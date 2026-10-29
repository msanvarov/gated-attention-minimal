"""Self-checks for gated attention. Run: python test_gated_attention.py"""

import torch

from gated_attention import GatedAttention, gate_sparsity, sink_mass


def test_shapes():
    for gate in (True, False):
        attn = GatedAttention(64, n_heads=8, gate=gate)
        assert attn(torch.randn(2, 16, 64)).shape == (2, 16, 64)


def test_manual_attention_matches_the_fused_kernel():
    """`return_attn=True` takes a hand-written path; it must agree with SDPA."""
    torch.manual_seed(0)
    attn = GatedAttention(64, n_heads=4).eval()
    x = torch.randn(2, 24, 64)
    fused = attn(x)
    manual, a, g = attn(x, return_attn=True)
    assert torch.allclose(fused, manual, atol=1e-5), (fused - manual).abs().max()
    assert a.shape == (2, 4, 24, 24)
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    assert torch.triu(a, diagonal=1).abs().max() == 0  # strictly causal


def test_open_gate_recovers_plain_attention():
    """sigmoid(+30) = 1, so the gated module must collapse onto the ungated one."""
    torch.manual_seed(0)
    gated = GatedAttention(64, n_heads=4, gate=True).eval()
    plain = GatedAttention(64, n_heads=4, gate=False).eval()
    plain.load_state_dict(
        {k: v for k, v in gated.state_dict().items() if not k.startswith("to_gate")}
    )
    with torch.no_grad():
        gated.to_gate.weight.zero_()
        gated.to_gate.bias.fill_(30.0)
    x = torch.randn(2, 24, 64)
    assert torch.allclose(gated(x), plain(x), atol=1e-5), \
        (gated(x) - plain(x)).abs().max()


def test_closed_gate_silences_the_layer():
    """The gate can remove a head's contribution entirely -- the sparsity claim."""
    torch.manual_seed(0)
    attn = GatedAttention(64, n_heads=4).eval()
    with torch.no_grad():
        attn.to_gate.weight.zero_()
        attn.to_gate.bias.fill_(-30.0)
    assert attn(torch.randn(2, 16, 64)).abs().max() < 1e-6


def test_gate_is_head_specific_and_query_dependent():
    torch.manual_seed(0)
    attn = GatedAttention(64, n_heads=4).eval()
    _, _, g = attn(torch.randn(4, 32, 64), return_attn=True)  # (B, H, T, Dh)
    assert g.shape == (4, 4, 32, 16)
    # varies across query positions -> query-dependent
    assert g.std(dim=2).mean() > 0.01, g.std(dim=2).mean()
    # heads do not share a gate -> head-specific
    per_head = g.mean(dim=(0, 2, 3))
    assert per_head.std() > 0, per_head
    assert not torch.allclose(g[:, 0], g[:, 1], atol=1e-3)


def test_causality():
    torch.manual_seed(0)
    attn = GatedAttention(64, n_heads=4).eval()
    x = torch.randn(1, 24, 64)
    y = attn(x)
    x2 = x.clone()
    x2[:, 12:] += 5 * torch.randn_like(x2[:, 12:])
    assert torch.allclose(y[:, :12], attn(x2)[:, :12], atol=1e-5)


def test_gradients_reach_the_gate():
    attn = GatedAttention(64, n_heads=4)
    attn(torch.randn(2, 16, 64)).pow(2).mean().backward()
    assert attn.to_gate.weight.grad.abs().sum() > 0
    assert attn.to_gate.bias.grad.abs().sum() > 0


def test_gate_starts_fully_open_when_asked():
    """gate_bias_init is how you retrofit the gate onto a trained checkpoint."""
    attn = GatedAttention(32, n_heads=4, gate_bias_init=4.0).eval()
    with torch.no_grad():
        attn.to_gate.weight.zero_()
    _, _, g = attn(torch.randn(1, 8, 32), return_attn=True)
    assert g.min() > 0.98, g.min()


def test_metric_helpers():
    B, H, T = 2, 3, 8
    all_on_zero = torch.zeros(B, H, T, T)
    all_on_zero[..., 0] = 1.0
    assert abs(sink_mass(all_on_zero) - 1.0) < 1e-6
    uniform = torch.full((B, H, T, T), 1.0 / T)
    assert abs(sink_mass(uniform) - 1.0 / T) < 1e-6
    assert gate_sparsity(torch.tensor([0.0, 0.05, 0.5, 1.0])) == 0.5


if __name__ == "__main__":
    for fn in [
        test_shapes,
        test_manual_attention_matches_the_fused_kernel,
        test_open_gate_recovers_plain_attention,
        test_closed_gate_silences_the_layer,
        test_gate_is_head_specific_and_query_dependent,
        test_causality,
        test_gradients_reach_the_gate,
        test_gate_starts_fully_open_when_asked,
        test_metric_helpers,
    ]:
        fn()
        print(f"ok  {fn.__name__}")
    print("\nall checks passed")
