# gated-attention-minimal

A minimal PyTorch implementation of **gated attention** — one sigmoid gate after
SDPA, and the measurements to see what it does.

> **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free**
> Qiu et al., Qwen Team, Alibaba — [arXiv:2505.06708](https://arxiv.org/abs/2505.06708)

The whole change:

```
G   = sigmoid(X W_θ)                     # query-dependent, per (head, channel)
out = W_O ( SDPA(Q, K, V) ⊙ G )
```

A head-specific, elementwise sigmoid gate on the attention output — the paper's
**G1** variant. Three claimed effects: a non-linearity inside attention's
otherwise low-rank value path, a sparse gate that lets a head contribute
*nothing* for a given query, and — as a consequence of that sparsity — no
attention sink. Validated across 30 variants of 15B MoE and 1.7B dense models.

## Install

```bash
pip install torch
```

## Use

Drop-in replacement for causal MHA. `gate=False` gives the identical module
without the gate, so an A/B differs in exactly one thing:

```python
import torch
from gated_attention import GatedAttention

attn = GatedAttention(dim=512, n_heads=8, gate=True)
y = attn(torch.randn(2, 1024, 512))

# inspect what the gate is doing
y, attn_map, gate = attn(x, return_attn=True)   # (B,H,T,T) and (B,H,T,Dh)
```

`gate_bias_init` controls the gate at init. The default `0.0` starts every gate
at 0.5, which halves the residual branch early on — training from scratch
absorbs that fine, but on one of five seeds it noticeably slowed the first few
hundred steps. Use `gate_bias_init=4.0` (sigmoid ≈ 0.98) when retrofitting the
gate onto a trained checkpoint.

Two metric helpers: `sink_mass(attn)` is the share of attention each query sends
to position 0; `gate_sparsity(gate)` is the fraction of gate values below a
threshold.

## Verify

```bash
python test_gated_attention.py   # 9 checks, instant
python train.py                  # gated vs plain tiny LM, ~1 min
```

What the checks pin down:

- **the gate is the only difference** — force it open (`sigmoid(+30) = 1`) and
  the gated module reproduces plain attention through the same weights, to 1e-5
- **the gate can close completely** — force it shut and the layer's output is
  exactly zero. That is the sparsity mechanism, not a metaphor.
- **the hand-written attention path agrees with the fused kernel** to 1e-5, rows
  sum to 1, and nothing above the diagonal is non-zero
- the gate really is head-specific and query-dependent (varies across both axes)
- causality, gradient flow to `W_θ`, `gate_bias_init`, and the two metric helpers

## What this does and does not show

`train.py` trains two 3-layer char LMs, same seed, gate on and off:

```
                                   gated                     plain
final loss                        0.0378                    0.0457
sink mass / layer  [0.045, 0.042, 0.041]     [0.040, 0.038, 0.037]
peak |hidden|                       22.8                      29.3
gate sparsity      [0.058, 0.086, 0.162]                         -
  ...at init       [0.000, 0.000, 0.000]                         -
```

**Reproduces: learned sparsity.** The gate starts with *zero* values below 0.1
and ends at 4–20% of channels switched off, rising with depth. Consistent across
5 seeds.

**Reproduces directionally: smaller activations.** Peak hidden-state magnitude
was lower for the gated model on 5/5 seeds, by 8–20%. That matches the paper's
massive-activations finding, at a scale far too small to call it confirmation.

**Does not reproduce: the attention sink.** No sink forms here — sink mass sits
near uniform (~1/T) in *both* models. The sink is a large-scale pretraining
phenomenon, and a 3-layer model on a few hundred characters never develops one.
So this run neither confirms nor refutes the paper's headline claim, and
`train.py` says so rather than asserting a result it cannot support.

**Not a loss comparison.** The gated model won on 3 of 5 seeds. At this scale
that is noise.

## Scope

Attention and the gate only. No RoPE (positional encoding belongs in your
model), no GQA/MQA, no KV-cache decoding, and none of the paper's other gate
placements — it ablates gating before/after SDPA, on Q/K/V, head-shared vs
head-specific, and sigmoid vs other activations. G1 is the one it recommends,
and G1 is what this implements.

## License

MIT
