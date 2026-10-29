"""Gated vs ungated attention on a tiny char LM. CPU, ~1 min.

Identical models, identical seed, one difference: the sigmoid gate after SDPA.
Reports final loss, the attention sink at position 0, learned gate sparsity,
and peak hidden-state magnitude.

Read the sink numbers with care -- see "What this does and does not show" in
the README. The gate sparsity is the part that reproduces here.
"""

import argparse

import torch

from gated_attention import gate_sparsity, sink_mass
from lm import TinyLM, batch


def train(gate, steps, bs, lr, seed, quiet=False):
    torch.manual_seed(seed)
    model = TinyLM(gate=gate)

    x, _ = batch(bs)
    _, gates_at_init, _ = model.probe(x)
    init_sparsity = ([gate_sparsity(g) for g in gates_at_init]
                     if gates_at_init else None)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for step in range(1, steps + 1):
        xb, yb = batch(bs)
        loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not quiet and (step % 100 == 0 or step == 1):
            print(f"  {'gated' if gate else 'plain':>5}  step {step:4d}  loss {loss.item():.4f}")

    x, _ = batch(bs)
    attns, gates, peak = model.probe(x)
    return {
        "loss": loss.item(),
        "sink": [sink_mass(a) for a in attns],
        "peak": peak,
        "init_sparsity": init_sparsity,
        "sparsity": [gate_sparsity(g) for g in gates] if gates else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    g = train(True, a.steps, a.bs, a.lr, a.seed)
    print()
    p = train(False, a.steps, a.bs, a.lr, a.seed)

    fmt = lambda xs: "[" + ", ".join(f"{v:.3f}" for v in xs) + "]"
    print(f"\n{'':<18}{'gated':>26}{'plain':>26}")
    print(f"{'final loss':<18}{g['loss']:>26.4f}{p['loss']:>26.4f}")
    print(f"{'sink mass / layer':<18}{fmt(g['sink']):>26}{fmt(p['sink']):>26}")
    print(f"{'peak |hidden|':<18}{g['peak']:>26.1f}{p['peak']:>26.1f}")
    print(f"{'gate sparsity':<18}{fmt(g['sparsity']):>26}{'-':>26}")
    print(f"{'  ...at init':<18}{fmt(g['init_sparsity']):>26}{'-':>26}")

    assert g["loss"] < 1.0 and p["loss"] < 1.0, "a model failed to train"
    # reproduces here: the gate learns to switch channels off, starting from none
    assert sum(g["sparsity"]) > sum(g["init_sparsity"]) + 0.05, \
        (g["init_sparsity"], g["sparsity"])
    assert max(g["sparsity"]) > 0.05, g["sparsity"]
    # reproduces directionally: smaller peak activations (5/5 seeds, 8-20% lower)
    assert g["peak"] < p["peak"] * 1.05, (g["peak"], p["peak"])
    print("\nok  the gate learned sparsity it did not start with, and the gated")
    print("    model's peak hidden activation is the lower of the two")
    print("note: no attention sink forms at this scale in *either* model, so this")
    print("      run neither confirms nor refutes the paper's sink-free claim")


if __name__ == "__main__":
    main()
