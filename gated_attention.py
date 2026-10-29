"""Gated attention, arXiv:2505.06708.

One head-specific sigmoid gate, applied elementwise to the SDPA output:

    G    = sigmoid(X W_theta)                     # query-dependent, per (head, channel)
    out  = W_O ( SDPA(Q, K, V) * G )

That is the entire change. The paper's claim is that it buys three things at
once: a non-linearity inside attention's otherwise low-rank value path, a
sparse gate that lets a head contribute nothing at all for a given query, and
-- as a consequence of the second -- no attention sink.

Set `gate=False` for the identical module without the gate, so A/B comparisons
differ in exactly one thing.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """Causal multi-head attention with the paper's G1 variant (post-SDPA,
    head-specific, elementwise, sigmoid).

    Args:
        gate: False gives plain attention through the same projections.
        gate_bias_init: the gate's bias at init. 0.0 starts every gate at 0.5,
            which halves the residual branch's scale early in training; the
            paper's own runs start open. Raise it to ~4.0 (sigmoid ~0.98) if you
            are retrofitting the gate onto a trained checkpoint.
    """

    def __init__(self, dim, n_heads=8, head_dim=None, gate=True, gate_bias_init=0.0,
                 dropout=0.0):
        super().__init__()
        head_dim = head_dim or dim // n_heads
        self.n_heads, self.head_dim, self.dropout = n_heads, head_dim, dropout
        inner = n_heads * head_dim

        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, dim, bias=False)

        self.to_gate = nn.Linear(dim, inner, bias=True) if gate else None
        if self.to_gate is not None:
            with torch.no_grad():
                self.to_gate.bias.fill_(gate_bias_init)

    def _heads(self, t, B, T):
        return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, return_attn=False):
        B, T, _ = x.shape
        q = self._heads(self.q_proj(x), B, T)
        k = self._heads(self.k_proj(x), B, T)
        v = self._heads(self.v_proj(x), B, T)

        if return_attn:
            t = torch.arange(T, device=x.device)
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn = logits.masked_fill(
                ~(t[:, None] >= t[None, :]), float("-inf")).softmax(-1)
            o = attn @ v
        else:
            attn = None
            o = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0)

        gate = None
        if self.to_gate is not None:
            gate = self._heads(self.to_gate(x), B, T).sigmoid()
            o = o * gate

        out = self.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        return (out, attn, gate) if return_attn else out


def sink_mass(attn):
    """Share of attention each query sends to position 0 -- the attention sink.

    attn: (B, H, T, T). Position 0 is excluded as a query since it can only
    attend to itself.
    """
    return attn[:, :, 1:, 0].mean().item()


def gate_sparsity(gate, threshold=0.1):
    """Fraction of gate values below `threshold`, i.e. channels switched off."""
    return (gate < threshold).float().mean().item()
