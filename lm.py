"""Tiny char-level LM used by train.py, so gated/ungated differ in one thing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gated_attention import GatedAttention

TEXT = (
    "the quick brown fox jumps over the lazy dog. pack my box with five dozen "
    "liquor jugs. how vexingly quick daft zebras jump! sphinx of black quartz, "
    "judge my vow. the five boxing wizards jump quickly at dawn. "
) * 48
VOCAB = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(VOCAB)}
DATA = torch.tensor([STOI[c] for c in TEXT])
CTX = 64


def batch(bs):
    i = torch.randint(0, len(DATA) - CTX - 1, (bs,))
    return (torch.stack([DATA[j:j + CTX] for j in i]),
            torch.stack([DATA[j + 1:j + CTX + 1] for j in i]))


class Block(nn.Module):
    def __init__(self, dim, heads, gate):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = GatedAttention(dim, n_heads=heads, gate=gate)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, h, return_attn=False):
        if return_attn:
            a, attn, g = self.attn(self.n1(h), return_attn=True)
            h = h + a
            return h + self.mlp(self.n2(h)), attn, g
        h = h + self.attn(self.n1(h))
        return h + self.mlp(self.n2(h))


class TinyLM(nn.Module):
    def __init__(self, gate, dim=128, heads=4, layers=3):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), dim)
        self.pos = nn.Embedding(CTX, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, gate) for _ in range(layers))
        self.norm_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, len(VOCAB), bias=False)

    def forward(self, idx, targets=None):
        h = self.embed(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.norm_f(h))
        if targets is None:
            return logits
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    @torch.no_grad()
    def probe(self, idx):
        """Returns (per-layer attention maps, per-layer gates, max |hidden|)."""
        h = self.embed(idx) + self.pos(torch.arange(idx.shape[1]))
        attns, gates, peak = [], [], h.abs().max().item()
        for b in self.blocks:
            h, attn, g = b(h, return_attn=True)
            attns.append(attn)
            if g is not None:
                gates.append(g)
            peak = max(peak, h.abs().max().item())
        return attns, gates, peak
