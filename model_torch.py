"""Decoder-only transformer (PyTorch).

Config: 4 layers, d_model 256, 8 heads (head_dim 32), MLP 4x with exact GELU,
RMSNorm pre-norm, RoPE on Q/K (interleaved pairs), tied embedding/unembedding,
vocab 1024, context 256, dropout 0.1. ~3.4M params.

GELU uses the exact erf form and RoPE uses the interleaved-pair layout so the
JAX implementation in model_jax.py matches numerically. Linear layers are
bias-free.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8
    max_seq_len: int = 256
    dropout: float = 0.1
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """y = g * x / RMS(x); no mean-centering, no bias (vs LayerNorm)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_freqs(head_dim, max_seq, base=10000.0):
    """Angle tables. theta_i = base^(-2i/d); entry (m, i) = m * theta_i."""
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    t = torch.arange(max_seq).float()
    f = torch.outer(t, inv)                      # (S, D/2)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    """Rotate interleaved pairs (x_{2i}, x_{2i+1}) by m*theta_i.
    x: (B, H, S, D); cos/sin: (S, D/2), sliced to these positions.
    Interleaved layout (not rotate-half); model_jax uses the same one."""
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                       dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        d = cfg.d_model
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, mask=None, cache=None):
        """mask: additive (0 / -inf), broadcastable to (B, H, S_q, S_k).
        cache: optional (k, v) of shape (B, H, S_past, D); returns updated.
        Attention is written out rather than using SDPA so the steps are
        explicit; F.scaled_dot_product_attention is the drop-in replacement."""
        B, S, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.wq(x).view(B, S, H, D).transpose(1, 2)   # (B, H, S, D)
        k = self.wk(x).view(B, S, H, D).transpose(1, 2)
        v = self.wv(x).view(B, S, H, D).transpose(1, 2)
        # RoPE before concatenating the cache: cached K already carries its
        # absolute positions; only the new positions get rotated here.
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            pk, pv = cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None:
            att = att + mask          # additive -inf before softmax
        att = F.softmax(att, dim=-1)  # row-max subtraction handled inside
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, S, H * D)
        return self.resid_drop(self.wo(y)), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.proj = nn.Linear(4 * d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # exact GELU (erf), matching jax.nn.gelu(approximate=False)
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm: norm sits inside the residual branch."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, mask=None, cache=None):
        a, new_cache = self.attn(self.ln1(x), cos, sin, mask, cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # additive causal mask: 0 below/on diagonal, -inf strictly above
        m = torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")).triu(1)
        self.register_buffer("mask", m, persistent=False)

        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # tied head shares storage with the embedding -> count once
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """Training/eval forward. idx: (B, S) int64. Returns (logits, loss)."""
        B, S = idx.shape
        assert S <= self.cfg.max_seq_len
        x = self.drop(self.tok_emb(idx))
        cos, sin, mask = self.cos[:S], self.sin[:S], self.mask[:S, :S]
        for blk in self.blocks:
            x, _ = blk(x, cos, sin, mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def forward_with_cache(self, idx, caches=None, start_pos=0):
        """Inference path (no dropout). Prefill: idx (B, S), caches=None,
        start_pos=0. Decode: idx (B, 1), start_pos = the new token's absolute
        position, i.e. its 0-based index, which equals the sequence length
        before this token is appended. RoPE is applied at start_pos, so an
        error here shows up as degraded generation after the first token;
        test_kv_cache_* checks this against the full forward."""
        B, S = idx.shape
        x = self.tok_emb(idx)
        cos = self.cos[start_pos:start_pos + S]
        sin = self.sin[start_pos:start_pos + S]
        mask = None
        if S > 1:
            past = caches[0][0].size(2) if caches is not None else 0
            m = torch.zeros(S, past + S, device=idx.device)
            m[:, past:] = self.mask[:S, :S]   # cached positions fully visible
            mask = m
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, cos, sin, mask, caches[i] if caches else None)
            new_caches.append(c)
        x = self.ln_f(x)
        return self.lm_head(x), new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=None):
        self.eval()
        idx = idx[:, -self.cfg.max_seq_len:]
        logits, caches = self.forward_with_cache(idx, None, 0)
        for _ in range(max_new_tokens):
            nxt = _sample(logits[:, -1], temperature, top_p)
            idx = torch.cat([idx, nxt], dim=1)
            if idx.size(1) >= self.cfg.max_seq_len:
                break
            # new token's absolute position = its 0-based index = len - 1
            logits, caches = self.forward_with_cache(nxt, caches, idx.size(1) - 1)
        return idx


def _sample(logits, temperature=1.0, top_p=None):
    logits = logits / max(temperature, 1e-8)
    if top_p is not None:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        drop = (cum - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, order, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg)
    print(f"params: {model.num_params() / 1e6:.2f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(x, x)
    print("logits:", tuple(logits.shape),
          f"| init loss {loss.item():.3f} (expect ~ln(1024)={math.log(1024):.3f})")
