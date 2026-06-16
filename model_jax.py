"""Decoder-only transformer (JAX), matching model_torch.py.

Parameters are stored in a PyTree of nested dicts; the forward pass is a pure
function of (params, tokens) -> logits.

Details that need to line up with the PyTorch version for the equivalence
tests to pass:
  - RoPE uses interleaved pairs (x[..., ::2], x[..., 1::2]), not rotate-half.
  - GELU uses the exact erf form (jax.nn.gelu(approximate=False)).
  - Matmuls are x @ W with W shaped (in, out); nn.Linear stores (out, in),
    so convert.py transposes each Linear weight.
  - The causal mask is additive (0 / -inf on the upper triangle) and applied
    before the softmax.

Dropout matches the PyTorch rate during training and is disabled at eval.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Config:
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


# ---------------------------------------------------------------- params ---

def init_params(key, cfg: Config):
    def dense(key, shape):
        return 0.02 * jax.random.normal(key, shape, dtype=jnp.float32)

    keys = jax.random.split(key, 1 + 6 * cfg.n_layers)
    d = cfg.d_model
    params = {
        "tok_emb": dense(keys[0], (cfg.vocab_size, d)),  # also the (tied) head
        "ln_f_g": jnp.ones(d),
        "blocks": [],
    }
    for i in range(cfg.n_layers):
        k = keys[1 + 6 * i: 1 + 6 * (i + 1)]
        params["blocks"].append({
            "ln1_g": jnp.ones(d),
            "wq": dense(k[0], (d, d)),
            "wk": dense(k[1], (d, d)),
            "wv": dense(k[2], (d, d)),
            "wo": dense(k[3], (d, d)),
            "ln2_g": jnp.ones(d),
            "fc": dense(k[4], (d, 4 * d)),
            "proj": dense(k[5], (4 * d, d)),
        })
    return params


def num_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# ------------------------------------------------------------ primitives ---

def rmsnorm(g, x, eps=1e-6):
    return g * x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)


def rope_tables(head_dim, max_seq, base=10000.0):
    inv = 1.0 / base ** (np.arange(0, head_dim, 2) / head_dim)
    f = np.outer(np.arange(max_seq), inv)            # (S, D/2)
    return jnp.asarray(np.cos(f), jnp.float32), jnp.asarray(np.sin(f), jnp.float32)


def apply_rope(x, cos, sin):
    """Interleaved-pairs convention; identical to model_torch.apply_rope."""
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                     axis=-1).reshape(x.shape)


def split_heads(x, n_heads):
    B, S, d = x.shape
    return x.reshape(B, S, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x):
    B, H, S, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, S, H * D)


def causal_mask(S, dtype=jnp.float32):
    """Additive mask: 0 on/below diagonal, -inf strictly above."""
    return jnp.triu(jnp.full((S, S), -jnp.inf, dtype=dtype), k=1)


def dropout(key, x, rate, train):
    """Inverted dropout, matching torch.nn.Dropout: at train time keep each
    unit w.p. (1-rate) and rescale survivors by 1/(1-rate); identity at eval.
    key=None / train=False / rate=0 => no-op, so the inference and oracle-test
    paths stay deterministic and bit-comparable with PyTorch eval."""
    if not train or rate == 0.0 or key is None:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(key, keep, x.shape).astype(x.dtype)
    return x * mask / keep


# ----------------------------------------------------------------- model ---

def attention(p, x, cos, sin, mask, n_heads, cache=None, *,
              key_w=None, key_r=None, drop=0.0, train=False):
    q, k, v = x @ p["wq"], x @ p["wk"], x @ p["wv"]      # (B, S, d)
    q, k, v = (split_heads(t, n_heads) for t in (q, k, v))  # (B, H, S, D)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if cache is not None:
        k = jnp.concatenate([cache[0], k], axis=2)
        v = jnp.concatenate([cache[1], v], axis=2)
    new_cache = (k, v)
    D = q.shape[-1]
    att = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(jnp.float32(D))
    if mask is not None:
        att = att + mask                                  # 0 / -inf, pre-softmax
    att = jax.nn.softmax(att, axis=-1)
    att = dropout(key_w, att, drop, train)                # on attention weights
    out = merge_heads(att @ v) @ p["wo"]
    out = dropout(key_r, out, drop, train)                # on residual output
    return out, new_cache


def mlp(p, x):
    return jax.nn.gelu(x @ p["fc"], approximate=False) @ p["proj"]


def block(p, x, cos, sin, mask, n_heads, cache=None, *,
          keys=(None, None, None), drop=0.0, train=False):
    a, new_cache = attention(p, rmsnorm(p["ln1_g"], x), cos, sin, mask, n_heads,
                             cache, key_w=keys[0], key_r=keys[1],
                             drop=drop, train=train)
    x = x + a
    m = dropout(keys[2], mlp(p, rmsnorm(p["ln2_g"], x)), drop, train)
    x = x + m
    return x, new_cache


def forward(params, idx, cfg: Config, cos=None, sin=None, *,
            key=None, train=False):
    """Full causal forward. idx: (B, S) int32. Returns logits (B, S, V).
    Dropout is applied only when train=True AND key is not None; the oracle
    tests and all inference call this with defaults => deterministic."""
    S = idx.shape[1]
    if cos is None:
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
    cos, sin = cos[:S], sin[:S]
    mask = causal_mask(S)
    x = params["tok_emb"][idx]
    drop = cfg.dropout if train else 0.0
    n_keys = 1 + 3 * cfg.n_layers          # 1 embed + (attn_w, attn_r, mlp)/blk
    ks = (list(jax.random.split(key, n_keys)) if (train and key is not None)
          else [None] * n_keys)
    x = dropout(ks[0], x, drop, train)     # embedding dropout
    for i, bp in enumerate(params["blocks"]):
        x, _ = block(bp, x, cos, sin, mask, cfg.n_heads,
                     keys=ks[1 + 3 * i: 4 + 3 * i], drop=drop, train=train)
    x = rmsnorm(params["ln_f_g"], x)
    return x @ params["tok_emb"].T                        # tied unembedding


def forward_with_cache(params, idx, cfg: Config, caches=None, start_pos=0,
                       cos=None, sin=None):
    """Incremental-decode path, matching GPT.forward_with_cache.
    Not jitted: the cache grows by concatenation, so the shape changes every
    step and each shape would trigger a recompile. A preallocated cache with
    lax.dynamic_update_slice would avoid that."""
    B, S = idx.shape
    if cos is None:
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
    cos_s, sin_s = cos[start_pos:start_pos + S], sin[start_pos:start_pos + S]
    mask = None
    if S > 1:
        past = caches[0][0].shape[2] if caches is not None else 0
        mask = jnp.concatenate(
            [jnp.zeros((S, past)), causal_mask(S)], axis=1)
    x = params["tok_emb"][idx]
    new_caches = []
    for i, bp in enumerate(params["blocks"]):
        x, c = block(bp, x, cos_s, sin_s, mask, cfg.n_heads,
                     caches[i] if caches else None)
        new_caches.append(c)
    x = rmsnorm(params["ln_f_g"], x)
    return x @ params["tok_emb"].T, new_caches


def generate(params, idx, cfg: Config, key, max_new_tokens, temperature=1.0):
    """Simple temperature sampling with the KV cache."""
    cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
    logits, caches = forward_with_cache(params, idx, cfg, None, 0, cos, sin)
    for _ in range(max_new_tokens):
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, logits[:, -1] / temperature)[:, None]
        idx = jnp.concatenate([idx, nxt.astype(idx.dtype)], axis=1)
        if idx.shape[1] >= cfg.max_seq_len:
            break
        logits, caches = forward_with_cache(
            params, nxt.astype(idx.dtype), cfg, caches, idx.shape[1] - 1,
            cos, sin)
    return idx


def loss_fn(params, batch, cfg: Config, bf16=False, *, key=None, train=False):
    """Mean next-token cross-entropy. batch = (x, y), each (B, S) int32.
    bf16 policy: fp32 master params, bf16 compute, fp32 loss (no loss scaling
    needed -- bf16 shares fp32's exponent range). Pass key + train=True from
    the training step to enable dropout; eval calls omit them."""
    x, y = batch
    p = params
    if bf16:
        p = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), params)
    logits = forward(p, x, cfg, key=key, train=train).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1).squeeze(-1)
    return nll.mean()


if __name__ == "__main__":
    cfg = Config()
    params = init_params(jax.random.PRNGKey(0), cfg)
    print(f"params: {num_params(params) / 1e6:.2f}M | devices: {jax.devices()}")
    x = jax.random.randint(jax.random.PRNGKey(1), (2, 64), 0, cfg.vocab_size)
    fwd = jax.jit(partial(forward, cfg=cfg))
    print("logits:", fwd(params, x).shape)
    print(f"init loss {loss_fn(params, (x, x), cfg):.3f} "
          f"(expect ~ln(1024)={np.log(1024):.3f})")
