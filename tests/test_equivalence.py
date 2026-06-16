"""Equivalence tests between the PyTorch and JAX implementations, checked
against each other and against simple NumPy expectations.

  1. test_cross_framework_logits: with shared weights (PyTorch init copied
     through NumPy into the JAX PyTree), the two forward passes agree to
     < 1e-4 max abs. Matmul precision is forced to fp32 on the JAX side;
     otherwise the accelerator backends lower matmuls to reduced precision
     and the comparison drifts.
  2. test_causal_mask_*: perturbing the token at position t+1 leaves the
     logits at positions <= t unchanged, confirming no information flows
     backward from future positions.
  3. test_kv_cache_*: a full forward over S tokens matches incremental
     decoding that appends K,V one position at a time, to < 1e-4 at the final
     position. This catches RoPE absolute-position errors in the decode path.

These check a single batch at fixed weights, so they do not say anything
about training dynamics: optimizer state, dropout, and accumulation order
still differ between independent runs.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import model_jax as mj
from convert import torch_to_jax
from model_torch import GPT, GPTConfig

SEED = 0
ATOL_LOGITS = 1e-4   # cross-framework / cache tolerance
ATOL_MASK = 1e-6     # same-framework, same-graph: near bit-identical


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(SEED)
    tcfg = GPTConfig(dropout=0.0)          # deterministic for equivalence
    tmodel = GPT(tcfg).eval().float()
    jparams = torch_to_jax(tmodel)
    jcfg = mj.Config()
    rng = np.random.default_rng(SEED)
    x = rng.integers(0, tcfg.vocab_size, size=(2, 64), dtype=np.int64)
    return tmodel, tcfg, jparams, jcfg, x


def test_cross_framework_logits(setup):
    tmodel, tcfg, jparams, jcfg, x = setup
    with torch.no_grad():
        t_logits = tmodel(torch.from_numpy(x))[0].numpy()
    # Force fp32 matmuls: on TPU/GPU, JAX's default matmul precision lowers
    # accumulation precision and the comparison drifts to ~1e-2.
    with jax.default_matmul_precision("float32"):
        j_logits = np.asarray(mj.forward(jparams, jnp.asarray(x, jnp.int32), jcfg))
    max_diff = np.abs(t_logits - j_logits).max()
    print(f"max |delta logit| = {max_diff:.2e}")
    assert max_diff < ATOL_LOGITS


def test_causal_mask_torch(setup):
    tmodel, tcfg, _, _, x = setup
    t = 10  # perturb position t+1; positions 0..t must be unaffected
    x2 = x.copy()
    x2[:, t + 1] = (x2[:, t + 1] + 1) % tcfg.vocab_size
    with torch.no_grad():
        a = tmodel(torch.from_numpy(x))[0][:, : t + 1]
        b = tmodel(torch.from_numpy(x2))[0][:, : t + 1]
    assert torch.allclose(a, b, atol=ATOL_MASK), "future token leaked backward"


def test_causal_mask_jax(setup):
    _, tcfg, jparams, jcfg, x = setup
    t = 10
    x2 = x.copy()
    x2[:, t + 1] = (x2[:, t + 1] + 1) % tcfg.vocab_size
    a = mj.forward(jparams, jnp.asarray(x, jnp.int32), jcfg)[:, : t + 1]
    b = mj.forward(jparams, jnp.asarray(x2, jnp.int32), jcfg)[:, : t + 1]
    assert np.abs(np.asarray(a - b)).max() < ATOL_MASK, "future token leaked"


def test_kv_cache_torch(setup):
    tmodel, _, _, _, x = setup
    xt = torch.from_numpy(x)
    with torch.no_grad():
        full = tmodel(xt)[0][:, -1]
        caches, logits = None, None
        for pos in range(xt.size(1)):          # one token at a time
            logits, caches = tmodel.forward_with_cache(
                xt[:, pos: pos + 1], caches, start_pos=pos)
        inc = logits[:, -1]
    diff = (full - inc).abs().max().item()
    print(f"torch cache max diff = {diff:.2e}")
    assert diff < ATOL_LOGITS


def test_kv_cache_jax(setup):
    _, _, jparams, jcfg, x = setup
    xj = jnp.asarray(x, jnp.int32)
    full = mj.forward(jparams, xj, jcfg)[:, -1]
    caches, logits = None, None
    for pos in range(xj.shape[1]):
        logits, caches = mj.forward_with_cache(
            jparams, xj[:, pos: pos + 1], jcfg, caches, start_pos=pos)
    inc = logits[:, -1]
    diff = np.abs(np.asarray(full - inc)).max()
    print(f"jax cache max diff = {diff:.2e}")
    assert diff < ATOL_LOGITS


def test_init_loss_sanity(setup):
    """At init, predictions are ~uniform -> loss ~ ln(vocab) = ln(1024).
    Catches softmax-over-wrong-axis / broken weight tying before any
    training time is wasted.

    Subtlety: with TIED embeddings the input token's logit against its own
    embedding is ||e||^2 > 0, so the init loss sits systematically a few
    tenths BELOW ln(V) (measured ~6.6 here), not exactly on it. A grossly
    wrong axis/tying bug lands far outside +-0.5."""
    tmodel, tcfg, jparams, jcfg, x = setup
    xt = torch.from_numpy(x)
    with torch.no_grad():
        _, t_loss = tmodel(xt, xt)
    j_loss = mj.loss_fn(jparams, (jnp.asarray(x, jnp.int32),) * 2, jcfg)
    target = math.log(tcfg.vocab_size)
    assert abs(t_loss.item() - target) < 0.5
    assert abs(float(j_loss) - target) < 0.5
