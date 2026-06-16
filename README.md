# Decoder-only transformer in PyTorch and JAX

A small GPT-style language model written two ways, once in PyTorch and once in
JAX, with a test suite that checks the two implementations against each other.
With shared weights the forward passes agree to about 1e-6, the causal mask is
verified to block any backward information flow, and incremental KV-cache
decoding is checked against a full forward pass in both frameworks.

It trains on TinyShakespeare with a byte-level BPE tokenizer built from
scratch. The JAX training loop runs single-device or data-parallel (pmap or
`jax.sharding`) with bf16.

```
tokens - tied embedding - [ RMSNorm -> MHA(RoPE, causal) -> + residual
                            RMSNorm -> MLP(4x, GELU)      -> + residual ] x4
        - RMSNorm - tied unembedding - logits
```

## Model

| | |
|---|---|
| Architecture | decoder-only, pre-norm |
| Layers / d_model / heads | 4 / 256 / 8 (head_dim 32) |
| Normalization | RMSNorm |
| Positions | RoPE on Q and K (interleaved pairs) |
| MLP | 4x expansion, exact (erf) GELU |
| Embedding | weight-tied input and output |
| Tokenizer | byte-level BPE, vocab 1024 |
| Context length | 256 |
| Parameters | ~3.4M |
| Data | TinyShakespeare (~1.1 MB) |

Training uses AdamW with decoupled weight decay applied to matrices only,
linear warmup into cosine decay, gradient clipping at 1.0, batch 64 x 256, and
bf16 compute.

## Running

```bash
pip install -r requirements.txt   # install the jax wheel for your accelerator
python prepare_data.py            # download corpus, train BPE, tokenize
python train_torch.py --steps 3000
python train_jax.py   --steps 3000                  # single device
python train_jax.py   --parallel pmap  --bf16       # data-parallel
python train_jax.py   --parallel shard --bf16       # data-parallel via sharding
pytest tests/ -v                  # equivalence tests
```

`prepare_data.py --max-bytes 300000` learns the BPE merges on a slice of the
corpus for speed and still tokenizes the whole thing.

## Tests

`tests/test_equivalence.py` contains:

1. Cross-framework logits. PyTorch weights are exported to NumPy and loaded
   into the JAX PyTree, transposing each `nn.Linear` weight from `(out, in)`
   to `(in, out)`. With matmul precision forced to fp32 the two forward passes
   agree to under 1e-4.
2. Causal mask. Perturbing the token at position t+1 leaves the logits at
   positions up to t unchanged, in both frameworks.
3. KV cache. A full forward over the sequence matches token-by-token
   incremental decoding at the final position, in both frameworks.
4. Init-loss sanity. The loss at initialization is close to ln(1024), the
   value for uniform predictions over the vocabulary.

The tests compare a single batch at fixed weights. They do not check training
dynamics, which differ between independent runs because of dropout, optimizer
state, and floating-point accumulation order.

## Results

Measured by the test suite with shared weights and fp32 matmuls:

| Check | Result |
|---|---|
| Cross-framework logits (PyTorch vs JAX) | max abs diff 1.3e-06 |
| Causal mask, both frameworks | unchanged to < 1e-6 |
| KV-cache vs full forward, both frameworks | max abs diff < 1e-06 |
| Init loss vs ln(1024) = 6.93 | 6.6 (offset explained below) |
| BPE compression on TinyShakespeare | 2.33 chars/token |

The init loss sits slightly below ln(vocab) rather than exactly on it: with a
tied embedding, a token's logit against its own embedding row is positive, so
the distribution at initialization is not perfectly uniform.

### Training

Both models train for 3,000 steps on TinyShakespeare (batch 64 x 256, AdamW
lr 3e-4, cosine schedule). The PyTorch run uses bf16 autocast on a single GPU;
the JAX runs are single-device and data-parallel on 2x T4. Validation loss is
nats per token; dividing by the 2.33 chars/token compression gives roughly
1.6 nats/char.

| Run | final val loss | final val ppl | best val ppl |
|---|---|---|---|
| PyTorch (1 GPU, bf16) | 3.771 | 43.4 | 39.5 |
| JAX (1 device) | 3.727 | 41.6 | 38.6 |
| JAX (pmap, 2x T4, bf16) | 3.741 | 42.2 | 39.4 |
| JAX (shard, 2x T4, bf16) | 3.727 | 41.6 | 38.6 |

Curves are in `out/loss_torch.png`, `out/loss_jax.png`, and a twin comparison
in `out/comparison.png`.

Observations:

- The two implementations are trained independently and land within a few
  percent on validation perplexity (43.4 vs 41.6 at the final step, 39.5 vs
  38.6 at best). Together with the sub-1e-6 logit agreement at shared weights,
  this is the sense in which the JAX port matches the PyTorch reference. An
  earlier JAX run without dropout diverged to ppl ~176 while training loss
  fell below 1.0; the equivalence tests stayed green throughout, because they
  check the forward pass at fixed weights and say nothing about training-time
  regularization. Adding the dropout the PyTorch model already had brought the
  two back into agreement.
- The `shard` run lands at 3.727, matching the single-device run almost
  exactly, because it uses one RNG key that the compiler shards across
  devices. The `pmap` run is slightly different (3.741) because it draws an
  independent dropout key per device, so each replica applies a different
  mask. Both are valid; the gap is dropout noise, not a bug.
- Data-parallel throughput: the single-device run took 542s, the 2x T4 runs
  303s, about 1.8x. It is below 2x because the model is small enough that the
  gradient all-reduce and the unparallelized eval loop are a visible fraction
  of each step.
- bf16 is essentially free here: the shard+bf16 run (3.7268) and the
  single-device fp32 run (3.7270) differ by 0.0002 nats. No loss scaling is
  needed because bf16 keeps fp32's exponent range.

## Notes

- RoPE uses the interleaved-pair layout in both implementations. The
  rotate-half layout used by some other codebases is not interchangeable with
  it, so both files have to use the same one.
- `jax.nn.gelu` defaults to the tanh approximation while PyTorch defaults to
  exact erf, so the JAX side passes `approximate=False`.
- The cross-framework test forces `jax.default_matmul_precision("float32")`.
  Without it the accelerator backends use lower-precision matmuls and the
  comparison drifts to around 1e-2.
- Attention is written out instead of using
  `F.scaled_dot_product_attention`, which would be the faster drop-in. The
  algorithm and the numbers are the same.
- The JAX decode path is not jitted, because the cache grows by concatenation
  and each new shape would recompile. A preallocated cache with
  `lax.dynamic_update_slice` would avoid that.
