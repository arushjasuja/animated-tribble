"""JAX training loop, with single-device and data-parallel modes.

  python train_jax.py --steps 3000                      # single device
  python train_jax.py --parallel pmap                   # pmap + pmean
  python train_jax.py --parallel shard                  # Mesh / NamedSharding
  python train_jax.py --parallel pmap --bf16            # data-parallel + bf16

Both data-parallel modes split the global batch across jax.devices() and
average gradients: pmap does this with an explicit lax.pmean in the step,
shard lets the compiler insert the all-reduce from the sharding annotations.
Run on a multi-device host (e.g. 2x T4 or a TPU runtime) to use more than one
device.

Writes out/ckpt_jax.pkl, out/loss_jax.csv, out/loss_jax.png and prints a
sample generation.
"""

import argparse
import csv
import math
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

import model_jax as mj
from bpe import BPETokenizer


def get_batch(tokens, batch_size, seq_len, rng):
    ix = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    x = np.stack([tokens[i: i + seq_len] for i in ix]).astype(np.int32)
    y = np.stack([tokens[i + 1: i + seq_len + 1] for i in ix]).astype(np.int32)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="GLOBAL batch; split across devices when parallel")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--parallel", choices=["none", "pmap", "shard"],
                    default="none")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    devices = jax.devices()
    n_dev = len(devices)
    print(f"devices: {devices} | parallel={args.parallel} bf16={args.bf16}")
    if args.parallel != "none":
        assert args.batch_size % n_dev == 0, "global batch must divide n_dev"

    tokens = np.load("data/tokens.npy")
    n = int(0.9 * len(tokens))
    train_tok, val_tok = tokens[:n], tokens[n:]

    cfg = mj.Config()
    params = mj.init_params(jax.random.PRNGKey(0), cfg)
    print(f"params: {mj.num_params(params) / 1e6:.2f}M")

    # optax's decay_steps is the TOTAL schedule length (warmup included),
    # so warmup must stay strictly below it.
    warmup = min(args.warmup, max(1, args.steps - 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr, warmup_steps=warmup,
        decay_steps=args.steps, end_value=0.1 * args.lr)
    # decoupled decay on matrices only (mask), matching the PyTorch model
    decay_mask = jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95,
                    weight_decay=args.weight_decay, mask=decay_mask))
    opt_state = tx.init(params)

    loss_fn = partial(mj.loss_fn, cfg=cfg, bf16=args.bf16)

    def step_body(params, opt_state, batch, key, axis_name=None):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, batch, key=key, train=True)          # dropout ON in training
        if axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name)   # gradient all-reduce
            loss = jax.lax.pmean(loss, axis_name)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    eval_loss = jax.jit(loss_fn)                          # train defaults False

    if args.parallel == "pmap":
        # SPMD: replicate params/opt_state, shard the batch on a leading
        # device axis. lax.pmean inside the step averages gradients across
        # devices, the same all-reduce DDP performs in its backward hook.
        train_step = jax.pmap(partial(step_body, axis_name="data"),
                              axis_name="data", donate_argnums=(0, 1))
        # replicate by stacking on a leading device axis (pmap's expected
        # layout; jax.device_put_replicated was removed in newer JAX)
        replicate = lambda t: jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * n_dev), t)
        params = replicate(params)
        opt_state = replicate(opt_state)

        def shard(b):
            return tuple(a.reshape(n_dev, -1, *a.shape[1:]) for a in b)

        def unreplicate(tree):
            return jax.tree_util.tree_map(lambda x: np.asarray(x[0]), tree)

        def split_key(k):
            return jax.random.split(k, n_dev)   # one dropout key per device

    elif args.parallel == "shard":
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        mesh = Mesh(np.array(devices), ("data",))
        repl = NamedSharding(mesh, P())
        data_sh = NamedSharding(mesh, P("data"))
        params = jax.device_put(params, repl)
        opt_state = jax.device_put(opt_state, repl)
        # donate_argnums frees the old params/opt_state buffers in place,
        # roughly halving peak memory for the update.
        train_step = jax.jit(step_body, donate_argnums=(0, 1))

        def shard(b):
            return tuple(jax.device_put(a, data_sh) for a in b)

        def unreplicate(tree):
            return jax.tree_util.tree_map(np.asarray, tree)

        def split_key(k):
            return k

    else:
        train_step = jax.jit(step_body, donate_argnums=(0, 1))

        def shard(b):
            return b

        def unreplicate(tree):
            return jax.tree_util.tree_map(np.asarray, tree)

        def split_key(k):
            return k

    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    os.makedirs("out", exist_ok=True)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        batch = shard(get_batch(train_tok, args.batch_size,
                                cfg.max_seq_len, rng))
        key, sub = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, batch,
                                             split_key(sub))
        if step % args.eval_every == 0 or step == args.steps - 1:
            p_eval = params
            if args.parallel == "pmap":
                p_eval = jax.tree_util.tree_map(lambda x: x[0], params)
                loss_v = float(np.asarray(loss)[0])
            else:
                loss_v = float(loss)
            vals = [float(eval_loss(p_eval,
                                    get_batch(val_tok, args.batch_size,
                                              cfg.max_seq_len, rng)))
                    for _ in range(10)]
            val = float(np.mean(vals))
            dt = time.time() - t0
            print(f"step {step:5d} | train {loss_v:.4f} | val {val:.4f} "
                  f"| ppl {math.exp(val):7.2f} | {dt:6.1f}s")
            log.append((step, loss_v, val))

    with open("out/loss_jax.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss"])
        w.writerows(log)
    final_params = unreplicate(params)
    with open("out/ckpt_jax.pkl", "wb") as f:
        pickle.dump(final_params, f)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, tr, va = zip(*log)
        plt.plot(steps, tr, label="train")
        plt.plot(steps, va, label="val")
        plt.xlabel("step"); plt.ylabel("loss (nats/token)")
        plt.legend(); plt.title(f"JAX ({args.parallel}, bf16={args.bf16})")
        plt.savefig("out/loss_jax.png", dpi=120)
    except ImportError:
        pass

    tok = BPETokenizer().load("data/merges.json")
    fp = jax.tree_util.tree_map(jnp.asarray, final_params)
    prompt = jnp.asarray([tok.encode("ROMEO:")], jnp.int32)
    out = mj.generate(fp, prompt, cfg, jax.random.PRNGKey(42), 200,
                      temperature=0.8)
    print("--- sample ---")
    print(tok.decode(np.asarray(out[0]).tolist()))


if __name__ == "__main__":
    main()
