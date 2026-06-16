"""PyTorch training loop. AdamW with decoupled weight decay (applied to 2-D
parameters only), linear warmup into cosine decay, gradient clipping at 1.0,
and bf16 autocast on CUDA.

  python prepare_data.py
  python train_torch.py --steps 3000

Writes out/ckpt.pt, out/loss_torch.csv, out/loss_torch.png and prints a
sample generation.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from bpe import BPETokenizer
from model_torch import GPT, GPTConfig


def lr_at(step, peak, warmup, total, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def get_batch(tokens, batch_size, seq_len, rng, device):
    ix = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    x = np.stack([tokens[i: i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1: i + seq_len + 1] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


@torch.no_grad()
def estimate_loss(model, tokens, batch_size, seq_len, rng, device, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(tokens, batch_size, seq_len, rng, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = args.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={device} bf16={use_bf16}")

    tokens = np.load("data/tokens.npy")
    n = int(0.9 * len(tokens))
    train_tok, val_tok = tokens[:n], tokens[n:]

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    print(f"params: {model.num_params() / 1e6:.2f}M")

    # decoupled weight decay on matrices only; norms/1-D params undecayed
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    rng = np.random.default_rng(0)
    os.makedirs("out", exist_ok=True)
    log = []
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_tok, args.batch_size, cfg.max_seq_len, rng, device)
        # bf16 autocast: fp32 master weights, bf16 compute. No GradScaler
        # needed (bf16 keeps fp32's exponent range -- unlike fp16).
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=use_bf16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            val = estimate_loss(model, val_tok, args.batch_size,
                                cfg.max_seq_len, rng, device)
            dt = time.time() - t0
            print(f"step {step:5d} | train {loss.item():.4f} | val {val:.4f} "
                  f"| ppl {math.exp(val):7.2f} | lr {lr:.2e} | {dt:6.1f}s")
            log.append((step, loss.item(), val))

    with open("out/loss_torch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss"])
        w.writerows(log)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__},
               "out/ckpt.pt")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, tr, va = zip(*log)
        plt.plot(steps, tr, label="train")
        plt.plot(steps, va, label="val")
        plt.xlabel("step"); plt.ylabel("loss (nats/token)")
        plt.legend(); plt.title("PyTorch")
        plt.savefig("out/loss_torch.png", dpi=120)
    except ImportError:
        pass

    tok = BPETokenizer().load("data/merges.json")
    prompt = torch.tensor([tok.encode("ROMEO:")], dtype=torch.long, device=device)
    out = model.generate(prompt, 200, temperature=0.8, top_p=0.95)
    print("--- sample ---")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
