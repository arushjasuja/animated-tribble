"""Download TinyShakespeare, train the BPE tokenizer, tokenize the corpus.

Outputs (under data/):
  input.txt    raw corpus (~1.1 MB)
  merges.json  learned BPE merges (vocab 1024)
  tokens.npy   full corpus tokenized, uint16

Run once before either training script:
  python prepare_data.py            # full corpus (~5 min of pure-Python BPE)
  python prepare_data.py --max-bytes 300000   # faster: train merges on a slice
"""

import argparse
import os
import urllib.request

import numpy as np

from bpe import BPETokenizer

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="train merges on only the first N bytes (speed)")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/input.txt"):
        print("downloading TinyShakespeare...")
        urllib.request.urlretrieve(URL, "data/input.txt")
    with open("data/input.txt", encoding="utf-8") as f:
        text = f.read()
    print(f"corpus: {len(text):,} chars")

    tok = BPETokenizer()
    if args.max_bytes is not None and args.max_bytes < len(text):
        print(f"training BPE on first {args.max_bytes:,} bytes...")
        tok.train(text[:args.max_bytes], args.vocab_size, verbose=True)
        print("encoding full corpus...")
        ids = tok.encode(text)
    else:
        print("training BPE on full corpus (this is pure Python; ~5 min)...")
        ids = tok.train(text, args.vocab_size, verbose=True)

    # sanity checks: round-trip and compression ratio
    probe = text[10_000:12_000]
    assert tok.decode(tok.encode(probe)) == probe, "round-trip FAILED"
    ratio = len(text) / len(ids)
    print(f"round-trip OK | {len(ids):,} tokens | {ratio:.2f} chars/token")

    tok.save("data/merges.json")
    assert max(ids) < 65536
    np.save("data/tokens.npy", np.array(ids, dtype=np.uint16))
    print("saved data/merges.json, data/tokens.npy")


if __name__ == "__main__":
    main()
