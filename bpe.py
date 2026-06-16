"""Byte-level BPE tokenizer from scratch (no external tokenizer libraries).

Algorithm (training):
  1. Start from the 256 raw byte values as the base vocabulary.
  2. Count frequencies of all adjacent token pairs in the corpus.
  3. Merge the most frequent pair into a new token id; record the merge.
  4. Repeat until vocab_size is reached.

Encoding applies the learned merges greedily *in learned order* (merge order,
not encode-time frequency). Decoding concatenates the byte sequences each
token id maps to. Round-trip property: decode(encode(s)) == s for any UTF-8 s.
"""

import json


def get_stats(ids):
    """Count adjacent pair frequencies in a token-id sequence."""
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` (a, b) with the new token `idx`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (a, b) -> new_idx; insertion order == learned order
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> raw bytes

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def train(self, text, vocab_size=1024, verbose=False):
        """Learn merges on `text`. Returns the tokenization of the training
        text itself (useful to avoid re-encoding the corpus afterwards)."""
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        n0 = len(ids)
        for new_idx in range(256, vocab_size):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            ids = merge(ids, pair, new_idx)
            self.merges[pair] = new_idx
            self.vocab[new_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (new_idx - 256) % 128 == 0:
                print(f"merge {new_idx - 255:4d}: {pair} -> {new_idx} "
                      f"({self.vocab[new_idx]!r}), seq len {len(ids)}")
        if verbose:
            print(f"compression: {n0} bytes -> {len(ids)} tokens "
                  f"({n0 / len(ids):.2f} bytes/token)")
        return ids

    def encode(self, text):
        """Apply learned merges greedily in learned order."""
        ids = list(text.encode("utf-8"))
        for pair, idx in self.merges.items():  # dict preserves learned order
            if len(ids) < 2:
                break
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        data = [[a, b, idx] for (a, b), idx in self.merges.items()]
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for a, b, idx in data:  # list order == learned order
            self.merges[(a, b)] = idx
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        return self


if __name__ == "__main__":
    tok = BPETokenizer()
    s = "First Citizen:\nBefore we proceed any further, hear me speak."
    tok.train(s * 50, vocab_size=300)
    assert tok.decode(tok.encode(s)) == s, "round-trip failed"
    print("round-trip OK; vocab", tok.vocab_size)
