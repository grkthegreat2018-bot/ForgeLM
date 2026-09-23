"""Bench FluxLM: ingest tok/s, generation tok/s, RSS memory, recall probe.

    python scripts/bench_flux.py [--vocab 65536] [--tokens 200000]
"""
import argparse
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.model.flux import FluxConfig, FluxLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=200_000)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tracemalloc.start()
    cfg = FluxConfig(vocab_size=args.vocab, device=args.device)
    m = FluxLM(cfg)

    # synthetic corpus: repeated sentence patterns + noise
    import random
    rng = random.Random(0)
    words = [rng.randrange(args.vocab) for _ in range(256)]
    corpus = []
    for i in range(args.tokens):
        if i % 97 < 40:
            corpus.append(words[i % 256])
        else:
            corpus.append(rng.randrange(args.vocab))

    t0 = time.perf_counter()
    m.ingest(corpus, tag="bench")
    dt = time.perf_counter() - t0
    print(f"ingest: {len(corpus) / dt:,.0f} tok/s "
          f"({dt:.2f}s for {len(corpus):,})")

    # generation (predict-only loop, argmax)
    m.soft_reset()
    t0 = time.perf_counter()
    import torch
    for _ in range(args.gen):
        logits = m._predict()
        nxt = int(torch.argmax(logits).item())
        m._learn_token(nxt)
    dt = time.perf_counter() - t0
    print(f"generate: {args.gen / dt:,.0f} tok/s ({dt:.2f}s for {args.gen})")

    rep = m.memory_report()
    cur, peak = tracemalloc.get_traced_memory()
    print(f"memory_est: {rep['total_MB_est']:.0f} MB | "
          f"tracemalloc peak: {peak / 1e6:.0f} MB")
    print(f"channels: {m.channel_weights()}")
    print(f"journal: {len(m.journal):,} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
