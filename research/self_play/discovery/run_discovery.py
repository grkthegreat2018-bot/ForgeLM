"""Entry point for the discovery self-play loop.

Usage:
    python -m research.self_play.discovery.run_discovery [--steps N] [--temp T]
                [--max-gen N] [--idle N] [--no-resume] [--cpu]

Examples:
    # Default: 200 steps, resume from last session's findings
    python -m research.self_play.discovery.run_discovery

    # Short exploratory run
    python -m research.self_play.discovery.run_discovery --steps 40 --temp 0.8

    # Inspect results afterwards:
    python -m research.self_play.discovery.discovery_monitor stats
    python -m research.self_play.discovery.discovery_monitor theories
    python -m research.self_play.discovery.discovery_monitor schema
    python -m research.self_play.discovery.discovery_monitor watch
"""
from __future__ import annotations

import argparse
import sys

from research.paths import DATA_DIR
from research.self_play.discovery.discovery_db import DiscoveryDB
from research.self_play.discovery.discovery_loop import DiscoveryLoop


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Discovery-based self-play loop")
    p.add_argument("--steps", type=int, default=200, help="max agentic steps")
    p.add_argument("--temp", type=float, default=0.7, help="sampling temperature")
    p.add_argument("--max-gen", type=int, default=320, help="max tokens per turn")
    p.add_argument("--idle", type=int, default=3,
                   help="stop after N consecutive turns with no tool call")
    p.add_argument("--no-resume", action="store_true",
                   help="don't seed the prompt with the last session's recap")
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    p.add_argument("--db", type=str, default=None, help="override DB path")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="override checkpoint path (default: best epoch or ForgeLM V2 base)")
    p.add_argument("--no-auto-advance", action="store_true",
                   help="disable automatic fine-tune/distill after each session")
    args = p.parse_args(argv)

    db_path = args.db or str(DATA_DIR / "discovery" / "discovery.sqlite3")
    db = DiscoveryDB(db_path)

    device = "cpu" if args.cpu else "cuda"
    print(f"[discovery] device={device} db={db_path} steps={args.steps}")

    if args.checkpoint:
        # Explicit override — load the given checkpoint directly.
        from research.model_loader import load_default_model
        print(f"[discovery] loading explicit checkpoint: {args.checkpoint}")
        model, tok = load_default_model("forgelm_v7",
                                        checkpoint_path=args.checkpoint)
        loop = DiscoveryLoop(model, tok, db, max_gen_tokens=args.max_gen,
                            temperature=args.temp, idle_limit=args.idle,
                            device=device, auto_advance=not args.no_auto_advance)
    else:
        # Auto-resolve: best epoch > ForgeLM V2 base > random.
        loop = DiscoveryLoop.from_default_model(
            db_path=db_path, max_gen_tokens=args.max_gen,
            temperature=args.temp, idle_limit=args.idle, device=device,
            auto_advance=not args.no_auto_advance)
    loop.device = device

    result = loop.run(max_steps=args.steps, resume=not args.no_resume)
    print("[discovery] done:", result)
    print("[discovery] inspect with: python -m research.self_play.discovery.discovery_monitor stats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
