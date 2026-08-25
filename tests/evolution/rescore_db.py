"""ForgeEvolve DB rescore tool — re-evaluate discoveries with updated scoring.

When a domain's `evaluate()` function is fixed (e.g. to penalize a synthetic-
metric artifact), existing discoveries in the DB still have their old (wrong)
scores. This tool:

1. Re-evaluates every discovery in the specified domains with the current
   scoring function (deterministic via fixed seed).
2. Updates the score, behavioral, and metadata fields in-place.
3. Optionally deletes "bad ideas" — discoveries whose score dropped sharply
   (the old fake winners) or that fall below the domain's 25th percentile.
4. Updates the runs table best_score / best_config_json to match.
5. Prints a before/after summary.

Usage:
    # Dry-run: see what would change for the 7 fixed domains
    python rescore_db.py --db forge_evolve.db --dry-run

    # Apply rescore + remove bad ideas
    python rescore_db.py --db forge_evolve.db --prune

    # Rescore specific domains only
    python rescore_db.py --db forge_evolve.db --domains moe_routing,rope_config

    # Custom prune threshold (default: drop >50% = artifact)
    python rescore_db.py --db forge_evolve.db --prune --drop-threshold 0.4

The 7 domains fixed on 2026-08-24 (scoring artifacts):
  moe_routing, rope_config, scheduler_config, fp8_training_config,
  loss_config, mod_config, factorized_embed
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import argparse
import sqlite3
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parents[2] / "research" / "results"

# Domains whose scoring was fixed (class name → domain name()).
# When adding new scoring fixes, just add the domain name here.
FIXED_DOMAINS = {
    # Round 1 (2026-08-24): synthetic-metric artifacts
    "moe_routing": "MoeRouting",
    "rope_config": "RopeConfig",
    "rope_config_refine_d1": "RopeConfig",
    "scheduler_config": "SchedulerConfig",
    "scheduler_config_refine_d1": "SchedulerConfig",
    "fp8_training_config": "Fp8TrainingConfig",
    "loss_config": "LossConfig",
    "mod_config": "ModConfig",
    "factorized_embed": "FactorizedEmbed",
    # Round 2 (2026-08-24): decoded-but-not-scored + trivial solutions + missing tradeoffs
    "speculative_decode": "SpeculativeDecode",
    "mtp_config": "MtpConfig",
    "batched_decode": "BatchedDecode",
    "sampling_config": "SamplingConfig",
    "beam_search": "BeamSearch",
    "titan_memory": "TitanMemory",
    "gla_attention": "GlaAttention",
    "gta_attention": "GtaAttention",
    "cross_layer_kv": "CrossLayerKV",
    "xquant_kv": "XQuantKV",
    "xquant_kv_refine_d1": "XQuantKV",
    "kv_recompute": "KvRecompute",
    "hybrid_offload": "HybridOffload",
    "streaming_kv": "StreamingKV",
    # Round 3 (2026-08-25): R&D round 14 training speedup domains
    "apollo_config": "ApolloConfig",
    "bread_config": "BreadConfig",
    "flashoptim_config": "FlashOptimConfig",
    "triton_kernel_config": "TritonKernelConfig",
    "varlen_config": "VarlenConfig",
}


def get_db_path(db_arg: str) -> str:
    p = Path(db_arg)
    if not p.is_absolute():
        p = RESULTS_DIR / db_arg
    return str(p)


def load_domain(domain_name: str):
    """Instantiate the domain class for a given domain name (from DB)."""
    from research.evolution.domains import DOMAINS
    # Map domain name (from DB, e.g. "moe_routing") to class name.
    # Strip _refine_dN suffix for the class lookup.
    base_name = domain_name.replace("_refine_d1", "").replace("_refine_d2", "")
    class_name = FIXED_DOMAINS.get(domain_name) or FIXED_DOMAINS.get(base_name)
    if class_name is None:
        # Try direct class name match (capitalize)
        class_name = "".join(w.capitalize() for w in base_name.split("_"))
    if class_name not in DOMAINS:
        raise KeyError(f"No domain class '{class_name}' for DB domain '{domain_name}'. "
                       f"Available: {list(DOMAINS.keys())}")
    return DOMAINS[class_name]()


def rescore_domain(conn, domain_name: str, seed: int = 42,
                   prune: bool = False, prune_low: bool = False,
                   drop_threshold: float = 0.5,
                   dry_run: bool = False) -> dict:
    """Re-evaluate all discoveries for one domain. Returns summary stats."""
    c = conn.cursor()
    c.execute(
        "SELECT id, config_json, score, metadata_json, behavioral_json "
        "FROM discoveries WHERE domain = ? ORDER BY score DESC",
        (domain_name,),
    )
    rows = c.fetchall()
    if not rows:
        return {"domain": domain_name, "n": 0, "rescored": 0, "pruned": 0,
                "old_best": None, "new_best": None}

    domain = load_domain(domain_name)
    old_scores = [r[2] for r in rows]
    old_best = max(old_scores)

    updates = []      # (id, new_score, new_metadata, new_behavioral)
    prune_ids = []    # ids to delete
    new_scores = []

    for row in rows:
        did, config_json, old_score, meta_json, behav_json = row
        config = json.loads(config_json)

        # Deterministic re-evaluation: set seed per-discovery for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            result = domain.evaluate(config)
            new_score = float(result["score"])
            # Guard against NaN/inf scores — mark for pruning
            if not np.isfinite(new_score):
                new_score = -1e9
                new_meta = json.dumps({"rescore_error": f"non-finite score: {result['score']}"})
                new_behav = json.dumps(())
            else:
                new_meta = json.dumps(result.get("metadata", {}))
                new_behav = json.dumps(result.get("behavioral", ()))
        except Exception as e:
            # If evaluation fails (e.g. NaN), mark for pruning
            new_score = -1e9
            new_meta = json.dumps({"rescore_error": str(e)})
            new_behav = json.dumps(())

        new_scores.append(new_score)
        updates.append((did, new_score, new_meta, new_behav))

        # Prune fake winners: discoveries that dropped >drop_threshold (relative)
        # were synthetic-metric artifacts → remove them.
        if prune and old_score > 0:
            relative_drop = (old_score - new_score) / abs(old_score)
            if relative_drop > drop_threshold:
                prune_ids.append(did)

    new_best = max(new_scores) if new_scores else None

    # Optional aggressive cleanup: also prune below 25th percentile
    if prune_low and new_scores:
        q25 = float(np.percentile(new_scores, 25))
        for (did, ns, _, _) in updates:
            if ns < q25 and did not in prune_ids:
                prune_ids.append(did)

    if not dry_run:
        # Update scores in-place
        c.executemany(
            "UPDATE discoveries SET score = ?, metadata_json = ?, behavioral_json = ? "
            "WHERE id = ?",
            [(ns, nm, nb, did) for (did, ns, nm, nb) in updates],
        )
        # Delete pruned discoveries
        if prune_ids:
            placeholders = ",".join("?" * len(prune_ids))
            c.execute(f"DELETE FROM discoveries WHERE id IN ({placeholders})",
                      prune_ids)
        # Update runs table best_score / best_config_json
        c.execute(
            "SELECT run_id FROM runs WHERE domain = ?",
            (domain_name,),
        )
        for (run_id,) in c.fetchall():
            c.execute(
                "SELECT config_json, score FROM discoveries "
                "WHERE run_id = ? AND domain = ? "
                "ORDER BY score DESC LIMIT 1",
                (run_id, domain_name),
            )
            best = c.fetchone()
            if best:
                c.execute(
                    "UPDATE runs SET best_score = ?, best_config_json = ? "
                    "WHERE run_id = ?",
                    (best[1], best[0], run_id),
                )
        conn.commit()

    n_pruned = len(prune_ids) if prune else 0
    return {
        "domain": domain_name,
        "n": len(rows),
        "rescored": len(updates),
        "pruned": n_pruned,
        "old_best": old_best,
        "new_best": new_best,
        "old_mean": float(np.mean(old_scores)),
        "new_mean": float(np.mean(new_scores)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="forge_evolve.db", help="DB filename in research/results/")
    ap.add_argument("--domains", default=None,
                    help="Comma-separated domain names to rescore (default: all FIXED_DOMAINS)")
    ap.add_argument("--dry-run", action="store_true", help="Show changes without writing to DB")
    ap.add_argument("--prune", action="store_true",
                    help="Delete fake winners: discoveries whose score dropped >threshold (artifacts)")
    ap.add_argument("--prune-low", action="store_true",
                    help="Also delete discoveries below 25th percentile (aggressive cleanup)")
    ap.add_argument("--drop-threshold", type=float, default=0.5,
                    help="Relative score drop threshold for pruning (default: 0.5 = 50%% drop)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for deterministic re-eval")
    args = ap.parse_args()

    db_path = get_db_path(args.db)
    if not os.path.exists(db_path):
        print(f"ERROR: DB not found: {db_path}")
        sys.exit(1)

    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    else:
        domains = list(FIXED_DOMAINS.keys())

    print(f"DB: {db_path}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}"
          f"{' + PRUNE' if args.prune else ''}"
          f"{' + PRUNE-LOW' if args.prune_low else ''}")
    print(f"Domains: {len(domains)}")
    print(f"Seed: {args.seed}")
    print()

    conn = sqlite3.connect(db_path)
    summaries = []
    for domain_name in domains:
        # Check if domain exists in DB
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM discoveries WHERE domain = ?", (domain_name,))
        count = c.fetchone()[0]
        if count == 0:
            print(f"  {domain_name:40s}  (0 discoveries, skipping)")
            continue
        print(f"  Rescoring {domain_name} ({count} discoveries)...", end=" ", flush=True)
        try:
            s = rescore_domain(conn, domain_name, seed=args.seed,
                               prune=args.prune, prune_low=args.prune_low,
                               drop_threshold=args.drop_threshold,
                               dry_run=args.dry_run)
            summaries.append(s)
            delta_best = (s["new_best"] - s["old_best"]) if s["old_best"] else 0
            delta_mean = s["new_mean"] - s["old_mean"]
            prune_str = f", pruned={s['pruned']}" if args.prune else ""
            print(f"old_best={s['old_best']:.2f} -> new_best={s['new_best']:.2f}"
                  f" (d={delta_best:+.2f}), mean d={delta_mean:+.2f}{prune_str}")
        except Exception as e:
            print(f"ERROR: {e}")

    conn.close()

    # Summary table
    if summaries:
        print("\n" + "=" * 90)
        print(f"{'Domain':40s} {'N':>6s} {'Old Best':>10s} {'New Best':>10s}"
              f" {'d Best':>8s} {'Pruned':>7s}")
        print("-" * 90)
        total_pruned = 0
        for s in summaries:
            delta = (s["new_best"] - s["old_best"]) if s["old_best"] else 0
            total_pruned += s["pruned"]
            print(f"{s['domain']:40s} {s['n']:>6d} {s['old_best']:>10.2f}"
                  f" {s['new_best']:>10.2f} {delta:>+8.2f} {s['pruned']:>7d}")
        print("-" * 90)
        print(f"Total pruned: {total_pruned}")
        if args.dry_run:
            print("\n(DRY-RUN: no changes written. Re-run without --dry-run to apply.)")


if __name__ == "__main__":
    main()
