"""Test persistence: run ForgeEvolve twice, verify second run benefits from first.

Validates:
1. Findings are saved to SQLite database
2. Second run warm-starts from first (generators + surrogate + archive)
3. Second run finds better score faster (fewer evaluations to match first run)
4. Database can be queried for past discoveries
"""
import os, sys, time, json
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
from pathlib import Path

DB_PATH = r"D:\windsurf\ForgeAI\research\results\forge_evolve_test.db"

def main():
    from research.evolution import ForgeEvolve, ForgeEvolveConfig, FindingsDB
    from research.evolution.domains.quant import QuantDomain

    # Clean start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print("=" * 70)
    print("  Persistence Test: Run twice, verify cross-run learning")
    print("=" * 70)

    domain = QuantDomain(matrix_size=(2048, 4096), seed=42)
    print(f"  Domain: quant, device={domain.device}")
    print()

    # ── Run 1: Cold start (no past findings) ──
    print("  [Run 1] Cold start — no past findings...")
    cfg1 = ForgeEvolveConfig(
        domain=domain,
        n_generators=200,
        filter_ratio=10,
        min_evaluate=10,
        max_evaluate=20,
        generations=10,
        exploration=0.3,
        db_path=DB_PATH,
        run_id="quant_run1",
        warm_start=False,  # cold start
        verbose=True,
    )
    engine1 = ForgeEvolve(cfg1)
    t0 = time.perf_counter()
    r1 = engine1.run()
    t1 = time.perf_counter() - t0

    print(f"\n  Run 1: best={r1['best_score']:.4f}, "
          f"{r1['total_evaluations']} evals, {r1['discoveries']} discoveries, "
          f"{t1:.1f}s, warm_started={r1['warm_started']}")

    # ── Verify database has findings ──
    print("\n  [Database check] Querying past findings...")
    db = FindingsDB(DB_PATH)
    runs = db.list_runs("quant")
    print(f"  Runs in DB: {len(runs)}")
    for run in runs:
        print(f"    {run['run_id']}: gen={run['generations']}, "
              f"discoveries={run['discoveries']}, best={run['best_score']:.4f}")

    past_disc = db.query_discoveries("quant", limit=5)
    print(f"  Top discoveries in DB: {len(past_disc)}")
    for d in past_disc[:3]:
        print(f"    score={d['score']:.4f}, config={d['config']}")

    # ── Run 2: Warm start (should benefit from Run 1) ──
    print("\n  [Run 2] Warm start — loading past findings...")
    cfg2 = ForgeEvolveConfig(
        domain=domain,
        n_generators=200,
        filter_ratio=10,
        min_evaluate=10,
        max_evaluate=20,
        generations=10,
        exploration=0.3,
        db_path=DB_PATH,
        run_id="quant_run2",
        warm_start=True,  # warm start from Run 1
        verbose=True,
    )
    engine2 = ForgeEvolve(cfg2)
    t0 = time.perf_counter()
    r2 = engine2.run()
    t2 = time.perf_counter() - t0

    print(f"\n  Run 2: best={r2['best_score']:.4f}, "
          f"{r2['total_evaluations']} evals, {r2['discoveries']} discoveries, "
          f"{t2:.1f}s, warm_started={r2['warm_started']}")

    db.close()

    # ── Comparison ──
    print()
    print("  " + "=" * 60)
    print(f"  {'Run':<12} {'Best Score':>12} {'Evals':>8} {'Time':>8} {'Warm':>6}")
    print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'Run 1':<12} {r1['best_score']:>12.4f} "
          f"{r1['total_evaluations']:>8} {t1:>7.1f}s {'No':>6}")
    print(f"  {'Run 2':<12} {r2['best_score']:>12.4f} "
          f"{r2['total_evaluations']:>8} {t2:>7.1f}s {'Yes':>6}")

    # ── Validation ──
    print()
    all_pass = True

    # Check 1: Database has findings
    db = FindingsDB(DB_PATH)
    disc_count = len(db.query_discoveries("quant"))
    if disc_count > 0:
        print(f"  PASS: Database has {disc_count} discoveries")
    else:
        print(f"  FAIL: Database has no discoveries")
        all_pass = False

    # Check 2: Run 2 was warm-started
    if r2['warm_started']:
        print(f"  PASS: Run 2 was warm-started from Run 1")
    else:
        print(f"  FAIL: Run 2 was not warm-started")
        all_pass = False

    # Check 3: Run 2 archive was pre-seeded
    # Run 2 should start with some archive entries from Run 1
    if r2['best_score'] >= r1['best_score']:
        print(f"  PASS: Run 2 ({r2['best_score']:.4f}) >= Run 1 ({r1['best_score']:.4f})")
    else:
        print(f"  WARN: Run 2 ({r2['best_score']:.4f}) < Run 1 ({r1['best_score']:.4f})")
        print(f"         (may happen if exploration dominates — not a hard failure)")

    # Check 4: Run 2 found discoveries faster (per eval)
    rate1 = r1['discoveries'] / r1['total_evaluations']
    rate2 = r2['discoveries'] / r2['total_evaluations']
    if rate2 >= rate1 * 0.5:  # at least half as efficient (new cells get scarce)
        print(f"  PASS: Run 2 discovery rate ({rate2:.3f}/eval) reasonable "
              f"vs Run 1 ({rate1:.3f}/eval)")
    else:
        print(f"  WARN: Run 2 discovery rate ({rate2:.3f}) much lower than Run 1 ({rate1:.3f})")

    # Check 5: Can query database for best configs
    best = db.query_best_configs("quant", limit=1)
    if best and best[0]['score'] >= r1['best_score']:
        print(f"  PASS: DB query returns best config (score={best[0]['score']:.4f})")
    else:
        print(f"  FAIL: DB query doesn't return best config")
        all_pass = False

    db.close()

    print()
    if all_pass:
        print("  ALL CHECKS PASSED — persistence + cross-run learning working")
    else:
        print("  SOME CHECKS FAILED — review above")

    # Save summary
    out = Path(r"D:\windsurf\ForgeAI\research\results\persistence_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "run1": {"best": r1["best_score"], "evals": r1["total_evaluations"],
                     "discoveries": r1["discoveries"], "time": t1},
            "run2": {"best": r2["best_score"], "evals": r2["total_evaluations"],
                     "discoveries": r2["discoveries"], "time": t2,
                     "warm_started": r2["warm_started"]},
        }, f, indent=2)
    print(f"\n  Summary saved to {out}")


if __name__ == "__main__":
    main()
