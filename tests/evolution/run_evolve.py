"""ForgeEvolve runner — single entry point for all evolution runs.

Driven by JSON configs in tests/evolution/configs/.
Replaces boot_evolve.py, deep_evolve.py, test_multi_domain.py, test_long.py.

Usage:
    # Boot run (short, all domains)
    python run_evolve.py --profile boot

    # Deep run (long, all domains)
    python run_evolve.py --profile deep

    # Specific category
    python run_evolve.py --profile deep --category quantization

    # Specific domains
    python run_evolve.py --profile deep --domains QuantDomain,AaacQuant

    # Custom overrides
    python run_evolve.py --profile deep --gens 200 --gen-pop 1000

    # Smoke test
    python run_evolve.py --profile smoke --domains SyntheticDomain
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import time
import json
import argparse
import torch
import numpy as np
from pathlib import Path

from research.evolution import ForgeEvolve, ForgeEvolveConfig
from research.evolution.domains import DOMAINS, list_domains
from research.evolution.database import FindingsDB

CONFIG_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "research" / "results"


def load_json(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return json.load(f)


def resolve_domains(category: str | None, domain_list: str | None) -> list[str]:
    """Resolve which domains to run from category or explicit list."""
    if domain_list:
        names = [d.strip() for d in domain_list.split(",")]
        for n in names:
            if n not in DOMAINS:
                raise KeyError(f"Unknown domain '{n}'. Available: {list_domains()}")
        return names

    if category is None or category == "all":
        return list_domains()

    cats = load_json("domain_categories.json")["categories"]
    if category not in cats:
        raise KeyError(f"Unknown category '{category}'. Available: {list(cats.keys())}")
    domains = cats[category]["domains"]
    return [d for d in domains if d in DOMAINS]


def build_config(profile: str, overrides: dict, domain_name: str, db_path: str) -> ForgeEvolveConfig:
    """Build ForgeEvolveConfig from a JSON profile + CLI overrides."""
    profiles = load_json("run_profiles.json")
    if profile not in profiles["profiles"]:
        raise KeyError(f"Unknown profile '{profile}'. Available: {list(profiles['profiles'].keys())}")

    p = profiles["profiles"][profile]
    defaults = profiles.get("defaults", {})

    cfg_kwargs = {
        "domain": DOMAINS[domain_name](),
        "n_generators": overrides.get("gen_pop", p["n_generators"]),
        "filter_ratio": p["filter_ratio"],
        "min_evaluate": p["min_evaluate"],
        "max_evaluate": p["max_evaluate"],
        "generations": overrides.get("gens", p["generations"]),
        "exploration": p["exploration"],
        "parallel_eval": defaults.get("parallel_eval", 1),
        "db_path": db_path,
        "run_id": f"{domain_name}_{profile}",
        "warm_start": p["warm_start"],
        "verbose": defaults.get("verbose", False),
        "log_every": p["log_every"],
    }
    return ForgeEvolveConfig(**cfg_kwargs)


def run_one_domain(name: str, profile: str, overrides: dict, db_path: str) -> tuple:
    """Run ForgeEvolve on one domain. Returns (results, time_s, error)."""
    try:
        cfg = build_config(profile, overrides, name, db_path)
    except Exception as e:
        return None, 0, str(e)

    t0 = time.time()
    try:
        engine = ForgeEvolve(cfg)
        results = engine.run()
        elapsed = time.time() - t0
        torch.cuda.empty_cache()
        return results, elapsed, None
    except Exception as e:
        torch.cuda.empty_cache()
        return None, time.time() - t0, str(e)[:200]


def extract_top_discoveries(db_path: str, top_n: int = 20) -> dict:
    """Extract top discoveries per domain from the DB."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT domain, config_json, score, behavioral_json, metadata_json,
               run_id, generation
        FROM discoveries WHERE score IS NOT NULL
        ORDER BY domain, score DESC
    """)
    by_domain = {}
    for row in c.fetchall():
        domain = row[0]
        if domain not in by_domain:
            by_domain[domain] = []
        try:
            by_domain[domain].append({
                "config": json.loads(row[1]), "score": row[2],
                "behavioral": json.loads(row[3]) if row[3] else [],
                "metadata": json.loads(row[4]) if row[4] else {},
                "run_id": row[5], "generation": row[6],
            })
        except (json.JSONDecodeError, TypeError):
            continue
    conn.close()
    return {d: sorted(items, key=lambda x: x["score"], reverse=True)[:top_n]
            for d, items in by_domain.items()}


def generate_ideas_report(top: dict, profile: str) -> str:
    """Generate markdown report of best optimization ideas."""
    L = [f"# ForgeEvolve '{profile}' Run: Top Optimization Ideas\n",
         f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
         f"Domains: {len(top)}\n\n"]

    # Tier 1: highest scores overall
    all_items = [(d, i) for d, items in top.items() for i in items]
    all_items.sort(key=lambda x: x[1]["score"], reverse=True)

    L.append("## Tier 1: Top 50 Configurations\n\n")
    L.append("| Rank | Domain | Score | Config |\n|------|--------|-------|--------|\n")
    for idx, (domain, item) in enumerate(all_items[:50]):
        cfg = json.dumps(item["config"], default=str)
        if len(cfg) > 80: cfg = cfg[:77] + "..."
        L.append(f"| {idx+1} | {domain} | {item['score']:.2f} | `{cfg}` |\n")

    # Tier 2: best per domain
    L.append("\n## Tier 2: Best Per Domain\n\n")
    for domain in sorted(top.keys()):
        items = top[domain]
        if not items: continue
        best = items[0]
        L.append(f"### {domain} (score={best['score']:.2f})\n")
        L.append(f"- **Config**: `{json.dumps(best['config'], default=str)}`\n")
        L.append(f"- **Metadata**: `{json.dumps(best['metadata'], default=str)}`\n\n")

    # Tier 3: cross-domain patterns
    L.append("## Tier 3: Cross-Domain Parameter Patterns\n\n")
    key_patterns = {}
    for domain, items in top.items():
        for item in items[:3]:
            for key, val in item["config"].items():
                key_patterns.setdefault(key, []).append((domain, val, item["score"]))
    for key in sorted(key_patterns.keys()):
        entries = key_patterns[key]
        if len(entries) < 3: continue
        L.append(f"### `{key}`\n")
        val_counts = {}
        for d, v, s in entries:
            vs = str(v)
            vc = val_counts.setdefault(vs, {"n": 0, "scores": [], "domains": set()})
            vc["n"] += 1; vc["scores"].append(s); vc["domains"].add(d)
        for val, info in sorted(val_counts.items(), key=lambda x: -x[1]["n"])[:5]:
            L.append(f"- `{val}`: {info['n']}x, avg={np.mean(info['scores']):.1f}, "
                    f"domains={','.join(sorted(info['domains'])[:5])}\n")
        L.append("")

    return "".join(L)


def print_summary(all_results: list, total_t: float, profile: str):
    """Print summary table."""
    valid = [(n, r, t, e) for n, r, t, e in all_results if r is not None]
    failed = [(n, r, t, e) for n, r, t, e in all_results if r is None]

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {len(all_results)} domains in {total_t:.1f}s ({total_t/60:.1f}m) [{profile}]")
    print(f"{'='*70}")
    print(f"  {'Domain':<30s} {'Best Score':>12s} {'Disc':>5s} {'Archive':>8s} {'Time':>7s}")
    print(f"  {'-'*30} {'-'*12} {'-'*5} {'-'*8} {'-'*7}")

    for name, results, t, _ in sorted(valid, key=lambda x: x[1]["best_score"], reverse=True):
        print(f"  {name:<30s} {results['best_score']:12.4f} "
              f"{results['discoveries']:5d} {results['archive_coverage']*100:7.0f}% {t:6.1f}s")

    if failed:
        print(f"\n  FAILED ({len(failed)}):")
        for name, _, t, err in failed:
            print(f"    {name}: {err[:80]}")

    print(f"\n  Valid: {len(valid)}/{len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description="ForgeEvolve runner")
    parser.add_argument("--profile", default="boot", help="Run profile from config (boot/deep/ultra/smoke)")
    parser.add_argument("--category", default=None, help="Domain category from config")
    parser.add_argument("--domains", default=None, help="Comma-separated domain names (overrides category)")
    parser.add_argument("--gens", type=int, default=None, help="Override generations")
    parser.add_argument("--gen-pop", type=int, default=None, help="Override generator population")
    parser.add_argument("--no-ideas", action="store_true", help="Skip ideas report generation")
    args = parser.parse_args()

    names = resolve_domains(args.category, args.domains)
    profiles = load_json("run_profiles.json")
    db_suffix = profiles["profiles"][args.profile]["db_suffix"]
    db_path = str(RESULTS_DIR / f"forge_evolve{db_suffix}.db")

    overrides = {}
    if args.gens is not None:
        overrides["gens"] = args.gens
    if args.gen_pop is not None:
        overrides["gen_pop"] = args.gen_pop

    print(f"{'='*70}")
    print(f"  ForgeEvolve [{args.profile}] — {len(names)} domains")
    print(f"  DB: {db_path}")
    print(f"  Warm start: {profiles['profiles'][args.profile]['warm_start']}")
    print(f"{'='*70}\n")

    all_results = []
    t_start = time.time()

    for i, name in enumerate(names):
        print(f"[{i+1:3d}/{len(names)}] {name}...", end=" ", flush=True)
        torch.cuda.empty_cache()
        results, t, err = run_one_domain(name, args.profile, overrides, db_path)
        if err:
            print(f"ERROR ({t:.1f}s): {err[:60]}")
            all_results.append((name, None, t, err))
        else:
            best = results["best_score"]
            disc = results["discoveries"]
            cov = results["archive_coverage"]
            flag = " <<< FREEZE" if t > 30 else (" << slow" if t > 15 else "")
            print(f"best={best:10.4f}, {disc:4d} disc, archive={cov*100:.0f}%, {t:.1f}s{flag}")
            all_results.append((name, results, t, None))

    total_t = time.time() - t_start
    print_summary(all_results, total_t, args.profile)

    # Save JSON summary
    valid = [(n, r, t) for n, r, t, e in all_results if r is not None]
    summary = {
        "profile": args.profile, "total_domains": len(names),
        "total_time_s": total_t, "total_time_m": total_t / 60,
        "valid_runs": len(valid), "failed_runs": len(all_results) - len(valid),
        "results": [{"domain": n, "best_score": r["best_score"],
                      "discoveries": r["discoveries"],
                      "archive_coverage": r["archive_coverage"],
                      "time_s": t, "best_config": r.get("best_config")}
                     for n, r, t in valid],
    }
    summary_path = RESULTS_DIR / f"evolve{db_suffix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary: {summary_path}")

    # Ideas report
    if not args.no_ideas and valid:
        print(f"  Extracting discoveries...")
        top = extract_top_discoveries(db_path, top_n=20)
        ideas = generate_ideas_report(top, args.profile)
        ideas_path = RESULTS_DIR / f"evolve{db_suffix}_ideas.md"
        with open(ideas_path, "w") as f:
            f.write(ideas)
        print(f"  Ideas: {ideas_path}")

    print(f"\n  Done in {total_t/60:.1f}m")


if __name__ == "__main__":
    main()
