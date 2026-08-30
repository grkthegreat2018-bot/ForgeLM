"""Scan the evolution DB for results analysis."""
import sqlite3, json, os
from collections import defaultdict

db_path = r"D:\windsurf\ForgeAI\research\results\forge_evolve.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# 1. Overall counts
total = c.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
applied = c.execute("SELECT COUNT(*) FROM discoveries WHERE applied=1").fetchone()[0]
unapplied = c.execute("SELECT COUNT(*) FROM discoveries WHERE applied=0 OR applied IS NULL").fetchone()[0]
print(f"=== DB Overview ===")
print(f"Total discoveries: {total:,}")
print(f"Applied: {applied:,} | Unapplied: {unapplied:,}")
print()

# 2. Per-domain summary
print(f"=== Per-Domain Summary (top 20 by best score) ===")
print(f"{'Domain':<35} {'Count':>6} {'Best':>10} {'Mean':>10} {'Applied':>8}")
print("-" * 75)
rows = c.execute("""
    SELECT domain, COUNT(*) as n, MAX(score) as best, AVG(score) as mean,
           SUM(CASE WHEN applied=1 THEN 1 ELSE 0 END) as n_applied
    FROM discoveries
    GROUP BY domain
    ORDER BY best DESC
    LIMIT 30
""").fetchall()
for r in rows:
    print(f"{r['domain']:<35} {r['n']:>6} {r['best']:>10.2f} {r['mean']:>10.2f} {r['n_applied']:>8}")
print()

# 3. Domains with 0 score or negative (potential scoring issues)
print(f"=== Domains with best score <= 0 (potential scoring issues) ===")
rows = c.execute("""
    SELECT domain, COUNT(*) as n, MAX(score) as best, MIN(score) as worst
    FROM discoveries
    GROUP BY domain
    HAVING best <= 0
    ORDER BY best ASC
""").fetchall()
for r in rows:
    print(f"  {r['domain']:<35} n={r['n']:>5} best={r['best']:>8.2f} worst={r['worst']:>8.2f}")
print()

# 4. Domains with very few discoveries (potential eval failures)
print(f"=== Domains with < 10 discoveries (potential eval issues) ===")
rows = c.execute("""
    SELECT domain, COUNT(*) as n, MAX(score) as best
    FROM discoveries
    GROUP BY domain
    HAVING n < 10
    ORDER BY n ASC
""").fetchall()
for r in rows:
    print(f"  {r['domain']:<35} n={r['n']:>5} best={r['best']:>8.2f}")
print()

# 5. Top 10 best configs overall
print(f"=== Top 10 Discoveries Overall ===")
rows = c.execute("""
    SELECT domain, score, config_json, behavioral_json
    FROM discoveries
    ORDER BY score DESC
    LIMIT 10
""").fetchall()
for r in rows:
    cfg = json.loads(r['config_json']) if r['config_json'] else {}
    cfg_short = {k: (round(v,3) if isinstance(v,float) else v) for k,v in list(cfg.items())[:5]}
    print(f"  {r['domain']:<30} score={r['score']:>10.2f} cfg={cfg_short}")
print()

# 6. Score distribution
print(f"=== Score Distribution ===")
bins = [(-1e9, -100), (-100, -50), (-50, -10), (-10, 0), (0, 10), (10, 50), (50, 100), (100, 1e9)]
for lo, hi in bins:
    n = c.execute("SELECT COUNT(*) FROM discoveries WHERE score >= ? AND score < ?", (lo, hi)).fetchone()[0]
    label = f"[{lo}, {hi})" if lo > -1e9 else f"[-inf, {hi})"
    bar = "#" * min(n // 100, 50)
    print(f"  {label:>15}: {n:>6} {bar}")
print()

# 7. Recent runs
print(f"=== Recent Runs (last 10) ===")
rows = c.execute("""
    SELECT run_id, domain, best_score, discoveries, generations, start_time, end_time
    FROM runs
    ORDER BY end_time DESC
    LIMIT 10
""").fetchall()
for r in rows:
    print(f"  {r['run_id']:<40} best={r['best_score']:>10.2f} disc={r['discoveries']:>4} gen={r['generations']:>3}")
print()

# 8. Refinement domain performance
print(f"=== Refinement Domains ===")
rows = c.execute("""
    SELECT domain, COUNT(*) as n, MAX(score) as best
    FROM discoveries
    WHERE domain LIKE '%refine%'
    GROUP BY domain
    ORDER BY best DESC
    LIMIT 15
""").fetchall()
for r in rows:
    print(f"  {r['domain']:<45} n={r['n']:>5} best={r['best']:>8.2f}")

conn.close()
