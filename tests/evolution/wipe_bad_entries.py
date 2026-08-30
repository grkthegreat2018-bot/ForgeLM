"""Wipe bad entries from DB + gen model knowledge."""
import sqlite3
import os
import shutil
from datetime import datetime

db_path = r"D:\windsurf\ForgeAI\research\results\forge_evolve.db"

# Backup first
bak_path = db_path + f".bak_wipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"Backing up DB to {bak_path}...")
shutil.copy2(db_path, bak_path)
print(f"  Backup: {os.path.getsize(bak_path) / 1e9:.2f} GB")

conn = sqlite3.connect(db_path)
c = conn.cursor()

# List tables
c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in c.fetchall()]
print(f"\nTables: {tables}")

# Domains with broken scoring that we just fixed — wipe their old discoveries
# since the scores are wrong
scoring_fixed_domains = [
    'quant_domain',
    'cross_layer_kv',
    'gla_attention',
    'cpu_kv_offload',
    'flashoptim_config',
]

# Domains that failed every loop due to surrogate grad error — their discoveries
# from loop 1 are stale (gen models trained on bad data)
# Actually these domains DID produce discoveries in loop 1 before crashing.
# The issue is they crashed in loops 2-15. Their loop 1 discoveries are fine.
# But we should wipe gen model knowledge so they start fresh.

print("\n=== Wiping discoveries from scoring-fixed domains ===")
total_wiped = 0
for domain in scoring_fixed_domains:
    # Wipe discoveries
    n = c.execute("SELECT COUNT(*) FROM discoveries WHERE domain=?", (domain,)).fetchone()[0]
    c.execute("DELETE FROM discoveries WHERE domain=?", (domain,))
    # Wipe runs
    n_runs = c.execute("SELECT COUNT(*) FROM runs WHERE domain=?", (domain,)).fetchone()[0]
    c.execute("DELETE FROM runs WHERE domain=?", (domain,))
    # Wipe canonical knowledge
    try:
        n_canon = c.execute("SELECT COUNT(*) FROM canonical_knowledge WHERE domain=?", (domain,)).fetchone()[0]
        c.execute("DELETE FROM canonical_knowledge WHERE domain=?", (domain,))
    except sqlite3.OperationalError:
        n_canon = 0
    print(f"  {domain}: wiped {n} discoveries, {n_runs} runs, {n_canon} canonical")
    total_wiped += n

print(f"\nTotal discoveries wiped: {total_wiped}")

# Now wipe ALL gen model knowledge (gen models trained on stale/bad data)
print("\n=== Wiping ALL gen model knowledge ===")
for table in ['gen_model_state', 'gen_models', 'gen_model_performance']:
    try:
        n = c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        c.execute(f"DELETE FROM {table}")
        print(f"  {table}: wiped {n} rows")
    except sqlite3.OperationalError:
        print(f"  {table}: doesn't exist")

# Also wipe canonical_knowledge for ALL domains (gen models are wiped, so
# canonical generators/surrogates are stale)
print("\n=== Wiping ALL canonical knowledge ===")
try:
    n = c.execute("SELECT COUNT(*) FROM canonical_knowledge").fetchone()[0]
    c.execute("DELETE FROM canonical_knowledge")
    print(f"  canonical_knowledge: wiped {n} rows")
except sqlite3.OperationalError:
    print("  canonical_knowledge: doesn't exist")

# Reset applied flags on all remaining discoveries (gen models are wiped,
# so TrainFirst should retrain on all past discoveries)
print("\n=== Resetting applied flags on all remaining discoveries ===")
n = c.execute("SELECT COUNT(*) FROM discoveries WHERE applied=1").fetchone()[0]
c.execute("UPDATE discoveries SET applied=0")
print(f"  Reset {n} discoveries to unapplied")

conn.commit()

# Final counts
total = c.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
total_runs = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
print(f"\n=== Final DB state ===")
print(f"  Discoveries: {total:,}")
print(f"  Runs: {total_runs:,}")

# Show remaining domains
rows = c.execute("""
    SELECT domain, COUNT(*) as n, MAX(score) as best
    FROM discoveries
    GROUP BY domain
    ORDER BY best DESC
    LIMIT 10
""").fetchall()
print(f"\n  Top 10 remaining domains:")
for r in rows:
    print(f"    {r[0]:<40} n={r[1]:>5} best={r[2]:>10.2f}")

conn.close()
print(f"\nDone. Backup at {bak_path}")
