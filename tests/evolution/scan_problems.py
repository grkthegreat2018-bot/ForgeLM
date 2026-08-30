"""Deep dive into problem domains."""
import sqlite3, json
from collections import Counter

db_path = r"D:\windsurf\ForgeAI\research\results\forge_evolve.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Focus domains with issues
problem_domains = [
    'quant_domain', 'cross_layer_kv', 'gla_attention', 'sharq_quant',
    'bitnet_config', 'mixed_precision', 'ffn_skip', 'offq_quant',
    'flashoptim_config', 'cpu_kv_offload',
]

for domain in problem_domains:
    rows = c.execute("""
        SELECT score, config_json, behavioral_json, metadata_json
        FROM discoveries WHERE domain=?
        ORDER BY score DESC LIMIT 3
    """, (domain,)).fetchall()
    n = c.execute("SELECT COUNT(*) FROM discoveries WHERE domain=?", (domain,)).fetchone()[0]
    if not rows:
        print(f"\n--- {domain} (n={n}) --- NO DISCOVERIES")
        continue
    print(f"\n--- {domain} (n={n}) ---")
    for r in rows:
        cfg = json.loads(r['config_json']) if r['config_json'] else {}
        beh = json.loads(r['behavioral_json']) if r['behavioral_json'] else {}
        meta = json.loads(r['metadata_json']) if r['metadata_json'] else {}
        cfg_short = {k: (round(v,3) if isinstance(v,float) else v) for k,v in cfg.items()}
        beh_short = beh if isinstance(beh, list) else {k: (round(v,3) if isinstance(v,float) else v) for k,v in list(beh.items())[:5]}
        print(f"  score={r['score']:>10.2f} | cfg={cfg_short}")
        if beh_short:
            print(f"           beh={beh_short}")
        if meta:
            meta_short = {k: str(v)[:60] for k,v in list(meta.items())[:3]}
            print(f"           meta={meta_short}")

# Check what domains are in the DB but NOT in the current focus
print("\n\n=== All domains in DB (sorted by best score) ===")
rows = c.execute("""
    SELECT domain, COUNT(*) as n, MAX(score) as best, MIN(score) as worst,
           AVG(score) as mean
    FROM discoveries
    GROUP BY domain
    ORDER BY best DESC
""").fetchall()
for r in rows:
    print(f"  {r['domain']:<45} n={r['n']:>5} best={r['best']:>10.2f} mean={r['mean']:>10.2f} worst={r['worst']:>10.2f}")

conn.close()
