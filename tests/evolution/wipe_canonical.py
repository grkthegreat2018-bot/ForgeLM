"""Wipe canonical generator/surrogate tables."""
import sqlite3

db_path = r"D:\windsurf\ForgeAI\research\results\forge_evolve.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

for t in ['canonical_generators', 'canonical_surrogate', 'generators', 'surrogate']:
    try:
        n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        c.execute(f"DELETE FROM {t}")
        print(f"  {t}: wiped {n} rows")
    except sqlite3.OperationalError:
        print(f"  {t}: doesn't exist")

conn.commit()
conn.close()
print("Done.")
