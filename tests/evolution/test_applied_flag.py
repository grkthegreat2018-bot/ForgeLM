"""Test the applied-flag + train-first flow."""
import sys, os, json, tempfile
sys.path.insert(0, r"D:\windsurf\ForgeAI")
os.environ['PYTHONUTF8'] = '1'

from research.evolution.database import FindingsDB

# Use a temp DB
tmp = tempfile.mktemp(suffix=".db")
db = FindingsDB(tmp)

# Save a run first (needed for FK)
db.save_run("test_run", "test_domain", {"foo": 1},
            {"generations": 1, "total_evaluations": 1, "discoveries": 1,
             "best_score": 10.0, "best_config": {"a": 1}, "device": "cpu"},
            0.0)

# Save some discoveries
discs = [
    {"config": {"x": 1}, "score": 10.0, "metadata": {"problem": "2+2", "model_answer": "4"}},
    {"config": {"x": 2}, "score": 8.0, "metadata": {"problem": "3+3", "model_answer": "6"}},
    {"config": {"x": 3}, "score": 5.0, "metadata": {}},
]
n_saved, n_dup = db.save_discoveries("test_run", "test_domain", discs)
print(f"Saved {n_saved} discoveries")

# Check unapplied count
n = db.count_unapplied()
print(f"Unapplied count: {n}")
assert n == 3, f"Expected 3 unapplied, got {n}"

# Get unapplied
unapplied = db.get_unapplied_discoveries()
print(f"Got {len(unapplied)} unapplied discoveries")
assert len(unapplied) == 3
assert unapplied[0]["score"] == 10.0  # sorted by score DESC

# Check that problem/answer are in metadata
for d in unapplied:
    print(f"  id={d['id']} score={d['score']} meta={d.get('metadata')}")

# Mark first 2 as applied
db.mark_applied([unapplied[0]["id"], unapplied[1]["id"]])
n = db.count_unapplied()
print(f"After marking 2 applied: {n} unapplied")
assert n == 1

# Mark all applied
n_marked = db.mark_all_applied()
print(f"mark_all_applied returned: {n_marked}")
n = db.count_unapplied()
print(f"After mark_all: {n} unapplied")
assert n == 0

# Test domain filter
discs2 = [{"config": {"y": 1}, "score": 3.0, "metadata": {}}]
db.save_discoveries("test_run2", "other_domain", discs2)
db.save_run("test_run2", "other_domain", {}, {}, 0.0)
n_all = db.count_unapplied()
n_test = db.count_unapplied("test_domain")
n_other = db.count_unapplied("other_domain")
print(f"Unapplied: all={n_all}, test_domain={n_test}, other_domain={n_other}")
assert n_all == 1
assert n_test == 0
assert n_other == 1

# Test get_unapplied with domain filter
unapplied_other = db.get_unapplied_discoveries("other_domain")
assert len(unapplied_other) == 1
print(f"other_domain unapplied: {unapplied_other[0]['domain']}")

# Cleanup
os.unlink(tmp)
print("\nALL TESTS PASSED")
