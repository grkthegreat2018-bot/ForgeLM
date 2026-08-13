"""Discovery monitor — CLI to inspect the LLM's discovery database.

Lets the user catch issues and ensure quality by reading everything the LLM
has saved: thoughts, scripts, research, theories, discoveries, schema changes,
and a live event tail.

Usage:
    python -m research.self_play.discovery.discovery_monitor <command> [options]

Commands:
    stats       — row counts per table
    tail [N]    — last N events (default 30)
    thoughts [N]— recent thoughts (incl. sudo_think / musing)
    scripts [N] — recent scripts with stdout/stderr (truncated)
    research [N]— recent web research findings
    theories    — all theories with status + evidence tallies
    discoveries [N] — confirmed findings
    schema      — audited schema migrations (catch LLM-initiated changes)
    sessions    — session list with summaries
    epochs      — all epochs with quality/skill/compute + best/archived status
    distills    — distill run history (bloat removed, items filtered)
    readiness   — DB readiness score for triggering fine-tune
    query SQL   — run a read-only SQL query and print rows
    watch       — tail events live (polls every 2s, Ctrl-C to stop)
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

from research.paths import DATA_DIR
from research.self_play.discovery.discovery_db import DiscoveryDB


_DB_PATH = DATA_DIR / "discovery" / "discovery.sqlite3"


def _db() -> DiscoveryDB:
    return DiscoveryDB(_DB_PATH)


def _trunc(s: Any, n: int = 200) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "…"


def _print_rows(rows: list[dict], cols: list[str] | None = None) -> None:
    if not rows:
        print("  (none)")
        return
    cols = cols or list(rows[0].keys())
    widths = {c: max(len(c), max(len(_trunc(r.get(c, ""), 40)) for r in rows)) for c in cols}
    print("  " + " | ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + " | ".join(_trunc(r.get(c, ""), 40).ljust(widths[c]) for c in cols))


def cmd_stats(_args):
    counts = _db().table_counts()
    print("Discovery DB row counts:")
    for t, n in counts.items():
        print(f"  {t:22s} {n}")


def cmd_tail(args):
    n = int(args[0]) if args else 30
    rows = _db().recent("events", n)
    print(f"Last {len(rows)} events:")
    for r in rows:
        payload = json.loads(r["payload"]) if r["payload"] else {}
        ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        print(f"  [{ts}] {r['kind']:18s} {_trunc(json.dumps(payload, ensure_ascii=False), 120)}")


def cmd_thoughts(args):
    n = int(args[0]) if args else 20
    rows = _db().recent("thoughts", n)
    print(f"Recent thoughts ({len(rows)}):")
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        print(f"  [{ts}] #{r['id']} ({r['kind']}) {_trunc(r['content'], 160)}")


def cmd_scripts(args):
    n = int(args[0]) if args else 10
    rows = _db().recent("scripts", n)
    print(f"Recent scripts ({len(rows)}):")
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        ok = "OK" if r["returncode"] == 0 else f"rc={r['returncode']}"
        print(f"  [{ts}] #{r['id']} {ok} {r.get('exec_ms', 0):.0f}ms")
        print(f"    code: {_trunc(r['code'], 120)}")
        if r["stdout"]:
            print(f"    out:  {_trunc(r['stdout'], 120)}")
        if r["stderr"]:
            print(f"    err:  {_trunc(r['stderr'], 120)}")


def cmd_research(args):
    n = int(args[0]) if args else 10
    rows = _db().recent("research", n)
    print(f"Recent research ({len(rows)}):")
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        print(f"  [{ts}] q={_trunc(r['query'], 80)}")
        if r.get("url"):
            print(f"    url: {_trunc(r['url'], 120)}")
        if r.get("summary"):
            print(f"    sum: {_trunc(r['summary'], 160)}")


def cmd_theories(_args):
    rows = _db().query("SELECT * FROM theories ORDER BY ts DESC")
    print(f"Theories ({len(rows)}):")
    _print_rows(rows, cols=["id", "status", "evidence_for", "evidence_against", "statement"])


def cmd_discoveries(args):
    n = int(args[0]) if args else 20
    rows = _db().recent("discoveries", n)
    print(f"Discoveries ({len(rows)}):")
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        print(f"  [{ts}] #{r['id']} conf={r.get('confidence')} {_trunc(r['summary'], 200)}")


def cmd_schema(_args):
    rows = _db().query("SELECT * FROM schema_migrations ORDER BY ts DESC")
    print(f"Schema migrations ({len(rows)}) — LLM-initiated DDL audit:")
    for r in rows:
        ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        flag = "OK " if r["success"] else "FAIL"
        print(f"  [{ts}] {flag} reason={_trunc(r['reason'], 80)}")
        print(f"    sql: {_trunc(r['sql'], 160)}")
        if r["error"]:
            print(f"    err: {_trunc(r['error'], 160)}")


def cmd_sessions(_args):
    rows = _db().query("SELECT * FROM sessions ORDER BY started DESC")
    print(f"Sessions ({len(rows)}):")
    for r in rows:
        st = time.strftime("%H:%M:%S", time.localtime(r["started"]))
        en = time.strftime("%H:%M:%S", time.localtime(r["ended"])) if r["ended"] else "—"
        print(f"  {r['id']} {st}..{en}  {_trunc(r['summary'], 120)}")


def cmd_query(args):
    sql = " ".join(args)
    try:
        rows = _db().query(sql)
        print(f"{len(rows)} rows:")
        _print_rows(rows)
    except Exception as e:
        print(f"query error: {e}")


def cmd_epochs(_args):
    rows = _db().query(
        "SELECT id, epoch_num, kind, status, quality, skill, compute, composite, "
        "checkpoint_path, ts FROM epochs ORDER BY epoch_num DESC")
    print(f"Epochs ({len(rows)}):")
    if not rows:
        print("  (none — no fine-tune has run yet)")
        return
    for r in rows:
        ts = time.strftime("%m-%d %H:%M", time.localtime(r["ts"]))
        comp = r["composite"]
        comp_s = f"{comp:.3f}" if comp is not None else "—"
        q = r["quality"]; s = r["skill"]; c = r["compute"]
        q_s = f"{q:.2f}" if q is not None else "—"
        s_s = f"{s:.2f}" if s is not None else "—"
        c_s = f"{c:.2f}" if c is not None else "—"
        print(f"  ep#{r['epoch_num']:>3} [{r['kind']:7s}] {r['status']:9s} "
              f"comp={comp_s} (q={q_s} s={s_s} c={c_s}) {ts}")
        print(f"    ckpt: {_trunc(r['checkpoint_path'], 100)}")


def cmd_distills(_args):
    rows = _db().query("SELECT * FROM distill_runs ORDER BY ts DESC")
    print(f"Distill runs ({len(rows)}):")
    if not rows:
        print("  (none — distill triggers every 12 epochs)")
        return
    for r in rows:
        ts = time.strftime("%m-%d %H:%M", time.localtime(r["ts"]))
        print(f"  ep#{r['epoch_num']} {ts}  from ep#{r['from_epoch']} -> ep#{r['to_epoch']}  "
              f"filtered={r['filtered_items']} bloat_removed={r['bloat_removed']}")
        if r["notes"]:
            print(f"    notes: {_trunc(r['notes'], 120)}")


def cmd_readiness(_args):
    from research.self_play.discovery.epoch_manager import _db_quality_score, db_is_ready
    db = _db()
    score = _db_quality_score(db)
    ready = db_is_ready(db)
    counts = db.table_counts()
    print(f"DB readiness score: {score:.3f}  (threshold 0.60)")
    print(f"Ready to fine-tune: {ready}")
    print(f"  thoughts={counts['thoughts']}  scripts={counts['scripts']}  "
          f"theories={counts['theories']}  discoveries={counts['discoveries']}  "
          f"research={counts['research']}")
    print(f"  epochs run: {counts['epochs']}  (distill triggers every 12)")


def cmd_watch(_args):
    db = _db()
    seen = 0
    print("Watching events (Ctrl-C to stop)…")
    try:
        while True:
            rows = db.recent("events", 200)
            new = rows[len(rows) - seen:] if seen < len(rows) else []
            if seen == 0:
                new = rows[-10:]
            for r in new:
                ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
                payload = json.loads(r["payload"]) if r["payload"] else {}
                print(f"  [{ts}] {r['kind']:18s} {_trunc(json.dumps(payload, ensure_ascii=False), 120)}")
            seen = len(rows)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nstopped.")


_COMMANDS = {
    "stats": cmd_stats, "tail": cmd_tail, "thoughts": cmd_thoughts,
    "scripts": cmd_scripts, "research": cmd_research, "theories": cmd_theories,
    "discoveries": cmd_discoveries, "schema": cmd_schema,
    "sessions": cmd_sessions, "query": cmd_query, "watch": cmd_watch,
    "epochs": cmd_epochs, "distills": cmd_distills, "readiness": cmd_readiness,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    fn = _COMMANDS.get(cmd)
    if fn is None:
        print(f"unknown command: {cmd}\n")
        print(__doc__)
        return 1
    fn(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
