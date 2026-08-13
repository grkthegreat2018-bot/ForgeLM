"""Anti-regression for the discovery loop.

Two mechanisms, both grounded in the research literature on preventing
self-play collapse and catastrophic forgetting:

1. Fingerprint block (T-SPIN / SERS style "don't repeat history"):
   A set of normalized content hashes is built from the current DB AND all
   archived epoch DBs. Before the loop executes a write-tool (think,
   run_script, propose_theory, record_discovery, save_research), it hashes
   the proposed content. If the hash is already in the set, the call is
   blocked and the LLM is told to iterate/improve instead of repeating.
   The LLM may refine, extend, or build on prior work — it just can't
   re-record the exact same content.

2. Stuck-loop rollback (GeRe / on-policy forgetting control):
   The loop tracks "productive" steps (ones that added a new thought /
   script / theory / discovery / research row, or updated a theory with
   new evidence). Each productive step creates a SQLite SAVEPOINT and
   marks a transcript index. If N consecutive steps are unproductive,
   the loop rolls the DB back to the last savepoint and truncates the
   transcript — reverting all "bad actions" from the stuck burst so the
   model can't loop on the same dead end.

Both are designed to be cheap (hashing + a SAVEPOINT per step) so they
don't slow the 1.2B model's loop.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from research.self_play.discovery.discovery_db import DiscoveryDB


# Tools that write new content and must be fingerprint-checked.
_WRITE_TOOLS = {"think", "sudo_think", "run_script", "propose_theory",
                "record_discovery", "save_research"}

# Tool -> DB column whose content is fingerprinted.
_TOOL_CONTENT_KEY = {
    "think": "content", "sudo_think": "content",
    "run_script": "code", "propose_theory": "statement",
    "record_discovery": "summary", "save_research": "query",
}


def _normalize(s: str) -> str:
    """Aggressive normalization so cosmetic edits don't dodge the block:
    lowercase, collapse whitespace, strip non-alphanumeric."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()


def fingerprint(content: str) -> str:
    return hashlib.sha1(_normalize(content).encode("utf-8")).hexdigest()[:16]


class FingerprintSet:
    """Set of content hashes the LLM must not reproduce exactly."""

    def __init__(self):
        self._seen: set[str] = set()

    @classmethod
    def from_db(cls, db: DiscoveryDB) -> "FingerprintSet":
        fs = cls()
        for row in db.all_content_rows():
            fs._seen.add(fingerprint(row["content"]))
        return fs

    def add(self, content: str) -> None:
        self._seen.add(fingerprint(content))

    def contains(self, content: str) -> bool:
        return fingerprint(content) in self._seen

    def __len__(self) -> int:
        return len(self._seen)


class StuckDetector:
    """Tracks productive steps and triggers rollback after N idle ones.

    Usage in the loop:
        det = StuckDetector(idle_limit=3)
        sp = db.savepoint(); det.mark_checkpoint(sp, len(transcript))
        ... run a step ...
        if det.is_productive(step_result):
            db.release(sp); det.reset_idle()
        else:
            det.tick_idle()
            if det.should_rollback():
                db.rollback_to(sp); truncate transcript to det.transcript_len
    """

    def __init__(self, idle_limit: int = 3):
        self.idle_limit = idle_limit
        self._idle = 0
        self._savepoint: str | None = None
        self.transcript_len: int = 0

    def mark_checkpoint(self, savepoint: str, transcript_len: int) -> None:
        self._savepoint = savepoint
        self.transcript_len = transcript_len

    def tick_idle(self) -> None:
        self._idle += 1

    def reset_idle(self) -> None:
        self._idle = 0

    def should_rollback(self) -> bool:
        return self._idle >= self.idle_limit and self._savepoint is not None

    @property
    def idle(self) -> int:
        return self._idle


def is_write_tool(tool: str) -> bool:
    return tool in _WRITE_TOOLS


def tool_content(tool: str, args: dict) -> str:
    """Extract the content to fingerprint from a tool call."""
    key = _TOOL_CONTENT_KEY.get(tool)
    return (args.get(key, "") if key else "") or ""


def is_productive_step(tool: str, result: dict) -> bool:
    """A step is productive if it added/changed a knowledge row, or ran a
    script that succeeded with non-empty output. Idle musings + failed
    scripts + blocked repeats don't count."""
    if tool in {"think", "sudo_think", "propose_theory", "record_discovery",
                "save_research"}:
        return bool(result.get("saved") or result.get("thought_id")
                    or result.get("theory_id") or result.get("discovery_id")
                    or result.get("research_id"))
    if tool == "run_script":
        return bool(result.get("ok") and result.get("stdout", "").strip())
    if tool == "update_theory":
        return bool(result.get("updated"))
    if tool == "migrate_schema":
        return bool(result.get("applied"))
    if tool == "query_db":
        # Reading is not productive by itself — but a non-empty result means
        # the model is grounding. We count it as neutral (not idle, not
        # productive) by returning False; the loop treats it as a non-idle
        # step so it doesn't trigger rollback, but it also doesn't reset idle.
        return False
    return False


def is_neutral_step(tool: str) -> bool:
    """A step that neither advances nor idles (e.g. query_db, web_search)."""
    return tool in {"query_db", "web_search"}
