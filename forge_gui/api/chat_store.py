"""ChatStore — persistent chat conversations with quality ratings + SFT export.

Pure-python (no Qt) so it is trivially unit-testable and reusable from
worker threads. Conversations live in ``data/chats/conversations.json``;
training-data exports are written to ``data/sft/*.jsonl`` in the exact
format ``research/training/runners/sft_train.py:load_examples`` consumes
(``{"messages": [{role, content}, ...]}`` per line).

Rating model (drives what becomes training data):
  - every assistant message carries ``rating``: "good" | "bad" | None
  - export emits, per good-rated assistant turn, the conversation prefix
    up to and including that turn (system + user + assistant), so the
    trainer learns to reproduce exactly the approved behavior.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

RATING_GOOD = "good"
RATING_BAD = "bad"


def _now() -> float:
    return time.time()


def _slug(title: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return (s[:limit] or "chat")


class ChatStore:
    """JSON-persisted conversation list with per-message ratings."""

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            root = Path(__file__).resolve().parents[2] / "data"
        self.root = Path(root)
        self.chats_dir = self.root / "chats"
        self.sft_dir = self.root / "sft"
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self.sft_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.chats_dir / "conversations.json"
        self.conversations: list[dict[str, Any]] = []
        self.load()

    # ── persistence ───────────────────────────────────────────────────
    def load(self) -> None:
        if not self.path.is_file():
            self.conversations = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.conversations = data.get("conversations", [])
        except Exception:
            self.conversations = []

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "conversations": self.conversations},
                      f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    # ── conversation CRUD ─────────────────────────────────────────────
    def create(self, title: str = "New chat", model: str = "") -> dict[str, Any]:
        conv = {
            "id": uuid.uuid4().hex[:12],
            "title": title,
            "model": model,
            "created_at": _now(),
            "updated_at": _now(),
            "messages": [],
        }
        self.conversations.insert(0, conv)
        self.save()
        return conv

    def get(self, conv_id: str) -> Optional[dict[str, Any]]:
        for c in self.conversations:
            if c["id"] == conv_id:
                return c
        return None

    def delete(self, conv_id: str) -> bool:
        before = len(self.conversations)
        self.conversations = [c for c in self.conversations if c["id"] != conv_id]
        changed = len(self.conversations) != before
        if changed:
            self.save()
        return changed

    def rename(self, conv_id: str, title: str) -> None:
        conv = self.get(conv_id)
        if conv is not None:
            conv["title"] = title or "Untitled"
            conv["updated_at"] = _now()
            self.save()

    def touch(self, conv_id: str, model: str = "") -> None:
        conv = self.get(conv_id)
        if conv is None:
            return
        conv["updated_at"] = _now()
        if model:
            conv["model"] = model
        # auto-title from the first user message
        if conv["title"] in ("New chat", "Untitled"):
            for m in conv["messages"]:
                if m.get("role") == "user" and m.get("content"):
                    conv["title"] = m["content"].strip().splitlines()[0][:60]
                    break
        self.save()

    # ── messages ──────────────────────────────────────────────────────
    def append_message(self, conv_id: str, role: str, content: str,
                       rating: Optional[str] = None,
                       image: str = "") -> int:
        conv = self.get(conv_id)
        if conv is None:
            raise KeyError(conv_id)
        msg = {
            "role": role, "content": content,
            "rating": rating, "ts": _now(),
        }
        if image:
            msg["image"] = image
        conv["messages"].append(msg)
        conv["updated_at"] = _now()
        self.save()
        return len(conv["messages"]) - 1

    def rate_message(self, conv_id: str, msg_idx: int,
                     rating: Optional[str]) -> Optional[str]:
        """Set/clear a message rating ('good' | 'bad' | None). Toggles."""
        conv = self.get(conv_id)
        if conv is None or not (0 <= msg_idx < len(conv["messages"])):
            return None
        cur = conv["messages"][msg_idx].get("rating")
        new = None if (rating is None or cur == rating) else rating
        conv["messages"][msg_idx]["rating"] = new
        conv["updated_at"] = _now()
        self.save()
        return new

    @staticmethod
    def count_ratings(conv: dict[str, Any]) -> tuple[int, int]:
        good = sum(1 for m in conv["messages"] if m.get("rating") == RATING_GOOD)
        bad = sum(1 for m in conv["messages"] if m.get("rating") == RATING_BAD)
        return good, bad

    # ── training-data export ──────────────────────────────────────────
    def export_training_data(self, conv_ids: Optional[list[str]] = None,
                             out_path: Optional[Path] = None,
                             include_system: bool = True) -> tuple[Path, int]:
        """Export good-rated turns as sft_train-compatible JSONL.

        Each good-rated assistant message becomes one line:
          {"messages": [system?], {"role": "user", ...}, {"role": "assistant", ...}}
        The prefix is truncated at the good turn, so multi-turn conversations
        yield one example per approved reply. Returns (path, n_examples).
        """
        ids = set(conv_ids) if conv_ids else {c["id"] for c in self.conversations}
        examples: list[dict] = []
        for conv in self.conversations:
            if conv["id"] not in ids:
                continue
            prefix: list[dict] = []
            for m in conv["messages"]:
                role, content = m.get("role"), m.get("content", "")
                if role == "system":
                    if include_system and content.strip():
                        prefix.append({"role": "system", "content": content})
                    continue
                if role == "user":
                    prefix.append({"role": "user", "content": content})
                elif role == "assistant":
                    if m.get("rating") == RATING_GOOD and content.strip():
                        messages = [p for p in prefix if p["content"].strip()]
                        messages.append({"role": "assistant", "content": content})
                        examples.append({"messages": messages})
                    prefix.append({"role": "assistant", "content": content})
        if out_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = self.sft_dir / f"forge_chats_{stamp}.jsonl"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return out_path, len(examples)

    def list_exports(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.sft_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            try:
                n = sum(1 for line in p.read_text(encoding="utf-8").splitlines()
                        if line.strip())
            except OSError:
                n = 0
            out.append({"path": str(p), "name": p.name, "examples": n,
                        "size": p.stat().st_size, "mtime": p.stat().st_mtime})
        return out

    @staticmethod
    def preview_title(conv: dict[str, Any]) -> str:
        good, bad = ChatStore.count_ratings(conv)
        n = len(conv["messages"])
        return f"{conv['title']}  ·  {n} msg" + (f"  ·  ★{good}" if good else "")
