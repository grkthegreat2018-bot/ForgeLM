"""Signal capture for the live-updating model pipeline (Phase 1).

Logs three types of signal to a JSONL file for later online training:
  - interaction: (messages, response) pair from /v1/chat/completions
  - feedback:    user action on a response (accept / edit / reject)
  - code_result: tool/code execution outcome (success / failure + error)

Each interaction gets a unique ID so feedback and code results can be
linked back to the original (prompt, response) pair.

The logger is thread-safe (serve.py uses ThreadingHTTPServer) and
append-only — safe to write from multiple request handlers concurrently.

Usage in serve.py:
    from research.runtime.signal_capture import SignalLogger
    logger = SignalLogger("research/data/live_training.jsonl")
    interaction_id = logger.log_interaction(messages, response, temperature)
    logger.log_feedback(interaction_id, action="accept")
    logger.log_code_result(interaction_id, success=True, output="...")
"""
import json
import sys
import threading
import time
import uuid
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


class SignalLogger:
    """Thread-safe append-only JSONL logger for live training signals."""

    def __init__(self, filepath, enabled=True):
        self.filepath = Path(filepath)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record):
        if not self.enabled:
            return
        record["timestamp"] = time.time()
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(line)

    def log_interaction(self, messages, response, temperature=0.7, max_tokens=512):
        """Log a (messages, response) pair. Returns interaction_id for linking feedback."""
        interaction_id = f"int_{uuid.uuid4().hex[:12]}"
        # Strip any non-serializable content from messages (e.g. images).
        clean_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), str):
                clean_messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                clean_messages.append({"role": msg["role"], "content": str(msg.get("content", ""))})
        self._write({
            "id": interaction_id,
            "type": "interaction",
            "messages": clean_messages,
            "response": response,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return interaction_id

    def log_feedback(self, interaction_id, action, edited_content=None):
        """Log user feedback on a previous interaction.

        action: "accept" | "edit" | "reject"
        edited_content: if action == "edit", the user's corrected version.
        """
        if action not in ("accept", "edit", "reject"):
            raise ValueError(f"action must be accept/edit/reject, got {action}")
        record = {
            "id": f"fb_{uuid.uuid4().hex[:12]}",
            "interaction_id": interaction_id,
            "type": "feedback",
            "action": action,
        }
        if edited_content is not None:
            record["edited_content"] = edited_content
        self._write(record)

    def log_code_result(self, interaction_id, success, output=None, error=None, language=None):
        """Log code execution outcome linked to a previous interaction."""
        record = {
            "id": f"code_{uuid.uuid4().hex[:12]}",
            "interaction_id": interaction_id,
            "type": "code_result",
            "success": bool(success),
        }
        if output is not None:
            record["output"] = output[:2000]  # cap to avoid huge logs
        if error is not None:
            record["error"] = error[:2000]
        if language is not None:
            record["language"] = language
        self._write(record)

    def log_self_verification(self, interaction_id, score, reasoning=None):
        """Log the model's self-verification score for its own output.

        score: float 0.0 to 1.0 (model's confidence in its own output quality).
        reasoning: optional short explanation string.
        """
        record = {
            "id": f"sv_{uuid.uuid4().hex[:12]}",
            "interaction_id": interaction_id,
            "type": "self_verification",
            "score": float(score),
        }
        if reasoning is not None:
            record["reasoning"] = reasoning[:500]
        self._write(record)


def load_signals(filepath):
    """Load a signal JSONL file and group by interaction_id.

    Returns dict: interaction_id -> {
        "messages": [...], "response": "...", "temperature": float,
        "feedback": {"action": "...", "edited_content": "..."} or None,
        "code_result": {"success": bool, "error": "..."} or None,
        "self_verification": {"score": float} or None,
    }
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return {}

    interactions = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")
            if rtype == "interaction":
                iid = record["id"]
                interactions[iid] = {
                    "messages": record["messages"],
                    "response": record["response"],
                    "temperature": record.get("temperature", 0.7),
                    "feedback": None,
                    "code_result": None,
                    "self_verification": None,
                    "timestamp": record.get("timestamp", 0),
                }
            elif rtype == "feedback":
                iid = record.get("interaction_id")
                if iid in interactions:
                    interactions[iid]["feedback"] = {
                        "action": record["action"],
                        "edited_content": record.get("edited_content"),
                    }
            elif rtype == "code_result":
                iid = record.get("interaction_id")
                if iid in interactions:
                    interactions[iid]["code_result"] = {
                        "success": record["success"],
                        "output": record.get("output"),
                        "error": record.get("error"),
                        "language": record.get("language"),
                    }
            elif rtype == "self_verification":
                iid = record.get("interaction_id")
                if iid in interactions:
                    interactions[iid]["self_verification"] = {
                        "score": record["score"],
                        "reasoning": record.get("reasoning"),
                    }
    return interactions


def filter_training_pairs(interactions, min_self_verify=0.5):
    """Filter interactions into training pairs for online learning.

    Returns list of {"prompt": str, "completion": str, "label": "positive"|"negative"|"dpo_chosen"|"dpo_rejected"}.

    Selection logic:
      - accept + self_verification >= min_self_verify → positive (prompt, response)
      - edit → DPO pair: chosen=edited_content, rejected=original response
      - reject → negative (prompt, response) — used for DPO rejected only
      - code_result success → positive (prompt, response)
      - code_result failure → negative (prompt, response)
      - no feedback + self_verification >= 0.7 → weak positive (lower weight in trainer)
    """
    pairs = []
    for iid, data in interactions.items():
        prompt = data["messages"]
        response = data["response"]
        fb = data["feedback"]
        cr = data["code_result"]
        sv = data["self_verification"]

        if fb and fb["action"] == "edit" and fb["edited_content"]:
            pairs.append({
                "prompt": prompt,
                "completion_chosen": fb["edited_content"],
                "completion_rejected": response,
                "label": "dpo",
                "interaction_id": iid,
            })
        elif fb and fb["action"] == "accept":
            pairs.append({
                "prompt": prompt,
                "completion": response,
                "label": "positive",
                "interaction_id": iid,
            })
        elif fb and fb["action"] == "reject":
            pairs.append({
                "prompt": prompt,
                "completion": response,
                "label": "negative",
                "interaction_id": iid,
            })
        elif cr and cr["success"]:
            pairs.append({
                "prompt": prompt,
                "completion": response,
                "label": "positive",
                "interaction_id": iid,
            })
        elif cr and not cr["success"]:
            pairs.append({
                "prompt": prompt,
                "completion": response,
                "label": "negative",
                "interaction_id": iid,
            })
        elif sv and sv["score"] >= 0.7:
            pairs.append({
                "prompt": prompt,
                "completion": response,
                "label": "weak_positive",
                "interaction_id": iid,
            })
        # If no signal at all, skip — don't train on unsupervised data.
    return pairs
