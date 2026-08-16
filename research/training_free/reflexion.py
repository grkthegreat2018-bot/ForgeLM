"""Reflexion — episodic memory for in-context reinforcement.

Instead of optimizer weight updates, failed attempts are reflected on and the
critique is appended to the prompt of the next attempt — policy iteration in
working memory (Shinn et al., 2023). Bounded context keeps it inside the
window. Entirely inference-time: no gradients, no parameter changes.
"""
from __future__ import annotations

from typing import Callable

from research.training_free.urial import build_prompt


def make_template_reflection(task: str, code: str = "", error: str = "") -> str:
    """Deterministic critique template (no model call needed)."""
    if error:
        return (
            f"Previous attempt at '{task}' failed with: {error.strip()[:200]}. "
            "Avoid repeating this mistake; check edge cases and syntax."
        )
    return (
        f"Previous attempt at '{task}' did not solve it. "
        "Re-analyze the problem and try a different, complete solution."
    )


class ReflexionBuffer:
    """Bounded episodic memory of past attempts, rendered into the prompt.

    Args:
        max_entries: how many attempts to keep (oldest dropped).
        max_chars: hard cap on the rendered context block.
        reflection_fn: optional callable(task, code, error) -> str used to
            generate the reflection; defaults to the deterministic template.
    """

    def __init__(
        self,
        max_entries: int = 8,
        max_chars: int = 1500,
        reflection_fn: Callable[[str, str, str], str] | None = None,
    ):
        self.max_entries = max(1, max_entries)
        self.max_chars = max(200, max_chars)
        self.reflection_fn = reflection_fn or make_template_reflection
        self._entries: list[dict] = []

    def add(self, task: str, code: str = "", error: str = "",
            success: bool = False, reflection: str | None = None) -> None:
        """Record one attempt; drops oldest once over max_entries."""
        if reflection is None:
            reflection = self.reflection_fn(task, code, error)
        self._entries.append({
            "task": task,
            "success": bool(success),
            "reflection": reflection,
        })
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def successes(self) -> int:
        return sum(1 for e in self._entries if e["success"])

    def context_block(self) -> str:
        """Render the memory as a context section (bounded by max_chars)."""
        if not self._entries:
            return ""
        lines = ["## Past attempts (learn from them, do not repeat mistakes)"]
        for e in self._entries:
            tag = "SUCCESS" if e["success"] else "FAILED"
            lines.append(f"- [{tag}] {e['reflection']}")
        block = "\n".join(lines)
        if len(block) > self.max_chars:
            suffix = "\n... (memory truncated)"
            block = block[: max(0, self.max_chars - len(suffix))] + suffix
        return block

    def prompt_for(self, task: str, n_examples: int = 3,
                   system_prompt: str | None = None) -> str:
        """URIAL-styled prompt with the memory block prepended to the task."""
        return build_prompt(
            task,
            system_prompt=system_prompt,
            n_examples=n_examples,
            extra_context=self.context_block(),
        )
