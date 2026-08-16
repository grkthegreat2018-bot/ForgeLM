"""URIAL — Untuned LLMs with Restyled In-context ALignment.

Alignment without any weight updates: a fixed system prompt plus a handful of
curated style demonstrations restyle the base model at inference time
("URIAL: Untuned LLMs with Restyled In-context ALignment", KDD 2024).

This module only assembles prompts — generation stays in the caller (or in
TrainingFreeSolver) so KV-cache/streaming choices live in one place.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and concise assistant. "
    "Solve the user's request directly and completely. "
    "When the answer is code, write clean, correct, efficient code "
    "with no unnecessary commentary."
)

# Three curated style demonstrations — URIAL shows that as few as three
# high-quality examples are enough to force the base model's style shift.
STYLE_EXAMPLES: list[dict[str, str]] = [
    {
        "user": "Write a function that returns the sum of all even numbers in a list.",
        "assistant": (
            "def sum_even(nums):\n"
            "    return sum(n for n in nums if n % 2 == 0)"
        ),
    },
    {
        "user": "Explain the difference between a list and a tuple in Python.",
        "assistant": (
            "A list is mutable and resizable; a tuple is immutable and "
            "fixed-size. Use lists for dynamic collections, tuples for fixed "
            "records or hashable keys."
        ),
    },
    {
        "user": "What is the output of 3 * 'ab' + 'c'?",
        "assistant": (
            "'abababc'. The * operator repeats the string 3 times, then "
            "+ concatenates 'c'."
        ),
    },
]


def build_prompt(
    user_prompt: str,
    system_prompt: str | None = None,
    n_examples: int = 3,
    extra_context: str = "",
) -> str:
    """Assemble the URIAL in-context alignment prompt.

    Args:
        user_prompt: the actual request.
        system_prompt: override the default system instruction.
        n_examples: how many style demonstrations to include (max 3).
        extra_context: optional extra block (e.g. Reflexion memory) inserted
            right before the user request.

    Returns:
        Full styled prompt string.
    """
    system = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    parts = [system]
    for ex in STYLE_EXAMPLES[: max(0, n_examples)]:
        parts.append(f"User: {ex['user']}\nAssistant: {ex['assistant']}")
    if extra_context:
        parts.append(extra_context)
    parts.append(f"User: {user_prompt}\nAssistant:")
    return "\n\n".join(parts)
