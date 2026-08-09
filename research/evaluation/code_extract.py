"""Shared code extraction utility for evaluation and self-play modules.

Extracts Python code from markdown fences or raw text. Used by:
- research.evaluation.reasoning_benchmarks
- research.evaluation.livecodebench_eval
- research.architecture.thinking_model
"""
import re

# Pre-compiled pattern for markdown code blocks
_RE_CODEBLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, last_block: bool = True) -> str:
    """Extract Python code from markdown fences or raw text.

    Args:
        text: input text that may contain ```python ... ``` blocks
        last_block: if True, return the last code block (most complete);
                    if False, return the first

    Returns:
        Extracted code string, or the original text stripped if no blocks found.
    """
    matches = _RE_CODEBLOCK.findall(text)
    if matches:
        return matches[-1].strip() if last_block else matches[0].strip()
    return text.strip()
