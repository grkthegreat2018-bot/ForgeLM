"""Thin API wrapper for chat message rendering and tool-call parsing.

Delegates to ``forge.self_play.discovery.qwen_adapter`` so that GUI pages
never import engine internals directly (GUI → API → engine layering).
"""
from __future__ import annotations


def render_messages_for_config(
    messages: list[dict],
    config_name: str | None = None,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render a chat message list into a model-specific prompt string."""
    from forge.self_play.discovery.qwen_adapter import (
        render_messages_for_config as _impl,
    )
    return _impl(
        messages,
        config_name=config_name,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
    )


def qwen_parse_tool_calls(text: str) -> tuple[list[dict] | None, str]:
    """Parse tool calls from model output text.

    Returns ``(tool_calls | None, remaining_text)``.
    """
    from forge.self_play.discovery.qwen_adapter import (
        qwen_parse_tool_calls as _impl,
    )
    return _impl(text)
