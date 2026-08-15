"""Context manager for multi-turn tool-use conversations.

Handles:
  - Token counting per message and per conversation
  - Context window budget tracking (prompt + generation must fit in max_seq_len)
  - Summarization of old turns when approaching the limit
  - Preservation of recent turns + tool call/result pairs (never split them)
  - Model-generated summaries (the model summarizes its own history)

The model has RoPE theta=1M (128K theoretical context, 32K VRAM budget).
We target 24K for the prompt, leaving 8K for generation. When the conversation
exceeds the budget, older turns are compressed into a summary.

Architecture:
  1. ConversationTokenTracker: counts tokens per message
  2. ContextBudget: tracks prompt + generation budget
  3. ContextSummarizer: compresses old turns into a summary message
  4. ContextManager: orchestrates tracking + summarization

Summarization strategy:
  - Keep the system message + tool definitions (always)
  - Keep the last N turns intact (recency window)
  - Summarize older turns into a single {"role": "system", "content": "..."} message
  - Never split a tool_call + tool_result pair
  - The summary preserves: what tools were called, what results were obtained,
    what was learned, what the current goal is
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from research.self_play.discovery.qwen_adapter import (
    qwen_render_messages, IM_START, IM_END,
)


# ── Token counting ─────────────────────────────────────────────────────────

def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text. Falls back to char-based estimate if no tokenizer."""
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    # Rough estimate: ~4 chars per token for English text
    return max(1, len(text) // 4)


def count_message_tokens(msg: dict, tokenizer=None) -> int:
    """Count tokens for a single message including role markers.

    Qwen format: <|im_start|>role\n{content}<|im_end|>\n
    Overhead: ~4 tokens for markers per message.
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls")

    token_count = 4  # <|im_start|>role\n ... <|im_end|>\n

    if tool_calls:
        for tc in tool_calls:
            token_count += count_tokens(json.dumps(tc, ensure_ascii=False), tokenizer)
    if content:
        token_count += count_tokens(str(content), tokenizer)
    if msg.get("name"):
        token_count += count_tokens(msg["name"], tokenizer)

    return token_count


def count_conversation_tokens(messages: list[dict], tokenizer=None) -> int:
    """Total token count for a conversation."""
    return sum(count_message_tokens(m, tokenizer) for m in messages)


# ── Context budget ─────────────────────────────────────────────────────────

@dataclass
class ContextBudget:
    """Tracks token budget for a conversation.

    max_seq_len: model's max context (32768 for LFM2.5)
    reserved_for_generation: tokens kept free for the model's response
    prompt_budget: max_seq_len - reserved_for_generation
    """
    max_seq_len: int = 32768
    reserved_for_generation: int = 4096

    @property
    def prompt_budget(self) -> int:
        return self.max_seq_len - self.reserved_for_generation

    def fits(self, token_count: int) -> bool:
        return token_count <= self.prompt_budget

    def overhead(self, token_count: int) -> int:
        """How many tokens over budget."""
        return max(0, token_count - self.prompt_budget)


# ── Conversation structure analysis ────────────────────────────────────────

def find_tool_call_groups(messages: list[dict]) -> list[tuple[int, int]]:
    """Find (assistant_with_tool_calls, tool_result) groups.

    Returns list of (start_idx, end_idx_exclusive) for each group.
    A group is: assistant(tool_calls) + one or more tool results.
    """
    groups = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            start = i
            # Find following tool messages
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            groups.append((start, j))
            i = j
        else:
            i += 1
    return groups


def split_conversation(messages: list[dict], keep_recent_turns: int = 4,
                       tokenizer=None) -> tuple[list[dict], list[dict]]:
    """Split conversation into (to_summarize, to_keep).

    Keeps:
    - All messages up to and including the first user message
    - The last `keep_recent_turns` messages
    - Never splits a tool_call + tool_result group

    Returns:
    - to_summarize: older messages that should be compressed
    - to_keep: recent messages that stay intact
    """
    if len(messages) <= keep_recent_turns + 1:
        return [], messages[:]

    # Find tool call groups to avoid splitting them
    groups = find_tool_call_groups(messages)

    # Find the split point: keep the last keep_recent_turns messages,
    # but adjust to not split a tool group
    split_idx = len(messages) - keep_recent_turns

    # Adjust split_idx to not break a tool group
    for start, end in groups:
        if start < split_idx < end:
            # The split would break this group — move split before the group
            split_idx = start
            break

    # Always keep the first user message
    first_user = 0
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            first_user = i
            break

    # Don't summarize the first user message
    if split_idx <= first_user + 1:
        return [], messages[:]

    to_summarize = messages[:split_idx]
    to_keep = messages[split_idx:]

    return to_summarize, to_keep


# ── Summarization ──────────────────────────────────────────────────────────

def build_summary_message(messages_to_summarize: list[dict]) -> dict:
    """Build a summary of older conversation turns.

    Creates a system message that captures:
    - What tools were called and their results (compressed)
    - What the model learned/discovered
    - What the current goal/sub-goal is
    - Key facts and findings

    This is a heuristic summarizer (not model-generated) for speed.
    For higher quality, use model_generate_summary() instead.
    """
    parts = []
    tool_calls_made = []
    findings = []
    user_goal = None

    for msg in messages_to_summarize:
        role = msg.get("role")
        if role == "user" and user_goal is None:
            user_goal = msg.get("content", "")[:200]
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("arguments", {})
                    # Compress args to key info
                    args_str = json.dumps(args, ensure_ascii=False)[:100]
                    tool_calls_made.append(f"  - {name}({args_str})")
            content = msg.get("content", "")
            if content and len(content) > 10:
                # Capture the model's reasoning/findings (first 150 chars)
                findings.append(content[:150])
        elif role == "tool":
            content = msg.get("content", "")
            name = msg.get("name", "tool")
            # Compress tool result to key info
            if len(content) > 200:
                # Try to extract key data from JSON result
                try:
                    result = json.loads(content)
                    if isinstance(result, dict):
                        if "stdout" in result and result["stdout"]:
                            findings.append(f"    [{name} output]: {result['stdout'][:150]}")
                        elif "results" in result and result["results"]:
                            n = len(result["results"])
                            findings.append(f"    [{name} returned {n} results]")
                        elif "error" in result:
                            findings.append(f"    [{name} error]: {result['error'][:100]}")
                        else:
                            findings.append(f"    [{name} result]: {content[:100]}")
                    else:
                        findings.append(f"    [{name}]: {content[:100]}")
                except Exception:
                    findings.append(f"    [{name}]: {content[:100]}")
            elif content:
                findings.append(f"    [{name}]: {content[:100]}")

    summary_parts = []
    if user_goal:
        summary_parts.append(f"Original task: {user_goal}")
    if tool_calls_made:
        summary_parts.append("Tools called so far:")
        summary_parts.extend(tool_calls_made)
    if findings:
        summary_parts.append("Key findings/results:")
        summary_parts.extend(findings)

    summary_text = "\n".join(summary_parts) if summary_parts else "No prior context."

    return {
        "role": "system",
        "content": f"[Context summary — earlier conversation compressed]\n{summary_text}",
    }


def model_generate_summary(model, tokenizer, messages_to_summarize: list[dict],
                           device: str = "cuda",
                           max_new_tokens: int = 256) -> dict:
    """Use the model itself to generate a summary of older turns.

    This produces higher-quality summaries than the heuristic, but costs
    a generation pass. Use when context quality matters more than speed.
    """
    from research.self_play.discovery.qwen_adapter import qwen_generate

    # Build a prompt asking the model to summarize
    conv_text = qwen_render_messages(messages_to_summarize, add_generation_prompt=False)
    # Truncate if too long
    if len(conv_text) > 4000:
        conv_text = conv_text[:4000] + "\n[...truncated...]"

    summary_prompt = (
        f"<|im_start|>system\n"
        f"You are a context summarizer. Summarize the conversation below into a "
        f"compact summary that preserves: (1) the user's goal, (2) what tools were "
        f"called and their key results, (3) important findings, (4) what remains "
        f"to be done. Be concise — max 200 words.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Summarize this conversation:\n{conv_text}\n"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )

    output = qwen_generate(model, tokenizer, summary_prompt,
                           max_new_tokens=max_new_tokens,
                           temperature=0.3, device=device)

    return {
        "role": "system",
        "content": f"[Context summary — earlier conversation compressed]\n{output.strip()}",
    }


# ── Context manager ────────────────────────────────────────────────────────

@dataclass
class ContextManagerConfig:
    """Configuration for context management."""
    max_seq_len: int = 32768
    reserved_for_generation: int = 4096
    keep_recent_turns: int = 6  # messages to keep intact
    summarize_threshold: float = 0.75  # summarize when at 75% of budget
    use_model_summary: bool = False  # use model-generated summaries (slower)
    summary_max_tokens: int = 512


class ContextManager:
    """Manages conversation context for multi-turn tool-use.

    Tracks token count and summarizes old turns when approaching the budget.
    Call `maybe_compress()` after each turn to check if compression is needed.
    """

    def __init__(self, config: ContextManagerConfig | None = None,
                 tokenizer=None, model=None, device: str = "cuda"):
        self.config = config or ContextManagerConfig()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.budget = ContextBudget(
            max_seq_len=self.config.max_seq_len,
            reserved_for_generation=self.config.reserved_for_generation,
        )
        self._summary_count = 0  # how many times we've summarized

    def token_count(self, messages: list[dict]) -> int:
        """Current token count of the conversation."""
        return count_conversation_tokens(messages, self.tokenizer)

    def needs_compression(self, messages: list[dict]) -> bool:
        """Check if the conversation exceeds the summarization threshold."""
        count = self.token_count(messages)
        threshold = self.budget.prompt_budget * self.config.summarize_threshold
        return count > threshold

    def compress(self, messages: list[dict]) -> list[dict]:
        """Compress the conversation by summarizing old turns.

        Returns a new message list with old turns replaced by a summary.
        """
        count = self.token_count(messages)
        if count <= self.budget.prompt_budget:
            return messages  # no compression needed

        to_summarize, to_keep = split_conversation(
            messages, keep_recent_turns=self.config.keep_recent_turns,
            tokenizer=self.tokenizer)

        if not to_summarize:
            return messages  # can't compress further

        # Generate summary
        if self.config.use_model_summary and self.model is not None:
            summary = model_generate_summary(
                self.model, self.tokenizer, to_summarize,
                device=self.device,
                max_new_tokens=self.config.summary_max_tokens)
        else:
            summary = build_summary_message(to_summarize)

        self._summary_count += 1

        # Build new conversation: summary + kept messages
        # Insert summary after the first user message if present,
        # otherwise prepend it
        if to_keep and to_keep[0].get("role") == "user":
            new_messages = [to_keep[0], summary] + to_keep[1:]
        else:
            new_messages = [summary] + to_keep

        return new_messages

    def maybe_compress(self, messages: list[dict]) -> tuple[list[dict], bool]:
        """Compress if needed. Returns (new_messages, was_compressed)."""
        if not self.needs_compression(messages):
            return messages, False
        new_messages = self.compress(messages)
        old_count = self.token_count(messages)
        new_count = self.token_count(new_messages)
        return new_messages, True

    def stats(self) -> dict:
        return {
            "summaries_made": self._summary_count,
            "prompt_budget": self.budget.prompt_budget,
            "reserved_for_generation": self.budget.reserved_for_generation,
            "max_seq_len": self.config.max_seq_len,
        }
