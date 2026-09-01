"""ForgeLM V10 ChatML template + Pythonic tool-call parsing.

ForgeLM V10 uses a ChatML-like format with special tokens:
  <|startoftext|>  — BOS, start of conversation
  <|im_start|>     — start of message (followed by role + newline)
  <|im_end|>       — end of message (= EOS)
  <|tool_list_start|> / <|tool_list_end|>     — wrap tool definitions in system prompt
  <|tool_call_start|> / <|tool_call_end|>     — wrap tool calls from assistant
  <|tool_response_start|> / <|tool_response_end|> — wrap tool results in tool role

Tool calls use PYTHONIC syntax (not JSON):
  <|tool_call_start|>[function_name(arg1='value1', arg2='value2')]<|tool_call_end|>

Tool definitions are JSON in the system prompt:
  List of tools: [{"name": "...", "description": "...", "parameters": {...}}]

Reference: https://docs.liquid.ai/lfm/key-concepts/chat-template
           https://docs.liquid.ai/lfm/key-concepts/tool-use
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any


# Special token strings (match the tokenizer's special_tokens_map.json).
BOS = "<|startoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_LIST_START = "<|tool_list_start|>"
TOOL_LIST_END = "<|tool_list_end|>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_RESP_START = "<|tool_response_start|>"
TOOL_RESP_END = "<|tool_response_end|>"


def format_tool_definitions(tools: list[dict]) -> str:
    """Render tool schemas as the ForgeLM V10 'List of tools:' string.

    Each tool is a dict with name, description, parameters.
    The Jinja template joins them as: List of tools: [{json}, {json}]
    """
    parts = []
    for t in tools:
        parts.append(json.dumps(t, ensure_ascii=False))
    return "List of tools: [" + ", ".join(parts) + "]"


def apply_chat_template(
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render a conversation in ForgeLM V10 ChatML format.

    Args:
        messages: list of {"role": "system"|"user"|"assistant"|"tool",
                           "content": str, "tool_calls": list (optional)}
        tools: list of tool definition dicts (name/description/parameters)
        add_generation_prompt: if True, append <|im_start|>assistant\n
    Returns:
        The formatted prompt string.
    """
    out = BOS

    # Extract system message + append tool definitions.
    system_text = ""
    if messages and messages[0]["role"] == "system":
        system_text = messages[0].get("content", "")
        messages = messages[1:]

    if tools:
        tool_str = format_tool_definitions(tools)
        system_text = (system_text + "\n" + tool_str) if system_text else tool_str

    if system_text:
        out += f"{IM_START}system\n{system_text}{IM_END}\n"

    # Render conversation messages.
    for msg in messages:
        role = msg["role"]
        out += f"{IM_START}{role}\n"

        if role == "assistant":
            # Content + optional tool calls.
            content = msg.get("content", "")
            if content:
                out += content
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                out += render_tool_calls(tool_calls)
            out += f"{IM_END}\n"
        elif role == "tool":
            # Tool response wrapped in response tokens.
            content = msg.get("content", "")
            out += f"{TOOL_RESP_START}{content}{TOOL_RESP_END}{IM_END}\n"
        else:
            # user or other roles.
            out += msg.get("content", "") + f"{IM_END}\n"

    if add_generation_prompt:
        out += f"{IM_START}assistant\n"

    return out


def render_tool_calls(tool_calls: list[dict]) -> str:
    """Render tool calls in ForgeLM V10 Pythonic syntax.

    Input: [{"name": "func", "args": {"arg1": "val1"}}]
    Output: <|tool_call_start|>[func(arg1='val1')]<|tool_call_end|>
    """
    calls = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args", {})
        arg_strs = []
        for k, v in args.items():
            if isinstance(v, str):
                arg_strs.append(f"{k}='{v}'")
            elif isinstance(v, bool):
                arg_strs.append(f"{k}={v}")
            elif isinstance(v, (int, float)):
                arg_strs.append(f"{k}={v}")
            else:
                arg_strs.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
        calls.append(f"{name}({', '.join(arg_strs)})")
    return f"{TOOL_CALL_START}[{', '.join(calls)}]{TOOL_CALL_END}"


# ── tool-call parsing ────────────────────────────────────────────────
_RE_TOOL_CALL = re.compile(
    re.escape(TOOL_CALL_START) + r'\[(.+?)\]' + re.escape(TOOL_CALL_END),
    re.DOTALL)


def parse_tool_calls(text: str) -> tuple[list[dict] | None, str]:
    """Parse ForgeLM V10 Pythonic tool calls from model output.

    Returns (tool_calls | None, musing_text).
    Tool calls are in format: <|tool_call_start|>[func(arg='val')]<|tool_call_end|>
    The musing is everything outside the tool-call tokens.

    Falls back to JSON parsing if the model emits JSON instead (robustness).
    """
    m = _RE_TOOL_CALL.search(text)
    if m:
        raw = m.group(1).strip()
        span = m.span()
        musing = (text[:span[0]] + text[span[1]:]).strip()
        calls = _parse_pythonic_calls(raw)
        if calls:
            return calls, musing
        # Fall through to JSON attempt if Pythonic parse failed.

    # Fallback: try JSON format {"tool": "...", "args": {...}}
    # (for robustness — some outputs might use JSON)
    calls_json = _parse_json_tool_call(text)
    if calls_json:
        # Find and remove the JSON from the musing.
        hint = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
        if hint:
            musing = (text[:hint.start()] + text[hint.end():]).strip()
        else:
            musing = text.strip()
        return calls_json, musing

    return None, text.strip()


def _parse_pythonic_calls(raw: str) -> list[dict] | None:
    """Parse 'func1(arg1='val1', arg2=42), func2(arg='x')' into dicts.

    Uses Python's ast module for safe parsing of the call syntax.
    """
    # Wrap in a list for ast.parse: [func1(...), func2(...)]
    try:
        tree = ast.parse(f"[{raw}]", mode="eval")
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Get function name from the Call node's func attribute.
                func = node.func
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                else:
                    continue
                args = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        continue
                    try:
                        args[kw.arg] = _ast_literal(kw.value)
                    except Exception:
                        args[kw.arg] = ast.unparse(kw.value)
                calls.append({"name": name, "args": args})
        return calls if calls else None
    except (SyntaxError, ValueError):
        return None


def _ast_literal(node: ast.AST) -> Any:
    """Safely evaluate an AST literal node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_ast_literal(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ast_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_ast_literal(k): _ast_literal(v)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ast_literal(node.operand)
    # Fallback: unparse to string.
    return ast.unparse(node)


def _parse_json_tool_call(text: str) -> list[dict] | None:
    """Fallback JSON parser for {"tool": "...", "args": {...}}."""
    # Find balanced JSON with "tool" key.
    hint = re.search(r'"tool"\s*:', text)
    if not hint:
        return None
    brace_start = text.rfind("{", 0, hint.start())
    if brace_start < 0:
        return None
    raw = _find_balanced_json(text, brace_start)
    if not raw:
        return None
    for candidate in (raw, raw.replace("'", '"').replace(",}", "}")):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "tool" in obj:
                return [{"name": obj["tool"], "args": obj.get("args", {})}]
        except json.JSONDecodeError:
            continue
    return None


def _find_balanced_json(text: str, start: int) -> str | None:
    """Return the substring of the first balanced {...} starting at start."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
