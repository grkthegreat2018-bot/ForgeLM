"""ForgeLM V2 ChatML template + tool-call parsing (Jamba Reasoning 3B parent).

ForgeLM V2 uses ChatML format with Jamba's special tokens:
  <|startoftext|>  — BOS (id=1), start of conversation
  <|im_start|>     — start of message (id=518, followed by role + newline)
  <|im_end|>       — end of message (id=519, also EOS)
  <tool_call>      — tool call start (id=531, single token)
  </tool_call>     — tool call end (id=532, single token)
  <tool_response>  — tool result start (id=539)
  </tool_response> — tool result end (id=540)
  <think>          — thinking start (id=541)
  </think>         — thinking end (id=542)

Tool calls use JSON syntax (Jamba format):
  <tool_call>
  {"name": "function_name", "arguments": {"arg1": "value1"}}
  </tool_call>

Tool results use:
  <tool_response>
  {"result": "..."}
  </tool_response>

Thinking mode (Jamba Reasoning):
  <think>
  reasoning steps here...
  </think>

  final answer here...

Legacy LFM2.5 format (Pythonic tool calls) is still supported for backward
compatibility via the LFM25_COMPAT flag.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

# Special token strings — Jamba Reasoning 3B (ForgeLM V2 parent).
BOS = "<|startoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"

# Jamba tool call markers (single tokens: 531/532)
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

# Jamba tool response markers (single tokens: 539/540)
TOOL_RESP_START = "<tool_response>"
TOOL_RESP_END = "</tool_response>"

# Trained thinking instruction — prepended to the final user message (and to
# user messages that precede an assistant <think> turn) so the model reasons
# inside <think>...</think> before answering. Matches the tokenizer's
# chat_template `thinking_prefix` verbatim.
THINKING_PREFIX = (
    "Begin by thinking about the reasoning process in the mind within "
    "<think> </think> tags and then proceed to give your response.\n")

# Canonical tools block (matches tokenizer chat_template verbatim):
#   # Tools
#   ... <tools>\n{json}\n...</tools> ... <tool_call>\n{"name": ...}\n</tool_call>
_TOOLS_BLOCK_HEAD = (
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within <tools></tools> "
    "XML tags:\n<tools>")
_TOOLS_BLOCK_TAIL = (
    "\n</tools>\n\nFor each function call, return a json object with function "
    "name and arguments within <tool_call></tool_call> XML tags:\n"
    f"{TOOL_CALL_START}\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
    f"{TOOL_CALL_END}")

# Legacy LFM2.5 markers (for backward compat, token IDs 10/11)
_LFM25_TOOL_CALL_START = bytes.fromhex(
    "3c7c746f6f6c5f63616c6c5f73746172747c3e").decode("ascii")
_LFM25_TOOL_CALL_END = bytes.fromhex(
    "3c7c746f6f6c5f63616c6c5f656e647c3e").decode("ascii")


def format_tool_definitions(tools: list[dict]) -> str:
    """Render tool schemas in the canonical Jamba <tools> block format.

    Matches the tokenizer's native chat_template: one JSON object per line
    inside <tools>...</tools>, preceded by the "# Tools" preamble and
    followed by the <tool_call> instruction.
    """
    parts = [_TOOLS_BLOCK_HEAD]
    for t in tools:
        parts.append("\n" + json.dumps(t, ensure_ascii=False))
    parts.append(_TOOLS_BLOCK_TAIL)
    return "".join(parts)


def _render_one_tool_call(tc: dict) -> str:
    """Render a single call as the template does:
    <tool_call>\n{"name": "NAME", "arguments": ARGS}\n</tool_call>

    Accepts {"name", "args"|"arguments"} and OpenAI {"function": {...}}.
    """
    if "function" in tc:
        tc = tc["function"]
    name = tc.get("name", "")
    args = tc.get("args", tc.get("arguments", {}))
    args_str = args if isinstance(args, str) else json.dumps(
        args, ensure_ascii=False)
    return (f'{TOOL_CALL_START}\n{{"name": "{name}", "arguments": '
            f'{args_str}}}\n{TOOL_CALL_END}')


def apply_chat_template(
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    bos: bool = True,
    thinking: bool = True,
) -> str:
    """Render a conversation in ForgeLM V2 (Jamba Reasoning 3B) ChatML format.

    This is a faithful Python port of the tokenizer's built-in Jinja
    ``chat_template`` (tokenizer_config.json) — the format the model was
    actually trained on:

      - system prompt carries the canonical ``<tools>`` block + ``<tool_call>``
        JSON instruction when tools are present
      - assistant ``tool_calls`` render as one ``<tool_call>{json}</tool_call>``
        block per call
      - consecutive ``tool`` messages group inside ONE ``<|im_start|>user``
        turn, each wrapped in ``<tool_response>...</tool_response>``
      - the generation prompt opens a ``<think>`` block (Jamba Reasoning
        always thinks before answering); the final user message gets the
        trained ``thinking_prefix`` instruction

    Args:
        messages: [{"role": "system"|"user"|"assistant"|"tool",
                    "content": str, "tool_calls": [...] (optional),
                    "reasoning_content": str (optional)}]
        tools: tool definition dicts (name/description/parameters)
        add_generation_prompt: append ``<|im_start|>assistant\\n<think>\\n``
            (or just ``<|im_start|>assistant\\n`` when thinking=False)
        bos: prepend ``<|startoftext|>``. Engine paths tokenize with
            add_special_tokens=True (BOS auto-added) — pass bos=False there.
            Direct tokenizer calls with add_special_tokens=False need bos=True.
        thinking: include the trained thinking instruction + open a ``<think>``
            block in the generation prompt.
    """
    out = BOS if bos else ""

    # ── system message + tool definitions ─────────────────────────────
    system_text = ""
    rest = messages
    if rest and rest[0].get("role") == "system":
        system_text = rest[0].get("content", "") or ""
        rest = rest[1:]

    if tools:
        sys_block = f"{IM_START}system\n"
        if system_text:
            sys_block += system_text + "\n\n"
        sys_block += format_tool_definitions(tools)
        out += sys_block + f"{IM_END}\n"
    elif system_text:
        out += f"{IM_START}system\n{system_text}{IM_END}\n"

    # ── last real query index (skips tool-response user turns) ───────
    n = len(rest)
    last_query_index = n - 1
    for i in range(n - 1, -1, -1):
        m = rest[i]
        if m.get("role") == "user":
            c = m.get("content", "") or ""
            if not (c.startswith(TOOL_RESP_START) and c.endswith(TOOL_RESP_END)):
                last_query_index = i
                break

    tp = THINKING_PREFIX if thinking else ""

    # ── conversation ──────────────────────────────────────────────────
    for i, msg in enumerate(rest):
        role = msg.get("role")
        if role == "user" or (role == "system" and i > 0):
            content = msg.get("content", "") or ""
            prefix = ""
            if role == "user" and tp and THINK_START not in content:
                nxt = rest[i + 1] if i + 1 < n else None
                if i == n - 1:
                    prefix = tp
                elif nxt is not None and nxt.get("role") == "assistant":
                    nc = nxt.get("content", "") or ""
                    if nc.startswith(THINK_START) or nxt.get("reasoning_content"):
                        prefix = tp
            out += f"{IM_START}{role}\n{prefix}{content}{IM_END}\n"
        elif role == "assistant":
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content")
            if reasoning is None and THINK_END in content:
                reasoning = (content.split(THINK_END)[0].rstrip("\n")
                             .split(THINK_START)[-1].lstrip("\n"))
                content = content.split(THINK_END)[-1].lstrip("\n")
            if i > last_query_index and reasoning:
                out += (f"{IM_START}assistant\n{THINK_START}\n"
                        f"{reasoning.strip(chr(10))}\n{THINK_END}\n\n"
                        f"{content.lstrip(chr(10))}")
            else:
                out += f"{IM_START}assistant\n{content}"
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for j, tc in enumerate(tool_calls):
                    if (j == 0 and content) or j > 0:
                        out += "\n"
                    out += _render_one_tool_call(tc)
            out += f"{IM_END}\n"
        elif role == "tool":
            # consecutive tool messages share one <|im_start|>user turn
            if i == 0 or rest[i - 1].get("role") != "tool":
                out += f"{IM_START}user"
            out += f"\n{TOOL_RESP_START}\n{msg.get('content', '')}\n{TOOL_RESP_END}"
            if i == n - 1 or rest[i + 1].get("role") != "tool":
                out += f"{IM_END}\n"

    if add_generation_prompt:
        out += f"{IM_START}assistant\n"
        if thinking:
            out += f"{THINK_START}\n"

    return out


def render_tool_calls(tool_calls: list[dict]) -> str:
    """Render tool calls in Jamba JSON format — one <tool_call> block per
    call, matching the native chat_template.

    Input: [{"name": "func", "args": {"arg1": "val1"}}]
    Also accepts OpenAI format: [{"function": {"name": ..., "arguments": ...}}]
    """
    return "\n".join(_render_one_tool_call(tc) for tc in tool_calls)


# ── tool-call parsing ────────────────────────────────────────────────
# Jamba format: {"name": "...", "arguments": {...}}
_RE_TOOL_CALL_JAMBA = re.compile(
    re.escape(TOOL_CALL_START) + r'(.*?)' + re.escape(TOOL_CALL_END),
    re.DOTALL)

# Legacy LFM2.5 format: [func(arg='val')]
_RE_TOOL_CALL_LFM25 = re.compile(
    re.escape(_LFM25_TOOL_CALL_START) + r'\[(.+?)\]' + re.escape(_LFM25_TOOL_CALL_END),
    re.DOTALL)


def parse_tool_calls(text: str) -> tuple[list[dict] | None, str]:
    """Parse tool calls from model output (Jamba JSON format).

    Returns (tool_calls | None, musing_text).
    Tool calls are in Jamba format:
      {"name": "func", "arguments": {...}}

    The musing is everything outside the tool-call tokens.

    Falls back to legacy LFM2.5 Pythonic format for backward compat.
    """
    # Try Jamba JSON format first
    m = _RE_TOOL_CALL_JAMBA.search(text)
    if m:
        raw = m.group(1).strip()
        span = m.span()
        musing = (text[:span[0]] + text[span[1]:]).strip()
        calls = _parse_json_calls(raw)
        if calls:
            return calls, musing

    # Try legacy LFM2.5 Pythonic format
    m = _RE_TOOL_CALL_LFM25.search(text)
    if m:
        raw = m.group(1).strip()
        span = m.span()
        musing = (text[:span[0]] + text[span[1]:]).strip()
        calls = _parse_pythonic_calls(raw)
        if calls:
            return calls, musing

    # Fallback: try bare JSON format {"tool": "...", "args": {...}}
    calls_json = _parse_json_tool_call(text)
    if calls_json:
        hint = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
        if hint:
            musing = (text[:hint.start()] + text[hint.end():]).strip()
        else:
            musing = text.strip()
        return calls_json, musing

    return None, text.strip()


def _parse_json_calls(raw: str) -> list[dict] | None:
    """Parse Jamba JSON tool calls: {"name": "...", "arguments": {...}}"""
    calls = []
    # May contain multiple JSON objects (one per line or comma-separated)
    for line in raw.strip().split('\n'):
        line = line.strip().rstrip(',')
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "name" in obj:
                calls.append({
                    "name": obj["name"],
                    "args": obj.get("arguments", obj.get("args", {})),
                })
        except json.JSONDecodeError:
            continue
    return calls if calls else None


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
