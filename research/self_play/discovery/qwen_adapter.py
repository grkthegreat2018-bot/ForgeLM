"""Qwen-format tool-call adapter for the discovery self-play loop.

The discovery loop was built for ForgeLM V10's native Pythonic tool-call format
(<|tool_call_start|>[func(arg='val')]<|tool_call_end|>). Our SFT model uses
Qwen chat format with JSON tool calls wrapped in the native special tokens
(ids 10/11):
  <|im_start|>assistant
  {start_token}\n{"name": "tool_name", "arguments": {...}}\n{end_token}<|im_end|>

This module provides:
  - qwen_render_messages: render a conversation in Qwen format with tool defs
  - qwen_parse_tool_calls: parse JSON tool calls from model output
  - QwenDiscoveryAdapter: wraps the discovery loop to use Qwen format

This lets the SFT-trained model participate in self-play using the format
it was trained on, without rewriting the entire discovery infrastructure.
"""
from __future__ import annotations

import json
import re
from typing import Any

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOS_ID = 7  # <|im_end|> token id in the LFM2.5 tokenizer

# Tool call markers — ForgeLM V10's native special tokens (single-token ids 10/11).
# Built from hex to avoid IDE tool-call parsing confusion.
TOOL_CALL_START = bytes.fromhex("3c7c746f6f6c5f63616c6c5f73746172747c3e").decode("ascii")
TOOL_CALL_END = bytes.fromhex("3c7c746f6f6c5f63616c6c5f656e647c3e").decode("ascii")
TOOL_CALL_START_FIRST_ID = 10   # special token id for the start marker
TOOL_CALL_END_FIRST_ID = 11     # special token id for the end marker


def qwen_render_tool_defs(tools: list[dict]) -> str:
    """Render tool definitions as a text block for the system/user message.

    Includes explicit instructions on how to call tools using the
    <|tool_call_start|>/<|tool_call_end|> marker format so the model
    knows to emit structured tool calls rather than writing code.
    """
    lines = [
        "You have access to the following tools. To use a tool, output a tool call "
        "wrapped in the special markers <|tool_call_start|> and <|tool_call_end|>, "
        "containing a JSON object with the tool name and arguments.",
        "",
        "Example tool call:",
        '<|tool_call_start|>',
        '{"name": "list_dir", "arguments": {"path": "."}}',
        '<|tool_call_end|>',
        "",
        "Rules:",
        "- Always use the tool call markers to invoke a tool. Never write Python "
        "code to perform actions — use the tools instead.",
        "- You can make multiple tool calls in one response.",
        "- After each tool call, you will receive the result and can continue.",
        "- Only call tools that are listed below.",
        "",
        "Available tools:",
    ]
    for t in tools:
        # unwrap OpenAI-style {"type": "function", "function": {...}}
        if t.get("type") == "function" and "function" in t:
            t = t["function"]
        params = t.get("parameters", {})
        if isinstance(params, dict) and "properties" in params:
            # Already in JSON schema format
            param_str = json.dumps(params["properties"], ensure_ascii=False)
        else:
            param_str = json.dumps(params, ensure_ascii=False)
        lines.append(f"- {t['name']}: {t.get('description', '')}\n"
                     f"    arguments: {param_str}")
    return "\n".join(lines)


def qwen_render_messages(messages: list[dict],
                         tools: list[dict] | None = None,
                         add_generation_prompt: bool = True) -> str:
    """Render a conversation in Qwen chat format.

    Messages: [{"role": "system"|"user"|"assistant"|"tool", "content": str,
                "tool_calls": [...], "name": str}]

    Tool calls are rendered as JSON blocks:
      {"name": "...", "arguments": {...}}

    Tool results use:
      <|im_start|>tool
      {tool_name}
      {result_json}<|im_end|>
    """
    parts = []

    # System message with tool definitions
    system_text = ""
    if messages and messages[0]["role"] == "system":
        system_text = messages[0].get("content", "")
        messages = messages[1:]

    if tools:
        tool_text = qwen_render_tool_defs(tools)
        system_text = (system_text + "\n\n" + tool_text) if system_text else tool_text

    if system_text:
        parts.append(f"{IM_START}system\n{system_text}{IM_END}\n")

    for msg in messages:
        role = msg["role"]
        if role == "user":
            parts.append(f"{IM_START}user\n{msg.get('content', '')}{IM_END}\n")
        elif role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Render each tool call in the training format: {start}\n{json}\n{end}
                tc_parts = []
                for tc in tool_calls:
                    tc_parts.append(
                        TOOL_CALL_START + "\n" +
                        json.dumps(tc, ensure_ascii=False) + "\n" +
                        TOOL_CALL_END
                    )
                body = "\n".join(tc_parts)
                body = body if not content else f"{content}\n{body}"
            else:
                body = content or ""
            parts.append(f"{IM_START}assistant\n{body}{IM_END}\n")
        elif role == "tool":
            name = msg.get("name", "tool")
            content = msg.get("content", "")
            parts.append(f"{IM_START}tool\n{name}\n{content}{IM_END}\n")

    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")

    return "".join(parts)


# ── Tool-call parsing ─────────────────────────────────────────────────────

# The training format uses the native special tokens as tool call markers:
#   {start}\n{"name": "...", "arguments": {...}}\n{end}
# We parse these markers first, then fall back to bare JSON detection.

_RE_NAME_HINT = re.compile(r'\{"name"\s*:\s*"([^"]+)"')


def _find_balanced_json(text: str, start: int) -> str | None:
    """Return the substring of the first balanced {...} starting at start.

    Handles nested braces and strings with escaped quotes.
    """
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


def qwen_parse_tool_calls(text: str) -> tuple[list[dict] | None, str]:
    """Parse tool calls from model output.

    Returns (tool_calls | None, musing_text).

    Primary format (training): {start}\n{json}\n{end}
    Fallback: bare {"name": ...} JSON objects
    Fallback: ForgeLM V10 Pythonic format
    """
    calls = []
    consumed_spans = []  # (start, end) ranges of parsed content

    # Phase 1: Parse start-marker ... end-marker wrapped tool calls
    pos = 0
    while True:
        start_idx = text.find(TOOL_CALL_START, pos)
        if start_idx == -1:
            break
        end_idx = text.find(TOOL_CALL_END, start_idx + 1)
        if end_idx == -1:
            # No end marker — try to parse from start to end of text
            end_idx = len(text)
        # Extract JSON between markers
        json_region = text[start_idx + len(TOOL_CALL_START):end_idx].strip()
        raw = _find_balanced_json(json_region, 0)
        if raw:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    import json_repair
                    obj = json_repair.loads(raw)
                except Exception:
                    obj = None
            if isinstance(obj, dict) and "name" in obj:
                name = obj["name"]
                args = obj.get("arguments", obj.get("args", {}))
                if not isinstance(args, dict):
                    args = {}
                calls.append({"name": name, "args": args, "arguments": args})
                consumed_spans.append((start_idx, min(end_idx + len(TOOL_CALL_END), len(text))))
        pos = end_idx + 1

    # Phase 2: If no marker-wrapped calls found, try bare JSON
    if not calls:
        for hint in _RE_NAME_HINT.finditer(text):
            brace_start = hint.start()
            if any(s <= brace_start < e for s, e in consumed_spans):
                continue
            raw = _find_balanced_json(text, brace_start)
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    import json_repair
                    obj = json_repair.loads(raw)
                except Exception:
                    continue
            if isinstance(obj, dict) and "name" in obj:
                name = obj["name"]
                args = obj.get("arguments", obj.get("args", {}))
                if not isinstance(args, dict):
                    args = {}
                calls.append({"name": name, "args": args, "arguments": args})
                consumed_spans.append((brace_start, brace_start + len(raw)))

    if calls:
        # Build musing from text outside consumed spans
        musing_parts = []
        last = 0
        for s, e in sorted(consumed_spans):
            if s > last:
                musing_parts.append(text[last:s].strip())
            last = e
        if last < len(text):
            musing_parts.append(text[last:].strip())
        musing = " ".join(musing_parts).strip()
        return calls, musing

    # Fallback: try ForgeLM V10 Pythonic format
    from research.self_play.discovery.chat_template import parse_tool_calls as _lfm_parse
    return _lfm_parse(text)


def qwen_generate(model, tokenizer, prompt: str, max_new_tokens: int = 256,
                  temperature: float = 0.2, device: str = "cuda",
                  top_k: int = 80, repetition_penalty: float = 1.05,
                  grammar_matcher=None, bitmask=None) -> str:
    """Generate text from the model, stopping at <|im_end|> (EOS_ID=7).

    Returns only the generated tokens (not the prompt), preserving special
    tokens so tool-call JSON is visible to the parser.

    Uses ForgeLM V10-recommended generation defaults:
      - temperature: 0.2 (low randomness for reliable tool calls)
      - top_k: 80 (limits sampling to top-k logits)
      - repetition_penalty: 1.05 (discourages token repetition)

    If grammar_matcher is provided, uses two-phase constrained decoding:
    Phase 1: Generate freely (for musings/reasoning text).
    Phase 2: When the model emits the start marker (token 10), switch to
             constrained mode — only valid JSON with correct tool names allowed.
    Structural tokens (end marker, newlines) are skipped in constrained mode.
    This prevents hallucinated tool names while allowing free-form reasoning.
    """
    import torch
    import torch.nn.functional as F

    ids = tokenizer(prompt, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
    prompt_len = ids.shape[1]

    # Constrained decoding: trigger on the start marker (token 10).
    # In constrained mode, skip the end marker (token 11) and newline tokens
    # (structural tokens between markers and JSON, not part of the JSON grammar).
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    skip_token_ids = {TOOL_CALL_END_FIRST_ID} | newline_ids

    constrained = False  # start in free-text mode
    generated_ids: list[int] = []  # track generated token ids for repetition penalty

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Apply xgrammar bitmask only when in constrained mode.
            # Structural tokens (end marker, newlines) are ONLY un-masked when
            # the JSON grammar is complete. If they're always un-masked, the
            # model emits <|tool_call_end|> or newline spam mid-JSON and the
            # tool call becomes unparseable (epoch-3 collapse: 50/50 tasks
            # with 0 tools because the JSON was garbage).
            if constrained and grammar_matcher is not None and bitmask is not None:
                import xgrammar as xgr
                grammar_matcher.fill_next_token_bitmask(bitmask)
                # Save skip-list logits before masking — only when grammar done
                skip_logits = {}
                grammar_done = grammar_matcher.is_terminated()
                if grammar_done:
                    for sid in skip_token_ids:
                        if 0 <= sid < next_logits.shape[-1]:
                            skip_logits[sid] = next_logits[:, sid].clone()
                xgr.apply_token_bitmask_inplace(next_logits, bitmask)
                # Restore skip-list logits (un-mask structural tokens)
                for sid, val in skip_logits.items():
                    next_logits[:, sid] = val

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                # Repetition penalty: penalize tokens already generated
                # (look at last 64 tokens to limit compute)
                if generated_ids:
                    for tid in set(generated_ids[-64:]):
                        next_logits[:, tid] /= repetition_penalty

                # Top-k filtering: keep only top_k logits before softmax
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(
                        next_logits, top_k)[0][..., -1, None]
                    next_logits.masked_fill_(indices_to_remove, float('-inf'))

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tok_id = next_token.item()
            generated_ids.append(tok_id)
            ids = torch.cat([ids, next_token], dim=1)

            if tok_id == EOS_ID:
                break

            # Switch to constrained mode when the start marker is emitted
            if not constrained and grammar_matcher is not None and tok_id == TOOL_CALL_START_FIRST_ID:
                constrained = True
                continue

            # Accept token in grammar matcher when constrained.
            # Skip structural tokens (end marker + newlines) — not part of JSON.
            if constrained and grammar_matcher is not None:
                if tok_id in skip_token_ids:
                    if tok_id == TOOL_CALL_END_FIRST_ID:
                        constrained = False  # back to free text after end marker
                    continue
                try:
                    grammar_matcher.accept_token(tok_id)
                except Exception:
                    break

    gen_ids = ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


# ── xgrammar constrained decoding helpers ─────────────────────────────────

import xgrammar as xgr

# Cache tokenizer info + compiler (expensive to create, reuse across calls)
_xgr_tokenizer_info: xgr.TokenizerInfo | None = None
_xgr_compiler: xgr.GrammarCompiler | None = None


def _get_xgr_compiler(tokenizer) -> xgr.GrammarCompiler:
    """Get or create a cached xgrammar compiler for the tokenizer."""
    global _xgr_tokenizer_info, _xgr_compiler
    if _xgr_compiler is None or _xgr_tokenizer_info is None:
        # Load a fresh HF tokenizer for xgrammar (gigatoken wrapper not supported)
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained("research/checkpoints/lfm25_tokenizer")
        # Pass vocab_size=65536 to match model config (tokenizer has 64416)
        # This ensures the bitmask covers all model logits
        _xgr_tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            hf_tok, vocab_size=65536, stop_token_ids=[7])
        _xgr_compiler = xgr.GrammarCompiler(_xgr_tokenizer_info)
    return _xgr_compiler


def create_tool_grammar(tokenizer, tools: list[dict]) -> tuple[xgr.GrammarMatcher, Any]:
    """Create an xgrammar matcher that constrains tool call JSON to valid names.

    The grammar enforces:
      - "name" must be one of the registered tool names (enum)
      - "arguments" must be a JSON object
      - Valid JSON structure

    Used with qwen_generate() in two-phase mode: free text first, then
    constrained JSON when the model starts emitting a tool call.

    Returns (matcher, bitmask) for use with qwen_generate().
    """
    compiler = _get_xgr_compiler(tokenizer)

    tool_names = [t["name"] for t in tools]

    # Schema for a single tool call object
    tool_call_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": tool_names},
            "arguments": {"type": "object"}
        },
        "required": ["name", "arguments"]
    }

    compiled = compiler.compile_json_schema(json.dumps(tool_call_schema))
    matcher = xgr.GrammarMatcher(compiled)
    bitmask = xgr.allocate_token_bitmask(1, _xgr_tokenizer_info.vocab_size)

    return matcher, bitmask


def make_grammar_logits_processor(grammar_matcher, bitmask, tokenizer):
    """Build a logits_processor callback for ForgeEngine.generate_raw().

    Implements the same two-phase constrained decoding as qwen_generate():
      - Phase 1 (free text): no masking. When TOOL_CALL_START_FIRST_ID (10)
        is emitted, switch to constrained mode.
      - Phase 2 (constrained): apply xgrammar bitmask to enforce valid
        tool-call JSON. Skip structural tokens (end marker, newlines).
        When TOOL_CALL_END_FIRST_ID (11) is emitted, switch back to free text.

    The processor is stateful (tracks constrained mode + accepted tokens).
    """
    newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
    skip_token_ids = {TOOL_CALL_END_FIRST_ID} | newline_ids
    state = {"constrained": False}

    def logits_processor(next_logits: torch.Tensor,
                         generated_ids: list[int]) -> torch.Tensor:
        # FIRST accept the previous token into the grammar matcher, so the
        # bitmask for the NEXT token reflects the full state. Doing this
        # after fill_next_token_bitmask leaves the mask one step stale and
        # lets the model re-emit tokens it already generated (garbage like
        # {"{"namename"") — only visible when the model emits the real
        # <|tool_call_start|> special token (id 10).
        if state["constrained"] and generated_ids and grammar_matcher is not None:
            last_tok = generated_ids[-1]
            if last_tok == TOOL_CALL_END_FIRST_ID:
                # End marker → back to free text
                state["constrained"] = False
            elif last_tok not in skip_token_ids and last_tok != TOOL_CALL_START_FIRST_ID:
                try:
                    grammar_matcher.accept_token(last_tok)
                except Exception:
                    pass  # grammar mismatch — let it slide

        # Check if we just entered a tool call (last token was start marker)
        if generated_ids and not state["constrained"]:
            if generated_ids[-1] == TOOL_CALL_START_FIRST_ID:
                state["constrained"] = True

        if not state["constrained"]:
            return next_logits

        # In constrained mode: apply grammar bitmask (state is now current)
        if grammar_matcher is not None and bitmask is not None:
            grammar_matcher.fill_next_token_bitmask(bitmask)
            # Move bitmask to same device as logits (xgrammar allocates on CPU)
            bitmask_dev = bitmask.to(next_logits.device)
            # Save skip-list logits before masking — only when grammar done.
            # If always un-masked, model emits newline spam or the end marker
            # mid-JSON, making the tool call unparseable.
            skip_logits = {}
            grammar_done = grammar_matcher.is_terminated()
            if grammar_done:
                for sid in skip_token_ids:
                    if 0 <= sid < next_logits.shape[-1]:
                        skip_logits[sid] = next_logits[:, sid].clone()
            xgr.apply_token_bitmask_inplace(next_logits, bitmask_dev)
            # Restore skip-list logits
            for sid, val in skip_logits.items():
                next_logits[:, sid] = val

        return next_logits

    return logits_processor
