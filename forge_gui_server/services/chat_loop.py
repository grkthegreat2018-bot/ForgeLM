"""Chat generation loop — async port of pages/chat.py::_ChatWorker.

Streams raw model chunks to the SSE consumer (the React client does the
<think>/<tool_call> marker splitting for display, same role the Qt
_StreamParser played). After each generation the accumulated text is
parsed for tool calls; calls execute via ToolHarness and the loop
continues, mirroring the Qt worker's max_rounds=6 behavior.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chat")

# Matches the Qt chat worker's hardcoded stop ids (im_end + eos for the
# Jamba-derived tokenizer).
_EOS_IDS = [2, 519]

# Jamba structural token ids — when tools are enabled, generation must
# stop at the end of a <tool_call> block (532) or the start of a
# <tool_response> block (539). Without these stops the model continues
# past its call and hallucinates the tool's output itself.
_TOOL_CALL_END_ID = 532       # </tool_call>
_TOOL_RESP_START_ID = 539     # <tool_response>
_TOOL_CALL_START = "<tool_call>"
_RE_NAME_HINT = re.compile(r'\{"name"\s*:\s*"')

# Special tokens that must never be persisted in stored message content —
# they are structural markers, not user-visible text.
_SPECIAL_TOKENS = (
    "<|im_start|>", "<|im_end|>", "<|startoftext|>", "<|endoftext|>",
    "<tool_response>", "</tool_response>",
)


def _clean_content(text: str) -> str:
    for t in _SPECIAL_TOKENS:
        text = text.replace(t, "")
    return text


class ChatCancel:
    def __init__(self) -> None:
        self._flag = threading.Event()

    def cancel(self) -> None:
        self._flag.set()

    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()


async def run_chat(runtime, harness, messages: list[dict],
                   config_name: str = "", max_new_tokens: int = 512,
                   temperature: float = 0.7, top_p: float = 0.95,
                   top_k: int = 80, repetition_penalty: float = 1.05,
                   tools_enabled: bool = True, thinking: bool = True,
                   max_rounds: int = 6,
                   cancel: ChatCancel | None = None,
                   ) -> AsyncIterator[dict]:
    """Yield chat events: token / tool_call / tool_result / done / error."""
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancel = cancel or ChatCancel()

    def emit(item):
        loop.call_soon_threadsafe(q.put_nowait, item)

    def worker():
        from forge.self_play.discovery.qwen_adapter import (
            qwen_parse_tool_calls, render_messages_for_config)
        try:
            if not runtime.is_ready():
                emit(("error", "No model resident — load one on the "
                      "Models page first."))
                return
            defs = (harness.chat_tool_defs()
                    if (harness and tools_enabled) else None)
            rounds = max_rounds if tools_enabled else 1
            conv = list(messages)
            new_messages: list[dict] = []

            for _round in range(rounds):
                if cancel.cancelled:
                    break
                rendered = render_messages_for_config(
                    conv, config_name=config_name, tools=defs,
                    add_generation_prompt=True, thinking=thinking)
                chunks: list[str] = []
                eos_ids = (_EOS_IDS + [_TOOL_CALL_END_ID, _TOOL_RESP_START_ID]
                           if defs else _EOS_IDS)
                with runtime.acquire(timeout_s=120.0) as engine:
                    for tok in engine.generate_stream(
                            rendered, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            skip_special_tokens=False,
                            eos_token_ids=eos_ids):
                        if cancel.cancelled:
                            break
                        chunks.append(tok)
                        emit(("token", tok))
                raw = "".join(chunks)
                tool_calls, content = qwen_parse_tool_calls(raw)
                if tool_calls:
                    # anything the model emitted after its first call is a
                    # hallucinated continuation (a simulated tool response)
                    # — keep only the musing that precedes the call
                    cut = raw.find(_TOOL_CALL_START)
                    if cut < 0:
                        m = _RE_NAME_HINT.search(raw)
                        cut = m.start() if m else -1
                    if cut >= 0:
                        content = raw[:cut]
                msg = {"role": "assistant", "content": _clean_content(content or ""),
                       "tool_calls": tool_calls or None}
                conv.append(msg)
                new_messages.append(msg)

                if not tool_calls or not harness:
                    break

                for tc in tool_calls:
                    if cancel.cancelled:
                        break
                    call = tc if isinstance(tc, dict) else {"name": str(tc)}
                    emit(("tool_call", call))
                    rec = harness.execute_calls([call])[0]
                    emit(("tool_result", rec))
                    from forge_gui.api.agent_tools import (
                        tool_results_to_text)
                    tmsg = {"role": "tool",
                            "name": call.get("name", "tool"),
                            "content": tool_results_to_text(rec)}
                    conv.append(tmsg)
                    new_messages.append(tmsg)

            emit(("done", {"messages": new_messages}))
        except Exception as e:
            logger.warning("chat loop failed: %s", e, exc_info=True)
            emit(("error", f"{type(e).__name__}: {e}"))
        finally:
            emit(("end", None))

    loop.run_in_executor(_pool, worker)
    while True:
        kind, data = await q.get()
        if kind == "end":
            break
        yield {"type": kind, "data": data}
