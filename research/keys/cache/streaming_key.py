"""StreamingLLM attention sink key — cache eviction policy, no weight changes.

Reference: Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
(ICLR 2024, MIT). Enables infinite-length generation by pinning the first N
tokens (attention sinks) and keeping a sliding window of recent K tokens.
"""
from typing import Dict

import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class StreamingLLMKey(Key):
    """StreamingLLM attention sink key — pin first N tokens + sliding window.

    No weight changes — produces a cache mask that keeps attention sinks
    (first N tokens) and a sliding window of recent K tokens.
    Enables stable generation over millions of tokens.

    Key class: TRIVIAL — fixed policy, no data or training.
    """

    @property
    def name(self) -> str:
        return "streaming_llm"

    @property
    def description(self) -> str:
        return "Attention sink + sliding window cache eviction policy (Xiao et al. ICLR 2024)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Build a cache mask: first n_sinks + last window_size tokens kept."""
        seq_len = int(data["seq_len"])
        n_sinks = int(data.get("n_sinks", 4))
        window_size = int(data.get("window_size", 1024))

        cache_mask = torch.zeros(seq_len, dtype=torch.bool)
        n_keep_sinks = min(n_sinks, seq_len)
        cache_mask[:n_keep_sinks] = True
        n_keep_window = min(window_size, max(seq_len - n_keep_sinks, 0))
        if n_keep_window > 0:
            cache_mask[seq_len - n_keep_window:] = True

        n_kept = int(cache_mask.sum().item())
        return KeyResult(
            success=True,
            weights={"cache_mask": cache_mask},
            data={"n_sinks": n_sinks, "window_size": window_size, "n_kept": n_kept},
            metadata={"seq_len": seq_len, "n_kept": n_kept},
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """Passthrough — no weights to recover (policy is not invertible)."""
        return KeyResult(success=True, weights=weights,
                         metadata={"note": "trivial key, nothing to recover"})


class StreamingKVCache:
    """KV cache that implements StreamingLLM policy.

    Keeps first n_sinks tokens + sliding window of recent tokens.
    Automatically evicts tokens outside this range.
    """

    def __init__(self, n_sinks: int = 4, window_size: int = 1024):
        self.n_sinks = n_sinks
        self.window_size = window_size
        self._sink_k: list = []
        self._sink_v: list = []
        self._win_k: list = []
        self._win_v: list = []

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Add new token KV. Evicts old tokens outside window (except sinks)."""
        if len(self._sink_k) < self.n_sinks:
            self._sink_k.append(k)
            self._sink_v.append(v)
        else:
            self._win_k.append(k)
            self._win_v.append(v)
            if len(self._win_k) > self.window_size:
                self._win_k.pop(0)
                self._win_v.pop(0)

    def get(self):
        """Returns current K, V tensors (sinks + window)."""
        ks = self._sink_k + self._win_k
        vs = self._sink_v + self._win_v
        if not ks:
            return None, None
        return torch.stack(ks), torch.stack(vs)

    @property
    def size(self) -> int:
        return len(self._sink_k) + len(self._win_k)
