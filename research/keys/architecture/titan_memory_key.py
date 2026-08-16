"""TITAN Neural Memory — test-time training long-term memory.

TITAN (Behrouz et al., ICLR 2025) augments short-range attention with a
trainable long-term memory module whose weights update *at test time* via a
"surprise" signal — context is compressed into memory weights instead of an
ever-growing KV tensor, keeping VRAM flat for long sequences.

This implementation is config-driven and lossless-at-start:
  - memory weight matrix zero-initialized  -> output = 0
  - gate zero-initialized                  -> residual path unchanged
so a pretrained checkpoint loads and produces identical outputs until the
memory is trained.

Memory update (TTT-lite): a Hebbian-style surprise step
    dW += lr * (x_cur outer x_cur) - lr * (pred outer x_cur)
computed with plain tensor ops in forward; with `training=True` the module
also exposes `update()` for explicit test-time adaptation during inference.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from research.keys.misc.base import Key, KeyClass, KeyResult


class TitanMemory(nn.Module):
    """Gated neural long-term memory with surprise-based updates.

    Args:
        d_model: hidden size (memory is (d_model, d_model) or low-rank).
        rank: 0 = full (d_model, d_model); >0 = low-rank U(d, r) @ V(r, d).
        update_lr: surprise-update learning rate (0 = updates disabled).
    """

    def __init__(self, d_model: int = 2048, rank: int = 0,
                 update_lr: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.update_lr = update_lr

        if rank > 0:
            self.u = nn.Parameter(torch.zeros(d_model, rank))
            self.v = nn.Parameter(torch.zeros(rank, d_model))
        else:
            self.memory = nn.Parameter(torch.zeros(d_model, d_model))
        # Zero-init gate => lossless at start.
        self.gate = nn.Parameter(torch.zeros(1))

    def _read(self, x: torch.Tensor) -> torch.Tensor:
        """memory read: x @ W."""
        if self.rank > 0:
            return x @ (self.u @ self.v)
        return x @ self.memory

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read memory and gate it. Lossless while gate == 0."""
        return self.gate * self._read(x)

    def update(self, x: torch.Tensor, pred: torch.Tensor | None = None) -> None:
        """Test-time memory update (surprise = x - pred, Hebbian form).

        With pred=None, uses the current memory's own prediction for x
        (self-supervised surprise). Updates in-place — no gradients.
        """
        if self.update_lr == 0.0:
            return
        xf = x.detach().float()
        if pred is None:
            pred = self._read(xf)
        else:
            pred = pred.detach().float()
        # Surprise outer-product accumulation over batch*tokens.
        dW = (xf.reshape(-1, self.d_model).T @ (xf - pred).reshape(-1, self.d_model))
        if self.rank > 0:
            with torch.no_grad():
                self.u.add_(self.update_lr * (dW @ self.v.detach().float()) /
                            max(xf.numel() // self.d_model, 1))
                self.v.add_(self.update_lr * (self.u.detach().float().T @ dW) /
                            max(xf.numel() // self.d_model, 1))
        else:
            with torch.no_grad():
                self.memory.add_(self.update_lr * dW /
                                 max(xf.numel() // self.d_model, 1))


class TitanMemoryKey(Key):
    """Adds/removes the TITAN memory parameters on a state dict.

    forward: inject zero-init memory (+ gate) into every block.
    reverse: strip the memory keys (back to a plain transformer state).
    Key class: PARTIAL.
    """

    def __init__(self, d_model: int = 2048, rank: int = 0):
        self.d_model = d_model
        self.rank = rank

    @property
    def name(self) -> str:
        return "titan_memory"

    @property
    def description(self) -> str:
        return f"TITAN neural memory (d={self.d_model}, rank={self.rank})"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(data)
            n_layers = 0
            for k in state:
                if k.startswith("blocks.") and ".ln1.weight" in k:
                    n_layers = max(n_layers, int(k.split(".")[1]) + 1)
            added = 0
            for i in range(n_layers):
                base = f"blocks.{i}.memory"
                if self.rank > 0:
                    state[f"{base}.u"] = torch.zeros(self.d_model, self.rank)
                    state[f"{base}.v"] = torch.zeros(self.rank, self.d_model)
                    added += 2
                else:
                    state[f"{base}.memory"] = torch.zeros(
                        self.d_model, self.d_model)
                    added += 1
                state[f"{base}.gate"] = torch.zeros(1)
                added += 1
            return KeyResult(success=True, weights=state,
                             metadata={"n_layers": n_layers, "keys_added": added})
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        try:
            state = dict(weights)
            removed = 0
            for k in list(state.keys()):
                if ".memory." in k:
                    del state[k]
                    removed += 1
            return KeyResult(success=True, weights=state,
                             metadata={"keys_removed": removed})
        except Exception as e:
            return KeyResult(success=False, error=str(e))
