"""Logit Capping Key — clamp logits for training/inference stability.

Logit capping (used in Gemma 2) clamps the logits to a fixed range before
softmax, preventing extremely large logits that cause numerical issues.

This is a TRIVIAL key — no weights, just a runtime clamp operation.

Usage:
    from research.keys.logit_cap_key import LogitCapKey, apply_logit_cap
    # Apply to model
    apply_logit_cap(model, cap=30.0)
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from .base import Key, KeyClass, KeyResult


class LogitCapKey(Key):
    """Logit capping key — clamp logits to ±cap."""

    def __init__(self, cap: float = 30.0):
        self.cap = cap

    @property
    def name(self) -> str:
        return "logit_cap"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL  # No weights, runtime only

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, weights={}, metadata={"cap": self.cap})

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, data={})


def apply_logit_cap(model, cap: float = 30.0):
    """Wrap model's head with logit capping.

    After applying, model.head output is clamped to [-cap, cap].
    """
    original_head = model.head

    class CappedHead(nn.Module):
        def __init__(self, head, cap):
            super().__init__()
            self.head = head
            self.cap = cap

        def forward(self, x):
            logits = self.head(x)
            return logits.clamp(-self.cap, self.cap)

    model.head = CappedHead(original_head, cap)
    print(f"  [LogitCap] Applied cap=±{cap} to model head")
    return model


def apply_logit_cap_to_forward(model, cap: float = 30.0):
    """Patch model forward to clamp logits (non-invasive).

    Instead of wrapping the head, this patches the forward method to
    clamp logits before returning.
    """
    original_forward = model.forward

    def patched_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        # result is (logits, loss) or (logits, loss, presents) or (logits, loss, presents, hidden)
        if isinstance(result, tuple) and result[0] is not None:
            result = list(result)
            result[0] = result[0].clamp(-cap, cap)
            result = tuple(result)
        return result

    model.forward = patched_forward
    print(f"  [LogitCap] Applied cap=±{cap} to model forward")
    return model
