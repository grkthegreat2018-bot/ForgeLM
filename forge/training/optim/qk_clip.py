"""QK-Clip: attention-logit overflow clipping (Kimi K2 MuonClip).

Tracks the per-head maximum attention logit S_max = max|q·k| / sqrt(d) during
training. After each optimizer step, any head whose S_max exceeded tau has its
Q/K projection rows rescaled by sqrt(gamma) with gamma = tau / S_max, which
caps the logit at tau without touching the softmax output distribution's
argmax structure (uniform logit rescale per head).

Kimi K2 pre-trained 15.5T tokens with zero loss spikes using this mechanism
(tau = 30 or 100). Overhead: one (B, H, T, T) logit computation per forward
when monitoring is enabled (training-only, opt-in), plus a cheap row rescale.

Usage:
    from forge.training.optim.qk_clip import QKClipMonitor
    monitor = QKClipMonitor.attach(model, tau=100.0)
    ... training loop ...
    optimizer.step()
    monitor.clip(model)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QKClipMonitor:
    """Collects per-head max attention logits and applies QK-Clip rescaling."""

    def __init__(self, tau: float = 100.0):
        self.tau = float(tau)
        self.attention_modules: list[nn.Module] = []
        self.head_max: dict[int, torch.Tensor] = {}
        self.clip_events = 0

    @classmethod
    def attach(cls, model: nn.Module, tau: float = 100.0) -> QKClipMonitor:
        monitor = cls(tau)
        for layer_idx, m in enumerate(model.modules()):
            if type(m).__name__ == "GroupedQueryAttention":
                m._qk_clip_monitor = monitor
                m._qk_clip_layer_idx = layer_idx
                monitor.attention_modules.append(m)
        return monitor

    @torch.no_grad()
    def observe(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor):
        """Record per-head max |logit|. q: (B, H, T, hd), k: (B, H, S, hd)."""
        hd = q.shape[-1]
        logits = (q.float() @ k.float().transpose(-2, -1)) / (hd ** 0.5)
        head_max = logits.abs().amax(dim=(0, 2, 3))  # (H,)
        prev = self.head_max.get(layer_idx)
        self.head_max[layer_idx] = head_max if prev is None else torch.maximum(prev, head_max)

    @torch.no_grad()
    def clip(self, model: nn.Module) -> int:
        """Rescale Q/K projection rows for heads whose max logit exceeded tau.

        GQA note: query heads sharing a KV head are clipped with the group-max
        gamma so the shared K rows stay consistent (worst head is capped
        exactly; other heads in the group are over-capped, which is safe).
        """
        n_clipped = 0
        for m in self.attention_modules:
            layer_idx = getattr(m, "_qk_clip_layer_idx", None)
            head_max = self.head_max.get(layer_idx)
            if head_max is None:
                continue
            hd = m.head_dim
            n_rep = max(1, m.n_heads // m.n_kv_heads)
            kv_gamma = torch.ones(m.n_kv_heads, device=head_max.device)
            q_gamma = torch.ones(m.n_heads, device=head_max.device)
            exceed = head_max > self.tau
            if not exceed.any():
                continue
            gamma = (self.tau / head_max.clamp_min(1e-8)).clamp(max=1.0)
            q_gamma = torch.where(exceed, gamma, q_gamma)
            for kv_idx in range(m.n_kv_heads):
                q_slice = slice(kv_idx * n_rep, (kv_idx + 1) * n_rep)
                group_gamma = gamma[q_slice].min()
                kv_gamma[kv_idx] = group_gamma
            sqrt_q = q_gamma.sqrt()
            sqrt_k = kv_gamma.sqrt()
            m.q_proj.weight.mul_(sqrt_q.repeat_interleave(hd).view(-1, 1))
            m.k_proj.weight.mul_(sqrt_k.repeat_interleave(hd).view(-1, 1))
            n_clipped += int(exceed.sum().item())
            self.head_max[layer_idx] = torch.zeros_like(head_max)
        self.clip_events += n_clipped
        return n_clipped
