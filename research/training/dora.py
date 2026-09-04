"""R38-2: DoRA — Weight-Decomposed Low-Rank Adaptation.

Decomposes each pre-trained weight matrix into a *magnitude* (per-output
scalar) and a *direction* (unit-norm rows). LoRA is applied to the
direction only; the magnitude is fine-tuned as a separate scalar
parameter. At merge time the adapted weight is folded back into a single
matrix so the result is a drop-in for any downstream Linear.

  W = m ⊙ (V / ‖V‖)                  (decomposition)
  V' = V + scale * (B @ A)            (LoRA on direction)
  W' = m ⊙ (V' / ‖V'‖)               (DoRA forward)

When delta_V == 0 (fresh adapter, B zero-init) the reconstruction is
lossless up to fp32 numerical precision: W' ≈ W. The merge path produces
``W_merged = m ⊙ normalize(V + scale * B @ A)`` which is a standard
nn.Linear weight with no runtime overhead.

Source: DoRA (Liu et al. ICML 2024, arXiv 2402.09353).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def _row_normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each row of ``v`` (out_features, in_features) to unit L2."""
    norm = v.norm(dim=1, keepdim=True).clamp(min=eps)
    return v / norm


# ── DoRA linear layer ─────────────────────────────────────────────────────

class DoRALinear(nn.Module):
    """DoRA-wrapped linear layer.

    Stores the decomposed magnitude (out_features,) and direction
    V (out_features, in_features) as frozen buffers, plus a trainable
    magnitude vector and a LoRA adapter (A, B) on the direction.

    Forward:  y = x @ (m ⊙ normalize(V + scale * B @ A)).T  [+ bias]
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: int = 16, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank
        # Direction (frozen) — set by load_from_weight.
        self.register_buffer('V', torch.zeros(out_features, in_features))
        # Magnitude (trainable).
        self.magnitude = nn.Parameter(torch.ones(out_features))
        # LoRA on direction: delta_V = scale * (B @ A).
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Cached normalized base direction (frozen) for the no-adapter path.
        self.register_buffer('V_norm', torch.zeros(out_features, in_features))

    @torch.no_grad()
    def load_from_weight(self, w: torch.Tensor):
        """Decompose a weight tensor (out, in) into magnitude + direction."""
        assert w.shape == (self.out_features, self.in_features)
        w = w.to(torch.float32)
        norm = w.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.magnitude.data.copy_(norm.squeeze(1))
        self.V.data.copy_(w / norm)
        self.V_norm.data.copy_(self.V.data)  # already unit-norm

    def _effective_weight(self) -> torch.Tensor:
        """Compute the effective weight m ⊙ normalize(V + delta_V)."""
        delta = self.lora_B @ self.lora_A          # (out, in)
        v_adapted = self.V + self.scale * delta
        v_norm = _row_normalize(v_adapted)
        return self.magnitude.unsqueeze(1) * v_norm  # (out, in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._effective_weight().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, bias)

    @torch.no_grad()
    def merge(self) -> torch.Tensor:
        """Merge adapter + magnitude into a single weight matrix.

        Returns the merged (out_features, in_features) weight and resets
        the adapter to zero so a subsequent forward is identical.
        """
        w = self._effective_weight().to(torch.float32)
        # Zero the adapter → delta_V = 0, magnitude already baked in.
        self.lora_B.zero_()
        # Re-decompose so V/magnitude reflect the merged weight.
        self.load_from_weight(w)
        return w

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, dora=True")


# ── DoRA manager ──────────────────────────────────────────────────────────

class DoRA:
    """Applies DoRA to a model by replacing nn.Linear with DoRALinear.

    Tracks wrapped layers so they can be merged back into plain
    nn.Linear weights for inference / checkpointing.
    """

    def __init__(self, rank: int = 8, alpha: int = 16):
        self.rank = rank
        self.alpha = alpha
        self.wrapped: list[tuple[nn.Module, str, DoRALinear]] = []

    def apply_to_model(self, model: nn.Module,
                       target_modules: list[str] | None = None,
                       min_size: int = 64) -> tuple[int, list[nn.Parameter]]:
        """Replace eligible nn.Linear with DoRALinear.

        Returns (n_wrapped, trainable_params).
        """
        n_wrapped = 0
        params: list[nn.Parameter] = []

        def find_and_replace(module, prefix=""):
            nonlocal n_wrapped
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear):
                    is_target = True
                    if target_modules is not None:
                        is_target = any(t in full_name
                                        for t in target_modules)
                    if (child.in_features < min_size
                            or child.out_features < min_size):
                        is_target = False
                    if is_target:
                        dora = DoRALinear(
                            child.in_features, child.out_features,
                            rank=self.rank, alpha=self.alpha,
                            bias=child.bias is not None)
                        dora.load_from_weight(child.weight.data)
                        if child.bias is not None:
                            dora.bias.data.copy_(child.bias.data)
                        dora = dora.to(child.weight.device).to(
                            child.weight.dtype
                            if child.weight.dtype != torch.float32
                            else torch.float32)
                        setattr(module, name, dora)
                        self.wrapped.append((module, name, dora))
                        params.extend([dora.magnitude, dora.lora_A,
                                       dora.lora_B])
                        n_wrapped += 1
                find_and_replace(child, full_name)

        find_and_replace(model)
        return n_wrapped, params

    def merge(self, model: nn.Module) -> int:
        """Merge all DoRA layers back into plain nn.Linear.

        Replaces each DoRALinear with an nn.Linear whose weight is the
        merged DoRA weight. Returns the number of layers merged.
        """
        n_merged = 0
        for parent, name, dora in list(self.wrapped):
            w = dora.merge().to(dora.V.dtype)
            lin = nn.Linear(dora.in_features, dora.out_features,
                            bias=dora.bias is not None)
            with torch.no_grad():
                lin.weight.data.copy_(w)
                if dora.bias is not None:
                    lin.bias.data.copy_(dora.bias.data)
            lin = lin.to(dora.V.device).to(dora.V.dtype)
            setattr(parent, name, lin)
            n_merged += 1
        self.wrapped.clear()
        return n_merged


def apply_to_model(model: nn.Module, rank: int = 8, alpha: int = 16,
                   target_modules: list[str] | None = None,
                   min_size: int = 64
                   ) -> tuple[DoRA, int, list[nn.Parameter]]:
    """Convenience wrapper: build a DoRA manager and apply it.

    Returns (dora_manager, n_wrapped, trainable_params).
    """
    mgr = DoRA(rank=rank, alpha=alpha)
    n, params = mgr.apply_to_model(model, target_modules=target_modules,
                                   min_size=min_size)
    return mgr, n, params
