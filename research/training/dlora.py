"""R38-1: DLoRA — Decoupled LoRA with Dynamic Rank.

LoRA adapters whose rank grows during training. Each adapter starts at a
low rank (e.g. r=4) and is promoted when its gradient norm exceeds a
threshold, allocating capacity only to layers that need it.

Growing appends new singular directions to the decomposition. The new
rows of A and columns of B are zero-initialised so a grow is an exact
no-op until the optimiser moves them — the forward output is unchanged
at the moment of growth, preserving training stability.

Forward:  y = x + scale * (x @ A[:r].T @ B[:r].T)
          where r is the *current* (possibly grown) rank.

Compatible with the manual-LoRA patterns in ``bitnet_lora.py``: the
adapter exposes ``lora_A`` / ``lora_B`` / ``scale`` attributes and a
``forward`` method, so ``merge_lora_adapters`` and the BitNet/NF4 forward
hooks work unchanged. The dynamic-rank bookkeeping (``current_rank``,
``grow_rank``, ``should_grow``) lives on the adapter and on the
container returned by ``apply_to_model``.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ── DLoRA adapter ─────────────────────────────────────────────────────────

class DLoRAAdapter(nn.Module):
    """LoRA adapter with a growable rank.

    A: (max_rank, in_features)  — kaiming-init for the first ``rank`` rows,
        zero-init for the reserved (max_rank - rank) rows.
    B: (out_features, max_rank) — zero-init (LoRA starts as a no-op).

    Only the first ``current_rank`` rows/cols participate in the forward
    pass; the rest are dormant capacity that ``grow_rank`` can activate.
    """

    def __init__(self, in_features: int, out_features: int,
                 initial_rank: int = 4, max_rank: int = 32,
                 alpha: int = 8):
        super().__init__()
        assert 0 < initial_rank <= max_rank
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.initial_rank = initial_rank
        self.current_rank = initial_rank
        # alpha is fixed; effective scale shrinks as rank grows so the
        # *contribution magnitude* of the active block stays controlled.
        self.alpha = alpha
        # Full-size buffers; dormant slices are zero.
        self.lora_A = nn.Parameter(torch.zeros(max_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))
        # Kaiming-init the active block of A; B stays zero (no-op start).
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.lora_A[:initial_rank], a=math.sqrt(5))
        # Gradient-norm history for growth decisions.
        self.grad_norm_history: list[float] = []

    @property
    def scale(self) -> float:
        """Effective scaling = alpha / initial_rank (fixed).

        Using initial_rank (not current_rank) ensures that growing the
        rank is a no-op at growth time — the new B columns are zero, so
        the additional rank contributes nothing, and the scale doesn't
        change, so the output is identical.
        """
        return self.alpha / max(self.initial_rank, 1)

    # ── rank management ───────────────────────────────────────────────────
    def should_grow(self, grad_norm: float, threshold: float) -> bool:
        """Return True if the adapter should grow its rank.

        Growth is triggered when ``grad_norm`` exceeds ``threshold`` AND
        the adapter has not yet reached ``max_rank``. The grad norm is
        recorded so callers can inspect the history.
        """
        self.grad_norm_history.append(float(grad_norm))
        return self.current_rank < self.max_rank and grad_norm > threshold

    def grow_rank(self, new_rank: int | None = None) -> int:
        """Activate additional rank up to ``new_rank`` (or +1 if None).

        Newly activated rows of A are kaiming-initialised; the
        corresponding columns of B stay zero, so the grow is an exact
        no-op for the forward pass at the moment of growth. Returns the
        new current rank.
        """
        if new_rank is None:
            new_rank = self.current_rank + 1
        new_rank = min(new_rank, self.max_rank)
        if new_rank <= self.current_rank:
            return self.current_rank
        with torch.no_grad():
            # Initialise the freshly activated rows of A. B columns stay
            # zero → forward unchanged at growth time.
            nn.init.kaiming_uniform_(
                self.lora_A[self.current_rank:new_rank], a=math.sqrt(5))
        self.current_rank = new_rank
        return self.current_rank

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.current_rank
        if r <= 0:
            return torch.zeros_like(x)
        A = self.lora_A[:r]          # (r, in)
        B = self.lora_B[:, :r]       # (out, r)
        return self.scale * (x @ A.T @ B.T)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.current_rank}/{self.max_rank}, "
                f"alpha={self.alpha}")


# ── Container tracking all adapters ───────────────────────────────────────

class DLoRA:
    """Manages a set of DLoRA adapters applied to a model.

    Tracks adapters by layer index so external training loops can call
    ``maybe_grow(step_grad_norms)`` once per step to promote layers whose
    gradients are hot.
    """

    def __init__(self, initial_rank: int = 4, max_rank: int = 32,
                 growth_threshold: float = 0.01, alpha: int = 8):
        self.initial_rank = initial_rank
        self.max_rank = max_rank
        self.growth_threshold = growth_threshold
        self.alpha = alpha
        self.adapters: list[DLoRAAdapter] = []
        # layer_idx → adapter, for targeted growth.
        self.layer_map: dict[int, DLoRAAdapter] = {}

    # ── application ───────────────────────────────────────────────────────
    def apply_to_model(self, model: nn.Module,
                       target_modules: list[str] | None = None,
                       min_size: int = 64) -> tuple[int, list[nn.Parameter]]:
        """Replace eligible Linear layers' forward with a DLoRA adapter.

        Mirrors ``add_lora_adapters`` from bitnet_lora.py: freezes base
        weights, attaches a ``lora_adapter`` attribute, and wraps forward
        for plain nn.Linear. Returns (n_adapters, trainable_params).
        """
        n_adapters = 0
        params: list[nn.Parameter] = []
        idx = 0

        def is_quant_linear(mod):
            return type(mod).__name__ in ("IRIFP4Linear", "NF4Linear")

        def find_and_add(module, prefix=""):
            nonlocal n_adapters, idx
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                has_weight = isinstance(getattr(child, 'weight', None),
                                        nn.Parameter)
                has_dims = (hasattr(child, 'in_features')
                            and hasattr(child, 'out_features'))
                is_quant = is_quant_linear(child)
                if has_dims and (has_weight or is_quant):
                    is_target = True
                    if target_modules is not None:
                        is_target = any(t in full_name
                                        for t in target_modules)
                    if (child.in_features < min_size
                            or child.out_features < min_size):
                        is_target = False
                    if is_target:
                        adapter = DLoRAAdapter(
                            child.in_features, child.out_features,
                            initial_rank=self.initial_rank,
                            max_rank=self.max_rank,
                            alpha=self.alpha)
                        dtype = (torch.bfloat16 if is_quant else
                                 child.weight.dtype)
                        dev = (child.weight_packed.device if is_quant
                               else child.weight.device)
                        adapter = adapter.to(dev).to(dtype)
                        setattr(child, 'lora_adapter', adapter)
                        if not is_quant:
                            orig = child.forward
                            child._lora_orig_forward = orig

                            def make_fwd(o, la):
                                def f(x):
                                    return o(x) + la(x)
                                return f
                            child.forward = make_fwd(orig, adapter)
                        # Freeze base weights.
                        if has_weight:
                            child.weight.requires_grad = False
                        if (hasattr(child, 'bias') and child.bias
                                is not None
                                and isinstance(child.bias, nn.Parameter)):
                            child.bias.requires_grad = False
                        self.adapters.append(adapter)
                        self.layer_map[idx] = adapter
                        params.extend([adapter.lora_A, adapter.lora_B])
                        n_adapters += 1
                        idx += 1
                find_and_add(child, full_name)

        find_and_add(model)
        return n_adapters, params

    # ── dynamic rank ──────────────────────────────────────────────────────
    def grow_rank(self, layer_idx: int, new_rank: int) -> int:
        """Grow the rank of a specific layer's adapter."""
        adapter = self.layer_map[layer_idx]
        return adapter.grow_rank(new_rank)

    def should_grow(self, grad_norm: float, threshold: float) -> bool:
        """Stateless threshold check (instance threshold used if omitted)."""
        return grad_norm > threshold

    def maybe_grow(self, grad_norms: dict[int, float],
                   threshold: float | None = None) -> list[int]:
        """Grow adapters whose grad norm exceeds the threshold.

        Args:
            grad_norms: {layer_idx: grad_norm} for this step.
            threshold: override the instance growth_threshold.
        Returns the list of layer indices that grew this step.
        """
        thr = threshold if threshold is not None else self.growth_threshold
        grew: list[int] = []
        for layer_idx, gn in grad_norms.items():
            adapter = self.layer_map.get(layer_idx)
            if adapter is None:
                continue
            if adapter.should_grow(gn, thr):
                adapter.grow_rank()
                grew.append(layer_idx)
        return grew

    def total_rank(self) -> int:
        """Sum of current ranks across all adapters."""
        return sum(a.current_rank for a in self.adapters)


def apply_to_model(model: nn.Module, initial_rank: int = 4,
                   max_rank: int = 32, growth_threshold: float = 0.01,
                   alpha: int = 8, target_modules: list[str] | None = None,
                   min_size: int = 64
                   ) -> tuple[DLoRA, int, list[nn.Parameter]]:
    """Convenience wrapper: build a DLoRA manager and apply it.

    Returns (dlora_manager, n_adapters, trainable_params).
    """
    mgr = DLoRA(initial_rank=initial_rank, max_rank=max_rank,
                growth_threshold=growth_threshold, alpha=alpha)
    n, params = mgr.apply_to_model(model, target_modules=target_modules,
                                   min_size=min_size)
    return mgr, n, params
