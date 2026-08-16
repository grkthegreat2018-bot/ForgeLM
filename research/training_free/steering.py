"""Activation steering / task vectors — inference-time behavior control.

Representation engineering: find a direction in the residual stream that
separates two behaviors (e.g. successful vs. failed runs), then add a scaled
copy of that direction at chosen layers during generation. No gradients, no
parameter updates — pure forward-pass intervention.

Usage:
    steerer = ActivationSteerer(model)
    acts_ok = steerer.collect_activations(tokenizer, success_prompts)
    acts_bad = steerer.collect_activations(tokenizer, failure_prompts)
    vectors = steerer.task_vectors(acts_ok, acts_bad)   # {layer: (D,)}
    steerer.apply(vectors, alpha=0.8)                    # hooks installed
    ... generate ...                                     # steered behavior
    steerer.remove()                                     # back to baseline
"""
from __future__ import annotations

from typing import Any

import torch


def _make_capture_hook(storage: dict[int, torch.Tensor], layer_idx: int,
                       use_last_token: bool):
    """Return a forward pre-hook that stores the block-input residual."""
    def hook(module, args):
        x = args[0].detach()
        if use_last_token:
            storage[layer_idx] = x[:, -1]
        else:
            storage[layer_idx] = x
        return None
    return hook


class ActivationSteerer:
    """Capture residual-stream activations and inject steering vectors.

    Args:
        model: ConfigurableResearchLLM (has .blocks ModuleList).
        layers: which blocks to instrument (default: all).
        use_last_token: capture only the final token's residual instead of
            the full sequence (cheaper, often enough for task vectors).
    """

    def __init__(self, model, layers: list[int] | None = None,
                 use_last_token: bool = False):
        self.model = model
        self.n_layers = len(model.blocks)
        self.layers = layers if layers is not None else list(range(self.n_layers))
        self.use_last_token = use_last_token
        self._hooks: list[Any] = []
        self._active_vectors: dict[int, torch.Tensor] = {}
        self._alpha: float = 0.0

    # ── capture ────────────────────────────────────────────────────

    def collect_activations(
        self,
        tokenizer,
        prompts: list[str],
        device: str = "cuda",
    ) -> dict[int, torch.Tensor]:
        """Run the model on *prompts* and return mean per-layer activations.

        Returns {layer_idx: (D,) float32 tensor} — mean over prompts and
        tokens (or last token when use_last_token=True).
        """
        storage: dict[int, torch.Tensor] = {}
        hooks = [
            self.model.blocks[i].register_forward_pre_hook(
                _make_capture_hook(storage, i, self.use_last_token))
            for i in self.layers
        ]
        self.model.eval()
        try:
            with torch.inference_mode():
                for p in prompts:
                    input_ids = tokenizer(
                        p, return_tensors="pt").input_ids.to(device)
                    try:
                        self.model(input_ids, use_cache=False)
                    except Exception:
                        # Some models raise on short inputs; skip prompt.
                        continue
        finally:
            for h in hooks:
                h.remove()

        if not storage:
            return {}
        means = {}
        for i in self.layers:
            x = storage.get(i)
            if x is None:
                continue
            # Mean over batch + sequence (or just batch for last-token mode).
            dims = (0, 1) if x.dim() == 3 else (0,)
            means[i] = x.float().mean(dim=dims)
        return means

    # ── task vectors ───────────────────────────────────────────────

    @staticmethod
    def task_vectors(
        positive: dict[int, torch.Tensor],
        negative: dict[int, torch.Tensor],
        normalize: bool = True,
    ) -> dict[int, torch.Tensor]:
        """Direction separating two behaviors: mean(positive) - mean(negative).

        Args:
            positive: activations from successful / desired runs.
            negative: activations from failed / undesired runs.
            normalize: L2-normalize each layer vector (standard practice so
                alpha has a consistent meaning across layers).

        Returns:
            {layer_idx: (D,) tensor}.
        """
        vectors = {}
        for i, v_pos in positive.items():
            v_neg = negative.get(i)
            if v_neg is None or v_pos.shape != v_neg.shape:
                continue
            v = v_pos - v_neg
            if normalize and v.norm() > 0:
                v = v / (v.norm() + 1e-8)
            vectors[i] = v
        return vectors

    # ── injection ─────────────────────────────────────────────────

    def apply(self, vectors: dict[int, torch.Tensor], alpha: float = 1.0) -> None:
        """Install forward pre-hooks that add alpha * vector to each layer's
        residual input. Replaces any previously active steering."""
        self.remove()
        if alpha == 0.0 or not vectors:
            return
        self._alpha = alpha
        self._active_vectors = {i: v.detach() for i, v in vectors.items()}

        def make_hook(vec: torch.Tensor, alpha: float):
            def hook(module, args):
                x = args[0]
                return (x + alpha * vec.to(x.dtype),)
            return hook

        for i, vec in self._active_vectors.items():
            if 0 <= i < self.n_layers:
                self._hooks.append(
                    self.model.blocks[i].register_forward_pre_hook(
                        make_hook(vec, alpha)))

    def remove(self) -> None:
        """Remove all steering hooks (model returns to baseline)."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._active_vectors = {}

    def clear(self) -> None:
        """Alias of remove() for symmetry with the buffer APIs."""
        self.remove()

    @property
    def active(self) -> bool:
        return len(self._hooks) > 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False
