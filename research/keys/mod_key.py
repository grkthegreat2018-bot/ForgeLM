"""Mixture-of-Depths (MoD) key — layer skipping with router init.

MoD (Google, 2024) dynamically skips transformer layers for tokens that
don't need full computation. A router at each layer decides:
- "process" (apply the layer) or "skip" (pass through residual)

Router-Tuning (EMNLP 2025) shows you can fine-tune ONLY the router
(< 0.01% of params) while freezing the backbone.

The key initializes the router to always "process" (never skip).
This makes MoD behave like standard transformer at init.
Fine-tuning then learns which tokens can skip layers.

Key class: TRIVIAL — identity-init router (always process), no data.

Reference: MoD, arxiv 2404.02258; Router-Tuning, EMNLP 2025
"""
import torch
import torch.nn as nn
from research.keys.base import Key, KeyClass, KeyResult


class MoDKey(Key):
    """Mixture-of-Depths key — router init for layer skipping.

    Initializes per-layer router to always "process" (never skip).
    This makes MoD behave like standard transformer at init.
    Fine-tuning the router (< 0.01% params) learns skip patterns.

    Key class: TRIVIAL — identity-init router, no data or training.
    """

    @property
    def name(self) -> str:
        return "mod"

    @property
    def description(self) -> str:
        return "Mixture-of-Depths router (always-process init for layer skipping)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Initialize MoD routers for all layers.

        Args:
            data: {"n_layers": int,
                   "d_model": int,
                   "capacity": float (default 1.0 — process all tokens)}

        Returns:
            {"routers": list of tensors — router_weight[i] is (d_model, 1)}
        """
        try:
            n_layers = data["n_layers"]
            d_model = data["d_model"]
            capacity = data.get("capacity", 1.0)

            routers = []
            for i in range(n_layers):
                # Router: linear projection d_model → 1 (process logit)
                # Init to large positive → always "process" (never skip)
                # The skip logit is implicitly 0, so process logit = large
                # means softmax always chooses "process"
                router = torch.zeros(d_model, 1)
                # Bias toward processing: large positive weight on first dim
                # Actually, use a bias-only approach: router weight = 0,
                # and the "process" bias = large positive
                # Simplified: router produces large positive → always process
                routers.append(router)

            # Process bias (per layer)
            process_bias = torch.full((n_layers,), 10.0)  # large → always process

            return KeyResult(
                success=True,
                weights={"routers": routers, "process_bias": process_bias},
                metadata={
                    "n_layers": n_layers, "d_model": d_model,
                    "capacity": capacity, "init": "always_process",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Identity init — reverse is passthrough."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"reversible": True},
        )


def apply_mod_to_model(model, capacity=1.0):
    """Add MoD routers to a model (in-place).

    Args:
        model: ConfigurableResearchLLM
        capacity: fraction of tokens to process (1.0 = all, 0.5 = half)

    Returns:
        Number of routers added.
    """
    n_layers = len(model.blocks)
    d_model = getattr(model.config, 'd_model', 256)

    key = MoDKey()
    result = key.forward({"n_layers": n_layers, "d_model": d_model, "capacity": capacity})

    if not result.success:
        raise RuntimeError(f"MoD key failed: {result.error}")

    for i, block in enumerate(model.blocks):
        router = nn.Linear(d_model, 1, bias=True)
        router.weight.data = result.weights["routers"][i].t()
        router.bias.data = torch.tensor([result.weights["process_bias"][i].item()])
        block.mod_router = router

    return n_layers


if __name__ == "__main__":
    key = MoDKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"n_layers": 4, "d_model": 256, "capacity": 1.0})
    print(f"Forward: {r.success}")
    print(f"  Routers: {len(r.weights['routers'])}")
    print(f"  Process bias: {r.weights['process_bias'].tolist()}")
    # Verify always-process init
    assert all(b == 10.0 for b in r.weights["process_bias"])
    print("  Always-process init verified ✓")
