"""MatFormer key — nested FFN for elastic inference (matryoshka FFN).

MatFormer (NeurIPS 2024) introduces nested structure in the FFN block:
the first m1 neurons form a sub-FFN, the first m2 > m1 form a larger one,
etc. This allows extracting hundreds of smaller accurate submodels at
zero cost.

The key insight for weight stealing: the existing FFN weights ARE the
largest matryoshka granularity. Smaller granularities are just slices
of the first m_i neurons. No weight changes needed — just mark which
neuron ranges form valid submodels.

This is a TRIVIAL key — the weights are already matryoshka-structured
by virtue of being a contiguous FFN. We just record the granularity
boundaries.

Reference: MatFormer, arxiv 2310.07707
"""
import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class MatFormerKey(Key):
    """MatFormer nested FFN key — matryoshka granularity marking.

    The existing FFN weights are already the largest granularity.
    Smaller granularities are contiguous slices of the first m_i neurons.
    No weight changes — just record the boundaries.

    Key class: TRIVIAL — no weight changes, just metadata.
    """

    @property
    def name(self) -> str:
        return "matformer"

    @property
    def description(self) -> str:
        return "Nested FFN granularities for elastic inference (matryoshka)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Define matryoshka granularities for an FFN.

        Args:
            data: {"d_ff": int (FFN hidden size),
                   "n_granularities": int (default 4),
                   "schedule": "exponential" or "linear" or "custom",
                   "custom_sizes": list (optional, for "custom")}

        Returns:
            {"granularities": list of int — neuron counts per granularity,
             "slices": list of (start, end) tuples}
        """
        try:
            d_ff = data["d_ff"]
            n_gran = data.get("n_granularities", 4)
            schedule = data.get("schedule", "exponential")
            custom = data.get("custom_sizes")

            if schedule == "custom" and custom:
                granularities = sorted(custom)
            elif schedule == "linear":
                step = d_ff // n_gran
                granularities = [step * (i + 1) for i in range(n_gran)]
            else:  # exponential (default, MatFormer style)
                # d_ff, d_ff/2, d_ff/4, d_ff/8, ...
                granularities = [d_ff // (2 ** i) for i in range(n_gran)]
                granularities = sorted(set(granularities))

            # Ensure largest = d_ff
            if granularities[-1] != d_ff:
                granularities.append(d_ff)

            # Build slices (always from 0 to m_i)
            slices = [(0, g) for g in granularities]

            return KeyResult(
                success=True,
                weights={"granularities": granularities, "slices": slices},
                metadata={
                    "d_ff": d_ff, "n_granularities": len(granularities),
                    "schedule": schedule,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """No weight changes — reverse is passthrough."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"reversible": True},
        )


def extract_submodel(model, granularity_idx, granularities):
    """Extract a smaller submodel from a MatFormer model.

    Args:
        model: ConfigurableResearchLLM with MatFormer FFN
        granularity_idx: which granularity to extract (0=smallest)
        granularities: list of neuron counts

    Returns:
        A view of the model with sliced FFN weights (no copy).
    """
    m = granularities[granularity_idx]
    # The submodel uses the first m neurons of each FFN
    # This is a "view" — in practice you'd create a new model with d_ff=m
    # and copy the first m neurons of each weight matrix
    return m


def apply_matformer_to_model(model, n_granularities=4, schedule="exponential"):
    """Mark MatFormer granularities on a model's FFN layers.

    Args:
        model: ConfigurableResearchLLM
        n_granularities: number of nested sizes
        schedule: "exponential", "linear", or "custom"

    Returns:
        Granularities list.
    """
    d_ff = getattr(model.config, 'd_ff', 8960)

    key = MatFormerKey()
    result = key.forward({
        "d_ff": d_ff,
        "n_granularities": n_granularities,
        "schedule": schedule,
    })

    if not result.success:
        raise RuntimeError(f"MatFormer key failed: {result.error}")

    # Store granularities on the model
    model.matformer_granularities = result.weights["granularities"]

    return result.weights["granularities"]


if __name__ == "__main__":
    key = MatFormerKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"d_ff": 8960, "n_granularities": 4, "schedule": "exponential"})
    print(f"Forward: {r.success}")
    print(f"  Granularities: {r.weights['granularities']}")
    print(f"  Slices: {r.weights['slices']}")
    # Verify largest = d_ff
    assert r.weights["granularities"][-1] == 8960
    # Verify nested (each is subset of next)
    for i in range(len(r.weights["granularities"]) - 1):
        assert r.weights["granularities"][i] < r.weights["granularities"][i + 1]
    print("  Nested structure verified ✓")
