"""Wanda pruning key — prune weights by |weight| × ||activation||.

Wanda (ICLR 2024, Princeton) prunes LLM weights without retraining:
- Score each weight by |w_ij| × ||X_j|| (weight magnitude × input activation norm)
- Prune the lowest-scoring weights per output row
- No weight updates needed — the pruned model works as-is

This is a PARTIAL key — needs calibration data for activation norms,
but no training or weight updates.

Reference: Wanda, arxiv 2306.11695
"""
import torch

from research.keys.base import Key, KeyClass, KeyResult


class WandaKey(Key):
    """Wanda pruning key — prune by weight × activation norm.

    Prunes weights with the smallest |w| × ||activation|| scores.
    No retraining or weight updates — the pruned model works directly.

    Key class: PARTIAL — needs calibration data for activation norms.
    """

    @property
    def name(self) -> str:
        return "wanda"

    @property
    def description(self) -> str:
        return "Prune weights by |w| × ||activation|| (no retraining)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """Prune weights using Wanda scoring.

        Args:
            data: {"weight": tensor (out_features, in_features),
                   "activations": tensor (n_samples, in_features),
                   "sparsity": float (fraction to prune, e.g. 0.5),
                   "sparsity_type": "unstructured" or "2:4" or "4:8"}

        Returns:
            {"pruned_weight": tensor (same shape, zeros for pruned),
             "mask": tensor (1=keep, 0=prune),
             "n_pruned": int}
        """
        try:
            weight = data["weight"].clone()
            activations = data["activations"]
            sparsity = data.get("sparsity", 0.5)
            sparsity_type = data.get("sparsity_type", "unstructured")

            out_features, in_features = weight.shape

            # Compute activation norms per input feature
            act_norms = activations.norm(dim=0)  # (in_features,)

            # Wanda score: |w_ij| × ||X_j||
            scores = weight.abs() * act_norms.unsqueeze(0)  # (out, in)

            if sparsity_type == "unstructured":
                # Prune lowest scores per output row
                n_prune = int(in_features * sparsity)
                mask = torch.ones_like(weight)
                for i in range(out_features):
                    threshold = scores[i].kthvalue(n_prune).values
                    mask[i][scores[i] <= threshold] = 0

            elif sparsity_type in ("2:4", "4:8"):
                # N:M structured sparsity: at most N non-zero per M consecutive
                n, m = map(int, sparsity_type.split(":"))
                mask = torch.ones_like(weight)
                for i in range(out_features):
                    for j in range(0, in_features, m):
                        block_end = min(j + m, in_features)
                        block_scores = scores[i, j:block_end]
                        block_size = block_end - j
                        n_keep = min(n, block_size)
                        if n_keep < block_size:
                            threshold = block_scores.kthvalue(block_size - n_keep + 1).values
                            mask[i, j:block_end][block_scores < threshold] = 0
            else:
                return KeyResult(success=False, error=f"Unknown sparsity_type: {sparsity_type}")

            pruned_weight = weight * mask
            n_pruned = (mask == 0).sum().item()

            return KeyResult(
                success=True,
                weights={"pruned_weight": pruned_weight, "mask": mask},
                metadata={
                    "sparsity": sparsity, "sparsity_type": sparsity_type,
                    "n_pruned": n_pruned, "n_total": weight.numel(),
                    "actual_sparsity": n_pruned / weight.numel(),
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Cannot recover pruned weights (lossy)."""
        return KeyResult(
            success=True,
            data={"weight": weights["pruned_weight"], "mask": weights["mask"]},
            metadata={"lossy": True},
        )


def apply_wanda_to_model(model, calibration_activations, sparsity=0.5, sparsity_type="unstructured"):
    """Apply Wanda pruning to all Linear layers in a model.

    Args:
        model: the model to prune
        calibration_activations: dict of {layer_name: activation tensor}
        sparsity: fraction of weights to prune
        sparsity_type: "unstructured", "2:4", or "4:8"

    Returns:
        Total number of weights pruned.
    """
    import torch.nn as nn
    key = WandaKey()
    total_pruned = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in calibration_activations:
            result = key.forward({
                "weight": module.weight.data,
                "activations": calibration_activations[name],
                "sparsity": sparsity,
                "sparsity_type": sparsity_type,
            })
            if result.success:
                module.weight.data = result.weights["pruned_weight"]
                total_pruned += result.metadata["n_pruned"]

    return total_pruned


if __name__ == "__main__":
    key = WandaKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test unstructured pruning
    weight = torch.randn(64, 128)
    activations = torch.randn(100, 128)
    r = key.forward({"weight": weight, "activations": activations,
                     "sparsity": 0.5, "sparsity_type": "unstructured"})
    print(f"Unstructured: {r.success}, pruned {r.metadata['n_pruned']}/{r.metadata['n_total']} "
          f"({r.metadata['actual_sparsity']:.1%})")

    # Test 2:4 structured pruning
    r2 = key.forward({"weight": weight, "activations": activations,
                      "sparsity": 0.5, "sparsity_type": "2:4"})
    print(f"2:4 structured: {r2.success}, pruned {r2.metadata['n_pruned']}/{r2.metadata['n_total']} "
          f"({r2.metadata['actual_sparsity']:.1%})")

    # Verify pruning preserves high-scoring weights
    scores = weight.abs() * activations.norm(dim=0).unsqueeze(0)
    pruned_scores = scores[r2.weights["mask"] == 0]
    kept_scores = scores[r2.weights["mask"] == 1]
    print(f"Pruned avg score: {pruned_scores.mean():.4f}")
    print(f"Kept avg score:   {kept_scores.mean():.4f}")
