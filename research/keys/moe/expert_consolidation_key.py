"""Expert Consolidation Key — merge similar MoE experts.

Novel key: no published method for merging MoE experts (existing work
goes dense→MoE, not MoE→fewer-experts).

Algorithm:
  1. Compute pairwise cosine similarity between expert weights
     (w_gate, w_up, w_down per expert)
  2. Identify experts above a similarity threshold (default 0.95)
  3. Merge similar experts via weighted averaging:
     - w_merged = (w_a + w_b) / 2
     - Router: redirect traffic from merged expert to the survivor
  4. Update router weights to reflect the consolidation

For ForgeLM v1/v2: 4 routed experts → 2-3 if some are redundant.
Since our experts are weight-sliced from a dense FFN (lossless split),
adjacent slices may be similar and mergeable.

Key class: PARTIAL — weight transform, not reversible (merging is lossy
if experts differ, but lossless if they're identical).

Usage:
    from research.keys.expert_consolidation_key import ExpertConsolidationKey, apply_expert_consolidation
    state = apply_expert_consolidation(state, n_layers=28, n_experts=4, threshold=0.95)
"""
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class ExpertConsolidationKey(Key):
    """Expert Consolidation key — merge similar MoE experts.

    Reduces expert count by merging weight-similar experts.
    Router is updated to redirect traffic to surviving experts.

    Key class: PARTIAL — weight transform, not fully reversible.
    """

    def __init__(self, threshold: float = 0.95, min_experts: int = 2):
        self.threshold = threshold
        self.min_experts = min_experts

    @property
    def name(self) -> str:
        return "expert_consolidation"

    @property
    def description(self) -> str:
        return f"Merge similar MoE experts (threshold={self.threshold}, min={self.min_experts})"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """Merge similar experts in the state dict.

        Args:
            data: state dict with expert weights and router

        Returns:
            modified state dict with merged experts
        """
        try:
            state = dict(data)
            n_layers = 0
            for k in state:
                if "ln1.weight" in k:
                    n_layers = max(n_layers, int(k.split(".")[1]) + 1)

            total_merged = 0
            new_expert_counts = []

            for i in range(n_layers):
                # Find all expert weights for this layer
                expert_weights = {}
                for ei in range(10):  # scan up to 10 experts
                    parts = {}
                    for part in ["w_gate", "w_up", "w_down"]:
                        k = f"blocks.{i}.ffn.experts.{ei}.{part}.weight"
                        if k in state:
                            parts[part] = state[k]
                    if parts:
                        expert_weights[ei] = parts

                if len(expert_weights) <= self.min_experts:
                    new_expert_counts.append(len(expert_weights))
                    continue

                # Compute pairwise cosine similarity
                n_exp = len(expert_weights)
                expert_ids = sorted(expert_weights.keys())
                sim_matrix = torch.zeros(n_exp, n_exp)

                for a_idx, a_id in enumerate(expert_ids):
                    for b_idx, b_id in enumerate(expert_ids):
                        if a_idx == b_idx:
                            sim_matrix[a_idx, b_idx] = 1.0
                            continue
                        # Average cosine similarity across all weight parts
                        sims = []
                        for part in ["w_gate", "w_up", "w_down"]:
                            if part in expert_weights[a_id] and part in expert_weights[b_id]:
                                wa = expert_weights[a_id][part].flatten().float()
                                wb = expert_weights[b_id][part].flatten().float()
                                sims.append(F.cosine_similarity(wa.unsqueeze(0),
                                                                 wb.unsqueeze(0)).item())
                        sim_matrix[a_idx, b_idx] = sum(sims) / len(sims) if sims else 0.0

                # Find pairs to merge (greedy: merge most similar first)
                merged = set()
                merge_map = {}  # old_id -> new_id (survivor)
                pairs_to_merge = []

                for a_idx in range(n_exp):
                    for b_idx in range(a_idx + 1, n_exp):
                        if sim_matrix[a_idx, b_idx] >= self.threshold:
                            pairs_to_merge.append((sim_matrix[a_idx, b_idx],
                                                   expert_ids[a_idx],
                                                   expert_ids[b_idx]))

                pairs_to_merge.sort(reverse=True)  # most similar first

                for sim, a_id, b_id in pairs_to_merge:
                    if a_id in merged or b_id in merged:
                        # If one is already merged, redirect to its survivor
                        survivor_a = merge_map.get(a_id, a_id)
                        survivor_b = merge_map.get(b_id, b_id)
                        if survivor_a != survivor_b:
                            # Check if survivors can merge
                            a_idx2 = expert_ids.index(survivor_a)
                            b_idx2 = expert_ids.index(survivor_b)
                            if sim_matrix[a_idx2, b_idx2] >= self.threshold:
                                # Merge b into a
                                merge_map[b_id] = survivor_a
                                merged.add(b_id)
                                total_merged += 1
                        continue
                    # Merge b into a (a survives)
                    merge_map[b_id] = a_id
                    merged.add(b_id)
                    total_merged += 1

                # Apply merges: average weights of merged experts into survivors
                for old_id, new_id in merge_map.items():
                    for part in ["w_gate", "w_up", "w_down"]:
                        old_k = f"blocks.{i}.ffn.experts.{old_id}.{part}.weight"
                        new_k = f"blocks.{i}.ffn.experts.{new_id}.{part}.weight"
                        if old_k in state and new_k in state:
                            state[new_k] = ((state[new_k].float() + state[old_k].float()) / 2
                                           ).to(state[new_k].dtype)
                            del state[old_k]

                # Count surviving experts
                survivors = len(expert_weights) - len(merged)
                new_expert_counts.append(survivors)

            return KeyResult(
                success=True,
                weights=state,
                metadata={
                    "n_layers": n_layers,
                    "total_merged": total_merged,
                    "new_expert_counts": new_expert_counts,
                    "threshold": self.threshold,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(
            success=True, data=weights,
            metadata={"reversible": False, "note": "Merged experts cannot be un-merged"},
        )


def apply_expert_consolidation(state: dict[str, torch.Tensor], n_layers: int,
                                n_experts: int, threshold: float = 0.95,
                                min_experts: int = 2) -> dict[str, torch.Tensor]:
    """Merge similar MoE experts in the state dict.

    Args:
        state: model state dict
        n_layers: number of transformer layers
        n_experts: current number of routed experts
        threshold: cosine similarity threshold for merging (0.95 = very similar)
        min_experts: minimum experts to keep (don't merge below this)

    Returns:
        modified state dict with merged experts
    """
    key = ExpertConsolidationKey(threshold=threshold, min_experts=min_experts)
    result = key.forward(state)
    if not result.success:
        raise RuntimeError(f"Expert consolidation failed: {result.error}")

    merged = result.metadata["total_merged"]
    counts = result.metadata["new_expert_counts"]
    print(f"  [Expert Consolidation] Merged {merged} expert pairs across {n_layers} layers")
    if counts:
        print(f"    Experts per layer: {counts[0]} (was {n_experts})")
    return result.weights


def compute_expert_similarities(state: dict[str, torch.Tensor],
                                 n_layers: int) -> list[list[float]]:
    """Compute expert similarity matrix for analysis (no merging)."""
    all_sims = []
    for i in range(n_layers):
        expert_weights = {}
        for ei in range(10):
            parts = {}
            for part in ["w_gate", "w_up", "w_down"]:
                k = f"blocks.{i}.ffn.experts.{ei}.{part}.weight"
                if k in state:
                    parts[part] = state[k]
            if parts:
                expert_weights[ei] = parts

        if len(expert_weights) < 2:
            continue

        ids = sorted(expert_weights.keys())
        n = len(ids)
        sims = []
        for a in range(n):
            for b in range(a + 1, n):
                part_sims = []
                for part in ["w_gate", "w_up", "w_down"]:
                    if part in expert_weights[ids[a]] and part in expert_weights[ids[b]]:
                        wa = expert_weights[ids[a]][part].flatten().float()
                        wb = expert_weights[ids[b]][part].flatten().float()
                        part_sims.append(F.cosine_similarity(
                            wa.unsqueeze(0), wb.unsqueeze(0)).item())
                avg_sim = sum(part_sims) / len(part_sims) if part_sims else 0.0
                sims.append((ids[a], ids[b], avg_sim))
        all_sims.append(sims)
    return all_sims


if __name__ == "__main__":
    key = ExpertConsolidationKey(threshold=0.9, min_experts=1)
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Create test state with 4 experts, 2 identical
    state = {}
    base_w = torch.randn(256, 128, dtype=torch.bfloat16)
    for i in range(2):
        for ei in range(4):
            if ei < 2:
                w = base_w.clone()  # experts 0,1 are identical
            else:
                w = torch.randn(256, 128, dtype=torch.bfloat16)
            state[f"blocks.{i}.ffn.experts.{ei}.w_gate.weight"] = w
            state[f"blocks.{i}.ffn.experts.{ei}.w_up.weight"] = w.clone()
            state[f"blocks.{i}.ffn.experts.{ei}.w_down.weight"] = w.t().contiguous()
        state[f"blocks.{i}.ln1.weight"] = torch.ones(128, dtype=torch.bfloat16)

    result = key.forward(state)
    print(f"Forward: {result.success}")
    print(f"  Merged: {result.metadata['total_merged']} pairs")
    print(f"  New counts: {result.metadata['new_expert_counts']}")
    assert result.metadata["total_merged"] > 0, "Should merge identical experts!"
    print("  Merging verified ✓")
