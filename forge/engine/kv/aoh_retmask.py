"""AoH: Autonomy-of-Heads + RetMask: retrieval head optimization.

Based on two 2026 papers:
  1. AoH (arXiv 2608.06849): data-free sparse attention from frozen QK geometry.
     Uses effective rank of M_h = W_K^T W_Q to classify heads:
       - Low effective rank → retrieval head (needs global context)
       - High effective rank → streaming head (sink + recent window suffices)
     50% sparsity: 96.5% of full attention, 66% decode latency reduction.
  2. RetMask (ACL 2026 Findings): optimize retrieval heads via contrastive masking.
     +2.28 HELMET at 128K, +70% citation generation, +32% passage re-ranking.
     Gains correlate with retrieval score sparsity.

For our model (6 GQA attention layers, 32 heads):
  - AoH: classify heads as retrieval vs streaming (data-free, from weights)
  - Streaming heads: use sink + recent window (skip global attention)
  - Retrieval heads: full attention (preserve retrieval capability)
  - RetMask: training signal from contrasting normal vs retrieval-masked outputs
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AutonomyOfHeads:
    """AoH: data-free head classification from frozen QK geometry.

    Computes effective rank of M_h = W_K^T W_Q for each head:
      - Low rank → retrieval head (few dominant matching directions)
      - High rank → streaming head (diffuse spectrum, no dominant direction)

    This is computed ONCE from frozen weights (no calibration data needed).
    """

    def __init__(self, n_heads: int, head_dim: int,
                 d_model: int,
                 sparsity_ratio: float = 0.5,
                 sink_size: int = 4,
                 window_size: int = 256):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.sparsity_ratio = sparsity_ratio
        self.sink_size = sink_size
        self.window_size = window_size

        # Head classification (computed from weights)
        self.head_types: list[str] = []  # 'retrieval' or 'streaming'
        self.effective_ranks: list[float] = []

    def classify_heads(self, w_q: torch.Tensor, w_k: torch.Tensor):
        """Classify heads from frozen Q/K weight matrices.

        Args:
            w_q: (n_heads * head_dim, d_model) query projection weight
                 (PyTorch Linear weight layout: out_features × in_features)
            w_k: (n_kv * head_dim, d_model) key projection weight
        """
        # Reshape to per-head weights
        # PyTorch Linear: weight is (out, in) = (n_heads * head_dim, d_model)
        w_q_heads = w_q.view(self.n_heads, self.head_dim, self.d_model)
        n_kv = w_k.shape[0] // self.head_dim
        w_k_heads = w_k.view(n_kv, self.head_dim, self.d_model)

        for h in range(self.n_heads):
            # M_h = W_K_h^T @ W_Q_h (head_dim × head_dim)
            # W_Q_h: (head_dim, d_model), W_K_h: (head_dim, d_model)
            # M_h = W_K_h @ W_Q_h^T → (head_dim, head_dim)
            kv_idx = h % n_kv  # GQA mapping
            M_h = w_k_heads[kv_idx] @ w_q_heads[h].T  # (head_dim, head_dim)

            # Compute effective rank via singular values
            singular_vals = torch.linalg.svdvals(M_h.float())

            # Effective rank = exp(entropy of normalized singular values)
            normalized = singular_vals / singular_vals.sum().clamp(min=1e-8)
            entropy = -(normalized * torch.log(normalized + 1e-8)).sum()
            eff_rank = entropy.exp().item()

            self.effective_ranks.append(eff_rank)

        # Classify: bottom (1-sparsity) fraction = retrieval, top = streaming
        # Low rank → retrieval (keep full attention)
        # High rank → streaming (use sink + window)
        sorted_indices = sorted(range(self.n_heads),
                                key=lambda i: self.effective_ranks[i])
        n_retrieval = int(self.n_heads * (1 - self.sparsity_ratio))

        self.head_types = ['streaming'] * self.n_heads
        for i in sorted_indices[:n_retrieval]:
            self.head_types[i] = 'retrieval'

        print(f"  [AoH] Classified {n_retrieval} retrieval + "
              f"{self.n_heads - n_retrieval} streaming heads")
        print(f"  [AoH] Effective ranks: {[f'{r:.2f}' for r in self.effective_ranks]}")

    def get_attention_mask(self, head_idx: int, seq_len: int,
                           device: str = "cuda") -> torch.Tensor:
        """Get attention mask for a head.

        Retrieval heads: full attention (no mask)
        Streaming heads: sink + recent window only

        Args:
            head_idx: which attention head
            seq_len: sequence length
            device: torch device

        Returns:
            mask: (seq_len,) boolean mask (True = attend)
        """
        if self.head_types[head_idx] == 'retrieval':
            # Full attention
            return torch.ones(seq_len, dtype=torch.bool, device=device)

        # Streaming: sink + recent window
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        # Sink tokens
        mask[:self.sink_size] = True
        # Recent window
        mask[max(0, seq_len - self.window_size):] = True
        return mask

    def stats(self) -> dict:
        return {
            "n_retrieval": sum(1 for t in self.head_types if t == 'retrieval'),
            "n_streaming": sum(1 for t in self.head_types if t == 'streaming'),
            "effective_ranks": self.effective_ranks,
            "sparsity_ratio": self.sparsity_ratio,
            "sink_size": self.sink_size,
            "window_size": self.window_size,
        }


class RetMask:
    """RetMask: retrieval head optimization via contrastive masking.

    Training signal: contrast normal model output with output from
    ablated variant (retrieval heads masked).

    This creates a training signal that strengthens retrieval heads:
      loss = -log P(correct | normal) + λ * log P(correct | masked)

    The contrastive signal teaches the model to rely on retrieval heads
    for long-context information extraction.
    """

    def __init__(self, retrieval_head_indices: list[int],
                 contrast_weight: float = 0.5):
        self.retrieval_indices = set(retrieval_head_indices)
        self.contrast_weight = contrast_weight

    def mask_retrieval_heads(self, attention_output: torch.Tensor,
                              head_dim: int) -> torch.Tensor:
        """Zero out retrieval heads' contributions.

        Args:
            attention_output: (B, n_heads, T, head_dim) per-head attention output
            head_dim: dimension per head

        Returns:
            masked: same shape with retrieval heads zeroed
        """
        masked = attention_output.clone()
        for h in self.retrieval_indices:
            if h < masked.shape[1]:
                masked[:, h, :, :] = 0
        return masked

    def compute_contrastive_loss(self,
                                  normal_logits: torch.Tensor,
                                  masked_logits: torch.Tensor,
                                  target_tokens: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute RetMask contrastive loss.

        Args:
            normal_logits: (B, T, V) logits from normal forward
            masked_logits: (B, T, V) logits from retrieval-masked forward
            target_tokens: (B, T) target token IDs
            mask: (B, T) validity mask

        Returns:
            loss: contrastive loss
        """
        # Normal: standard CE (maximize correct prediction)
        normal_loss = F.cross_entropy(
            normal_logits.view(-1, normal_logits.shape[-1]),
            target_tokens.view(-1),
            reduction='none'
        ).view(target_tokens.shape)

        # Masked: reverse CE (minimize correct prediction → force reliance on retrieval)
        masked_loss = -F.cross_entropy(
            masked_logits.view(-1, masked_logits.shape[-1]),
            target_tokens.view(-1),
            reduction='none'
        ).view(target_tokens.shape)

        # Contrastive: normal should be much better than masked
        loss = normal_loss + self.contrast_weight * masked_loss

        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    def stats(self) -> dict:
        return {
            "n_retrieval_heads": len(self.retrieval_indices),
            "contrast_weight": self.contrast_weight,
            "retrieval_indices": list(self.retrieval_indices),
        }
