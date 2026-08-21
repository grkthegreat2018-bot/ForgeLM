"""Factorized Embedding — decompose vocab×d_model into vocab×rank × rank×d_model.

Reduces embedding parameters by (vocab×d_model) / (vocab×rank + rank×d_model).
For vocab=65536, d_model=2048, rank=256:
  Standard:   65536 × 2048 = 134.2M params
  Factorized: 65536 × 256 + 256 × 2048 = 16.8M + 0.5M = 17.3M (7.8x reduction)

Init via SVD of original embedding (lossless at start):
  W = U @ S @ Vh  →  embed = U * sqrt(S),  project = sqrt(S) @ Vh
  Forward: x = project(embed(token_id))  =  U*S*Vh[token_id]  =  W[token_id]

Weight tying: the LM head reuses the factorized embedding via a shared
projection. head(x) = x @ project.T @ embed.weight.T = x @ (U*S*Vh).T

Usage:
    from research.keys.architecture.factorized_embed_key import FactorizedEmbedding
    embed = FactorizedEmbedding(vocab_size, d_model, rank=256)
    # Or init from existing embedding:
    embed = FactorizedEmbedding.from_embedding(original_embedding, rank=256)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FactorizedEmbedding(nn.Module):
    """Two-stage embedding: vocab×rank lookup + rank→d_model projection.

    Args:
        vocab_size: vocabulary size
        d_model: model hidden dimension
        rank: factorization rank (<< d_model)
    """

    def __init__(self, vocab_size: int, d_model: int, rank: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rank = rank
        # Stage 1: compact lookup table (vocab × rank)
        self.embed = nn.Embedding(vocab_size, rank)
        # Stage 2: project rank → d_model (shared across all tokens)
        self.project = nn.Linear(rank, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        # Kaiming init for both stages
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.kaiming_normal_(self.project.weight, nonlinearity="linear")

    @classmethod
    def from_embedding(cls, original: nn.Embedding, rank: int = 256
                       ) -> "FactorizedEmbedding":
        """Initialize from an existing full-rank embedding via SVD (lossless).

        W = U @ diag(S) @ Vh
        embed.weight = U * sqrt(S)   (vocab × rank)
        project.weight = sqrt(S)[:, None] * Vh  (rank × d_model)
        Forward: project(embed(token)) = U*S*Vh[token] = W[token]
        """
        d_model = original.embedding_dim
        vocab_size = original.num_embeddings
        emb = cls(vocab_size, d_model, rank=rank)
        with torch.no_grad():
            W = original.weight.float()  # (vocab, d_model)
            # SVD: W = U @ S @ Vh
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # Truncate to rank
            U_r = U[:, :rank]          # (vocab, rank)
            S_r = S[:rank]             # (rank,)
            Vh_r = Vh[:rank, :]        # (rank, d_model)
            sqrt_S = torch.sqrt(S_r.clamp(min=1e-8))
            # embed = U * sqrt(S)
            emb.embed.weight.copy_((U_r * sqrt_S.unsqueeze(0)).to(original.weight.dtype))
            # project = sqrt(S)[:, None] * Vh  → weight shape (d_model, rank)
            # nn.Linear weight is (out_features, in_features) = (d_model, rank)
            proj = (sqrt_S.unsqueeze(1) * Vh_r).T  # (d_model, rank)
            emb.project.weight.copy_(proj.to(original.weight.dtype))
        return emb

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (...) → output: (..., d_model)"""
        compact = self.embed(token_ids)  # (..., rank)
        return self.project(compact)     # (..., d_model)

    @property
    def weight(self) -> torch.Tensor:
        """Effective full embedding weight (for weight tying compat).

        Returns U @ project.weight.T which equals the full vocab×d_model matrix.
        Used by checkpoint I/O and weight tying logic.
        NOTE: This is a computed property — assigning to it won't work.
        Use from_embedding() for initialization.
        """
        # embed.weight: (vocab, rank), project.weight: (d_model, rank)
        # Full W = embed.weight @ project.weight.T  → (vocab, d_model)
        return self.embed.weight @ self.project.weight.T


class FactorizedLMHead(nn.Module):
    """LM head that reuses factorized embedding weights (weight tying).

    head(x) = x @ project.weight @ embed.weight.T
            = x @ (effective_embedding).T

    This shares the same embed + project parameters as FactorizedEmbedding,
    so no extra params for the head.
    """

    def __init__(self, embed: FactorizedEmbedding):
        super().__init__()
        self.embed_ref = embed  # reference, not a copy
        self.d_model = embed.d_model
        self.vocab_size = embed.vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) → logits: (..., vocab_size)"""
        # project.weight: (d_model, rank) → need x @ project.weight → (..., rank)
        # nn.Linear weight is (out, in) = (d_model, rank), so weight.T = (rank, d_model)
        # x @ weight.T = (..., d_model) @ (d_model, rank) = (..., rank)
        compact = x @ self.embed_ref.project.weight  # (..., d_model) @ (d_model, rank) = (..., rank)
        # embed.weight: (vocab, rank) → need compact @ embed.weight.T = (..., rank) @ (rank, vocab)
        return compact @ self.embed_ref.embed.weight.T  # (..., vocab)

    @property
    def weight(self) -> torch.Tensor:
        """Effective full head weight (for compat with code expecting .weight)."""
        return self.embed_ref.weight  # (vocab, d_model)
