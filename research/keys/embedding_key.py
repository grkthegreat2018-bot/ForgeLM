"""Embedding key — token lookup table.

Architecture: y = W[token_id]  (W is [vocab_size, d_model])

Forward key: W[token_ids] = target_vectors  (direct copy)
Reverse key: data = W[token_ids]            (read rows)

Classification: Bi (exact both directions, trivial)
"""
import torch
from .base import Key, KeyClass, KeyResult


class EmbeddingKey(Key):
    @property
    def name(self) -> str:
        return "embedding"

    @property
    def description(self) -> str:
        return "Token embedding lookup. W[token_id] = vector. Direct copy."

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def forward(self, data: dict) -> KeyResult:
        """data -> weights. Expects: 'token_ids', 'target_vectors', 'vocab_size', 'd_model'."""
        try:
            token_ids = data['token_ids']
            target_vectors = data['target_vectors']
            vocab_size = data['vocab_size']
            d_model = data['d_model']

            W = torch.zeros(vocab_size, d_model, dtype=target_vectors.dtype)
            W[token_ids] = target_vectors

            return KeyResult(
                success=True,
                weights={'W': W},
                metadata={'vocab_size': vocab_size, 'd_model': d_model,
                         'n_seen': len(token_ids)}
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """weights -> data. Returns the full embedding table as data."""
        W = weights.get('W')
        if W is None:
            return KeyResult(success=False, error="Missing 'W' in weights")
        # The data IS the weight table — every row is a token's vector
        vocab_size, d_model = W.shape
        token_ids = torch.arange(vocab_size)
        return KeyResult(
            success=True,
            data={'token_ids': token_ids, 'target_vectors': W,
                  'vocab_size': vocab_size, 'd_model': d_model},
            metadata={'n_tokens': vocab_size}
        )
