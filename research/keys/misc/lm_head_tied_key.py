"""LM Head (tied) key — output projection shares weights with embedding.

Architecture: logits = hidden @ embed.weight^T  (head.weight = embed.weight)

Classification: Bi (trivial — direct copy)
"""
import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class LMHeadTiedKey(Key):
    @property
    def name(self) -> str:
        return "lm_head_tied"

    @property
    def description(self) -> str:
        return "Tied LM head. head.weight = embed.weight. Direct copy."

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    def forward(self, data: dict) -> KeyResult:
        """data -> weights. Expects 'embed_weight'."""
        try:
            embed_weight = data['embed_weight']
            head_weight = embed_weight.clone()
            return KeyResult(
                success=True,
                weights={'head_weight': head_weight},
                metadata={'source': 'embed_weight', 'tied': True}
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """weights -> data. Returns embed_weight from head_weight."""
        head_weight = weights.get('head_weight')
        if head_weight is None:
            return KeyResult(success=False, error="Missing 'head_weight' in weights")
        return KeyResult(
            success=True,
            data={'embed_weight': head_weight.clone()},
            metadata={'tied': True}
        )
