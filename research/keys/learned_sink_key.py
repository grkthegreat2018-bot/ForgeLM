"""GPT-OSS learned attention sink key — per-head bias logits.

GPT-OSS (OpenAI, 2025) uses learned attention sinks: instead of pinning
the first few tokens as attention sinks (StreamingLLM style), it adds
a per-head bias logit to the attention scores. This achieves the same
stabilization effect without modifying the input sequence.

The sink is a scalar per head: attention_score[i] += sink[h]
It's initialized to a small positive value (e.g. 1.0) so that there's
always a "virtual sink" that absorbs excess attention.

This is a TRIVIAL key — fixed initialization, no data or training.
The sink values can be fine-tuned but work well with simple init.

Reference: GPT-OSS, openai/gpt-oss GitHub
"""
import torch
import torch.nn as nn
from research.keys.base import Key, KeyClass, KeyResult


class LearnedSinkKey(Key):
    """GPT-OSS learned attention sink key — per-head bias init.

    Initializes per-head sink bias logits. These are added to attention
    scores before softmax, creating a "virtual sink" that stabilizes
    attention without pinning input tokens.

    Key class: TRIVIAL — fixed init, no data or training.
    """

    @property
    def name(self) -> str:
        return "learned_sink"

    @property
    def description(self) -> str:
        return "Per-head attention sink bias (GPT-OSS style)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Initialize learned sink biases.

        Args:
            data: {"n_heads": int,
                   "init_value": float (default 1.0),
                   "init_method": "constant" or "random" or "zero"}

        Returns:
            {"sinks": tensor (n_heads,) — per-head sink bias logits}
        """
        try:
            n_heads = data["n_heads"]
            init_value = data.get("init_value", 1.0)
            init_method = data.get("init_method", "constant")

            if init_method == "constant":
                sinks = torch.full((n_heads,), init_value, dtype=torch.float32)
            elif init_method == "random":
                sinks = torch.randn(n_heads) * 0.1 + init_value
            elif init_method == "zero":
                sinks = torch.zeros(n_heads)
            else:
                return KeyResult(success=False, error=f"Unknown init_method: {init_method}")

            return KeyResult(
                success=True,
                weights={"sinks": sinks},
                metadata={
                    "n_heads": n_heads, "init_method": init_method,
                    "init_value": init_value,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Sinks are additive bias — reverse removes them."""
        return KeyResult(
            success=True,
            data={"sinks": weights["sinks"]},
            metadata={"reversible": True},
        )


def apply_learned_sinks_to_model(model, init_value=1.0, init_method="constant"):
    """Add learned attention sinks to all attention layers (in-place).

    Args:
        model: ConfigurableResearchLLM
        init_value: initial sink bias value
        init_method: "constant", "random", or "zero"

    Returns:
        Number of layers modified.
    """
    modified = 0
    for block in model.blocks:
        attn = block.attn
        n_heads = getattr(attn, 'n_heads', 12)

        key = LearnedSinkKey()
        result = key.forward({
            "n_heads": n_heads,
            "init_value": init_value,
            "init_method": init_method,
        })

        if result.success:
            attn.sinks = nn.Parameter(result.weights["sinks"].clone())
            modified += 1

    return modified


if __name__ == "__main__":
    key = LearnedSinkKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"n_heads": 8, "init_value": 1.0, "init_method": "constant"})
    print(f"Forward: {r.success}")
    print(f"  Sinks: {r.weights['sinks'].tolist()}")
    assert all(s == 1.0 for s in r.weights["sinks"])
    print("  All sinks = 1.0 ✓")

    # Random init
    r2 = key.forward({"n_heads": 8, "init_value": 1.0, "init_method": "random"})
    print(f"  Random sinks: {[f'{s:.3f}' for s in r2.weights['sinks'].tolist()]}")
