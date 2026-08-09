"""Sliding window attention key — alternating local/global attention pattern.

GPT-OSS and Mistral use alternating attention patterns:
- Even layers: sliding window (local attention, window=W)
- Odd layers: full attention (global context)

This reduces attention complexity from O(N²) to O(N·W) for half the layers.
No weight changes needed — only the attention mask changes.

This is a TRIVIAL key — pure mask modification, no data or training.

Reference: GPT-OSS, Mistral 7B
"""
import torch

from research.keys.base import Key, KeyClass, KeyResult


class SlidingWindowKey(Key):
    """Alternating sliding window / full attention key.

    Even layers use sliding window (local), odd layers use full attention.
    No weight changes — only the attention mask pattern.

    Key class: TRIVIAL — fixed mask pattern, no data or training.
    """

    @property
    def name(self) -> str:
        return "sliding_window"

    @property
    def description(self) -> str:
        return "Alternating sliding window / full attention (GPT-OSS style)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Compute attention masks for all layers.

        Args:
            data: {"n_layers": int,
                   "seq_len": int,
                   "window_size": int (default 128),
                   "pattern": "alternating" or "all_sliding" or "all_full"}

        Returns:
            {"masks": list of tensors (seq_len, seq_len) bool,
             "layer_types": list of str ("sliding" or "full")}
        """
        try:
            n_layers = data["n_layers"]
            seq_len = data["seq_len"]
            window = data.get("window_size", 128)
            pattern = data.get("pattern", "alternating")

            # Causal mask base
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

            masks = []
            layer_types = []

            for i in range(n_layers):
                if pattern == "alternating":
                    use_sliding = (i % 2 == 0)  # even = sliding
                elif pattern == "all_sliding":
                    use_sliding = True
                else:
                    use_sliding = False

                if use_sliding and window > 0:
                    # Sliding window: only attend to last W tokens
                    mask = causal.clone()
                    for j in range(seq_len):
                        start = max(0, j - window + 1)
                        mask[j, :start] = False
                    layer_types.append("sliding")
                else:
                    mask = causal
                    layer_types.append("full")

                masks.append(mask)

            return KeyResult(
                success=True,
                weights={"masks": masks, "layer_types": layer_types},
                metadata={
                    "n_layers": n_layers, "seq_len": seq_len,
                    "window_size": window, "pattern": pattern,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Mask-only — no weights to reverse."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"runtime_only": True},
        )


def apply_sliding_window_to_model(model, window_size=128, pattern="alternating"):
    """Configure sliding window attention on a model.

    Args:
        model: ConfigurableResearchLLM
        window_size: sliding window size
        pattern: "alternating", "all_sliding", or "all_full"

    Returns:
        List of layer types.
    """
    n_layers = len(model.blocks)

    key = SlidingWindowKey()
    result = key.forward({
        "n_layers": n_layers,
        "seq_len": getattr(model.config, 'max_seq_len', 2048),
        "window_size": window_size,
        "pattern": pattern,
    })

    if not result.success:
        raise RuntimeError(f"Sliding window key failed: {result.error}")

    for i, block in enumerate(model.blocks):
        block.attention_type = result.weights["layer_types"][i]
        if result.weights["layer_types"][i] == "sliding":
            block.sliding_window = window_size
        else:
            block.sliding_window = 0

    return result.weights["layer_types"]


if __name__ == "__main__":
    key = SlidingWindowKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"n_layers": 6, "seq_len": 32, "window_size": 8, "pattern": "alternating"})
    print(f"Forward: {r.success}")
    print(f"  Layer types: {r.weights['layer_types']}")
    # Verify alternating
    assert r.weights["layer_types"] == ["sliding", "full", "sliding", "full", "sliding", "full"]
    # Verify sliding window mask
    sliding_mask = r.weights["masks"][0]
    # Token 10 should only see tokens 3-10 (window=8)
    assert sliding_mask[10, 2] == False  # outside window
    assert sliding_mask[10, 3] == True   # inside window
    assert sliding_mask[10, 10] == True  # self
    print("  Alternating pattern verified ✓")
    print("  Sliding window mask verified ✓")
