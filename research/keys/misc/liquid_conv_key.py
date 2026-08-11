"""Liquid conv key — derive conv kernel from local attention pattern.

The Liquid (LFM2) architecture replaces some attention layers with
double-gated short convolutions. The conv kernel can be derived from
the local attention pattern of the original attention layer.

Key insight: A short causal conv with kernel size k acts like attention
that only looks at the k most recent tokens. If we extract the average
attention weights for the k most recent positions, we can derive a conv
kernel that approximates the local attention behavior.

For a depthwise conv (one filter per channel), the kernel weight for
position offset j is proportional to the average attention weight at
that offset across all calibration tokens.

Key class: PARTIAL — needs calibration data for attention pattern.
The gate and forget_gate linear layers can be initialized from the
attention Q/K projection (approximate).
"""
import torch
import torch.nn as nn

from research.keys.misc.base import Key, KeyClass, KeyResult


class LiquidConvKey(Key):
    """Derive Liquid conv kernel from local attention pattern.

    Converts an attention layer to a double-gated conv block by:
    1. Extracting the average local attention weights (positions t, t-1, ..., t-k+1)
    2. Using them as the depthwise conv kernel
    3. Initializing gate/forget_gate from the attention Q projection (approximate)
    4. Initializing out_proj from the attention O projection

    Key class: PARTIAL — approximate, needs calibration data.
    """

    @property
    def name(self) -> str:
        return "liquid_conv"

    @property
    def description(self) -> str:
        return "Derive conv kernel from local attention pattern (LFM2-style)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """Derive conv + gate weights from attention layer.

        Args:
            data: {
                "attention_weights": tensor (n_heads, seq, seq) — softmaxed attn,
                "q_weight": tensor (d_model, d_model) — attention Q projection,
                "o_weight": tensor (d_model, d_model) — attention O projection,
                "kernel_size": int (default 3),
                "d_model": int,
            }

        Returns:
            {"conv_weight": tensor (d_model, 1, kernel_size),
             "gate_weight": tensor (d_model, d_model),
             "forget_gate_weight": tensor (d_model, d_model),
             "out_proj_weight": tensor (d_model, d_model)}
        """
        try:
            attn_weights = data["attention_weights"]  # (n_heads, seq, seq)
            q_weight = data["q_weight"]  # (d_model, d_model)
            o_weight = data.get("o_weight")  # (d_model, d_model)
            kernel_size = data.get("kernel_size", 3)
            d_model = data["d_model"]
            n_heads, seq_len, _ = attn_weights.shape

            # 1. Extract local attention pattern (last kernel_size positions)
            # For each query position t, get attention to t, t-1, ..., t-k+1
            # Average across heads and positions
            local_weights = torch.zeros(kernel_size)
            count = 0
            for t in range(kernel_size - 1, seq_len):
                for j in range(kernel_size):
                    src = t - j  # position being attended to
                    if src >= 0:
                        # Average across heads
                        local_weights[j] += attn_weights[:, t, src].mean().item()
                        count += 1
            local_weights /= max(count, 1)
            # Normalize so kernel sums to 1 (like attention)
            local_weights = local_weights / local_weights.sum().clamp(min=1e-8)

            # Depthwise conv: each channel gets the same kernel (from attention pattern)
            # Conv1d weight shape: (d_model, 1, kernel_size) for depthwise
            conv_weight = local_weights.unsqueeze(0).unsqueeze(0).expand(d_model, 1, -1).clone()

            # 2. Gate weights: approximate from Q projection
            # gate = sigmoid(W_g @ x) — controls what information passes through
            # Use Q projection (queries = "what am I looking for") as gate
            gate_weight = q_weight.clone()

            # 3. Forget gate: also from Q (with different sign to differentiate)
            # forget_gate = sigmoid(W_f @ x) — controls what to forget
            # Use a random projection near Q (so gates are correlated but not identical)
            forget_gate_weight = q_weight.clone() * 0.5  # weaker forget gate

            # 4. Output projection: from O projection if available
            if o_weight is not None:
                out_proj_weight = o_weight.clone()
            else:
                out_proj_weight = torch.eye(d_model)  # identity fallback

            return KeyResult(
                success=True,
                weights={
                    "conv_weight": conv_weight,
                    "gate_weight": gate_weight,
                    "forget_gate_weight": forget_gate_weight,
                    "out_proj_weight": out_proj_weight,
                    "local_attention_pattern": local_weights,
                },
                metadata={
                    "kernel_size": kernel_size,
                    "d_model": d_model,
                    "method": "local_attention_extraction",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Cannot recover attention from conv (lossy, not invertible)."""
        return KeyResult(
            success=True,
            data={"local_attention_pattern": weights.get("local_attention_pattern")},
            metadata={"approximate": True, "lossy": True},
        )


def init_liquid_conv_from_attention(conv_block, attention_layer, attention_weights,
                                     kernel_size=3):
    """Initialize a DoubleGatedConvBlock from an attention layer (in-place).

    Args:
        conv_block: DoubleGatedConvBlock to initialize
        attention_layer: the attention module (GQA or MLA)
        attention_weights: softmaxed attention weights from calibration (n_heads, seq, seq)
        kernel_size: conv kernel size
    """
    d_model = conv_block.d_model

    # Get Q and O projection weights
    q_weight = None
    o_weight = None
    for name in ["q_proj", "q", "query"]:
        if hasattr(attention_layer, name):
            q_weight = getattr(attention_layer, name).weight.data
            break
    for name in ["o_proj", "o", "out_proj"]:
        if hasattr(attention_layer, name):
            o_weight = getattr(attention_layer, name).weight.data
            break

    if q_weight is None:
        raise ValueError("Could not find Q projection in attention layer")

    key = LiquidConvKey()
    result = key.forward({
        "attention_weights": attention_weights,
        "q_weight": q_weight,
        "o_weight": o_weight,
        "kernel_size": kernel_size,
        "d_model": d_model,
    })
    if not result.success:
        raise RuntimeError(f"Liquid conv key failed: {result.error}")

    w = result.weights

    # Copy conv kernel
    conv_block.conv.conv.weight.data.copy_(w["conv_weight"])

    # Copy gate weights
    conv_block.gate.weight.data.copy_(w["gate_weight"])
    conv_block.forget_gate.weight.data.copy_(w["forget_gate_weight"])

    # Copy output projection
    conv_block.out_proj.weight.data.copy_(w["out_proj_weight"])

    return result


if __name__ == "__main__":
    key = LiquidConvKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Synthetic test: attention that focuses on recent tokens
    d_model = 64
    n_heads = 4
    seq_len = 32
    kernel_size = 3

    # Create attention weights that focus on recent positions (like local attention)
    attn = torch.zeros(n_heads, seq_len, seq_len)
    for t in range(seq_len):
        for j in range(min(kernel_size, t + 1)):
            attn[:, t, t - j] = 1.0 / (j + 1)  # more weight on recent
    # Normalize (softmax-like)
    attn = attn / attn.sum(-1, keepdim=True).clamp(min=1e-8)

    q_weight = torch.randn(d_model, d_model)
    o_weight = torch.randn(d_model, d_model)

    r = key.forward({
        "attention_weights": attn,
        "q_weight": q_weight,
        "o_weight": o_weight,
        "kernel_size": kernel_size,
        "d_model": d_model,
    })
    print(f"Forward: {r.success}")
    print(f"  Conv kernel: {r.weights['conv_weight'][0, 0]}")
    print(f"  Local pattern: {r.weights['local_attention_pattern']}")
    print(f"  Gate shape: {r.weights['gate_weight'].shape}")
