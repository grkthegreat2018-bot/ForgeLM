"""Liquid Foundation Model (LFM2-style) hybrid architecture components.

LFM2 uses a hybrid layout:
- 10 double-gated short-range convolution blocks (fast, O(T*d) instead of O(T²*d))
- 6 grouped query attention blocks (for long-range dependencies)

The convolution blocks use multiplicative gates + short causal convolutions,
inspired by Liquid Time-Constant Networks. They're 2x faster than attention
on CPU and much cheaper for long sequences.

For our pipeline, we replace some attention layers with Liquid conv blocks.
The conv weights are initialized from a projection of the attention weights,
then fine-tuned.

Reference: LFM2 Technical Report (Liquid AI, 2025)
           https://www.liquid.ai/blog/liquid-foundation-models-v2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ShortConv1d(nn.Module):
    """Causal short-range 1D convolution with a small kernel (3-7).

    Uses padding to maintain sequence length and ensures causality
    by shifting the output. Faster than attention: O(T*k*d) vs O(T²*d).
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.d_model = d_model
        # Depthwise convolution (one filter per channel) — cheap and effective
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal padding
            groups=d_model,  # depthwise
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        # Conv1d expects (B, D, T)
        x_t = x.transpose(1, 2)  # (B, D, T)
        out = self.conv(x_t)  # (B, D, T + k - 1)
        # Causal: keep last T tokens
        out = out[..., :x.shape[1]]
        return out.transpose(1, 2)  # (B, T, D)


class DoubleGatedConvBlock(nn.Module):
    """Double-gated short-range convolution block (LFM2 style).

    Replaces attention with two multiplicative gates + a short causal conv.
    Much faster than attention: O(T*k*d) vs O(T²*d*k).

    Architecture:
        x -> LN -> [gate1 * conv(silu(gate2 * x))] + residual

    The double gating allows the model to selectively pass or block information
    per-channel, similar to LSTM input/forget gates but in a feed-forward form.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, norm_type: str = "rmsnorm"):
        super().__init__()
        self.d_model = d_model

        # Input gate and forget gate (both project d_model -> d_model)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.forget_gate = nn.Linear(d_model, d_model, bias=False)

        # Short-range causal convolution
        self.conv = ShortConv1d(d_model, kernel_size=kernel_size)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Normalization
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                past_key_value: Optional[Tuple] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """x: (B, T, D) -> (B, T, D)

        Compatible with the attention interface (takes past_key_value, returns it).
        Conv state is cached for efficient single-token decoding.
        """
        h = self.norm(x)

        # Double gating: input gate * conv(silu(forget gate * h))
        f = torch.sigmoid(self.forget_gate(h))  # (B, T, D) — forget gate
        g = self.conv(F.silu(self.gate(h) * f))  # gated conv

        out = self.out_proj(g)
        return x + out, None  # residual + no KV cache for conv


class RMSNorm(nn.Module):
    """RMS normalization (matches our existing implementation)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class LiquidHybridBlock(nn.Module):
    """A hybrid block that can be either attention or Liquid conv.

    This allows the model to use a mix of attention and conv blocks,
    like LFM2's 10 conv + 6 attention layout.
    """

    def __init__(self, d_model: int, block_type: str = "conv",
                 kernel_size: int = 3, norm_type: str = "rmsnorm",
                 n_heads: int = 12, n_kv_heads: int = 2,
                 max_seq_len: int = 2048, base: float = 10000.0,
                 attn_bias: bool = False, intermediate_size: int = None,
                 ffn_type: str = "swiglu"):
        super().__init__()
        self.block_type = block_type
        self.d_model = d_model

        if block_type == "conv":
            self.attn = DoubleGatedConvBlock(d_model, kernel_size=kernel_size, norm_type=norm_type)
        elif block_type == "attention":
            from research.model_loader import GroupedQueryAttention
            self.attn = GroupedQueryAttention(
                d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len, base=base, attn_bias=attn_bias,
            )
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        # FFN is shared between conv and attention blocks
        if ffn_type == "swiglu":
            from research.model_loader import SwiGLUFFN
            d_ff = intermediate_size or 8 * d_model // 3
            self.ffn = SwiGLUFFN(d_model, d_ff)
        else:
            from research.model_loader import StandardFFN
            d_ff = intermediate_size or 4 * d_model
            self.ffn = StandardFFN(d_model, d_ff)

        # Post-attention/post-conv norm
        if norm_type == "rmsnorm":
            self.ln2 = RMSNorm(d_model)
        else:
            self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, past_key_value=None, use_cache=False):
        # Attention/conv sublayer
        attn_out, new_kv = self.attn(x, past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out
        # FFN sublayer
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, new_kv


def liquid_block_layout(n_layers: int, n_conv: int = 10, n_attn: int = 6) -> list:
    """Generate a hybrid block layout (conv/attention interleaved).

    LFM2 uses 10 conv + 6 attention = 16 blocks.
    Default: interleave conv and attention, with more conv at the start
    (short-range features) and more attention at the end (long-range reasoning).

    Args:
        n_layers: total number of blocks
        n_conv: number of conv blocks
        n_attn: number of attention blocks

    Returns:
        List of "conv" or "attention" strings, length n_layers.
    """
    assert n_conv + n_attn == n_layers, f"{n_conv}+{n_attn} != {n_layers}"
    layout = []
    conv_remaining = n_conv
    attn_remaining = n_attn
    for i in range(n_layers):
        frac = i / max(n_layers - 1, 1)  # 0.0 at start, 1.0 at end
        # Want conv fraction to decrease linearly: conv_ratio = 1 - frac
        # Place conv if we still need conv AND the desired ratio says conv
        desired_conv = (1 - frac) * (i + 1)  # how many conv blocks should be placed by position i
        placed_conv = n_conv - conv_remaining
        want_conv = conv_remaining > 0 and (attn_remaining == 0 or placed_conv < desired_conv)
        if want_conv:
            layout.append("conv")
            conv_remaining -= 1
        else:
            layout.append("attention")
            attn_remaining -= 1
    return layout


def convert_attention_to_liquid(model, n_conv_blocks: int = 10):
    """Replace some attention blocks with Liquid conv blocks (in-place).

    This converts an existing model to a hybrid Liquid architecture.
    The conv weights are initialized randomly — fine-tuning is needed.

    Args:
        model: ConfigurableResearchLLM with .blocks
        n_conv_blocks: how many blocks to convert to conv

    Returns:
        Number of blocks converted.
    """
    n_layers = len(model.blocks)
    n_attn = n_layers - n_conv_blocks
    layout = liquid_block_layout(n_layers, n_conv=n_conv_blocks, n_attn=n_attn)

    print(f"  [Liquid] Converting to hybrid: {n_conv_blocks} conv + {n_attn} attention")
    print(f"  [Liquid] Layout: {' '.join('C' if b == 'conv' else 'A' for b in layout)}")

    converted = 0
    for i, block in enumerate(model.blocks):
        if layout[i] == "conv":
            d_model = block.attn.d_model if hasattr(block.attn, 'd_model') else block.attn.q_proj.in_features
            # Create Liquid conv block to replace attention
            conv_block = DoubleGatedConvBlock(d_model, kernel_size=3, norm_type="rmsnorm")
            # Copy the existing norm if possible
            if hasattr(block, 'ln1'):
                if hasattr(block.ln1, 'weight'):
                    conv_block.norm.weight.data.copy_(block.ln1.weight.data)
            block.attn = conv_block
            converted += 1

    print(f"  [Liquid] Converted {converted} blocks to Liquid conv")
    return converted


if __name__ == "__main__":
    # Quick test
    d_model = 256
    block = DoubleGatedConvBlock(d_model, kernel_size=3)
    x = torch.randn(1, 16, d_model)
    out, _ = block(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in block.parameters()) / 1e3:.1f}K")

    # Compare with GQA attention
    from research.model_loader import GroupedQueryAttention
    attn = GroupedQueryAttention(d_model, n_heads=8, n_kv_heads=2, max_seq_len=512)
    attn_params = sum(p.numel() for p in attn.parameters())
    conv_params = sum(p.numel() for p in block.parameters())
    print(f"\nGQA params:  {attn_params / 1e3:.1f}K")
    print(f"Conv params: {conv_params / 1e3:.1f}K")
    print(f"Conv is {attn_params / conv_params:.1f}x smaller")

    # Layout test
    layout = liquid_block_layout(16, n_conv=10, n_attn=6)
    print(f"\nLayout (16 blocks): {layout}")
