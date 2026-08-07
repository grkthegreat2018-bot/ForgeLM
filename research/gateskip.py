"""GateSkip — Token-wise layer skipping via lightweight sigmoid gates.

Each transformer block gets a sigmoid gate on its residual stream:
    g = sigmoid(W_g @ x)   # gate score per token, per layer
    out = x + g * block(x)  # skip block if g ≈ 0

This saves up to 15% compute (simple tokens skip layers) while retaining
>90% accuracy. The gate is a single nn.Linear(d_model, 1) — minimal overhead
(~1.7% parameter increase).

Training: gates start biased toward 1.0 (always execute) and gradually learn
to skip easy tokens. Differentiable, compatible with quantization/pruning.

Usage:
    from research.gateskip import GateSkipBlock
    # Wrap your existing transformer block:
    block = GateSkipBlock(existing_block, d_model=1024)

References:
    - GateSkip: https://doi.org/10.48550/arxiv.2510.13876
    - Dr.LLM: dynamic layer skip/execute/repeat
    - TSA: Token-Selective Attention (1.7% param overhead)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GateSkipBlock(nn.Module):
    """Wraps a transformer block with a learned skip gate.

    The gate is a single linear layer (d_model -> 1) + sigmoid.
    On forward, if gate value < threshold, the block is skipped (residual only).

    Args:
        block: the transformer block to wrap (e.g. ModularBlock)
        d_model: model dimension
        skip_threshold: gate values below this skip the block (inference only).
            During training, all blocks execute (gate scales the output).
        init_bias: initial bias for the gate (positive = start executing).
            Default 2.0 → sigmoid(2.0) ≈ 0.88 (mostly execute at start).
    """

    def __init__(self, block, d_model, skip_threshold=0.1, init_bias=2.0):
        super().__init__()
        self.block = block
        self.gate = nn.Linear(d_model, 1, bias=True)

        # Initialize gate to mostly-execute (positive bias).
        nn.init.zeros_(self.gate.weight)
        self.gate.bias.data.fill_(init_bias)

        self.skip_threshold = skip_threshold
        self.d_model = d_model

        # Track skip statistics for analysis.
        self.register_buffer("_skip_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_total_count", torch.tensor(0, dtype=torch.long))

    def forward(self, x, **kwargs):
        # Gate: per-token score, shape (B, T, 1).
        gate_score = torch.sigmoid(self.gate(x))  # (B, T, 1)

        if self.training:
            # Training: scale block output by gate (differentiable).
            block_out = self.block(x, **kwargs)
            # Handle tuple returns (some blocks return (out, kv_cache)).
            if isinstance(block_out, tuple):
                block_out, *rest = block_out
                return x + gate_score * block_out, *rest
            return x + gate_score * block_out
        else:
            # Inference: skip block entirely if gate < threshold for all tokens.
            # Per-token decision: if any token needs the block, compute it.
            # For batch efficiency, compute block if mean gate > threshold.
            mean_gate = gate_score.mean()
            if mean_gate < self.skip_threshold:
                # Skip: return input unchanged (with None kv_cache if expected).
                self._skip_count += 1
                self._total_count += 1
                # Check if block expects to return cache.
                if hasattr(self.block, "forward"):
                    # Return input + None cache to match signature.
                    try:
                        return x, None
                    except Exception:
                        return x
            else:
                self._total_count += 1
                block_out = self.block(x, **kwargs)
                if isinstance(block_out, tuple):
                    block_out, *rest = block_out
                    return x + gate_score * block_out, *rest
                return x + gate_score * block_out

    def skip_rate(self):
        """Return the fraction of times this block was skipped (inference)."""
        if self._total_count == 0:
            return 0.0
        return (self._skip_count.float() / self._total_count.float()).item()

    def reset_stats(self):
        self._skip_count.zero_()
        self._total_count.zero_()


def add_gateskip_to_model(model, d_model, skip_threshold=0.1):
    """Wrap each transformer block in model with GateSkipBlock.

    Assumes model has a list of blocks (model.blocks or model.transformer.h).

    Args:
        model: the LLM model
        d_model: model dimension
        skip_threshold: inference skip threshold

    Returns:
        count of wrapped blocks.
    """
    # Find the block list (handle different model architectures).
    blocks = None
    block_attr = None
    for attr in ("blocks", "h", "layers", "transformer_blocks"):
        if hasattr(model, attr):
            blocks = getattr(model, attr)
            block_attr = attr
            break

    if blocks is None:
        # Search one level deep.
        for name, module in model.named_children():
            for attr in ("blocks", "h", "layers"):
                if hasattr(module, attr):
                    blocks = getattr(module, attr)
                    block_attr = f"{name}.{attr}"
                    break
            if blocks is not None:
                break

    if blocks is None:
        raise ValueError("Could not find block list in model. "
                         "Expected model.blocks, model.h, or model.layers.")

    count = 0
    for i in range(len(blocks)):
        block = blocks[i]
        if isinstance(block, GateSkipBlock):
            continue  # already wrapped
        blocks[i] = GateSkipBlock(block, d_model=d_model, skip_threshold=skip_threshold)
        count += 1

    return count


def get_skip_rates(model):
    """Return per-layer skip rates for analysis."""
    rates = {}
    for name, module in model.named_modules():
        if isinstance(module, GateSkipBlock):
            rates[name] = module.skip_rate()
    return rates
