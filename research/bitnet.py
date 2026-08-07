"""BitNet 1.58 — Ternary weight Linear layer for extreme model compression.

Each weight is constrained to {-1, 0, +1} (1.58 bits), trained from scratch
using the straight-through estimator (STE). Activations are quantized to 8-bit
per-token via absmax. This reduces model weights by ~10x vs FP16 while
matching FP16 quality at the same parameter count (per Microsoft's paper).

Usage:
    from research.bitnet import BitLinear
    # Drop-in replacement for nn.Linear:
    layer = BitLinear(1024, 4096, bias=False)
    # In your model, replace nn.Linear with BitLinear for all projections.

Training:
    Weights are quantized to ternary on every forward pass using absmean:
        W_quant = round(W / mean(|W|)) clipped to {-1, 0, +1}
    Gradients flow through the round() via straight-through estimator
    (round() has zero gradient, so we use the identity gradient: dW_quant = dW).

    Activations are quantized to 8-bit per-token:
        a_quant = round(a * 127 / max(|a|)) / 127
    This keeps activations in [-1, 1] with 8-bit precision.

Memory savings (360M model, non-embedding weights):
    FP16:    ~720 MB
    Ternary: ~72 MB  (10x reduction, 1.58 bits/weight packed)
    Unpacked ternary (int8 storage): ~360 MB (5x reduction)

References:
    - "The Era of 1-bit LLMs" (arXiv:2402.17764)
    - BitNet b1.58 2B4T: microsoft/bitnet-b1.58-2B-4T
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """Ternary weight Linear layer (BitNet 1.58).

    Stores full-precision weights for training, quantizes to ternary {-1, 0, +1}
    on every forward pass. At inference, weights can be permanently quantized
    to save memory (call .freeze_ternary()).

    Args:
        in_features, out_features: same as nn.Linear
        bias: if True, adds a bias (usually False for LLMs)
        weight_bits: 1.58 (ternary) — only ternary supported for now
        act_bits: 8 (activation quantization bits)
    """

    def __init__(self, in_features, out_features, bias=False,
                 weight_bits=1.58, act_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.act_bits = act_bits

        # Full-precision weight for training (quantized on forward).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize like nn.Linear.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # If True, weights are permanently ternary (inference mode).
        self._frozen = False

    def _quantize_weight_ternary(self, w):
        """Quantize weights to ternary {-1, 0, +1} using absmean scaling.

        W_quant = clip(round(W / absmean(W)), -1, 1)
        """
        # absmean scaling factor.
        abs_mean = w.abs().mean().clamp(min=1e-8)
        # Scale, round, clip to {-1, 0, +1}.
        w_scaled = w / abs_mean
        w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        return w_quant

    def _quantize_activation(self, x, bits=8):
        """Quantize activations to int8 per-token using absmax.

        a_quant = round(a * (2^(bits-1) - 1) / max(|a|)) / (2^(bits-1) - 1)
        This keeps activations in [-1, 1] with 8-bit precision.
        """
        if not self.training:
            return x  # skip activation quant at inference for speed
        qmax = 2 ** (bits - 1) - 1  # 127 for 8-bit
        # Per-token absmax (along last dim).
        abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x_scaled = x * qmax / abs_max
        x_quant = torch.round(x_scaled) / qmax
        return x_quant

    def forward(self, x):
        # Quantize activations (per-token absmax, 8-bit).
        x = self._quantize_activation(x, self.act_bits)

        if self._frozen:
            # Inference: weight is already ternary int8, just do matmul.
            weight = self.weight  # stored as int8-like float
        else:
            # Training: quantize weight on every forward (STE via detach trick).
            weight = self._quantize_weight_ternary(self.weight)
            # Straight-through estimator: gradient flows through as if identity.
            weight = self.weight + (weight - self.weight).detach()

        # The matmul: with ternary weights, this is just additions/subtractions.
        # In PyTorch we still use F.linear (GPU-optimized matmul), but a custom
        # CUDA kernel could use integer-only ops for 4-6x speedup.
        return F.linear(x, weight, self.bias)

    def freeze_ternary(self):
        """Permanently quantize weights to ternary (for inference).

        After calling this, the weight parameter holds ternary values {-1, 0, +1}
        and no quantization happens on forward (saves compute).
        """
        with torch.no_grad():
            self.weight.copy_(self._quantize_weight_ternary(self.weight))
        self._frozen = True
        self.weight.requires_grad = False

    def unfreeze(self):
        """Reverse of freeze_ternary — allow training again."""
        self._frozen = False
        self.weight.requires_grad = True

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, weight_bits={self.weight_bits}, "
                f"act_bits={self.act_bits}, frozen={self._frozen}")


def convert_model_to_bitnet(model, target_modules=None, skip_embeddings=True):
    """Replace nn.Linear modules with BitLinear in-place.

    Args:
        model: nn.Module to convert
        target_modules: set of attribute name substrings to match.
            If None, targets all Linear except embeddings.
        skip_embeddings: if True, skip layers with 'embed' or 'lm_head' in name.

    Returns:
        count of replaced layers.
    """
    if target_modules is None:
        target_modules = {"q_proj", "k_proj", "v_proj", "out_proj", "kv_down_proj",
                          "kv_up_proj", "k_up_proj", "v_up_proj", "w_gate", "w_up",
                          "w_down", "c_fc", "c_proj"}

    count = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if skip_embeddings and any(s in child_name for s in ("embed", "lm_head", "wte", "wpe")):
                continue
            if any(t in child_name for t in target_modules):
                # Create BitLinear with same shape, copy weights.
                bit_layer = BitLinear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                )
                bit_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None and bit_layer.bias is not None:
                    bit_layer.bias.data.copy_(child.bias.data)
                setattr(module, child_name, bit_layer)
                count += 1
    return count


def pack_ternary_to_int8(weight):
    """Pack ternary {-1, 0, +1} weights into int8 storage.

    Each ternary value needs 1.58 bits, but for simplicity we store as int8
    (one value per byte). A more compact packing could fit 5 values per byte
    (5 * 1.58 = 7.9 bits ≈ 1 byte), giving 5x storage compression.

    Args:
        weight: tensor with values in {-1, 0, 1}

    Returns:
        int8 tensor with same shape.
    """
    return weight.to(torch.int8)


def unpack_int8_to_ternary(packed):
    """Unpack int8 storage back to ternary float tensor."""
    return packed.to(torch.float32)


def freeze_ternary(model):
    """Permanently quantize BitLinear weights to ternary {-1, 0, +1} for inference.

    After training, call this to round all BitLinear weights to exact ternary
    values and freeze them (no gradients). This reduces memory and ensures
    the model uses true ternary weights at inference time.

    Args:
        model: the model with BitLinear layers

    Returns:
        number of layers frozen
    """
    n_frozen = 0
    for module in model.modules():
        if isinstance(module, BitLinear):
            with torch.no_grad():
                # Round to ternary {-1, 0, +1}.
                module.weight.data = module.weight.data.sign().to(torch.int8)
            module.weight.requires_grad_(False)
            n_frozen += 1
    return n_frozen
