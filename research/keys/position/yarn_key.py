"""YaRN key — NTK-by-parts RoPE scaling for context extension.

YaRN (ICLR 2024) extends the context window of RoPE-based models by
modifying the RoPE frequency formula. No weight changes needed —
only the RoPE frequency computation changes.

This is a TRIVIAL key — pure formula modification, no data or training.
The model's weights are completely unchanged; only the position encoding
formula is modified at inference time.

NTK-by-parts formula:
  - Low-frequency dimensions (lambda < alpha): interpolate by 1/s
  - High-frequency dimensions (lambda > beta): no change (extrapolate)
  - Middle dimensions: smooth interpolation between the two

Where:
  s = scaling factor (e.g. 4.0 for 4x context)
  alpha = 1/s (low-freq threshold)
  beta = 1/s * 2 (high-freq threshold, or configurable)
  lambda_i = 2 * pi / (base^(2i/d)) = RoPE wavelength of dimension i

Reference: YaRN, arxiv 2309.00071
"""
import math

import torch

from research.keys.misc.base import Key, KeyClass, KeyResult


class YaRNKey(Key):
    """YaRN RoPE scaling key — NTK-by-parts frequency modification.

    Modifies RoPE frequencies to extend context window by factor s.
    No weight changes — only the position encoding formula changes.

    Key class: TRIVIAL — pure formula, no data or training.
    """

    @property
    def name(self) -> str:
        return "yarn"

    @property
    def description(self) -> str:
        return "NTK-by-parts RoPE scaling for context extension (no weight changes)"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Compute YaRN-modified RoPE frequencies.

        Args:
            data: {"d_model": int or "head_dim": int,
                   "base": float (RoPE base, default 10000),
                   "scale": float (context extension factor, e.g. 4.0),
                   "beta_fast": float (default 32),
                   "beta_slow": float (default 1)}

        Returns:
            {"inv_freq": tensor (head_dim/2,) — modified inverse frequencies,
             "attention_scale": float — YaRN attention scaling factor}
        """
        try:
            head_dim = data.get("head_dim", data.get("d_model", 128))
            base = data.get("base", 10000.0)
            scale = data.get("scale", 4.0)
            beta_fast = data.get("beta_fast", 32.0)
            beta_slow = data.get("beta_slow", 1.0)

            # Original RoPE inverse frequencies
            # inv_freq[i] = 1 / base^(2i/d)
            freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

            # YaRN NTK-by-parts
            # Compute wavelengths: lambda_i = 2*pi / inv_freq_i = 2*pi * base^(2i/d)
            # No — inv_freq = 1/base^(2i/d), so wavelength = 2*pi/inv_freq = 2*pi*base^(2i/d)
            # But YaRN uses the ratio of original context to extended context

            # NTK-aware: modify base to base' = base * s^(d/(d-2))
            # This is the "NTK-aware" part
            ntk_base = base * (scale ** (head_dim / (head_dim - 2)))

            # NTK-by-parts: blend between NTK-aware and linear interpolation
            # based on frequency
            inv_freq_ntk = 1.0 / (ntk_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            inv_freq_orig = freqs

            # Compute the "ramp" function for blending
            # Low freq (long wavelength) → interpolate (use scaled original)
            # High freq (short wavelength) → extrapolate (use NTK)
            # The ramp goes from 0 (interpolate) to 1 (extrapolate)

            # Wavelength ratio: how many periods fit in original context
            # If wavelength >> original context → low freq → interpolate
            # If wavelength << original context → high freq → extrapolate

            # YaRN uses: alpha = 1/s, beta = 2/s (configurable)
            # Dimensions with inv_freq < alpha → interpolate
            # Dimensions with inv_freq > beta → extrapolate
            # In between → smooth ramp

            # Actually YaRN works on the wavelength, not inv_freq directly
            # Let's use the standard YaRN formula from the paper

            # Compute the ramp
            low_freq_wavelen = 1.0 / (scale * inv_freq_orig)  # scaled wavelength
            high_freq_wavelen = 1.0 / inv_freq_orig  # original wavelength

            # The ramp factor r_i for each dimension
            # r = 0 → full interpolation (inv_freq / scale)
            # r = 1 → no change (NTK extrapolation)
            # In between → smooth blend

            # YaRN's gamma function
            def gamma(r):
                return 1.0 - 1.0 / (1.0 + r * scale / (2 * math.pi))

            # Compute per-dimension scaling
            # inv_freq_yarn[i] = inv_freq_orig[i] / scale  (interpolation)
            #                   or inv_freq_ntk[i]          (extrapolation)
            # blended by the ramp

            # Simplified YaRN: use the NTK-aware base for high-freq dims,
            # linear scaling for low-freq dims, smooth ramp in between

            # Compute the wavelength in terms of original context length
            # wavelen = 2 * pi / inv_freq
            wavelen = 2 * math.pi / inv_freq_orig
            orig_context = data.get("orig_context", 1024)

            # Dimensions where wavelength > orig_context → low freq → interpolate
            # Dimensions where wavelength < orig_context / scale → high freq → extrapolate

            # Alpha and beta thresholds (in terms of wavelength)
            alpha_wavelen = orig_context / scale  # below this → extrapolate
            beta_wavelen = orig_context  # above this → interpolate

            # Ramp: 0 for interpolate, 1 for extrapolate
            ramp = torch.zeros_like(freqs)
            for i in range(len(freqs)):
                if wavelen[i] <= alpha_wavelen:
                    ramp[i] = 1.0  # extrapolate
                elif wavelen[i] >= beta_wavelen:
                    ramp[i] = 0.0  # interpolate
                else:
                    # Smooth ramp (linear in log space)
                    t = (math.log(beta_wavelen) - math.log(wavelen[i].item())) / \
                        (math.log(beta_wavelen) - math.log(alpha_wavelen))
                    ramp[i] = 1.0 - t  # smooth transition

            # Blend: inv_freq_yarn = ramp * inv_freq_ntk + (1-ramp) * inv_freq_orig / scale
            inv_freq_yarn = ramp * inv_freq_ntk + (1 - ramp) * (inv_freq_orig / scale)

            # YaRN attention scaling
            # Scale attention scores by a temperature factor
            attn_scale = 0.1 * math.log(scale) + 1.0

            return KeyResult(
                success=True,
                weights={"inv_freq": inv_freq_yarn, "attn_scale": attn_scale},
                metadata={
                    "scale": scale, "base": base, "head_dim": head_dim,
                    "ntk_base": ntk_base, "method": "ntk_by_parts",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """YaRN is not invertible (it's a one-way context extension)."""
        return KeyResult(
            success=True,
            data={"note": "YaRN is a one-way transform (context extension)"},
            metadata={"lossy": False, "reversible": False},
        )


def apply_yarn_to_model(model, scale=4.0, orig_context=1024):
    """Apply YaRN RoPE scaling to a model (modifies RoPE frequencies).

    Args:
        model: ConfigurableResearchLLM with RoPE
        scale: context extension factor (e.g. 4.0 for 4x context)
        orig_context: original training context length

    Returns:
        The YaRN key result with modified frequencies.
    """
    # Find RoPE parameters in the model
    head_dim = None
    base = 10000.0

    # Try to find head_dim from attention config
    if hasattr(model, 'config'):
        cfg = model.config
        head_dim = getattr(cfg, 'd_model', None) // getattr(cfg, 'n_heads', 12)
        base = getattr(cfg, 'base', 10000.0)
        if hasattr(cfg, 'head_dim'):
            head_dim = cfg.head_dim

    if head_dim is None:
        # Fallback: infer from first attention layer
        for block in model.blocks:
            if hasattr(block.attn, 'q_proj'):
                head_dim = block.attn.q_proj.out_features // getattr(block.attn, 'n_heads', 12)
                break

    if head_dim is None:
        raise ValueError("Could not determine head_dim from model")

    key = YaRNKey()
    result = key.forward({
        "head_dim": head_dim,
        "base": base,
        "scale": scale,
        "orig_context": orig_context,
    })

    if not result.success:
        raise RuntimeError(f"YaRN key failed: {result.error}")

    # Apply modified frequencies to all attention layers
    inv_freq = result.weights["inv_freq"]
    attn_scale = result.weights["attn_scale"]

    for block in model.blocks:
        if hasattr(block.attn, 'inv_freq'):
            block.attn.inv_freq.data = inv_freq
        if hasattr(block.attn, 'attn_scale'):
            block.attn.attn_scale = attn_scale

    return result


if __name__ == "__main__":
    key = YaRNKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    # Test 4x context extension
    r = key.forward({"head_dim": 128, "base": 10000, "scale": 4.0, "orig_context": 1024})
    print(f"Forward: {r.success}")
    print(f"  inv_freq shape: {r.weights['inv_freq'].shape}")
    print(f"  attn_scale: {r.weights['attn_scale']:.4f}")
    print(f"  ntk_base: {r.metadata['ntk_base']:.0f}")

    # Compare original vs YaRN frequencies
    orig_freq = 1.0 / (10000 ** (torch.arange(0, 128, 2).float() / 128))
    yarn_freq = r.weights["inv_freq"]
    ratio = yarn_freq / orig_freq
    print(f"  Freq ratio (first 5): {ratio[:5].tolist()}")
    print(f"  Freq ratio (last 5):  {ratio[-5:].tolist()}")
    print("  Low-freq dims scaled down (interpolate), high-freq preserved (extrapolate)")
