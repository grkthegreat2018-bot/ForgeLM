"""Activation configuration for ForgeEngine.

Encapsulates the 50+ parameters of ``ForgeEngine.activate()`` into a
single dataclass, making the configuration inspectable, serialisable,
and reusable (e.g. for level-2 wake re-activation).

Usage::

    from research.inference.activation import ActivationConfig

    cfg = ActivationConfig(kv_cache="s4r", use_compile=True, warmup=True)
    engine.activate_config(cfg)

    # Backwards-compatible kwargs still work:
    engine.activate(kv_cache="s4r", use_compile=True)

The ``from_kwargs`` classmethod builds an ``ActivationConfig`` from the
same keyword arguments that ``activate()`` accepts, so the old API is
preserved without duplicating the parameter list.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class ActivationConfig:
    """All knobs for ``ForgeEngine.activate()`` in one place.

    Mirrors the parameter list of ``activate()`` exactly — same names,
    same defaults — so it can be used as a drop-in replacement.
    """

    # Core strategies
    kv_cache: str = "paged"
    decoding: str = "standard"
    quantize: str | None = None
    acceleration: str | None = None

    # Innovation parameters
    mrl_keep_ratio: float | None = None
    kv_bits: int = 4
    use_v0_warm: bool = False
    use_progressive_kv: bool = False

    # Runtime / compile
    use_compile: bool = False
    use_triton_conv: bool = False
    use_prefix_cache: bool = False
    use_spec_attn: bool = False
    kv_cache_tokens: int | None = None
    use_chunked_prefill: bool = False

    # Feature-registry flags (all default False)
    use_wavelength_pruning: bool = False
    use_pod_attention: bool = False
    use_adaptive_spec: bool = False
    use_cosa: bool = False
    use_seq_split: bool = False
    use_compact_attn: bool = False
    use_suffix_spec: bool = False
    use_fused_qk_norm_rope_cache: bool = False
    use_block_fusion: bool = False
    use_compile_autotune: bool = False
    use_hybrid_prefill: bool = False
    use_learned_prefix_cache: bool = False
    use_hotprefix: bool = False
    use_mosa: bool = False
    use_triroute: bool = False
    use_breakable_cuda_graph: bool = False
    use_corun: bool = False
    use_foundry: bool = False
    use_fa4: bool = False
    use_kara_kv: bool = False
    use_moment_kv: bool = False
    use_kvpop: bool = False
    use_conf_kv: bool = False
    use_jet_long: bool = False
    use_rope_id: bool = False
    use_lerope: bool = False
    use_lampe: bool = False
    use_peagle: bool = False
    use_lookahead_gate: bool = False
    use_fastserve: bool = False
    use_libra: bool = False
    use_faser: bool = False
    use_kairos: bool = False
    use_unified_radix: bool = False
    use_elbow_moe: bool = False
    use_alloc_moe: bool = False
    use_lda_moe: bool = False
    use_adamx: bool = False
    use_sharq: bool = False
    use_mosaic_quant: bool = False
    use_aoh: bool = False

    # Warmup
    warmup: bool = True

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "ActivationConfig":
        """Build from keyword arguments, ignoring unknown keys.

        This allows ``activate(**kwargs)`` to pass through extra
        parameters (e.g. from ``activate_optimal`` overrides) without
        raising ``TypeError``.
        """
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)

    # ── Derived views ────────────────────────────────────────────────────

    @property
    def feature_flags(self) -> dict[str, bool]:
        """Dict of all ``use_*`` flags plus ``warmup``.

        This is the dict consumed by ``_apply_feature_registry`` and
        ``_activate_compile_runtime``.
        """
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name.startswith("use_") or f.name == "warmup"
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for level-2 wake re-activation)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    # Optimal preset ─────────────────────────────────────────────────────

    @classmethod
    def optimal(cls, **overrides: Any) -> "ActivationConfig":
        """Best combination for max VRAM efficiency + speed on current hw.

        Mirrors ``ForgeEngine.activate_optimal()`` defaults.
        """
        defaults = {
            "kv_cache": "rotorquant",
            "decoding": "standard",
            "quantize": None,
            "acceleration": None,
            "kv_bits": 4,
            "use_compile": True,
            "use_triton_conv": True,
            "use_prefix_cache": True,
            "use_fused_qk_norm_rope_cache": True,
            "use_chunked_prefill": True,
            "use_seq_split": True,
            "warmup": True,
        }
        defaults.update(overrides)
        return cls.from_kwargs(**defaults)
