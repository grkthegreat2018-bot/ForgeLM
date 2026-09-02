"""Forge Inference Engine — unified runtime backend for ForgeAI models.

Pluggable strategy architecture with auto-detection and auto-activation:
  - KV cache: standard, paged, rotorquant, hadamard_int4, compressed,
    streaming, snapkv, snapkv_4bit, paged_eviction, xquant, cpu_offload,
    s4r (15x compression, default), hqe_kv, 2bit
  - Decoding: standard, speculative, medusa, dspark, eagle3, mtp_selfspec
  - Quantization: none, int8, int4, fp8, w8a8, nvfp4, BitNet ternary
    (auto-selected based on VRAM + GPU capability)
  - Acceleration: none, cuda_graph, airllm_streaming, megakernel, flex_decoding
  - Innovations: MRL-AdaptiveContext, QuaRot-KV, V0-WarmStart, ProgressiveKV
  - 42 feature-registry flags for attention, prefill, KV, scheduling, MoE, etc.

Auto-detects KeyStack features from checkpoint metadata and auto-activates
optimal strategies (auto_activate=True by default). Picks the highest
quantization that fits VRAM (nvfp4 on Blackwell, w8a8 default, int4 when
tight). Uses RotorQuant 4-bit KV cache by default (Givens rotation +
Lloyd-Max quantization, ~8x compression, 0.94% error, deferred quantization
for zero error compounding during prefill).

Loading strategies (auto-selected by VRAM capacity):
  1. Pre-quantized BitNet → int8 direct load (4x VRAM cut)
  2. Fits in VRAM → fast meta-init + background thread weight load
  3. Hybrid offload → conv on CPU, attention on GPU (LFM2.5 hybrid arch)
  4. Too large → AirLLM layer-streaming with CPU RAM shard caching

Usage:
    from research.inference.forge_engine import ForgeEngine

    # Auto-activates optimal strategies (S4R KV, torch.compile, prefix cache, etc.)
    engine = ForgeEngine.from_checkpoint(
        checkpoint="research/checkpoints/ForgeLM_V2_Light.safetensors",
        config_name="forgelm_v2_light",
        tokenizer_path="research/checkpoints/lfm25_tokenizer",
    )
    # Strategies auto-activated — just generate:
    output = engine.generate("def fibonacci(n):", max_new_tokens=50)

    # Or manually control activation:
    engine = ForgeEngine.from_checkpoint(..., auto_activate=False)
    engine.activate_optimal(kv_cache="hadamard_int4", decoding="mtp_selfspec")

    # Streaming, raw control, benchmarking, diagnostics:
    for chunk in engine.generate_stream("Hello", max_new_tokens=100):
        print(chunk, end="", flush=True)
    engine.benchmark("test prompt", max_new_tokens=50)
    engine.bottleneck()  # per-layer timing
    engine.diagnose()    # full health report

    # Sleep/wake for VRAM management:
    engine.sleep(level=1)  # offload to CPU
    engine.wake()          # restore to GPU

    # Context manager (auto-sleeps on exit):
    with ForgeEngine.from_checkpoint(...) as engine:
        engine.generate("...")
"""
import os
import time
import json
import threading
from pathlib import Path
from collections.abc import Iterator

import torch
import torch.nn.functional as F

_DEFAULT_CPU_MEMORY_BYTES = 32 * 1024**3
_DEFAULT_EOS_TOKEN_IDS = frozenset({7, 151643, 151645})

# Module-level caches to avoid repeated disk I/O for checkpoint metadata and sizes.
# Keyed by (path, mtime) so stale entries are invalidated when the file changes.
# Bounded with LRU eviction to prevent unbounded growth in long-running servers.
# Thread-safe: guarded by _ckpt_cache_lock (forge_server runs generate() from
# multiple worker threads via SessionManager/BatchQueue).
from collections import OrderedDict as _OrderedDict
_CKPT_CACHE_MAX = 64
_checkpoint_metadata_cache: _OrderedDict[tuple[str, float], dict] = _OrderedDict()
_checkpoint_size_cache: _OrderedDict[str, int] = _OrderedDict()
_ckpt_cache_lock = threading.Lock()

from research.inference.activation import ActivationConfig
from research.inference.airllm_streamer import AirLLMStreamer
from research.inference.decoding import DecodingStrategy, StandardDecoding, build_decoding
from research.inference.diagnostics import (
    EngineProfiler,
    EventLog,
    OutputHistory,
    build_health_report,
)
from research.inference.feature_registry import _FEATURE_REGISTRY
from research.inference.innovations import (
    MRLAdaptiveContext,
    ProgressiveKV,
    QuaRotKV,
    V0WarmStart,
)
from research.inference.kv_backend import KVCacheStrategy, build_kv_cache
from research.inference.prefix_cache import (
    LRUPrefixCache,
    ChunkedPrefixCache,
    cache_prompt_prefix as _cache_prompt_prefix,
    generate_from_prefix_cache as _generate_from_prefix_cache,
)
from research.inference.kv.cacheblend import CacheBlend
from research.inference.crash_recovery import CrashRecoveryManager
from research.inference.errors import (
    ActivationError,
    CheckpointError,
    ConfigurationError,
    ForgeEngineError,
    GenerationError,
    GenerationOOMError,
    GenerationTimeoutError,
)
from research.inference.session_cache import SessionCacheManager
from research.inference.hotswap import HotSwapManager, EngineSettings
from research.inference.library import Library
from research.inference.engine_tools import EngineToolRegistry
from research.model_loader import unpack_output_with_kv


class ForgeEngine:
    """Unified inference engine for ForgeAI XP models.

    Orchestrates all runtime strategies and innovations. Auto-detects
    KeyStack features from checkpoint and activates matching strategies.
    """

    def __init__(self, model, tokenizer, device="cuda",
                 checkpoint_path: str | None = None):
        # Reduce CUDA memory fragmentation (critical for 12GB VRAM)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # LoRA hot-loading state
        self._lora_config: dict | None = None

        # Strategy slots
        self.kv_cache: KVCacheStrategy | None = None
        self.decoding: DecodingStrategy = StandardDecoding()
        self.quantize: str | None = None
        self.acceleration: str | None = None
        self._active_kv_bits = 8  # default; updated by _activate_kv_cache
        self._active_kv_cache_name: str | None = None

        # Innovation slots
        self.mrl_adapter: MRLAdaptiveContext | None = None
        self.quarot_kv: QuaRotKV | None = None
        self.v0_warm: V0WarmStart | None = None
        self.progressive_kv: ProgressiveKV | None = None

        # Detected KeyStack features
        self.keystack_features: list[str] = []

        # Stats
        self.generation_count = 0
        self.total_tokens_generated = 0
        self._prefix_cache = None
        self._cache_blend: CacheBlend | None = None  # R&D14: CacheBlend
        self._graph_runner = None
        self._stop_tokens = None
        self._awake = True
        self._sleep_level = 0
        self._last_activation_params: dict | None = None

        # Built-in diagnostics (replaces need for one-off scripts)
        self.events = EventLog(capacity=500)
        self.outputs = OutputHistory(capacity=100)
        self._profiler = EngineProfiler(self.model, self.device)
        self._log("ForgeEngine initialized",
                  device=str(self.device),
                  checkpoint=checkpoint_path or "none")

        # Crash recovery: signal handlers + atexit + disk checkpointing
        self._recovery = CrashRecoveryManager(self, enabled=True)

        # Session-aware KV cache: multi-turn optimization with radix prefix matching + TTL
        self._session_cache = SessionCacheManager(self)

        # Hot-swap manager: runtime config changes without reload
        self.hotswap = HotSwapManager(self)

        # Library: persistent knowledge base with lorebook-style injection
        from research.paths import LIBRARY_DIR
        self.library = Library(
            tokenizer=tokenizer, path=str(LIBRARY_DIR),
        )
        self._library_enabled = True
        self._library_injection_budget = 2048

        # Built-in tool registry: gives the LLM tools to use engine features
        self.tools = EngineToolRegistry(self)

        # Move model to device (unless it's on meta — streaming mode)
        self._needs_streaming = False
        first_param = next(self.model.parameters(), None)
        if first_param is not None and first_param.device.type != "meta":
            self.model.to(self.device)
        self.model.eval()

        # Auto-detect KeyStack features
        self._checkpoint_metadata = {}
        if checkpoint_path:
            self._detect_keystack_features()
            # Note: int8 conversion is handled in from_checkpoint() for pre-quant
            # checkpoints, since it requires casting int8->fp32 before load_state_dict

    # ── Logging ──────────────────────────────────────────────────────────

    def _log(self, message: str, level: str = "info",
             source: str = "engine", **data):
        """Print a status message and record it in the event log.

        Replaces scattered ``print()`` calls with a single chanel that
        both shows the message to the user and stores it for
        ``read_log()`` / ``diagnose()`` diagnostics.
        """
        print(f"  [{source.title()}] {message}" if source != "engine"
              else f"  [ForgeEngine] {message}")
        self.events.log(message, level=level, source=source, **data)

    def _clear_cuda_cache(self):
        """Synchronize + empty CUDA cache. Deduplicates the 6x repeated
        ``synchronize / empty_cache / synchronize`` pattern across generate,
        sleep, and OOM-recovery paths."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _clear_cuda_cache_static(device):
        """Static version for use in classmethods (before an instance exists)."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    def _release_acceleration_resources(self):
        """Release CUDA graph / megakernel / flex-decoding / chunked-prefill
        resources that hold GPU memory independent of ``self.model``.

        Called from ``sleep(level=2)`` before discarding the model to prevent
        leaks of CUDA graph pools and megakernel capture buffers.
        """
        for attr in ("_graph_runner", "_megakernel",
                     "_flex_decoding", "_chunked_prefill"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            # If the object exposes a cleanup/release method, call it.
            for method_name in ("release", "cleanup", "destroy", "close"):
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
                    break
            setattr(self, attr, None)
        self.acceleration = None

    # ── Convenience properties ──────────────────────────────────────────
    # These derive from the loaded model so callers don't need to reach
    # into self.model.config / self.model.parameters() manually.

    @property
    def config(self):
        """ModelConfig attached to the loaded model (or None)."""
        return getattr(self.model, "config", None)

    # ── Library control ──────────────────────────────────────────────────

    def library_save(self, content: str, category: str = "custom",
                     tags: list[str] | None = None,
                     description: str = "",
                     triggers: list[str] | None = None,
                     priority: int = 0) -> str:
        """Save an entry to the library (model self-write).

        Categories: "failure", "win", "research", "common_data", "custom".
        Returns entry_id.
        """
        return self.library.save(
            content=content, category=category, tags=tags,
            description=description, triggers=triggers,
            priority=priority, source="model")

    def library_set_enabled(self, enabled: bool) -> None:
        """Enable/disable library lorebook injection globally."""
        self._library_enabled = enabled

    def library_set_budget(self, tokens: int) -> None:
        """Set the injection token budget (max tokens injected per request)."""
        self._library_injection_budget = tokens

    def library_lookup(self, **kwargs) -> list:
        """Lookup library entries by tags/category."""
        return self.library.lookup(**kwargs)

    def library_search(self, query: str, limit: int = 20) -> list:
        """Full-text search the library."""
        return self.library.search(query, limit=limit)

    def library_optimize(self) -> dict:
        """Run library optimization (merge similar, trim, re-index)."""
        return self.library.optimize()

    def library_stats(self) -> dict:
        """Get library statistics."""
        return self.library.stats()

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the model's first parameter (bf16 by default)."""
        if self.model is None:
            return getattr(self, "_stored_dtype", torch.bfloat16)
        parameter = next(self.model.parameters(), None)
        return parameter.dtype if parameter is not None else torch.bfloat16

    @staticmethod
    def _memory_info(device: torch.device) -> tuple[int, int]:
        if device.type == "cuda":
            return torch.cuda.mem_get_info(device)
        return _DEFAULT_CPU_MEMORY_BYTES, _DEFAULT_CPU_MEMORY_BYTES

    @property
    def _kv_dimensions(self) -> tuple[int, int]:
        return (
            getattr(self.config, "n_kv_heads", 8),
            getattr(self.config, "head_dim", 64),
        )

    def _require_awake(self):
        if not self._awake:
            raise ForgeEngineError(
                "ForgeEngine is asleep; call wake() before inference",
                context={"sleep_level": self._sleep_level},
                suggestion="Call engine.wake() to restore the model to GPU.")

    @property
    def active_config(self) -> ActivationConfig | None:
        """The ``ActivationConfig`` from the last ``activate()`` call, or None."""
        params = getattr(self, "_last_activation_params", None)
        if params is None:
            return None
        return ActivationConfig.from_kwargs(**params)

    def reset_stats(self) -> None:
        """Reset generation counters and diagnostics history.

        Useful for benchmarking: call before a measurement run to get
        clean stats without interference from prior generations.
        """
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.events.clear()
        self.outputs = OutputHistory(capacity=100)

    def __repr__(self) -> str:
        status = "awake" if self._awake else f"asleep(L{self._sleep_level})"
        n_params = 0
        if self.model is not None:
            n_params = sum(p.numel() for p in self.model.parameters())
        cfg_name = getattr(self.config, "name", "unknown")
        return (f"ForgeEngine(model={cfg_name}, "
                f"params={n_params/1e6:.0f}M, "
                f"device={self.device}, "
                f"kv={self.kv_cache.info().get('type', 'none') if self.kv_cache else 'none'}, "
                f"decoding={self.decoding.name}, "
                f"{status})")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-sleep level 1 on context exit to release VRAM."""
        if self._awake:
            self.sleep(level=1)
        return False

    # ── Checkpoint loading ────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, checkpoint: str, config_name: str = "forgelm_v2_light",
                        tokenizer_path: str | None = None,
                        device: str = "cuda",
                        auto_activate: bool = True,
                        **kwargs) -> "ForgeEngine":
        """Build engine from a KeyStack checkpoint.

        Auto-checks VRAM capacity and picks the best loading strategy:
          1. Pre-quantized BitNet → int8 direct load
          2. Fits in VRAM → fast meta-init load
          3. Fits with hybrid offload → conv on CPU, attention on GPU
          4. Too large → AirLLM layer-streaming (meta device + shard loading)

        Args:
            auto_activate: If True (default), automatically calls
                ``activate_optimal()`` with keystack-aware overrides after
                loading. Detected features (MTP, value_residual, etc.) are
                auto-enabled. Set to False for manual activation.
        """
        from research.config import get_config
        from research.tokenizer_cache import get_tokenizer

        cfg = get_config(config_name, device=device)
        tok_path = tokenizer_path or "research/checkpoints/lfm25_tokenizer"
        tokenizer = get_tokenizer(tok_path)

        ckpt_size = _checkpoint_size_cache.get(checkpoint)
        if ckpt_size is None:
            try:
                ckpt_size = Path(checkpoint).stat().st_size
            except OSError as e:
                raise CheckpointError(
                    f"Cannot access checkpoint '{checkpoint}': {e}",
                    context={"checkpoint": checkpoint},
                    suggestion="Verify the path exists and is a valid "
                               "safetensors file or directory of shards.")
            with _ckpt_cache_lock:
                _checkpoint_size_cache[checkpoint] = ckpt_size
                while len(_checkpoint_size_cache) > _CKPT_CACHE_MAX:
                    _checkpoint_size_cache.popitem(last=False)
        dev = torch.device(device)
        vram_free, _ = cls._memory_info(dev)
        needed = int(ckpt_size * 1.3)
        fits = vram_free > needed

        metadata = cls._read_checkpoint_metadata(checkpoint)
        is_prequant = metadata.get("_bitnet_prequant") == "1"

        engine = None
        try:
            if is_prequant:
                engine = cls._load_prequant(
                    cfg, checkpoint, tokenizer, device, metadata, **kwargs)
            elif fits:
                engine = cls._load_standard(
                    cfg, checkpoint, tokenizer, device, **kwargs)
            else:
                # Check if hybrid offload can bridge the gap
                engine = cls._load_with_fallback(
                    cfg, checkpoint, tokenizer, device,
                    ckpt_size, vram_free, **kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # If the primary load path fails (OOM, corrupt weights, etc.),
            # fall through to the streaming path as a last resort.
            print(f"  [ForgeEngine] Load path failed ({e}), "
                  f"falling back to AirLLM streaming...")
            cls._clear_cuda_cache_static(dev)
            engine = cls._load_streaming(
                cfg, checkpoint, tokenizer, device,
                ckpt_size, vram_free, **kwargs)

        if auto_activate and engine is not None:
            engine._auto_activate_optimal()
        return engine

    def _auto_activate_optimal(self):
        """Auto-activate optimal strategies based on detected KeyStack features.

        Inspects ``self.keystack_features`` and enables feature-appropriate
        strategies that are safe and beneficial:
          - MTP detected → mtp_selfspec decoding (2-4x speedup)
          - value_residual detected → use_v0_warm (lossless quality boost)
          - bitnet_prequant → skip quantize (already ternary)
          - CUDA device → enable block fusion + breakable CUDA graphs
          - VRAM-aware: auto-select highest quantization that fits
        """
        overrides = {}

        # torch.compile is broken on some triton/GPU stacks (inductor
        # mis-types scalar args of @triton.jit kernels → InductorError on
        # every forward). FORGE_NO_COMPILE=1 opts out at load time — the
        # GUI fast-load path sets this before importing the engine.
        if os.environ.get("FORGE_NO_COMPILE", "").strip().lower() in (
                "1", "true", "yes"):
            overrides["use_compile"] = False

        # Keystack-aware overrides
        if "mtp" in self.keystack_features:
            overrides["decoding"] = "mtp_selfspec"
        if "value_residual" in self.keystack_features:
            overrides["use_v0_warm"] = True
        if "bitnet_prequant" in self.keystack_features:
            # Weights are already ternary int8 — no quantize needed
            overrides["quantize"] = None
        else:
            # Auto-select highest quantization that fits VRAM
            quant = self._auto_select_quantization()
            if quant:
                overrides["quantize"] = quant

        # VRAM-aware feature selection
        if self.device.type == "cuda":
            vram_free, vram_total = self._memory_info(self.device)
            vram_ratio = vram_free / vram_total if vram_total > 0 else 0

            # RotorQuant KV cache: Givens rotation + Lloyd-Max quantization
            # More refined than TurboQuant (block-diagonal vs dense rotation,
            # deferred quantization for zero error compounding during prefill).
            # 3-4 bit KV with 0.94% error, ~8x compression.
            overrides.setdefault("kv_cache", "rotorquant")
            overrides.setdefault("kv_bits", 4)

            if vram_ratio > 0.5:
                # Ample VRAM — enable aggressive speed features
                overrides["use_block_fusion"] = True
                overrides["use_breakable_cuda_graph"] = True
                overrides["use_learned_prefix_cache"] = True
            elif vram_ratio < 0.25:
                # Tight VRAM — skip graph features, keep RotorQuant + 4-bit
                overrides["use_block_fusion"] = False
                overrides["use_breakable_cuda_graph"] = False

        # Don't activate if model is on meta (streaming mode)
        if self._needs_streaming:
            self._log("Auto-activate skipped (streaming mode)", level="info")
            return

        self._log(f"Auto-activating optimal strategies "
                  f"(overrides: {list(overrides.keys()) or 'none'})")
        self.activate_optimal(**overrides)

    def _auto_select_quantization(self) -> str | None:
        """Pick the highest quantization level that fits available VRAM.

        Priority (highest quality first):
          1. nvfp4 — if Blackwell GPU (native FP4, ~99% quality, 3.8x compression)
             NOW THE DEFAULT on Blackwell — quality is near-lossless and it
             frees VRAM for larger KV cache / longer context.
          2. None (bf16) — only if nvfp4 unavailable and VRAM is ample
          3. fp8 — if Hopper+ (hardware-native FP8)
          4. w8a8 — 2-3x speedup, minimal quality loss
          5. int4 — 4x compression, last resort

        Returns None if no quantization is needed (ample VRAM + no Blackwell).
        """
        if self.device.type != "cuda":
            return None

        # Check GPU capability for hardware-native quantization
        cap = torch.cuda.get_device_capability(self.device)
        # Blackwell = SM 100+ (RTX 5070 = SM 120)
        is_blackwell = cap[0] >= 10
        # Hopper = SM 90 (H100)
        is_hopper = cap[0] >= 9

        # NVFP4 is now the default on Blackwell — near-lossless quality,
        # 3.8x compression, frees VRAM for KV cache / longer context.
        # Only skip if the model is already BitNet (ternary) or explicitly
        # configured to use a different quant.
        if is_blackwell:
            try:
                from research.inference.quant.nvfp4_quant import quantize_model_nvfp4  # noqa
                return "nvfp4"
            except ImportError:
                pass
            try:
                from research.quantization.fp8_infer import quantize_model_fp8  # noqa
                return "fp8"
            except ImportError:
                pass

        # Estimate model size in VRAM for non-Blackwell path
        n_params = sum(p.numel() for p in self.model.parameters())
        model_bytes_bf16 = n_params * 2  # bf16 = 2 bytes/param
        vram_free, _ = self._memory_info(self.device)

        # If we have > 2x model size free, no quantization needed
        if vram_free > model_bytes_bf16 * 2:
            return None

        if is_hopper:
            try:
                from research.quantization.fp8_infer import quantize_model_fp8  # noqa
                return "fp8"
            except ImportError:
                pass

        # W8A8: 2-3x speedup, works on all CUDA GPUs with torch._int_mm
        if vram_free > model_bytes_bf16 * 0.6:
            return "w8a8"

        # INT4: 4x compression, last resort for very tight VRAM
        return "int4"

    @staticmethod
    def _read_checkpoint_metadata(checkpoint: str) -> dict:
        """Read safetensors metadata from a checkpoint (single or sharded).

        Uses a module-level cache keyed by (path, mtime) to avoid re-reading
        metadata from the same file on every call.
        """
        try:
            mtime = Path(checkpoint).stat().st_mtime
        except OSError:
            mtime = 0.0
        cache_key = (checkpoint, mtime)
        with _ckpt_cache_lock:
            cached = _checkpoint_metadata_cache.get(cache_key)
        if cached is not None:
            return cached

        from safetensors import safe_open
        metadata = {}
        try:
            with safe_open(checkpoint, framework="pt") as checkpoint_file:
                metadata = checkpoint_file.metadata() or {}
        except (OSError, RuntimeError, ValueError):
            # File missing, corrupt, or not a valid safetensors archive —
            # return empty metadata rather than crashing the engine.
            metadata = {}
        with _ckpt_cache_lock:
            _checkpoint_metadata_cache[cache_key] = metadata
            while len(_checkpoint_metadata_cache) > _CKPT_CACHE_MAX:
                _checkpoint_metadata_cache.popitem(last=False)
        return metadata

    @classmethod
    def _load_prequant(cls, cfg, checkpoint, tokenizer, device,
                       metadata, **kwargs):
        """Load a pre-quantized BitNet checkpoint directly into int8 storage.

        Avoids the wasteful int8→bf16→int8 round-trip of the old path.
        Instead, loads int8 tensors from the safetensors file and calls
        BitNetLinear.load_prequantized() to store them directly as int8
        buffers. Non-BitNet tensors (embeddings, norms, etc.) are loaded
        normally as bf16. Peak CPU RAM is ~6 GB (int8 state dict) instead
        of ~17 GB (int8 + bf16 intermediate).
        """
        from research.keys.quantization.bitnet_b158_key import (
            BitNetLinear, BitNetConv1d, BitNetEmbedding,
        )
        from research.model_loader import ModelLoader
        from research.config import ModelConfig
        from safetensors import safe_open
        import time, gc

        t0 = time.time()
        print("  [ForgeEngine] Pre-quantized BitNet checkpoint: "
              "direct int8 loading (no bf16 intermediate)")

        # 1. Build model on meta device (fast, no real tensors)
        cfg_meta = ModelConfig(**{**cfg.__dict__, "device": "meta"})
        with torch.device("meta"):
            from research.model_loader import ConfigurableResearchLLM
            model = ConfigurableResearchLLM(cfg_meta)
        print(f"  [FastBuild] Meta-init architecture in {time.time()-t0:.1f}s")

        # 2. Load safetensors to CPU (int8 tensors stay int8 — no cast!)
        t_weights = time.time()
        state = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        n_int8 = sum(1 for t in state.values() if t.dtype == torch.int8)
        n_bf16 = sum(1 for t in state.values() if t.dtype != torch.int8)
        int8_params = sum(t.numel() for t in state.values() if t.dtype == torch.int8)
        other_params = sum(t.numel() for t in state.values() if t.dtype != torch.int8)
        print(f"  [FastBuild] Loaded {len(state)} tensors "
              f"({n_int8} int8={int8_params/1e9:.2f}B, "
              f"{n_bf16} other={other_params/1e9:.2f}B) "
              f"in {time.time()-t_weights:.1f}s")

        # 3. Build a map from parameter name → module for all BitNet types
        #    (BitNetLinear, BitNetConv1d, BitNetEmbedding) so we can call
        #    load_prequantized for int8 weights.
        bitnet_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, (BitNetLinear, BitNetConv1d, BitNetEmbedding)):
                bitnet_modules[name + ".weight"] = module
        # Also handle head (nn.Linear) — store int8 as a buffer manually
        head_module = getattr(model, 'head', None)

        # 4. Assign tensors: int8 → load_prequantized, others → assign
        t_gpu = time.time()
        int8_loaded = 0
        other_keys = {}
        # Collect qscale tensors to pair with their int8 weights
        qscale_map = {}
        for key, tensor in state.items():
            if key.endswith(".qscale") and tensor.dtype != torch.int8:
                qscale_map[key[:-len(".qscale")] + ".weight"] = tensor
        for key, tensor in state.items():
            if key in bitnet_modules and tensor.dtype == torch.int8:
                # Direct int8 loading — no bf16 intermediate!
                module = bitnet_modules[key]
                # Use checkpoint's qscale if available, else compute from absmean
                # (use bf16 not fp32 to minimize memory for the fallback)
                if key in qscale_map:
                    qscale = qscale_map[key]
                else:
                    absmean = tensor.to(torch.bfloat16).abs().float().mean().clamp(min=1e-8)
                    qscale = absmean / 0.7
                # Move to target device as int8
                module.load_prequantized(
                    tensor.to(device).to(torch.int8), qscale.to(device))
                int8_loaded += 1
            elif key == "head.weight" and tensor.dtype == torch.int8 and head_module is not None:
                # Head is nn.Linear — convert to int8 buffer storage manually
                dev = torch.device(device)
                w_int8 = tensor.to(dev).to(torch.int8)
                qscale = qscale_map.get(key)
                if qscale is None:
                    absmean = tensor.to(torch.bfloat16).abs().float().mean().clamp(min=1e-8)
                    qscale = absmean / 0.7
                qscale = qscale.to(dev)
                del head_module.weight
                head_module.register_buffer("weight_int8", w_int8)
                head_module.register_buffer("qscale_buf", qscale)
                head_module._prequantized = True
                # Monkey-patch forward to use int8 buffer
                _orig_forward = head_module.forward
                def _int8_forward(x, _w=w_int8, _s=qscale, _b=head_module.bias):
                    return F.linear(x, _w.to(x.dtype), _b) * _s.to(x.dtype)
                head_module.forward = _int8_forward
                int8_loaded += 1
            elif key.endswith(".qscale") and (
                key[:-len(".qscale")] + ".weight" in bitnet_modules
                or key[:-len(".qscale")] + ".weight" == "head.weight"):
                # qscale for a BitNet/head layer already handled — skip
                continue
            else:
                other_keys[key] = tensor

        # 5. Load remaining (non-int8) tensors via assign=True
        if other_keys:
            missing, unexpected = model.load_state_dict(
                other_keys, strict=False, assign=True)
            if missing:
                real_missing = [k for k in missing if k != "head.weight"]
                if real_missing:
                    print("  Missing keys:", real_missing[:5],
                          "..." if len(real_missing) > 5 else "")
            if unexpected:
                print("  Unexpected keys:", unexpected[:5],
                      "..." if len(unexpected) > 5 else "")

        # 6. Re-tie weights if needed (assign breaks sharing)
        if getattr(cfg, 'tie_word_embeddings', True) \
                and not getattr(cfg, 'use_pit', False):
            model.head.weight = model.embed.weight

        # 7. Move non-int8 params/buffers to target device
        dev = torch.device(device)
        for module in model.modules():
            for pname, param in list(module._parameters.items()):
                if param is None:
                    continue
                if param.is_meta:
                    module._parameters[pname] = torch.nn.Parameter(
                        torch.zeros(param.shape, dtype=param.dtype, device=dev),
                        requires_grad=param.requires_grad)
                elif param.device != dev:
                    module._parameters[pname] = torch.nn.Parameter(
                        param.data.to(dev),
                        requires_grad=param.requires_grad)
            for bname, buf in list(module._buffers.items()):
                if buf is not None:
                    if buf.is_meta:
                        module._buffers[bname] = torch.zeros(
                            buf.shape, dtype=buf.dtype, device=dev)
                    elif buf.device != dev:
                        module._buffers[bname] = buf.to(dev)

        # 8. Reset non-persistent buffers (RoPE cos/sin)
        ModelLoader._reset_non_persistent_buffers(model, dev)

        if dev.type == "cuda":
            torch.cuda.synchronize()

        # 9. Free the state dict immediately
        del state, other_keys, bitnet_modules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 10. Post-load QK-norm identity scan
        for block in model.blocks:
            attn = block.attn
            if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
                q_id = (attn.q_norm.weight == 1.0).all()
                k_id = (attn.k_norm.weight == 1.0).all()
                attn._qk_norm_identity = bool(q_id and k_id)

        t_total = time.time() - t0
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [FastBuild] Direct int8 load: {int8_loaded} BitNet layers | "
              f"assign: {time.time()-t_gpu:.1f}s | Total: {t_total:.1f}s "
              f"({param_count:.1f}M params)")
        model.eval()

        engine = cls(model, tokenizer, device=device,
                     checkpoint_path=checkpoint, **kwargs)
        engine._checkpoint_metadata = metadata
        engine.keystack_features = ["bitnet_prequant", "quarot", "mrl"]
        engine._log(f"KeyStack features: {engine.keystack_features}")
        return engine

    @classmethod
    def _load_standard(cls, cfg, checkpoint, tokenizer, device, **kwargs):
        """Fast path: model fits in VRAM, load normally."""
        from research.model_loader import ModelLoader

        # Pass dtype=bfloat16 to prevent fp32 upcasting (saves 2x VRAM)
        model = ModelLoader.build_model_fast(
            cfg, checkpoint_path=checkpoint, dtype=torch.bfloat16)
        return cls(model, tokenizer, device=device,
                   checkpoint_path=checkpoint, **kwargs)

    @classmethod
    def _load_with_fallback(cls, cfg, checkpoint, tokenizer, device,
                            ckpt_size, vram_free, **kwargs):
        """Model doesn't fit entirely — try hybrid offload before streaming.

        Decision tree:
          1. If model has hybrid layer types (conv + attention), try
             hybrid_offload (conv on CPU, attention on GPU). This is much
             faster than full streaming since only conv weights go to CPU.
          2. If hybrid offload still doesn't fit, fall back to AirLLM
             layer-streaming (meta device + per-forward shard loading).
        """
        from research.model_loader import ModelLoader

        layer_types = getattr(cfg, "layer_types", None)
        has_hybrid = layer_types and any(
            lt != "attention" for lt in layer_types)

        if has_hybrid:
            # Estimate: attention layers on GPU, conv on CPU
            n_attn = sum(1 for lt in layer_types if lt == "attention")
            n_conv = len(layer_types) - n_attn
            # Rough estimate: attention layers are ~70% of model params
            attn_size = int(ckpt_size * 0.7)
            if vram_free > int(attn_size * 1.3):
                print(f"  [HybridOffload] Checkpoint {ckpt_size/1e9:.2f} GB > "
                      f"VRAM free {vram_free/1e9:.2f} GB, but model has "
                      f"{n_attn} attn + {n_conv} conv layers")
                print("  [HybridOffload] Trying hybrid: attention on GPU, "
                      "conv on CPU...")
                try:
                    model = ModelLoader.build_model_fast(
                        cfg, checkpoint_path=checkpoint, dtype=torch.bfloat16)
                    model = ModelLoader.hybrid_offload(
                        model, gpu_layers=-1, device=device)
                    engine = cls(model, tokenizer, device=device,
                                 checkpoint_path=checkpoint, **kwargs)
                    engine._log("Hybrid offload active: conv layers on CPU, "
                                "attention on GPU")
                    return engine
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print(f"  [HybridOffload] Failed ({e}), "
                          f"falling back to AirLLM streaming...")
                    cls._clear_cuda_cache_static(torch.device(device))

        # Fall back to full streaming
        return cls._load_streaming(
            cfg, checkpoint, tokenizer, device,
            ckpt_size, vram_free, **kwargs)

    @classmethod
    def _load_streaming(cls, cfg, checkpoint, tokenizer, device,
                        ckpt_size, vram_free, **kwargs):
        """Slow path: model too large for VRAM, build on meta for streaming."""
        from research.model_loader import ModelLoader

        print(f"  [AirLLM-Smart] Checkpoint {ckpt_size/1e9:.2f} GB > "
              f"VRAM free {vram_free/1e9:.2f} GB")
        print("  [AirLLM-Smart] Building model on meta device (zero VRAM)...")
        model = ModelLoader.build_model(cfg, checkpoint_path=None)
        model.eval()
        engine = cls(model, tokenizer, device=device,
                     checkpoint_path=checkpoint, **kwargs)
        engine._needs_streaming = True
        return engine

    def _detect_keystack_features(self):
        """Detect which KeyStack transforms are in the checkpoint.

        Degrades gracefully on I/O errors (missing/corrupt checkpoint):
        returns empty features + metadata instead of crashing.
        """
        features = []
        ckpt = Path(self.checkpoint_path) if self.checkpoint_path else None
        metadata = {}
        keys = set()
        if ckpt is None or not ckpt.exists():
            self._log("KeyStack feature detection skipped (no checkpoint)",
                      level="warn")
            self.keystack_features = features
            self._checkpoint_metadata = metadata
            return
        try:
            from safetensors import safe_open
            if ckpt.is_dir():
                shards = sorted(ckpt.glob("model-*.safetensors"))
                if shards:
                    with safe_open(str(shards[0]), framework="pt") as f:
                        keys = set(f.keys())
                        metadata = f.metadata() or {}
            else:
                with safe_open(str(ckpt), framework="pt") as f:
                    keys = set(f.keys())
                    metadata = f.metadata() or {}
        except (OSError, RuntimeError, ValueError, ImportError) as e:
            self._log(
                f"KeyStack feature detection failed: {e} — "
                f"continuing with empty features", level="warn")
            self.keystack_features = features
            self._checkpoint_metadata = metadata
            return

        if "value_residual_v0" in keys:
            features.append("value_residual")
        if "rotorquant_rotations" in keys:
            features.append("rotorquant")
        if "mtp_head.heads.0.weight" in keys:
            features.append("mtp")
        if "_airllm_streamable" in keys:
            features.append("airllm")
        if metadata.get("_bitnet_prequant") == "1":
            features.append("bitnet_prequant")
            self._log(f"Pre-quantized BitNet checkpoint detected "
                      f"(mode={metadata.get('_prequant_mode', 'int8')})")
        # QuaRot detection: check if V/O weights are Hadamard-rotated
        # (heuristic: compare against original if available)
        features.append("quarot")  # Assume applied by pipeline
        features.append("mrl")     # Assume applied by pipeline

        self.keystack_features = features
        self._checkpoint_metadata = metadata
        self._log(f"KeyStack features detected: {features}")

    # ── Activation ────────────────────────────────────────────────────────

    def activate_optimal(self, **overrides) -> None:
        """Activate with optimal settings for max VRAM efficiency + speed.

        Picks the best combination of strategies for the current hardware:
          - RotorQuant KV cache (Givens rotation + Lloyd-Max, ~8x compression)
          - torch.compile (1.3-2x decode speedup)
          - Fused QK-Norm+RoPE+Cache-Write (5-10% decode speedup)
          - Triton conv kernel (89% conv bottleneck cut)
          - Prefix cache (avoids re-computing repeated prefixes)
          - Chunked prefill (interleaves with decode for long prompts)

        For pre-quantized BitNet checkpoints, weights are already int8 —
        no quantize= needed (the model uses ternary GEMM natively).

        Args:
            **overrides: Override any activate() parameter.
        """
        config = ActivationConfig.optimal(**overrides)
        return self.activate_config(config)

    def activate(self, **kwargs) -> None:
        """Activate runtime strategies.

        All keyword arguments map directly to ``ActivationConfig`` fields.
        See ``ActivationConfig`` for the full parameter list and defaults.

        Common args:
            kv_cache: "standard", "paged", "rotorquant", "hadamard_int4", "compressed",
                      "streaming", "snapkv", "snapkv_4bit", "paged_eviction", "xquant",
                      "cpu_offload", "s4r", "hqe_kv"
            decoding: "standard", "speculative", "medusa", "dspark", "eagle3", "mtp_selfspec"
            quantize: None, "int8", "int4", "fp8", "w8a8", "nvfp4"
            acceleration: None, "cuda_graph", "airllm_streaming", "megakernel", "flex_decoding"
            mrl_keep_ratio: if set (e.g. 0.75), truncate to that fraction of dims
            kv_bits: 4 or 8, for KV cache quantization
            use_compile: torch.compile the model for 1.3-2x decode speedup
            use_prefix_cache: cache KV for repeated prompt prefixes
            warmup: pre-run a dummy token to initialize CUDA kernels (avoids
                    first-generation slowdown). Like llama.cpp's graph reservation.

        For the full set of 50+ feature flags, see ``ActivationConfig``.
        """
        self._require_awake()
        config = ActivationConfig.from_kwargs(**kwargs)
        self.activate_config(config)

    def activate_config(self, config: ActivationConfig) -> None:
        """Activate runtime strategies from an ``ActivationConfig``.

        This is the primary activation path — ``activate()`` and
        ``activate_optimal()`` both delegate here.
        """
        self._require_awake()
        feature_flags = config.feature_flags

        self._activate_core_innovations(
            config.quantize, config.mrl_keep_ratio, config.kv_cache,
            config.kv_bits, feature_flags)
        self._activate_kv_cache(config.kv_cache, config.kv_cache_tokens)
        self._activate_decoding(config.decoding)
        self._activate_acceleration(config.acceleration)
        self._activate_compile_runtime(feature_flags)
        self._apply_feature_registry(feature_flags)
        self._finalize_activation(feature_flags)

        # Store activation parameters for level-2 wake re-activation.
        self._last_activation_params = config.to_dict()

    def _activate_core_innovations(self, quantize, mrl_keep_ratio, kv_cache,
                                   kv_bits, feature_flags):
        # 1. Quantization
        if quantize:
            self._apply_quantization(quantize)
            self.quantize = quantize

        # 2. MRL adaptive context
        if mrl_keep_ratio and mrl_keep_ratio < 1.0:
            cfg = getattr(self.model, "config", None)
            d_model = getattr(cfg, "d_model", 1536)
            self.mrl_adapter = MRLAdaptiveContext(d_model, mrl_keep_ratio)
            self.mrl_adapter.apply_to_model(self.model)

        # 3. QuaRot-KV
        if "quarot" in self.keystack_features and kv_cache in ("hadamard_int4", "rotorquant"):
            self.quarot_kv = QuaRotKV(bits=kv_bits, has_quarot=True)
            self._log(f"QuaRot-KV active: V pre-rotated, "
                      f"K runtime-Hadamard, {kv_bits}-bit")

        # 4. V0 warm start
        if feature_flags["use_v0_warm"] and "value_residual" in self.keystack_features:
            self.v0_warm = V0WarmStart.from_checkpoint(self.checkpoint_path)
            if self.v0_warm:
                self._log(f"V0-WarmStart active: {self.v0_warm.info()}")
            else:
                self._log("V0-WarmStart: no V_0 found in checkpoint", level="warn")

        # 5. Progressive KV
        if feature_flags["use_progressive_kv"]:
            self.progressive_kv = ProgressiveKV(anchor_bits=8, residual_bits=8)
            self._log(f"ProgressiveKV active: {self.progressive_kv.info()}")

    # Fallback order for KV cache init failures (OOM, unsupported backend, etc.)
    _KV_FALLBACK_CHAIN = {
        "rotorquant": ["s4r", "hadamard_int4", "standard", "cpu_offload"],
        "hadamard_int4": ["s4r", "standard", "cpu_offload"],
        "paged": ["s4r", "standard", "cpu_offload"],
        "compressed": ["s4r", "standard", "cpu_offload"],
        "snapkv": ["s4r", "standard", "cpu_offload"],
        "snapkv_4bit": ["s4r", "standard", "cpu_offload"],
        "paged_eviction": ["s4r", "standard", "cpu_offload"],
        "xquant": ["s4r", "standard", "cpu_offload"],
        "hqe_kv": ["s4r", "standard", "cpu_offload"],
        "spectral": ["s4r", "standard", "cpu_offload"],
        "s4r": ["standard", "cpu_offload"],
        "standard": ["cpu_offload"],
        "cpu_offload": [],
        "streaming": [],
    }

    def _activate_kv_cache(self, kv_cache, kv_cache_tokens):
        cfg = getattr(self.model, "config", None)
        n_heads = getattr(cfg, "n_heads", 12)
        n_kv = getattr(cfg, "n_kv_heads", 2) or n_heads
        head_dim = getattr(cfg, "head_dim", None) or (
            getattr(cfg, "d_model", 1536) // n_heads)
        max_seq = getattr(cfg, "max_seq_len", 4096)
        if kv_cache_tokens is not None and kv_cache_tokens < max_seq:
            self._log(f"KV cache limited to {kv_cache_tokens} tokens "
                      f"(was {max_seq})")
            max_seq = kv_cache_tokens

        # VRAM-aware cap: don't allocate KV cache larger than free VRAM allows
        if self.device.type == "cuda":
            try:
                from research.runtime.vram_manager import VRAMManager
                n_layers = getattr(cfg, "n_layers", 16)
                dtype_bytes = self.dtype.itemsize if hasattr(self.dtype, 'itemsize') else 2
                vram_mgr = VRAMManager(safety_margin_gb=0.5)
                vram_max = vram_mgr.max_gen_tokens(
                    n_layers=n_layers, n_heads=n_kv,
                    head_dim=head_dim, dtype_bytes=dtype_bytes,
                    overhead_mb=256,
                )
                if vram_max < max_seq:
                    self._log(f"VRAM-aware KV cap: {vram_max} tokens "
                              f"(was {max_seq}, free VRAM limited)")
                    max_seq = vram_max
            except Exception as e:
                self._log(f"VRAM-aware KV sizing skipped: {e}", level="warn")

        # Try the requested KV cache, then fall back through progressively
        # simpler caches on OOM or unsupported-backend errors.
        chain = [kv_cache] + self._KV_FALLBACK_CHAIN.get(kv_cache, ["standard"])
        last_error = None
        for try_cache in chain:
            try:
                self.kv_cache = build_kv_cache(try_cache)
                self.kv_cache.init(n_heads, head_dim, n_kv, max_seq,
                                   str(self.device), self.dtype)
                # Track active KV bits for OOM-recovery fallback (s4r 4-bit, etc.)
                self._active_kv_bits = getattr(self.kv_cache, "bits", 8)
                self._active_kv_cache_name = try_cache
                if try_cache != kv_cache:
                    self._log(
                        f"KV cache fallback: '{kv_cache}' failed, "
                        f"using '{try_cache}' instead", level="warn")
                self._log(f"KV cache: {self.kv_cache.info()}")
                return
            except (torch.cuda.OutOfMemoryError, RuntimeError, ImportError,
                    ValueError) as e:
                last_error = e
                self._log(
                    f"KV cache '{try_cache}' init failed: {e}",
                    level="warn")
                self._clear_cuda_cache()
        # All fallbacks failed — this should be extremely rare
        raise ActivationError(
            f"All KV cache strategies failed (last: {last_error})",
            context={"requested": kv_cache, "tried": chain},
            suggestion="Try kv_cache='cpu_offload' or reduce max_seq_len.")

    def _activate_decoding(self, decoding):
        decode_kwargs = {}
        if decoding == "mtp_selfspec":
            decode_kwargs["k"] = 4
            if hasattr(self.model, "mtp_head"):
                decode_kwargs["mtp_module"] = self.model.mtp_head
        elif decoding == "eagle3":
            if hasattr(self.model, "eagle_head"):
                decode_kwargs["eagle_head"] = self.model.eagle_head
            elif self.checkpoint_path:
                eagle_path = self.checkpoint_path.replace(
                    ".safetensors", ".eagle3.safetensors")
                if os.path.exists(eagle_path):
                    from research.decoding.eagle import Eagle3Head, add_eagle3_to_model
                    head = add_eagle3_to_model(self.model)
                    from safetensors.torch import load_file
                    head.load_state_dict(load_file(eagle_path))
                    head = head.to(self.device)
                    decode_kwargs["eagle_head"] = head
                    self._log(f"EAGLE-3 head loaded from {eagle_path}")
            decode_kwargs.setdefault("draft_length", 4)
        self.decoding = build_decoding(decoding, **decode_kwargs)
        self._log(f"Decoding: {self.decoding.name}")

    def _activate_acceleration(self, acceleration):
        if acceleration == "cuda_graph" and self.device.type == "cuda":
            from research.runtime.cuda_graph import CudaGraphRunner
            self._graph_runner = CudaGraphRunner(
                self.model, batch_size=1, seq_len=1,
                device=str(self.device), use_cache=True)
            self._graph_runner.capture()
            self.acceleration = "cuda_graph"
            self._log("CUDA graphs: active")
        elif acceleration == "megakernel" and self.device.type == "cuda":
            from research.decoding.megakernel import CompiledMegakernelDecode
            self._megakernel = CompiledMegakernelDecode(
                self.model, device=str(self.device))
            try:
                self._megakernel.capture()
                self.acceleration = "megakernel"
                self._log("Megakernel decode: active (compiled + graph)")
            except Exception as e:
                self._log(f"Megakernel decode: failed ({e}), falling back",
                          level="warn")
                self._megakernel = None
                self.acceleration = None
        elif acceleration == "flex_decoding" and self.device.type == "cuda":
            from research.inference.attention.flex_decoding import FlexDecodingWrapper
            self._flex_decoding = FlexDecodingWrapper()
            if self._flex_decoding.apply(self.model):
                self.acceleration = "flex_decoding"
            else:
                self._flex_decoding = None
                self.acceleration = None
        elif acceleration == "airllm_streaming":
            AirLLMStreamer.setup(self)
        else:
            self._graph_runner = None
            self.acceleration = None

    def _activate_compile_runtime(self, feature_flags):
        # torch.compile
        if feature_flags["use_compile"] and self.device.type == "cuda":
            try:
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", dynamic=True)
                self._log("torch.compile: active (reduce-overhead)")
            except Exception as e:
                self._log(f"torch.compile: failed ({e})", level="warn")

        # Triton fused conv kernel
        if feature_flags["use_triton_conv"] and self.device.type == "cuda":
            try:
                from research.decoding.triton_conv import patch_conv_layers
                patch_conv_layers(self.model)
            except Exception as e:
                self._log(f"Triton conv: failed ({e})", level="warn")

        # Prefix caching — bounded LRU by default; ChunkedPrefixCache
        # (LMCache-style rolling-hash, R&D14) when use_chunked_prefix_cache.
        if feature_flags["use_prefix_cache"]:
            use_chunked = feature_flags.get("use_chunked_prefix_cache", False)
            if use_chunked:
                if not isinstance(self._prefix_cache, ChunkedPrefixCache):
                    self._prefix_cache = ChunkedPrefixCache(max_entries=64)
                self._log("Prefix caching: active (ChunkedPrefixCache, "
                          "256-token rolling hash)")
            else:
                if not isinstance(self._prefix_cache, LRUPrefixCache):
                    self._prefix_cache = LRUPrefixCache(max_entries=64)
                self._log("Prefix caching: active (LRU, max 64 entries)")
        else:
            self._prefix_cache = None

        # CacheBlend (R&D14): non-prefix KV reuse for RAG / tool-use.
        if feature_flags.get("use_cache_blend", False):
            if not isinstance(self._cache_blend, CacheBlend):
                self._cache_blend = CacheBlend()
            self._log("CacheBlend: active (non-prefix KV reuse)")
        else:
            self._cache_blend = None

        # Chunked prefill
        self._chunked_prefill = None
        if (feature_flags["use_chunked_prefill"]
                and not feature_flags["use_hybrid_prefill"]):
            from research.inference.prefill import ChunkedPrefiller
            self._chunked_prefill = ChunkedPrefiller(
                self.model, chunk_size=512, device=str(self.device))
            self._log("Chunked prefill: active (chunk_size=512)")

    def _apply_feature_registry(self, feature_flags):
        """Activate all registered inference features in a single pass.

        Iterates ``_FEATURE_REGISTRY`` (see ``feature_registry.py``) and
        invokes each enabled feature's handler.  Replaces the 11 per-category
        ``_activate_*`` methods (~500 lines) with one declarative loop.

        Each handler performs its own lazy import, instantiation, and
        ``.apply(model)`` call, then returns a status message (or ``None``).
        Exceptions are caught per-feature so one failure doesn't block the rest.
        """
        for spec in _FEATURE_REGISTRY:
            if not feature_flags.get(spec.flag, False):
                continue
            if spec.cuda_only and self.device.type != "cuda":
                continue
            try:
                message = spec.handler(self, feature_flags)
                if message:
                    self._log(message)
            except Exception as e:
                self._log(f"{spec.flag}: failed ({e})", level="warn")

    def _finalize_activation(self, feature_flags):
        # Warmup — pre-run a dummy token to initialize CUDA kernels
        if feature_flags["warmup"] and self.device.type == "cuda" and not self._needs_streaming:
            self._warmup()

        if self.device.type == "cuda":
            vram_free, vram_total = torch.cuda.mem_get_info(self.device)
            used_gb = (vram_total - vram_free) / 1e9
            free_gb = vram_free / 1e9
            self._log(f"VRAM: {used_gb:.2f} GB used, {free_gb:.2f} GB free",
                      level="profile")

    @torch.no_grad()
    def _warmup(self):
        """Pre-compile all CUDA kernels with dummy forward passes.

        The first real generation triggers JIT compilation of CUDA kernels,
        cuDNN algorithm selection, and memory pool initialization. This warmup
        runs a multi-token dummy pass through every layer type (conv + attention)
        with KV cache enabled, so all kernel variants are compiled upfront.

        This reduces the Layer 0 cold start from ~300ms (JIT compile) to ~1ms.
        """
        try:
            vocab_size = getattr(self.model, 'config', None)
            vocab_size = getattr(vocab_size, 'vocab_size', 65536) if vocab_size else 65536
            dummy = torch.randint(0, min(vocab_size, 32767), (1, 4),
                                  device=self.device, dtype=torch.long)
            with torch.inference_mode():
                self.model(dummy, use_cache=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._log("Warmup: all CUDA kernels pre-compiled "
                      "(conv + attn + KV cache)")
        except Exception as e:
            self._log(f"Warmup: skipped ({e})", level="warn")

    # Fallback order: if a quantization mode fails, try the next lower one.
    _QUANT_FALLBACK_CHAIN = {
        "nvfp4": ["w8a8", "fp8", "int8", "int4", None],
        "w8a8": ["fp8", "int8", "int4", None],
        "fp8": ["int8", "int4", None],
        "int8": ["int4", None],
        "int4": [None],
        None: [],
    }

    def _apply_quantization(self, mode: str):
        """Apply weight-only quantization with automatic fallback.

        If the requested mode fails (unsupported hardware, OOM, ImportError),
        falls back through progressively lower-bit modes, finally to
        unquantized bf16. Never leaves the model in a broken state.
        """
        chain = [mode] + self._QUANT_FALLBACK_CHAIN.get(mode, [None])
        last_error = None
        for try_mode in chain:
            if try_mode is None:
                self._log(
                    f"Quantization '{mode}' failed; falling back to "
                    f"unquantized bf16 (no compression)",
                    level="warn")
                if last_error:
                    self._log(f"  Last error: {last_error}", level="warn")
                return  # leave model unquantized
            try:
                self._apply_quantization_single(try_mode)
                if try_mode != mode:
                    self._log(
                        f"Quantization fallback: '{mode}' failed, "
                        f"using '{try_mode}' instead", level="warn")
                return
            except (ImportError, RuntimeError, ValueError,
                    torch.cuda.OutOfMemoryError) as e:
                last_error = e
                self._log(
                    f"Quantization '{try_mode}' failed: {e}",
                    level="warn")
                self._clear_cuda_cache()
        # Should not reach here (chain always ends with None), but just in case:
        raise ConfigurationError(
            f"All quantization modes failed for '{mode}'",
            context={"mode": mode, "last_error": str(last_error)},
            suggestion="Try quantize=None to run unquantized.")

    def _apply_quantization_single(self, mode: str):
        """Apply a single quantization mode (no fallback). Raises on failure."""
        if mode == "int8":
            from research.quantization.inference_quant import quantize_model_int8
            fast = torch.cuda.is_available()
            quantize_model_int8(self.model, fast=fast)
        elif mode == "int4":
            from research.quantization.inference_quant import quantize_model_int4
            quantize_model_int4(self.model, group_size=128)
        elif mode == "fp8":
            from research.quantization.fp8_infer import quantize_model_fp8
            quantize_model_fp8(self.model)
        elif mode == "w8a8":
            from research.inference.quant.w8a8_quant import quantize_model_w8a8
            w8a8_mode = getattr(self.model, 'config', None)
            w8a8_mode = getattr(w8a8_mode, 'w8a8_mode', 'int8') if w8a8_mode else 'int8'
            quantize_model_w8a8(self.model, mode=w8a8_mode)
            self._log(f"W8A8 quantization: {w8a8_mode}")
        elif mode == "nvfp4":
            from research.inference.quant.nvfp4_quant import quantize_model_nvfp4
            cfg = getattr(self.model, 'config', None)
            block_size = getattr(cfg, 'nvfp4_block_size', 32) if cfg else 32
            w4a8 = getattr(cfg, 'nvfp4_w4a8', False) if cfg else False
            quantize_model_nvfp4(self.model, block_size=block_size, w4a8=w4a8)
            self._log(f"NVFP4 quantization: active (Blackwell native FP4, "
                      f"block={block_size}, w4a8={w4a8})")
        else:
            raise ConfigurationError(
                f"Unknown quantization mode: {mode}",
                context={"mode": mode},
                suggestion="Use one of: int8, int4, fp8, w8a8, nvfp4")

    # ── Generation ────────────────────────────────────────────────────────

    _LOW_VRAM_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB

    def _check_vram_and_offload_if_needed(self):
        """Proactively check free VRAM and switch to CPU offload KV if low.

        If free VRAM drops below 500 MB, switches the KV cache strategy to
        ``cpu_offload`` before an OOM can occur. This avoids the expensive
        OOM recovery path (which retries the entire generation).
        """
        if self.device.type != "cuda":
            return
        try:
            free, _ = torch.cuda.mem_get_info(self.device)
        except Exception:
            return
        if free < self._LOW_VRAM_THRESHOLD_BYTES:
            kv_info = self.kv_cache.info() if self.kv_cache else {}
            kv_type = kv_info.get("type", kv_info.get("name", "none"))
            if kv_type != "cpu_offload":
                self._log(
                    f"Low VRAM ({free / 1e9:.2f} GB free < "
                    f"{self._LOW_VRAM_THRESHOLD_BYTES / 1e9:.2f} GB) — "
                    f"proactively switching KV cache to cpu_offload",
                    level="warn")
                self._clear_cuda_cache()
                try:
                    self._activate_kv_cache("cpu_offload", None)
                except Exception as e:
                    self._log(
                        f"Proactive CPU offload failed: {e}", level="warn")

    # Evolution-discovered creative sampling preset (score 10.45):
    # High temperature + high top_p + moderate top_k + low penalties.
    # Use for creative/diverse generation tasks.
    CREATIVE_SAMPLING = {
        "temperature": 1.98, "top_p": 0.989, "top_k": 69,
        "repetition_penalty": 1.011, "frequency_penalty": 0.014,
    }

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 80, repetition_penalty: float = 1.05,
                 finish_sentence: bool = True,
                 context_limit: int | None = None,
                 skip_special_tokens: bool = True) -> str:
        """Generate text from a prompt using active strategies.

        Args:
            top_k: LFM2.5-recommended top-k sampling (only applied when
                temperature > 0; ignored for greedy decoding).
            repetition_penalty: LFM2.5-recommended repetition penalty (only
                applied when temperature > 0; ignored for greedy decoding).
            finish_sentence: If True, when max_new_tokens is hit mid-sentence,
                continue generating up to 32 extra tokens to reach a natural
                stopping point (period, newline, code block close, EOS).
            context_limit: Override max context tokens for this request only.
                If None, uses the hotswap current setting. Set to a large
                number (e.g. 1_000_000) for infinite context with eviction.
            skip_special_tokens: If True (default), strips special tokens from
                output. Set to False for tool-call parsing (preserves
                <|tool_call_start|>/<|tool_call_end|> markers).
        """
        return self._generate_with_oom_recovery(
            self._generate_impl, prompt, max_new_tokens, temperature,
            top_p, top_k, repetition_penalty, finish_sentence,
            context_limit, skip_special_tokens)

    def _generate_impl(self, prompt, max_new_tokens, temperature, top_p,
                       top_k, repetition_penalty, finish_sentence,
                       context_limit, skip_special_tokens) -> str:
        """Internal generate implementation (no OOM wrapper)."""
        self._require_awake()
        self._validate_generation_params(
            prompt, max_new_tokens, temperature, top_p, top_k,
            repetition_penalty)
        self._check_vram_and_offload_if_needed()

        # Apply pending hot-swap changes before generation
        self.hotswap.apply_pending()

        # Library lorebook injection: augment prompt with relevant entries
        if self._library_enabled and self.library is not None:
            prompt = self.library.inject(
                prompt, max_tokens=self._library_injection_budget)

        # Per-request context limit override
        ctx_limit = context_limit or self.hotswap.current.max_context_tokens

        _t0 = time.perf_counter()
        ids = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=ctx_limit,
            add_special_tokens=True,  # BOS <|startoftext|> — required for sane raw prompts
        ).input_ids.to(self.device)

        # CacheBlend (R&D14): non-prefix KV reuse for RAG / tool-use.
        # Attempted before prefix caching; on a productive blend it
        # assembles a KV buffer from pre-computed chunks and decodes the
        # suffix, skipping most of the prefill.  Falls through to the
        # prefix-cache / standard path on a miss (zero overhead).
        output_ids = None
        if self._cache_blend is not None:
            blend_result = self._cache_blend.blend_prefill(self, ids)
            if blend_result is not None:
                blend_kv, covered_len = blend_result
                suffix = ids[:, covered_len:]
                if suffix.shape[1] > 0 and blend_kv is not None:
                    with torch.inference_mode():
                        out = self.model(
                            suffix, past_key_values=blend_kv, use_cache=True)
                        logits, past_kv = unpack_output_with_kv(out)
                    output_ids = self._decode_with_kv(
                        ids, logits, past_kv, max_new_tokens, temperature,
                        top_p, top_k=top_k,
                        repetition_penalty=repetition_penalty)
                    print(f"  [CacheBlend] HIT (reused {covered_len} tokens, "
                          f"suffix {suffix.shape[1]} to prefill)")

        # Prefix caching: check if we've seen this prompt prefix before
        if output_ids is None:
            output_ids = _generate_from_prefix_cache(
                self, ids, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty)
        if output_ids is None and self.acceleration == "airllm_streaming":
            output_ids = AirLLMStreamer.generate(
                self, ids, max_new_tokens, temperature)
        elif output_ids is None:
            output_ids = self.decoding.generate(
                self.model, ids, max_new_tokens, temperature, top_p,
                top_k=top_k, repetition_penalty=repetition_penalty)

        # Capture KV cache from decoding step for fast finish-to-stop path
        captured_kv = getattr(self.model, '_forge_last_kv', None)

        # Smart cutoff: if we hit max_new_tokens without EOS, extend to next
        # natural stopping point (up to 32 extra tokens).
        if finish_sentence and output_ids.shape[1] - ids.shape[1] >= max_new_tokens:
            output_ids = self._finish_to_stop(
                output_ids, ids.shape[1], temperature, top_p,
                extra_budget=32, past_kv=captured_kv,
                top_k=top_k, repetition_penalty=repetition_penalty)

        # Store prefix KV cache for future reuse
        _cache_prompt_prefix(self, ids)

        n_gen = output_ids.shape[1] - ids.shape[1]
        self._record_generation(n_gen)
        prompt_len = ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=skip_special_tokens)

        _gen_ms = (time.perf_counter() - _t0) * 1000
        self._record_output(prompt, result, n_gen, _gen_ms, temperature)
        return result

    # ── CacheBlend public API (R&D14) ────────────────────────────────

    def enable_cache_blend(self, chunk_size: int = 256,
                           max_chunks: int = 512) -> CacheBlend:
        """Enable CacheBlend non-prefix KV reuse at runtime.

        Returns the ``CacheBlend`` instance so the caller can register
        chunks via ``register_chunk`` / ``register_text``.
        """
        if not isinstance(self._cache_blend, CacheBlend):
            self._cache_blend = CacheBlend(
                chunk_size=chunk_size, max_chunks=max_chunks)
        self._log("CacheBlend: enabled (non-prefix KV reuse)")
        return self._cache_blend

    def register_blend_chunk(self, text: str) -> int:
        """Pre-compute and store a chunk's KV for CacheBlend reuse."""
        if self._cache_blend is None:
            self.enable_cache_blend()
        return self._cache_blend.register_text(self, text)

    @torch.no_grad()
    def generate_adaptive(
        self,
        prompt: str,
        think_max_tokens: int = 512,
        no_think_max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        think_prefix: str = "Let me think about this step by step.\n",
    ) -> tuple[str, bool]:
        """Adaptive thinking generation (RPO-trained models).

        Uses the model's root token to decide whether to think or not:
        1. Forward pass on prompt to get root token logits
        2. If root token indicates thinking → generate with think_prefix
           and higher token budget
        3. If root token indicates direct answer → generate without
           think_prefix and lower token budget

        This gives ~50% token reduction on easy problems while maintaining
        accuracy on hard ones (ACL 2026, Kim et al.).

        Args:
            think_max_tokens: token budget when thinking is triggered
            no_think_max_tokens: token budget for direct answers
            think_prefix: text prepended when thinking mode is selected

        Returns:
            (generated_text, did_think) tuple
        """
        self._require_awake()
        self._check_vram_and_offload_if_needed()

        # Step 1: Forward pass on prompt to get root token logits
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=("cuda" in str(self.device)),
        ):
            logits, _ = self.model(ids)

        root_logits = logits[0, -1, :].float()
        root_token = root_logits.argmax().item()

        # Step 2: Decide think vs no-think based on root token
        # After RPO training, the root token encodes this decision.
        # Heuristic: if the root token matches common thinking markers,
        # use thinking mode; otherwise direct answer.
        think_markers = set()
        for marker in ["Let", "Let me", "First", "To solve", "I need"]:
            t_ids = self.tokenizer(marker, add_special_tokens=False).input_ids
            if t_ids:
                think_markers.add(t_ids[0])

        did_think = root_token in think_markers

        # Step 3: Generate with appropriate budget
        if did_think:
            full_prompt = prompt + think_prefix
            result = self.generate(
                full_prompt, max_new_tokens=think_max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty)
            return result, True
        else:
            result = self.generate(
                prompt, max_new_tokens=no_think_max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty)
            return result, False

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
    ) -> list[str]:
        """Generate text for multiple prompts in a single batched forward pass.

        Uses BatchedDecoding for 3-5x throughput vs serial generation.
        All prompts are processed simultaneously — the model's KV cache
        handles all sequences in parallel.

        Args:
            prompts: list of prompt strings (1-8 recommended for 12GB VRAM)
            max_new_tokens: max tokens to generate per prompt
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling threshold
            top_k: top-k sampling limit
            repetition_penalty: repetition penalty

        Returns:
            list of generated text strings (same order as prompts)
        """
        self._require_awake()
        self.hotswap.apply_pending()
        self._check_vram_and_offload_if_needed()

        if not prompts:
            return []

        # For single prompt, fall back to regular generate
        if len(prompts) == 1:
            return [self.generate(
                prompts[0], max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty)]

        # Tokenize all prompts
        all_ids = []
        for p in prompts:
            ids = self.tokenizer(
                p, return_tensors="pt", truncation=True,
                max_length=self.hotswap.current.max_context_tokens
            ).input_ids.to(self.device)
            all_ids.append(ids)

        # Use BatchedDecoding
        from research.inference.batched_decoding import BatchedDecoding
        batched = BatchedDecoding()

        _t0 = time.perf_counter()
        try:
            output_ids = batched.generate_batch(
                self.model, all_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        except torch.cuda.OutOfMemoryError:
            # Fallback: serial generation
            self._log(
                f"Batched OOM ({len(prompts)} prompts), falling back to serial",
                level="warn")
            self._clear_cuda_cache()
            return [self.generate(
                p, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty) for p in prompts]

        # Decode each output
        results = []
        for i, ids in enumerate(output_ids):
            prompt_len = all_ids[i].shape[1]
            generated = ids[0, prompt_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text)

        _gen_ms = (time.perf_counter() - _t0) * 1000
        total_tokens = sum(len(r) for r in results)
        self._record_generation(total_tokens)
        self._log(
            f"Batch generate: {len(prompts)} prompts, "
            f"{total_tokens} tokens, {_gen_ms:.0f}ms",
            source="batch")

        return results

    @torch.no_grad()
    def generate_with_tools(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        max_tool_rounds: int = 5,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        extra_tools: list[dict] | None = None,
    ) -> dict:
        """Agentic generation loop with built-in tool execution.

        The model generates a response. If it makes tool calls, they are
        executed server-side and the results are fed back. This continues
        for up to `max_tool_rounds` rounds or until the model stops
        calling tools.

        Built-in tools available to the model:
          - library_save/search/lookup/get/delete/stats/optimize/set_config
          - engine_set_kv_cache/decoding/context_limit/infinite_context/
            generation_params/feature/apply_changes
          - engine_get_settings/stats/pending
          - engine_batch_generate
          - engine_generate_adaptive

        Args:
            prompt: the user's prompt
            max_new_tokens: max tokens per generation round
            max_tool_rounds: max number of tool execution rounds
            temperature: sampling temperature
            extra_tools: additional tool definitions from the caller

        Returns:
            dict with:
              - "content": final text response
              - "tool_calls": list of all tool calls made
              - "tool_results": list of all tool results
              - "rounds": number of rounds executed
        """
        from research.self_play.discovery.qwen_adapter import (
            qwen_render_messages, qwen_parse_tool_calls,
        )

        # Build tool definitions: built-in + extra
        tool_defs = self.tools.get_tool_defs()
        if extra_tools:
            tool_defs.extend(extra_tools)

        all_tool_calls: list[dict] = []
        all_tool_results: list[dict] = []
        messages = [
            {"role": "user", "content": prompt}
        ]

        for round_idx in range(max_tool_rounds):
            # Render conversation with tools
            rendered = qwen_render_messages(
                messages, tools=tool_defs, add_generation_prompt=True)

            # Library injection on the full conversation
            if self._library_enabled and self.library is not None:
                rendered = self.library.inject(
                    rendered, max_tokens=self._library_injection_budget)

            # Generate
            raw = self.generate(
                rendered, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty)

            # Parse tool calls
            tool_calls, content = qwen_parse_tool_calls(raw)
            # Convert to OpenAI format if needed
            if tool_calls:
                parsed_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict) and "name" in tc:
                        parsed_calls.append(tc)
                    elif isinstance(tc, dict) and "function" in tc:
                        fn = tc["function"]
                        parsed_calls.append({
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", {})
                            if isinstance(fn.get("arguments"), dict)
                            else json.loads(fn.get("arguments", "{}"))
                        })
                tool_calls = parsed_calls

            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls or None,
            })

            if not tool_calls:
                # No more tool calls — we're done
                break

            # Execute tool calls
            results = self.tools.execute_calls(tool_calls)
            all_tool_calls.extend(tool_calls)
            all_tool_results.extend(results)

            # Feed results back to the model
            for call, result in zip(tool_calls, results):
                tool_name = call.get("name", "tool")
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return {
            "content": content or "",
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "rounds": round_idx + 1,
        }

    def _tokenize(self, prompt: str,
                  add_special_tokens: bool = True) -> torch.Tensor:
        """Tokenize a prompt and move token IDs to the engine's device.

        Shared by ``generate_raw`` and ``generate_stream`` so they use
        identical tokenization semantics. Special tokens (BOS
        ``<|startoftext|>``) are added by default — the model was trained
        with a BOS-prefixed sequence and raw prompts tokenize to garbage
        without it.
        """
        return self.tokenizer(
            prompt, return_tensors="pt",
            add_special_tokens=add_special_tokens).input_ids.to(self.device)

    def _prefill(self, ids: torch.Tensor):
        """Run prefill: process ``ids`` through the model with KV cache.

        Returns ``(logits, past_kv)`` — the full-sequence logits and the KV
        cache state ready for autoregressive decoding.

        Shared by ``generate_raw``, ``generate_stream``, and
        ``_finish_to_stop`` (slow path) to avoid duplicating the
        ``model(ids, use_cache=True) + unpack`` pattern.
        """
        with torch.inference_mode():
            out = self.model(ids, use_cache=True)
            return unpack_output_with_kv(out)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float,
                           top_k: int, top_p: float,
                           repetition_penalty: float,
                           generated_ids: list[int]) -> torch.Tensor:
        """Sample the next token from logits with top-k / top-p / rep-penalty.

        Centralised sampling used by generate_raw, generate_stream,
        _decode_with_kv and _finish_to_stop so they all share identical
        filtering semantics.

        Args:
            logits: (batch, vocab) already-sliced logits for the next
                position (i.e. ``logits[:, -1, :]`` from the model output).
            temperature: 0 = greedy argmax, >0 = probabilistic sampling.
            top_k: top-k filter (0 = disabled).
            top_p: nucleus filter (1.0 = disabled).
            repetition_penalty: divisor applied to last 64 generated tokens.
            generated_ids: token IDs generated so far (for rep penalty).

        Returns:
            next_token tensor of shape (batch, 1).
        """
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        if repetition_penalty <= 0:
            raise ConfigurationError("repetition_penalty must be positive")

        next_logits = logits / temperature
        # Repetition penalty (last 64 tokens)
        if generated_ids:
            repeated = torch.tensor(
                tuple(set(generated_ids[-64:])), device=logits.device)
            next_logits[:, repeated] /= repetition_penalty
        # Top-k filtering
        if top_k > 0:
            candidate_logits, candidate_ids = torch.topk(
                next_logits, min(top_k, next_logits.shape[-1]), dim=-1)
        else:
            candidate_logits = next_logits
            candidate_ids = torch.arange(
                next_logits.shape[-1], device=logits.device).expand_as(next_logits)
        # Top-p filtering
        if top_p < 1.0:
            candidate_logits, order = torch.sort(
                candidate_logits, descending=True, dim=-1)
            candidate_ids = candidate_ids.gather(-1, order)
            cumulative_probs = torch.cumsum(
                torch.softmax(candidate_logits, dim=-1), dim=-1)
            remove = cumulative_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            candidate_logits = candidate_logits.masked_fill(
                remove, float("-inf"))
        sampled = torch.multinomial(
            torch.softmax(candidate_logits, dim=-1), num_samples=1)
        return candidate_ids.gather(-1, sampled)

    def _eos_token_ids(self, custom_ids=None) -> set[int]:
        token_ids = set(custom_ids) if custom_ids else set(_DEFAULT_EOS_TOKEN_IDS)
        sources = (
            self.tokenizer,
            self.model,
            getattr(self.model, "config", None),
        )
        for source in sources:
            token_id = getattr(source, "eos_token_id", None)
            if isinstance(token_id, (list, tuple, set, frozenset)):
                token_ids.update(token_id)
            elif token_id is not None:
                token_ids.add(token_id)
        return token_ids

    def _decode_tokens(
        self, logits, past_kv, max_new_tokens, temperature, top_p, top_k,
        repetition_penalty, stop_token_ids, generated_ids=None,
        logits_processor=None,
    ):
        generated_ids = generated_ids if generated_ids is not None else []
        try:
            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :]

                # Constrained decoding: apply logits processor BEFORE sampling
                if logits_processor is not None:
                    next_logits = logits_processor(next_logits, generated_ids)

                next_token = self._sample_next_token(
                    next_logits, temperature, top_k, top_p,
                    repetition_penalty, generated_ids)
                token_id = next_token.item()
                generated_ids.append(token_id)
                should_stop = token_id in stop_token_ids
                yield next_token, should_stop
                if should_stop:
                    break

                # Crash recovery: checkpoint generation + KV snapshot
                n_gen = len(generated_ids)
                if hasattr(self, '_recovery') and self._recovery.enabled:
                    partial = self._safe_decode_ids(
                        generated_ids, skip_special_tokens=False)
                    self._recovery.checkpoint_generation(
                        "", partial, n_gen)  # prompt filled by caller
                    self._recovery.snapshot_kv_cache(n_gen)

                with torch.inference_mode():
                    out = self.model(
                        next_token, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)
        finally:
            if self.model is not None:
                self.model._forge_last_kv = past_kv

    @torch.no_grad()
    def generate_raw(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        logits_processor=None,
        eos_token_ids: list[int] | None = None,
        skip_special_tokens: bool = False,
    ) -> str:
        """Generate text with raw control — for self-play / agentic loops.

        Unlike ``generate()``, this method:
          - Supports a ``logits_processor`` callback for constrained decoding
            (e.g. xgrammar bitmask for tool-call JSON).
          - Returns the decoded string with configurable ``skip_special_tokens``
            (self-play needs special tokens preserved for tool-call parsing).
          - Does NOT do prefix caching or finish_sentence extension (the
            agentic loop manages its own stopping logic).
          - Uses the active KV cache strategy + Triton conv + torch.compile
            if activated on the engine.

        Args:
            prompt: input prompt string
            max_new_tokens: max tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling threshold
            top_k: top-k sampling
            repetition_penalty: repetition penalty
            logits_processor: optional callback ``(logits, token_ids) -> logits``
                called BEFORE top-k/temperature. Use for grammar constraints.
            eos_token_ids: custom EOS token IDs to stop on. If None, uses
                {7, 151643, 151645} (LFM2.5 + Qwen2.5 defaults).
            skip_special_tokens: if True, strips special tokens from output.
                Self-play needs False to preserve tool-call markers.

        Returns:
            Decoded string of generated tokens (not including prompt).
        """
        self._require_awake()
        self._validate_generation_params(
            prompt, max_new_tokens, temperature, top_p, top_k,
            repetition_penalty)
        self._check_vram_and_offload_if_needed()
        self.hotswap.apply_pending()

        def _run():
            ids = self._tokenize(prompt)
            eos_set = self._eos_token_ids(eos_token_ids)
            generated_ids: list[int] = []

            logits, past_kv = self._prefill(ids)
            for _ in self._decode_tokens(
                logits, past_kv, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, eos_set, generated_ids, logits_processor,
            ):
                pass

            self._record_generation(len(generated_ids))
            return self._safe_decode_ids(generated_ids, skip_special_tokens)

        return self._generate_with_oom_recovery(_run)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        skip_special_tokens: bool = True,
        logits_processor=None,
        eos_token_ids: list[int] | None = None,
    ) -> Iterator[str]:
        """Token-by-token streaming generator.

        Yields decoded text chunks (one per generated token) as they are
        produced, enabling true token-level SSE streaming with low
        time-to-first-token.

        Mirrors :meth:`generate_raw` decoding logic but yields each token's
        decoded text incrementally instead of collecting all tokens first.

        Args:
            logits_processor: optional callback ``(logits, token_ids) -> logits``
                called BEFORE top-k/temperature. Use for grammar constraints.
            eos_token_ids: custom EOS token IDs to stop on. If None, uses
                {7, 151643, 151645} (LFM2.5 + Qwen2.5 defaults).
        """
        self._require_awake()
        self._validate_generation_params(
            prompt, max_new_tokens, temperature, top_p, top_k,
            repetition_penalty)
        self._check_vram_and_offload_if_needed()
        self.hotswap.apply_pending()

        # Note: OOM recovery wraps the entire generator, but since generators
        # can't be retried mid-yield, we wrap the setup + first yield.
        # If OOM occurs, the generator raises GenerationOOMError.
        try:
            ids = self._tokenize(prompt)
            eos_set = self._eos_token_ids(eos_token_ids)
            generated_ids: list[int] = []

            logits, past_kv = self._prefill(ids)
            for next_token, _ in self._decode_tokens(
                logits, past_kv, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, eos_set, generated_ids, logits_processor,
            ):
                chunk = self._safe_decode_ids([next_token.item()],
                                              skip_special_tokens)
                if chunk:
                    yield chunk

            self._record_generation(len(generated_ids))
        except torch.cuda.OutOfMemoryError as e:
            self._clear_cuda_cache()
            vram = self.vram_usage() if self.device.type == "cuda" else {}
            raise GenerationOOMError(
                f"OOM during streaming generation: {e}",
                context={"vram": vram},
                suggestion=("Use generate_raw() instead of generate_stream() "
                            "for OOM recovery, or reduce max_new_tokens."))

    def _record_generation(self, n_gen: int):
        """Update generation counters (shared by all generate methods)."""
        self.generation_count += 1
        self.total_tokens_generated += n_gen

    def _validate_generation_params(self, prompt: str, max_new_tokens: int,
                                    temperature: float, top_p: float,
                                    top_k: int, repetition_penalty: float):
        """Validate generation parameters before starting.

        Raises ``ConfigurationError`` or ``GenerationError`` on invalid input.
        """
        if not isinstance(prompt, str) or not prompt:
            raise ConfigurationError("prompt must be a non-empty string")
        if max_new_tokens <= 0:
            raise ConfigurationError(
                f"max_new_tokens must be positive, got {max_new_tokens}")
        if max_new_tokens > 100_000:
            raise ConfigurationError(
                f"max_new_tokens={max_new_tokens} is unreasonably large "
                f"(max 100000)", suggestion="Use a smaller value or add a timeout.")
        if not (0 <= temperature <= 2.0):
            raise ConfigurationError(
                f"temperature must be in [0, 2.0], got {temperature}")
        if not (0 < top_p <= 1.0):
            raise ConfigurationError(
                f"top_p must be in (0, 1.0], got {top_p}")
        if top_k <= 0:
            raise ConfigurationError(
                f"top_k must be positive, got {top_k}")
        if repetition_penalty <= 0:
            raise ConfigurationError(
                f"repetition_penalty must be positive, got {repetition_penalty}")

        # Check prompt length vs model max_seq_len (warn, don't block —
        # tokenizer may handle truncation, and estimate is rough)
        # With infinite context mode, the limit is the KV cache budget
        # (adjustable via hotswap.set_context_limit()).
        max_seq = self.hotswap.current.max_context_tokens
        est_prompt_tokens = len(prompt) // 4
        if est_prompt_tokens > max_seq and not self.hotswap.current.infinite_context:
            self._log(
                f"Prompt ~{est_prompt_tokens} tokens may exceed max_seq_len "
                f"({max_seq}). Generation may truncate.",
                level="warn")

    def _generate_with_oom_recovery(self, generate_fn, *args, **kwargs):
        """Wrap a generation function with OOM detection and auto-degradation.

        If ``torch.cuda.OutOfMemoryError`` is raised during generation, this
        attempts recovery by:
          1. Clearing CUDA cache and retrying
          2. Reducing KV bits to 4 and switching to S4R cache
          3. Switching to CPU offload KV cache
          4. Giving up with a helpful error message
        """
        if self.device.type != "cuda":
            return generate_fn(*args, **kwargs)

        try:
            return generate_fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            self._log(f"OOM during generation: {e}", level="error")
            self._clear_cuda_cache()

            # Attempt 1: retry after cache clear
            try:
                self._log("Retrying after CUDA cache clear...")
                return generate_fn(*args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                pass

            # Attempt 2: reduce KV bits + switch to S4R
            old_kv = self.kv_cache
            old_bits = getattr(self, '_active_kv_bits', 8)
            try:
                self._log("Retrying with S4R 4-bit KV cache...")
                self._activate_kv_cache("s4r", None)
                result = generate_fn(*args, **kwargs)
                self._log("OOM recovery successful (S4R 4-bit)")
                return result
            except torch.cuda.OutOfMemoryError:
                self.kv_cache = old_kv  # restore
                pass

            # Attempt 3: CPU offload KV cache
            try:
                self._log("Retrying with CPU offload KV cache...")
                self._activate_kv_cache("cpu_offload", None)
                result = generate_fn(*args, **kwargs)
                self._log("OOM recovery successful (CPU offload KV)")
                return result
            except torch.cuda.OutOfMemoryError:
                self.kv_cache = old_kv
                pass

            # All recovery attempts failed
            vram = self.vram_usage()
            raise GenerationOOMError(
                f"Out of memory after all recovery attempts. "
                f"VRAM: {vram['used_gb']:.1f}/{vram['total_gb']:.1f} GB used.",
                context={"vram_used_gb": vram["used_gb"],
                         "vram_total_gb": vram["total_gb"],
                         "model": getattr(self.config, "name", "unknown")},
                suggestion=("Try engine.sleep(1) to offload weights to CPU, "
                            "or use quantize='int4' for 4x weight compression."))

    def _safe_decode_ids(self, token_ids: list[int],
                         skip_special_tokens: bool = True) -> str:
        """Decode token IDs, clamping to tokenizer vocab range.

        The model vocab may be larger than the tokenizer vocab (e.g. padding
        for tensor-parallel alignment). This clamps out-of-range IDs to the
        last valid token before decoding, preventing IndexError.

        Shared by ``generate_raw`` and ``generate_stream``.
        """
        tok_vocab = len(self.tokenizer)
        safe_ids = [t if t < tok_vocab else tok_vocab - 1 for t in token_ids]
        return self.tokenizer.decode(
            safe_ids, skip_special_tokens=skip_special_tokens)

    def _record_output(self, prompt, result, n_gen, gen_ms, temperature):
        """Record output + timing event for diagnostics."""
        kv_cache_info = self.kv_cache.info() if self.kv_cache else {}
        self.outputs.record(
            prompt, result, n_gen, gen_ms, temperature=temperature,
            kv_cache=kv_cache_info.get("name", kv_cache_info.get("type", "none")),
            decoding=self.decoding.name,
        )
        self.events.log(f"generate: {n_gen} tokens in {gen_ms:.0f}ms",
                        source="engine", level="profile",
                        tokens=n_gen, time_ms=round(gen_ms, 1),
                        tok_s=round(n_gen / (gen_ms / 1000), 1) if gen_ms > 0 else 0)

    @torch.no_grad()
    def _decode_with_kv(self, ids, logits, past_kv,
                        max_new_tokens, temperature, top_p,
                        top_k: int = 80, repetition_penalty: float = 1.05):
        """Standard autoregressive decode from existing KV cache state.

        Used by prefix cache fast path: prefill already done, just decode.
        """
        eos_set = self._eos_token_ids()
        generated_ids: list[int] = []
        generated_tokens = []

        for next_token, is_eos in self._decode_tokens(
            logits, past_kv, max_new_tokens, temperature, top_p, top_k,
            repetition_penalty, eos_set, generated_ids,
        ):
            if not is_eos:
                generated_tokens.append(next_token)

        if not generated_tokens:
            return ids
        return torch.cat([ids, *generated_tokens], dim=-1)

    # Token IDs for natural stopping points (Qwen2.5)
    def _get_stop_tokens(self) -> set[int]:
        """Get token IDs that indicate natural sentence/code boundaries."""
        if self._stop_tokens is not None:
            return self._stop_tokens
        tok = self.tokenizer
        stops = self._eos_token_ids()
        for text in [".", "!", "?", ".\n", "!\n", "?\n", ".\"", "!", "?",
                     "```\n", "```\n\n", ")\n", ")\n\n", "}\n", "}\n\n"]:
            ids = tok.encode(text, add_special_tokens=False)
            if ids:
                stops.add(ids[-1])
        self._stop_tokens = stops
        return stops

    @torch.no_grad()
    def _finish_to_stop(self, output_ids, prompt_len,
                        temperature, top_p, extra_budget=32,
                        past_kv=None, top_k: int = 80,
                        repetition_penalty: float = 1.05):
        """Continue generation until a natural stopping point or extra_budget.

        If past_kv is provided (captured from the decoding step), skips the
        expensive full-sequence re-run and continues directly from the last state.
        Otherwise falls back to a full prefill to recover KV cache state.
        """
        stop_tokens = self._get_stop_tokens()
        generated_ids = output_ids[0, prompt_len:].tolist()

        if past_kv is not None:
            last_token = output_ids[:, -1:]
            with torch.inference_mode():
                out = self.model(last_token, past_key_values=past_kv, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)
        else:
            logits, past_kv = self._prefill(output_ids)

        generated_tokens = [
            next_token
            for next_token, _ in self._decode_tokens(
                logits, past_kv, extra_budget, temperature, top_p, top_k,
                repetition_penalty, stop_tokens, generated_ids,
            )
        ]
        if not generated_tokens:
            return output_ids
        return torch.cat([output_ids, *generated_tokens], dim=-1)

    # ── Benchmarking & stats ──────────────────────────────────────────────

    def benchmark(self, prompt: str, max_new_tokens: int = 50,
                  n_runs: int = 3) -> dict:
        """Benchmark generation speed."""
        self._require_awake()
        if n_runs < 1:
            raise ConfigurationError("n_runs must be positive")
        self.generate(prompt, max_new_tokens=10, finish_sentence=False)  # warmup
        times = []
        token_counts = []
        for _ in range(n_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            tokens_before = self.total_tokens_generated
            start = time.perf_counter()
            self.generate(
                prompt, max_new_tokens=max_new_tokens, finish_sentence=False)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            times.append(time.perf_counter() - start)
            token_counts.append(self.total_tokens_generated - tokens_before)
        total_time = sum(times)
        total_tokens = sum(token_counts)
        average = total_time / n_runs
        tokens_per_second = total_tokens / total_time if total_time else 0
        self._log(f"{tokens_per_second:.0f} tok/s | "
                  f"{average * 1000:.0f}ms average", level="profile")
        return {
            "tokens_per_sec": tokens_per_second,
            "latency_ms": average * 1000,
            "tokens": max_new_tokens,
            "generated_tokens": total_tokens,
            "runs": n_runs,
        }

    def stats(self) -> dict:
        """Get engine statistics."""
        vram_info = self.vram_usage() if self.device.type == "cuda" else {}
        return {
            "generation_count": self.generation_count,
            "total_tokens_generated": self.total_tokens_generated,
            "keystack_features": self.keystack_features,
            "kv_cache": self.kv_cache.info() if self.kv_cache else None,
            "decoding": self.decoding.name,
            "quantization": self.quantize,
            "acceleration": self.acceleration,
            "mrl_adapter": self.mrl_adapter.info() if self.mrl_adapter else None,
            "quarot_kv": self.quarot_kv.info() if self.quarot_kv else None,
            "v0_warm": self.v0_warm.info() if self.v0_warm else None,
            "progressive_kv": self.progressive_kv.info() if self.progressive_kv else None,
            "vram": vram_info,
            "active_config": (self.active_config.to_dict()
                              if self.active_config else None),
        }

    # ── Built-in diagnostics ────────────────────────────────────────────
    # These methods eliminate the need for one-off profiling/log-reading scripts.

    def bottleneck(self, prompt: str = "The quick brown fox",
                   max_new_tokens: int = 16) -> dict:
        """Profile a generation pass and identify the slowest transformer layers.

        Runs a short generation with per-layer forward hooks to measure
        wall-clock time per block. Returns a dict with per-layer timings,
        top-5 bottlenecks, and overall throughput.

        No external profiling script needed — call this directly:
            report = engine.bottleneck()
            print(report["bottlenecks"])
        """
        self._require_awake()
        self.events.log("Starting bottleneck profiling", source="profile",
                        max_new_tokens=max_new_tokens)
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        result = self._profiler.profile_generate(ids, max_new_tokens=max_new_tokens)
        bottlenecks = result.get("bottlenecks", [])
        if not bottlenecks:
            self.events.warn(
                result.get("error", "No bottlenecks found"), source="profile")
            return result
        slowest = bottlenecks[0]
        self.events.log(
            f"Bottleneck: {result.get('tok_s', 0)} tok/s, "
            f"slowest={slowest['type']}#{slowest['index']} "
            f"({slowest['time_ms']}ms)",
            source="profile", level="profile", bottlenecks=bottlenecks,
        )
        return result

    def read_log(self, n: int = 50, level: str | None = None,
                 source: str | None = None) -> list[dict]:
        """Read recent engine events as structured dicts.

        Replaces log-tailing scripts. Optional filters by level/source:
            engine.read_log(n=20, level="error")  # recent errors
            engine.read_log(n=10, source="profile")  # recent timings
        """
        return self.events.read_log(n=n, level=level, source=source)

    def read_output(self, n: int = 10) -> list[dict]:
        """Read recent generation outputs with metadata.

        Replaces output-capture scripts. Returns last n generations:
            engine.read_output(n=5)  # last 5 generations with tok/s, timing
        """
        return self.outputs.read_output(n=n)

    def diagnose(self) -> dict:
        """Full health report: stats + VRAM + warnings + recent errors.

        Non-invasive (does not run generation). Combines everything a
        debugging script would check into one call:
            report = engine.diagnose()
            if report["status"] != "healthy":
                print(report["warnings"])
        """
        report = build_health_report(self)
        self.events.log(f"Diagnose: {report['status']}",
                        source="engine",
                        warnings=len(report.get("warnings", [])))
        return report

    # ── LoRA hot-loading ───────────────────────────────────────────────────

    def load_lora(self, lora_path: str, rank: int = 32, alpha: int | None = None,
                  target_modules: list[str] | None = None) -> int:
        """Hot-load a LoRA adapter onto the model.

        Attaches LoRA adapters to the specified modules and loads weights from
        a safetensors checkpoint. The base model weights are frozen — only the
        LoRA adapters are trainable. This enables dynamic skill injection at
        runtime (e.g. loading a tool-calling LoRA for self-play, then swapping
        it for a code-generation LoRA).

        Args:
            lora_path: Path to LoRA adapter safetensors file.
            rank: LoRA rank (must match the checkpoint).
            alpha: LoRA alpha (scale = alpha / rank). If None, uses rank * 2.
            target_modules: Module name substrings to attach LoRA to.
                If None, defaults to FFN+attention modules:
                ["w_gate", "w_up", "w_down", "q_proj", "v_proj",
                 "out_proj", "in_proj"]
                Pass ["w_gate", "w_up", "w_down"] for FFN-only.

        Returns:
            Number of LoRA parameters loaded.

        Raises:
            FileNotFoundError: If lora_path doesn't exist.
            ValueError: If LoRA checkpoint doesn't match model structure.
        """
        from pathlib import Path
        from safetensors.torch import load_file as _load
        from research.training.bitnet_lora import add_lora_adapters

        if alpha is None:
            alpha = rank * 2
        if target_modules is None:
            target_modules = ["w_gate", "w_up", "w_down", "q_proj", "v_proj",
                              "out_proj", "in_proj"]

        lora_file = Path(lora_path)
        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

        # Remove any existing LoRA adapters first
        self.unload_lora()

        self._log(f"Loading LoRA adapter: {lora_file.name} (rank={rank})")
        n_adapters, _ = add_lora_adapters(
            self.model, rank=rank, alpha=alpha,
            target_modules=target_modules)

        state = _load(str(lora_file))
        loaded = 0
        for name, param in self.model.named_parameters():
            if "lora_" in name and name in state:
                param.data.copy_(state[name])
                loaded += 1

        if loaded < len(state):
            missing = set(state.keys()) - {
                n for n, _ in self.model.named_parameters() if "lora_" in n}
            self._log(f"WARNING: {len(state) - loaded} LoRA tensors not found "
                      f"in model (checkpoint has {len(state)}, loaded {loaded}). "
                      f"Missing: {list(missing)[:5]}...",
                      level="warning")

        n_params = sum(v.numel() for v in state.values())
        self._lora_config = {
            "path": str(lora_file), "rank": rank, "alpha": alpha,
            "target_modules": target_modules, "n_adapters": n_adapters,
            "n_params": n_params,
        }
        self._log(f"LoRA loaded: {loaded}/{len(state)} tensors, "
                  f"{n_adapters} adapters, {n_params / 1e6:.1f}M params")
        return loaded

    def unload_lora(self) -> bool:
        """Remove LoRA adapters from the model.

        Detaches all LoRA modules and restores the original forward functions.
        The base model weights are unaffected.

        Returns:
            True if any LoRA adapters were removed, False if none were attached.
        """
        removed = 0
        for name, module in self.model.named_modules():
            # IRIFP4Linear: lora_adapter is a submodule attribute
            if hasattr(module, 'lora_adapter') and module.lora_adapter is not None:
                del module.lora_adapter
                removed += 1
            # nn.Linear with monkey-patched forward: restore original
            if hasattr(module, '_lora_orig_forward'):
                module.forward = module._lora_orig_forward
                del module._lora_orig_forward
                # Also remove the LoRA adapter module if present
                if hasattr(module, 'lora_adapter'):
                    del module.lora_adapter
                removed += 1

        if removed > 0:
            self._log(f"LoRA unloaded: {removed} adapters removed")
            self._lora_config = None
        return removed > 0

    def has_lora(self) -> bool:
        """Check if LoRA adapters are currently loaded."""
        return getattr(self, '_lora_config', None) is not None

    def lora_info(self) -> dict | None:
        """Return info about the currently loaded LoRA, or None."""
        return getattr(self, '_lora_config', None)

    # ── Sleep / wake ──────────────────────────────────────────────────────

    def sleep(self, level: int = 1) -> None:
        """Release GPU memory by offloading model weights.

        Level 1 (default): Move weights to CPU RAM. Fast wake (~2-3s).
            Preserves tokenizer, config, KV cache strategies, and CUDA context.
        Level 2: Discard weights entirely. Slower wake (reload from disk).
            Use for model switching when Level 1 CPU RAM is insufficient.

        After sleep, generation will fail until wake() is called.
        """
        if level not in (1, 2):
            raise ConfigurationError(
                f"sleep level must be 1 or 2, got {level}")
        if not self._awake and level <= self._sleep_level:
            return  # Already asleep
        if level == 2 and not self.checkpoint_path:
            raise RuntimeError("Sleep level 2 requires a checkpoint path")

        if level == 1:
            self.model.to("cpu", non_blocking=True)
            self._clear_cuda_cache()
            self._awake = False
            self._sleep_level = 1
            self._log("Sleep level 1: weights offloaded to CPU")
            return

        # Store minimal state, discard model
        self._stored_config = getattr(self.model, "config", None)
        self._stored_dtype = self.dtype
        self._stored_checkpoint = self.checkpoint_path
        self._profiler.model = None
        # Release acceleration resources that hold CUDA memory/graphs.
        # Without this, sleep(level=2) leaks CUDA graph + megakernel memory.
        self._release_acceleration_resources()
        self.model = None
        self._clear_cuda_cache()
        self._awake = False
        self._sleep_level = 2
        if self.device.type == "cuda":
            free, _ = self._memory_info(self.device)
            self._log(f"Sleep level 2: weights discarded, "
                      f"{free / 1e9:.1f}GB free")
        else:
            self._log("Sleep level 2: weights discarded")

    def wake(self) -> None:
        """Restore model to GPU and resume inference.

        Level 1 wake: CPU→GPU copy (~2-3s). Preserves all strategies.
        Level 2 wake: Reload from checkpoint (~5-10s). Strategies must be re-activated.
        """
        if self._awake:
            return  # Already awake

        if self._sleep_level == 1:
            self.model.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._awake = True
            self._sleep_level = 0
            self._log("Woke from level 1 sleep")
            return

        if not getattr(self, "_stored_checkpoint", None):
            raise CheckpointError(
                "Level 2 wake requires stored checkpoint path",
                suggestion="Load from checkpoint again with from_checkpoint().")
        from research.model_loader import ModelLoader
        self.model = ModelLoader.build_model_fast(
            self._stored_config, checkpoint_path=self._stored_checkpoint,
            dtype=self._stored_dtype)
        self.model.to(self.device)
        self.model.eval()
        self._profiler.model = self.model
        del self._stored_config
        del self._stored_dtype
        del self._stored_checkpoint
        self._awake = True
        self._sleep_level = 0
        self._log("Woke from level 2 sleep (reloaded from checkpoint)")

        # Re-activate strategies that were lost when the model was discarded.
        params = getattr(self, "_last_activation_params", None)
        if params:
            self._awake = True
            config = ActivationConfig.from_kwargs(**params)
            self.activate_config(config)
            self._log("Re-activated strategies after level 2 wake")

    @property
    def is_awake(self) -> bool:
        return self._awake

    def vram_usage(self) -> dict:
        """Report current VRAM usage for this engine."""
        if self.device.type != "cuda":
            return {
                "total_gb": 0,
                "free_gb": 0,
                "used_gb": 0,
                "model_weights_gb": 0,
                "percent": 0,
            }
        free, total = self._memory_info(self.device)
        used = total - free
        model_bytes = 0
        if self.is_awake and self.model is not None:
            model_bytes = sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
                if parameter.device.type == "cuda"
            )
        return {
            "total_gb": total / 1e9,
            "free_gb": free / 1e9,
            "used_gb": used / 1e9,
            "model_weights_gb": model_bytes / 1e9,
            "percent": used / total * 100,
        }

    def recover(self) -> dict | None:
        """Recover state from disk after a crash or restart.

        Returns a dict with recovered data:
          - ``generation``: last partial generation (prompt + output + token count)
          - ``event_log``: event log history
          - ``output_history``: past generation outputs
          - ``kv_snapshot``: KV cache state (if available)

        Returns None if no recovery data found.
        """
        return self._recovery.recover()

    def clear_recovery(self) -> None:
        """Remove all crash recovery files from disk."""
        self._recovery.clear()

    # ── Session-aware generation ────────────────────────────────────────

    def begin_session(self, session_id: str, ttl: float | None = None) -> None:
        """Start a new generation session with persistent KV cache.

        Sessions maintain KV cache state across multiple ``continue_session``
        calls, enabling O(Δt) per-turn cost instead of O(n) re-prefill.

        Args:
            session_id: unique identifier for this conversation/session.
            ttl: time-to-live in seconds. If set, the session's KV cache
                is auto-evicted after this many seconds of inactivity.
                None = no TTL (persists until end_session or LRU eviction).
        """
        return self._session_cache.begin_session(session_id, ttl)

    def continue_session(self, session_id: str, prompt: str,
                         max_new_tokens: int = 100,
                         temperature: float = 0.0,
                         top_p: float = 1.0,
                         top_k: int = 80,
                         repetition_penalty: float = 1.05) -> str:
        """Continue a session with a new prompt — reuses cached KV.

        Only the delta (new tokens not in the session's cached prefix)
        needs prefilling. Previous turns' KV is reused as-is.

        Args:
            session_id: must match a session started with ``begin_session``.
            prompt: new user input (appended to session history).
            max_new_tokens, temperature, top_p, top_k, repetition_penalty:
                same as ``generate()``.

        Returns:
            Generated text string.
        """
        self._require_awake()
        self._validate_generation_params(
            prompt, max_new_tokens, temperature, top_p, top_k,
            repetition_penalty)

        ids, past_kv, cached_len = self._session_cache.continue_session(
            session_id, prompt)

        # If we have cached KV, only prefill the delta
        generated_ids: list[int] = []
        if cached_len > 0 and past_kv is not None:
            delta_ids = ids[:, cached_len:]
            if delta_ids.shape[1] > 0:
                with torch.inference_mode():
                    out = self.model(
                        delta_ids, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)
            # Decode from current logits
            eos_set = self._eos_token_ids()
            for _ in self._decode_tokens(
                logits, past_kv, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, eos_set, generated_ids,
            ):
                pass
            result = self._safe_decode_ids(generated_ids)
        else:
            # No cache hit — full prefill + decode
            result = self._generate_with_oom_recovery(
                lambda: self._full_generate(
                    ids, max_new_tokens, temperature, top_p, top_k,
                    repetition_penalty))
            # _full_generate records its own token count; generated_ids stays empty here.

        # Update session KV state
        last_kv = getattr(self.model, '_forge_last_kv', None)
        if last_kv is not None:
            self._session_cache.update_session_kv(session_id, last_kv)

        self._record_generation(len(generated_ids))
        return result

    def pin_session(self, session_id: str, ttl: float | None = None) -> None:
        """Pin a session's KV cache to prevent eviction during tool calls.

        Args:
            session_id: the session to pin.
            ttl: optional TTL for the pin. If set, the session auto-evicts
                after this many seconds (useful for tool call timeouts).
        """
        self._session_cache.pin_session(session_id, ttl)

    def unpin_session(self, session_id: str) -> None:
        """Unpin a session (allow eviction again)."""
        self._session_cache.unpin_session(session_id)

    def end_session(self, session_id: str) -> None:
        """End a session and release its KV cache."""
        self._session_cache.end_session(session_id)

    def session_stats(self) -> dict:
        """Get session cache statistics."""
        return self._session_cache.stats()

    def _full_generate(self, ids, max_new_tokens, temperature, top_p, top_k,
                       repetition_penalty) -> str:
        """Full prefill + decode (no session cache)."""
        eos_set = self._eos_token_ids()
        generated_ids: list[int] = []
        logits, past_kv = self._prefill(ids)
        for _ in self._decode_tokens(
            logits, past_kv, max_new_tokens, temperature, top_p, top_k,
            repetition_penalty, eos_set, generated_ids,
        ):
            pass
        self._record_generation(len(generated_ids))
        return self._safe_decode_ids(generated_ids)
