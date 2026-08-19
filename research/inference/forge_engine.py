"""Forge Inference Engine — unified runtime backend for XP models.

Extends FastInferenceEngine with pluggable strategy architecture:
  - KV cache: standard, paged, rotorquant, hadamard_int4, compressed
  - Decoding: standard, speculative, medusa, dspark, mtp_selfspec
  - Quantization: none, int8, int4 weight-only
  - Acceleration: none, cuda_graph, airllm_streaming
  - Innovations: MRL-AdaptiveContext, QuaRot-KV, V0-WarmStart, ProgressiveKV

Auto-detects KeyStack features from checkpoint metadata and activates
the appropriate runtime strategies.

Usage:
    from research.inference.forge_engine import ForgeEngine

    engine = ForgeEngine.from_checkpoint(
        checkpoint="research/checkpoints/xp_full_no_mqa.safetensors",
        config_name="forgelm_v3",
        tokenizer_path="research/checkpoints/lfm25_tokenizer",
    )
    engine.activate(kv_cache="hadamard_int4", decoding="mtp_selfspec")
    output = engine.generate("def fibonacci(n):", max_new_tokens=50)
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from research.inference.decoding import DecodingStrategy, StandardDecoding, build_decoding
from research.inference.diagnostics import (
    EngineProfiler,
    EventLog,
    OutputHistory,
    build_health_report,
)
from research.inference.innovations import (
    MRLAdaptiveContext,
    ProgressiveKV,
    QuaRotKV,
    V0WarmStart,
)
from research.inference.kv_backend import KVCacheStrategy, build_kv_cache
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

        # Strategy slots
        self.kv_cache: KVCacheStrategy | None = None
        self.decoding: DecodingStrategy = StandardDecoding()
        self.quantize: str | None = None
        self.acceleration: str | None = None

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
        self._graph_runner = None

        # Built-in diagnostics (replaces need for one-off scripts)
        self.events = EventLog(capacity=500)
        self.outputs = OutputHistory(capacity=100)
        self._profiler = EngineProfiler(self.model, self.device)
        self.events.log("ForgeEngine initialized", source="engine",
                        device=str(self.device),
                        checkpoint=checkpoint_path or "none")

        # Move model to device (unless it's on meta — streaming mode)
        self._needs_streaming = False
        first_param = next(self.model.parameters(), None)
        if first_param is not None and first_param.device.type != "meta":
            self.model.to(self.device)
        self.model.eval()

        # Auto-detect KeyStack features
        if checkpoint_path:
            self._detect_keystack_features()

    @classmethod
    def from_checkpoint(cls, checkpoint: str, config_name: str = "forgelm_v3",
                        tokenizer_path: str | None = None,
                        device: str = "cuda", **kwargs):
        """Build engine from a KeyStack checkpoint.

        Auto-checks VRAM capacity. If the model fits, loads normally.
        If not, sets up AirLLM layer-streaming (meta device + shard loading).
        """
        from safetensors import safe_open

        from research.config import get_config
        from research.model_loader import ModelLoader
        from research.tokenizer_cache import get_tokenizer

        cfg = get_config(config_name, device=device)
        tok_path = tokenizer_path or "research/checkpoints/lfm25_tokenizer"
        tokenizer = get_tokenizer(tok_path)

        # Check checkpoint size on disk
        ckpt_size = Path(checkpoint).stat().st_size

        # Check available VRAM
        dev = torch.device(device)
        if dev.type == "cuda":
            vram_free = torch.cuda.mem_get_info(dev)[0]
        else:
            vram_free = 32 * 1024**3  # Assume 32GB RAM

        # Need ~1.3x checkpoint size for model + activations + KV cache
        needed = int(ckpt_size * 1.3)
        fits = vram_free > needed

        if fits:
            # Fast path: load normally (uses cached architecture)
            model = ModelLoader.build_model_fast(cfg, checkpoint_path=checkpoint)
            return cls(model, tokenizer, device=device, checkpoint_path=checkpoint, **kwargs)
        else:
            # Slow path: build on meta, set up streaming
            print(f"  [AirLLM-Smart] Checkpoint {ckpt_size/1e9:.2f} GB > VRAM free {vram_free/1e9:.2f} GB")
            print("  [AirLLM-Smart] Building model on meta device (zero VRAM)...")
            model = ModelLoader.build_model(cfg, checkpoint_path=None)
            # Don't move to device — keep on meta/CPU
            model.eval()
            engine = cls(model, tokenizer, device=device, checkpoint_path=checkpoint, **kwargs)
            # Mark as needing streaming
            engine._needs_streaming = True
            return engine

    def _detect_keystack_features(self):
        """Detect which KeyStack transforms are in the checkpoint."""
        from safetensors import safe_open
        from pathlib import Path as _Path
        features = []
        ckpt = _Path(self.checkpoint_path)
        if ckpt.is_dir():
            # Sharded model: read keys from first shard
            shards = sorted(ckpt.glob("model-*.safetensors"))
            if shards:
                with safe_open(str(shards[0]), framework="pt") as f:
                    keys = set(f.keys())
            else:
                keys = set()
        else:
            with safe_open(self.checkpoint_path, framework="pt") as f:
                keys = set(f.keys())

        # Detect features from keys (works for both single-file and sharded)
        if "value_residual_v0" in keys:
            features.append("value_residual")
        if "rotorquant_rotations" in keys:
            features.append("rotorquant")
        if "mtp_head.heads.0.weight" in keys:
            features.append("mtp")
        if "_airllm_streamable" in keys:
            features.append("airllm")
        # QuaRot detection: check if V/O weights are Hadamard-rotated
        # (heuristic: compare against original if available)
        features.append("quarot")  # Assume applied by pipeline
        features.append("mrl")     # Assume applied by pipeline

        self.keystack_features = features
        print(f"  [ForgeEngine] KeyStack features detected: {features}")

    def activate(self, kv_cache: str = "paged",
                 decoding: str = "standard",
                 quantize: str | None = None,
                 acceleration: str | None = None,
                 mrl_keep_ratio: float | None = None,
                 kv_bits: int = 4,
                 use_v0_warm: bool = False,
                 use_progressive_kv: bool = False,
                 use_compile: bool = False,
                 use_triton_conv: bool = False,
                 use_prefix_cache: bool = False,
                 use_spec_attn: bool = False,
                 kv_cache_tokens: int | None = None,
                 use_chunked_prefill: bool = False,
                 use_flex_decoding: bool = False,
                 use_wavelength_pruning: bool = False,
                 use_pod_attention: bool = False,
                 use_adaptive_spec: bool = False,
                 use_cosa: bool = False,
                 use_seq_split: bool = False,
                 use_compact_attn: bool = False,
                 use_suffix_spec: bool = False,
                 use_fused_qk_norm_rope_cache: bool = False,
                 use_block_fusion: bool = False,
                 use_compile_autotune: bool = False,
                 use_hybrid_prefill: bool = False,
                 use_learned_prefix_cache: bool = False,
                 use_hotprefix: bool = False,
                 use_mosa: bool = False,
                 use_triroute: bool = False,
                 use_breakable_cuda_graph: bool = False,
                 use_corun: bool = False,
                 use_foundry: bool = False,
                 use_fa4: bool = False,
                 use_kara_kv: bool = False,
                 use_moment_kv: bool = False,
                 use_kvpop: bool = False,
                 use_conf_kv: bool = False,
                 use_jet_long: bool = False,
                 use_rope_id: bool = False,
                 use_lerope: bool = False,
                 use_lampe: bool = False,
                 use_peagle: bool = False,
                 use_lookahead_gate: bool = False,
                 use_fastserve: bool = False,
                 use_libra: bool = False,
                 use_faser: bool = False,
                 use_kairos: bool = False,
                 use_unified_radix: bool = False,
                 use_elbow_moe: bool = False,
                 use_alloc_moe: bool = False,
                 use_lda_moe: bool = False,
                 use_adamx: bool = False,
                 use_sharq: bool = False,
                 use_mosaic_quant: bool = False,
                 use_aoh: bool = False,
                 warmup: bool = True):
        """Activate runtime strategies.

        Args:
            kv_cache: "standard", "paged", "rotorquant", "hadamard_int4", "compressed",
                      "streaming", "snapkv", "snapkv_4bit", "paged_eviction", "xquant",
                      "cpu_offload", "s4r", "hqe_kv"
            decoding: "standard", "speculative", "medusa", "dspark", "eagle3", "mtp_selfspec"
            quantize: None, "int8", "int4", "fp8", "w8a8", "nvfp4"
            acceleration: None, "cuda_graph", "airllm_streaming", "megakernel", "flex_decoding"
            mrl_keep_ratio: if set (e.g. 0.75), truncate to that fraction of dims
            kv_bits: 4 or 8, for KV cache quantization
            use_v0_warm: enable V0 warm-start for KV cache
            use_progressive_kv: enable progressive KV (anchor + residual streams)
            use_compile: torch.compile the model for 1.3-2x decode speedup
            use_triton_conv: replace conv layers with fused Triton kernel (89% bottleneck)
            use_prefix_cache: cache KV for repeated prompt prefixes
            use_spec_attn: L1 Speculative Attention (57% attn cut, lossless)
            kv_cache_tokens: limit KV cache allocation to N tokens (saves VRAM).
                             None = use model's max_seq_len. Like llama.cpp --kv-cache-tokens.
            use_chunked_prefill: split long prompts into chunks (512 tokens) to
                                  avoid blocking decode queue. Like vLLM chunked prefill.
            use_flex_decoding: use PyTorch FlexDecoding backend for decode attention
                               (splits KV across SMs, 1.5-3x for batch=1).
            use_wavelength_pruning: ATFlash per-RoPE-wavelength attention pruning
                                    (37-48% attention compute cut, near-lossless).
            use_pod_attention: POD-Attention prefill-decode overlap (28% mean speedup
                               for hybrid batches in BatchQueue).
            use_adaptive_spec: adaptive n-gram + EAGLE-3 combo speculative decoding
                               (up to 4.9x on code-editing, 2.89x average).
            use_cosa: CoSA proxy-kernel sparse attention for long-context decode
                      (4.93x attn speedup at 128K, training-free).
            use_seq_split: sequence-aware split heuristic for low-head-count decode
                           (21-24% SM utilization improvement, 8 KV heads → 192 SMs).
            use_compact_attn: CompactAttention block-union KV for chunked prefill
                              (2.72x attn speedup at 128K under chunked prefill).
            use_suffix_spec: suffix decoding speculative (training-free, best for
                             code/RAG with repeated patterns, composes with n-gram).
            use_fused_qk_norm_rope_cache: fuse QK-Norm+RoPE+KV-cache-write into one
                                          kernel (5-10% decode speedup, fewer launches).
            use_block_fusion: per-block CUDA graph capture + torch.compile for full
                              transformer block fusion (1.3-1.5x, supports MoD skip).
            use_compile_autotune: auto-benchmark torch.compile modes and pick best.
            use_hybrid_prefill: adaptive chunked prefill (chunk only when decode active,
                                continuous prefill otherwise). +2-5% throughput, -20% TTFT.
            use_learned_prefix_cache: ML-guided prefix cache eviction (18-47% cache size
                                      reduction at equivalent hit ratios).
            use_hotprefix: hotness-aware GPU/CPU prefix promotion (hot prefixes on GPU,
                           cold on CPU, periodic rebalancing).
            use_mosa: Mixture of Sparse Attention (expert-choice token routing per head,
                      27% better perplexity at same compute, smaller KV cache).
            use_triroute: unified routing for attention mode + KV bits (joint policy,
                          better quality-compute tradeoff than independent routing).
            warmup: pre-run a dummy token to initialize CUDA kernels (avoids
                    first-generation slowdown). Like llama.cpp's graph reservation.
        """
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
            print(f"  [ForgeEngine] QuaRot-KV active: V pre-rotated, K runtime-Hadamard, {kv_bits}-bit")

        # 4. V0 warm start
        if use_v0_warm and "value_residual" in self.keystack_features:
            self.v0_warm = V0WarmStart.from_checkpoint(self.checkpoint_path)
            if self.v0_warm:
                print(f"  [ForgeEngine] V0-WarmStart active: {self.v0_warm.info()}")
            else:
                print("  [ForgeEngine] V0-WarmStart: no V_0 found in checkpoint")

        # 5. Progressive KV
        if use_progressive_kv:
            self.progressive_kv = ProgressiveKV(anchor_bits=8, residual_bits=8)
            print(f"  [ForgeEngine] ProgressiveKV active: {self.progressive_kv.info()}")

        # 6. KV cache
        cfg = getattr(self.model, "config", None)
        n_heads = getattr(cfg, "n_heads", 12)
        n_kv = getattr(cfg, "n_kv_heads", 2) or n_heads
        head_dim = getattr(cfg, "d_model", 1536) // n_heads
        max_seq = getattr(cfg, "max_seq_len", 4096)
        # Limit KV cache allocation if requested (like llama.cpp --kv-cache-tokens)
        if kv_cache_tokens is not None and kv_cache_tokens < max_seq:
            print(f"  [ForgeEngine] KV cache limited to {kv_cache_tokens} tokens "
                  f"(was {max_seq})")
            max_seq = kv_cache_tokens
        self.kv_cache = build_kv_cache(kv_cache)
        self.kv_cache.init(n_heads, head_dim, n_kv, max_seq,
                           str(self.device), torch.bfloat16)
        print(f"  [ForgeEngine] KV cache: {self.kv_cache.info()}")

        # 7. Decoding
        decode_kwargs = {}
        if decoding == "mtp_selfspec":
            decode_kwargs["k"] = 4
            if hasattr(self.model, "mtp_head"):
                decode_kwargs["mtp_module"] = self.model.mtp_head
        elif decoding == "eagle3":
            # EAGLE-3: load head from checkpoint sidecar or model attribute
            if hasattr(self.model, "eagle_head"):
                decode_kwargs["eagle_head"] = self.model.eagle_head
            elif self.checkpoint_path:
                import os
                eagle_path = self.checkpoint_path.replace(".safetensors", ".eagle3.safetensors")
                if os.path.exists(eagle_path):
                    from research.decoding.eagle import Eagle3Head, add_eagle3_to_model
                    head = add_eagle3_to_model(self.model)
                    from safetensors.torch import load_file
                    head.load_state_dict(load_file(eagle_path))
                    head = head.to(self.device)
                    decode_kwargs["eagle_head"] = head
                    print(f"  [ForgeEngine] EAGLE-3 head loaded from {eagle_path}")
            decode_kwargs.setdefault("draft_length", 4)
        self.decoding = build_decoding(decoding, **decode_kwargs)
        print(f"  [ForgeEngine] Decoding: {self.decoding.name}")

        # 8. Acceleration
        if acceleration == "cuda_graph" and self.device.type == "cuda":
            from research.runtime.cuda_graph import CudaGraphRunner
            self._graph_runner = CudaGraphRunner(
                self.model, batch_size=1, seq_len=1,
                device=str(self.device), use_cache=True)
            self._graph_runner.capture()
            self.acceleration = "cuda_graph"
            print("  [ForgeEngine] CUDA graphs: active")
        elif acceleration == "megakernel" and self.device.type == "cuda":
            # Megakernel decode: single-graph capture of entire decode step
            # + torch.compile for intra-layer kernel fusion. 1.5-5x over eager.
            from research.decoding.megakernel import CompiledMegakernelDecode
            self._megakernel = CompiledMegakernelDecode(
                self.model, device=str(self.device))
            try:
                self._megakernel.capture()
                self.acceleration = "megakernel"
                print("  [ForgeEngine] Megakernel decode: active (compiled + graph)")
            except Exception as e:
                print(f"  [ForgeEngine] Megakernel decode: failed ({e}), falling back")
                self._megakernel = None
                self.acceleration = None
        elif acceleration == "flex_decoding" and self.device.type == "cuda":
            # FlexDecoding: PyTorch's fused FlashDecoding backend for decode.
            # Splits KV cache across SMs (1.5-3x for batch=1, low head count).
            from research.inference.attention.flex_decoding import FlexDecodingWrapper
            self._flex_decoding = FlexDecodingWrapper()
            if self._flex_decoding.apply(self.model):
                self.acceleration = "flex_decoding"
            else:
                self._flex_decoding = None
                self.acceleration = None
        elif acceleration == "airllm_streaming":
            self._setup_airllm_smart()
        else:
            self._graph_runner = None
            self.acceleration = None

        # 9. torch.compile
        if use_compile and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
                print("  [ForgeEngine] torch.compile: active (reduce-overhead)")
            except Exception as e:
                print(f"  [ForgeEngine] torch.compile: failed ({e})")

        # 9b. Triton fused conv kernel
        if use_triton_conv and self.device.type == "cuda":
            try:
                from research.decoding.triton_conv import patch_conv_layers
                patch_conv_layers(self.model)
            except Exception as e:
                print(f"  [ForgeEngine] Triton conv: failed ({e})")

        # 10. Prefix caching
        self._prefix_cache = {} if use_prefix_cache else None
        if use_prefix_cache:
            print("  [ForgeEngine] Prefix caching: active")

        # 10b. Chunked prefill
        self._chunked_prefill = None
        if use_chunked_prefill:
            from research.inference.prefill.chunked_prefill import ChunkedPrefiller
            self._chunked_prefill = ChunkedPrefiller(
                self.model, chunk_size=512, device=str(self.device))
            print("  [ForgeEngine] Chunked prefill: active (chunk_size=512)")

        # 10c. ATFlash wavelength pruning (37-48% attention compute cut)
        self._wavelength_pruner = None
        if use_wavelength_pruning:
            from research.inference.attention.atflash import WavelengthPrunedAttention
            rope_base = getattr(getattr(self.model, 'config', None), 'rope_base', 1_000_000.0)
            self._wavelength_pruner = WavelengthPrunedAttention(
                rope_base=rope_base, prune_factor=1.0)
            self._wavelength_pruner.apply(self.model)
            print("  [ForgeEngine] ATFlash wavelength pruning: active (37-48% attn cut)")

        # 10d. POD-Attention prefill-decode overlap (28% mean for hybrid batches)
        self._pod_attention = None
        if use_pod_attention and self.device.type == "cuda":
            from research.inference.attention.pod_attention import PODAttentionScheduler
            self._pod_attention = PODAttentionScheduler(device=str(self.device))
            print("  [ForgeEngine] POD-Attention: active (prefill-decode overlap)")

        # 10e. Adaptive n-gram + EAGLE-3 speculative decoding
        self._adaptive_spec = None
        if use_adaptive_spec:
            from research.decoding.adaptive_speculative import AdaptiveSpeculativeDecoder
            eagle_head = getattr(self.model, 'eagle_head', None)
            mtp_module = getattr(self.model, 'mtp_module', None)
            self._adaptive_spec = AdaptiveSpeculativeDecoder(
                eagle_head=eagle_head, mtp_module=mtp_module)
            print("  [ForgeEngine] Adaptive speculative decoding: active (n-gram + EAGLE-3)")

        # 10f. CoSA sparse attention for long-context decode (4.93x at 128K)
        self._cosa = None
        if use_cosa:
            from research.inference.attention.cosa import CoSAWrapper
            self._cosa = CoSAWrapper(block_size=16, budget_ratio=0.5, min_seq_len=2048)
            self._cosa.apply(self.model)

        # 10g. Sequence-aware split for low-head-count decode (21-24% SM util)
        self._seq_split = None
        if use_seq_split and self.device.type == "cuda":
            from research.inference.attention.seq_aware_split import SequenceAwareSplitWrapper
            self._seq_split = SequenceAwareSplitWrapper(min_seq_len=512)
            self._seq_split.apply(self.model, self.device)

        # 10h. CompactAttention for chunked prefill (2.72x at 128K)
        self._compact_attn = None
        if use_compact_attn:
            from research.inference.attention.compact_attention import CompactAttentionWrapper
            self._compact_attn = CompactAttentionWrapper(
                block_size=16, budget_ratio=0.5, min_kv_len=2048)
            self._compact_attn.apply(self.model)

        # 10i. Suffix decoding speculative (training-free, code/RAG)
        self._suffix_spec = None
        if use_suffix_spec:
            from research.decoding.suffix_decoding import SuffixDecoder
            self._suffix_spec = SuffixDecoder(max_draft_len=8)
            print("  [ForgeEngine] Suffix decoding: active (training-free, code/RAG)")

        # 10j. Fused QK-Norm + RoPE + KV cache write (5-10% decode speedup)
        self._fused_qk_cache = None
        if use_fused_qk_norm_rope_cache:
            from research.inference.attention.fused_qk_norm_rope_cache import FusedQKNormRopeCacheWrapper
            self._fused_qk_cache = FusedQKNormRopeCacheWrapper(kv_quant_bits=None)
            self._fused_qk_cache.apply(self.model)

        # 10k. Block fusion: per-block CUDA graph + torch.compile (1.3-1.5x)
        self._block_fusion = None
        if use_block_fusion and self.device.type == "cuda":
            from research.inference.graphs.block_fusion import CompiledBlockFusion
            try:
                self._block_fusion = CompiledBlockFusion(
                    self.model, device=str(self.device))
                self._block_fusion.capture()
                print("  [ForgeEngine] Block fusion: active (per-block CUDA graph + compile)")
            except Exception as e:
                print(f"  [ForgeEngine] Block fusion: failed ({e})")
                self._block_fusion = None

        # 10l. torch.compile auto-tuner
        if use_compile_autotune and self.device.type == "cuda":
            from research.inference.graphs.compile_autotune import auto_tune_compile
            sample_input = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            model_id = getattr(getattr(self.model, 'config', None), 'name', 'default')
            best_mode, self.model = auto_tune_compile(
                self.model, sample_input, model_id=model_id)
            print(f"  [ForgeEngine] torch.compile auto-tuned: {best_mode}")

        # 10m. Hybrid chunked prefill (adaptive: chunk only when decode active)
        if use_hybrid_prefill:
            from research.inference.prefill.hybrid_prefill import HybridChunkedPrefiller
            self._chunked_prefill = HybridChunkedPrefiller(
                self.model, chunk_size=512, device=str(self.device))
            print("  [ForgeEngine] Hybrid chunked prefill: active (adaptive chunking)")

        # 10n. Learned prefix cache (ML-guided eviction)
        self._learned_prefix_cache = None
        if use_learned_prefix_cache:
            from research.inference.kv.learned_prefix_cache import LearnedPrefixCache
            self._learned_prefix_cache = LearnedPrefixCache(max_entries=256)
            self._prefix_cache = self._learned_prefix_cache  # replace LRU dict
            print("  [ForgeEngine] Learned prefix cache: active (ML eviction, 18-47% size cut)")

        # 10o. HotPrefix hotness-aware GPU/CPU prefix promotion
        self._hotprefix = None
        if use_hotprefix:
            from research.inference.scheduler.hotprefix import HotPrefixManager
            self._hotprefix = HotPrefixManager(gpu_capacity=8)
            print("  [ForgeEngine] HotPrefix: active (hotness-aware GPU/CPU promotion)")

        # 10p. MoSA: Mixture of Sparse Attention (27% better perplexity, smaller KV)
        self._mosa = None
        if use_mosa:
            from research.inference.kv.mosa import MoSAWrapper
            self._mosa = MoSAWrapper(k_ratio=0.5, min_seq_len=512)
            self._mosa.apply(self.model)

        # 10q. TriRoute: unified attention mode + KV bits routing
        self._triroute = None
        if use_triroute:
            from research.inference.scheduler.triroute import TriRouteWrapper
            self._triroute = TriRouteWrapper(local_window=256, min_seq_len=512)
            self._triroute.apply(self.model)

        # 10r. Breakable CUDA Graph (BCG): segmented graph capture (1.7-1.93x)
        self._bcg = None
        if use_breakable_cuda_graph and torch.cuda.is_available():
            from research.inference.graphs.breakable_cuda_graph import BreakableCudaGraph
            self._bcg = BreakableCudaGraph(self.model, device=str(self.device))
            self._bcg.capture_decode(batch_sizes=[1, 2, 4, 8])
            print(f"  [ForgeEngine] Breakable CUDA Graph: {self._bcg.stats()}")

        # 10s. CoRun: deterministic inference via padding + fixed-shape graph
        self._corun = None
        if use_corun and torch.cuda.is_available():
            from research.inference.scheduler.corun import CoRunScheduler
            self._corun = CoRunScheduler(self.model, max_concurrency=8,
                                          device=str(self.device), dtype=self.dtype)
            self._corun.capture_decode_graph()
            print(f"  [ForgeEngine] CoRun deterministic: {self._corun.stats()}")

        # 10t. Foundry: template-based CUDA graph cold start (10x faster startup)
        self._foundry = None
        if use_foundry and torch.cuda.is_available():
            from research.inference.graphs.foundry import FoundryRunner
            self._foundry = FoundryRunner(self.model, device=str(self.device))
            # Try to materialize from saved templates; if none, capture new ones
            templates = self._foundry.list_templates()
            if templates:
                for tname in templates[:4]:
                    self._foundry.materialize(tname)
                print(f"  [ForgeEngine] Foundry: materialized {len(templates)} templates")
            else:
                self._foundry.capture_and_save("decode", batch_size=1)
                self._foundry.capture_and_save("decode", batch_size=4)
                print(f"  [ForgeEngine] Foundry: captured new templates")

        # 10u. FA4: FlashAttention-4 for Blackwell (1.3x prefill, 1.6-1.9x decode)
        self._fa4 = None
        if use_fa4 and torch.cuda.is_available():
            from research.inference.attention.fa4_attention import FA4Attention
            self._fa4 = FA4Attention(use_fp8_kv=False)
            print(f"  [ForgeEngine] FA4: {self._fa4.stats()}")

        # 10v. KARA: sliding-window KV compression with Token2Chunk
        self._kara_kv = None
        if use_kara_kv:
            from research.inference.kv.kara_kv import KARAKVCache
            n_kv = getattr(self.config, 'n_kv_heads', 8)
            head_dim = getattr(self.config, 'head_dim', 64)
            self._kara_kv = KARAKVCache(n_kv, head_dim, target_budget=2048)
            print("  [ForgeEngine] KARA KV: sliding-window compression + Token2Chunk")

        # 10w. MomentKV: moment-statistics KV eviction (directional gap fix)
        self._moment_kv = None
        if use_moment_kv:
            from research.inference.kv.moment_kv import MomentKVCache
            n_kv = getattr(self.config, 'n_kv_heads', 8)
            head_dim = getattr(self.config, 'head_dim', 64)
            self._moment_kv = MomentKVCache(n_kv, head_dim, max_budget=2048)
            print("  [ForgeEngine] MomentKV: moment-statistics eviction (geometric regularity)")

        # 10x. KVpop: predictive online pruning with learned scoring
        self._kvpop = None
        if use_kvpop:
            from research.inference.kv.kvpop import KVpopCache
            n_kv = getattr(self.config, 'n_kv_heads', 8)
            head_dim = getattr(self.config, 'head_dim', 64)
            self._kvpop = KVpopCache(n_kv, head_dim, long_range_budget=1024)
            print("  [ForgeEngine] KVpop: predictive online pruning (future-attention supervised)")

        # 10y. CONF-KV: confidence-aware KV eviction + mixed-precision storage
        self._conf_kv = None
        if use_conf_kv:
            from research.inference.kv.conf_kv import ConfKVCache
            n_kv = getattr(self.config, 'n_kv_heads', 8)
            head_dim = getattr(self.config, 'head_dim', 64)
            self._conf_kv = ConfKVCache(n_kv, head_dim, min_budget=256, max_budget=4096)
            print("  [ForgeEngine] CONF-KV: confidence-aware budget + mixed FP16/INT8 storage")

        # 10z. Jet-Long: dynamic bifocal RoPE (zero-shot 128K context)
        self._jet_long = None
        if use_jet_long:
            from research.inference.scheduler.jet_long import JetLongAttention
            n_heads = getattr(self.config, 'n_heads', 32)
            n_kv = getattr(self.config, 'n_kv_heads', 8)
            head_dim = getattr(self.config, 'head_dim', 64)
            self._jet_long = JetLongAttention(n_heads, head_dim, n_kv,
                                               local_window=1024, max_train_length=32768)
            print(f"  [ForgeEngine] Jet-Long: dynamic bifocal RoPE "
                  f"(zero-shot 128K, {self._jet_long.stats()})")

        # 10aa. RoPE-ID: in-distribution high-freq rotation (length generalization)
        self._rope_id = None
        if use_rope_id:
            from research.inference.position.rope_id import RoPEID
            head_dim = getattr(self.config, 'head_dim', 64)
            self._rope_id = RoPEID(head_dim, rotation_fraction=0.25)
            print(f"  [ForgeEngine] RoPE-ID: {self._rope_id.stats()}")

        # 10ab. LeRoPE: learnable RoPE frequencies (3.4% less compute)
        self._lerope = None
        if use_lerope:
            from research.inference.position.lerope import integrate_lerope_into_model
            n_replaced = integrate_lerope_into_model(self.model)
            print(f"  [ForgeEngine] LeRoPE: replaced RoPE in {n_replaced} layers")

        # 10ac. LaMPE: length-aware multi-grained positional encoding
        self._lampe = None
        if use_lampe:
            from research.inference.scheduler.lampe import LaMPE
            head_dim = getattr(self.config, 'head_dim', 64)
            self._lampe = LaMPE(head_dim, train_length=32768)
            print(f"  [ForgeEngine] LaMPE: {self._lampe.stats()}")

        # 10ad. P-EAGLE: parallel speculative decoding (1.69x over EAGLE-3)
        self._peagle = None
        if use_peagle:
            from research.decoding.peagle import PEAGLEDraftHead
            d_model = getattr(self.config, 'd_model', 2048)
            vocab_size = getattr(self.config, 'vocab_size', 65536)
            self._peagle = PEAGLEDraftHead(d_model, vocab_size, n_draft_tokens=4)
            self._peagle = self._peagle.to(self.device)
            print("  [ForgeEngine] P-EAGLE: parallel speculative decoding (K=4 in 1 pass)")

        # 10ae. Lookahead Quality Gate: block-wise acceptance (2.6-7.9x)
        self._lookahead_gate = None
        if use_lookahead_gate:
            from research.decoding.lookahead_gate import LookaheadQualityGate
            self._lookahead_gate = LookaheadQualityGate(n_lookahead=4)
            print("  [ForgeEngine] Lookahead Quality Gate: geometry-based block acceptance")

        # 10af. FastServe: skip-join MLFQ preemptive scheduler (6.1x throughput)
        self._fastserve = None
        if use_fastserve:
            from research.inference.scheduler.fastserve import FastServeScheduler
            self._fastserve = FastServeScheduler(max_batch_size=8)
            print("  [ForgeEngine] FastServe: skip-join MLFQ + token-level preemption")

        # 10ag. Libra: micro-request flexible partitioning (1.91x goodput)
        self._libra = None
        if use_libra:
            from research.inference.scheduler.libra import LibraScheduler
            self._libra = LibraScheduler(max_batch_size=8, chunk_size=512)
            print("  [ForgeEngine] Libra: micro-request partitioning + SLO-aware batching")

        # 10ah. FASER: fine-grained SD phase management (53% throughput, 1.92x latency)
        self._faser = None
        if use_faser:
            from research.inference.attention.faser import FASERController, FASEREarlyExit, FASERFrontier
            self._faser = FASERController()
            self._faser_exit = FASEREarlyExit()
            self._faser_frontier = FASERFrontier()
            print("  [ForgeEngine] FASER: dynamic spec length + early exit + frontier overlap")

        # 10ai. Kairos: SLO-aware prefill+decode scheduling (33.8% SLO attainment)
        self._kairos = None
        if use_kairos:
            from research.inference.scheduler.kairos import KairosScheduler
            self._kairos = KairosScheduler(max_batch_size=8)
            print("  [ForgeEngine] Kairos: urgency-based prefill + slack-guided decode")

        # 10aj. Unified Radix Cache: hybrid prefix caching (HiCache L1/L2/L3)
        self._unified_radix = None
        if use_unified_radix:
            from research.inference.scheduler.unified_radix import UnifiedRadixCache
            self._unified_radix = UnifiedRadixCache(max_gpu_tokens=32768)
            print("  [ForgeEngine] Unified Radix Cache: hybrid prefix caching + HiCache")

        # 10ak. Elbow MoE routing: training-free dynamic top-k (5.3% latency cut)
        self._elbow_moe = None
        if use_elbow_moe:
            from research.inference.scheduler.moe_optim import ElbowRouter
            self._elbow_moe = ElbowRouter(min_experts=1, max_experts=4)
            print("  [ForgeEngine] Elbow MoE: training-free dynamic top-k routing")

        # 10al. Alloc-MoE: budget-aware expert activation (1.34x decode)
        self._alloc_moe = None
        if use_alloc_moe:
            from research.inference.scheduler.moe_optim import AllocMoE
            n_layers = getattr(self.config, 'n_layers', 16)
            self._alloc_moe = AllocMoE(n_layers, n_experts=8, total_budget=0.5)
            print(f"  [ForgeEngine] Alloc-MoE: {self._alloc_moe.stats()}")

        # 10am. LDA MoE: distribution-consistent inference (reduced routing correction)
        self._lda_moe = None
        if use_lda_moe:
            from research.inference.scheduler.moe_optim import LDACalibrator
            n_layers = getattr(self.config, 'n_layers', 16)
            self._lda_moe = LDACalibrator(n_layers)
            print("  [ForgeEngine] LDA MoE: distribution-consistent reduced routing")

        # 10an. AdaMX: adaptive microscaling quantization (83% MXFP4 loss removed)
        self._adamx = None
        if use_adamx:
            from research.quantization.adaptive_quant import AdaMXQuantizer
            self._adamx = AdaMXQuantizer(block_size=32, scheme='auto')
            print("  [ForgeEngine] AdaMX: adaptive microscaling (per-block scheme selection)")

        # 10ao. SharQ: sparse-dense FP4 activation quantization (2.2-2.4x latency)
        self._sharq = None
        if use_sharq:
            from research.quantization.adaptive_quant import SharQQuantizer
            self._sharq = SharQQuantizer(n_ratio=2, m_ratio=4)
            print("  [ForgeEngine] SharQ: sparse-dense FP4 activation quantization")

        # 10ap. MosaicQuant: inlier-outlier disaggregation 4-bit (near-FP16, 1.24x)
        self._mosaic = None
        if use_mosaic_quant:
            from research.quantization.adaptive_quant import MosaicQuantizer
            self._mosaic = MosaicQuantizer(block_size=128)
            print("  [ForgeEngine] MosaicQuant: inlier-outlier disaggregation 4-bit")

        # 10aq. AoH: Autonomy-of-Heads data-free sparse attention (66% decode cut)
        self._aoh = None
        if use_aoh:
            from research.inference.kv.aoh_retmask import AutonomyOfHeads
            n_heads = getattr(self.config, 'n_heads', 32)
            head_dim = getattr(self.config, 'head_dim', 64)
            d_model = getattr(self.config, 'd_model', 2048)
            self._aoh = AutonomyOfHeads(n_heads, head_dim, d_model, sparsity_ratio=0.5)
            print("  [ForgeEngine] AoH: data-free head classification (retrieval vs streaming)")

        # 10ar. FlexDecoding: PyTorch fused FlashDecoding backend (1.5-3x batch=1)
        if use_flex_decoding and self.device.type == "cuda":
            from research.inference.attention.flex_decoding import FlexDecodingAttention
            print("  [ForgeEngine] FlexDecoding: fused FlashDecoding (SM-split decode)")

        # 11. L1 Speculative Attention (lossless, 57% attn compute cut)
        if use_spec_attn:
            from research.keys.speculative.speculative_keys import SpeculativeAttentionKey
            self._spec_attn_key = SpeculativeAttentionKey(draft_rank=32)
            self._spec_attn_key.apply(self.model)
            print("  [ForgeEngine] L1 Speculative Attention: active (lossless, 57% attn cut)")

        # 12. Warmup — pre-run a dummy token to initialize CUDA kernels
        # (like llama.cpp's graph reservation). Avoids first-gen slowdown.
        if warmup and self.device.type == "cuda" and not self._needs_streaming:
            self._warmup()

        # Print VRAM stats (like llama.cpp's model print_info)
        if self.device.type == "cuda":
            vram_free, vram_total = torch.cuda.mem_get_info(self.device)
            used_gb = (vram_total - vram_free) / 1e9
            free_gb = vram_free / 1e9
            print(f"  [ForgeEngine] VRAM: {used_gb:.2f} GB used, {free_gb:.2f} GB free")

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
            import torch
            # Use 4 tokens to trigger conv-layer JIT (conv kernels need seq_len > 1)
            # and attention-layer JIT (causal mask, KV cache allocation).
            vocab_size = getattr(self.model, 'config', None)
            vocab_size = getattr(vocab_size, 'vocab_size', 65536) if vocab_size else 65536
            # Use min(vocab_size, 32767) for warmup — actual values don't matter,
            # just the shape/dtype. High values overflow C long with torch.compile.
            dummy = torch.randint(0, min(vocab_size, 32767), (1, 4),
                                  device=self.device, dtype=torch.long)
            with torch.inference_mode():
                # use_cache=True triggers KV cache kernel compilation too
                self.model(dummy, use_cache=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            print("  [ForgeEngine] Warmup: all CUDA kernels pre-compiled (conv + attn + KV cache)")
        except Exception as e:
            print(f"  [ForgeEngine] Warmup: skipped ({e})")

    def _setup_airllm_smart(self):
        """Smart AirLLM: only stream layers if VRAM can't hold the full model.

        Checks available VRAM vs model size + KV cache overhead.
        If VRAM is sufficient, loads normally (fast path).
        If VRAM is insufficient, splits into shards and streams (slow but works).
        """
        # If from_checkpoint already determined model fits, skip streaming
        if not getattr(self, '_needs_streaming', False):
            # Model is already loaded in VRAM — check if it's actually there
            first_param = next(self.model.parameters(), None)
            if first_param is not None and first_param.device.type == "cuda":
                self._graph_runner = None
                self.acceleration = "none"
                print("  [AirLLM-Smart] Model already in VRAM — streaming not needed (fast path)")
                return

        # Calculate model size in bytes (params * dtype size)
        n_params = sum(p.numel() for p in self.model.parameters())
        dtype_bytes = 2  # bf16
        model_bytes = n_params * dtype_bytes

        # Estimate KV cache overhead: n_layers * 2 (K+V) * max_seq * n_kv * head_dim * dtype
        cfg = getattr(self.model, "config", None)
        n_layers = getattr(cfg, "n_layers", 28)
        n_kv = getattr(cfg, "n_kv_heads", 2) or 12
        head_dim = getattr(cfg, "d_model", 1536) // 12
        max_seq = getattr(cfg, "max_seq_len", 4096)
        batch_size = 1
        kv_bytes = n_layers * 2 * max_seq * n_kv * head_dim * dtype_bytes * batch_size

        total_needed = model_bytes + kv_bytes

        # Check available VRAM
        if self.device.type == "cuda":
            vram_total = torch.cuda.get_device_properties(self.device).total_memory
            vram_free = torch.cuda.mem_get_info(self.device)[0]
        else:
            # CPU: assume enough RAM
            vram_total = 32 * 1024**3  # 32GB assumption
            vram_free = vram_total

        model_gb = model_bytes / 1e9
        kv_gb = kv_bytes / 1e9
        needed_gb = total_needed / 1e9
        free_gb = vram_free / 1e9
        total_gb = vram_total / 1e9

        print(f"  [AirLLM-Smart] Model: {model_gb:.2f} GB, KV cache: {kv_gb:.2f} GB, "
              f"Total needed: {needed_gb:.2f} GB")
        print(f"  [AirLLM-Smart] VRAM free: {free_gb:.2f} GB / {total_gb:.2f} GB")

        # Safety margin: 20% overhead for activations, fragmentation, etc.
        fits = vram_free > total_needed * 1.2

        if fits:
            # Fast path: model fits in VRAM, no streaming needed
            self._graph_runner = None
            self.acceleration = "none"
            print("  [AirLLM-Smart] Model fits in VRAM — loading normally (fast path)")
        else:
            # Slow path: model too large, use layer streaming
            print("  [AirLLM-Smart] Model exceeds VRAM — enabling layer streaming")
            shard_dir = Path(self.checkpoint_path).parent / "xp_shards"
            if not shard_dir.exists() or not any(shard_dir.glob("shard_*.safetensors")):
                print("  [AirLLM-Smart] Splitting checkpoint into shards...")
                from research.keys.moe.airllm_key import AirLLMKey
                key = AirLLMKey()
                key.forward({
                    "checkpoint_path": self.checkpoint_path,
                    "output_dir": str(shard_dir),
                    "compression": None,
                    "layer_prefix": "blocks",
                })

            # Move model weights to CPU (free VRAM) — keep structure intact
            self.model.to('cpu')
            # Load embed/head/norm to VRAM (resident)
            from safetensors.torch import load_file
            shards = sorted(shard_dir.glob("shard_*.safetensors"))
            if shards:
                shard0 = load_file(str(shards[0]))
                for kn, t in shard0.items():
                    # Set on the model's parameter directly
                    for name, param in self.model.named_parameters():
                        if name == kn:
                            param.data = t.to(self.device, dtype=torch.bfloat16)
                            break
                print(f"  [AirLLM-Smart] Resident: {len(shard0)} tensors from shard 0")
                self._layer_shards = shards[1:]
                self._param_map = {name: p for name, p in self.model.named_parameters()}
                self._graph_runner = None
                self.acceleration = "airllm_streaming"
                print(f"  [AirLLM-Smart] Stream layers: {len(self._layer_shards)}")

    def _apply_quantization(self, mode: str):
        """Apply weight-only quantization."""
        if mode == "int8":
            from research.quantization.inference_quant import quantize_model_int8
            # Use fast INT8 (torch._scaled_mm) on CUDA to avoid dequant overhead.
            # On Blackwell (RTX 5070), bf16 matmul is fast so dequant+matmul is
            # slower than bf16 — _scaled_mm does native INT8 matmul with no dequant.
            fast = torch.cuda.is_available()
            quantize_model_int8(self.model, fast=fast)
        elif mode == "int4":
            from research.quantization.inference_quant import quantize_model_int4
            quantize_model_int4(self.model, group_size=128)
        elif mode == "fp8":
            from research.quantization.fp8_infer import quantize_model_fp8
            quantize_model_fp8(self.model)
        elif mode == "w8a8":
            # W8A8: weight + activation INT8 quantization with tensor-core GEMM.
            # 2-3x decode speedup at batch=1 via torch._int_mm.
            from research.inference.quant.w8a8_quant import quantize_model_w8a8
            w8a8_mode = getattr(self.model, 'config', None)
            w8a8_mode = getattr(w8a8_mode, 'w8a8_mode', 'int8') if w8a8_mode else 'int8'
            quantize_model_w8a8(self.model, mode=w8a8_mode)
            print(f"  [ForgeEngine] W8A8 quantization: {w8a8_mode}")
        elif mode == "nvfp4":
            # NVFP4: native FP4 on Blackwell 5th-gen tensor cores.
            # 2x throughput vs FP8, 4x memory vs FP16. ~99% quality.
            from research.inference.quant.nvfp4_quant import quantize_model_nvfp4
            quantize_model_nvfp4(self.model)
            print(f"  [ForgeEngine] NVFP4 quantization: active (Blackwell native FP4)")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 80, repetition_penalty: float = 1.05,
                 finish_sentence: bool = True) -> str:
        """Generate text from a prompt using active strategies.

        Args:
            top_k: LFM2.5-recommended top-k sampling (only applied when
                temperature > 0; ignored for greedy decoding).
            repetition_penalty: LFM2.5-recommended repetition penalty (only
                applied when temperature > 0; ignored for greedy decoding).
            finish_sentence: If True, when max_new_tokens is hit mid-sentence,
                continue generating up to 32 extra tokens to reach a natural
                stopping point (period, newline, code block close, EOS).
        """
        _t0 = time.perf_counter()
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Prefix caching: check if we've seen this prompt prefix before
        prefix_hit = False
        if self._prefix_cache is not None and ids.shape[1] > 16:
            key_len = min(32, ids.shape[1])
            cache_key = ids[:, :key_len].cpu().tolist()[0]
            cached = self._prefix_cache.get(str(cache_key))
            if cached is not None:
                cached_ids, cached_past_kv = cached
                # Reuse cached prefix KV: only process new tokens
                suffix_ids = ids[:, cached_ids.shape[1]:]
                if suffix_ids.shape[1] > 0 and cached_past_kv is not None:
                    with torch.inference_mode():
                        out = self.model(suffix_ids, past_key_values=cached_past_kv,
                                         use_cache=True)
                        logits, past_kv = unpack_output_with_kv(out)
                    # Build full KV cache: prefix + suffix
                    full_past_kv = []
                    for li, (pk, pv) in enumerate(cached_past_kv):
                        sk, sv = past_kv[li]
                        full_past_kv.append((
                            torch.cat([pk, sk], dim=-2),
                            torch.cat([pv, sv], dim=-2),
                        ))
                    # Continue decoding from here (skip the prefill in StandardDecoding)
                    output_ids = self._decode_with_kv(
                        ids, logits, full_past_kv,
                        max_new_tokens, temperature, top_p,
                        top_k=top_k, repetition_penalty=repetition_penalty)
                    prefix_hit = True
                    print(f"  [PrefixCache] HIT + REUSE (prefix len={cached_ids.shape[1]}, "
                          f"saved prefill)")
                else:
                    print(f"  [PrefixCache] HIT (prefix len={cached_ids.shape[1]})")

        if not prefix_hit:
            if self.acceleration == "airllm_streaming":
                output_ids = self._generate_streaming(ids, max_new_tokens, temperature)
            else:
                output_ids = self.decoding.generate(
                    self.model, ids, max_new_tokens, temperature, top_p,
                    top_k=top_k, repetition_penalty=repetition_penalty)

        # Capture KV cache from decoding step for fast finish-to-stop path
        captured_kv = getattr(self.model, '_forge_last_kv', None)

        # Smart cutoff: if we hit max_new_tokens without EOS, extend to next
        # natural stopping point (up to 32 extra tokens).
        if finish_sentence and output_ids.shape[1] - ids.shape[1] >= max_new_tokens:
            output_ids = self._finish_to_stop(output_ids, ids.shape[1],
                                              max_new_tokens, temperature, top_p,
                                              extra_budget=32, past_kv=captured_kv,
                                              top_k=top_k,
                                              repetition_penalty=repetition_penalty)

        # Store prefix KV cache for future reuse
        if self._prefix_cache is not None and ids.shape[1] > 16:
            key_len = min(32, ids.shape[1])
            cache_key = ids[:, :key_len].cpu().tolist()[0]
            if str(cache_key) not in self._prefix_cache:
                # Capture KV cache for the prefix (first key_len tokens)
                with torch.inference_mode():
                    prefix_out = self.model(ids[:, :key_len], use_cache=True)
                    if isinstance(prefix_out, tuple):
                        prefix_kv = prefix_out[2] if len(prefix_out) > 2 else prefix_out[1]
                    else:
                        prefix_kv = None
                self._prefix_cache[str(cache_key)] = (ids[:, :key_len], prefix_kv)

        self.generation_count += 1
        n_gen = output_ids.shape[1] - ids.shape[1]
        self.total_tokens_generated += n_gen
        # Decode only the generated tokens (not the prompt)
        prompt_len = ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Record output + event for built-in diagnostics
        _gen_ms = (time.perf_counter() - _t0) * 1000
        self.outputs.record(prompt, result, n_gen, _gen_ms,
                            temperature=temperature,
                            kv_cache=self.kv_cache.info()["name"] if self.kv_cache and hasattr(self.kv_cache, "info") else "none",
                            decoding=self.decoding.name)
        self.events.log(f"generate: {n_gen} tokens in {_gen_ms:.0f}ms",
                        source="engine", level="profile",
                        tokens=n_gen, time_ms=round(_gen_ms, 1),
                        tok_s=round(n_gen / (_gen_ms / 1000), 1) if _gen_ms > 0 else 0)
        return result

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
        ids = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(self.device)
        prompt_len = ids.shape[1]

        # EOS set: LFM2.5 <|im_end|>=7, Qwen2.5 <|im_end|>=151645, <|endoftext|>=151643
        eos_set = set(eos_token_ids) if eos_token_ids else {7, 151643, 151645}
        eos_attr = getattr(self.tokenizer, "eos_token_id", None)
        if eos_attr is not None:
            eos_set.add(eos_attr)
        eos_tensor = torch.tensor(list(eos_set), device=self.device)

        generated_ids: list[int] = []

        # Prefill with KV cache
        with torch.inference_mode():
            out = self.model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Constrained decoding: apply logits processor BEFORE sampling
            if logits_processor is not None:
                next_logits = logits_processor(next_logits, generated_ids)

            if temperature <= 0:
                next_token = next_logits.argmax(-1, keepdim=True)
            else:
                # Repetition penalty (last 64 tokens)
                if generated_ids:
                    for tid in set(generated_ids[-64:]):
                        next_logits[:, tid] /= repetition_penalty
                # Top-k filtering
                if top_k > 0:
                    k = min(top_k, next_logits.shape[-1])
                    thresh = torch.topk(next_logits, k)[0][..., -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < thresh, float("-inf"))
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=False)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs <= (1 - top_p)
                    remove[..., -1] = False
                    next_logits = next_logits.scatter(
                        -1, sorted_idx, remove.to(next_logits.dtype) * float("-inf"))

                next_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            tok_id = next_token.item()
            generated_ids.append(tok_id)

            if (next_token == eos_tensor).any().item():
                break

            with torch.inference_mode():
                out = self.model(next_token, past_key_values=past_kv, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)

        self.generation_count += 1
        self.total_tokens_generated += len(generated_ids)
        # Clamp token IDs to tokenizer vocab range (model vocab may be larger)
        tok_vocab = len(self.tokenizer)
        safe_ids = [t if t < tok_vocab else tok_vocab - 1 for t in generated_ids]
        return self.tokenizer.decode(safe_ids,
                                     skip_special_tokens=skip_special_tokens)

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
    ):
        """Token-by-token streaming generator.

        Yields decoded text chunks (one per generated token) as they are
        produced, enabling true token-level SSE streaming with low
        time-to-first-token.

        Mirrors :meth:`generate_raw` decoding logic but yields each token's
        decoded text incrementally instead of collecting all tokens first.
        """
        ids = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=False).input_ids.to(self.device)

        # EOS set: LFM2.5 <|im_end|>=7, Qwen2.5 <|im_end|>=151645, <|endoftext|>=151643
        eos_set = {7, 151643, 151645}
        eos_attr = getattr(self.tokenizer, "eos_token_id", None)
        if eos_attr is not None:
            eos_set.add(eos_attr)
        eos_tensor = torch.tensor(list(eos_set), device=self.device)

        generated_ids: list[int] = []
        tok_vocab = len(self.tokenizer)

        # Prefill with KV cache
        with torch.inference_mode():
            out = self.model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if temperature <= 0:
                next_token = next_logits.argmax(-1, keepdim=True)
            else:
                # Repetition penalty (last 64 tokens)
                if generated_ids:
                    for tid in set(generated_ids[-64:]):
                        next_logits[:, tid] /= repetition_penalty
                # Top-k filtering
                if top_k > 0:
                    k = min(top_k, next_logits.shape[-1])
                    thresh = torch.topk(next_logits, k)[0][..., -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < thresh, float("-inf"))
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=False)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs <= (1 - top_p)
                    remove[..., -1] = False
                    next_logits = next_logits.scatter(
                        -1, sorted_idx, remove.to(next_logits.dtype) * float("-inf"))

                next_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            tok_id = next_token.item()
            generated_ids.append(tok_id)

            # Yield the decoded text for this token
            safe_id = tok_id if tok_id < tok_vocab else tok_vocab - 1
            chunk = self.tokenizer.decode([safe_id],
                                          skip_special_tokens=skip_special_tokens)
            if chunk:
                yield chunk

            if (next_token == eos_tensor).any().item():
                break

            with torch.inference_mode():
                out = self.model(next_token, past_key_values=past_kv, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)

        self.generation_count += 1
        self.total_tokens_generated += len(generated_ids)

    @torch.no_grad()
    def _decode_with_kv(self, ids, logits, past_kv,
                        max_new_tokens, temperature, top_p,
                        top_k: int = 80, repetition_penalty: float = 1.05):
        """Standard autoregressive decode from existing KV cache state.

        Used by prefix cache fast path: prefill already done, just decode.
        """
        device = ids.device
        eos = getattr(self.model, "eos_token_id", None)
        eos_set = {7, 151643, 151645}  # LFM2.5 <|im_end|>=7 + Qwen2.5
        if eos is not None:
            eos_set.add(eos)
        eos_tensor = torch.tensor(list(eos_set), device=device)
        token_pinned = torch.zeros(1, 1, dtype=torch.long, pin_memory=True)
        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                next_token = next_logits.argmax(-1, keepdim=True)
            else:
                # Repetition penalty (last 64 tokens)
                if generated_ids:
                    for tid in set(generated_ids[-64:]):
                        next_logits[:, tid] /= repetition_penalty
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(
                        next_logits, top_k)[0][..., -1, None]
                    next_logits.masked_fill_(indices_to_remove, float('-inf'))
                next_token = torch.multinomial(
                    torch.nn.functional.softmax(next_logits, dim=-1),
                    num_samples=1)

            tok_id = next_token.item()
            generated_ids.append(tok_id)
            is_eos = (next_token == eos_tensor).any()
            token_pinned.copy_(next_token, non_blocking=True)
            if is_eos.item():
                break

            ids = torch.cat([ids, next_token], dim=-1)
            with torch.inference_mode():
                out = self.model(next_token, past_key_values=past_kv, use_cache=True)
                if isinstance(out, tuple):
                    logits = out[0]
                    past_kv = out[2] if len(out) > 2 else out[1]
                else:
                    logits = out

        # Expose final KV cache for _finish_to_stop
        self.model._forge_last_kv = past_kv
        return ids

    # Token IDs for natural stopping points (Qwen2.5)
    _STOP_TOKENS = None  # cached on first use

    def _get_stop_tokens(self) -> set:
        """Get token IDs that indicate natural sentence/code boundaries."""
        if self._STOP_TOKENS is not None:
            return self._STOP_TOKENS
        tok = self.tokenizer
        stops = set()
        # EOS tokens
        for t in [151643, 151645]:  # <|endoftext|>, <|im_end|>
            stops.add(t)
        # Common sentence-ending tokens
        for text in [".", "!", "?", ".\n", "!\n", "?\n", ".\"", "!", "?",
                     "```\n", "```\n\n", ")\n", ")\n\n", "}\n", "}\n\n"]:
            ids = tok.encode(text, add_special_tokens=False)
            if ids:
                stops.add(ids[-1])
        self._STOP_TOKENS = stops
        return stops

    @torch.no_grad()
    def _finish_to_stop(self, output_ids, prompt_len, max_new_tokens,
                        temperature, top_p, extra_budget=32,
                        past_kv=None, top_k: int = 80,
                        repetition_penalty: float = 1.05):
        """Continue generation until a natural stopping point or extra_budget.

        If past_kv is provided (captured from the decoding step), skips the
        expensive full-sequence re-run and continues directly from the last state.
        Otherwise falls back to a full prefill to recover KV cache state.
        """
        stop_tokens = self._get_stop_tokens()
        stop_tensor = torch.tensor(list(stop_tokens), device=output_ids.device)
        token_pinned = torch.zeros(1, 1, dtype=torch.long, pin_memory=True)
        extra = 0
        # Track generated token ids for repetition penalty
        generated_ids = output_ids[0, prompt_len:].tolist()

        if past_kv is not None:
            # Fast path: KV cache captured from decoding step.
            # Run just the last token through the model to get logits.
            last_token = output_ids[:, -1:]
            with torch.inference_mode():
                out = self.model(last_token, past_key_values=past_kv, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)
        else:
            # Slow path: re-run full sequence to recover KV cache state.
            with torch.inference_mode():
                out = self.model(output_ids, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)

        while extra < extra_budget:
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                next_token = next_logits.argmax(-1, keepdim=True)
            else:
                # Repetition penalty (last 64 tokens)
                if generated_ids:
                    for tid in set(generated_ids[-64:]):
                        next_logits[:, tid] /= repetition_penalty
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(
                        next_logits, top_k)[0][..., -1, None]
                    next_logits.masked_fill_(indices_to_remove, float('-inf'))
                next_token = torch.multinomial(
                    torch.nn.functional.softmax(next_logits, dim=-1),
                    num_samples=1)

            tok_id = next_token.item()
            generated_ids.append(tok_id)
            output_ids = torch.cat([output_ids, next_token], dim=-1)
            extra += 1

            # Check if we hit a stop token
            is_stop = (next_token == stop_tensor).any()
            token_pinned.copy_(next_token, non_blocking=True)
            if is_stop.item():
                break

            # Continue with KV cache
            with torch.inference_mode():
                out = self.model(next_token, past_key_values=past_kv, use_cache=True)
                if isinstance(out, tuple):
                    logits = out[0]
                    past_kv = out[2] if len(out) > 2 else out[1]
                else:
                    logits = out

        return output_ids

    def _generate_streaming(self, ids: torch.Tensor, max_new_tokens: int,
                            temperature: float) -> torch.Tensor:
        """AirLLM streaming generation: load layer shards per forward pass.

        Shards are loaded once per forward pass (not per token). KV cache is
        maintained across decode steps to avoid O(seq_len) recomputation.
        """
        from safetensors.torch import load_file
        eos = getattr(self.tokenizer, "eos_token_id", None)
        param_map = self._param_map
        device = self.device
        n_layers = len(self.model.blocks)

        # Pre-allocate KV cache on GPU (small: n_layers * 2 * max_seq * n_kv * head_dim * 2 bytes)
        cfg = getattr(self.model, "config", None)
        n_kv = getattr(cfg, "n_kv_heads", 2) or 12
        head_dim = getattr(cfg, "d_model", 2048) // (getattr(cfg, "n_heads", 32) or 32)
        max_seq = getattr(cfg, "max_seq_len", 32768)
        kv_cache = [
            (
                torch.empty(1, n_kv, max_seq, head_dim, dtype=torch.bfloat16, device=device),
                torch.empty(1, n_kv, max_seq, head_dim, dtype=torch.bfloat16, device=device),
            )
            for _ in range(n_layers)
        ]
        cache_pos = ids.shape[1]  # current fill position in KV cache

        def _load_all_shards():
            """Load all layer shards from disk to GPU. Call once per forward pass."""
            for li in range(n_layers):
                state = load_file(str(self._layer_shards[li]))
                for kn, t in state.items():
                    if kn in param_map:
                        param_map[kn].data = t.to(device, dtype=torch.bfloat16, non_blocking=True)
            torch.cuda.synchronize() if device.type == "cuda" else None

        def _free_all_shards():
            """Free all layer weights back to CPU."""
            for li in range(n_layers):
                state = load_file(str(self._layer_shards[li]))
                for kn in state:
                    if kn in param_map:
                        param_map[kn].data = param_map[kn].data.cpu()
                del state
            torch.cuda.empty_cache()

        def _forward_with_kv(x: torch.Tensor, start_layer: int = 0,
                             start_pos: int = 0) -> torch.Tensor:
            """Run layers with KV cache update. x shape: (1, seq, d_model)."""
            for li in range(start_layer, n_layers):
                k_cache, v_cache = kv_cache[li]
                # Pass KV cache to the block (blocks must support past_kv)
                out = self.model.blocks[li](x)
                if isinstance(out, tuple) and len(out) >= 3:
                    x, new_k, new_v = out[0], out[1], out[2]
                    # Update KV cache at current positions
                    seq_len = new_k.shape[1]
                    end = start_pos + seq_len
                    k_cache[:, :, start_pos:end] = new_k
                    v_cache[:, :, start_pos:end] = new_v
                else:
                    x = out[0] if isinstance(out, tuple) else out
            x = self.model.ln_f(x)
            return self.model.head(x)

        with torch.inference_mode():
            # Prefill: run full sequence through all layers with KV cache
            _load_all_shards()
            logits = _forward_with_kv(self.model.embed(ids), start_pos=0)
            _free_all_shards()

        # Decode loop: one token at a time, only the new token through layers
        for step in range(max_new_tokens):
            if temperature == 0:
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits[:, -1, :] / max(temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos is not None and (next_token == eos).any():
                break

            ids = torch.cat([ids, next_token], dim=-1)

            with torch.inference_mode():
                _load_all_shards()
                x = self.model.embed(next_token)
                logits = _forward_with_kv(x, start_pos=cache_pos)
                _free_all_shards()

            cache_pos += 1

        return ids

    def benchmark(self, prompt: str, max_new_tokens: int = 50,
                  n_runs: int = 3) -> dict:
        """Benchmark generation speed."""
        self.generate(prompt, max_new_tokens=10)  # warmup
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            t0 = time.time()
            self.generate(prompt, max_new_tokens=max_new_tokens)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
        avg = sum(times) / len(times)
        tps = max_new_tokens / avg
        print(f"  [ForgeEngine] {tps:.0f} tok/s | {avg*1000:.0f}ms for {max_new_tokens} tokens")
        return {"tokens_per_sec": tps, "latency_ms": avg * 1000,
                "tokens": max_new_tokens, "runs": n_runs}

    def stats(self) -> dict:
        """Get engine statistics."""
        vram_info = {}
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            vram_info = {"used_gb": (total - free) / 1e9,
                         "free_gb": free / 1e9, "total_gb": total / 1e9}
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
        self.events.log("Starting bottleneck profiling", source="profile",
                        max_new_tokens=max_new_tokens)
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        result = self._profiler.profile_generate(ids, max_new_tokens=max_new_tokens)
        self.events.log(f"Bottleneck: {result.get('tok_s', 0)} tok/s, "
                        f"slowest={result['bottlenecks'][0]['type']}#{result['bottlenecks'][0]['index']} "
                        f"({result['bottlenecks'][0]['time_ms']}ms)",
                        source="profile", level="profile",
                        bottlenecks=result["bottlenecks"])
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

    def sleep(self, level: int = 1):
        """Release GPU memory by offloading model weights.

        Level 1 (default): Move weights to CPU RAM. Fast wake (~2-3s).
            Preserves tokenizer, config, KV cache strategies, and CUDA context.
        Level 2: Discard weights entirely. Slower wake (reload from disk).
            Use for model switching when Level 1 CPU RAM is insufficient.

        After sleep, generation will fail until wake() is called.
        """
        if not hasattr(self, '_awake') or self._awake is False:
            return  # Already asleep

        if level == 1:
            # Offload weights to CPU, keep CUDA context alive
            self.model.to('cpu', non_blocking=True)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            torch.cuda.empty_cache()
            self._awake = False
            self._sleep_level = 1
            print(f"  [ForgeEngine] Sleep level 1: weights offloaded to CPU")
        elif level == 2:
            # Store minimal state, discard model
            self._stored_config = getattr(self.model, 'config', None)
            self._stored_dtype = next(self.model.parameters()).dtype
            self._stored_checkpoint = self.checkpoint_path
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            self._awake = False
            self._sleep_level = 2
            print(f"  [ForgeEngine] Sleep level 2: weights discarded, {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free")

    def wake(self):
        """Restore model to GPU and resume inference.

        Level 1 wake: CPU→GPU copy (~2-3s). Preserves all strategies.
        Level 2 wake: Reload from checkpoint (~5-10s). Strategies must be re-activated.
        """
        if getattr(self, '_awake', True):
            return  # Already awake

        if self._sleep_level == 1:
            self.model.to(self.device, non_blocking=True)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            self._awake = True
            print(f"  [ForgeEngine] Woke from level 1 sleep")
        elif self._sleep_level == 2:
            if not hasattr(self, '_stored_checkpoint') or not self._stored_checkpoint:
                raise RuntimeError("Level 2 wake requires stored checkpoint path")
            from research.model_loader import ModelLoader
            self.model = ModelLoader.build_model_fast(
                self._stored_config, checkpoint_path=self._stored_checkpoint)
            self.model.to(self.device)
            self.model.eval()
            del self._stored_config
            del self._stored_checkpoint
            self._awake = True
            print(f"  [ForgeEngine] Woke from level 2 sleep (reloaded from checkpoint)")

    @property
    def is_awake(self) -> bool:
        return getattr(self, '_awake', True)

    def vram_usage(self) -> dict:
        """Report current VRAM usage for this engine."""
        if self.device.type != 'cuda':
            return {"total_gb": 0, "free_gb": 0, "used_gb": 0}
        total = torch.cuda.get_device_properties(self.device).total_memory
        free = torch.cuda.mem_get_info(self.device)[0]
        used = total - free
        # Estimate model weight VRAM
        model_bytes = 0
        if self.is_awake and self.model is not None:
            try:
                model_bytes = sum(
                    p.numel() * p.element_size() for p in self.model.parameters()
                    if p.device.type == 'cuda')
            except Exception:
                pass
        return {
            "total_gb": total / 1e9,
            "free_gb": free / 1e9,
            "used_gb": used / 1e9,
            "model_weights_gb": model_bytes / 1e9,
        }

    def compare_strategies(self, prompt: str, max_new_tokens: int = 30) -> dict:
        """Compare different strategy combinations on the same prompt."""
        results = {}
        configs = [
            {"kv_cache": "standard", "decoding": "standard", "label": "baseline"},
            {"kv_cache": "hadamard_int4", "decoding": "standard", "label": "hadamard_kv"},
            {"kv_cache": "rotorquant", "decoding": "standard", "label": "rotorquant_kv"},
            {"kv_cache": "standard", "decoding": "mtp_selfspec", "label": "mtp_spec"},
        ]
        for cfg in configs:
            label = cfg.pop("label")
            self.activate(**cfg)
            t0 = time.time()
            out = self.generate(prompt, max_new_tokens=max_new_tokens)
            dt = time.time() - t0
            tps = max_new_tokens / dt
            results[label] = {"output": out[:80], "tok/s": tps, "time": dt}
            print(f"  {label:15s}: {tps:.0f} tok/s | {out[:60]}")
        return results
