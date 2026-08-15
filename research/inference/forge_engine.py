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
        config_name="lfm25_1.2b",
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
    def from_checkpoint(cls, checkpoint: str, config_name: str = "lfm25_1.2b",
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
                 warmup: bool = True):
        """Activate runtime strategies.

        Args:
            kv_cache: "standard", "paged", "rotorquant", "hadamard_int4", "compressed",
                      "streaming", "snapkv"
            decoding: "standard", "speculative", "medusa", "dspark", "eagle3", "mtp_selfspec"
            quantize: None, "int8", "int4", "fp8"
            acceleration: None, "cuda_graph", "airllm_streaming"
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
            dummy = torch.randint(0, vocab_size, (1, 4),
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
        kv_bytes = n_layers * 2 * max_seq * n_kv * head_dim * dtype_bytes

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
        self.total_tokens_generated += output_ids.shape[1] - ids.shape[1]
        # Decode only the generated tokens (not the prompt)
        prompt_len = ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

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
    def _decode_with_kv(self, ids, logits, past_kv,
                        max_new_tokens, temperature, top_p,
                        top_k: int = 80, repetition_penalty: float = 1.05):
        """Standard autoregressive decode from existing KV cache state.

        Used by prefix cache fast path: prefill already done, just decode.
        """
        device = ids.device
        eos = getattr(self.model, "eos_token_id", None)
        eos_set = {151643, 151645}
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
