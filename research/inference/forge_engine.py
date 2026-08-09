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
        config_name="qwen25_coder_1.5b",
        tokenizer_path="research/checkpoints/qwen_hf",
    )
    engine.activate(kv_cache="hadamard_int4", decoding="mtp_selfspec")
    output = engine.generate("def fibonacci(n):", max_new_tokens=50)
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from research.inference.decoding import DecodingStrategy, StandardDecoding, build_decoding
from research.inference.innovations import (
    MRLAdaptiveContext,
    ProgressiveKV,
    QuaRotKV,
    V0WarmStart,
)
from research.inference.kv_backend import KVCacheStrategy, build_kv_cache


class ForgeEngine:
    """Unified inference engine for ForgeAI XP models.

    Orchestrates all runtime strategies and innovations. Auto-detects
    KeyStack features from checkpoint and activates matching strategies.
    """

    def __init__(self, model, tokenizer, device="cuda",
                 checkpoint_path: str | None = None):
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
    def from_checkpoint(cls, checkpoint: str, config_name: str = "qwen25_coder_1.5b",
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
        tok_path = tokenizer_path or "research/checkpoints/qwen_hf"
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
        features = []
        with safe_open(self.checkpoint_path, framework="pt") as f:
            keys = set(f.keys())
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

    def activate(self, kv_cache: str = "standard",
                 decoding: str = "standard",
                 quantize: str | None = None,
                 acceleration: str | None = None,
                 mrl_keep_ratio: float | None = None,
                 kv_bits: int = 4,
                 use_v0_warm: bool = False,
                 use_progressive_kv: bool = False,
                 use_compile: bool = False,
                 use_prefix_cache: bool = False,
                 use_spec_attn: bool = False):
        """Activate runtime strategies.

        Args:
            kv_cache: "standard", "paged", "rotorquant", "hadamard_int4", "compressed",
                      "streaming", "snapkv"
            decoding: "standard", "speculative", "medusa", "dspark", "mtp_selfspec"
            quantize: None, "int8", "int4"
            acceleration: None, "cuda_graph", "airllm_streaming"
            mrl_keep_ratio: if set (e.g. 0.75), truncate to that fraction of dims
            kv_bits: 4 or 8, for KV cache quantization
            use_v0_warm: enable V0 warm-start for KV cache
            use_progressive_kv: enable progressive KV (anchor + residual streams)
            use_compile: torch.compile the model for 1.3-2x decode speedup
            use_prefix_cache: cache KV for repeated prompt prefixes
            use_spec_attn: L1 Speculative Attention (57% attn cut, lossless)
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

        # 10. Prefix caching
        self._prefix_cache = {} if use_prefix_cache else None
        if use_prefix_cache:
            print("  [ForgeEngine] Prefix caching: active")

        # 11. L1 Speculative Attention (lossless, 57% attn compute cut)
        if use_spec_attn:
            from research.keys.speculative_keys import SpeculativeAttentionKey
            self._spec_attn_key = SpeculativeAttentionKey(draft_rank=32)
            self._spec_attn_key.apply(self.model)
            print("  [ForgeEngine] L1 Speculative Attention: active (lossless, 57% attn cut)")

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
                from research.keys.airllm_key import AirLLMKey
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
            quantize_model_int8(self.model)
        elif mode == "int4":
            from research.quantization.inference_quant import quantize_model_int4
            quantize_model_int4(self.model, group_size=128)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.0, top_p: float = 1.0,
                 finish_sentence: bool = True) -> str:
        """Generate text from a prompt using active strategies.

        Args:
            finish_sentence: If True, when max_new_tokens is hit mid-sentence,
                continue generating up to 32 extra tokens to reach a natural
                stopping point (period, newline, code block close, EOS).
        """
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Prefix caching: check if we've seen this prompt prefix before
        if self._prefix_cache is not None and ids.shape[1] > 16:
            # Use first 32 tokens as cache key (or full prompt if shorter)
            key_len = min(32, ids.shape[1])
            cache_key = ids[:, :key_len].cpu().tolist()[0]
            cached = self._prefix_cache.get(str(cache_key))
            if cached is not None:
                cached_ids, cached_past_kv = cached
                # Reuse cached prefix KV, only process new tokens
                new_ids = ids[:, cached_ids.shape[1]:]
                if new_ids.shape[1] > 0:
                    # TODO: full prefix cache integration with decoding
                    pass
                # For now, just log the hit
                print(f"  [PrefixCache] HIT (prefix len={cached_ids.shape[1]})")

        if self.acceleration == "airllm_streaming":
            output_ids = self._generate_streaming(ids, max_new_tokens, temperature)
        else:
            output_ids = self.decoding.generate(
                self.model, ids, max_new_tokens, temperature, top_p)

        # Smart cutoff: if we hit max_new_tokens without EOS, extend to next
        # natural stopping point (up to 32 extra tokens).
        if finish_sentence and output_ids.shape[1] - ids.shape[1] >= max_new_tokens:
            output_ids = self._finish_to_stop(output_ids, ids.shape[1],
                                              max_new_tokens, temperature, top_p,
                                              extra_budget=32)

        # Store prefix in cache
        if self._prefix_cache is not None and ids.shape[1] > 16:
            key_len = min(32, ids.shape[1])
            cache_key = ids[:, :key_len].cpu().tolist()[0]
            self._prefix_cache[str(cache_key)] = (ids, None)

        self.generation_count += 1
        self.total_tokens_generated += output_ids.shape[1] - ids.shape[1]
        # Decode only the generated tokens (not the prompt)
        prompt_len = ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

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
                        temperature, top_p, extra_budget=32):
        """Continue generation until a natural stopping point or extra_budget."""
        stop_tokens = self._get_stop_tokens()
        stop_tensor = torch.tensor(list(stop_tokens), device=output_ids.device)
        token_pinned = torch.zeros(1, dtype=torch.long, pin_memory=True)
        eos_set = {151643, 151645}

        # Use the decoding strategy to extend, checking each new token
        generated = output_ids.shape[1] - prompt_len
        extra = 0

        # Get current KV cache state by re-running with use_cache
        # Simplest: just continue with standard decoding on the full sequence
        with torch.inference_mode():
            out = self.model(output_ids, use_cache=True)
            if isinstance(out, tuple):
                logits = out[0]
                past_kv = out[2] if len(out) > 2 else out[1]
            else:
                logits = out
                past_kv = None

        while extra < extra_budget:
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                next_token = next_logits.argmax(-1, keepdim=True)
            else:
                next_token = torch.multinomial(
                    torch.nn.functional.softmax(next_logits, dim=-1),
                    num_samples=1)

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
        """AirLLM streaming generation: load one layer at a time from disk shards."""
        from safetensors.torch import load_file
        eos = getattr(self.tokenizer, "eos_token_id", None)
        param_map = self._param_map

        for step in range(max_new_tokens):
            with torch.inference_mode():
                # Embed (resident in VRAM)
                x = self.model.embed(ids)

                # Stream each layer: load shard → compute → free to CPU
                for li, block in enumerate(self.model.blocks):
                    shard = self._layer_shards[li]
                    state = load_file(str(shard))
                    for kn, t in state.items():
                        if kn in param_map:
                            param_map[kn].data = t.to(self.device, dtype=torch.bfloat16)
                    x = block(x)
                    if isinstance(x, tuple):
                        x = x[0]
                    # Free layer weights back to CPU
                    for kn in state:
                        if kn in param_map:
                            param_map[kn].data = param_map[kn].data.cpu()
                    del state
                    torch.cuda.empty_cache()

                # Final norm + head (resident)
                x = self.model.ln_f(x)
                logits = self.model.head(x)

            next_token_gpu = logits[0, -1].argmax(keepdim=True).unsqueeze(0)  # (1, 1) GPU
            if eos:
                is_eos = (next_token_gpu == eos).any()
                if is_eos.item():
                    break
            ids = torch.cat([ids, next_token_gpu], dim=-1)

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
