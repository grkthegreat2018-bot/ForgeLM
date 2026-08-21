"""AirLLM layer-streaming for models that exceed available VRAM.

Extracted from ``ForgeEngine`` to isolate the streaming load/generate
logic.  Two public entry points:

- ``AirLLMStreamer.setup(engine)`` — splits the checkpoint into per-layer
  shards (if needed), moves the model to CPU, and keeps embed/head/norm
  resident on GPU.
- ``AirLLMStreamer.generate(engine, ids, max_new_tokens, temperature)`` —
  runs prefill + decode with per-forward-pass shard loading.

The streamer is intentionally stateless beyond what it stores on the
engine (``_layer_shards``, ``_param_map``) so it can be used as a
drop-in replacement for the old inline methods.
"""
from __future__ import annotations

from pathlib import Path

import torch

from research.model_loader import unpack_output_with_kv

# Lazy imports inside functions to avoid loading safetensors at module import.


class AirLLMStreamer:
    """Manages AirLLM layer-streaming for a ForgeEngine."""

    # ── Setup ────────────────────────────────────────────────────────────

    @staticmethod
    def setup(engine) -> None:
        """Configure the engine for layer-streaming if VRAM is insufficient.

        If the model already fits in VRAM, this is a no-op (fast path).
        Otherwise, splits the checkpoint into shards and sets up the
        ``_layer_shards`` / ``_param_map`` attributes on the engine.
        """
        # If from_checkpoint already determined model fits, skip streaming
        if not getattr(engine, "_needs_streaming", False):
            first_param = next(engine.model.parameters(), None)
            if first_param is not None and first_param.device.type == "cuda":
                engine._graph_runner = None
                engine.acceleration = None
                print("  [AirLLM-Smart] Model already in VRAM — "
                      "streaming not needed (fast path)")
                return

        model_bytes, kv_bytes = AirLLMStreamer._estimate_memory(engine)
        total_needed = model_bytes + kv_bytes
        vram_free, vram_total = engine._memory_info(engine.device)

        model_gb = model_bytes / 1e9
        kv_gb = kv_bytes / 1e9
        needed_gb = total_needed / 1e9
        free_gb = vram_free / 1e9
        total_gb = vram_total / 1e9

        print(f"  [AirLLM-Smart] Model: {model_gb:.2f} GB, "
              f"KV cache: {kv_gb:.2f} GB, Total needed: {needed_gb:.2f} GB")
        print(f"  [AirLLM-Smart] VRAM free: {free_gb:.2f} GB / "
              f"{total_gb:.2f} GB")

        # 20% safety margin for activations, fragmentation, etc.
        if vram_free > total_needed * 1.2:
            engine._graph_runner = None
            engine.acceleration = None
            print("  [AirLLM-Smart] Model fits in VRAM — "
                  "loading normally (fast path)")
            return

        # Slow path: model too large, use layer streaming
        print("  [AirLLM-Smart] Model exceeds VRAM — enabling layer streaming")
        AirLLMStreamer._prepare_shards(engine)
        AirLLMStreamer._load_resident_shard(engine)

    @staticmethod
    def _estimate_memory(engine) -> tuple[int, int]:
        """Return (model_bytes, kv_bytes) estimate."""
        n_params = sum(p.numel() for p in engine.model.parameters())
        dtype_bytes = 2  # bf16
        model_bytes = n_params * dtype_bytes

        cfg = getattr(engine.model, "config", None)
        n_layers = getattr(cfg, "n_layers", 28)
        n_kv = getattr(cfg, "n_kv_heads", 2) or 12
        head_dim = getattr(cfg, "d_model", 1536) // 12
        max_seq = getattr(cfg, "max_seq_len", 4096)
        kv_bytes = n_layers * 2 * max_seq * n_kv * head_dim * dtype_bytes
        return model_bytes, kv_bytes

    @staticmethod
    def _prepare_shards(engine) -> None:
        """Split checkpoint into per-layer shards if not already done."""
        shard_dir = Path(engine.checkpoint_path).parent / "xp_shards"
        if not shard_dir.exists() or not any(shard_dir.glob("shard_*.safetensors")):
            print("  [AirLLM-Smart] Splitting checkpoint into shards...")
            from research.keys.moe.airllm_key import AirLLMKey
            key = AirLLMKey()
            key.forward({
                "checkpoint_path": engine.checkpoint_path,
                "output_dir": str(shard_dir),
                "compression": None,
                "layer_prefix": "blocks",
            })

    @staticmethod
    def _load_resident_shard(engine) -> None:
        """Move model to CPU, load embed/head/norm (shard 0) to VRAM."""
        from safetensors.torch import load_file

        engine.model.to("cpu")
        shard_dir = Path(engine.checkpoint_path).parent / "xp_shards"
        shards = sorted(shard_dir.glob("shard_*.safetensors"))
        if not shards:
            return

        shard0 = load_file(str(shards[0]))
        for kn, t in shard0.items():
            for name, param in engine.model.named_parameters():
                if name == kn:
                    param.data = t.to(engine.device, dtype=torch.bfloat16)
                    break
        print(f"  [AirLLM-Smart] Resident: {len(shard0)} tensors from shard 0")
        engine._layer_shards = shards[1:]
        engine._param_map = dict(engine.model.named_parameters())
        engine._graph_runner = None
        engine.acceleration = "airllm_streaming"
        print(f"  [AirLLM-Smart] Stream layers: {len(engine._layer_shards)}")

    # ── Generation ───────────────────────────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def generate(engine, ids: torch.Tensor, max_new_tokens: int,
                 temperature: float) -> torch.Tensor:
        """Run generation with per-forward-pass shard loading.

        Shards are loaded once per forward pass (not per token). KV cache
        is maintained across decode steps to avoid O(seq_len) recomputation.

        Shards are cached in CPU RAM with LRU eviction to avoid re-reading
        from disk on every decode step. The shard cache size is limited
        to ``_SHARD_CACHE_MAX`` entries — when full, the least-recently-used
        shard is evicted from CPU RAM.
        """
        from safetensors.torch import load_file

        eos_set = engine._eos_token_ids()
        param_map = engine._param_map
        device = engine.device
        n_layers = len(engine.model.blocks)

        # Pre-allocate KV cache on GPU
        cfg = getattr(engine.model, "config", None)
        n_kv = getattr(cfg, "n_kv_heads", 2) or 12
        head_dim = (getattr(cfg, "d_model", 2048)
                    // (getattr(cfg, "n_heads", 32) or 32))
        max_seq = getattr(cfg, "max_seq_len", 32768)
        kv_cache = [
            (
                torch.empty(1, n_kv, max_seq, head_dim,
                            dtype=torch.bfloat16, device=device),
                torch.empty(1, n_kv, max_seq, head_dim,
                            dtype=torch.bfloat16, device=device),
            )
            for _ in range(n_layers)
        ]
        cache_pos = ids.shape[1]

        # Shard cache: keeps loaded shards in CPU RAM to avoid re-reading
        # from disk on every decode step. LRU eviction when full.
        shard_cache: dict[int, dict[str, torch.Tensor]] = {}
        shard_lru: list[int] = []  # most-recently-used at end
        _SHARD_CACHE_MAX = n_layers  # cache all shards if we can

        def _load_shard(li: int) -> dict[str, torch.Tensor]:
            """Load shard li from cache or disk."""
            if li in shard_cache:
                shard_lru.remove(li)
                shard_lru.append(li)
                return shard_cache[li]
            state = load_file(str(engine._layer_shards[li]))
            # Cache in CPU RAM
            shard_cache[li] = state
            shard_lru.append(li)
            # Evict oldest if over capacity
            while len(shard_cache) > _SHARD_CACHE_MAX:
                old_li = shard_lru.pop(0)
                del shard_cache[old_li]
            return state

        def _load_all_shards():
            for li in range(n_layers):
                state = _load_shard(li)
                for kn, t in state.items():
                    if kn in param_map:
                        param_map[kn].data = t.to(
                            device, dtype=torch.bfloat16, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

        def _free_all_shards():
            """Move params back to CPU (shards stay cached in CPU RAM)."""
            for li in range(n_layers):
                state = shard_cache.get(li)
                if state is None:
                    state = _load_shard(li)
                for kn in state:
                    if kn in param_map:
                        param_map[kn].data = param_map[kn].data.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        def _forward_with_kv(x, start_pos=0):
            for li in range(n_layers):
                k_cache, v_cache = kv_cache[li]
                out = engine.model.blocks[li](x)
                if isinstance(out, tuple) and len(out) >= 3:
                    x, new_k, new_v = out[0], out[1], out[2]
                    seq_len = new_k.shape[1]
                    end = start_pos + seq_len
                    k_cache[:, :, start_pos:end] = new_k
                    v_cache[:, :, start_pos:end] = new_v
                else:
                    x = out[0] if isinstance(out, tuple) else out
            x = engine.model.ln_f(x)
            return engine.model.head(x)

        # Prefill
        with torch.inference_mode():
            _load_all_shards()
            logits = _forward_with_kv(engine.model.embed(ids), start_pos=0)
            _free_all_shards()

        # Decode loop
        generated_ids: list[int] = []
        generated_tokens: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            next_token = engine._sample_next_token(
                logits[:, -1, :], temperature, 0, 1.0, 1.0, generated_ids)
            token_id = next_token.item()
            generated_ids.append(token_id)
            if token_id in eos_set:
                break
            generated_tokens.append(next_token)

            with torch.inference_mode():
                _load_all_shards()
                x = engine.model.embed(next_token)
                logits = _forward_with_kv(x, start_pos=cache_pos)
                _free_all_shards()
            cache_pos += 1

        if not generated_tokens:
            return ids
        return torch.cat([ids, *generated_tokens], dim=-1)
