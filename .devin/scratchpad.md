# R&D Round: Boot Time / Load Optimization (2026-08-18/19)

## Baseline (forgelm_v3, ForgeLM_V3_Base.safetensors, RTX 5070, bf16)

| Stage | Cold (ms) | Warm (ms) | % (warm) | Notes |
|-------|-----------|-----------|----------|-------|
| D_arch_build | 6689-11398 | 6689 | 59% | ConfigurableResearchLLM.__init__ + .to(device) |
| F_weights_load | 1283-29165 | 1283 | 11% | fastsafetensors direct-to-GPU |
| J_tokenizer_load | 2720-46244 | 2720 | 24% | Serial AFTER model build |
| L_first_forward | 176-2414 | 376 | 3.3% | cache_devices() + JIT |
| H_load_state_dict | 37-378 | 37 | 0.3% | GPU→GPU copy |
| I_post_load_scan | 20-342 | 20 | 0.2% | QK-norm identity scan |
| **TOTAL** | **11279-90382** | **11279** | 100% | |

## Variation Results (all correct: "Paris. The capital" tokens [5242,523,509,1098,5706])

| Variation | TOTAL (ms) | Speedup | VRAM | Correct | Key Change |
|-----------|-----------|---------|------|---------|------------|
| baseline | 11279 | 1.0x | 2.68 | ✓ | reference |
| V1 skip_init | 10190 | 1.1x | 2.74 | ✓ | torch.nn.utils.skip_init |
| V2 parallel tok | 7185 | 1.6x | 2.68 | ✓ | tokenizer in ThreadPoolExecutor |
| V3 OS prefetch | 6363 | 1.8x | 2.68 | ✓ | background 16MB block reads |
| V4 PrefetchVM | 6355 | 1.8x | 2.68 | ✓ | Windows PrefetchVirtualMemory (failed, fell back) |
| V5 meta+assign | 4414 | 2.6x | 2.74 | ✓ | meta init + load_state_dict(assign=True) |
| V6 V1+V2+V3 | 8209 | 1.4x | 2.74 | ✓ | skip_init + parallel tok + prefetch |
| **V7 V5+V2+V3** | **3316** | **3.4x** | **2.74** | **✓** | **meta+assign + parallel tok + prefetch** |

## V7 Breakdown (the winner)

| Stage | ms | Notes |
|-------|------|-------|
| D_arch_build | 1098 | meta init (no param alloc, no init kernels) |
| F_weights_load | 1489 | fastsafetensors + OS prefetch warmed cache |
| J_tokenizer_load | 392 | wait time only; 2713ms hidden behind arch+weights |
| L_first_forward | 181 | first kernel JIT (unavoidable) |
| E_dtype_convert | 56 | RoPE buffer re-init |
| H_load_state_dict | 46 | assign=True (direct param replacement) |
| I_post_load_scan | 22 | QK-norm identity scan |
| M_first_decode | 28 | second token |
| **TOTAL** | **3316** | |

## Key Findings

### What Worked
1. **Meta device init + assign=True (V5/V7)** — 6.1x speedup on D_arch_build (6.7s → 1.1s)
   - Build model on `torch.device("meta")` (zero storage, just shape metadata)
   - `load_state_dict(state, assign=True)` directly replaces meta params with state_dict tensors
   - Skips both init kernels AND the .to(device) copy
   - Requires: re-tie weights (assign breaks sharing), re-init RoPE buffers (non-persistent)
2. **Parallel tokenizer (V2/V7)** — hides 2.7s behind arch build
   - `ThreadPoolExecutor` starts tokenizer load before arch build
   - Only 392ms wait time (2713ms hidden)
3. **OS page cache prefetch (V3/V7)** — modest weight load improvement
   - Background thread reads 16MB blocks during arch build
   - Warms OS page cache before fastsafetensors reads

### What Failed
1. **V4 PrefetchVirtualMemory** — API call failed on Windows (mmap buffer access issue)
   - `mm.__buffer__()` not available in Python's mmap module
   - Would need ctypes-level virtual memory mapping to work
   - Fell back to no-prefetch, same as V3
2. **V1 skip_init** — works but slower than V5 (10.2s vs 4.4s)
   - `skip_init` uses meta + `to_empty` internally
   - `to_empty` materializes ALL params (extra copy), then `load_state_dict` fills them
   - V5's `assign=True` skips the intermediate materialization
3. **V6 (skip_init combo)** — 8.2s, dominated by skip_init overhead
   - Replaced by V7 (meta+assign combo) which is 2.5x faster

### Future Optimizations (not implemented)
1. **Lazy key instantiation** — 698ms (92% of meta init!) is key module creation overhead
   - TITAN/MoD/MHC/AttnRes/DiffAttn/BitNet created per-block in __init__
   - Could defer to first forward (they're zero-init=lossless)
   - Would drop D_arch_build from 1098ms to ~400ms
   - Requires invasive ModularBlock changes
2. **fastsafetensors pipeline mode** — `use_pipeline=true, queue_size=0`
   - Overlaps copy with broadcast (for multi-GPU)
   - Single-GPU benefit unclear
3. **CUDA stream weight copy** — overlap H2D with kernel JIT
   - Would hide F_weights_load behind L_first_forward JIT
4. **Persistent compile cache warming** — pre-compile kernels offline
   - Would eliminate L_first_forward JIT overhead (181ms)

## Randomizer Output (step 7)
- Combo 3: os_page_prefetch + cuda_stream_weight_copy + meta_device_init (≈ V7 + stream copy)
- Combo 4: buffer_reuse + spectral_decomposition_load + torch_compile_warm_cache (cross-domain)
- Combo 5: cuda_stream_weight_copy + skip_init + moe_expert_bake + kv_cache_lazy_alloc
- Most promising next: lazy_key_instantiation (698ms savings, noted above)

## Production Change

Implemented in `research/model_loader.py` + `research/tokenizer_cache.py`:
- Added `fast_load` parameter to `load_default_model()` and `build_model_fast()`
- When `fast_load=True`: uses meta init + assign=True + parallel tokenizer + OS prefetch
- Added `self.base`, `self.max_seq_len`, `self.rope_scaling` to RotaryEmbedding (needed for buffer re-init)
- Added `_reset_non_persistent_buffers()` helper
- Default: `fast_load=True` (opt-in to disable with `fast_load=False`)

## Round 2: Lazy Key Instantiation Investigation (2026-08-19)

### Profiling Result
- Meta init (full V3, cold): 780ms — dominated by Python import overhead
- Meta init (full V3, warm): 52ms — only 40ms is key module creation
- BitNet is 97% of key overhead (762ms cold, 33ms warm)
- **Conclusion: lazy key instantiation NOT worth it** — the 700ms cold cost is
  Python import overhead that must be paid somewhere. Moving to first forward
  is a bad trade (first forward is already 180ms JIT, adding 700ms = 880ms).

### Pivot: Fast Tokenizer (the real bottleneck)

Profiled tokenizer load:
- `from transformers import AutoTokenizer`: 3957ms (!!) — dominant cost
- `AutoTokenizer.from_pretrained()`: 89ms
- `gigatoken.Tokenizer().as_hf()`: 204ms
- Total via transformers: 4254ms

**Fast path: `tokenizers` Rust library directly**
- `from tokenizers import Tokenizer`: 18.5ms
- `Tokenizer.from_file()`: 77ms
- `gigatoken.Tokenizer(rust_tok).as_hf()`: 113ms
- Read tokenizer_config.json for special tokens: <1ms
- Total: 223ms — **19x faster**

### Implementation
- `tokenizer_cache.py`: `_load_fast_tokenizer()` uses `tokenizers.Tokenizer.from_file()`
  + gigatoken wrap + sets `eos_token`/`bos_token`/`pad_token` strings (gigatoken's
  `eos_token_id` property reads from these)
- Falls back to `_load_hf_tokenizer()` (transformers) if fast path fails
- Token IDs verified identical: [1, 1098, 5706, 803, 4481, 856] for both paths

### Final Results (with fast tokenizer)

| Metric | Original Baseline | V7 (meta+assign) | V7 + fast tokenizer |
|--------|------------------|------------------|---------------------|
| TOTAL  | 11279 ms         | 3316 ms          | **2696 ms**         |
| Speedup| 1.0x             | 3.4x             | **4.2x**            |
| D_arch | 6689 ms          | 1098 ms          | 1142 ms             |
| F_weights | 1283 ms       | 1489 ms          | 1254 ms             |
| J_tok_wait | 2720 ms      | 392 ms           | **0 ms**            |
| L_first | 376 ms          | 181 ms           | 170 ms              |

### Production test (test_boot.py)
- Boot: 2.5s (was 11.3s original, 2.8s with V7 only)
- Output: "Paris. The capital of France." — correct
- 292.5 tok/s, 2.69 GB VRAM
- 84 unit tests pass
