# Round 33: KV Cache & Auto-Context Handling

**Date**: 2026-09-03
**Status**: COMPLETE — 37/37 tests passing
**Theme**: Dynamic KV management + automatic context window optimization

## Features Implemented

### R33-1: EvoSparse — Evolving Token Importance KV Cache
- **File**: `research/inference/kv/evo_sparse.py` (256 lines)
- **Source**: ACL 2026 long.530 (iLearn-Lab/ACL26-EvoSparse)
- **What**: Models token importance as a dynamic process across decoding
  steps and layers. Cross-Step Accumulation (decayed running average of
  attention scores) + Cross-Layer Propagation (retrieval heads compute
  query-aware indices, propagate across layers).
- **Key methods**: `update_importance(attention_scores, layer_idx)`,
  `_propagate_importance()`, `_maybe_evict()`
- **Parameters**: `decay=0.95`, `keep_ratio=0.5`, `n_layers=32`,
  `n_sinks=4`, `propagation_weight=0.3`
- **VRAM**: keep_ratio=0.5 → ~1.0GB for V10 at 4096 ctx; importance
  vectors only ~256KB
- **Sink protection**: First 4 tokens + most recent token always retained
  (StreamingLLM finding)

### R33-2: Vegas — Verification-Guided KV Selection
- **File**: `research/inference/kv/vegas_kv.py` (171 lines)
- **Source**: arXiv 2602.07223
- **What**: Reuses attention scores from the verification/full-attention
  phase to identify critical KV entries for the next draft step. Zero-
  overhead KV selection.
- **Key methods**: `set_attention_hints(attention_scores)` — column sums
  → per-position criticality → top-k retention
- **Fallback**: If no hints set, uses sliding window (most recent
  keep_ratio fraction)
- **Parameters**: `keep_ratio=0.5`

### R33-3: HiSparse — Hierarchical HBM-DRAM KV Management
- **File**: `research/inference/kv/hisparse_kv.py` (302 lines)
- **Source**: arXiv 2608.07009 (merged into SGLang)
- **What**: Full KV history in host memory (DRAM), fixed-size GPU cache
  (HBM). LRU eviction from GPU→CPU. Layer-wise prefetching.
- **Key methods**: `prefetch_layer(layer_idx, positions)`, `hit_rate()`
- **Parameters**: `gpu_cache_size` (max GPU slots)
- **VRAM**: Fixed GPU budget regardless of context length. CPU DRAM
  stores full history with pinned memory for fast H2D.
- **Note**: Fused CUDA kernel (hit detection + LRU + H2D inside decode
  CUDA graph) documented as future work. Current implementation is
  synchronous Python reference.

### R33-4: Capture/HybridServe — Activation Cache
- **File**: `research/inference/kv/capture_kv.py` (326 lines)
- **Source**: Capture (casys-kaist), HybridServe (ICCD 2025)
- **What**: Stores input activations instead of K,V. K,V regenerated via
  linear projection on demand. 50% memory reduction per cached block.
- **Three modes**: `"act"` (activations only), `"kv"` (full K/V fallback),
  `"hybrid"` (recent=KV, old=ACT)
- **Key methods**: `set_projection_weights(w_k, w_v)`,
  `append_activation(activation, position)`
- **VRAM**: ACT mode uses ~50% of KV mode (hidden_dim vs 2*n_kv*head_dim)

### R33-5: vToken — Token-Level Virtualization
- **File**: `research/inference/kv/vtoken_kv.py` (281 lines)
- **Source**: arXiv 2608.13263
- **What**: Decouples logical token liveness from physical block
  placement. Token-table indirection + async repacking of live tokens.
  27-72% retained KV block reduction.
- **Key methods**: `mark_dead(positions)`, `repack()` (returns blocks
  freed), `get_live_mask()`
- **Parameters**: `block_size=16`, `max_blocks`
- **Innovation vs paged_eviction**: Token-level (not block-level)
  reclamation. A block with 16 tokens but 3 dead → 13 live repacked,
  full block freed.

### R33-6: AutoContext — Automatic Context Window Management (NOVEL)
- **File**: `research/inference/kv/auto_context.py` (454 lines)
- **Source**: Novel combination — no paper proposes entropy→KV strategy
  selection. Inspired by KVFlow (NeurIPS 2025) + LPC (NeurIPS 2025).
- **What**: Meta-manager that automatically selects and dynamically
  switches KV cache strategies based on 3 signals:
  1. **Context length**: short→standard, medium→snapkv, long→cpu_offload
  2. **Task type** (from entropy trajectory): high→coding→snapkv,
     low→chat→streaming, mixed→rag→snapkv
  3. **VRAM pressure**: <70%→full KV, >70%→compressed, >90%→offload
- **Dynamic hot-swap**: `migrate_kv(old_cache, new_cache)` extracts KV
  from old strategy, injects into new — no context loss
- **Growth prediction**: EMA of per-turn seq_len diffs → pre-emptive
  strategy switch before OOM
- **Novel twist**: Uses the conversation's token-level entropy trajectory
  (already computed for sampling) as the task-type signal. Cross-domain
  combination (entropy signal → KV strategy selection) that no paper
  proposes.
- **Classes**: `AutoContextManager` (meta-manager) +
  `AutoContextKVCache` (KVCacheStrategy wrapper with hot-swap)

## Engine Integration

All 6 new strategies wired into:
- `build_kv_cache()` in `kv_backend.py` — lazy import dispatch
- `ForgeEngine._KV_FALLBACK_CHAIN` in `forge_engine.py` — fallback chains:
  - `evo_sparse` → snapkv → s4r → standard → cpu_offload
  - `vegas` → snapkv → s4r → standard → cpu_offload
  - `hisparse` → cpu_offload → s4r → standard
  - `capture` → cpu_offload → s4r → standard
  - `vtoken` → paged_eviction → snapkv → standard → cpu_offload
  - `auto_context` → s4r → standard → cpu_offload
- Docstring updated with all 21 KV strategy names

## R40 Phase 1: forge/ Package Skeleton
- **Files**: `forge/__init__.py`, `forge/engine/__init__.py`
- **Status**: Skeleton created — re-exports from `research.inference`
- **Bug fix**: `research/self_play/discovery/qwen_adapter.py` line 39 had
  an unterminated string literal (pre-existing). Fixed
  `TOOL_RESP_START`/`TOOL_RESP_END` to proper string literals.
- **Verification**: `from forge.engine import ForgeEngine` works

## Test Results

```
tests/unit/test_round33_kv.py — 37 passed, 0 failed
```

Test coverage:
- EvoSparse: init/append/get, update_importance, eviction, clear, info (5)
- Vegas: init/append/get, attention_hints, fallback, clear (4)
- HiSparse: init/append/get, GPU overflow to CPU, hit_rate, info (4)
- Capture: KV mode fallback, ACT mode with projections, info (3)
- vToken: init/append/get, mark_dead+repack, info, clear (4)
- AutoContext: entropy_to_task_type, 5 strategy selection scenarios,
  maybe_switch, predict_growth, KV cache wrapper (10)
- Dispatch: build_kv_cache for all 6 + engine fallback chain (7)

## Bugs Fixed During Testing

1. **HiSparse `get()`**: `positions.tolist()` called on a list (double
   conversion). Fixed to flatten 2D positions first, then iterate.
2. **CaptureKVCache `get()`**: `N = positions.shape[0]` gave wrong size
   for 2D position tensors (returned batch dim, not total positions).
   Fixed to `N = positions.numel()`. Also fixed mask-based scatter
   assignment to use `torch.where()` for correct index placement.
3. **VTokenKVCache `get()`**: `permute(1,0,2)` failed on 4D tensors from
   2D position indexing. Fixed with `reshape(-1, n_kv, head_dim)` before
   permute.
4. **VTokenKVCache `repack()`**: Dead token flags not cleared after
   repack. Added `self.token_live[:self.seq_len] = True` after repack.
5. **qwen_adapter.py**: Pre-existing unterminated string literal on line
   39. Fixed `TOOL_RESP_START`/`TOOL_RESP_END`.

## VRAM Budget Summary

| Strategy | GPU Memory | CPU Memory | Notes |
|----------|-----------|-----------|-------|
| EvoSparse | keep_ratio × full KV | None | Importance vectors ~256KB |
| Vegas | keep_ratio × full KV | None | Zero-overhead (reuses attn) |
| HiSparse | gpu_cache_size × slot | Full history (pinned) | Fixed GPU budget |
| Capture (ACT) | ~50% of KV | None | Regenerates K/V via matmul |
| vToken | max_blocks × block_size | None | Reclaims dead token blocks |
| AutoContext | Varies (wraps strategy) | Varies | Hot-swaps based on pressure |
