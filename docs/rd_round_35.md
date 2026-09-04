# Round 35 — I/O Optimizations & Boot Performance

## Overview
Reduce time-to-first-token and improve memory utilization through progressive
loading, virtual tensor abstraction, and asymmetric memory paging for hybrid
(KV+SSM) models. Plus GUI boot improvements (lazy page construction, splash).

## Features

### R35-1: Lazy GUI Page Construction
- **File**: `forge_gui/app.py`
- **Problem**: `MainWindow.__init__` eagerly constructed all 14 GUI pages,
  blocking window appearance until every page's imports + CUDA init + disk
  scans completed.
- **Solution**: Factory-based lazy construction. Pages are built on-demand
  when first selected via `_ensure_page(idx)`. Placeholder widgets inserted
  at init; replaced with real pages on first visit.
- **Impact**: Window appears immediately; only the visible page pays
  construction cost.

### R35-2: Splash Screen
- **File**: `forge_gui/app.py` (`run()`)
- **Problem**: No visual feedback during QApplication + MainWindow init.
- **Solution**: `QSplashScreen` shown before MainWindow construction,
  finished via `splash.finish(win)` after window appears.
- **Impact**: User sees branding immediately on launch.

### R35-3: eLLM — Virtual Tensor Abstraction
- **File**: `research/inference/memory/virtual_tensor.py`
- **Paper**: arXiv 2506.15155
- **Problem**: GPU memory is a hard ceiling for batch size and context
  length. CPU RAM (32GB) sits mostly idle during inference.
- **Solution**: `VirtualTensor` transparently spans GPU and CPU memory.
  Elastic inflation/deflation moves elements between GPU (hot) and CPU
  (cold) on demand. `VirtualTensorPool` manages multiple tensors with a
  shared GPU budget and access-pattern-based rebalancing.
- **Key classes**: `VirtualTensor`, `VirtualTensorPool`
- **VRAM budget**: GPU holds hot portion; CPU (pinned) holds cold portion.
  On 12GB + 32GB, enables ~3x larger effective memory.
- **Tests**: 8 (init, allocate, inflate/deflate, device_location, to_gpu/cpu,
  memory_stats, pool create, pool rebalance)

### R35-4: AVMP — Asymmetric Virtual Memory Paging
- **File**: `research/inference/memory/avmp.py`
- **Paper**: arXiv 2605.22416
- **Problem**: Hybrid models (KV cache + SSM state) share a single GPU
  memory pool. KV grows with context; SSM is fixed. A single pool can't
  adapt when one component needs more.
- **Solution**: `AVMPManager` separates KV and SSM into physically distinct
  pools behind a unified virtual address space. On allocation failure,
  capacity migrates from the pool with spare to the pool that's full.
  `rebalance()` adjusts pool sizes based on usage pressure.
- **Key classes**: `AVMPManager`, `AVMPPool`
- **VRAM budget**: 12GB split 60/40 (KV/SSM) by default. Cross-pool
  migration on OOM. 7.6% OOM reduction, 1.83-13.3x throughput.
- **Tests**: 8 (init, request_kv, request_ssm, migration, pressure, free,
  rebalance, stats)

### R35-6: Progressive GGUF Loading
- **File**: `research/inference/loader/progressive_loader.py`
- **Problem**: Loading all model tensors before first forward pass blocks
  time-to-first-token.
- **Solution**: `ProgressiveLoader` loads only essential tensors (embedding
  + first 25% of layers + final layer + unembedding) to GPU initially.
  Remaining tensors stream in background via a daemon thread. On-demand
  loading for any tensor not yet loaded.
- **Key classes**: `ProgressiveLoader`
- **VRAM budget**: Only essential tensors on GPU initially (~30% of model).
  Rest streams in background.
- **Tests**: 7 (init, essential computation, load_essential, on-demand,
  background, is_loaded, stats)

## Bug Fix: threading.Lock Deadlock
- **Root cause**: `ProgressiveLoader.stats()` acquired `self._lock` then
  called `self.loading_progress()` which also acquired `self._lock`.
  `threading.Lock` is non-reentrant → deadlock.
- **Fix**: Changed `threading.Lock()` → `threading.RLock()` in both
  `progressive_loader.py` and `virtual_tensor.py`.
- **Lesson**: Any lock that protects methods calling other locked methods
  must be reentrant.

## Test Results
- **R35 tests**: 23 passed (8 VirtualTensor + 2 Pool + 8 AVMP + 7 ProgressiveLoader)
- **Full suite**: 121 passed (R32+R33+R34+R35), 0 failed

## Files Created
- `research/inference/memory/virtual_tensor.py` (251 lines)
- `research/inference/memory/avmp.py` (159 lines)
- `research/inference/loader/progressive_loader.py` (172 lines)
- `tests/unit/test_round35_io.py` (233 lines)

## Files Modified
- `forge_gui/app.py` (lazy page construction + splash screen)
