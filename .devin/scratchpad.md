# ForgeAI Complete Bottleneck Map — Refactor Reference

Compiled from 9 subagent scans: GPU sync, Python overhead, VRAM waste, disk I/O, CPU overhead, GPU utilization, concurrency, startup/architecture, Windows/data pipeline, error handling, testing/CI, dependency/build.

---

## TIER 1 — CRITICAL (per-token, every generation)

### T1.1 `.item()` GPU→CPU syncs (30+ locations)
Every generation loop calls `.item()` for EOS checks, causing full GPU pipeline stall per token.
- `decoding.py:72` — `if next_token.item() in eos_set:`
- `self_play_sandbox.py:397,399,403,413,419,421,423` — multiple .item() per token
- `recursive_self_play.py:202,207,210` — log_probs + token sync
- `dspark.py:335,347,349,760,826` — 5 syncs per drafted token
- `fast_infer.py:155`, `cuda_graph.py:192`, `dpo_align.py:194`
- `infinite_curriculum.py:440,460`
- `forge_engine.py:479,526`
**Fix**: Batch EOS check on GPU, only sync when decoding text (every N tokens).

### T1.2 O(n²) tensor growth — KV cache + token sequence
Every token append reallocates entire tensor.
- `kv_backend.py:69-70` — `torch.cat([self.k_cache, k], dim=2)` per token
- `model_loader.py:178-179,293-294,341-342,395-396` — KV cat per layer per token
- `recursive_self_play.py:205` — `torch.cat([cur_ids, next_token...])` per token
- `self_play_sandbox.py:566` — same pattern
- `dspark.py:662-663,853-854` — token cat in generation
- `rotorquant.py:293-294,313-316` — quantized KV cat
**Fix**: Pre-allocate max-length tensor, write by index. O(1) append.

### T1.3 `torch.tensor([[next_token.item()]])` per token
Creates new 1×1 tensor every token — allocation + sync.
- `self_play_sandbox.py:413`
**Fix**: Reuse pre-allocated single-token buffer, write in-place.

### T1.4 Subprocess spawn per code execution (Windows-critical)
Python interpreter startup ~200ms on Windows vs ~10ms on Unix.
- `self_play_sandbox.py:139-144` — subprocess.Popen per task
- `recursive_self_play.py:943-947` — subprocess.run per retry round (up to 5/task)
- `infinite_curriculum.py:705-708` — subprocess.Popen per validation
- `livecodebench_eval.py:175-182` — subprocess.run per test case
- `reasoning_benchmarks.py:134-138` — subprocess.run per benchmark
**Cost**: 200ms × 1500 calls (5 rounds × 300 tasks) = 5 minutes overhead per session.
**Fix**: Persistent worker process with stdin/stdout pipe, or in-process exec with sandboxing.

### T1.5 `torch.sort()` on full vocab for top-p sampling
O(n log n) sort on 151K tokens every token when top_p < 1.0.
- `recursive_self_play.py:119`, `infinite_curriculum.py:452`, `self_play_sandbox.py:347`
**Fix**: Sort only top-k candidates (k=50), not full vocab. Or cache sorted order.

### T1.6 Temp file creation per execution (2 files each)
- `self_play_sandbox.py:90-93` — NamedTemporaryFile + wrapper file
- `recursive_self_play.py:896-973` — same pattern
- `infinite_curriculum.py:699-725` — same pattern
**Cost**: 2 file creates + 2 writes + 2 reads + 2 unlinks per execution.
**Fix**: Persistent temp files reused across executions (overwrite content).

### T1.7 Busy-wait sleep in subprocess polling
- `self_play_sandbox.py:152` — `time.sleep(0.01)` in poll loop
**Fix**: Use `proc.wait(timeout=...)` or async subprocess.

---

## TIER 2 — HIGH (per-forward, every layer)

### T2.1 RMSNorm fp32 upcast — 113 norms × 28 layers
- `model_loader.py:60-62` — `x.float().pow(2).mean(-1,keepdim=True)` per norm
**Cost**: 226 extra kernel launches per forward (~40% overhead). ~50MB temp per norm.
**Fix**: Use `torch.nn.functional.rms_norm` (PyTorch 2.4+) — bf16 native with fp32 accumulation.

### T2.2 MoE expert Python loop — 4 experts × 28 layers
- `moe.py:195-211` — `for i, expert in enumerate(self.experts):` with nonzero(), indexing, scatter
- `speculative_keys.py:339-341,365-368,372-374` — same pattern in speculative paths
**Cost**: 112 Python iterations per forward, each with 3-5 GPU ops on tiny tensors.
**Fix**: Batch all experts with grouped matmul, or stack weights for single matmul in dense_bypass mode.

### T2.3 `.contiguous()` after attention transpose — 4 per layer × 28 layers
- `model_loader.py:218,306,354,410` — `out.transpose(1,2).contiguous().view(B,T,C)`
- `attn_reuse_key.py:226,279,322` — same in reuse path
**Cost**: 112+ memory copies per forward.
**Fix**: Use `.reshape()` instead of `.contiguous().view()` where possible, or SDPA with reshape.

### T2.4 `hasattr()`/`getattr()` in attention forward — 3 checks × 28 layers
- `model_loader.py:283` — `getattr(self, '_qk_norm_identity', True)`
- `attn_reuse_key.py:171,179,202` — `hasattr(self, 'kv_down_proj')`, `getattr(self, 'use_qk_norm', False)`
**Cost**: 84 Python function calls per forward.
**Fix**: Cache as boolean attributes at `__init__`.

### T2.5 RoPE `.to(x.dtype)` conversion per forward
- `model_loader.py:132-133` — `.to(x.dtype)` on cos/sin slices per layer
**Cost**: 56 dtype conversions per forward.
**Fix**: Pre-convert buffers to bf16 at init.

### T2.6 Manual attention (non-SDPA) in 3 locations
- `model_loader.py:207-215` — manual matmul + softmax + matmul (6 kernels) in chunked prefill
- `eagle.py:126-130` — manual attention in draft loop (4 kernels)
- `speculative_keys.py:92-99` — manual low-rank attention (3 kernels)
**Fix**: Use `F.scaled_dot_product_attention` for all paths.

### T2.7 Attention reuse cache clones — 3 GPU copies per cache miss
- `attn_reuse_key.py:126-128` — `.detach().clone()` × 3 per cache entry
- `attn_reuse_key.py:189` — `.detach().clone()` per forward for matching
**Fix**: Use views or share storage where possible.

### T2.8 Adaptive TopK aux loss loop
- `adaptive_topk_key.py:188-195` — rebuilds dispatch_mask in loop over k values
**Cost**: 3 iterations × 5 GPU ops per MoE layer per forward.
**Fix**: Compute directly from routing logits without loop.

### T2.9 Gated DeltaNet per-token sequential loop
- `gated_deltanet_key.py:133-159` — `for t in range(T):` with GPU ops per token
**Cost**: T iterations instead of 1 batched op during prefill.
**Fix**: Chunked formulation (process in blocks, not per-token).

### T2.10 Triton import in forward path (not cached at init)
- `wisparse_key.py:140-147` — `try: from .wisparse_triton import wisparse_fused` in forward
- `wisparse_triton.py:12-13` — no import guard, crashes if triton missing
**Fix**: Cache import at module level. Add try/except guard in wisparse_triton.py.

---

## TIER 3 — MEDIUM (per-batch / per-epoch)

### T3.1 Print statements in training loop — 30+ per epoch
- `self_play_expert_training.py:161-455` — extensive f-string print() calls
**Cost**: Synchronous I/O blocks GPU. ~30 print calls × 5 epochs = 150 blocking I/O calls.
**Fix**: Accumulate logs in buffer, print once per epoch. Or use logging module with buffer.

### T3.2 Regex without pre-compilation — 9+ per task proposal
- `infinite_curriculum.py:474,478,481,485,499,510,557,665,690,779` — re.search/match/finditer
- `recursive_self_play.py:380,384,537` — re.search/findall
- `self_play_sandbox.py:481,483,545,625,631` — re.search/findall
**Cost**: Pattern compilation on every call. 50 task proposals × 9 regex = 450 recompilations.
**Fix**: Pre-compile patterns at module level with `re.compile()`.

### T3.3 JSON dumps/loads in test verification
- `recursive_self_play.py:677-680,689,712,723` — JSON serialize for every test case
**Cost**: Thousands of JSON dumps/loads during training.
**Fix**: Use pickle or direct string formatting for internal data.

### T3.4 Redundant forward passes in DSpark training
- `dspark.py:500-532` — computes target logits, then iterates again for loss
**Cost**: ~30% redundant compute in DSpark training.
**Fix**: Cache target logits, compute loss in same pass.

### T3.5 Checkpoint double-read on save
- `checkpoint_io.py:70-90` — writes .tmp, reads back for verification, then renames
**Cost**: 2× I/O per checkpoint save (7.2GB for 3.6GB checkpoint).
**Fix**: Make verification optional, or verify metadata only (shapes/dtypes from header).

### T3.6 `eval()` in curriculum parsing
- `infinite_curriculum.py:503-504,514-515` — `eval()` per test case pattern
**Cost**: Python eval overhead × 10 test cases × 50 tasks = 500 eval calls.
**Fix**: Use `ast.literal_eval()` (safer + faster for literals).

### T3.7 No torch.compile in self-play training path
- `self_play_expert_training.py` — no torch.compile, no CUDA graphs
- Only used in `forge_engine.py:248` and `model_loader.py:699,769` (not training)
**Cost**: Missing 1.3-2x decode speedup from kernel fusion + reduced Python overhead.
**Fix**: Apply `torch.compile(model, mode="reduce-overhead")` to generation path. Note: dynamic KV shapes may cause recompilation — use `dynamic=True` or static shapes with padding.

### T3.8 No CUDA graphs in generation loop
- CUDA graphs exist (`runtime/cuda_graph.py`) but not used in self-play
- `fast_infer.py:80` uses them, but self-play doesn't call FastInferenceEngine
**Cost**: Missing ~30-50% decode speedup from eliminating kernel launch overhead.
**Fix**: Use CudaGraphRunner for the single-token decode path in generation loops.

### T3.9 Single CUDA stream — no overlap
- `cuda_graph.py:75` — only one stream created (for graph capture)
- No multi-stream usage anywhere in generation
**Cost**: No overlap of data transfers with compute.
**Fix**: Use separate streams for H2D transfers and compute kernels.

### T3.10 1121 print() calls, zero logging module usage
- No `logging.info/debug/warning/error` calls anywhere in `research/`
- 1121 `print()` calls across the codebase
**Cost**: No log levels, no structured logging, no file output, synchronous I/O.
**Fix**: Replace print() with logging module. Use WARNING/INFO/DEBUG levels.

---

## TIER 4 — CONCURRENCY / PARALLELISM (architectural)

### T4.1 Sequential expert loading (28 files, no parallelism)
- `airmoe_infinite.py:196-219` — loads 28 expert files sequentially from disk
**Cost**: 56MB sequential read. Could be 10-20x faster with ThreadPoolExecutor.
**Fix**: `ThreadPoolExecutor(max_workers=8)` for parallel disk reads.

### T4.2 Sequential SVD compression (28 layers, CPU-bound)
- `self_play_expert_training.py:852-878` — SVD compression per layer, sequential
**Cost**: 28 SVD decompositions on CPU, sequential.
**Fix**: `ProcessPoolExecutor` for parallel SVD across layers.

### T4.3 No CPU-GPU overlap in generation
- `recursive_self_play.py:172` — tokenization (CPU) blocks before GPU generation
- All generation loops: CPU parsing/decoding blocks GPU
**Fix**: Pipeline: tokenize next prompt while GPU generates current. Use pinned memory + non_blocking transfers.

### T4.4 No pipeline parallelism in self-play
- `recursive_self_play.py:345-390` — generate → execute → verify, all sequential
- `infinite_curriculum.py:238-272` — propose → validate, sequential
**Fix**: Producer-consumer: propose next task while validating current. Async subprocess for execution.

### T4.5 No async/await anywhere in Python code
- Only in frontend JavaScript (`app.js`, `tools.js`)
**Fix**: Use `asyncio.create_subprocess_exec()` for sandbox execution to not block GPU.

### T4.6 Prefetch infrastructure exists but unused
- `training_utils.py:56-62` — BinaryDataset has prefetch support, default `prefetch=0`
- `airmoe_key.py:662-670` — `prefetch()` method defined but never called
- `airllm_key.py:232` — AirLLM has prefetch (good), but AirMoE doesn't use it
**Fix**: Enable prefetch in training. Call AirMoE prefetch based on router predictions.

### T4.7 No batch generation for self-consistency
- `recursive_self_play.py:184-213` — generates one token at a time, single sample
- k_samples for self-consistency done sequentially, not batched
**Fix**: Batch k independent generations in parallel (if VRAM allows).

### T4.8 Sequential model + tokenizer loading
- `self_play_expert_training.py:990-991` — model.to("cuda") then tokenizer load
**Fix**: Load tokenizer in parallel thread while model loads.

---

## TIER 5 — STARTUP / INITIALIZATION

### T5.1 Eager torch import in config.py
- `config.py:29` — `device: str = "cuda" if __import__("torch"...).cuda.is_available() else "cpu"`
**Cost**: Triggers torch CUDA init (~2-3s) on every `import research.config`.
**Fix**: Move to lazy function: `def get_device(): ...`

### T5.2 All 12 MODEL_CONFIGS instantiated at import time
- `config.py:82-261` — 12 ModelConfig objects with validation at module level
**Fix**: Lazy instantiation — only create config when `get_config()` is called.

### T5.3 Tokenizer loaded 15+ times, no caching
- 15+ files call `AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")`
**Cost**: 0.5-1s per load, no cross-module caching.
**Fix**: Create `research/tokenizer_cache.py` with `get_tokenizer(path)` LRU cache.

### T5.4 Keys/__init__.py imports all 18 key classes eagerly
- `keys/__init__.py:1-56` — imports all key implementations at module level
**Fix**: Use lazy imports or registry pattern with `importlib.import_module()`.

### T5.5 GoalTaskGenerator singleton at module level
- `self_play_expert_training.py:45` — `_goal_gen = GoalTaskGenerator()` at import
**Fix**: Move inside `main()` or use lazy initialization.

### T5.6 Sequential key application (28 model walks per key)
- `self_play_expert_training.py:1002-1024` — each key traverses full model graph
**Fix**: Create `KeyBatch` class that applies multiple keys in single model traversal.

### T5.7 Mandatory 16-token warmup before training
- `vram_manager.py:164-209` — runs full generation loop for VRAM profiling
**Fix**: Cache VRAM profile per config signature. Make warmup optional if profile cached.

### T5.8 Test blocks in key files (72 files with __main__ blocks)
- Heavy test code in `model_loader.py:842`, `airmoe_key.py:697`, `bake_v4.py:485`
**Fix**: Move test blocks to `tests/` directory to reduce import-time overhead.

---

## TIER 6 — VRAM / MEMORY WASTE

### T6.1 fp32 temporary tensors in RMSNorm
- `model_loader.py:60-62` — fp32 upcast for variance
**Cost**: ~50MB temp per layer, freed immediately but causes allocator pressure.
**Fix**: Same as T2.1 — use F.rms_norm.

### T6.2 fp32 KV cache in SnapKV
- `snapkv.py:67,93` — attention scores stored as fp32
**Cost**: 2× VRAM for attention scores (~128MB for 4096 tokens).
**Fix**: Use bf16 or fp16.

### T6.3 KV cache dequant to fp32
- `kv_compress.py:64` — `kv_quant.to(torch.float32) * scale`
**Fix**: Dequant directly to bf16.

### T6.4 SVD decompression upcasts to fp32
- `airmoe_key.py:249-251` — `U.float().to(device)`, `S.float().to(device)`, `Vh.float().to(device)`
**Cost**: 3× fp32 upcast per expert load.
**Fix**: Compute in bf16 with fp32 accumulation only for the matmul.

### T6.5 Forward cache GPU→CPU for hashing
- `forward_cache.py:45` — `input_ids[0].cpu().tolist()` for cache key
**Fix**: Use GPU tensor hash (tensor.sum().item() as cheap hash, or torch.hash).

### T6.6 Model cache deepcopy (3.2GB RAM)
- `model_loader.py:688-690` — full model copy for architecture cloning
**Note**: Now on CPU (fixed). Only during model loading, not hot path.

### T6.7 Duplicate RoPE buffers (non-persistent)
- `model_loader.py:87-88` — each layer creates cos/sin buffers (persistent=False)
**Note**: Already addressed by `rope_share_key.py`. Buffers not in checkpoint.

### T6.8 All training data in-memory, no streaming
- `self_play_expert_training.py:267-268` — `successes = []`, `failures = []` in memory
- `self_play_sandbox.py:245` — `self.packets: List[Dict] = []` in memory
**Cost**: OOM risk for large datasets. No incremental checkpointing.
**Fix**: Stream to disk (JSONL append) instead of accumulating in memory.

---

## TIER 7 — DISK I/O

### T7.1 AirLLM streaming — per-layer per-token disk load
- `forge_engine.py:500-520` — loads layer shard from disk for EVERY token
**Cost**: 28 layers × 100 tokens = 2,800 disk loads. ~280GB transferred.
**Note**: Only used when model doesn't fit in VRAM. Not relevant for 1.5B on 12GB GPU.
**Fix**: Cache layers in VRAM, or async prefetch next layer while computing current.

### T7.2 AirMoE expert loading — 28 files per topic switch
- `airmoe_infinite.py:196-222` — loads 28 expert files (~56MB) per topic
**Note**: LRU cache exists. Prefetch method defined but never called.
**Fix**: Call existing `prefetch()` method based on router predictions.

### T7.3 AirMoE expert saving — 28 file writes per topic
- `self_play_expert_training.py:852-874` — saves 28 separate .safetensors files
**Fix**: Batch into single file with tensor namespacing.

### T7.4 AirMoE module build — 112 file writes
- `airmoe_key.py:339-365` — 28 layers × 4 experts = 112 separate writes
**Fix**: Batch into fewer files (one per layer, or one per topic).

### T7.5 Orphaned .tmp files
- `checkpoint_io.py:70-96` — .tmp files left if process crashes
**Fix**: Scan and clean .tmp files on startup.

### T7.6 JSONL log files accumulate (no rotation)
- `self_play_sandbox.py:808-816` — new timestamped JSONL file per session
**Fix**: Implement log rotation policy, or append to single file.

### T7.7 Manifest.json loaded from disk every time
- `airmoe_infinite.py:62-64`, `airmoe_key.py:186-187` — no caching
**Fix**: Cache manifest in memory (module-level singleton).

---

## TIER 8 — WINDOWS-SPECIFIC

### T8.1 17 hardcoded absolute Windows paths
- `recursive_self_play.py:69,888,1004` — `"D:/windsurf/ForgeAI/.devin/tmp"`
- `infinite_curriculum.py:186` — `"D:/windsurf/ForgeAI/research/data/curriculum"`
- `self_play_expert_training.py:71,1051` — hardcoded data/expert paths
- `reasoning_benchmarks.py:49,50,125` — hardcoded cache/temp dirs
- `livecodebench_eval.py:45,157,294` — hardcoded cache/temp dirs
- `vram_manager.py:55` — hardcoded torch cache dir
- `airmoe_infinite.py:435` — hardcoded v4 checkpoint path
- `airmoe_hotswap.py:196,295` — hardcoded module paths
**Impact**: Code won't run on any other machine.
**Fix**: Replace with `Path(__file__).parent.parent / "data"` or environment variables.

### T8.2 Manual path normalization instead of pathlib
- `launch.py:66,193` — `rel.replace('\\', '/').split('/')`
**Fix**: Use `pathlib.Path` consistently for all path operations.

### T8.3 Windows path length limits (260 chars)
- Nested temp paths could exceed Windows MAX_PATH
**Fix**: Use `\\?\` prefix for long paths, or keep temp paths short.

### T8.4 No multiprocessing (good for Windows)
- No `multiprocessing` usage found — avoids spawn overhead on Windows.
**Status**: ✅ Correct decision. Use ThreadPoolExecutor for I/O, ProcessPoolExecutor only for CPU-bound SVD.

---

## TIER 9 — ARCHITECTURE / CODE QUALITY

### T9.1 God class: ExpertSelfPlayTrainer (861 lines)
- `self_play_expert_training.py:59-920`
- Responsibilities: training loop, self-play orchestration, expert loading, benchmark eval, curriculum
**Fix**: Split into `SelfPlayOrchestrator`, `ExpertLoader`, `BenchmarkEvaluator`, `CurriculumManager`.

### T9.2 God function: train_topic (784 lines)
- `self_play_expert_training.py:136-920`
**Fix**: Split into `generate_tasks_for_topic()`, `run_self_play_epoch()`, `fine_tune_expert()`, `evaluate_benchmarks()`.

### T9.3 God class: InfiniteAirMoE (352 lines)
- `airmoe_infinite.py:125-477`
- Responsibilities: expert routing, LRU caching, disk I/O, SVD decompression, manifest management
**Fix**: Split caching logic into `ExpertCache` class.

### T9.4 10 nn.Module subclasses in one file
- `model_loader.py` — RMSNorm, RoPE, Attention variants, Block, ConfigurableResearchLLM, etc.
**Fix**: Split into `research/model/norm.py`, `attention.py`, `block.py`, `model.py`.

### T9.5 8 KV cache strategies in one file
- `kv_backend.py` — StandardKVCache, PagedKVCache, H2OCache, SnapKVCache, etc.
**Fix**: Split into `research/kv/` package with one file per strategy.

### T9.6 Duplicate code: AutoTokenizer loading (15+ locations)
**Fix**: Centralize via `research/tokenizer_cache.py`.

### T9.7 Duplicate code: Model loading pattern (10+ locations)
**Fix**: Create `load_default_model(config_name, checkpoint_path)` helper.

### T9.8 Direct safetensors imports (30+ files) instead of checkpoint_io
**Fix**: Enforce usage of `checkpoint_io` module for all tensor I/O.

### T9.9 No data augmentation
- No code transformation, test case perturbation, or prompt rephrasing
**Opportunity**: Add augmentations to increase effective dataset size.

### T9.10 Individual sample processing instead of true batching
- `self_play_expert_training.py:719-773` — processes samples one-by-one with gradient accumulation
**Fix**: Implement true batching with DataLoader for training efficiency.

### T9.11 33 files with sys.path.insert hacks
- `self_play_expert_training.py:29`, `recursive_self_play.py:49,980`, `infinite_curriculum.py:48`, `self_play_sandbox.py:62,850`, `bake_v4.py:40`, `chat_ui.py:15`, `airmoe_infinite.py:425`, `airmoe_hotswap.py:276`, `reasoning_benchmarks.py:45`, `livecodebench_eval.py:39`, `thinking_model.py:406`, plus root scripts and .devin/ scripts
**Impact**: Breaks IDE auto-completion, test discovery, distribution.
**Fix**: Create `pyproject.toml`, make project installable with `pip install -e .`, remove all sys.path hacks.

---

## TIER 10 — NUMERICAL / CORRECTNESS

### T10.1 Mixed precision in SVD storage
- `airmoe_key.py:240-241` — U in bf16, S in fp16, Vh in bf16
- `bake_v4.py:169,180` — same mixed precision
**Risk**: Precision mismatch between U (bf16) and S (fp16) in reconstruction.
**Fix**: Use consistent dtype (all bf16 or all fp16).

### T10.2 Quantization dequant precision loss
- `kv_compress.py:64` — dequant to fp32 then use in attention (bf16)
- `kv4bit_key.py:143-144` — dequant to fp16, may lose precision vs fp32
**Risk**: Accumulated precision loss in long sequences.
**Fix**: Dequant directly to compute dtype, use fp32 accumulation in attention.

### T10.3 Bitnet ternary unpack to fp32
- `bitnet.py:201` — `packed.to(torch.float32)` for ternary values
**Fix**: Use bf16 (values are -1, 0, 1 — no precision needed).

---

## TIER 11 — ERROR HANDLING / RESILIENCE

### T11.1 31 silent `except Exception: pass` — hide bugs
- `recursive_self_play.py:588-589` — swallows reference execution errors
- `infinite_curriculum.py:506-507,517-518` — swallows test case parsing errors
- `self_play_sandbox.py:177-178,293-294` — swallows subprocess cleanup + confidence scoring
- `checkpoint_io.py:26-27,32-33,153-154` — swallows import + load errors
- `launch.py:91-92,145-146,169-170,204-205,226-227,238-239,255-256` — 7 silent swallows in HTTP server
- `task_logger.py:54-55,62-63` — swallows log write errors
**Impact**: Bugs hidden, cascading failures, false positives in training data, zombie processes.
**Fix**: Replace with logging.warning() + specific exception types.

### T11.2 Unsafe pickle fallback in checkpoint load (SECURITY RISK)
- `checkpoint_io.py:153-154` — falls back to `weights_only=False` (arbitrary code execution)
**Fix**: Warn user before fallback. Never auto-fallback to unsafe pickle.

### T11.3 No CUDA OOM handling in training loops
- `self_play_expert_training.py:286-375` — no try/except around model forward
- `recursive_self_play.py:184-225` — no try/except around generation
**Impact**: Single OOM crashes entire session without checkpoint save.
**Fix**: Add `except torch.cuda.OutOfMemoryError` handler with `empty_cache()` + skip sample.

### T11.4 No SIGTERM handler in main training script
- `dpo_align.py:321-325` has SIGINT handler (good)
- `self_play_expert_training.py` has NO signal handler
**Impact**: `kill` command loses all training progress.
**Fix**: Add SIGTERM + SIGINT handlers with emergency checkpoint save.

### T11.5 Retry loops without backoff
- `infinite_curriculum.py:770-816` — fixed temperature increase, no backoff
- `recursive_self_play.py:345-380` — fixed retry loop, no jitter
**Fix**: Add exponential backoff + jitter to retry loops.

### T11.6 Subprocess cleanup doesn't force-kill
- `self_play_sandbox.py:172-178` — `proc.wait(timeout=5)` may timeout without force-kill
- `recursive_self_play.py:961-964` — TimeoutExpired handler doesn't kill
**Fix**: Add `proc.terminate()` then `proc.kill()` fallback after timeout.

### T11.7 VRAM leaks on error paths
- `self_play_expert_training.py:719-773` — no VRAM cleanup if backward pass fails
- `recursive_self_play.py:184-225` — cleanup only if loop completes (finally helps but tensors may leak before)
**Fix**: Wrap critical sections in try/finally with `del` + `empty_cache()`.

### T11.8 Error messages lack context
- `model_loader.py:482,495` — "Unknown attention type" without valid options
- `airmoe_key.py:560,604-606` — "Expert file not found" without available topics
- `checkpoint_io.py:59,76,80,82,89` — generic errors without checkpoint path
**Fix**: Include valid options, file paths, and state in error messages.

---

## TIER 12 — TESTING / CI INFRASTRUCTURE (CRITICAL FOR REFACTOR)

### T12.1 No test framework — all tests manual
- No pytest, no unittest in requirements.txt
- All test files run via `python test_xxx.py` manually
**Fix**: Add pytest to requirements, convert tests to pytest format.

### T12.2 No tests/ directory — tests scattered
- Test files in root (`test_integration_keys.py`, `test_novel_keys.py`, `test_novel_keys2.py`)
- Test files in `.devin/` (6 files)
- Test files in `research/keys/`, `research/evaluation/`
**Fix**: Create `tests/` directory, move all test files there.

### T12.3 Core modules have ZERO tests
- `research/config.py` — NO TESTS
- `research/model_loader.py` (842 lines) — NO TESTS
- `research/checkpoint_io.py` — NO TESTS
- `research/training/` — NO TESTS
- `research/self_play/` — NO TESTS
- `research/inference/` — NO TESTS
- `research/decoding/` — NO TESTS
- `research/quantization/` — NO TESTS
- `research/moe/` — NO TESTS
- `research/runtime/` — NO TESTS
- `research/serving/` — NO TESTS
**Impact**: Refactor has no safety net.
**Fix**: Write tests for config, model_loader, checkpoint_io first.

### T12.4 No CI/CD
- No `.github/workflows/`, no `.gitlab-ci.yml`
- No automated test execution on commit
**Fix**: Add GitHub Actions workflow for test + lint.

### T12.5 No linting or formatting config
- No `.flake8`, no `ruff.toml`, no `pyproject.toml [tool.black]`, no `.isort.cfg`
**Fix**: Add ruff (lint + format in one tool).

### T12.6 No type checking enforcement
- 152 files have type hints but no mypy/pyright config
- No `py.typed` markers, no stub files
**Fix**: Add pyright config (faster than mypy, IDE-compatible).

### T12.7 No pre-commit hooks
- No `.pre-commit-config.yaml`
**Fix**: Add pre-commit with ruff + pytest --fast.

### T12.8 No test fixtures or conftest.py
- No shared model loading helpers, no mock objects
- Cosine similarity functions duplicated across test files
**Fix**: Create `tests/conftest.py` with shared fixtures.

### T12.9 No performance regression tests
- `ablation_benchmark.py` exists but no baseline comparison
**Fix**: Add performance regression test with golden baseline.

### T12.10 No coverage measurement
- No `.coveragerc`, no pytest-cov
**Fix**: Add pytest-cov, set minimum 60% coverage threshold.

---

## TIER 13 — DEPENDENCY / BUILD / PACKAGING

### T13.1 No pyproject.toml — project not installable
- No `setup.py`, `setup.cfg`, or `pyproject.toml`
- Cannot `pip install -e .`
**Fix**: Create `pyproject.toml` with `[project]` + `[tool.setuptools.packages.find]`.

### T13.2 6 unused dependencies in requirements.txt
- `tokenizers==0.21.4` — never imported (transitive)
- `accelerate==1.9.0` — never imported (transitive)
- `huggingface_hub==0.30.0` — never imported (transitive)
- `wandb==0.19.11` — never imported
- `tqdm==4.67.1` — never imported
**Fix**: Remove from requirements.txt (transitive deps auto-installed).

### T13.3 Missing dependencies from requirements.txt
- `safetensors` — used in 54+ files, not in requirements (relies on transformers transitive)
- `gradio` — used in chat_ui.py, not in requirements
**Fix**: Add to requirements.txt explicitly.

### T13.4 Triton not guarded in wisparse_triton.py
- `wisparse_triton.py:12-13` — `import triton` with no try/except
- Will crash if triton not installed
**Fix**: Add try/except ImportError with fallback to PyTorch ops.

### T13.5 Bleeding-edge version pins
- `torch==2.8.0+cu128` — very new, requires CUDA 12.8
- `transformers==4.51.3` — outdated by 1 major version (current 5.8.1)
- `numpy==2.2.6` — NumPy 2.x has breaking changes
**Fix**: Pin to stable versions. Test version bumps carefully.

### T13.6 No environment.yml or Dockerfile
- No conda environment spec, no containerization
**Fix**: Add `environment.yml` for conda users, `Dockerfile` for containerized deployment.

### T13.7 No .python-version file
- `setup_env.ps1` says Python 3.13, no explicit pin
**Fix**: Add `.python-version` file with `3.13`.

---

## SUMMARY BY RESOURCE

| Resource | # Issues | Top bottleneck | Est. waste |
|----------|----------|---------------|------------|
| **GPU sync** | 30+ | `.item()` per token | 10-15% gen speed |
| **GPU kernels** | 10 | 565 launches/token (should be ~250) | 2x kernel overhead |
| **VRAM** | 8 | KV cache O(n²), fp32 temps | ~2GB waste |
| **Disk I/O** | 7 | AirLLM 280GB/100tok, 28 files/topic | Major for offload |
| **CPU/Python** | 12 | subprocess spawn, regex recompilation | 60s+ per session |
| **Concurrency** | 8 | No parallelism anywhere | 10-20x expert load |
| **Startup** | 8 | Eager torch import, no tokenizer cache | 4-6s wasted |
| **Windows** | 4 | 17 hardcoded paths, subprocess overhead | Portability + 60s |
| **Architecture** | 11 | 3 god classes, 784-line function, 33 sys.path hacks | Maintainability |
| **Numerical** | 3 | Mixed precision in SVD | Correctness risk |
| **Error handling** | 8 | 31 silent excepts, no OOM handler, no SIGTERM | Session loss risk |
| **Testing** | 10 | No framework, no CI, core modules untested | Refactor risk HIGH |
| **Dependencies** | 7 | Not installable, 6 unused, 2 missing, triton unguarded | Installation fails |

---

## REFACTOR PRIORITY (by ROI for large refactor)

### Phase 0: Safety net (BEFORE any refactor)
| # | Fix | Effort | Risk reduction |
|---|-----|--------|---------------|
| 0a | Create pyproject.toml, make installable | ~30 lines | Enables all tooling |
| 0b | Add pytest + tests/ directory | ~50 lines | Test discovery works |
| 0c | Write tests for config, model_loader, checkpoint_io | ~200 lines | Core safety net |
| 0d | Add ruff lint + format config | ~10 lines | Code consistency |
| 0e | Add pre-commit hook | ~15 lines | Quality gate |
| 0f | Add GitHub Actions CI | ~30 lines | Automated checks |

### Phase 1: Hot path (generation loop) — biggest speedup
| # | Fix | Effort | Est. speedup |
|---|-----|--------|-------------|
| 1 | Pre-allocated KV cache | ~50 lines | 2-3x gen at 200+ tokens |
| 2 | Batch EOS (no .item() per token) | ~20 lines/file | 10-15% generation |
| 3 | F.rms_norm (eliminate fp32 upcast) | 3 lines | 15-20% forward |
| 4 | Pre-allocate token buffer | ~5 lines | Eliminates per-token alloc |
| 5 | torch.compile on generation path | ~10 lines | 1.3-2x decode |
| 6 | CUDA graphs for single-token decode | ~30 lines | 30-50% decode |

### Phase 2: Forward pass — kernel reduction
| # | Fix | Effort | Est. speedup |
|---|-----|--------|-------------|
| 7 | Cache hasattr/getattr at init | ~10 lines | 5% forward |
| 8 | Pre-convert RoPE to bf16 | 2 lines | 3% forward |
| 9 | SDPA for manual attention | ~20 lines | 3-6 kernels → 1 |
| 10 | Batch MoE expert dispatch | ~80 lines | 20% FFN |
| 11 | Eliminate .contiguous() where possible | ~20 lines | 112 fewer copies |
| 12 | Cache Triton import at module level | ~5 lines | Removes per-forward try/except |

### Phase 3: Execution pipeline — Windows + concurrency
| # | Fix | Effort | Est. speedup |
|---|-----|--------|-------------|
| 13 | Persistent worker process | ~100 lines | 60s+ per session |
| 14 | Pre-compile regex patterns | ~30 lines | 450 fewer recompilations |
| 15 | Parallel expert loading | ~20 lines | 10-20x disk I/O |
| 16 | Parallel SVD compression | ~30 lines | 5-10x CPU |
| 17 | Async subprocess (asyncio) | ~50 lines | 1.3-1.5x pipeline |
| 18 | CPU-GPU overlap (pinned mem) | ~40 lines | 1.2-1.5x |

### Phase 4: Startup + architecture
| # | Fix | Effort | Est. speedup |
|---|-----|--------|-------------|
| 19 | Lazy torch import in config | ~5 lines | 2-3s startup |
| 20 | Global tokenizer cache | ~15 lines | 0.5-1s per load |
| 21 | Lazy MODEL_CONFIGS | ~20 lines | Reduced import overhead |
| 22 | Key batch application | ~50 lines | Single model traversal |
| 23 | Split ExpertSelfPlayTrainer | ~200 lines | Maintainability |
| 24 | Split train_topic (784 lines) | ~150 lines | Maintainability |
| 25 | Replace hardcoded paths | ~30 lines | Portability |
| 26 | Remove sys.path hacks (33 files) | ~50 lines | IDE + test discovery |

### Phase 5: Data pipeline + VRAM
| # | Fix | Effort | Est. speedup |
|---|-----|--------|-------------|
| 27 | Top-p sort only top-k | ~10 lines | O(n log n) → O(k log k) |
| 28 | Replace print with logging | ~100 lines | Removes I/O blocking |
| 29 | Stream training data to disk | ~50 lines | OOM prevention |
| 30 | True batching with DataLoader | ~80 lines | Training efficiency |
| 31 | fp32 → bf16 in SnapKV/KV dequant | ~5 lines | 2x VRAM for KV |
| 32 | Cache VRAM profile per config | ~20 lines | Skip warmup |

### Phase 6: Error handling + resilience
| # | Fix | Effort | Est. speedup |
|---|-----|--------|-------------|
| 33 | Add CUDA OOM handler in training | ~20 lines | Session save on OOM |
| 34 | Add SIGTERM/SIGINT handler | ~15 lines | Emergency checkpoint |
| 35 | Replace silent excepts with logging | ~60 lines | Bug visibility |
| 36 | Fix unsafe pickle fallback | ~5 lines | Security |
| 37 | Add backoff to retry loops | ~20 lines | Prevent OOM cascades |
| 38 | Force-kill subprocess on timeout | ~10 lines | No zombie processes |

### Phase 7: Dependencies + packaging
| # | Fix | Effort | Est. speedup |
|---|-----|--------|-------------|
| 39 | Remove 6 unused dependencies | ~6 lines | Cleaner install |
| 40 | Add missing deps (safetensors, gradio) | ~2 lines | Install works |
| 41 | Guard triton import | ~5 lines | No crash if missing |
| 42 | Add environment.yml | ~20 lines | Conda support |
| 43 | Add Dockerfile | ~30 lines | Containerization |

### Phase 8: Polish
| # | Fix | Effort | Est. speedup |
|---|-----|--------|-------------|
| 44 | Orphaned .tmp cleanup on startup | ~15 lines | Disk hygiene |
| 45 | Log rotation for JSONL | ~20 lines | Disk hygiene |
| 46 | Manifest.json caching | ~10 lines | Minor |
| 47 | Data augmentation | ~100 lines | Dataset size |
| 48 | Multi-CUDA-stream generation | ~80 lines | 1.5-2x with overlap |
| 49 | Split model_loader.py into package | ~200 lines | Maintainability |
| 50 | Split kv_backend.py into package | ~100 lines | Maintainability |
| 51 | Add type checking (pyright) | ~20 lines config | Type safety |
| 52 | Add performance regression tests | ~100 lines | Catch slowdowns |
| 53 | Add property-based tests (hypothesis) | ~100 lines | Edge case coverage |
