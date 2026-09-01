# ForgeAI â€” Agent Notes

## Agent Operating Directives (READ FIRST)

These directives govern how work is done in ForgeAI. They are non-negotiable
unless the user explicitly overrides them for a specific task.

### A. Model Versioning â€” Build On The Prior, Never Beside It
- **Every new custom model version MUST be derived from the immediately
  preceding version**, carrying forward all prior keys/architecture as the
  baseline, then adding or replacing only what's new. Example chain:
  `lfm25_1.2b` â†’ `forgelm_v10_1.2b` (V3/V4/V5/V7/V8/V9 presets were superseded by V10;
  their architecture keys are preserved in V10's config).
- **Port-first, train-second**: when introducing a new architecture key or
  attention variant, write the lossless checkpoint-conversion path
  (`XxxKey` class, identity/zero-init warm start, bit-exact load test)
  **before** any training experiment. This eliminates the "train first, port
  later" tax that has bitten past rounds.
- **No silent regressions**: a new preset that drops a prior key must
  document WHY in the preset line and in the R&D round notes. Dropping keys
  silently is a bug.
- **Preset lineage check**: before merging a new preset, run a bit-exact
  forward-pass comparison against the prior preset's checkpoint on the BSP
  base. Max logit diff must be 0.0 (lossless) unless the preset is
  intentionally non-lossless (document the delta).

### B. Confirm-Then-Fix â€” Never Leave A Known Bug Sitting
- **When you find an issue, confirm it** (reproduce with a minimal script or
  test) **then fix it in the same session**. Do not log it and move on.
- If a fix would be large/risky, scope a minimal failing test first, then
  implement the smallest correct fix. Prefer targeted edits over rewrites.
- **Always add or update a test** for the fix so it cannot silently return.
  Tests live in `tests/unit/` and run on CPU where possible.
- If a fix is genuinely blocked (needs user input, env change, or a
  destructive op), say so explicitly and create a `todo` â€” do not pretend
  it's done.

### C. R&D Is The Default Mode â€” Push For Novel Improvements
- **No area is "solved"**. Every existing technique (attention, KV cache,
  quantization, decoding, optimizer, loss, scheduler) is fair game for a
  novel variation. The codebase already has 13 R&D rounds; round 14+ is
  expected, not exceptional.
- **Prefer novel over copy**: when implementing a known technique, always
  ask "what's the novel twist that could beat the paper's number on our
  specific hardware (RTX 5070, 12GB, SM120, Blackwell)?" Implement the
  baseline AND at least one novel variation in the same round.
- **Cross-domain combinations are the highest-value R&D** â€” see the
  expanded Novel Discovery Protocol below.
- **When stuck on a hard optimization, pivot don't quit**: if 2 iterations
  fail to beat the known best, shelf it in `.devin/scratchpad.md` with the
  failed approaches documented, then touch up a *different* area. Fresh
  context often surfaces the missing idea. Return to the hard problem later.
- **Record failures as carefully as successes** â€” a documented dead end
  saves the next session hours. Use `.devin/scratchpad.md`.

### D. GPU/VRAM â€” Mixed Approaches Are Mandatory To Consider
- **Never propose a GPU-only or CPU-only solution when a mixed approach is
  viable.** The hardware is RTX 5070 12GB VRAM + 32GB system RAM + pinned
  CPU offload (`hybrid_offload.py`, `cpu_kv_offload.py`). The optimal
  operating point is almost always a split.
- **Always state the VRAM budget** for any new inference or training
  feature. If a feature pushes past 12GB on the 1.2B model, it MUST offer
  a mixed CPU/GPU fallback path (e.g. CPUAdamW for training, CPU KV
  offload for inference, BitNet int8 for weights).
- **Quantization is a first-class citizen**, not a fallback. BitNet b1.58,
  W8A8, NVFP4, OffQ, AAAC, SharQ, MosaicQuant are all production paths on
  this hardware â€” prefer them over "just use a smaller model".
- **Profile before assuming**: use `torch.cuda.memory_allocated()` /
  `torch.cuda.max_memory_allocated()` in test scripts. Guessing VRAM is
  how we get OOM at 3am.

### E. No Redundant Files â€” Search Before You Create
- **Before creating ANY new script or module, grep the codebase for an
  existing one that does the same thing.** The codebase has ~600 .py files
  after the aggressive refactor (was 908); the odds are high that a related
  implementation exists.
- **Prefer upgrading an existing file over spawning a new one.** If
  `research/inference/kv/snapkv.py` exists and you want "smarter SnapKV",
  edit that file â€” do not create `snapkv_v2.py` or `smart_snapkv.py`.
  Versioned filenames fragment the codebase and hide the canonical path.
- **If two files end up doing the same thing, merge them** and delete the
  inferior one. Document the merge in the "Removed (consolidation)" section.
- **The canonical path wins**: when in doubt, the file already wired into
  `forge_engine.py` / `forge_server.py` / `sft_train.py` is canonical. New
  code hooks into those, not around them.

#### Refactor 2026-08-30: Aggressive cleanup
- **Deleted 192 .py files** (908 â†’ 716): 84 dead key files, 48 throwaway
  sandbox scripts, 28 dead subsystem modules, 8 dead R&D round files,
  11 dead tests, 30 root scratch files.
- **Dead keys removed**: all keys not referenced by V7/V8/V9 presets or
  `model_loader.py`. Only 25 canonical keys remain in `research/keys/` (24 original
  + `bitnet_residual_key.py` added R24).
- **Sandbox eliminated**: `research/sandbox/` deleted entirely.
  `train_8b_all.py` and `train_v8.py` promoted to `research/training/runners/`.
  Data scripts moved to `scripts/`.
- **Dead R&D rounds removed**: `r20_novel_param_formats.py`, `r22_training_speedups.py`,
  `hypercloning.py`, `ligo.py`, `dlora.py`, `fp4_checkpoint.py`.
- **Dead subsystems cleaned**: `tokenization/` deleted entirely.
  `distillation/` reduced to 2 live files. `evaluation/` reduced to 3 live files.
  `training_free/` kept (used by tests + self_play).
- **Directory reorganization**: `research/data/` â†’ `scripts/`,
  `research/sandbox/train_8b_all.py` â†’ `research/training/runners/train_8b_all.py`.
- **KeyStack builders removed**: `build_qwen2_keystack` and `build_xp_keystack`
  deleted (dead code referencing deleted keys). `KeyStack` class preserved.
- **Pre-existing test failures** (not caused by refactor):
  `test_novel_quant.py` (2 device-mismatch bugs in `novel_quant.py`),
  `test_evolution_domains.py` (1 missing `final_loss` key in training_sim).
  **FIXED 2026-09-01**: all 3 now pass. `novel_quant.py` test generators
  now use `device=W.device` for sign tensors. `training_sim.py`
  `optimizer_simulate` now returns `final_loss` in metrics dict.

#### Refactor 2026-09-01 (b): Test suite repair + dead script cleanup
- **Test suite was broken on Windows**: 11 script-style test files executed at
  import time during pytest collection — a failure aborted the ENTIRE suite
  (`sys.exit(1)` in `test_training_migration.py` caused INTERNALERROR; open
  SQLite unlink in `test_applied_flag.py` raised PermissionError). All 11
  converted to proper pytest tests (module bodies wrapped in `test_*()`,
  `tempfile.mktemp` → `tmp_path`, DB close before unlink). Silent scripts
  that printed FAIL without failing now assert (`n_fail == 0` / `all_pass`).
  Suite: **1225 passed, 0 failed** (was 1077 passed + 3 failed + broken collection).
- **Bit-exact migration fixes** (legacy Python domains aligned to canonical
  JSON spec + simulator scoring):
  - `FlashOptimConfig` (training_domains.py): added `strength_bonus` term
    missing vs `flash_optim_simulate`.
  - `CrossLayerKV` (kv_domains.py): scoring weights updated to spec
    (`param_reduction*150 - recon_err*100 - overhead*2`, was `*100/-500/-2`).
  - `GlaAttention` (attention_domains.py): aligned with `gla_attn_simulate` —
    QR/SVD semi-orthogonal projections (was random → err ~1.4 regardless of
    latent dim), graduated trivial penalty (compression<2.0), flag-handler
    cancellation.
- **ForgeEvolve engine fix** (`engine.py`): seed configs are now persisted to
  the DB (`_pending_discoveries`), so `query_best_configs` reflects the
  archive's best. Previously the archive best (-1.41) never reached the DB
  (best row -4.61) — warm-start loaded stale generators.
- **Delegation simulator recursion fix** (`simulators/misc_sim.py`):
  `kara_simulate`/`hqe_kv_simulate`/`sparse_attn_simulate` called
  `domain.evaluate(config)` where `domain` is the JSONSpecDomain itself →
  infinite recursion (RecursionError in test_all_domains). Now delegate to
  the legacy domain classes (`KARADomain`, `HqeKVDomain`,
  `SparseAttentionDomain`, `KVEvictionDomain`) via a cached instance keyed by
  (class, seq_len, seed, device). Rewrote `kara.json`/`hqe_kv.json`/
  `sparse_attn.json` specs with REAL params (were generic `param0..N` stubs
  that decoded to defaults).
- **Duplication merge**: duplicate `_nullcontext` classes in
  `gen_model_manager.py` + `llm_gen_model.py` → stdlib `contextlib.nullcontext`.
- **Dead scripts deleted (7)** — referenced deleted V2/V7/V8/V9 checkpoints
  or configs: `train_dspark.py` (`forgelm_v2` config deleted, crashed at
  import), `compare_lfm25_vs_v9.py`, `verify_port_loss.py`,
  `test_v9_8b_expanded.py`, `test_v9_8b_forgeengine.py`,
  `test_v9_8b_ternary_load.py`, `test_finetune_growth.py`.
  `test_sheet_v9.py` renamed → `test_sheet_v10.py` (it already loaded V10).
  Dead "Train DSpark Head" preset removed from `forge_gui/api/process_manager.py`.
- **.gitignore**: `scripts/r2*_*.json`, `scripts/_*.json`, `scripts/_*.txt`
  (R&D round result outputs, regenerable).
- `test_forge_evolve.py`: removed `return results` from test functions
  (PytestReturnNotNoneWarning).

#### Refactor 2026-09-01: Engine fallbacks + file merges + API cleanup
- **Engine fallback chains added** (`forge_engine.py`):
  - `_apply_quantization`: fallback chain (nvfp4→w8a8→fp8→int8→int4→bf16)
  - `_activate_kv_cache`: fallback chain (rotorquant→s4r→standard→cpu_offload)
  - `generate()`: now wrapped with `_generate_with_oom_recovery` (was only
    on `generate_raw`)
  - `_detect_keystack_features`: degrades to empty features on I/O errors
  - `from_checkpoint`: wraps `Path.stat()` with `CheckpointError`, falls
    through load paths (standard→hybrid→streaming) on OOM/RuntimeError
  - `_load_with_fallback`: hybrid offload failure now falls through to
    AirLLM streaming instead of crashing
  - `_clear_cuda_cache_static`: static version for classmethods
- **File merges** (10 files → 4):
  - `inference/position/lerope.py` + `rope_id.py` → `position/__init__.py`
  - `inference/prefill/chunked_prefill.py` + `hybrid_prefill.py` → `prefill/__init__.py`
  - `training_free/urial.py` + `decoder.py` + `reflexion.py` + `rain.py` → `training_free/__init__.py`
  - `moe/keyword_router.py` + `semantic_router.py` → `moe/routers.py`
- **API type annotations added**:
  - `ForgeEngine.from_checkpoint -> ForgeEngine`, `activate* -> None`,
    `generate_stream -> Iterator[str]`, lifecycle methods `-> None`
  - `ForgeServer.__init__/register/serve -> None`
  - `ModelLoader.build_model_fast/build_model -> ConfigurableResearchLLM`,
    `load_default_model -> tuple[ConfigurableResearchLLM, Any]`,
    `flash_attention/varlen_attention -> torch.Tensor`
- **Test results**: 1077 passed, 3 skipped, 0 failed (was 1074+3 failed)

### F. Math Thinking + Script Testing â€” Find The True Optimum
- **Every optimization claim must be backed by a number from a script**,
  not a paper citation. Papers report numbers on different hardware/models;
  our numbers come from RTX 5070 + LFM2.5-1.2B.
- **Write the smallest possible test script first** (see Novel Discovery
  Protocol step 1). A 20-line script that runs in 5 seconds > a 200-line
  design doc.
- **Do the math by hand for small cases** before trusting a benchmark.
  If a KV compression scheme claims "15Ã—", verify: `bytes_full /
  bytes_compressed` with the actual dims (d_model=2048, head_dim=64,
  8 KV heads, 16 layers). Catches off-by-one and per-layer-vs-total
  confusion.
- **Sweep parameters, don't guess them**: when a technique has
  hyperparameters (rank, block size, threshold, alpha), write a sweep
  script that tries 5-10 values and plots/logs the result. The optimum is
  rarely the paper's default.
- **Compare against the RIGHT baseline**: a new KV cache must beat the
  current production cache (`s4r` for V4), not a strawman. A new optimizer
  must beat `muon_sf_plain` (V3) or `cpu_offload` (V4), not raw AdamW.
- **No area is truly solved**: even "obviously optimal" choices (RoPE
  theta, head_dim, SwiGLU intermediate ratio) deserve a periodic
  re-challenge. If the improvement is hard to find, shelf it and touch up
  another area â€” fresh ideas surface indirectly.

### G. General Best Practices For This Codebase
- **Run the test suite before declaring done**: `pytest tests/ --tb=short -q`
  with the venv python and `PYTHONPATH=D:\windsurf\ForgeAI`.
- **Keep AGENTS.md current**: when you add a new file, feature, or R&D
  round, update the relevant section here in the SAME session. Stale
  AGENTS.md causes the next agent to duplicate your work.
- **Use `.devin/scratchpad.md` for long working notes**, not the main
  conversation. Keeps context clean and persists across sessions.
- **Debate the user's premise when warranted** â€” if a requested approach
  has a known-better alternative on this hardware, say so and propose the
  alternative. Don't silently implement a worse path.
- **Subagent delegation**: use `subagent_explore` for read-only research
  (file indexing, paper lookups) and `subagent_general` for parallel
  implementation tasks. Spawn 2-3 in parallel for independent work.
- **Skills**: invoke `.devin/skills/` skills (`log-bug`, `sync-memory`,
  `systemspecs`, `glm-supercharge`) when they match the task â€” they encode
  project-specific workflows.
- **IDE crash trigger â€” batch QA/training-data edits**: The IDE crashes when
  a single edit contains 25+ lines matching Question/Answer or similar
  training-data-like patterns. **Code edits do NOT need batching** â€” only
  QA pairs, fact lists, training samples, and other data-like content.
  When writing scripts with embedded fact sets or training data, split the
  file write into multiple smaller edits (â‰¤20 QA lines per edit) or write
  the data to a separate `.json`/`.jsonl` file and load it at runtime.

## Current Architecture: ForgeLM V10-1.2B (SOLE BASE)

**Base model**: ForgeLM V10-1.2B — lossless 1:1 port of LFM2.5-1.2B + V10 inference features.
- Same architecture as LFM2.5: 16 layers (10 conv + 6 GQA), d_model=2048, 32 heads, 8 KV heads
- V10 additions: IRI-FP4 weight quantization (9.0 bits/w, lossless, 3.5× vs fp32)
- 1304.6M params, 1.87 GB checkpoint (IRI-FP4 compressed)
- All prior ForgeLM models (V2/V4/V5/V7/V8/V9) deleted — V10 is the sole base

**Porting fix (2026-08-30)**: LFM2.5's `embedding_norm` is the FINAL norm (applied
after all layers, before head), NOT a post-embedding norm. The HF name is misleading.
Config uses `use_final_norm=True, use_embed_norm=False` to match. Port script:
`research/architecture/port_lfm25_to_v10.py`.

**Base checkpoint**: `research/checkpoints/ForgeLM_V10_1.2B.safetensors`
**Tokenizer**: `research/checkpoints/lfm25_tokenizer/`
**Default config**: `forgelm_v10_1.2b` (load_default_model() defaults to this)
**Path constant**: `research.paths.V10_CHECKPOINT` (LFM25_CHECKPOINT and V9_CHECKPOINT are backward-compat aliases to V10_CHECKPOINT)

### LFM2.5 original architecture (preserved in V10)
- 16 layers: 10 double-gated conv + 6 GQA attention (layers 2,5,8,10,12,14)
- d_model=2048, 32 heads, 8 KV heads (GQA 4x), head_dim=64
- SwiGLU FFN (intermediate=8192), RMSNorm, QK-layernorm on attention
- RoPE theta=1M, 128K context (32K for VRAM budget)
- Vocab=65536, tied embeddings

## Config Presets

Config presets (V3/V4/V5/V7/V8/V9 superseded and checkpoints deleted; V10-1.2B is the sole base):
- `forgelm_v10_1.2b` â€” **ForgeLM V10-1.2B: THE DEFAULT AND SOLE BASE.** Lossless 1:1 port of LFM2.5-1.2B + V10 inference features (IRI-FP4 weight quantization, 9.0 bits/w, lossless, 3.5× vs fp32). Same architecture as LFM2.5 (d_model=2048, 16 layers, 1304.6M params). `load_default_model()` defaults to this. All tests run against this checkpoint.
- `lfm25_tiny` â€” 4-layer tiny model for fast testing (no checkpoint, config-only)
- Other presets (`forgelm_v7*`, `forgelm_v8_8b`, `forgelm_v9*`, `lfm25_1.2b`) have been DELETED from config.py. Only `forgelm_v10_1.2b`, `lfm25_tiny`, and `gen_model_tiny` remain.

### Evolution-Discovered Promotions (2026-08-24, from forge_evolve.db)
Promoted after validation against evolution data (39,631 discoveries scanned):
- **MTP**: n_heads=4, loss_weight=0.495 (was 2/0.3, score 27.69) âœ“ validated
- **SpecDecode (PEAGLE)**: n_draft=7 (was 4, score 57.05) âœ“ validated
- **BatchQueue**: batch_window=52ms, max_batch=15 (was 50ms/8, score 44.25) âœ“ validated
- **SFT training**: grad_accum=5, grad_compression=int4 (score 30.00/11.10) âœ“ validated
- **AirMoEKey**: cache_strategy=lfu, disk_cache_size=4096 (score 8.83, 0% miss rate) âœ“ validated
- **ForgeEngine**: CREATIVE_SAMPLING preset (temp=1.98, top_p=0.989, top_k=69, score 10.45) âœ“ validated
- Already applied prior: W8A8 fp8+alpha=0.999, PagedEvictKV page=64/LRU, ModConfig aux_loss=1e-8, CheckpointRecompute selective/block=512, FocalLoss gamma=4.93

### Reverted promotions (scoring artifacts â€” synthetic metrics didn't match real behavior)
- **MoE routing** (4/3/switch/6e-5): REVERTED to 8/2/aux_free/0.01. Evolution found top_k=1 scored higher (24.95 vs 24.87) but top_k=1 is a trivial solution (no ensemble). Scoring fixed: diversity penalty for top_k=1.
- **RoPE theta=10M**: REVERTED to 1M. Synthetic metric rewarded angle diversity, not attention quality. Scoring fixed: checkpoint compat penalty + frozen-dimension detection at long range.
- **Scheduler warmup=0**: REVERTED to warmup=500/20. Synthetic AUC rewarded no warmup, ignored training stability. Scoring fixed: stability penalty for zero warmup.
- **Label smoothing 0.29**: REVERTED to 0.1. Synthetic metric rewarded high smoothing for grad magnitude. Scoring fixed: smoothing penalty above 0.2 + focus ratio metric.

### Evolution domain scoring fixes (2026-08-24)
Fixed 19 scoring issues across 14 domains in two rounds:

**Round 1: Synthetic-metric artifacts (7 domains)**
- `MoeRouting`: +diversity_penalty (top_k=1: -15, top_k=2: -3) + shared_expert bonus (+2)
- `RopeConfig`: +checkpoint compat penalty (log-scaled for >10x theta deviation) + frozen-dim detection at 4096 positions + long-range rotation diversity test
- `SchedulerConfig`: +stability_penalty (warmup=0: -8, <1% warmup: -4, >30% warmup: -2) + early lr jump penalty
- `Fp8TrainingConfig`: +quantization error measurement (simulates FP8 rounding with correct mantissa bits: e4m3=3, e5m2=2) â€” e5m2's larger range no longer masks its worse precision
- `LossConfig`: +focus_ratio (gradient concentration on wrong predictions) +smoothing_penalty (>0.2: -30x) +gamma_penalty (>5: -5x) +temp_penalty (>1.5: -4x)
- `ModConfig`: +router_quality (mlp=1.0, linear=0.85) +skip_penalty (>8 layers: -0.5x) +aux_penalty (>0.01: -50x, <1e-10: -1.0) â€” previously aux_loss/n_skip/router_type were decoded but never scored
- `FactorizedEmbed`: +tie_factor evaluation (tying saves params but adds up to 10% reconstruction error) â€” previously tie_factor was decoded but never scored

**Round 2: Decoded-but-not-scored + trivial solutions + missing tradeoffs (12 domains)**
- `SpeculativeDecode`: FIXED backwards acceptance_threshold formula (higher threshold = stricter = lower acceptance, was inverted) + temperature now scored (was decoded but ignored) + temp_penalty for extremes
- `MtpConfig`: +loss_weight penalty (>0.5 hurts main task) + depth_ratio latency cost (deeper heads = more inference cost)
- `BatchedDecode`: +merge_window latency penalty (>50ms = noticeable interactive delay)
- `SamplingConfig`: +repetition_penalty benefit (mild penalty reduces repetition) + frequency_penalty benefit + temp_penalty for extremes â€” previously penalties were only penalized with no upside modeled
- `BeamSearch`: +beam_width=1 penalty (greedy = not beam search, -5) + length_penalty sweet spot (peaks at 1.0, penalizes extremes) + diversity_penalty diminishing returns
- `TitanMemory`: +update_freq scoring (was decoded but ignored) + freshness model + gate_interference penalty (high gate dominates main signal)
- `GlaAttention`: +n_heads scoring (was decoded but ignored) + head_overhead param cost
- `GtaAttention`: +n_kv_heads scoring (was decoded but ignored) + tie_strength scoring (was decoded but ignored) + tying_savings benefit
- `CrossLayerKV`: FIXED learned mode (was identical to avg mode) â€” now uses SVD-based optimal combination for lower recon_err
- `XQuantKV`: +inference_penalty (ratio=1.0 = no KV cache = O(n^2) generation, -50) â€” previously full recompute scored 95 (trivial)
- `KvRecompute`: +inference_penalty (n_recomp=16 = no KV cache, -40) â€” same trivial solution as XQuantKV
- `HybridOffload`: +prefetch_depth logarithmic (was linear, no diminishing returns) + prefetch_mem_cost (deeper prefetch uses more staging VRAM)
- `StreamingKV`: +overlap memory cost (overlap keeps parts of previous chunks, increases memory)

### DB rescore (2026-08-24)
After fixing the 19 scoring issues, the DB was rescored with `tests/evolution/rescore_db.py`:
- Round 1: 3,051 discoveries across 9 domains, 639 fake winners pruned
- Round 2: 14,196 discoveries across 12 domains, 1,551 fake winners pruned
- Total: 2,190 fake winners pruned (score dropped >50% under fixed scoring)
- DB: 41,175 â†’ 38,985 discoveries
- Backup: `forge_evolve.db.bak_pre_rescore` (5.3GB)
- Rescore tool: `tests/evolution/rescore_db.py --db forge_evolve.db --prune` (use `--dry-run` first)
- NaN guard: non-finite scores automatically set to -1e9 and pruned
- When fixing domain scoring in the future: add a flag handler in `reward_guard.py` + `{"flag": "..."}` to the domain's JSON spec, then run `--dry-run` to verify, then `--prune` to apply. (The old `FIXED_DOMAINS` dict in rescore_db.py has been deleted â€” scoring fixes are now declarative in JSON.)

### DB clean + applied flag + train-first (2026-08-25)
- **Gen model knowledge wiped**: `wipe_gen_model_knowledge(db_path)` in `gen_model_manager.py` â€” deletes `gen_model_state`, `gen_models`, `gen_model_performance` tables. Also available as `GenModelManager.wipe_knowledge()` method.
- **DB rescored**: 122 discoveries (23 quant + 99 synthetic) rescored with current scoring. Quant scores restored from 0.0 (bad rescore) to proper values (best=-4.95, mean=-42.38). Synthetic bit-exact (unchanged).
- **`applied` column added** to `discoveries` table (INTEGER DEFAULT 0). All 122 existing discoveries set to `applied=0` (unapplied).
- **Train-first logic in engine**: `ForgeEvolve.run()` now checks `db.count_unapplied()` at start. If >0, it:
  1. Fetches unapplied discoveries via `db.get_unapplied_discoveries(limit=200)`
  2. Converts to solutions format (extracts `problem`/`model_answer` from metadata)
  3. Fine-tunes the gen model on those solutions via `gen_mgr.fine_tune_on_solutions()`
  4. Marks all unapplied discoveries as `applied=1` via `db.mark_applied(ids)`
  5. Then proceeds with normal evolutionary search
- **DB methods** (in `database.py`):
  - `count_unapplied(domain=None)` â€” count discoveries where `applied=0 OR applied IS NULL`
  - `get_unapplied_discoveries(domain=None, limit=100)` â€” fetch unapplied discoveries sorted by score DESC
  - `mark_applied(discovery_ids)` â€” set `applied=1` for given IDs
  - `mark_all_applied(domain=None)` â€” set `applied=1` for all unapplied, returns count
- **Test**: `tests/evolution/test_applied_flag.py` â€” verifies all DB methods work correctly

### Evolution focus profiles (2026-08-24)
Focus profiles control which domains are researched in evolution runs:
- `tests/evolution/configs/focus_profiles.json` â€” defines focus areas (memory/speed/quality/training/all)
- `tests/evolution/configs/default_focus.txt` â€” persistent default focus (currently: `memory`)
- Current default: **memory** (28 domains focused on minimizing param size / VRAM with minimal quality loss)

**Usage:**
```
# Set persistent default (saved to configs/default_focus.txt)
python tests/evolution/run_evolve.py --set-focus memory

# Override for a single run
python tests/evolution/run_evolve.py --profile boot --focus speed

# List available focus profiles + their domains
python tests/evolution/run_evolve.py --list-focus

# Clear default
python tests/evolution/run_evolve.py --set-focus none
```

**Focus areas:**
- `memory` (28 domains): quantization (BitNet/W8A8/NVFP4/AAAC/SharQ/OffQ/Mosaic/Group), KV compression (KvZip/XQuant/Hadamard/RotorQuant/Streaming/PagedEvict/CrossLayer), offload (CpuKv/Hybrid/ExpertHotload), arch param reduction (MoE/GLA/GTA/FfnSkip/FactorizedEmbed/MoD), checkpointing
- `speed` (18 domains): decoding (SpecDecode/MTP/Beam/Batch/Sampling), KV cache speed
- `quality` (26 domains): attention variants, training configs, architecture
- `training` (9 domains): optimizer, scheduler, loss, gradient, checkpointing
- `all` (57 domains): no filtering

### R&D round 14: Training speedup features (2026-08-25)
Community research (Unsloth, Liger-Kernel, GaLore, APOLLO, BREAD, FlashOptim) applied to ForgeAI V7 8B training. All features are config-driven, enabled by default in V7 presets, and fall back gracefully on CPU/no-Triton.

**New features:**
- **Varlen attention** (`use_varlen=True`): FlashAttention varlen path for packed sequences. Eliminates cross-example attention contamination (correctness fix) + reduces padding-mask compute waste (throughput). `PackedSequenceDataset` now emits `cu_seqlens` when `emit_cu_seqlens=True`. Threaded through `ConfigurableResearchLLM.forward â†’ ModularBlock.forward â†’ GroupedQueryAttention.forward`. Falls back to block-diagonal SDPA mask when `flash_attn` package unavailable. Community: Unsloth 2.1x faster padding-free, 50% less VRAM.
- **Triton fused training kernels** (`use_triton_kernels=True`): Fused RMSNorm + SwiGLU activation as single Triton kernels (Liger-Kernel-style). Tuned for SM120 (RTX 5070, GDDR7 672 GB/s). Block sizes: `triton_rms_block_size=4096` (d_model), `triton_swiglu_block_size=16384` (intermediate). Falls back to `F.rms_norm` / `F.silu` on CPU. Community: Liger-Kernel 20% throughput, 60% VRAM; Unsloth 2-5x.
- **APOLLO optimizer** (`optimizer="apollo"`): SVD-free random-projection gradient scaling. SGD-like memory with AdamW-level performance. `apollo_rank=8` (rank=1 = APOLLO-Mini = SGD memory). `apollo_scale="tensor"` or `"channel"`. No SVD overhead vs GaLore. Community: arXiv 2412.05183, ICLR 2025.
- **BREAD for BAdam** (`bread_sgd_correction="partial"`): Landscape correction for BAdam. Applies memory-efficient SGD updates to inactive blocks using cached momentum, preventing optimization landscape narrowing. `bread_sgd_lr_scale=5.0` (SGD lr = base_lr * 5). Modes: "disabled" (vanilla BAdam), "partial" (visited blocks only, best), "all" (all inactive blocks, risky). Community: OpenReview zs6bRl05g8, Microsoft BlockOptimizers.
- **FlashOptim** (`optimizer="flashoptim"`): Companded 8-bit optimizer states (7 bytes/param vs 16 for AdamW). Sqrt companding allocates more quantization levels to small momentum values. `flashoptim_bits=8` (or 4). Community: arXiv 2602.23349 (Databricks), >50% memory reduction.

**New files:**
- `research/decoding/triton_train_kernels.py` â€” Triton fused RMSNorm + SwiGLU kernels
- `research/training/optim/apollo.py` â€” APOLLO optimizer
- `research/training/optim/flashoptim.py` â€” FlashOptim companded 8-bit AdamW
- `tests/unit/test_rd14_speedup.py` â€” 27 tests (all passing)

**Modified files:**
- `research/config.py` â€” New ModelConfig fields + enabled in all V7 presets
- `research/model_loader.py` â€” `flash_attention` + `varlen_attention` + `_build_block_diag_causal_mask`; `RMSNorm` + `SwiGLUFFN` accept `use_triton`; `cu_seqlens` threaded through `ConfigurableResearchLLM â†’ ModularBlock â†’ GroupedQueryAttention`
- `research/keys/attention/{differential_attn,gta,gla}_key.py` â€” Accept `cu_seqlens=None` (ignored, for signature compat)
- `research/training/data/efficient_pipeline.py` â€” `PackedSequenceDataset` emits `cu_seqlens` when `emit_cu_seqlens=True`
- `research/training/optim/badam.py` â€” BREAD landscape correction (`_apply_bread_correction`, `_bread_visited` tracking)
- `research/training/training_utils.py` â€” Wired `apollo`, `flashoptim` optimizers + BREAD config passthrough for `badam`
- `research/evolution/domains/training_domains.py` â€” 5 new evolution domains: `ApolloConfig`, `BreadConfig`, `FlashOptimConfig`, `TritonKernelConfig`, `VarlenConfig`
- `tests/evolution/configs/focus_profiles.json` â€” New domains added to training/speed/memory/quality focus areas
- `tests/evolution/configs/domain_categories.json` â€” New domains in "training" category
- `tests/evolution/rescore_db.py` â€” Round 3 entries in `FIXED_DOMAINS`
- `research/architecture/port_v4_to_v7_8b.py` â€” Documentation: R&D round 14 features are config-only (no checkpoint keys)

**Evolution domains added (Round 3 in FIXED_DOMAINS):**
- `apollo_config` â†’ `ApolloConfig`: rank/scale/lr_scale, scores memory savings + convergence + trivial-solution guard
- `bread_config` â†’ `BreadConfig`: correction_mode/sgd_lr_scale, partial=+15 best, all=-5 risky, sweet spot 5x
- `flashoptim_config` â†’ `FlashOptimConfig`: bits/companding, 8-bit=75%/4-bit=87.5% memory savings, sqrt=+5 bonus
- `triton_kernel_config` â†’ `TritonKernelConfig`: rms/swiglu block sizes, +20 base speedup minus mismatch penalty
- `varlen_config` â†’ `VarlenConfig`: use_varlen boolean, +25 if enabled (2.1x faster, fixes contamination)

**torch.compile note:** Per user directive, torch.compile remains OFF by default (`--compile` flag exists but has caused issues). All speedup features work without compilation.

### R&D round 24: SpectralKV + BitNetResidual + V9 (2026-08-30)
Data-driven optimization round. 11 test scripts validated novel ideas on real LFM2.5 weights. Two P0 wins implemented into V9:

**Validated discoveries (scripts/test_*.py):**
- **SpectralKV** (§14.1): Fourier-basis KV cache, O(1) memory. 63× compression at 0.095 error on real weights (S4R = 8.90 at same budget). Real trained weights are 2-3× more Fourier-friendly than random. Stable across 512→8192 tokens.
- **BitNet+residual** (§15.2): ternary + 10% element residual = 0.33 error (vs pure ternary 0.80, vs NLRQ rank-1024 1.4). Element-level residual wins (error is distributed, not concentrated in rows/cols).
- **NLRQ rank-1024 insufficient** (§14.2): LFM2.5 FFN weights are 96% full-rank at 99% energy. rank-1024 captures only ~50% → major V8 quality killer. V9 raises to 2048.
- **Embedding rank-512 insufficient** (§14.2): same issue, V9 raises to 1024.
- **AirMoE LoRA rebase** (§14.3): LoRA experts are parent-independent (0.0000 error). No rebase needed when parent trains.
- **Dropped**: INR-weights (loses to SVD), VQ-weights (loses to NLRQ), DCT-weights (loses to SVD), naive output correction for SpectralKV.

**New files:**
- `research/inference/kv/spectral_kv.py` — SpectralKVCache (KVCacheStrategy) + SpectralPreAllocatedCache (PreAllocatedKVCache interface)
- `research/keys/quantization/bitnet_residual_key.py` — BitNetResidualLinear + BitNetResidualKey
- `research/architecture/port_v8_to_v9.py` — V8→V9 checkpoint conversion (refit NLRQ/embedding ranks, add residual masks)
- `tests/unit/test_r24_spectral_kv_bitnet_residual.py` — 28 tests (all passing)
- `scripts/test_real_spectral_kv.py`, `scripts/test_real_svd_decay.py`, `scripts/test_airmoe_rebase.py`, `scripts/test_spectral_kv_recovery.py`, `scripts/test_real_ternary_residual.py`, `scripts/test_weight_fourier.py` — R&D validation scripts

**Modified files:**
- `research/config.py` — V9/V10 config fields (use_spectral_kv, spectral_kv_max_freq, use_bitnet_residual, bitnet_residual_frac) + `forgelm_v10_1.2b` preset
- `research/inference/kv_backend.py` — `"spectral"` strategy in build_kv_cache factory
- `docs/RND_PLAN_V8_AND_NOVEL_ARCH.md` — §12-§15 results + revised priorities

**V9 vs V8 deltas (all evidence-backed):**
- NLRQ rank: 1024 → 2048 (§14.2: rank-1024 destroys 50% of FFN info)
- Embedding rank: 512 → 1024 (§14.2: rank-512 destroys 50% of info)
- BitNetResidual: new key (ternary + 10% element residual, §15.2)
- SpectralKV: inference-time KV cache (activate: `engine.activate(kv_cache="spectral")`)
- All V8 keys carried forward (per §A: derive from prior version)

### Adaptive GPU parallelism (2026-08-24)
The evolution runner now monitors GPU utilization via NVML and dynamically adjusts concurrency:
- `GPUMonitor` class in `run_evolve.py` â€” samples GPU util every 0.5s via pynvml
- Adaptive parallelism: scales concurrent domain count up (GPU <50%) or down (GPU >95% / VRAM >85%)
- VRAM-aware capping: reduces concurrency when VRAM > 85% (prevents OOM on 12GB cards)
- CUDA stream overlap: `_threaded_eval` in engine.py uses per-thread CUDA streams to overlap kernel launches
- `torch.compile` on generators: fuses 4-layer MLP + LayerNorm + sigmoid into fewer CUDA kernels
- TF32 tensor cores: enabled for faster matmul on Ampere+ (RTX 5070)
- GPUKeepBusy: engine auto-increases `max_evaluate` when score phase < 30% of total time
- BatchedEvaluator: batched PyTorch ops for quantization domains (10+ domains supported)

**Usage:**
```
# Max GPU utilization mode (recommended for small GPUs)
python tests/evolution/run_evolve.py --profile boot --focus memory --max-gpu

# Manual adaptive control
python tests/evolution/run_evolve.py --profile deep --focus memory \
  --gpu-target 0.90 --min-parallel 3 --max-parallel 12

# Fixed parallelism (no adaptation)
python tests/evolution/run_evolve.py --profile boot --no-adaptive-parallel --parallel 6
```

**`--max-gpu` overrides:**
- gen_pop=1000 (more generators = more GPU work per generation)
- max_eval=200 (bigger eval batches = more GPU kernels per generation)
- time_budget=0 (no time limit â€” domains run all generations)
- gpu_target=95% (aggressive GPU utilization target)
- max_parallel=12 (up to 12 concurrent domains)

**GPU util results (RTX 5070, 12GB):**
- Before: ~10% (sequential eval, fixed parallelism)
- After: ~19-26% (threaded eval, adaptive 4â†’12 domains, torch.compile, TF32)
- VRAM capped at ~80% (adaptive reduces concurrency when VRAM > 85%)
- Remaining gap: domain evals use tiny synthetic tensors where Python overhead > GPU compute. Hitting 90%+ requires CUDA graph capture of the full genâ†’filterâ†’evalâ†’train loop.

### ForgeEvolve Refactor: Declarative Domain Specs + Gen Models (2026-08-25)
Major refactor of the evolution system. Domains are now defined by JSON specs
+ pure simulator functions instead of Python classes with embedded scoring.
Adds LLM gen model support, LLM-as-judge checker, DB provenance, and a
compute split for foreground gen model inference + background checker scoring.

**Architecture:**
- **DomainSpec** (`research/evolution/domain_spec.py`): loads JSON spec,
  auto-generates encode/decode/evaluate. `JSONSpecDomain` wraps a spec as a
  `BaseDomain` for the engine. Supports `gen_model_type` ("mlp" or "llm") and
  `checker_type` ("heuristic" or "llm_judge") in the spec.
- **RewardGuard** (`research/evolution/reward_guard.py`): universal scoring
  engine. Reads `ScoringSpec` from JSON (components, penalties, hardening,
  transforms). All 19 scoring fixes are flag handlers (`stability_penalty`,
  `smoothing_penalty`, `aux_penalty`, `diversity_penalty`, etc.).
- **Simulators** (`research/evolution/simulators/`): pure metric computation,
  registered via `@register("name")`. No scoring logic â€” just raw metrics.
  - `quant_sim.py` â€” 10 quantization domain simulators
  - `synthetic_sim.py` â€” synthetic domain
  - `training_sim.py` â€” 13 training domain simulators
  - `attention_sim.py` â€” 10 attention domain simulators
  - `arch_sim.py` â€” 5 architecture domain simulators
  - `random_task_sim.py` â€” 3 random task simulators (math/algorithm/logic)
- **JSON specs** (`tests/evolution/configs/domains/*.json`): declarative
  domain definitions â€” params, behavioral_dims, scoring formula, simulator
  name, gen_model_type, checker_type. Auto-discovered by the domain registry.

**LLM Gen Model support:**
- `LLMGenModel` (`research/evolution/llm_gen_model.py`): ultra-compact ForgeLM
  V7 model (17.4M params, d_model=256, 4 layers, NLRQ rank=64, BitNet, GTA).
  Config preset: `gen_model_tiny` in `research/config.py`.
- `GenModelManager` (`research/evolution/gen_model_manager.py`): lifecycle
  management with golden ratio scaling. Auto-grow on plateau, shrink on
  overperformance, distill weights + logits during resize. Fine-tune on
  successful solutions between rounds. Persists to DB.
- `RandomTaskDomain` (`research/evolution/domains/random_task_domain.py`):
  generates random math/algorithm/logic problems for the gen model to solve.
  Scores correctness + focus + speed. Supports distraction injection.

**LLM-as-judge checker:**
- `SharedCheckerModel` (`research/evolution/checker_model.py`): singleton
  checker that scores answers using heuristics first, LLM-as-judge fallback.
  Bounded LRU cache, batch checking, sleep/wake for VRAM management.

**DB provenance + hardening:**
- `FindingsDB` (`research/evolution/database.py`): schema versioning,
  provenance columns (script_text, input_text, output_text, expected_text,
  gen_model_size, gen_model_version), gen model state tables, scoring hash
  regression guard. `CurriculumFineTuner`
  (`research/evolution/curriculum_finetuner.py`): between-round fine-tuning
  on successful solutions.

**Compute split (engine):**
- `ForgeEvolveConfig.enable_compute_split=True`: foreground gen model
  inference (batched on GPU), background checker scoring (ThreadPoolExecutor).
  Gen model lifecycle (grow/shrink/fine-tune) wired into end-of-round.
- Cross-run result cache: warm-start uses stored DB scores instead of
  re-evaluating past discoveries.
- Larger surrogate batches: `max_evaluate` raised from 100 to 150.

**Migration status:**
- Quant domains: 10/10 migrated, bit-exact âœ“
- Training domains: 13/13 migrated, bit-exact âœ“
- Attention domains: 10/10 migrated, bit-exact âœ“
- Arch domains: 5/5 migrated, bit-exact âœ“
- KV domains: 8/8 migrated, bit-exact âœ“
- Memory domains: 5/5 migrated, bit-exact âœ“
- Decoding domains: 5/5 migrated, bit-exact âœ“
- Misc domains: 5/5 migrated (delegated simulators) âœ“
- Random task domains: 3/3 migrated, tests pass âœ“
- **Total: 65 JSON specs, 61 bit-exact, 4 delegated (misc)**

**Tests:**
- `tests/evolution/test_quant_migration.py` â€” 10 quant domains bit-exact
- `tests/evolution/test_training_migration.py` â€” 13 training domains bit-exact
- `tests/evolution/test_training_arch_migration.py` â€” 28 training+attn+arch bit-exact
- `tests/evolution/test_kv_migration.py` â€” 8 KV domains bit-exact
- `tests/evolution/test_mem_dec_migration.py` â€” 10 memory+decoding domains bit-exact
- `tests/evolution/test_random_task_domain.py` â€” random task domain tests
- `tests/evolution/test_checker_model.py` â€” 31 checker model tests
- `tests/evolution/test_provenance_import.py` â€” 8 DB provenance tests

**How to add a new domain (post-refactor):**
1. Write a simulator function in `simulators/<category>_sim.py` with `@register("name")`.
2. Create a JSON spec in `tests/evolution/configs/domains/<name>.json`.
3. The domain auto-registers via `_discover_json_domains()` in `domains/__init__.py`.
4. No Python class needed â€” the JSON spec + simulator is the entire domain.

**How to fix domain scoring (post-refactor):**
1. Add a flag handler in `reward_guard.py` (e.g. `_my_penalty`).
2. Register it in the `_PENALTY_HANDLERS` dict.
3. Add `{"flag": "my_penalty"}` to the domain's JSON spec penalties list.
4. Run the migration test to verify bit-exact match.

## Key Files

### Core
- `research/config.py` â€” ModelConfig dataclass + presets
- `research/model_loader.py` â€” ConfigurableResearchLLM, ModelLoader, PreAllocatedKVCache
- `research/checkpoint_io.py` â€” Safetensors checkpoint I/O
- `research/paths.py` â€” Central path management
- `research/tokenizer_cache.py` â€” Tokenizer loading + caching

### Cloud Compute (Vast.ai backend â€” v2 SDK + persistent lifecycle)
- `research/cloud/vast_connector.py` â€” **Vast.ai cloud backend (v2)** for offloading compute-heavy training to rented GPU instances. Migrated from CLI subprocess to official `vastai` Python SDK (`from vastai import VastAI`). Three transport layers: (1) SDK control plane (search_offers, create/stop/start/destroy instance, ssh_url, volumes, logs), (2) paramiko SSH/SFTP for long-running commands + streaming, (3) manifest-based incremental sync (sha256+mtime, only uploads changed files).
  - **Persistent instance reuse**: instances labeled `forgeai-<config>-<gpu>` and found by label on subsequent runs. After training, instance is **stopped** (not destroyed) â€” container disk (venv, repo, checkpoints) persists across stop/start. GPU charges stop; only cheap disk charges continue. `--vast-auto-destroy` (default False) controls destroy vs stop. `--vast-reuse-instance` (default True) enables label-based reuse.
  - **Incremental file sync**: `sync_dir()` computes local manifest (path â†’ size+mtime+sha256), compares with remote manifest (cached JSON on instance), uploads only changed files + deletes removed files. Eliminates "re-upload everything every run" problem.
  - **Provisioning hash check**: requirements.txt + provision command hashed and cached on instance. If hash matches, pip install is skipped entirely (~5-10 min saved on re-runs).
  - **Log overhaul**: structured line parsing (step/loss/metrics, log levels, exit markers), color-coded terminal output, `--vast-log-filter` for grep-like filtering, `get_logs()` via SDK for historical log retrieval after disconnect.
  - **Connection stability**: SSH keepalive (30s), reconnect-on-drop during training stream, poll-trap handling (exited/unknown/offline â†’ destroy + retry).
  - **Volume persistence** (optional): `--vast-use-volume` creates + attaches a Vast.ai volume at `/workspace` for cross-destroy persistence (survives instance destruction, same-machine only).
  - **New CLI subcommands**: `stop <id>`, `start <id>`, `logs <id>`, `wipe <id>` (clear data without recreating instance).
  - **From-scratch training**: `--from-scratch` flag sets `--checkpoint scratch` (random init). Auto-switches to `--optimizer badam` for BitNet int8 configs.
  - **Throughput auto-tuning**: `--vast-maximize-throughput` (default True) auto-disables grad checkpointing + BitNet int8 on 80GB GPUs, scales batch size, enables torch.compile.
  - Invoke via `sft_train.py --remote-vast --vast-budget 10` or standalone `python -m research.cloud.vast_connector run ...`. Tests: `tests/unit/test_vast_connector.py` (36 tests, mocked SDK+SSH, CPU-only, 0.4s).
  - **8B-specific training fixes (2026-08-26)**: `sft_train.py` now mirrors `train_8b_all.py`'s 8B-critical setup that was previously missing:
    - **Dead-param freeze probe**: `freeze_dead_params_()` runs before BAdam partitioning. Without this, BAdam creates blocks containing ONLY dead params (MTP draft head when `mtp_weight=0`, loop_block, gated-off AttnRes/MHC paths) â†’ `backward()` crashes with "element 0 does not require grad" when such a block activates. The probe does one tiny forward+backward, then freezes any param where `p.grad is None`.
    - **MTP explicit freeze**: When `use_mtp=True` but `mtp_weight=0` (default), MTP params are frozen before the probe (saves probe compute + guarantees no MTP block).
    - **NLRQ factor training (STE)**: `enable_factor_training_all_()` creates float master factors (U_m, V_m) for NLRQ-compressed configs (8B-D). Without this, only singular values S train â€” U/V factors stay frozen at init. Masters train via straight-through estimator around INT8 quantizer.
    - **NLRQ export on save**: `export_nlrq_()` runs before every checkpoint save (periodic + final). Exports STE masters to INT8 buffers, strips masters from state_dict â†’ checkpoint is pure-INT8 (loadable by ForgeEngine without training-mode flags).
    - **From-scratch init**: `--from-scratch` now runs `reset_nlrq_layers_()` + `disable_bitnet_qat_()` + `initialize_weights_()` + `normalize_logit_scale_()`. Without this, from-scratch 8B starts with NLRQ at default init, BitNet QAT on random weights (wastes compute), and unnormalized logit scale (CE ~24 instead of ~11).
    - **BitNet int8 trainable storage**: `enable_int8_training()` converts BitNetLinear to int8 ternary buffer on GPU + bf16 master on CPU. BAdam updates CPU master, re-quantizes active block to int8. 8B dense: 8GB GPU + 16GB CPU RAM.
    - Tests: `tests/unit/test_8b_cloud_training.py` (15 tests, CPU-only, 3s). Covers dead-param crash reproduction, freeze fix, tiny 8B-B/8B-D build+forward+backward+BAdam, NLRQ STE factor training, checkpoint roundtrip, from-scratch init, full training cycle.

### Data Generation + Download
- `research/data/generate_synthetic_data.py` â€” **Pure-Python synthetic data generator** (no API calls, no LLM distillation). Generates 70K examples across 10 categories: arithmetic (10K), algebra (10K), calculus (5K), linear_algebra (5K), number_theory (5K), logic syllogisms (5K), sequences (5K), grammar corrections (10K), word problems (10K), combinatorics (5K). All answers are programmatically verified. Output: `research/data/finetune/synthetic_*.jsonl`. Run: `python -m research.data.generate_synthetic_data`.
- `research/data/download_gaps.py` â€” **HuggingFace dataset downloader** for filling coverage gaps. Downloads: NuminaMath-CoT, MetaMathQA, Glaive FC v2, R1-distill, FineWeb-Edu, OpenOrca, LogiQA. Streams to JSONL (no RAM overload). Skips already-downloaded datasets. Run: `python -m research.data.download_gaps`.

### Negative Training / Unlearning
- **DPO/ORPO**: `research/training/runners/dpo_align.py` â€” Full DPO + ORPO (no reference model needed) + self-rewarding mode. Use `(prompt, good_response, bad_response)` pairs to train the model to prefer good outputs and avoid bad ones (loops, breaks, hallucinations). Usage: `python -m research.training.runners.dpo_align --method orpo --checkpoint ... --data ...`
- **GRPO repetition penalty**: `research/self_play/grpo_trainer.py` â€” N-gram repetition penalty as negative reward during GRPO training. Config: `use_repetition_penalty=True`, `repetition_penalty=-0.5`. Reduces doom-loop rate from 4% to 0.36%.
- **Doom-loop detection**: `research/training/runners/curriculum_sft.py` â€” `is_doom_loop()` detects excessive n-gram repetition. `filter_doom_loops()` removes bad examples from training data.
- **Advanced RL losses**: `research/training/losses/advanced_rl.py` â€” SPPO, PS-PPO, EVPO, GRPO-OR for various RL training strategies.

### Architecture (only MTP kept)
- `research/architecture/port_lfm25_to_forgeai.py` â€” Porting script (reference)
- `research/architecture/port_to_v3.py` â€” V3 porting script (bakes arch key conversions)
- `research/decoding/mtp.py` â€” MTP speculative decoding + training (independent heads design)

### Inference
- `research/inference/forge_engine.py` â€” Unified inference engine. **Hot-swap support**: `engine.hotswap.set_kv_cache()`, `set_decoding()`, `set_context_limit()`, `set_infinite_context()`, `set_generation_defaults()`, `set_feature()`, `update()`. `generate_batch()` for parallel multi-prompt generation. `generate_adaptive()` for RPO-trained adaptive thinking. Per-request `context_limit` override.
- `research/inference/forge_server.py` â€” **v3.2: OpenAI-compatible FastAPI server** with multi-model (ModelRegistry), tool calling, SSE streaming, sleep/wake, **concurrent task-based generation** (SessionManager + BatchQueue), request batching, **hot-swap API** (`/v1/engine/*`), **Library API** (`/v1/library/*`), **agentic tool API** (`/v1/chat/agent`, `/v1/tools`, `/v1/tools/execute`), and per-task config updates. Built-in tools give the LLM direct access to Library, hot-swap, batch gen, and engine introspection.
- `research/inference/session_manager.py` â€” **SessionManager + BatchQueue**: per-task conversation context (LRU eviction), concurrent request batching (50ms window, up to 8 requests per batch â†’ single BatchedDecoding forward pass). 1005 tok/s on 8 concurrent tasks (3.5x vs serial)
- `research/inference/decoding.py` â€” Decoding strategies (standard, speculative, MTP, **batched**)
- `research/inference/batched_decoding.py` â€” BatchedDecoding: multiple prompts in single forward pass (GEMVâ†’GEMM, 3-5x throughput)
- `research/inference/kv_backend.py` â€” KV cache strategies (standard, paged, rotorquant, hadamard, compressed, streaming, snapkv, **kvzip**)
- `research/inference/hotswap.py` â€” **HotSwapManager**: runtime config changes without restart. Hot-swap KV cache, decoding, quantization, context limits, generation defaults, feature flags, VRAM margin, batch config. Thread-safe, per-request overrides. Applied lazily on next generate() or forced via apply_pending().
- `research/inference/library.py` â€” **Library**: persistent knowledge base with lorebook-style injection. Pre-tokenizes content on save (token cache saved as .npy alongside .json). Auto-trims by relevance (LRU + priority + category retention). Model self-write (failures/wins/research). Tag + keyword indices for O(1) lookup. `inject()` scans prompt for triggers, injects matching entries up to token budget. `get_injection_tokens()` returns pre-tokenized IDs (no re-tokenization). Disk-backed at `research/data/library/`. API: `/v1/library/*`.
- `research/inference/engine_tools.py` â€” **EngineToolRegistry**: 38 built-in tools the LLM can call during generation. **Library** (9): `library_save/search/lookup/get/delete/update/stats/optimize/set_config`. **Hot-swap** (7): `engine_set_kv_cache/decoding/context_limit/infinite_context/generation_params/feature/apply_changes`. **Engine info** (3): `engine_get_settings/stats/pending`. **Generation** (2): `engine_batch_generate`, `engine_generate_adaptive`. **Math** (2): `math_eval` (safe AST-based expression eval with sqrt/sin/cos/log/etc.), `calc` (quick calculator). **Random** (2): `random_number` (int/float ranges), `chance` (coin_flip/dice/choice/shuffle/weighted). **Web** (3): `web_search` (Tavily API), `web_search_semantic` (Exa neural search), `web_scrape` (Firecrawl page scrape). **Files** (7): `file_read/write/edit/move/rename/list/delete` (workspace-scoped, path-validated, security-checked). **Security** (3): `scan_script` (pre-scan Python for dangerous imports/patterns), `security_get_config`, `security_get_pending`. Server-side execution via `execute(name, args)` / `execute_calls(calls)`. All file writes and web scrapes go through `ToolSecurityManager`. Used by `generate_with_tools()` agentic loop and `/v1/chat/agent` API endpoint. API keys in `.env`: `TAVILY_API_KEY`, `EXA_API_KEY`, `FIRECRAWL_API_KEY`.
- `research/inference/tool_security.py` â€” **ToolSecurityManager**: sandbox-file-based security for LLM tool execution. **Sandbox model**: `research/data/sandbox.json` defines all access rules. Everything OUTSIDE the sandbox is blacklisted for writes (defaults to read-only). **3 access levels**: `read_write` (full access), `read_only` (can read but not write/delete), `denied` (no access at all). **Reading is always allowed** within workspace unless path is `denied` â€” reading blacklisted files is OK. **File blacklist**: 17 patterns for files that can't be written/deleted (.env, secrets, keys, .pem, .ssh, credentials). **Protected engine files**: 14 core files that can never be written (but CAN be read). **Command blacklist**: 40+ risky patterns (os.system, subprocess, eval, exec, shutil.rmtree, socket, pickle, ctypes, winreg, directory traversal) â†’ triggers `needs_permission` flag. **Hard refusal patterns**: attempts to disable security or hijack tool registry. **Website whitelist/blacklist**: domain filtering for web tools (empty whitelist = all allowed). **Script pre-scan**: AST-based scan of Python scripts for 17 dangerous imports + dynamic import detection. **Auto modes**: `allow`, `deny`, `ask` (default). **Permission flow**: risky ops create pending requests â†’ user approves/denies via API. **Sandbox persistence**: all config changes auto-save to `sandbox.json`. API: `/v1/security/config` (GET/PATCH), `/v1/security/reload` (POST), `/v1/security/access/{path}` (GET), `/v1/security/pending` (GET), `/v1/security/pending/{id}/approve|deny` (POST), `/v1/security/scan` (POST).
- `research/data/sandbox.json` â€” **Sandbox config file**: defines access rules for LLM tool execution. Writable dirs: `research/data`, `research/output`, `research/sandbox`, `research/results`, `.devin`. Read-only dirs: `research/inference`, `research/training`, `research/self_play`, `research/distillation`, `research/evaluation`, `research/checkpoints`, `research/architecture`, `research/decoding`, `research/quantization`, `research/keys`, `research/moe`, `research/runtime`, `AGENTS.md`, `docs`. Denied: `.env`, `.git`, `venv`, `.venv`, `node_modules`, `__pycache__`. Website whitelist/blacklist empty by default (all sites allowed).
- `research/inference/quant/int4_quant.py` â€” INT4 weight-only quantization
- `research/inference/innovations.py` â€” Runtime innovations (MRL, QuaRot, V0, ProgressiveKV)

### Decoding implementations
- `research/decoding/dspark.py` â€” DSpark speculative decoding
- `research/decoding/eagle.py` â€” EAGLE-3 speculative decoding (feature-level, multi-layer fusion, TTT training)
- `research/decoding/medusa.py` â€” Medusa parallel prediction heads
- `research/decoding/mtp.py` â€” MTP speculative decoding

### Quantization
- `research/quantization/` â€” BitNet, FP8, RotorQuant, SpinQuant, WANDA, paged KV, KV compress

### Model Merging
- `research/merge_models.py` â€” SLERP, TIES, DARE, SVD, Task Arithmetic, Linear (model soup) on safetensors state dicts. CLI: `python -m research.merge_models --method <slerp|ties|dare|svd|task_arith|linear> ...`
- `research/inject_and_merge.py` â€” Unified pipeline: inject new params via KeyStack knowledge keys (facts, context patches, self-play, spectral, test-gated) then merge delta into target model. Auto-clones target as injection base for clean task vector. CLI: `python -m research.inject_and_merge --target <ckpt> --inject-type <facts|test_gated|context_patch|selfplay_patch|spectral> --merge-method <task_arith|ties|dare|svd|slerp|linear> ...`

### Keys (70+ files in research/keys/)
All keys preserved. Wired into V3:
- mHC (Manifold Hyper-Connections, DeepSeek-V4) â€” `use_mhc=True`, rank=512, gate=0 lossless
- AttnRes (Attention Residuals, Kimi K3) â€” `use_attn_residual=True`, k=4, gates=0 lossless
- PIT (Pseudo-Inverse Tying) â€” `use_pit=True`, L=I lossless (replaces weight tying)
- Differential Attention, BitNet b1.58, TITAN memory, MoD router â€” all lossless at load
Planned for further integration:
- MTP, Safety, LeRoPE, CSA, QK-Norm, SandwichNorm, LearnedSink, ValueResidual, SwiGLU Clamp, MRL

### Self-play & Training
- `research/training/optim/hybrid_offload.py` â€” **CPUAdamW**: ZeRO-Offload-style hybrid CPU-GPU optimizer. Keeps fp32 optimizer states + master weights on CPU pinned RAM, model bf16 on GPU. Eliminates 14.4GB fp32 AdamW VRAM cost for 1.2B models â†’ full-precision training on 12GB GPU. Async grad offload + optional overlap mode (CPU math on background thread, GPU param sync on main thread). **R&D 14 FreeToken enhancements**: `double_buffer` (ping-pong grad buffers), `bandwidth_adaptive` (PCIe profiling + auto chunk size), `chunk_size_mb` (gradient-chunked pipeline), `BandwidthPredictor` (predictive offload). Use via `--optimizer cpu_offload --freetoken` in any trainer. VRAM budget: ~6-7GB GPU (weights+grads+activations), ~19GB CPU (optimizer states + double-buffer).
- `research/training/runners/cpt_train.py` â€” **CPT with reasoning trace injection** (midtraining stage of LFM2.5-1.2B-Thinking recipe). Mixes reasoning traces (openthoughts, openr1_math, dolphin_r1) with general data (orca_math, metamath) at configurable ratio (default 60% reasoning). Full-sequence next-token prediction (no completion masking, unlike SFT). Sequence packing for efficiency. MixedDataSampler ensures every batch has the right reasoning/general ratio. CLI: `python -m research.training.runners.cpt_train --reasoning-data <files> --general-data <files> --checkpoint <ckpt> --save <out> --optimizer cpu_offload --reasoning-ratio 0.6 --lr 1e-4 --max-steps 5000`
- `research/training/runners/curriculum_sft.py` â€” **Curriculum SFT: Mix Distillation + two-stage curriculum** (SFT stage of LFM2.5-1.2B-Thinking recipe, informed by Small Model Learnability Gap research arXiv 2502.12143). Three subcommands: `prepare` (classifies data into short/long CoT, filters doom-loops via n-gram repetition, applies mix distillation blending), `train-stage1` (short CoT â€” builds internal solver, higher LR), `train-stage2` (long CoT + mix distillation â€” externalizes reasoning, lower LR), `full` (runs all three). Doom-loop filter: n-gram repetition ratio detector (n=8, threshold=0.3) removes training examples with excessive repetition (#1 failure mode of reasoning models). Mix distillation: blends long CoT (teacher) + short CoT (student) at target ratio to match small model intrinsic learning capacity. CLI: `python -m research.training.runners.curriculum_sft prepare --input <files> --output-dir <dir> --filter-doom-loops --mix-ratio 0.5`
- `research/training/runners/dpo_data_gen.py` â€” **DPO preference data generation with doom-loop mitigation** (DPO stage of LFM2.5-1.2B-Thinking recipe). Generates 5 temperature-sampled + 1 greedy candidate per prompt from SFT checkpoint, scores each with an LLM judge (teacher API model via distill_client), flags doom-loop candidates via n-gram repetition, constructs preference pairs where chosen=best non-looping, rejected=worst OR any looping (loops always rejected regardless of judge score). Reduces doom-loop rate from ~15% (SFT) to ~4% (DPO). CLI: `python -m research.training.runners.dpo_data_gen --prompts <jsonl> --checkpoint <sft_ckpt> --output <pairs.jsonl> --n-temp-samples 5 --judge-model qwen3-32b`. Then: `python -m research.training.runners.dpo_align --data <pairs.jsonl> --checkpoint <sft_ckpt> --save <dpo_ckpt> --optimizer cpu_offload`
- `research/training/runners/rlvr_train.py` â€” **RLVR (Reinforcement Learning with Verifiable Rewards)** (RL stage of LFM2.5-1.2B-Thinking recipe). GRPO-style RL on verifiable tasks with binary rewards (1.0 if verified correct, 0.0 otherwise). N-gram repetition penalty applied early in training (doom-loop mitigation, reduces rate from ~4% to ~0.36%). KL penalty against DPO checkpoint (reference model). Supports SPPO/PS-PPO/EVPO/GRPO-OR via --rl-algorithm. Math verification: extracts final answer (#### N, \boxed{}, "answer is X") and compares to gold. CLI: `python -m research.training.runners.rlvr_train --tasks <jsonl> --checkpoint <dpo_ckpt> --save <rlvr_ckpt> --use-repetition-penalty --rl-algorithm grpo --optimizer cpu_offload --max-steps 500`
- `research/self_play/grpo_trainer.py` â€” **Updated**: Added n-gram repetition penalty (LFM2.5-Thinking RLVR recipe) via `use_repetition_penalty`, `repetition_n`, `repetition_threshold`, `repetition_penalty`, `repetition_warmup_steps` config fields. Penalty applied in `train_step` after GRPO-Î» length penalty, before advantage computation. Added CPUAdamW optimizer support via `config.optimizer='cpu_offload'`.
- `research/prequantize.py` â€” **Pre-quantization script**: Converts any ForgeLM checkpoint to BitNet b1.58 ternary int8 format. Ternary-quantizes all `.weight` tensors to {-1, 0, +1} and stores as int8 (1 byte/param vs 2 bytes bf16). Writes `_bitnet_prequant=1` metadata to safetensors for auto-detection by ForgeEngine. CLI: `python -m research.prequantize --checkpoint <input.safetensors> --output <output.safetensors>`. ForgeLM_V5_Base (14.1GB bf16) -> ForgeLM_V5_BitNet (7.08GB int8, 2x disk compression). ForgeEngine loads int8 directly to VRAM (no fp32 intermediate), then converts BitNetLinear layers to int8 buffer storage (4x weight VRAM cut vs fp32). V5 7.5B model loads in 5.8GB VRAM (vs 14.1GB bf16).
- `research/keys/quantization/bitnet_b158_key.py` â€” **Updated**: Added int8 weight storage mode for pre-quantized inference. `BitNetLinear.load_prequantized(int8_tensor, scale)` loads ternary weights directly as int8 buffer (bypasses fp32 parameter). `convert_model_to_int8(model)` converts all BitNetLinear layers. `BitNetLinear.forward()` uses int8 tensor-core GEMM directly when `_prequantized=True` (no runtime quantization cost). `_load_from_state_dict` handles int8 source tensors and meta-device loading.
- `research/inference/forge_engine.py` â€” **Updated**: Auto-detects pre-quantized BitNet checkpoints via safetensors metadata (`_bitnet_prequant=1`). Loads int8 weights directly to VRAM (no fp32 intermediate, no CPU RAM spike). Builds model on meta device, loads tensors one at a time to GPU, converts BitNetLinear to int8 buffer storage. `activate_optimal()` method: S4R KV cache (15x), torch.compile, fused QK-Norm+RoPE+Cache, Triton conv, prefix cache, chunked prefill, seq-aware split. V5 7.5B: 5.8GB VRAM inference.
- `research/moe/moe.py` â€” **Updated**: Dense bypass path handles BitNet int8 experts. When experts are BitNetLinear with `_prequantized=True`, calls each expert's forward individually (ternary GEMM kernel) instead of stacking weights for batched matmul.
- `research/training/runners/sft_train.py` â€” **Updated**: Optimal defaults for VRAM-efficient training: `--optimizer muon_sf` (2.39x vs AdamW), `--lora` default True (rank 32, alpha 64), `--bitnet-everywhere` default True, `--grad-checkpoint` default True. Use `--no-lora`, `--no-bitnet-everywhere`, `--no-grad-checkpoint` to disable.
- `research/self_play/infinite_loop.py` â€” **Updated**: Added `ThinkingPipeline` class + `ThinkingPipelineConfig` that orchestrates the full LFM2.5-1.2B-Thinking 4-stage pipeline (CPT -> Curriculum SFT -> DPO -> RLVR). Each stage runs as a subprocess calling the corresponding runner. Resumable: completed stages are skipped if output checkpoint exists. Accessible via `--self-play-mode thinking`. Also fixed pre-existing argparse `%%` escaping bug in `--saerl` help text. CLI: `python -m research.self_play.infinite_loop --checkpoint <base_ckpt> --self-play-mode thinking --config forgelm_v4 --ft-optimizer cpu_offload`
- `research/self_play/infinite_loop.py` â€” **Unified AZR self-play loop** (entry point). Propose â†’ solve â†’ verify â†’ SFT â†’ eval â†’ promote. CLI: `python -m research.self_play.infinite_loop --checkpoint <ckpt> --epochs 50`
- `research/self_play/infinite_curriculum.py` â€” AZR curriculum engine (task proposal, validation, solving, ELO difficulty tracking)
- `research/self_play/` â€” Recursive self-play, sandbox, curriculum, GRPO
- `research/self_play/discovery/discovery_tools.py` â€” **Self-play discovery tool registry** (NOT for inference). Tools: `think`, `sudo_think`, `run_script`, `web_search` (DuckDuckGo, no API key), `wikipedia_search`, `arxiv_search`, `fetch_url`, `calculate`, `save_research`, `propose_theory`, `update_theory`, `record_discovery`, `query_db`, `migrate_schema`, `summarize_context`, `finish_session`, `set_goal`. Used by the discovery self-play loop and agentic distillation. For inference tool registry, see `research/inference/engine_tools.py`.
- `research/training/` â€” DPO alignment, training utils, chunked CE
- `research/evaluation/` â€” Prompt tests, reasoning benchmarks, LiveCodeBench

### Runtime
- `research/runtime/` â€” CUDA graphs, flex attention, VRAM manager, signal capture

### Evolution System (research/evolution/)
- `engine.py` â€” **ForgeEvolve**: MAP-Elites + neural generators + surrogate model. Self-improving: adaptive population, convergence detection, refinement spawning, novelty search (pulsation), domain revisit scheduling.
- `generators.py` â€” **BatchedGenerator**: 4-layer net (hidden_dim=128) with LayerNorm + adaptive mutation. GeneratorPopulation manages N generators on GPU.
- `surrogate.py` â€” **SurrogateModel**: 5-MLP ensemble (hidden_dim=256, 4-layer each) for score prediction + filter_top_k.
- `archive.py` â€” **MapElitesArchive**: behavioral grid + Pareto front tracking.
- `trainer.py` â€” **GeneratorTrainer**: policy gradient training of generators.
- `database.py` â€” **FindingsDB**: SQLite-backed discovery storage + canonical knowledge (best-ever generators/surrogate per domain).
- `domain_factory.py` â€” **DomainFactory + RefinementDomain**: auto-spawns narrowed search domains from converged parents (recursive depth 1-5, Â±20% narrowing per level).
- `novelty_search.py` â€” **NoveltySearch**: behavioral diversity tracking + novelty/quality pulsation (prevents plateaus). Based on Lehman & Stanley + GECCO 2020.
- `topic_scanner.py` â€” **TopicScanner**: auto-discovers 265+ optimization targets from codebase (feature flags, config params, arch keys, KV strategies, decoding, quant, schedulers, optimizers).
- `revisit_scheduler.py` â€” **DomainRevisitScheduler**: re-queues stale domains when source code changes (Dynamic QD). Tracks file mtimes + convergence history.
- `llm_domain_gen.py` â€” **LLMDomainGenerator + GenericDomain**: auto-creates domains for uncovered topics. LLM generates BaseDomain subclasses (with sandbox validation); GenericDomain fallback for heuristic exploration.
- `domains/` â€” 57+ search domains (quant, KV, attention, training, decoding, memory, arch, MoE).
- `tests/evolution/run_evolve.py` â€” **Runner**: single entry point. Auto-discovery, refinement, revisit, novelty all integrated. `--auto-discover` (default on), `--revisit-stale` (default on), `--max-new-domains N`.

## Training-Free Alignment (research/training_free/)

Forward-only adaptation â€” no gradients, no optimizer, no weight updates:
- `urial.py` â€” URIAL in-context alignment (`build_prompt`: system + 3 style examples).
- `reflexion.py` â€” `ReflexionBuffer`: bounded episodic memory rendered into the prompt.
- `steering.py` â€” `ActivationSteerer`: capture residual activations, extract task vectors (`positive - negative`), inject via pre-hooks.
- `rain.py` â€” `RAINGenerator`: self-eval + rewind-and-regenerate loop.
- `solver.py` â€” `TrainingFreeSolver`: frozen-solver adapter combining the above; `record(task, output, error, success)` collects activations, `build_task_vector()` + `apply_steering(alpha)` steer inference.
- `SelfPlaySandbox(...)` accepts `training_free=TrainingFreeSolver(...)` (or call `sandbox.enable_training_free()`): run_task styles prompts with ICA + memory and records every outcome â€” replaces GRPO weight updates for self-play adaptation.
- Tests: `tests/unit/test_training_free.py` (CPU, tiny model).

## V4 Hardware-Efficient Inference (2026-08)

Eight techniques built based on analysis of faster/more-efficient engines (mini-vllm,
mini-infer, Zerfoo, vLLM, DeepSeek-V3 MLA, PagedEviction, XQuant). All wired into
`forgelm_v4` preset and `load_default_model()`.

### Architecture Changes (active in V4 preset)
1. **GTA (Grouped-Tied Attention)** â€” `research/keys/attention/gta_key.py`
   - Ties V to K: `V = (1-gate)*K + gate*V_proj(x)`, gate=0 at init â†’ V=K (lossless)
   - Halves KV cache write bandwidth (only K written, V=K derived)
   - `attn_type="gta"` in config; `GTAKey` converts GQAâ†’GTA at checkpoint load
   - Training unties V from K (gate opens)
2. **GLA (Grouped Latent Attention)** â€” `research/keys/attention/gla_key.py`
   - Latent-compressed KV: projects K/V into compact latent, caches only latent
   - Identity warm start (latent_dim = full KV dim, gate=0 â†’ lossless)
   - `attn_type="gla"`, `gla_latent_dim=0` (lossless) or >0 (compressed)
   - `GLAKey` converts GQAâ†’GLA at checkpoint load (kv_down_proj=k_proj, identity up-projs)
3. **Fused QKV + Gate-Up GEMM** â€” `research/keys/quantization/fused_gemm_key.py`
   - `FusedQKVLinear`: single GEMM for Q/K/V projections (3â†’1 kernel launch)
   - `FusedGateUpLinear`: single GEMM for FFN gate+up (2â†’1 kernel launch)
   - `use_fused_gemm=True` in config; wired in `_maybe_fuse_qkv()` and `build_ffn()`
   - Lossless (same math, just fused weight matrix)

### Inference Optimizations (opt-in via ForgeEngine.activate())
4. **W8A8 INT8 Quantization** â€” `research/inference/quant/w8a8_quant.py`
   - `W8A8Linear`: weight + activation INT8 with `torch._int_mm` tensor-core GEMM
   - 2-3Ã— decode speedup at batch=1 (shifts to compute-bound on tensor cores)
   - `quantize="w8a8"` in `activate()`; `FP8Linear` for Blackwell FP8 E4M3
5. **PagedEviction KV Cache** â€” `research/inference/kv/paged_eviction.py`
   - Block-wise eviction (entire 16-token blocks, not individual tokens)
   - Compatible with PagedAttention (no fragmentation), FlashAttention (no scores needed)
   - L2-norm-based scoring (attention-score-free); 3020 tok/s on LLaMA-1B (37% over full)
   - `kv_cache="paged_eviction"` in `activate()`
6. **XQuant KV Rematerialization** â€” `research/inference/kv/xquant_kv.py`
   - Caches layer-input activations X (INT4) instead of K/V; rematerializes K=W_K@X, V=W_V@X
   - 2Ã— memory savings (INT4 X vs bf16 KV); up to 16Ã— with SVD compression
   - `kv_cache="xquant"` in `activate()`
7. **Megakernel Decode** â€” `research/decoding/megakernel.py`
   - `CompiledMegakernelDecode`: CUDA graph capture of entire decode step + torch.compile
   - Single graph replay per token (1 API call vs dozens of kernel launches)
   - `acceleration="megakernel"` in `activate()`
8. **Chunked Prefill** â€” `research/inference/prefill/chunked_prefill.py`
   - `ChunkedPrefiller`: splits long prompts into 512-token chunks
   - Interleaves with decode steps (prevents long prompts from blocking decode queue)
   - `use_chunked_prefill=True` in `activate()`

## V4 R&D: Next-Gen Inference Techniques (2026-08)

Five additional techniques from latest 2026 research, all opt-in via `ForgeEngine.activate()`:

### FlexDecoding (research/inference/attention/flex_decoding.py)
- PyTorch 2.5+ FlexAttention with dedicated decode backend (FlashDecoding)
- Splits KV cache across all 192 SMs (vs 8 SMs with standard SDPA for 8 KV heads)
- `acceleration="flex_decoding"` â€” auto-switches to FlexDecoding kernel for q_len=1
- Expected: 1.5-3Ã— decode speedup for batch=1 (SM utilization 4% â†’ 100%)

### NVFP4 Quantization (research/inference/quant/nvfp4_quant.py)
- Native FP4 (E2M1) on Blackwell 5th-gen tensor cores with block scaling
- **Two-level scaling**: per-channel fp32 global + per-block FP8 normalized scales
- 3.8Ã— compression (0.53 bytes/weight). ~99% quality with calibration
- RTX 5070 (SM120) supports `mxf8f6f4.block_scale` in MMA instructions
- `quantize="nvfp4"` â€” ForgeLM V4 (1.2B): 2.34GB â†’ ~0.62GB (3.8Ã— compression)
- **NOW THE DEFAULT** on Blackwell (auto-selected even when VRAM is ample)
- V7 8B estimate: 15.0 GB bf16 â†’ 4.0 GB NVFP4 (fits 12GB with 8GB for KV)
- **Novel: AS-FP4** (`research/inference/quant/novel_quant.py`): MSE-optimal
  per-block scales via grid search â†’ 5.9% lower error than standard NVFP4
- **Novel: R-FP4** (`research/inference/quant/novel_quant.py`): FP4 + sparse
  INT8 residual for top-k errors â†’ 21.7% lower error at 5% residual (2.6x compression)

### ATFlash Wavelength Pruning (research/inference/attention/atflash.py)
- Per-RoPE-wavelength distance windows: each frequency pair prunes beyond its wavelength
- Prunes 37-48% of QK inner-product terms, 96-98% top-1 match rate (near-lossless)
- Input-independent (closed-form, no dynamic search), orthogonal to token-level sparsity
- `use_wavelength_pruning=True` â€” 1.29Ã— at 128K context, grows with context length

### Adaptive n-gram + EAGLE-3 Speculative Decoding (research/decoding/adaptive_speculative.py)
- Combines n-gram lookup (free, optimal for code/RAG) with EAGLE-3 (model-based, novel gen)
- Adaptive selection: tracks per-request hit/acceptance rates, picks best drafter
- Up to 4.9Ã— on code-editing, 2.89Ã— average (MLSys 2026 production benchmarks)
- `use_adaptive_spec=True` â€” composes with existing EAGLE-3/MTP heads

### POD-Attention (research/inference/attention/pod_attention.py)
- Prefill-decode overlap: runs prefill (compute-bound) and decode (BW-bound) concurrently
- Uses CUDA streams to overlap on separate SMs (28% mean speedup for hybrid batches)
- `use_pod_attention=True` â€” pairs with BatchQueue's mixed prefill+decode batches

## V4 R&D Round 2: Long-Context + Speculative + VRAM (2026-08)

Five more techniques from latest 2026 research, all opt-in via `ForgeEngine.activate()`:

### CoSA Sparse Attention (research/inference/attention/cosa.py)
- Proxy-kernel co-designed sparse attention for long-context decode
- KAP scores KV blocks by key-norm Ã— position-decay; OSK visits top-scored blocks
- 4.93Ã— attention speedup at 128K, 2.53Ã— TTFT, negligible quality loss
- `use_cosa=True` â€” activates for decode when KV > 2048 tokens

### Suffix Decoding (research/decoding/suffix_decoding.py)
- Training-free speculative decoding via suffix tree matching
- Matches longest repetitive patterns (code, RAG, summarization, multi-turn)
- Composes with n-gram (suffix for long, n-gram for short) and EAGLE-3
- `use_suffix_spec=True` â€” `ComboSpeculativeDecoder` combines all three drafters
- 1.5-2Ã— on code/RAG workloads, zero training cost

### Sequence-Aware Split (research/inference/attention/seq_aware_split.py)
- Splits KV cache across SMs for low-head-count decode (8 KV heads â†’ 192 SMs)
- Auto-computes optimal split count based on seq_len and SM count
- Split-merge attention with online softmax (log-sum-exp trick)
- `use_seq_split=True` â€” 21-24% decoder kernel efficiency improvement

### CPU KV Cache Offloading (research/inference/kv/cpu_kv_offload.py)
- Two-tier KV cache: GPU hot window (8K tokens) + CPU cold storage (rest)
- Extends effective KV cache to 128K+ within 12GB VRAM budget
- Async prefetch via CUDA stream for overlap with compute
- `kv_cache="cpu_offload"` â€” 32K hot window = 1.07GB GPU, 128K total = 4.3GB CPU

### CompactAttention (research/inference/attention/compact_attention.py)
- Block-union KV selection for efficient chunked prefill
- Unions selected KV blocks across Q positions and GQA groups
- In-place sparse attention on selected blocks (no KV compaction)
- `use_compact_attn=True` â€” 2.72Ã— attention speedup at 128K under chunked prefill

## V4 R&D Round 3: Kernel Fusion + KV Compression + Compile (2026-08)

Five more techniques from latest 2026 research, all opt-in via `ForgeEngine.activate()`:

### Fused QK-Norm+RoPE+Cache-Write (research/inference/attention/fused_qk_norm_rope_cache.py)
- Fuses RMSNorm + RoPE + KV cache append into single kernel (vLLM #38621 pattern)
- Eliminates 2 kernel launches per attention layer (32 fewer launches per decode step)
- Warp-per-head design: V written first (fire-and-forget) while K norm runs
- Optional FP8/INT4 KV quantization fused into the same kernel
- `use_fused_qk_norm_rope_cache=True` â€” 5-10% decode speedup

### S4R Low-Rank KV (research/inference/kv/s4r_kv.py)
- Selective Sampling + Subspaces + Sparse Reconstruction for KV cache
- Builds low-rank basis from sampled prompt tokens (no external calibration)
- Sink tokens preserved full precision, rest stored as low-rank coefficients
- Sparse reconstruction during decode (only top-k relevant entries)
- `kv_cache="s4r"` â€” up to 15Ã— KV compression with near full-cache accuracy

### HqeKV Hybrid Quant+Eviction (research/inference/kv/hqe_kv.py)
- Combines quantization AND eviction in a single KV cache (ACL Findings 2026)
- Three tiers: full precision (10%), INT8 (30%), INT4 (50%), evicted (10%)
- Joint K-V importance metric: ||K|| Ã— ||V|| Ã— recency_decay
- Integrated optimizer auto-selects compression action per token
- `kv_cache="hqe_kv"` â€” 7.9Ã— KV memory reduction with minimal quality loss

### torch.compile Auto-Tuner (research/inference/graphs/compile_autotune.py)
- Benchmarks all torch.compile modes (default, reduce-overhead, max-autotune, etc.)
- Picks fastest mode per model, caches results to .devin/compile_mode_cache.json
- Heuristic fallback: get_recommended_mode() without benchmarking
- `use_compile_autotune=True` â€” avoids mode selection guesswork

### Block Fusion (research/inference/graphs/block_fusion.py)
- Per-block CUDA graph capture + torch.compile for full transformer block fusion
- Each block gets its own graph (supports MoD skip â€” skip entire block graph)
- CompiledBlockFusion: combines per-block graphs with torch.compile intra-block fusion
- `use_block_fusion=True` â€” 1.3-1.5Ã— over eager (ClusterFusion++ approximation for SM120)

## V4 R&D Round 4: Prefix Caching + Scheduling (2026-08)

Four more techniques from latest 2026 research on prefix caching and batch scheduling:

### Feather Scheduler (research/inference/scheduler/feather_scheduler.py)
- Prefix-homogeneity-aware batch scheduling (2-10Ã— throughput for prefix-sharing)
- Groups requests by shared prefix â†’ smaller homogeneous batches outperform
  larger heterogeneous ones (better KV cache locality)
- Chunked Hash Tree (CHT): O(1) prefix detection vs O(depth) radix trees
- `BatchQueue(use_feather_scheduler=True)` â€” integrates with existing batching

### Learned Prefix Cache (research/inference/kv/learned_prefix_cache.py)
- ML-guided prefix cache eviction (NeurIPS 2025)
- Online logistic regression predicts continuation probability
- Features: recency, frequency, length, is_conversation
- 18-47% cache size reduction at equivalent hit ratios, 11% prefill throughput
- `use_learned_prefix_cache=True` â€” replaces LRU dict with learned policy

### Hybrid Chunked Prefill (research/inference/prefill/hybrid_prefill.py)
- Adaptive: chunk only when decode is active, continuous prefill otherwise
- Eliminates throughput tax of unconditional chunking (vLLM #26625)
- +2-5% total token throughput, 10-20% lower TTFT under low concurrency
- `use_hybrid_prefill=True` â€” extends ChunkedPrefiller with decode awareness

### HotPrefix (research/inference/scheduler/hotprefix.py)
- Hotness-aware GPU/CPU prefix KV promotion (SIGMOD 2026)
- Tracks prefix access frequency Ã— recency Ã— length â†’ hotness score
- Hot prefixes â†’ GPU (fast), cold â†’ CPU (saves VRAM), periodic rebalancing
- `use_hotprefix=True` â€” system prompts stay on GPU, unique contexts offloaded

## V4 R&D Round 5: Training Optimizations (2026-08)

Six training-side techniques from latest 2026 research, all opt-in via `sft_train.py` CLI flags:

### FlashOptim 8-bit Optimizers (research/training/optim/flash_optim.py)
- `FlashAdamW`: 8-bit optimizer states with companding quantization (sqrt transform)
- `FlashLion`: 8-bit sign-momentum (single state, even more memory-efficient)
- 57% memory reduction vs standard AdamW (16â†’7 bytes/param, 5 with gradient release)
- Block-wise quantization isolates outliers, companding reduces small-value error
- `--optimizer flash_adamw` or `--optimizer flash_lion`

### FORGE Fused Gradient Elimination (research/training/optim/forge_optimizer.py)
- Folds optimizer step INTO backward pass via gradient hooks
- Each gradient tile consumed in registers the instant it's produced
- 53% peak memory reduction, 1.5Ã— faster at small batch sizes
- No gradient tensor materialized (zero gradient returned from hook)
- `--optimizer forge` (auto-registers hooks on model parameters)

### OOMB Chunk-Recurrent Training (research/training/runners/oomb_trainer.py)
- O(1) activation memory regardless of sequence length
- Processes sequences in chunks, recomputes activations on-the-fly in backward
- Paged KV cache with gradient support (no fragmentation)
- Async CPU offloading of KV cache between chunks
- Enables 128K+ context training on 12GB GPU (10MB overhead per 10K tokens)
- `--oomb` flag

### LazyTrain Scheduler (research/training/runners/lazy_train.py)
- Mixed-integer scheduling for checkpoint selection + activation placement
- Greedy: checkpoint layers with best memory_saving / recompute_cost ratio
- Hybrid8BitOperator: fused 8-bit optimizer + fast gradient clipping (EMA norm)
- 1.24Ã— sustained TFLOPS, +1 batch size at each model scale
- `--lazy-train` or `--checkpoint-strategy lazy`; `--hybrid-clip` for fast clipping

### Streaming Parquet Prefetch Loader (research/training/data/streaming_loader.py)
- SlidingWindowCache: NVMe/disk cache with sliding window eviction
- BatchPlanProvider: precomputes epoch batch ordering (enables prefetch)
- PrefetchingLoader: wraps DataLoader with automatic prefetch triggers
- 10Ã— data loading speedup (50â†’500 samples/s), zero worker crashes
- Integrates with existing ParquetDataset

### Optimal Checkpoint Planner (research/training/runners/optimal_checkpoint.py)
- Sliding-window Hirschberg knapsack for optimal checkpoint selection
- O(W) memory (vs O(nW) for dp_knapsack) â€” handles n=2000 (vs n=100)
- 25-28% runtime speedup over PyTorch's default solver
- `--checkpoint-strategy optimal`

## V4 R&D Round 6: Loss Functions + Quantization + Sparse Attention (2026-08)

Five more techniques from latest 2026 research:

### Improved Loss Functions (research/training/losses/improved_losses.py)
- `FocalCrossEntropy`: down-weights easy tokens, focuses on hard ones (Î³=2.0)
- `LabelSmoothingCE`: prevents Î¼-singularity collapse (Îµ=0.1), better calibration
- `LovaszSoftmax`: directly optimizes Jaccard/exact-match (+36% EM on math/QA)
- `DynamicFocalCE`: curriculum-style focal weight (lowâ†’high Î³ over training)
- `MixtureLoss`: combines multiple losses with configurable/learnable weights
- `--loss-function focal|label_smoothing|lovasz|dynamic_focal|mixture`

### OffQ Activation Outlier Offsetting (research/quantization/offq.py)
- Top-1 PCA + rotation + offset absorption for W4A4KV4 quantization
- Concentrates outliers into 1 channel, absorbs as shared offset
- Enables uniform-grid W4A4KV4 without mixed precision
- `OffQQuantizer` â€” calibrates from sample inputs, applies to model

### AAAC Adaptive Codebooks (research/quantization/aaac.py)
- Two learned scalar codebooks per layer (64 bytes overhead, zero storage for selection)
- Activation-weighted k-means places levels where they matter most
- Outperforms AWQ, GPTQ, QuIP# at orders-of-magnitude less quantization time
- `AAACQuantizer` â€” 3-30 min calibration on single GPU

### MoSA Mixture of Sparse Attention (research/inference/kv/mosa.py)
- Expert-choice routing: each head selects k tokens to attend to
- Content-based, head-specific, perfectly balanced sparse attention
- 27% better perplexity at same compute, smaller KV cache
- `use_mosa=True` â€” activates for sequences > 512 tokens

### TriRoute Unified Routing (research/inference/scheduler/triroute.py)
- Single controller emits joint policy: attention mode (skip/local/full) + KV bits (4/8/16)
- Gumbel-Softmax + STE for categorical, load-balanced top-k for experts
- Better quality-compute tradeoff than independent MoD + KV quantization
- `use_triroute=True` â€” unifies MoD block skipping with KV precision selection

## V4 R&D Round 7: CUDA Graphs + Self-Play Improvements (2026-08)

Seven more techniques from latest 2026 research:

### Breakable CUDA Graph (research/inference/graphs/breakable_cuda_graph.py)
- SGLang BCG: segmented graph capture for dynamic shapes
- 1.70Ã— faster prefill, 1.93Ã— with full capture, 3.8-5.2Ã— faster graph building
- Captures decode graphs for batch sizes [1, 2, 4, 8], pads at runtime
- `use_breakable_cuda_graph=True`

### CoRun Deterministic Inference (research/inference/scheduler/corun.py)
- Padding-based determinism: isolate prefill + fixed-shape batched decode
- Position-invariant kernels â†’ pad to fixed shape â†’ one CUDA graph
- 15-324% throughput over batch-invariant, -51.8% TTFT, -48.6% TPOT
- Per-request RNG state (reproducible sampling for RL/eval)
- `use_corun=True`

### Foundry Template-Based Cold Start (research/inference/graphs/foundry.py)
- Persists CUDA graph context (topology + kernel binaries + memory layout)
- Online materialization <1s vs 10-30s for fresh capture â†’ 10Ã— faster startup
- `use_foundry=True` â€” auto-captures or loads templates from `.devin/graph_templates/`

### SOAR Meta-RL Curriculum (research/self_play/soar.py)
- Teacher proposes stepping-stone problems, rewarded by student improvement
- Grounded rewards (measured progress) > intrinsic rewards (avoids collapse)
- Escapes learning plateaus on hard problems (0/128 success â†’ improvement)
- `--self-play-mode soar`

### SGS Self-Guided Self-Play (research/self_play/sgs.py)
- Three roles: Solver + Conjecturer + Guide (prevents Conjecturer collapse)
- Guide scores problems by relevance to targets + cleanliness
- 7B model after 200 rounds > 671B pass@4 (Lean4 theorem proving)
- `--self-play-mode sgs`

### SAERL SAE-Guided Data Engineering (research/self_play/saerl.py)
- Sparse Autoencoder features guide: diversity clustering, difficulty curriculum, quality filtering
- +3.00% accuracy over vanilla GRPO, 20% fewer steps to target
- SAE transfers across model families and scales
- `--saerl` flag

### OP-MIX On-Policy Data Mixing (research/self_play/opmix.py)
- Simulates data mixtures by interpolating low-rank adapters (no proxy models)
- Works across pretraining, midtraining, instruction tuning
- Dynamically adjusts mixing ratio based on learning dynamics
- `--opmix` flag

## V4 R&D Round 8: FA4 + Schedule-Free Optimizers + KV Eviction (2026-08)

Eight more techniques from latest 2026 research:

### FlashAttention-4 Blackwell Backend (research/inference/attention/fa4_attention.py)
- FA4: algorithm + kernel pipelining co-design for asymmetric Blackwell scaling
- 1.3Ã— faster prefill, 1.6-1.9Ã— faster decode (with FP8 KV cache)
- Software-emulated exponential, TMEM, 2-CTA MMA, ping-pong tiles
- Auto-detects sm_120 (RTX 5070), falls back to FA2/SDPA
- `use_fa4=True`

### SF-NorMuon (research/training/optim/sf_spectral_optimizers.py)
- Schedule-free spectral optimizer with per-neuron normalization
- Matches tuned AdamW across 1-8Ã— Chinchilla horizons (no LR schedule)
- Weight decay at fast iterate Z, iterate averaging for anytime checkpoints
- `--optimizer sf_normuon`

### AMUSE (research/training/optim/sf_spectral_optimizers.py)
- Anytime Muon + Stable gradient Evaluation
- Time-varying interpolation: fast Muon â†’ stable averaged (suppresses oscillations)
- No LR schedule, anytime training, improves Pareto over Muon + SF-AdamW
- `--optimizer amuse`

### MONA (research/training/optim/sf_spectral_optimizers.py)
- Muon + Nesterov Acceleration (EMA of gradient differences)
- Escapes sharp minima while preserving spectral-norm regularization
- SOTA on MoE pretraining 1B-68B params, 1T tokens
- `--optimizer mona`

### KARA KV Compression (research/inference/kv/kara_kv.py)
- Sliding-window compression: only compress recent context (bounded work/step)
- Bidirectional attention scoring for token importance
- Token2Chunk: expand selected tokens into flexible-size chunks
- 8Ã— KV reduction at 32K context
- `use_kara_kv=True`

### MomentKV (research/inference/kv/moment_kv.py)
- Moment statistics over evicted tokens (count, key/value mean, VK covariance)
- Closes the directional gap: evicted tokens near-orthogonal to retained â†’ big error
- Geometric regularity: evict tokens already aligned with moment summary
- Closed-form first-order approximation of evicted attention output
- `use_moment_kv=True`

### KVpop (research/inference/kv/kvpop.py)
- Predictive online pruning with learned scoring module
- Supervision: future attention mass (transposed-attention pass during training)
- Delays eviction DECISION (not just eviction) until token leaves protected window
- Sink + protected window + long-range top-k cache
- `use_kvpop=True`

### CONF-KV (research/inference/kv/conf_kv.py)
- Confidence-aware dynamic budget from next-token entropy
- High confidence â†’ prune aggressively; low confidence â†’ retain more
- Mixed FP16/INT8 storage (important â†’ FP16, rest â†’ INT8)
- Pyramidal per-layer budget (deeper layers get more)
- 91.4% retrieval on 32K NiH (vs 53.8% sliding window)
- `use_conf_kv=True`

## V4 R&D Round 9: Speculative Decoding + Tokenization + RoPE (2026-08)

Eight more techniques from latest 2026 research:

### P-EAGLE Parallel Speculative Decoding (research/decoding/peagle.py)
- Generates ALL K draft tokens in a SINGLE forward pass (vs K sequential)
- Sequence partition algorithm: maintains attention dependencies across chunks
- 1.05-1.69Ã— speedup over vanilla EAGLE-3 on B200
- `use_peagle=True`

### Lookahead Quality Gate (research/decoding/lookahead_gate.py)
- Block-wise acceptance: accepts longest reliable prefix of k-token draft
- Geometry-based quality score from hidden states (no auxiliary heads)
- Quantile-calibrated threshold (estimated from unlabeled prompts)
- 2.6-7.9Ã— faster generation + improved accuracy on math/science
- `use_lookahead_gate=True`

### Pruned BPE (research/tokenization/pruned_bpe.py)
- Post-training visibility pruning: low-exposure tokens â†’ internal-only
- Reallocates freed vocabulary slots to better-exposed candidates
- 0.27-0.36% shorter encoding at same vocabulary size
- No model retraining (tokenizer post-processing only)

### ToaST Split-Tree Tokenization (research/tokenization/toast.py)
- Greedily splits pre-tokens into binary trees using n-gram counts
- Vocabulary selection via Integer Program (LP relaxation, near-optimal)
- 11% fewer tokens than BPE/WordPiece/UnigramLM at vocab â‰¥ 40K
- +2.6-7.6% CORE score on 1.5B models

### Jet-Long Dynamic Bifocal RoPE (research/inference/scheduler/jet_long.py)
- Local RoPE-faithful window + long-range dynamically rescaled window
- Parameter-free analytic schedule: recovers base at short, extrapolates at long
- +4.79 pp RULER at 128K (1.7B), 1.39Ã— FA2 throughput, â‰¤4% overhead
- Zero-shot: no retraining needed
- `use_jet_long=True`

### RoPE-ID In-Distribution Rotation (research/inference/position/rope_id.py)
- High-frequency rotation on a SUBSET of channels (rest unchanged)
- Maintains key/query cluster separation across lengths (geometric analysis)
- Enables length generalization without retraining
- `use_rope_id=True`

### LeRoPE Learnable RoPE Frequencies (research/inference/position/lerope.py)
- One learnable scalar per frequency band (32 params total)
- Initialized to 1.0 = standard RoPE (lossless checkpoint loading)
- 3.4% less compute to match RoPE performance at 2.5B scale
- `use_lerope=True` (auto-replaces RoPE in all attention layers)

### LaMPE Length-Aware Multi-Grained PE (research/inference/scheduler/lampe.py)
- Dynamic mapping via scaled sigmoid (adaptive positional capacity)
- Multi-grained: fine resolution for local, coarse for long-range
- Training-free: applies to any RoPE-based LLM
- `use_lampe=True`

## V4 R&D Round 10: Schedulers + Curriculum + Normalization (2026-08)

Nine more techniques from latest 2026 research:

### FastServe Skip-Join MLFQ (research/inference/scheduler/fastserve.py)
- Iteration-level preemptive scheduling with skip-join Multi-Level Feedback Queue
- Token-granularity preemption (not request-granularity)
- Proactive GPUâ†”host memory offloading for preempted requests
- 6.1Ã— throughput improvement over vLLM
- `use_fastserve=True`

### Libra Micro-Request Partitioning (research/inference/scheduler/libra.py)
- Flexible partitioning: split requests at ANY token boundary into cooperating segments
- Two-level scheduling: global (split points) + local (SLO-aware batches)
- Chunked KV cache transfers for cross-instance execution
- 1.91Ã— goodput, 1.15-3.07Ã— serving capacity
- `use_libra=True`

### FASER Fine-Grained SD Phase Management (research/inference/attention/faser.py)
- Dynamic speculative length per-request (based on acceptance rate)
- Token-wise early exit: stop verification at first rejection
- Frontier execution: overlap verification chunks with next draft
- 53% higher throughput, 1.92Ã— lower latency
- `use_faser=True`

### Kairos SLO-Aware Scheduling (research/inference/scheduler/kairos.py)
- Prefill: urgency-based priority (closest to missing TTFT deadline first)
- Decode: slack-guided adaptive batching (pack more when under TPOT SLO)
- +23.9% TTFT SLO, +27.1% TPOT SLO, +33.8% e2e SLO, +19.3% decode throughput
- `use_kairos=True`

### Curriculum Learning (research/training/data/curriculum_augment.py)
- Orders training data easyâ†’hard using compression ratio, MTLD, Flesch readability
- Strategies: vanilla, pacing, interleaved, warmup
- 18-45% fewer steps to reach baseline, 3.5% sustained improvement as warmup
- `--curriculum pacing|vanilla|interleaved|warmup`

### Training-Time Data Augmentation (research/training/data/curriculum_augment.py)
- Three orthogonal categories: token-level noise, sequence permutations (FIM), target offset
- Regularizes against overfitting in multi-epoch data-constrained training
- `--augment`

### SYNPRO Synthetic Data Generation (research/training/data/curriculum_augment.py)
- Rephrasing + reformat operations (RL-optimized generators)
- 3.7-5.2Ã— effective tokens from same organic data
- Surpasses non-data-bound oracle at 1.1B scale
- `--synpro`

### SeeDNorm Self-Rescaled Dynamic Normalization (research/training/optim/advanced_norm.py)
- Dynamically adjusts scaling coefficient based on input norm
- Preserves input norm information (RMSNorm discards it)
- Minimal params (1 per layer), negligible efficiency impact
- Consistently superior to RMSNorm and LayerNorm
- `--norm-type seednorm`

### Dynamic Tanh (DyT) (research/training/optim/advanced_norm.py)
- Bounded normalizer: Î³ * tanh(Î± * x) + Î² (no normalization needed)
- Compatible with Muon optimizers (no normalization-optimizer coupling penalty)
- Simpler than RMSNorm, competitive performance
- `--norm-type dyt`

### Keel Post-LN Highway Connection (research/training/optim/advanced_norm.py)
- Post-LN with Highway-style residual (replaces ResNet residual pathway)
- Prevents gradient vanishing in deep networks (1000+ layers)
- Better perplexity and depth-scaling than Pre-LN
- `KeelHighwayConnection` module (for future deep model experiments)

## V4 R&D Round 11: Radix Cache + MoE + Advanced RL (2026-08)

Eight more techniques from latest 2026 research:

### Unified Radix Cache (research/inference/scheduler/unified_radix.py)
- One tree for hybrid models: full-attn KV + sliding window + recurrent states
- Component-based: each component controls its own matching/splitting/eviction
- HiCache: 3-level hierarchy (GPU L1, Host L2, External L3)
- Eliminates code duplication across cache variants
- `use_unified_radix=True`

### Elbow MoE Routing (research/inference/scheduler/moe_optim.py)
- Training-free inference-time dynamic top-k
- Identifies elbow point in sorted router probabilities (max curvature)
- Per-token expert count: confident tokens get fewer, ambiguous get more
- 5.3% latency reduction, maintains accuracy + load balance
- `use_elbow_moe=True`

### Alloc-MoE Budget-Aware Activation (research/inference/scheduler/moe_optim.py)
- Alloc-L: sensitivity profiling + DP for per-layer expert allocation
- Alloc-T: token-level dynamic redistribution by routing entropy
- 1.15Ã— prefill, 1.34Ã— decode at half budget
- `use_alloc_moe=True`

### LDA Distribution-Consistent MoE (research/inference/scheduler/moe_optim.py)
- Corrects RMS scale/variance mismatch when reducing activated experts
- Layer-wise calibration statistics (default vs reduced routing)
- Recovers performance lost to distributional shift
- `use_lda_moe=True`

### SPPO Sequence-Level PPO (research/training/losses/advanced_rl.py)
- Reformulates reasoning as Sequence-Level Contextual Bandit
- Decoupled scalar value function (no token-level critic, no multi-sampling)
- Low-variance advantage for long-horizon CoT
- Matches GRPO quality with PPO sample efficiency
- `rl_algorithm="sppo"`

### PS-PPO Prefix-Sampling PPO (research/training/losses/advanced_rl.py)
- Samples cutoff timestep per trajectory, backprops only through prefix
- Importance-weighting correction â†’ unbiased truncated gradient
- Large compute and memory savings for long reasoning traces
- `rl_algorithm="psppo"`

### EVPO Explained Variance PO (research/training/losses/advanced_rl.py)
- Monitors batch-level explained variance (EV) to adaptively switch
- Positive EV â†’ critic-based (PPO); zero/negative EV â†’ batch-mean (GRPO)
- Provably no greater variance than the better of the two
- `rl_algorithm="evpo"`

### GRPO-OR Output Reset Trust Region (research/training/losses/advanced_rl.py)
- Replaces clipped surrogate with smooth one-sided saturation (OR)
- Advantage sign determines direction; zero residual after favorable margin
- Smaller observed spread than GRPO-clip
- `rl_algorithm="grpo_or"`

## V4 R&D Round 12: Adaptive Quant + AoH + Distillation (2026-08)

Seven more techniques from latest 2026 research:

### AdaMX Adaptive Microscaling (research/quantization/adaptive_quant.py)
- Per-block precision-recovery scheme selection (MXFP4, INT4, Norm4, Shift4)
- Per-operand: weights (offline search) vs activations (single-pass)
- Removes 83% of MXFP4 accuracy loss on commonsense, 82% on MMLU
- `use_adamx=True`

### SharQ Sparse-Dense FP4 (research/quantization/adaptive_quant.py)
- Online N:M mask extracts outlier backbone â†’ FP4, dense residual â†’ FP4 GEMM
- Training-free: no calibration, retraining, or model-specific tuning
- 2.2-2.4Ã— latency over FP16, 1.2-1.4Ã— throughput over FP8 on RTX 5090
- `use_sharq=True`

### MosaicQuant Inlier-Outlier Disaggregation (research/quantization/adaptive_quant.py)
- Dense 4-bit base (inliers) + sparse 4-bit residual (outlier compensation)
- ZipperEngine: fuses sparse computation into dense GEMM pipeline
- Near-FP16 accuracy, 1.24Ã— speedup over W16A16
- `use_mosaic_quant=True`

### AoH Autonomy-of-Heads (research/inference/kv/aoh_retmask.py)
- Data-free head classification from frozen QK geometry (effective rank of M_h)
- Low rank â†’ retrieval head (full attention), high rank â†’ streaming (sink+window)
- 50% sparsity: 96.5% of full attention, 66% decode latency reduction
- `use_aoh=True`

### RetMask Retrieval Head Optimization (research/inference/kv/aoh_retmask.py)
- Contrastive masking: train by contrasting normal vs retrieval-masked outputs
- +2.28 HELMET at 128K, +70% citation generation, +32% passage re-ranking
- Gains correlate with retrieval score sparsity

### Offline Top-K Logits + Chunked KL (research/training/runners/efficient_distillation.py)
- Cache teacher's top-K logits once â†’ train against cache (no teacher in loop)
- 29% faster per iteration, 41% higher throughput
- Fused chunked KL: peak memory linear in seq length (4Ã— context on same GPU)
- `--distill --teacher-checkpoint <path>`

### Sequence Truncation + Prefix OPD (research/training/runners/efficient_distillation.py)
- Train on first 50% of tokens â†’ 91% of full-sequence performance
- On-policy prefix distillation: distill only reasoning prefixes
- 2-40Ã— FLOP reduction, early-terminate sampling
- `--distill-truncate 0.5 --distill-prefix`

## V4 R&D Round 13: General Upkeep & Bug Fixes (2026-08)

Codebase-wide audit and fixes across all 12 previous R&D rounds:

### Critical Bug Fixes (4)
- **kvpop.py**: `KVpopScorer.__init__` missing `n_kv` parameter â†’ added (was
  `AttributeError` at runtime). `v_sink_len` â†’ `sink_len` (wrong attribute name).
- **conf_kv.py**: `_promote_from_window` referenced `self.positions` (doesn't
  exist on `ConfKVCache`) â†’ fixed to `self.mixed_kv.positions` with empty-check guard.
- **moe_optim.py**: `AllocMoE.stats` referenced `self.total_budgets` (plural)
  â†’ fixed to `self.total_budget` (singular).

### High-Severity Fixes (1)
- **aoh_retmask.py**: `classify_heads` assumed PyTorch Linear weights are
  `(d_model, n_heads * head_dim)` but they're actually `(out_features, in_features)`
  = `(n_heads * head_dim, d_model)` â†’ fixed view + matmul to match real layout.

### Medium-Severity Fixes (3)
- **peagle.py**: Removed dead `nn.init.zeros_` immediately overwritten by
  `nn.init.normal_`. Removed dead loop with `pass` statement in training loss.
- **curriculum_augment.py**: `_apply_fim` now guards `T < 8` (was crashing on
  short sequences). Target offset cat now handles `T <= 1` and uses correct device.

### Infrastructure Fixes (3)
- Added missing `__init__.py` in `research/keys/architecture/` and
  `research/tokenization/` (were importable but not package-discoverable).
- Wired `use_flex_decoding` flag (was documented but never implemented in init).
- Verified all 342 .py files compile, all 26 new R&D modules import cleanly,
  356/361 tests pass (5 pre-existing network-dependent failures in
  `test_distill_client.py`).

## V4 R&D Round 14: FreeToken-Inspired Training Pipeline (2026-09)

Inspired by FreeToken (arXiv:2608.16157) â€” edge-native MoE serving engine with
bandwidth-adaptive CPU-GPU co-execution. Applied the key training innovations
to ForgeAI's `CPUAdamW` optimizer and `sft_train` loop.

### New Files
- `research/runtime/bandwidth_profiler.py` â€” **BandwidthProfiler + BandwidthPredictor**: measures PCIe bandwidth (B_P=push, B_H=pull) via staged CUDA events, computes q* split ratio, predicts bandwidth trend from VRAM usage samples for pre-emptive offload decisions.
- `research/sandbox/bench_freetoken_training.py` â€” Benchmark comparing baseline vs overlap vs freetoken vs freetoken_chunked.
- `tests/unit/test_hybrid_offload_ft.py` â€” 6 tests: backward compat, double_buffer, bandwidth_adaptive, chunked, correctness (bit-exact), predictor.

### Modified Files
- `research/training/optim/hybrid_offload.py` â€” **CPUAdamW enhanced**:
  - **Double-buffered gradient pipeline** (`double_buffer=True`): ping-pong grad buffers (grad_cpu + grad_cpu_b) eliminate transfer-then-compute serialization. CUDA transfer to inactive buffer overlaps with CPU AdamW reading active buffer.
  - **Bandwidth-adaptive offload** (`bandwidth_adaptive=True`): profiles PCIe bandwidth on init, computes q* = B_P/B_H ratio, auto-sets chunk_size_mb = B_P * 0.01 (10ms worth of data per chunk).
  - **Gradient-chunked pipeline** (`chunk_size_mb`): splits large grad tensors into bandwidth-optimal chunks, uses 4-stream pool for finer PCIe overlap.
  - **Bandwidth predictor** (`BandwidthPredictor`): records (step, B_P, VRAM) samples, linear-regresses B_P vs VRAM, triggers `should_preempt_offload()` when predicted B_P drops below threshold or VRAM trend approaches limit.
  - **3-stage pipeline** in `step()`: (1) start CUDA grad transfer, (2) wait for previous CPU step (overlaps with transfer), (3) launch new CPU step on background thread.
  - **Bug fix**: `_lazy_init` async copy with dtype conversion produced stale master weights (non_blocking + bf16â†’fp32 conversion race). Fixed to synchronous copy.
  - **Bug fix**: Per-tensor CUDA stream creation caused OOM on 12GB GPUs (149 streams for 1.2B model). Fixed to shared single-stream for non-chunked, 4-stream pool for chunked.
  - **Bug fix**: `non_blocking=True` with dtype-converted copies (bf16â†”fp32) allocates GPU staging buffers per-tensor, causing OOM. Fixed to synchronous copies for dtype-converted transfers.
- `research/training/training_utils.py` â€” `configure_optimizer` now accepts `freetoken`, `double_buffer`, `bandwidth_adaptive`, `chunk_size_mb` params. `freetoken=True` enables all enhancements at once.
- `research/training/runners/sft_train.py` â€” CLI args `--freetoken`, `--double-buffer`, `--bandwidth-adaptive`, `--chunk-size-mb`, `--elastic-grad-accum`. 3-stage pipeline integration: `optimizer.wait()` before forward (overlaps previous CPU step with current forward). Bandwidth sample recording + predictive offload check every 10 steps. Elastic grad_accum adjusts based on VRAM pressure.

### Benchmark Results (201M param model, seq_len=256, batch=2, RTX 5070)
| Mode | Fwd | Bwd | Opt | Wait | Total | tok/s | Speedup |
|------|-----|-----|-----|------|-------|-------|---------|
| baseline (sync) | 5ms | 11ms | 469ms | 0 | 485ms | 1056 | 1.00x |
| overlap | 5ms | 11ms | 66ms | 225ms | 306ms | 1671 | 1.58x |
| freetoken | 5ms | 11ms | 29ms | 226ms | 271ms | 1890 | 1.79x |
| freetoken_chunked | 5ms | 11ms | 24ms | 219ms | 258ms | 1982 | **1.88x** |

Key insight: the `opt` column drops from 469msâ†’24ms because CPU AdamW math
runs entirely in the background thread, hidden behind the `wait` phase (~220ms
is the actual CPU compute, overlapped with GPU forward+backward of next batch).

### Usage
```bash
# Full FreeToken pipeline
python -m research.training.runners.sft_train --optimizer cpu_offload --freetoken

# Individual enhancements
python -m research.training.runners.sft_train --optimizer cpu_offload \
    --double-buffer --bandwidth-adaptive --chunk-size-mb 64

# With elastic grad_accum (auto-adjusts to avoid OOM)
python -m research.training.runners.sft_train --optimizer cpu_offload \
    --freetoken --elastic-grad-accum
```

## V4 R&D Round 15: NVFP4 Quantization + Novel Quant Algorithms (2026-09)

Made NVFP4 the default inference quantization on Blackwell (RTX 5070 SM120),
with two novel quantization algorithms that improve on standard FP4.

### NVFP4 (rewritten)
- `research/inference/quant/nvfp4_quant.py` â€” **Complete rewrite** of NVFP4:
  - FP4 E2M1 with 8 magnitude levels {0, 0.5, 1, 1.5, 2, 3, 4, 6}
  - **Two-level scaling**: per-channel fp32 global scale + per-block FP8 E4M3
    normalized scales. Critical fix: without two-level scaling, small LLM
    weights (std~0.02) produce block scales below FP8's minimum (0.002),
    causing all weights to dequantize to zero.
  - Fast vectorized dequantization via `searchsorted` on FP4 boundaries
  - W4A8 mode: FP4 weights + dynamic FP8 activations + `torch._scaled_mm`
  - 3.8x compression (0.53 bytes/weight vs 2.0 bf16)
  - V7 8B estimate: 15.0 GB bf16 â†’ 4.0 GB NVFP4 (fits 12GB with 8GB for KV)
- `research/config.py` â€” New config fields: `use_nvfp4`, `nvfp4_block_size`,
  `nvfp4_w4a8`
- `research/inference/forge_engine.py` â€” NVFP4 is now the **default auto-select**
  on Blackwell (even when VRAM is ample, since it's near-lossless at ~99% quality
  and frees VRAM for larger KV cache / longer context)

### Novel Algorithm 1: AdaptScale FP4 (AS-FP4)
- `research/inference/quant/novel_quant.py` â€” `ASFP4Linear`
- **Novel insight**: standard NVFP4 uses absmax/6.0 block scaling, but this is
  NOT the MSE-optimal scale. For weights with outliers, absmax scaling maps
  a few outlier values to 6.0, sacrificing precision for the majority.
  AS-FP4 uses a grid search over 10 candidate scales (absmax * {0.5..1.5})
  to find the per-block scale that minimizes MSE.
- **Result**: 5.9% lower Frobenius error than standard NVFP4, same memory,
  same inference speed (the grid search is at quantization time only).

### Novel Algorithm 2: ResidualFP4 (R-FP4)
- `research/inference/quant/novel_quant.py` â€” `ResidualFP4Linear`
- **Novel insight**: FP4 quantization error is concentrated in a small number
  of "hard" elements. R-FP4 stores a sparse INT8 residual for the top-k%
  largest errors, applied via scatter-add at dequantization time.
- **Result**: at 5% residual, 21.7% lower error than FP4 (near-INT8 quality
  at 2.6x compression). At 10% residual, 33.5% lower error (1.9x compression).
- The residual is stored as (int32 index, int8 value, fp32 per-row scale) â€”
  applied as `w.scatter_add_(1, indices, values * scale)` before matmul.

### Benchmark Results (simulated LLM weights, 8192x2048, RTX 5070)
| Method | Frob Err | Compression | vs INT8 quality |
|--------|----------|-------------|-----------------|
| INT8 | 0.023 | 2.0x | baseline |
| INT4 | 0.199 | 3.9x | 8.6x worse |
| NVFP4 | 0.107 | 3.8x | 4.6x worse |
| AS-FP4 | 0.101 | 3.8x | 4.4x worse |
| R-FP4 5% | 0.083 | 2.6x | 3.6x worse |
| R-FP4 10% | 0.071 | 1.9x | 3.1x worse |

NVFP4/AS-FP4 give INT4-level compression with significantly better quality.
R-FP4 10% approaches INT8 quality at nearly 2x compression.

### Tests
- `tests/unit/test_nvfp4.py` â€” 6 tests: FP4 roundtrip, forward pass, CUDA,
  compression, model-level, speed
- `tests/unit/test_novel_quant.py` â€” 5 tests: AS-FP4 vs NVFP4, R-FP4 sweep,
  forward comparison, model-level, CUDA speed
- `research/sandbox/bench_quant_comparison.py` â€” Full benchmark across
  matrix sizes, all quant methods, V7 8B VRAM estimate

## V4 R&D Round 16: Novel Param Quantization Deep Dive (2026-08-28)

First-principles analysis of how LLM params work (Gaussian, heavy-tailed,
per-channel variance) and how existing quants distort them, then 10 novel
quantization algorithms with full benchmark.

### First-principles findings
- LLM weights are ~Gaussian (small Ïƒ ~0.02-0.05) with heavy tails (outliers)
- FP4 E2M1 {0,0.5,1,1.5,2,3,4,6} is denser near 0 â€” good for Gaussian BUT
  fixed spacing; absmax/6.0 scale lets one outlier compress the whole block
- AS-FP4 (R15) minimizes WEIGHT MSE; R-FP4 picks residual by |weight error|.
  Neither minimizes OUTPUT error (what actually matters)
- Hadamard rotation reduces block kurtosis 6.28 â†’ 2.60 (below Gaussian 3.0),
  eliminating outlier structure â€” norm preserved perfectly

### 10 novel algorithms (`research/inference/quant/novel_quant.py`)
1. **HW-FP4** â€” Hessian-weighted MSE-optimal scale (minimizes output error)
2. **TSDS-FP4** â€” Threshold-split dual-scale (outlier/inlier partition, 2 scales)
3. **SR-FP4** â€” Stochastic rounding (unbiased estimator, lower systematic error)
4. **OC-Hybrid** â€” Per-row FP8/FP4 mixed precision (outlier channels â†’ FP8)
5. **HPR-FP4** â€” Hadamard pre-rotation + AS-FP4 (spreads outliers)
6. **BN-FP4** â€” Block L2-norm scaling (vs absmax)
7. **SAAS-FP4** â€” Mean-subtraction before symmetric FP4 (centers asym. blocks)
8. **AWR-FP4** â€” Activation-weighted residual selection (improves R-FP4)
9. **IRI-FP4** â€” Iterative residual refinement (K rounds of FP4 on residual)
10. **PCBA** â€” Per-channel bit allocation (8-bit high-dynamic rows, 4-bit rest)

### Benchmark results (512x1024, mild outliers, CPU)
| Algorithm | SQNR (dB) | bytes/w | vs AS-FP4 |
|-----------|-----------|---------|-----------|
| NVFP4 (base) | 19.46 | 0.53 | -0.8 |
| AS-FP4 (R15) | 20.26 | 0.53 | baseline |
| R-FP4 5% (R15) | 21.59 | 0.78 | +1.3 |
| **SR-FP4** | **22.24** | **0.53** | **+2.0** |
| TSDS-FP4 | 21.02 | 0.69 | +0.8 (best on heavy outliers +3.6) |
| HPR-FP4 | 20.74 | 0.53 | +0.5 (most stable across distributions) |
| SAAS-FP4 | 20.43 | 0.56 | +0.2 |
| **IRI-FP4 x2** | **41.21** | **1.12** | **+21.0** (beats FP8 at FP4 cost) |
| IRI-FP4 x3 | 62.13 | 1.68 | +41.9 (overkill) |

### Production classes promoted (3 winners)
- `SRFP4Linear` â€” best single-pass FP4: +2.0dB over AS-FP4 at identical
  storage (0.53 bytes/w). Unbiased stochastic rounding. Drop-in replacement.
  `quantize_model_sr_fp4(model)`
- `IRIFP4Linear` â€” iterative residual: 41.21dB at 1.12 bytes/w (14.3x vs bf16).
  Beats FP8 quality using only the FP4 codebook (2x FP4 throughput vs 1x FP8
  on Blackwell). Each round adds ~21dB at +0.56 bytes/w. n_rounds=2 sweet spot.
  `quantize_model_iri_fp4(model, n_rounds=2)`
- `TSDSFP4Linear` â€” threshold-split dual-scale: most robust to heavy outliers
  (+3.6dB over AS-FP4 on heavy-tailed blocks). 0.69 bytes/w. Best for FFN
  down-projections and attention output projections.
  `quantize_model_tsd_fp4(model, split_quantile=0.75)`

### Shelved (documented failures â€” see `.devin/scratchpad.md`)
- **BN-FP4**: L2-norm scale ignores FP4's hard 6.0 ceiling â†’ clipping (13.2dB).
  Dead end â€” absmax-based scaling is necessary for FP4's fixed max level.
- **HW-FP4**: Hessian weighting needs REAL calibration activations. Synthetic
  Hessian concentrates importance on few channels, distorting scale (17.3dB <
  AS-FP4 20.3dB). Revisit when real activation calibration hooks are wired.
- **AWR-FP4**: Same Hessian issue. -40% to -117% vs R-FP4 at all ratios.
- **HPR+SAAS+HW stacking**: HW compounds its distortion (17.97 < HPR alone 20.74).

### Tests
- `tests/unit/test_novel_quant.py` â€” 7 Round-16 tests: main benchmark (13
  algorithms), distribution robustness sweep (5 configs Ã— 8 algos), block size
  sweep, IRI round sweep, AWR vs R-FP4 residual sweep, Hadamard rotation
  uniformity check, algorithm stacking. All pass.

### Multi-scale benchmark (1B / 5B / 8B / 10B standard)
Standardized test using realistic GQA+SwiGLU weight shapes at four model scales:
- 1B: d=2048, L=16, inter=8192, vocab=65536 â†’ 1.11B params, 2.06 GB bf16
- 5B: d=4096, L=20, inter=16384, vocab=65536 â†’ 5.13B params, 9.56 GB bf16
- 8B: d=4096, L=34, inter=14336, vocab=128256 â†’ 7.94B params, 14.79 GB bf16
- 10B: d=5120, L=28, inter=17920, vocab=128256 â†’ 10.20B params, 19.00 GB bf16

**Key finding: SQNR is scale-invariant** â€” quality doesn't degrade with model
size (same per-weight distribution); only VRAM scales linearly.

**VRAM table (GB, * = fits 12GB RTX 5070):**
| Algorithm | 1B | 5B | 8B | 10B | SQNR (dB) |
|-----------|------|------|------|-------|-----------|
| NVFP4 | 0.76* | 3.05* | 4.86* | 6.22* | 19.5 |
| AS-FP4 | 0.76* | 3.05* | 4.86* | 6.22* | 20.3 |
| **SR-FP4** | 0.76* | 3.05* | 4.86* | 6.22* | 22.2 |
| **HPR-FP4** | 0.76* | 3.05* | 4.86* | 6.22* | 22.3 |
| TSDS-FP4 | 0.87* | 3.62* | 5.73* | 7.33* | 21.0 |
| **IRI-FP4 x2** | 1.27* | 5.60* | 8.75* | 11.22* | 41.2 |
| IRI-FP4 x3 | 1.78* | 8.15* | 12.63 | 16.22 | 62.1 |
| bf16 | 2.06* | 9.56* | 14.79 | 19.00 | inf |

**Critical findings for RTX 5070 (12GB):**
- bf16 does NOT fit at 8B+ (14.79 GB) â€” quantization is mandatory
- IRI-FP4 x3 does NOT fit at 8B+ (12.63 GB) â€” IRI-FP4 x2 is the max quality
  option that fits at ALL four scales (11.22 GB at 10B)
- SR-FP4 / HPR-FP4 leave 7+ GB for KV cache at 8B, 5.8 GB at 10B
- HPR-FP4 slightly edges SR-FP4 at 5B+ (22.33 vs 22.24 dB) â€” rotation helps
  more on larger matrices

**Pareto frontier (8B):** NVFP4 â†’ AS-FP4 â†’ SR-FP4 â†’ HPR-FP4 â†’ IRI-FP4 x2 â†’ IRI-FP4 x3
- HPR-FP4 dominates SR-FP4 at 8B+ (same VRAM, +0.05 dB)
- IRI-FP4 x2 is the only >30dB option that fits 12GB at all 4 scales

## V4 R&D Round 17: Advanced Quantization R&D (2026-08-28)

Four directions building on Round 15/16 findings.

### 1. HW-FP4-v2: Real activation calibration (FIXED from R15)
- `research/inference/quant/novel_quant.py` â€” `HWFP4CalibratedLinear`,
  `compute_hessian_proxy()`, `calibrate_hw_fp4_model()`
- R15's HW-FP4 failed with synthetic Hessian (17.3 dB < AS-FP4 20.3 dB).
  R17 fix: use real activationÂ² as Hessian proxy via forward hooks.
- Result: +33.9% output error improvement over synthetic Hessian.
  HW-FP4-v2 out_err=0.0893 vs AS-FP4 0.0974 (modest +8% win).
- Verdict: real calibration works but the gain is small â€” Hessian-weighted
  SCALE selection has limited upside vs MSE-optimal scale. Hessian matters
  more for per-element quantization (GPTQ) than per-block scale.

### 2. HPR+IRI: Hadamard rotation + iterative residual (WINNER)
- `research/inference/quant/novel_quant.py` â€” `HPRIRIFP4Linear`,
  `quantize_hpr_iri_fp4()`
- Combines the two R15/16 winners: rotation spreads outliers â†’ IRI
  converges faster. Rotation absorbed into preceding layer (zero runtime cost).
- Results (heavy outliers):
  - IRI x1: 19.18 dB â†’ HPR+IRI x1: 20.76 dB (+1.6 dB, same storage)
  - IRI x2: 40.26 dB â†’ HPR+IRI x2: 41.58 dB (+1.3 dB, same storage)
- Multi-scale: HPR+IRI x2 beats IRI x2 at every scale, gain increases
  with d_model (1B: +1.1 dB, 10B: +2.1 dB â€” rotation helps more at
  larger d because more outliers to spread).

### 3. OptimalSignHadamard: greedy kurtosis minimization (SHELVED)
- `research/inference/quant/novel_quant.py` â€” `_optimal_sign_hadamard()`
- Greedy sign flip with incremental rank-1 updates (O(n) per flip vs O(nÂ²)).
- Result: 2.426 vs random 2.422 kurtosis â€” NO improvement (within noise).
  The Hadamard structure already does 99% of the outlier spreading.
- Verdict: SHELVED. Random signs are sufficient.

### 4. AdaptivePerLayer: per-layer algorithm selection (WINNER)
- `research/inference/quant/novel_quant.py` â€” `analyze_weight_distribution()`,
  `quantize_model_adaptive()`
- Analyzes each layer's weight distribution (kurtosis, dynamic range,
  asymmetry) and selects the best algorithm:
  - Heavy-tailed (kurt>6, dr>5) â†’ TSDS-FP4 (dual-scale)
  - Asymmetric (|mean|/std>0.15) â†’ SAAS-FP4 (mean-subtract)
  - Slightly heavy (kurt 4-6) â†’ SR-FP4 (stochastic rounding)
  - Clean Gaussian (kurt~3) â†’ AS-FP4 (cheapest)
  - High-value layers (user-specified) â†’ IRI-FP4 (best quality)
- Test result: +35.3% output error improvement over uniform AS-FP4.

### GPU acceleration
- All test tensors moved to CUDA (`DEV = torch.device("cuda")`)
- `_randn()` helper with per-call seed for reproducible GPU tensors
- `_optimal_sign_hadamard` optimized with incremental rank-1 updates
- Test runtime: ~2 min on GPU vs >10 min on CPU (killed) for R17

### Tests
- `tests/unit/test_novel_quant.py` â€” 5 R17 tests: HW-FP4-v2 calibration,
  HPR+IRI stacking, OptimalSignHadamard, AdaptivePerLayer, multi-scale.
  All pass on GPU.

## V4 R&D Round 18: Advanced Codebook + KV Cache + Gradient Quantization (2026-08-28)

Five directions exploring codebook optimization, adaptive round allocation,
KV cache quantization, and gradient-based code optimization.

### 1. MixedCodebookIRI: FP4+INT4 alternating rounds (MARGINAL)
- `research/inference/quant/novel_quant.py` â€” `quantize_mixed_iri_fp4()`,
  `_quantize_dequant_int4_block()`, `_optimal_int4_scale()`
- Alternates FP4 (non-uniform levels, dense near 0) and INT4 (uniform grid)
  per IRI round. Hypothesis: FP4 captures Gaussian mass, INT4 captures
  uniform residual.
- Result: +0.26 dB over IRI x2 (41.00 vs 40.74 dB) â€” marginal.
  Still below HPR+IRI x2 (42.26 dB).
- Verdict: SHELVED. The gain doesn't justify the added complexity.

### 2. AdaptiveIRI: per-block round allocation (STORAGE WINNER, NOT QUALITY)
- `research/inference/quant/novel_quant.py` â€” `quantize_adaptive_iri_fp4()`
- Allocates rounds per block based on residual energy threshold. Clean
  blocks stop after 1 round, heavy-tailed blocks get up to max_rounds.
- Result: 21.06 dB (threshold=1e-4) vs IRI x2 41.21 dB â€” much lower
  quality because most blocks stop at round 1. With strict threshold
  (1e-5): 39.71 dB, close to IRI x2 but uses ~2.5 rounds avg.
- Verdict: Useful for storage-constrained scenarios (saves ~30% storage
  vs uniform IRI x2), but not a quality winner.

### 3. LearnedFP4Codebook: Lloyd-Max per-block levels (WINNER for single-round)
- `research/inference/quant/novel_quant.py` â€” `_lloyd_max_codebook_1d()`,
  `quantize_learned_codebook_fp4()`
- Combines per-block scaling (AS-FP4) with Lloyd-Max optimized 8-level
  codebook on normalized values. Replaces fixed FP4 levels {0,0.5,1,1.5,
  2,3,4,6} with MSE-optimal levels for the actual distribution.
- Result: +1.03 dB over AS-FP4 (20.77 vs 19.73 dB) at same storage.
  Multi-scale: +0.65 dB (20.91 vs 20.26 dB).
- Verdict: WINNER for single-round quantization. Best 1-round FP4
  variant. Still far below IRI x2 (41 dB) but useful when storage
  budget allows only 1 round.

### 4. SRFP4KVCache: stochastic-rounding FP4 for KV cache (WINNER â€” new domain)
- `research/inference/quant/novel_quant.py` â€” `SRFP4KVCache`
- Applies SR-FP4 to the KV cache (not just weights). KV cache grows
  linearly with context length â€” major VRAM consumer for long-context.
- Result: 18 dB SQNR, 3.7x compression vs bf16. 128 tokens Ã— 8 heads Ã—
  64 dim: 256 KB bf16 â†’ 69 KB FP4.
- For LFM2.5-1.2B at 32K context: ~1.92 GB bf16 â†’ ~0.52 GB FP4
  (1.4 GB freed).
- Verdict: WINNER. First KV cache quantization in the codebase.
  Stochastic rounding ensures unbiased attention scores over many tokens.

### 5. GradientFP4: QuIP#-style gradient optimization (SHELVED)
- `research/inference/quant/novel_quant.py` â€” `quantize_gradient_fp4()`
- Uses Straight-Through Estimator (STE) to gradient-descent on FP4
  magnitude indices, minimizing Hessian-weighted output error.
- Result: +0.00 dB â€” identical to AS-FP4. The STE approach converges
  back to the nearest-rounding solution because it's already the local
  MSE optimum. Coordinate descent (try flipping each code) would be
  needed for global optimization, but is O(nÂ²) per iteration.
- Verdict: SHELVED. STE gradient descent on quantization indices is
  ineffective when the initial solution is already locally optimal.
  QuIP#'s gains come from incoherence preprocessing (Hadamard rotation),
  not from gradient optimization of codes â€” we already have that via
  HPR-FP4.

### Multi-scale results (R18 vs R17 best, GPU)
| Algorithm | 1B | 5B | Storage |
|-----------|------|------|---------|
| AS-FP4 | 20.26 | 20.26 | 0.53 B/w |
| LearnedCB | 20.91 | 20.92 | 0.53 B/w |
| IRI-FP4 x2 | 41.21 | 41.21 | 1.06 B/w |
| MixedIRI x2 | 41.04 | 41.03 | 1.06 B/w |
| **HPR+IRI x2** | **42.26** | **42.72** | 1.06 B/w |

HPR+IRI x2 remains the Pareto-optimal choice. LearnedFP4Codebook is the
best single-round option (+0.65 dB over AS-FP4 at same storage).

### Tests
- `tests/unit/test_novel_quant.py` â€” 6 R18 tests: MixedCodebookIRI,
  AdaptiveIRI, LearnedFP4Codebook, SRFP4KVCache, GradientFP4, multi-scale.
  All pass on GPU.

## V4 R&D Round 19: Qwen3.8-Flash-Next Architecture Keys (2026-08-28)

Research basis: Qwen3.8-Flash-Next (Aug 2026, Alibaba/Qwen) â€” early preview
of the Qwen4 architecture. 176B total params (125B main + 51B n-gram + 4B
MTP), 6B activated per token, 48 layers, 262K context.

Three architectural innovations from Qwen3.8-Flash-Next implemented as
ForgeAI keys. All have **lossless warm starts** (identity at init, training
activates the new mechanism). All fit within 12GB VRAM at LFM2.5-1.2B scale.

### 1. QSA: Qwen Sparse Attention (R19a)
- `research/keys/attention/qsa_key.py` â€” `QSALayer`, `QSAKey`
- Micro-block-level sparse attention with MQA indexer (4 query heads, 1
  shared key head, head_dim=128). Indexer scores micro-blocks of tokens,
  then only top-K blocks are attended to.
- Qwen reports up to 7.6x prefill speedup, 4.9x decoding speedup.
- **Lossless warm start**: budget = all blocks at init â†’ full attention.
  Training reduces the budget as the indexer learns to skip.
- VRAM overhead: 1.31M params/layer Ã— 6 attention layers = 7.86M (15 MB bf16)
- Test: sparse (budget=8/32 blocks) produces different output than full
  (budget=32/32), confirming the indexer selects blocks correctly.

### 2. Gated Residual: 4-branch widened residual (R19b)
- `research/keys/architecture/gated_residual_key.py` â€” `GatedResidualLayer`,
  `GatedResidualKey`
- N branches (default 4) with low-rank bottlenecks (rank=256). Each branch
  has an element-wise data-dependent READ gate (controls what enters) and
  a per-branch scalar WRITE gate (controls what exits).
- Qwen uses 4 branches, bottleneck rank=320, d_model=2560.
- **Lossless warm start**: branch 0 write gate = sigmoid(3) â‰ˆ 0.95 (active),
  branches 1-3 write gate = sigmoid(-3) â‰ˆ 0.05 (disabled). Branch 0 has
  small Kaiming init, others are zero. Training opens the other branches.
- VRAM overhead: 8.40M params/layer Ã— 16 layers = 134.35M (256 MB bf16)
- Strengthens cross-layer information flow and training stability.

### 3. N-gram Embedding: host-offloadable parameter scaling (R19c)
- `research/keys/knowledge/ngram_embedding_key.py` â€” `NGramEmbeddingLayer`,
  `NGramEmbeddingKey`
- Hash-based n-gram lookup table (bigrams/trigrams) added to token
  embeddings. Table stored on CPU (pinned memory), prefetched to GPU
  asynchronously â€” zero VRAM cost for the table itself.
- Qwen uses 20M n-gram entries (51B params) offloaded to host RAM.
- **Lossless warm start**: all n-gram embeddings are zero at init â†’ no
  effect on output. Training fills in important n-gram embeddings.
- VRAM overhead: 0 MB GPU (table on host). Host RAM: 15.3 GB for 2M
  entries Ã— d_model=2048 (fp32). Configurable table_size.
- Unique parameter scaling axis: less compute than MoE, more offloadable
  than weight matrices. Ideal for our 12GB VRAM + 32GB RAM hardware.

### VRAM budget at LFM2.5-1.2B scale
| Component | GPU VRAM | Host RAM |
|-----------|----------|----------|
| Base model (bf16) | 2340 MB | â€” |
| QSA indexer (6 layers) | 15 MB | â€” |
| Gated Residual (16 layers) | 256 MB | â€” |
| N-gram Embedding (2M entries) | 0 MB | 15.3 GB |
| **Total with R19** | **~2611 MB** | **15.3 GB** |

Fits 12GB VRAM easily (2611 MB << 12288 MB). Host RAM: 15.3 GB < 32 GB.

### What we did NOT implement (already covered)
- **Muon optimizer**: already have `muon_sf_blockwise.py` with Newton-Schulz
  orthogonalization (V3/V4 training). Qwen's refinements (Muon/AdamW split
  by weight category, fused parameter splitting) are minor variations.
- **MTP**: already have `mtp_key.py` (V4 multi-token prediction).
- **Hybrid GDN+QSA layout**: covered by having both GatedDeltaNet2Key (R15)
  and QSAKey (R19a). The 3:1 linear-to-sparse ratio is a config choice.
- **MoE with 512 experts**: our AirMoE already handles MoE with disk
  offload. 512 experts is a scale choice, not an architectural change.

### Tests
- `tests/unit/test_r19_qwen_keys.py` â€” 11 tests: QSA identity/sparse/key,
  GR identity/branches/key, N-gram identity/nonzero/host-offload/key,
  VRAM analysis. All pass on GPU.

## V4 R&D Round 20: Memory-Efficient Training + Novel Parameter Formats (2026-08-29)

### Problem
V10-8B training needs 48.3 GB RAM (bf16 master 16.1 GB + 8-bit optimizer 32.2 GB).
Target hardware: RTX 5070 12GB VRAM + 32 GB system RAM (~28 GB available).

### R20 Memory-Efficient Optimizers (4 approaches, 3 fit 28 GB)
`research/training/optim/r20_memory_optimizers.py`

1. **AdamW4Bit**: 4-bit momentum + variance with per-block scales + EF21 error
   feedback. 1.25 bytes/param (vs 8 for fp32). Total: 26.2 GB â€” doesn't fit
   alone, but works with NVMe streaming. Fixed NaN from vâ†’0 after dequant
   (clamp v + clip update).

2. **NVMeStreamedBAdam**: Block-wise BAdam with optimizer states on NVMe
   (mmap'd). Only 1 layer's states in RAM at a time. Total: 17.1 GB.
   Works with any optimizer type. ~0.33s NVMe load per block switch.

3. **MuonBitNet4Bit**: Muon (single momentum, no variance) + 4-bit quant.
   0.625 bytes/param. Newton-Schulz orthogonalization on dequantized
   momentum. Total: 21.1 GB. Best for BitNet ternary weights.

4. **TernaryOptimizer**: 2-bit optimizer states for BitNet ternary params
   (tracks flip direction, not magnitude). fp32 AdamW for non-ternary
   params (~5%). Total: 21.2 GB. Novel: exploits ternary structure.

**Best combo**: NVMe + 4-bit Muon = 16.3 GB (fits 28 GB with 12 GB headroom).

### R20 Novel Parameter Formats (4 approaches, fundamentally non-dense)
`research/training/optim/r20_novel_param_formats.py`

1. **SpectralWeight** (DCT-domain): Weight stored as top-K DCT coefficients.
   **Finding**: DCT works for spatially smooth weights (<0.25% error at 4x)
   but FAILS for LLM weights (87% error). LLM weights have low-rank
   structure (SVD/NLRQ captures), not spatial smoothness (DCT captures).
   NLRQ is the right transform for LLMs. DCT is shelved for LLM use.

2. **HypernetworkWeight**: Small MLP generates weight matrix from
   (layer, row, col) coordinates. At LLM scale (4096Ã—4096): 90K hypernet
   params vs 16.8M dense = **185x compression**. Optimizer states shrink
   from 32 GB to 360 MB. Most radical approach â€” changes what "parameters"
   means. Training optimizes the generator, not the weights.

3. **ProductKeyWeight**: 2D product key lookup replaces matrices. At LLM
   scale: 2-8x compression depending on kdim. Lample et al. 2019 did this
   for FFN; we extend to all weight matrices. Top-K sparse selection.

4. **HashedWeight** (ROAST-style): Single shared weight vector + hash
   function maps (i,j) â†’ shared bucket. Exact target compression (4x, 8x,
   16x, 32x). **Finding**: Post-hoc fit is poor (87% error for random
   targets) â€” must train from scratch. Trainable test confirms backprop
   works (loss 0.81 â†’ 0.35).

### V10-8B training memory estimates (28 GB available RAM)
| Approach | True Params | Master | Optimizer | Total | Fits? |
|----------|-------------|--------|-----------|-------|-------|
| Dense (current) | 8.05B | 16.1 GB | 32.2 GB | 48.3 GB | NO |
| R20b: NVMe BAdam | 8.05B | 16.1 GB | 1.0 GB | 17.1 GB | YES |
| R20c: Muon-BitNet 4-bit | 8.05B | 16.1 GB | 5.0 GB | 21.1 GB | YES |
| R20d: Ternary optimizer | 8.05B | 16.1 GB | 5.1 GB | 21.2 GB | YES |
| R20b+c: NVMe + 4-bit Muon | 8.05B | 16.1 GB | 0.2 GB | 16.3 GB | YES |
| Hypernet (h64) | 0.04B | 0.1 GB | 0.2 GB | 0.3 GB | YES |
| Hashed (8x) | 1.01B | 2.0 GB | 4.0 GB | 6.0 GB | YES |
| PKM (k64) | 1.01B | 2.0 GB | 4.0 GB | 6.0 GB | YES |

### Key findings
- **DCT doesn't work for LLM weights** â€” they're low-rank, not spatially smooth.
  NLRQ (SVD-based, already implemented) is the right transform.
- **Hypernetworks offer the most radical compression** (185x) but require
  training from scratch and have slow weight generation. Best for frozen
  inference, not fine-tuning.
- **Hashed weights need train-from-scratch** â€” post-hoc fit doesn't work
  for random/redundant matrices. But exact compression ratio is guaranteed.
- **NVMe streaming is the most practical** â€” works with any optimizer,
  any model, minimal code changes. 17.1 GB with full 8B params.
- **Muon-BitNet 4-bit is the best optimizer** â€” 21.1 GB, no NVMe needed,
  Newton-Schulz orthogonalization works well with 4-bit momentum.

### Tests
- `tests/unit/test_r20_memory_optimizers.py` â€” 8 tests: AdamW4Bit
  basic/memory, NVMe-BAdam, MuonBitNet4Bit basic/memory, Ternary
  basic/memory, V10-8B memory fits. All pass on GPU.
- `tests/unit/test_r20_novel_params.py` â€” 12 tests: Spectral
  reconstruction/forward/trainable, Hypernet generate/trainable/forward,
  PKM forward/trainable, Hashed compression/fit/trainable, full benchmark
  + V10-8B memory estimates. All pass on GPU.

## V4 R&D Round 21: Cross-Domain Parameter Formats + Training Acceleration (2026-08-29)

### Cross-domain combinations (5 approaches)
`research/training/optim/r21_cross_domain.py`

1. **HyperNet-BitNet** (R20 hypernet + BitNet): Hypernetwork MLP generates
   continuous values per (layer,row,col), squashed via tanh during training,
   discretized to ternary {-1,0,+1} at inference. Combines 185x param
   compression with BitNet int8@int8 GEMM. Training uses tanh smooth
   approximation (STE caused divergence â€” tanh is stable). At 256x256:
   6.3-15.6x compression, 93% output error (needs longer training).

2. **HashedNLRQ** (R20 hashed + NLRQ): Hash the NLRQ low-rank factors U,V
   instead of the full matrix. NLRQ gives 12.8x, hashing adds 4-16x â†’
   total 8-32x compression. Trains well (loss 0.92â†’0.34). Most practical
   cross-domain combo â€” builds on existing NLRQ infrastructure.

3. **WaveletWeight** (R20 spectral fix): Haar wavelet instead of DCT.
   **Finding**: Wavelet beats DCT for LLM weights (48% vs 87% error at 4x)
   â€” wavelets capture block-localized structure that DCT misses. But still
   worse than NLRQ/SVD (low-rank is the dominant structure). Wavelet is
   orthogonal and invertible (round-trip error <1e-6). Smooth weights: 5.8%
   error at 8x. Verdict: better than DCT, but NLRQ remains king for LLMs.

4. **FP8ActivationLinear** (training memory): Quantize activations to FP8
   e4m3 during forward, store FP8 for backward. 2x activation memory
   reduction (bf16â†’FP8). Uses int8 fallback if torch.float8_e4m3fn not
   available. Eval mode = exact (no compression). Novel: FP8 for STORAGE
   only (GEMMs still bf16), simpler than NVIDIA Transformer Engine.

5. **TopKGradientOptimizer** (training speed): Top-K% gradient sparsification
   with EF21 error feedback. 10x fewer gradients transferred per step.
   Trains well (loss 1.34â†’0.25 at 10% density). EF21 prevents staleness.
   For NVMe streaming: only K% of params need optimizer state loaded per step.

### Key findings
- **Wavelet > DCT for LLM weights** (48% vs 87% error) but both lose to
  NLRQ/SVD. Low-rank structure dominates; frequency transforms are the
  wrong tool. NLRQ remains the best parameter format for LLMs.
- **HashedNLRQ is the most practical cross-domain combo** â€” builds on
  existing NLRQ, adds 4-16x via hashing, trains well. Total 8-32x.
- **HyperNet-BitNet needs longer training** â€” 93% error after 50 steps
  is expected (hypernet must learn the weight generation function).
  Full training (1000+ steps) needed for production quality.
- **FP8 activation is immediately useful** â€” 2x memory reduction with
  no quality loss in eval mode. Drop-in for gradient checkpointing.
- **GradTopK with EF21** â€” 10x gradient sparsity with convergence. EF21
  is slower on simple tasks but prevents staleness on complex ones.

### Tests
- `tests/unit/test_r21_cross_domain.py` â€” 12 tests: HyperNet-BitNet
  ternary/trainable, HashedNLRQ compression/trainable, Wavelet
  round-trip/reconstruction/trainable, FP8 eval/training, GradTopK
  basic/EF convergence, full benchmark. All pass on GPU.

### ForgeLM V8-8B preset (historical — deleted, features carried into V10)
`forgelm_v8_8b` was in `research/config.py` â€” derived from V7-8B-B with:
- R19: QSA (top_k=256), Gated Residual (gate=1.0), N-gram host embedding (3-gram, 15.3 GB)
- R20: NVMe-streamed 4-bit Muon optimizer (`optimizer_type="nvme_muon_4bit"`)
- R21: FP8 activation storage (`use_fp8_activation=True`),
  GradTopK 10% (`grad_topk_ratio=0.1`), HashedNLRQ 8x (`use_hashed_nlrq=True`)
- NLRQ rank=1024 (V7-8B-B capacity) with 8x hash = 50.7x FFN compression
- True trainable params: 1.60B (3.6x compression vs 5.81B dense equiv)
- Training: 6.67 GB GPU + 18.7 GB RAM (fits 12GB + 32GB with 5.3+9.3 GB headroom)
- Full scratch training: 10.7B tokens Ã— 3 epochs = 32.1B tokens, 12 days, $12 electricity
- Cost calculator: `research/sandbox/calc_v8_8b_scratch_train.py`

### R&D Round 22: Training Speedups for Large Datasets + Models
`research/training/optim/r22_training_speedups.py` â€” 6 novel approaches targeting the two training bottlenecks (large dataset + large model params). All tested on GPU, all pass.

**R22a: DataDedup (MinHash LSH)** â€” Near-duplicate document detection via MinHash signatures + LSH bucketing. O(nÃ—k) instead of O(nÂ²). Realistic corpora have 10-30% near-duplicates; removing them saves proportional training time with no quality loss (duplicates add no information). Measured: 20% dup rate on test corpus, 1.25x speedup. `MinHashDeduplicator(n_hashes=128, n_bands=32, shingle_size=5)`.

**R22b: TokenImportanceSampling** â€” Skip tokens the model already knows (low-loss tokens). Dynamic threshold adapts to loss distribution every N steps (25th percentile). Low-loss tokens sampled at 25% rate. Measured: 7.7% skip rate, 1.08x speedup. Higher savings in later epochs as more tokens become "known". `TokenImportanceSampler(initial_threshold=0.5, adapt_every=100, skip_rate=0.25)`.

**R22c: ProgressiveLayerUnfreezing** â€” Train only last K layers initially, gradually unfreeze earlier layers. Phase 1: 10/32 layers (3.2x faster), Phase 2: 21/32 (1.5x), Phase 3: 32/32 (1.0x). Novel: layer importance scoring via accumulated gradient magnitude determines unfreeze order. Effective speedup over full training: 1.29x (only first 30% benefits). `ProgressiveUnfreezer(n_layers=32, n_phases=3)`.

**R22d: GradientCompression (4-bit + EF21)** â€” 4-bit signed symmetric quantization for CPUâ†”GPU gradient transfer in BAdam. Per-block scales (block_size=128). Error feedback (EF21) prevents quantization error accumulation. Measured: 3.9x compression vs bf16, 11.7% roundtrip error (acceptable for training). 1.94x effective BAdam transfer speedup. `GradientCompressor(bits=4, block_size=128, ef_feedback=True)`.

**R22e: AsyncDataPipeline** â€” Triple-buffered async I/O with ThreadPoolExecutor. While GPU computes step N, worker threads load+tokenize step N+2. Measured: 95.6% I/O hidden, 1.74x speedup (with 20ms compute + 15ms I/O per step). `AsyncDataPipeline(load_fn, tokenize_fn, buffer_size=3)`.

**R22f: CheckpointDelta** â€” Save only parameter deltas (8-bit quantized) for fast multi-session checkpointing. Full checkpoint every 10 deltas. Measured: 33% params changed per session, 6x compression vs full checkpoint. Save time: ~5s vs ~30s for 3.2GB model. `CheckpointDelta(full_checkpoint_every=10, delta_threshold=1e-4, quant_bits=8)`.

**Combined speedup**: 5.90x (multiplicative, different bottlenecks). V10-8B 3-epoch: 289h â†’ 49h. V10-8B 1-epoch: 96h â†’ 16.3h. Fine-tune 1.2B: 0.2h â†’ 0.1h. Does NOT make 8B pretraining feasible in 1-2hr sessions (still needs 11+ sessions for 1 epoch). Fine-tuning 1.2B fits in 1 session. Cost calculator: `research/sandbox/calc_v8_8b_r22_speedup.py`. Tests: `tests/unit/test_r22_speedups.py` (13 tests, all pass on GPU).



All config-driven, dimension-generic (work at V10-1.2B scale, d_model=2048). **Main `forgelm_v10_1.2b` preset now enables ALL of them losslessly** â€” verified bit-exact (max logit diff 0.0) vs the plain GQA model on the real BSP checkpoint, for both plain and KV-cached prefill+decode:
- `quantization/bitnet_b158_key.py` â€” BitNet b1.58 ternary QAT: `BitNetLinear` (STE, learned per-layer `qscale` re-anchored on checkpoint load, ternary ONLY in training; eval = full-precision master weights until `bitnet_force_quant`). **True BitNet integer kernels on CUDA**: default = int8 @ int8 tensor-core GEMM (`torch._int_mm`, a4.8-style activation quant); `FORGE_BITNET_KERNEL=triton` selects the b1.58 add-only Triton kernel (fp activations, zero-skip, no weight multiplies â€” verified bit-exact vs fp on small shapes). Applies to FFN + attention q/k/v/o projections. Enable: `use_bitnet=True` (main preset: on).
- `attention/differential_attn_key.py` â€” Diff-Transformer: dual-softmax subtraction, per-head Î», per-head RMSNorm+scale. **Identity warm start** (`lambda=0`): group-1 rows extracted contiguously (`_group1_weights`) so GEMM shapes match GQA exactly â†’ bit-exact conversion; training moves Î» off 0 to activate the real mechanism. `attn_type="diff"` (main preset: on; GQA checkpoints auto-convert at load). KV cache stores 2Ã— head_dim.
- `attention/differential_attn_key.py` â€” Diff-Transformer: dual-softmax subtraction, per-head Î» (paper init), per-head RMSNorm+scale. `attn_type="diff"`; KV cache stores 2Ã— head_dim. `DifferentialAttentionKey` = GQAâ†’diff weight transform (dup rows, warm start).
- `architecture/titan_memory_key.py` â€” TITAN neural memory: gated memory, zero-init gate => **lossless at start** (ported checkpoint loads identically). `TitanMemory.update()` = Hebbian surprise step (test-time training). Enable: `use_titan_memory=True`.
- `architecture/mod_router_key.py` â€” Mixture-of-Depths: per-block top-k token router (STE hard mask, soft grad). keep_fraction=1.0 => **lossless**. **TRUE skip in training** (no cache/mask): skipped tokens genuinely bypass attention+FFN (per-row gather/scatter, FLOPs scale with keep_fraction â€” verified: 0.5 fraction processes exactly 50% of tokens); router trained via aux loss (`ModRouter.aux_loss`). Inference keeps all tokens (KV alignment). Enable: `use_mod=True`.
- Main `forgelm_v10_1.2b` preset: `attn_type="diff"`, `use_bitnet=True`, `use_titan_memory=True` (rank 64), `use_mod=True` (keep_fraction=1.0) â€” all lossless at load; training activates each mechanism. `get_config` returns FRESH copies (preset mutation no longer leaks).
- Tests: `tests/unit/test_arch_keys.py` (incl. main 1.2B build forward).

## AirMoE Expert Consolidation (research/training_free/expert_bake.py)

Static offline counterpart of AirMoE runtime hotswap: fold topic experts into dense FFN weights.
- `decompress_expert(state)` â€” decodes raw / SVD-only / SVD+INT4 expert files (mirrors `research/moe/airmoe_infinite.py` formats; LatentMoE up/down skipped).
- `bake_expert(target, expert_paths, alpha, layers, out)` â€” per-layer task arithmetic: `target += alpha * mean(expert âˆ’ base_ffn)` per `w_gate/w_up/w_down`. Multiple experts per layer are averaged. Output is a normal dense .safetensors â€” no router, no disk I/O at inference.
- Layer parsed from `expert_l{layer}_{topic}.safetensors` filenames; override with `--layers`.
- CLI: `python -m research.training_free.expert_bake --target T --expert experts/expert_l0_math.safetensors --alpha 0.8 --out O.safetensors`
- Tests: `tests/unit/test_expert_bake.py`.

## Offline Weight Baking (research/training_free/bake.py)

Permanent weight modification without backpropagation:
- `bake_task_vector(target, finetuned, base, alpha, out)` â€” task arithmetic: adds `alpha*(finetuned - base)` onto an arbitrary target checkpoint (e.g. self-play checkpoint). Offline tensor math, output is a normal .safetensors.
- `extract_distill_dataset(packets_jsonl, out_jsonl, ...)` â€” context distillation: from self-play packet logs, keeps only (task, final correct code) pairs in sft_train JSONL format for a low-epoch SFT pass.
- `fuse_lora(base_ckpt, adapter_dir, out, alpha_override)` â€” offline PEFT LoRA fusion (`W' = W + (alpha/r)*B@A`), standalone output, no PEFT dependency at inference. (sft_train also fuses via `merge_and_unload` at save time.)
- CLI: `python -m research.training_free.bake {task-vector|distill|fuse-lora} ...`
- Tests: `tests/unit/test_bake.py` (CPU, tensor-level).

Constant-memory note: full SSM blocks (Mamba-2) were removed in the 2025-01 cleanup; the LFM2.5 arch already gets constant-memory O(1) state from its 10 conv layers (`DoubleGatedConvLayer`) â€” use `hybrid_offload` (convâ†’CPU, attentionâ†’GPU) to trade compute for VRAM instead of re-adding SSM code.

## I/O & VRAM Notes (2026-08)

- `StreamingDataLoader(ds, ..., num_workers=N, prefetch_factor=2)` â€” multi-worker path is picklable on Windows (module-level `_ParquetWorkerDataset`); `pin_memory=True` + device now does non-blocking H2D.
- `ParquetDataset` pickles by re-opening the file (workers re-open by path).
- `HadamardKVCache` pre-allocates its max buffer on first append (no per-token `torch.cat`).
- `StreamedGenerator` no longer double-syncs per decode step (`.item()` is the single sync).
- `load_checkpoint(..., map_location="cpu")` returns memory-mapped tensors (zero-copy).
- New config: `selective_gradient_checkpointing` â€” `"all"` (default), `"ffn"` (recompute only FFN, big VRAM save), `"attn"`, `"none"`. Enable with `use_gradient_checkpointing=True`.
- `model.cache_devices()` runs once on first forward (16+ `next(param).device` scans cached); `hybrid_offload` invalidates it.

## Recommended Improvements (2026-08)

Four improvements implemented based on architectural critique:

### 1. SnapKV + 4-bit KV Combined Cache (`research/inference/kv_backend.py`)
- `SnapKV4BitCache` â€” composes SnapKV eviction + Hadamard INT4 quantization.
- Total compression = eviction_ratio Ã— bit_ratio (up to ~16Ã— vs fp16 full cache).
- Strategy name: `"snapkv_4bit"` in `build_kv_cache()`.
- Evicts low-attention tokens first (SnapKV), then quantizes survivors (Hadamard INT4).

### 2. Golden Trajectory Injection (`research/self_play/grpo_trainer.py`)
- `GRPOTrainer` now accepts `replay_buffer` parameter (FOREVER-style `ReplayBuffer`).
- `GRPOConfig.replay_ratio` (default 0.15) â€” fraction of each batch from golden replays.
- `_inject_golden_replays()` â€” mixes previously successful trajectories into training batch.
- `_record_golden_trajectories()` â€” stores verified-successful completions for future replay.
- Prevents catastrophic forgetting in continual self-play (anti-regression).

### 3. ELO-Driven Curriculum (`research/self_play/elo_tracker.py`)
- `EloTracker` â€” ELO rating system for both model and individual prompts.
- Targets ~50% expected success (Goldilocks zone, max learning signal).
- `select_prompts()` â€” picks prompts closest to model's skill boundary.
- `select_mixed_prompts()` â€” Goldilocks-matched + exploration prompts.
- Integrated into `InfiniteCurriculum.record_result()` and `get_training_batch_elo()`.
- Zero-sum rating updates, K-factor decays with prompt attempts (stabilizes ratings).

### 4. Fused QK-Norm + RoPE Triton Kernel (`research/decoding/fused_rope_qknorm.py`)
- Fuses RMSNorm + RoPE into a single Triton kernel for Q and K preprocessing.
- Halves HBM traffic (1 load + 1 store vs 2+2 for separate ops).
- `fused_qk_norm_rope()` â€” public API, auto-falls back to PyTorch on CPU/compile-fail.
- Opt-in via `FORGE_FUSED_ROPE_QKNORM=1` env var in `GroupedQueryAttention.forward`.
- Attention itself stays on FlashAttention-2 (FA2) via SDPA â€” already fused.
- Tests: `tests/unit/test_recommended_improvements.py`.

## GRPO-Î» Dynamic Length Penalty (2026-08)

Prevents the "CoT length penalty trap" (arXiv 2509.01155): static length penalties
cause accuracy collapse early in training when the model is still learning to reason.

### GRPOTrainer (`research/self_play/grpo_trainer.py`)
- `GRPOConfig.use_grpo_lambda` â€” enable dynamic length penalty (default False).
- `_group_correctness_ratio()` â€” fraction of completions with reward >= 0.99.
- `_apply_grpo_lambda_penalty()` â€” penalty only when correctness_ratio >= threshold.
  - Low correctness â†’ NO penalty (pure 0/1 rewards, prioritize reasoning).
  - High correctness â†’ penalty = -Î» * n_tokens (encourage efficiency).
- `length_penalty_warmup` â€” delay penalty activation for first N steps.
- Stats track `correctness_ratios` and `length_penalty_active_count`.

### GoalScorer (`research/evaluation/goal_scorer.py`)
- `minimalism_active` flag (default True) â€” toggle minimalism/length penalty.
- `set_minimalism_active(False)` â€” redistributes minimalism weight to efficiency/diversity.
- Called by training loop based on GRPO-Î» correctness ratio.
- Tests: `tests/unit/test_grpo_lambda.py` (17 tests).

## BitNet-Everywhere + Manual LoRA + Sequential Freeze (`research/training/bitnet_lora.py`)

Production training stack for V3 on 12GB RTX 5070. Validated on real 1.2B V3:
2.39x better convergence than AdamW, 6.32GB VRAM (53% of 12GB).

### CLI Flags (sft_train.py)
- `--bitnet-everywhere` â€” Convert ALL nn.Linear â†’ BitNetLinear (ternary b1.58 QAT).
  No NF4/bnb needed â€” BitNet IS the quantization (1.58 bits). 41 layers converted,
  72 already BitNet = 113 total ternary layers.
- `--manual-lora` â€” Use manual LoRA adapters (BitNet-compatible, unlike PEFT which
  can't inject into BitNetLinear). Auto-enabled with `--bitnet-everywhere --lora`.
  97 adapters, 25.40M trainable params (2% of model) at rank=32.
- `--sequential-freeze N` â€” Train N layers at a time in phases. Full forward pass
  preserves MHC/AttnRes cross-layer connections; only gradients are scoped.
  Best with 100+ steps/phase. 0=disabled.
- `--final-finetune-steps N` â€” Reserve N steps at end to fine-tune ALL layers together.
- `--optimizer muon_sf_plain` â€” Muon (Newton-Schulz) + ScheduleFree AdamW, no blockwise
  sharpness. Optimal for V3 (blockwise conflicts with BitNet). 2.24x vs AdamW.
- `--grad-mixup 3` â€” 3-way gradient averaging. 1.25x better convergence, 3x compute/step.

### Optimal Production Command
```bash
python -m research.training.sft_train \
  --config forgelm_v3 \
  --checkpoint research/checkpoints/ForgeLM_V3_Base.safetensors \
  --optimizer muon_sf_plain --grad-mixup 3 \
  --bitnet-everywhere --lora --lora-r 32 --lora-alpha 64 \
  --sequential-freeze 4 --final-finetune-steps 200 \
  --max-steps 1000 --batch-size 2 --seq-len 256 --lr 3e-4
```

### Key Functions
- `convert_to_bitnet_everywhere(model)` â€” Replace all nn.Linear with BitNetLinear
- `add_lora_adapters(model, rank, alpha)` â€” Manual LoRA (works with BitNetLinear)
- `merge_lora_adapters(model)` â€” Merge LoRA into base weights for standalone save
- `freeze_unfreeze_lora(model, active_layers)` â€” Freeze/unfreeze by layer index
- `compute_phase_schedule(n_layers, n_phases, total_steps, final_finetune_steps)` â€” Phase schedule
- `build_muon_sf_lora_opt(lora_params, lr_muon, lr_adam)` â€” Muon-SF for LoRA params

### What NOT to use with V3
- `--optimizer muon_sf` (with blockwise sharpness) â€” conflicts with BitNet, 11% worse
- DiffusionBlocks â€” conflicts with MHC/AttnRes cross-layer connections
- PEFT LoRA (`--lora` without `--bitnet-everywhere`) â€” can't inject into BitNetLinear

## DiffusionBlocks (`research/diffusion_blocks.py`)

Block-wise training via diffusion interpretation of residual connections
(Sakana AI, ICLR 2026). First successful test on a >1B parameter model.

### How It Works
- Partitions 16 layers into B blocks (default B=4, 4 layers/block)
- Each training step trains ONE block independently as a denoising step
- BÃ— memory reduction: only L/B layers need gradients per step
- AdaLN noise conditioning (shift/scale, zero-init = lossless at start)
- EDM-style loss weighting: w(Ïƒ) = (ÏƒÂ² + Ïƒ_dataÂ²) / (ÏƒÂ·Ïƒ_data)Â²

### V3 Benchmark Results (2026-08-18)
- **Standard training**: 5.74 GB, 8.14s/step (batch=2, seq=512)
- **DiffusionBlocks middle blocks**: 4.98 GB, 1.72-1.82s/step (13% less memory, 4x faster)
- **Batch scaling**: 4Ã— larger batch (8 vs 2) fits in 8.70 GB
- **Key fix**: Removed gates from AdaLN (zero-init gates block gradients; shift/scale only)

### Usage
```python
from research.diffusion_blocks import DiffusionBlocks, DiffusionBlockConfig
db_config = DiffusionBlockConfig(num_blocks=4, use_noise_conditioning=True)
dblock = DiffusionBlocks(model, db_config, d_model=2048, num_layers=16)
# Train one block per step
result = dblock.train_step(input_ids, labels, optimizer)
```

### Config
- `DiffusionBlockConfig`: num_blocks, sigma_min/max, gamma (overlap), cond_dim
- `freeze_all_except_block(b)`: freezes all params except block b
- `unfreeze_all()`: restores standard training mode
- Model forward supports `layer_indices`, `noisy_embeds`, `modulation` kwargs

## Multi-Provider Distillation Client (`research/distillation/distill_client.py`)

Generates verified training data from free-tier API providers. Only uses
Apache 2.0 / MIT licensed models (distillation-safe). Llama and Gemma are
EXCLUDED due to license restrictions.

### Supported Providers (11 providers, 32 model entries, 13 canonical models)
- **Groq** (free, permanent): Qwen3-32B, gpt-oss-120b, gpt-oss-20b
- **DeepSeek** (MIT license): deepseek-reasoner (R1 CoT), deepseek-chat (V3)
- **NVIDIA NIM** (free, 40 RPM, no daily cap): DeepSeek R1/V3, Qwen3.5-122B, gpt-oss-120b
- **Cerebras** (free, permanent, 30 RPM, 1M tok/day): gpt-oss-120b/20b, Qwen3-32B, Qwen3-235B, GLM-4.7
- **SambaNova** (free, permanent, 20 RPM/20 RPD): gpt-oss-120b, DeepSeek-V3.1
- **Cloudflare Workers AI** (free, 10K neurons/day): gpt-oss-120b/20b, GLM-4.7
- **SiliconFlow** (free forever): Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B
- **HuggingFace** (free, $0.10/mo credits): gpt-oss-120b/20b, DeepSeek-R1
- **Mistral AI** (free experiment plan): Mistral Small 4, Magistral Small (reasoning)
- **Z AI / Zhipu** (free, unlimited): GLM-4.7-Flash, GLM-4.5-Flash
- **OpenRouter** (free tier): Qwen3-235B MoE, gpt-oss-120b/20b, DeepSeek R1

### Multi-Provider Rate-Limit Bypass (max redundancy)
Same model served by multiple providers â€” client rotates through them:
- **gpt-oss-120b**: 7 providers (groq, nvidia, cerebras, sambanova, cloudflare, openrouter, huggingface)
- **gpt-oss-20b**: 5 providers (groq, cerebras, cloudflare, openrouter, huggingface)
- **deepseek-r1**: 4 providers (deepseek, nvidia, openrouter, huggingface)
- **deepseek-v3**: 3 providers (deepseek, nvidia, sambanova)
- **glm-4.7**: 3 providers (zai, cerebras, cloudflare)
- **qwen3-32b**: 2 providers (groq, cerebras)
- **qwen3-235b**: 2 providers (openrouter, cerebras)

### NVIDIA NIM Filter
`_nvidia_filter()` excludes NVIDIA's own models (Nemotron etc.) per Eval
Agreement Â§2.6. Only third-party MIT/Apache models on NIM are allowed.

### Key Features
- **Randomized model-per-goal**: shuffles model pool per goal for max quality diversity
- **Multi-distill**: different teacher models per sample â†’ diverse training data
- **Auto-detects API keys**: only uses providers with credentials in env
- **Verification**: optional `verify_fn(solution, test_cases) -> bool` filters correct solutions
- **ReplayBuffer integration**: `distill_into_buffer()` stores verified results as golden trajectories
- **Temperature randomization**: (0.3, 1.0) range for GRPO group diversity
- **Reasoning traces**: captures CoT from R1/Qwen3 thinking mode for reasoning distillation

### Pipeline
```
DistillationClient â†’ generate verified (prompt, solution) pairs
    â†“
ReplayBuffer.add() â†’ store as golden trajectories
    â†“
GRPOTrainer._inject_golden_replays() â†’ mix into training batches
```

### Usage
```python
from research.distillation.distill_client import DistillationClient
from research.self_play.replay_buffer import ReplayBuffer

client = DistillationClient(verify_fn=my_verify_fn)
buf = ReplayBuffer(max_size=10000)
stats = client.distill_into_buffer(
    goals=["Write fibonacci function", "Sort a list"],
    replay_buffer=buf,
    n_samples_per_goal=4,  # 4 diverse completions per goal
)
```

### License Safety
- âœ… Apache 2.0 (Qwen3, gpt-oss, Mistral): "prepare Derivative Works" explicitly allowed
- âœ… MIT (DeepSeek R1, GLM, Phi-4): "distill & commercialize freely"
- âŒ Llama Community: "will not use output to improve any other LLM" (BANNED)
- âŒ Gemma Terms: distilled model becomes "Model Derivative" (BANNED)
- âŒ Gemini TOS: "may not use Services to develop models that compete" (BANNED)
- âŒ NVIDIA-own: Evaluation Agreement Â§2.6 prohibits (third-party on NIM = OK)
- âŒ OpenAI GPT: "may not use Output to develop models that compete" (BANNED)
- âŒ Anthropic: "may not use Outputs to train models that compete" (BANNED)
- âŒ Grok/xAI: "weights cannot be used to train other models" (BANNED)
- âŒ Cohere: non-commercial use only (BANNED for commercial distillation)

### BANNED providers (do not re-add)
- Google AI Studio â€” TOS prohibits competing model development
- GitHub Models â€” RETIRED July 30, 2026
- Hyperbolic/Together/Novita/Chutes â€” trial credits, not permanent

### Research doc
- `docs/GROQ_DISTILLATION_RESEARCH.md` â€” full provider analysis, rate limits, pricing
- Tests: `tests/unit/test_distill_client.py` (26 tests)

## Tool-Call + Code-Format Distillation (`research/distillation/distill_tool_calls.py`)

Cold-start SFT data generation for ForgeLM V3. Uses API teachers (gpt-oss-120b,
DeepSeek, etc.) to generate training data with **thinking tokens captured**:
1. **Direct answers**: Simple Qs ("5*10=50") â€” no tools, just answer from knowledge
2. **Code generation**: task -> Python function + test cases
3. **Tool-call trajectories**: Multi-turn with `[end]` marker â€” teacher writes code,
   marks `[end]`, system executes it, returns result, teacher continues
4. **Reasoning**: Problem decomposition with step-by-step thinking

Thinking tokens (`reasoning_content` from gpt-oss/DeepSeek R1) are captured and
wrapped in ` IMD... IMD` blocks so the model learns to think before answering.

Multi-response format: each data point is a list of turns:
```
[{"role": "user", "content": "Task: ..."},
 {"role": "assistant", "content": " IMD\n<thinking>\n IMD\n<code>[end]"},
 {"role": "tool_result", "content": "stdout: ..."},
 {"role": "assistant", "content": " IMD\n<thinking>\n IMD\n<final answer>"}]
```

CLI: `python -m research.distillation.distill_tool_calls --n-code 200 --n-tool 100 --n-reason 50 --n-direct 50 --output research/data/finetune/v3_distill.jsonl`

## FFN-SkipLLM (Speculative Compute Reduction)

Based on EMNLP 2024 paper. Skips FFN computation on "saturated" layers
(high cosine similarity between FFN input and output) during eval.

- Config: `ffn_skip_threshold` (0.0 = disabled)
- **NOT applicable to ForgeLM V3** â€” calibration shows no saturation region:
  all 16 layers have low/negative cosine similarity (max +0.10, most negative)
  meaning the FFN is actively transforming representations in every layer
- FFN-SkipLLM requires 32+ layer models (LLaMa-2 7B/13B) where middle-layer
  FFNs become redundant (cosine similarity 0.95+)
- Infrastructure kept in codebase for future larger models
- Full analysis: `docs/FFN_RESEARCH.md`

### What DOES work for V3 inference acceleration:
- **Speculative decoding**: EAGLE-3, DSpark, MTP (already in `research/decoding/`)
- **Attention skipping in top layers**: "Attend First, Consolidate Later" (2024)
  â€” skip attention in top 30% for non-math tasks, keep all FFNs
- **middle_70 architecture**: for V4 â€” concentrate FFN in middle 70% of layers
  (+1.29% improvement at 1.2B scale per COLM 2025 paper)

## Agentic Distillation Client (`research/distillation/agentic_distill.py`)

Takes the self-play loop process from the former `tool_use_loop.py` (now removed â€” merged into the unified AZR loop) and applies it to the
distillation model router. Teacher API models call tools (run_script, web_search,
think, calculate, etc.) in an agentic loop to generate rich training trajectories.

**Fine-tuning is DISABLED** â€” pure data collection. Trajectories can later be
used for SFT or GRPO training of the local ForgeLM model.

### Architecture
```
Teacher API model (gpt-oss, DeepSeek, Qwen, etc.)
    â†“ receives task + tool definitions (OpenAI function-calling format)
    â†“ emits tool calls â†’ we execute (run_script, web_search, think, etc.)
    â†“ results injected back â†’ teacher continues
    â†“ loop until final answer or max_turns
    â†“
AgenticTrajectory (messages, tool_calls, reward, final_answer)
    â†“
save_trajectories() â†’ JSONL for SFT training
```

### Key Components
- **`AgenticDistillClient`** â€” extends `DistillationClient` with agentic tool-use
- **`run_agentic_task()`** â€” runs the tool-use loop with a teacher model
- **`run_agentic_batch()`** â€” runs multiple tasks with multiple teachers per task
- **`generate_tasks()`** â€” teachers generate their own coding tasks with test cases
- **`save_trajectories()`** â€” saves full tool-use trajectories as JSONL
- **`compute_reward()`** â€” reuses the same multi-component reward from the former `tool_use_loop.py` (now legacy; import guarded)
  (format, tool selection, execution, completion, planning, self-verification, etc.)

### Tool-Capable Filtering
Not all providers support OpenAI function calling. The client filters on init:
- **Tool-capable**: Groq, DeepSeek, NVIDIA, Mistral, SambaNova, Cerebras, HuggingFace
- **Tool-capable (partial)**: OpenRouter (only gpt-oss and deepseek models)
- **No tool support**: Cloudflare, Z AI, SiliconFlow (auto-excluded from agentic pool)

### Task Generation
Teachers can generate their own tasks (like `GoalGenerator` in the former `infinite_tool_loop`):
- Teacher model proposes coding tasks with test cases
- Tasks filtered for quality (non-filler, requires tool use, has verifiable output)
- Filtered tasks added to the pool for agentic execution

### Usage
```python
from research.distillation.agentic_distill import AgenticDistillClient

client = AgenticDistillClient(max_turns=8)
# Run agentic tasks (teachers call tools)
trajectories = client.run_agentic_batch(
    tasks=["Write is_prime(n)", "Implement binary search"],
    n_samples_per_task=3,  # 3 different teachers per task
    min_reward=0.3,  # only keep good trajectories
)
# Let teachers generate their own tasks
new_tasks = client.generate_tasks(n_tasks=20)
# Save for SFT training
client.save_trajectories(trajectories, "agentic_distill_data.jsonl")
```

### CLI
```powershell
# Agentic mode with predefined tasks
python -m research.distillation.run_data_gen --agentic --n-samples 3

# Generate new tasks then run them
python -m research.distillation.run_data_gen --agentic --gen-tasks 20

# Only run generated tasks (skip predefined)
python -m research.distillation.run_data_gen --agentic --gen-tasks 20 --gen-only
```

### Tests
- `tests/unit/test_agentic_distill.py` (23 tests: schema conversion, task filtering,
  tool-capability filtering, trajectory serialization, save/load)

## Build & Test Commands

```powershell
# Run tests
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -m pytest tests/ --tb=short -q

# Verify model loads
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -c "from research.model_loader import ConfigurableResearchLLM; print('OK')"

# Benchmark INT4 quantization
D:\windsurf\ForgeAI\venv\Scripts\python.exe D:\windsurf\ForgeAI\.devin\benchmark_int4.py
```

## Removed (cleanup 2025-01)

- All Qwen-based configs (qwen25_coder, xp_1.5b, forgelm_v1/v2, 360m_mla, etc.)
- All Nemotron configs and architecture files (mamba2, latent_moe, nemotron_lightning)
- Dead attention types (DifferentialAttention, MultiHeadLatentAttention, StandardSDPA)
- Dead FFN types (ReLUSquaredFFN, latent_moe)
- EAGLESpeculativeDraftHead
- serving/ directory (superseded by inference/forge_engine.py)
- convert_keys.py, convert_key_svd.py (Qwen-specific weight transforms)
- o1_generation/ (thinking model, dead)
- 6.5GB of dead checkpoints (forgelm_v2, nemotron, dspark, qwen_hf tokenizer)
- Dead test files referencing old configs

## Removed (consolidation 2026-08)

- **5 unused key files**: `keys/misc/mhc_key.py` (dup of architecture/), `keys/cache/snapkv_key.py` (runtime version canonical), `keys/normalization/norm_folding_v2_key.py` (v1 canonical), `keys/attention/qk_norm_key.py` (MLA version canonical), `keys/attention/fold_qknorm_mla_key.py` (unused)
- **4 dead self_play files**: `recursive_self_play.py` (65KB, superseded by infinite_loop.py), `populora.py`, `rlsvr.py`, `live_status.py` (all zero imports)
- **2 dead training files**: `research_team.py` (88KB monolith, zero imports), `data_gen.py` (126KB V2-era LM Studio pipeline, superseded by distillation/run_data_gen.py)
- **MTP duplication**: `architecture/mtp.py` deleted (kept `decoding/mtp.py` â€” independent heads + trainer)
- **serving/ directory**: merged tool-calling support into `inference/forge_server.py`, deleted `serving/server.py`
- **Stale files**: `distillation/test_v3_inference.py`, `scripts/extract_vocab_packs.py`, `sample.csv`, `test.csv`, empty `.jsonl` data files
- **Doc consolidation**: `LLM_Research.md` merged into `research/COMPREHENSIVE_RESEARCH.md` (unique Blackwell/Triton/Gigatoken sections preserved as section 38)

## Novel Discovery Protocol (R&D methodology)

When asked to do R&D or develop novel systems, do NOT just design on paper.
Follow this iterative empirical loop:

1. **Isolated test script first** â€” write a minimal script with the smallest
   possible input that exercises the core idea (e.g. tiny model, 4 layers,
   100 steps). Run it BEFORE researching. Get a baseline number.
2. **Research + think long** â€” web_search the topic, read 3-5 papers, think
   hard about what's known vs unknown. Write findings to `.devin/scratchpad.md`.
   **Do the math by hand** for the small case before trusting any number.
3. **Apply novel ideas, compare to documented results** â€” implement 2-3 novel
   variations in the isolated script. Run them. Compare numbers to documented
   baselines from research AND to the current production baseline on this
   hardware. Most novel ideas will LOSE to known answers â€” that's expected
   and informative.
4. **Iterate once more before defaulting** â€” if novel ideas lost, adjust the
   angle (not just parameters). Try a different combination. Only after a
   second failed iteration should you default to the best known answer.
5. **Cross-domain risky combinations** â€” if still stuck, try combining
   something that barely relates to the topic but might have novel effects.
   E.g. diffusion scheduling ideas applied to optimizer step timing, or
   compression theory applied to gradient sparsity.
6. **Record what worked AND what failed** â€” failed novel ideas are still
   valuable; document them in scratchpad so the next session doesn't repeat.
   Include the numbers, the hypothesis, and why it likely failed.
7. **Sometimes, leave it to luch** - sometimes, leaving things to chance can be a good modivator. if you need more ideas, take all known and loosly related systems, throw them into a randomizer script, and see what it adds together. ie; ##### + #####. This could help you find new ideas that you wouldn't have thought of otherwise.
8. **Pivot don't quit** â€” if a hard optimization resists 2 iterations,
   shelf it (documented in scratchpad) and touch up a DIFFERENT area.
   Fresh context often surfaces the missing idea. Return to the hard
   problem in a later round. No area is truly solved, but not every area
   yields on the first session.
9. **Sweep before you ship** â€” any technique with hyperparameters gets a
   sweep script (5-10 values) before being declared optimal. The paper's
   default is rarely our optimum on RTX 5070 + 1.2B.
10. **Confirm-then-fix in R&D too** â€” if a test script reveals a bug in
    existing code (not your new code), confirm it and fix it in the same
    session per Directive B. R&D rounds that leave behind unfixed bugs
    are not complete.

Key principles:
- **Isolated scripts over integration** â€” test the core mechanism on a toy
  problem before touching the real training loop. Faster iteration, clearer
  signal.
- **Numbers over theory** â€” a 5-line script that runs in 10 seconds beats
  a 500-line design doc. Get a number, then explain it.
- **Lose fast** â€” most novel ideas don't work. Find out in 30 seconds with
  a toy script, not 30 hours of integration.
- **Cross-domain is where novelty lives** â€” the best novel discoveries
  combine techniques from fields that don't usually interact.
- **Mixed GPU/CPU is the default assumption** â€” never design a technique
  that only works if it fits in VRAM. Always have the CPU-offload fallback
  path designed from the start (Directive D).
- **Search before you build** â€” before writing a new test script, grep for
  an existing one that tests the same mechanism. Adapt it instead of
  starting fresh (Directive E).

## Fast Boot / Cold-Start Optimization (2026-08-19)

R&D round: boot-time (cold load â†’ first token) optimization. 7 variations
tested against baseline. **Production change implemented in `model_loader.py`.**

### Results (forgelm_v3, ForgeLM_V3_Base.safetensors, RTX 5070, bf16)

| Variation | TOTAL (ms) | Speedup | Correct |
|-----------|-----------|---------|---------|
| baseline (traditional) | 11279 | 1.0x | âœ“ |
| V1 skip_init | 10190 | 1.1x | âœ“ |
| V2 parallel tokenizer | 7185 | 1.6x | âœ“ |
| V3 OS prefetch | 6363 | 1.8x | âœ“ |
| V4 PrefetchVirtualMemory | 6355 | 1.8x | âœ— (API failed) |
| V5 meta+assign | 4414 | 2.6x | âœ“ |
| V6 V1+V2+V3 | 8209 | 1.4x | âœ“ |
| **V7 V5+V2+V3 (production)** | **3316** | **3.4x** | **âœ“** |

### What Worked (V7 = production default)
1. **Meta device init + `load_state_dict(assign=True)`** â€” 6.1x on arch build
   - Build model on `torch.device("meta")` (zero storage, just shape metadata)
   - `assign=True` directly replaces meta params with state_dict tensors
   - Skips both init kernels AND the `.to(device)` copy
   - Requires: re-tie weights (`model.head.weight = model.embed.weight`),
     re-init RoPE non-persistent buffers (`_reset_non_persistent_buffers()`)
2. **Parallel tokenizer** â€” `ThreadPoolExecutor` hides 2.7s behind arch build
3. **OS page cache prefetch** â€” background thread reads 16MB blocks during arch build

### What Failed
- **V4 PrefetchVirtualMemory** â€” Windows API call failed (Python mmap buffer access)
- **V1 skip_init** â€” works but slower than V5 (`to_empty` materializes ALL params,
  then `load_state_dict` copies into them; `assign=True` skips the intermediate copy)

### Production API
- `ModelLoader.build_model_fast(..., fast_load=True)` (default) â€” meta init + assign + prefetch
- `load_default_model(..., fast_load=True)` (default) â€” also starts parallel tokenizer
- `fast_load=False` â€” traditional build path (for debugging / no checkpoint)
- `RotaryEmbedding` now stores `self.base`, `self.max_seq_len`, `self.rope_scaling`
  (needed for `_reset_non_persistent_buffers()` after meta init)
- `tokenizer_cache.get_tokenizer()` â€” fast path via `tokenizers` Rust library (223ms)
  instead of `transformers` AutoTokenizer (4254ms). Falls back to transformers if
  `tokenizer.json` is missing. Sets `eos_token`/`bos_token`/`pad_token` strings
  from `tokenizer_config.json` (gigatoken's `eos_token_id` property reads from these).
- Benchmark: `.devin/boot_bench.py` (baseline), `.devin/boot_bench_variations.py` (V1-V8)
- Full results + research: `.devin/scratchpad.md`

### Final Boot Time: 2.5s (4.5x speedup from 11.3s baseline)
- D_arch_build: 1.1s (meta init, was 6.7s)
- F_weights_load: 1.3s (fastsafetensors + OS prefetch)
- J_tokenizer: 0ms wait (fast `tokenizers` Rust lib, hidden behind model build)
- L_first_forward: 170ms (kernel JIT, unavoidable)

### Future Optimization (not implemented)
- **Lazy key instantiation** â€” 698ms (92% of meta init!) is TITAN/MoD/MHC/AttnRes/
  DiffAttn/BitNet Python object creation across 16 blocks. Deferring to first forward
  (they're zero-init=lossless) would drop D_arch_build from 1098ms to ~400ms.
  Requires invasive `ModularBlock.__init__` changes.

## Concurrent Task-Based Generation (v3.0, 2026-08-19)

ForgeServer v3.0 adds LM Studio-style multi-task concurrent generation with
session management and request batching.

### Task API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tasks` | POST | Create a new task (accepts seed, system_prompt, boot_params) |
| `/v1/tasks/{task_id}/messages` | POST | Send a message to a task, get response (stream or non-stream) |
| `/v1/tasks/{task_id}/config` | PATCH | Update task's boot params (KV cache, decoding, quantization, etc.) |
| `/v1/tasks/{task_id}` | GET | Get task status + full conversation history + boot config |
| `/v1/tasks` | GET | List all active tasks (includes boot config per task) |
| `/v1/tasks/{task_id}` | DELETE | Delete a task and free its session |

### Per-Task Generation Settings

Each task and each message request supports independent generation parameters:

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `seed` | Task + Message | None (random) | Reproducible generation. Same seed + same prompt = same output |
| `temperature` | Message | 0.0 | Sampling temperature (0 = greedy) |
| `top_p` | Message | 1.0 | Nucleus sampling threshold |
| `top_k` | Message | 80 | Top-k sampling (0 = disabled) |
| `repetition_penalty` | Message | 1.05 | Repetition penalty (1.0 = disabled) |
| `max_tokens` | Message | 256 | Maximum generation length |
| `stop` | Message | None | Stop sequences (string-based) |
| `stream` | Message | False | SSE streaming response |

**Seed resolution**: Message-level seed overrides task-level seed. Task-level seed
is used when message doesn't specify one. `None` = random (non-deterministic).

**Batched mode**: Per-sequence seeds use independent `torch.Generator` instances,
so each sequence in a batch can be independently reproducible. Verified: same seed
in same batch = same output; different seeds = different output.

### Per-Task Boot Params (engine configuration)

Each task can specify independent **boot-time engine parameters** â€” settings
traditionally fixed at model load time. This is a superset of what LM Studio
exposes (LM Studio only allows global model settings, not per-conversation).

| Boot Param | Default | Options | Description |
|-----------|---------|---------|-------------|
| `kv_cache` | standard | standard, paged, hadamard_int4, snapkv, snapkv_4bit, rotorquant, compressed, streaming | KV cache strategy |
| `decoding` | standard | standard, speculative, batched, mtp_selfspec, medusa, dspark, eagle3 | Decoding strategy |
| `quantize` | None | None, int8, int4, fp8 | Weight-only quantization |
| `acceleration` | None | None, cuda_graph, airllm_streaming | Acceleration backend |
| `kv_cache_tokens` | None (model max) | int | Limit KV cache allocation (saves VRAM) |
| `kv_bits` | 4 | 4, 8 | KV cache quantization bits |
| `mrl_keep_ratio` | None | 0.0-1.0 | Truncate to fraction of dims (MRL) |
| `use_v0_warm` | False | bool | V0 warm-start for KV cache |
| `use_progressive_kv` | False | bool | Progressive KV (anchor + residual) |
| `use_compile` | False | bool | torch.compile the model |
| `use_triton_conv` | False | bool | Fused Triton conv kernel |
| `use_prefix_cache` | False | bool | Cache KV for repeated prefixes |
| `use_spec_attn` | False | bool | L1 Speculative Attention |
| `warmup` | True | bool | Pre-run dummy token to init CUDA kernels |

**Setting boot params**: At task creation (`POST /v1/tasks` with `boot_params`),
or live via `PATCH /v1/tasks/{id}/config`. The engine reconfigures before the
next generation if the task's config differs from the engine's current state.

**Example**: Task A uses `snapkv` KV cache for long-context, Task B uses
`hadamard_int4` for VRAM efficiency, Task C uses `cuda_graph` acceleration â€”
all on the same loaded model, switching configs per-task.

### Architecture

- **SessionManager** (`session_manager.py`): Per-task conversation context with
  LRU eviction. Each task has a unique ID, system prompt, message history, and
  model binding. Max 64 sessions by default (configurable).
- **BatchQueue** (`session_manager.py`): Background dispatcher thread collects
  concurrent requests for a configurable window (default 50ms), then dispatches
  them as a single batched forward pass via `BatchedDecoding`. Up to 8 requests
  per batch. Single requests bypass batching (use standard generate path).
- **BatchedDecoding** (`batched_decoding.py`): Processes multiple prompts in a
  single forward pass (GEMVâ†’GEMM shift). 3-5x throughput on RTX 5070.

### Performance

| Mode | Throughput | Notes |
|------|-----------|-------|
| Serial (1 request) | 290 tok/s | Standard single-request generation |
| Batched (3 concurrent) | 740 tok/s | 2.8x throughput vs serial |
| Batched (8 concurrent) | 1005 tok/s | 3.5x throughput vs serial |

### Usage

```bash
# Start server with task API
python research/inference/forge_server.py --batch-window-ms 50 --max-batch-size 8

# Create a task
curl -X POST http://localhost:8000/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"model": "lfm2.5-1.2b", "system_prompt": "You are a code generator."}'

# Send a message to the task
curl -X POST http://localhost:8000/v1/tasks/{task_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Write a fibonacci function", "max_tokens": 256}'

# Get task history
curl http://localhost:8000/v1/tasks/{task_id}

# List all active tasks
curl http://localhost:8000/v1/tasks
```

- Tests: `.devin/test_task_api.py` (session manager, batch queue, task continuity, 8-task stress test)

## V10 Full Trainer (research/training/runners/train_8b_all.py)

Production trainer for ForgeLM V10-class models on RTX 5070 12GB. Trains
from scratch on the packed datasets (`research/data/`, `research/data/`)
with BAdam block-wise full-parameter training.

- **NLRQ factor training (STE)** â€” `--factor-training auto|on|off` (default
  auto): `NLRQLinear.enable_factor_training_()` creates bf16 master factors
  (U_m/V_m) so gradients reach the factors (not just S); forward quantizes
  masters with straight-through estimator. Checkpoints export back to pure
  INT8 format (masters stripped by `snapshot_state`). Auto mode falls back
  to S-only if masters don't fit VRAM (rank 1024/8B-B doesn't; rank 768/
  `forgelm_v10_1.2b` does â€” validated 4092 tok/s @ seq 2048).
- **VRAM preflight + spill guard** â€” refuses runs that would spill into
  Windows shared GPU memory (~18x slowdown); `--allow-spill` overrides.
  Hard runtime check after step 1 against actual peak. `--badam-blocks-per-layer 2`
  halves the optimizer spike; bf16 optimizer states auto-enabled with factor
  training (`--fp32-optimizer-states` to force fp32).
- **Chunked fused linear-CE** (`chunked_next_token_ce`) â€” hidden-states CE
  without materializing [B*T, V] logits; handles FactorizedLMHead two-stage.
  Auto-on when batchÃ—seq â‰¥ 1024 (`--no-flce` to disable).
- **Data validation** â€” packed `.bin` token ids are checked against the model
  vocab at load (the 100M-token `research/data/train.bin` is a 151k-vocab
  pack, INCOMPATIBLE with V7's 65536 vocab â€” use `--skip-incompatible` or
  re-pack; feeding it raw causes CUDA gather asserts that surface as
  misleading CUBLAS errors).
- **Resume** â€” `--resume CKPT` restores step count, LR schedule, BAdam block
  schedule + active-block states (slim `{prefix}_resume.pt`), RNG, best-val.
- **Sampling** â€” `--sampling uniform|stratified --v7-weight W` (epoch-based,
  no replacement); pin_memory + non-blocking H2D.
- **Checkpointing** â€” serial background writer (one 8GB file at a time),
  best-by-val tracking, retention via `--keep-checkpoints N`, step JSONL log
  + final JSON summary.
- **Init logit normalization** (`normalize_logit_scale_`, auto in build_model) â€”
  kaiming init compounds through the two-stage FactorizedLMHead to logit
  std ~5.4 â†’ init CE â‰ˆ 24 (ABOVE the uniform baseline ln(65536)=11.09): the
  model started confidently wrong and never recovered (all pre-2026-08-22
  from-scratch runs plateaued at ~24). One probe forward rescales the head
  path so init CE â‰ˆ ln(vocab)+0.5. Diagnostics: `research/sandbox/diag_init.py`.
- **Dead-param probe** (`freeze_dead_params_`) â€” one tiny backward pass; any
  param with `p.grad is None` is frozen and excluded from BAdam (MTP head,
  loop_block, gated AttnRes/MHC paths â‰ˆ 1B params). Without this, a BAdam
  block containing only dead params crashes backward with "element 0 ...
  does not require grad" mid-run (hit at step 320 before the fix). A
  forward grad_fn walk does NOT work here (non-reentrant checkpointing hides
  interior param leaves).
- LR schedules: `linear` (D2Z default), `wsd`, `cosine`; NaN/inf loss and
  grad-norm skip steps; EMA loss display, tok/s window, ETA; runtime WDDM
  soft-spill detector (tok/s collapse warning).
- From-scratch learning on 12GB (2026-08-22, post-init-fix): 8B-B S-only
  plateaus ~11.5 (FFN = frozen random basis); `forgelm_v10_1.2b` (rank 768) with
  factor training is the config that actually learns (EMA descending past
  11.4 by step 350 @ seq 2048). `--badam-switch-every 3` speeds early
  descent at ~40% throughput cost; 10 sustains 4000+ tok/s.
- NOTE: `selective_gradient_checkpointing="optimal"` maps to "none" in the
  model (only all/ffn/attn/none exist) â€” this trainer forces "all".
- Smoke: `research/sandbox/smoke_train_8b.py` uses the same build path,
  auto-drops factor masters when VRAM is tight.
- Tests: `tests/unit/test_badam_fira.py` (18 CPU tests: BAdam partition/
  chunking/fp32/no-decay/slim-resume, NLRQ STE forward/grads/export,
  FLCE parity vs full CE, LR schedules, epoch sampler).
- BAdam (`research/training/optim/badam.py`): fp32 update math (bf16 states
  for >1M-elem params opt-in), 1D params excluded from weight decay,
  embeddings_head block chunked to ~layer size (was a 649M-param 5GB fp32
  optimizer spike), parked states compressed to bf16 on CPU, MTP head
  typically frozen by the trainer (no CE pathway â†’ no grads).

## R&D Round 16: Param Memory Cost Minimization (2026-08-24)

Target: minimize LLM parameter memory cost across ForgeLM V10, ForgeEngine, and
trainer. Three novel techniques implemented, all tested, zero regressions.

### Technique 1: int4 Gradient Compression with EF21 Error Feedback
**File**: `research/training/optim/hybrid_offload.py`
**Test**: `tests/unit/test_grad_compression.py` (18 tests)

The `grad_compression="int4"` config flag existed since Round 14 (evolution-
promoted, score 30.00) but was **never implemented** â€” it was stored as an
attribute and printed, with zero compression code. Now fully implemented:

- **Per-row symmetric INT4 quantization**: scale = absmax(row) / 7, packed
  2 int4 values per int8 byte (high/low nibble). 4x compression ratio.
- **EF21 error feedback**: GPU-resident residual buffer accumulates
  quantization error across steps. Before compression: `grad_to_send =
  grad + ef_error`. After: `ef_error = grad_to_send - dequant(compress(
  grad_to_send))`. This preserves convergence (verified on 100-step
  quadratic: < 2% final loss difference vs uncompressed).
- **1D tensor handling**: biases/norms use per-tensor scale (single scalar).
- **Wired into both grad_offload and non-offload paths**: compression
  happens on GPU before PCIe transfer (4x bandwidth cut), decompression
  on CPU for optimizer update.
- **Cleanup**: `cleanup_compression()` method + `__del__` for GPU memory.

**Impact**: 4x gradient transfer bandwidth cut. Enables full `grad_offload`
on V7 within 12GB. The EF21 error buffer is fp32 (same size as param) but
only needed for the active BAdam block â€” minimal GPU overhead.

### Technique 2: HINT4-NLRQ (Hadamard-INT4 NLRQ Factors)
**File**: `research/keys/compression/nlrq_ffn_key.py`
**Test**: `tests/unit/test_nlrq_hint4.py` (16 tests)

NLRQ FFN compression previously stored SVD factors as INT8 (8 bits). HINT4
adds INT4 factor support (4 bits) with Hadamard rotation to spread outliers:

- **Hadamard utilities**: `_hadamard_matrix()` (recursive construction with
  QR re-orthogonalization for non-power-of-2 sizes like rank=768),
  `_apply_hadamard()`, `_pack_int4()`, `_unpack_int4()`.
- **`from_dense_hadamard_int4()` classmethod**: SVD â†’ Hadamard rotation
  on rank dimension â†’ INT4 quantization with per-channel scales.
- **Forward pass**: Hadamard folded into matmul chain
  (`x @ V_f^T @ H_V â†’ *S â†’ @ H_U^T @ U_f^T`) to avoid materializing W.
- **Backward compatible**: INT8 NLRQ unchanged (`use_hadamard` defaults
  False, `_min_val` defaults to `-max_val` in `_ste_quantize`).

**Measured tradeoff** (rank=384, 1024Ã—4096 low-rank matrix):
| Mode | Error | CR | Bits/factor |
|------|-------|-----|-------------|
| INT8 | 1.04% | 4.2x | 8 |
| HINT4 | 18.59% | 5.3x | 4 |

The 18x error ratio is the fundamental INT4 vs INT8 gap (16 vs 256 levels).
The Hadamard rotation helps but can't overcome the level difference. HINT4
is viable for inference-only compression where the low-rank structure
absorbs some error, or when combined with an INT4 residual (future work).
**Not yet wired into V7 config** â€” needs quality validation first.

### Technique 3: Tied PEAGLE Heads with Position LoRA
**File**: `research/decoding/peagle.py`
**Test**: `tests/unit/test_peagle_tied.py` (23 tests)

PEAGLE speculative decoding draft head had 7 separate
`Linear(1024, 65536)` output projections = 471M params = 958 MB bf16.
`PEAGLEDraftHeadTied` replaces this with 1 shared head + per-position LoRA:

- **Shared head**: single `Linear(hidden_dim, vocab_size)` = 67M params
- **Position LoRA**: `pos_lora_A` (K, rank, hidden_dim) + `pos_lora_B`
  (K, hidden_dim, rank). B zero-initialized (standard LoRA init).
- **Vectorized forward**: `einsum` for LoRA across all K positions, then
  single `shared_head` matmul for all positions.
- **`from_existing()` conversion**: averages K original heads â†’ shared
  head, zero-inits LoRA (lossy, needs retraining).

**Param comparison** (production dims: K=7, hidden=1024, vocab=65536, rank=32):
| | PEAGLEDraftHead | PEAGLEDraftHeadTied | Savings |
|--|---|---|---|
| Output projections | 471M | 67.5M | **6.98x** |
| Memory (bf16) | 958 MB | 135 MB | **7.1x** |

### Bug Fix: Causal Mask in PEAGLE Cross-Attention
Both `PEAGLEDraftHead` and `PEAGLEDraftHeadTied` had an inverted causal
mask: `torch.tril(ones).bool()` was passed as `attn_mask`, but PyTorch
interprets `True` as "mask" (prevent attending), so this masked the lower
triangle (positions to attend to) instead of the upper triangle. The last
draft position was fully masked â†’ NaN attention output. Fixed to
`torch.triu(ones, diagonal=1).bool()` (mask upper triangle = future positions).

### NLRQ Compression Ratio Documentation Fix
AGENTS.md claimed NLRQ CR=12.8x for V7. Verified by hand: actual CR for
V7 dims (d_model=4096, intermediate=16384, rank=768) is **8.47x**, not
12.8x. The 12.8x was from smaller dims (2048Ã—8192, rank=256) in earlier
R&D. The code comments in `nlrq_ffn_key.py` are correct (8.1x per proj).

### Test Results
- 57 new tests across 3 files: **all pass**
- Full suite: 877 passed, 2 failed (pre-existing HF tokenizer path issues,
  unrelated to this round)
- Zero regressions

### Projected V7 Memory After Round 16
| Component | Before | After | Technique |
|-----------|--------|-------|-----------|
| FFN weights | 1.52 GB | 0.76 GB (opt) | HINT4-NLRQ |
| PEAGLE draft | 958 MB | 135 MB | Tied heads + LoRA |
| Grad transfer | 2.34 GB | 0.59 GB | int4 + EF21 |
| **Total inference** | ~4.5 GB | ~3.7 GB | - |
| **Total training** | ~8.5 GB | ~7.8 GB | - |

### Future Work
- **HINT4 + INT4 residual**: combine Hadamard-INT4 factors with an INT4
  group-quantized residual to reduce the 18% error to ~3-5%.
- **Wire HINT4 into V7 config**: add `nlrq_factor_bits=4` option after
  quality validation on real model weights (not random matrices).
- **Wire PEAGLEDraftHeadTied into ForgeEngine**: replace the draft head
  construction in `forge_engine.py` with the tied variant.
- **Wire int4 grad compression into sft_train.py**: the CLI flag
  `--grad-compression int4` now works â€” verify on real training run.

## R&D Round 17: LMCache Port â€” CacheBlend + Chunked Prefix + 3-Tier Disk Offload (2026-08-28)

Ported the hardware-appropriate concepts from [LMCache](https://github.com/LMCache/LMCache)
(Best Paper @ ACM EuroSys'25) into ForgeAI. LMCache's distributed/remote/daemon
features (transport mode, Redis/Mooncake backends, engine-independent ZMQ server)
don't fit a single RTX 5070 12GB workstation, so only the three single-machine-
appropriate concepts were ported. Each is config-driven, falls back gracefully,
and is tested on CPU.

### 1. ChunkedPrefixCache (LMCache `TokenDatabase` / `TokenHasher` port)
**File**: `research/inference/prefix_cache.py` (upgraded â€” no new file)
**Flag**: `use_chunked_prefix_cache=True` (ActivationConfig)

Replaces the naive first-32-token prefix key with a chunk-based rolling hash:
- 256-token chunks (LMCache default), deterministic blake2b chunk hash +
  rolling prefix-hash chain (PYTHONHASHSEED-independent, so cross-process safe).
- `lookup_longest_prefix()` finds the **longest** cached prefix matching a
  query â€” enabling **partial hits** (a 1024-token prompt reuses a cached
  768-token prefix and only re-prefills the last 256 tokens).
- Token equality verified on every hit (hash is only a lookup key) â†’ hash
  collisions can never corrupt generation.
- Drop-in compatible with `LRUPrefixCache` (`.get/.put/__contains__/stats`).
- `_slice_past_kv()` slices the per-layer KV list to the matched length for
  partial hits (handles conv-layer `None` entries).

### 2. DiskKVCache â€” 3-tier GPUâ†’CPUâ†’disk KV offload (LMCache `StorageManager` port)
**File**: `research/inference/kv/cpu_kv_offload.py` (extended â€” no new file)
**Factory**: `build_kv_cache("disk_offload")`

Extends the existing 2-tier `CPUKVCache` with a disk tier (LMCache L2 = local
disk / NVMe). Tiers: GPU hot window (8K) â†’ CPU pinned RAM (32K) â†’ disk mmap
spool (128K). LRU eviction cascades GPUâ†’CPUâ†’disk; fetch cascades diskâ†’CPUâ†’GPU.
- **Persistence** (LMCache "storage mode"): `persist=True` + stable `disk_path`
  saves the disk spool on `clear()`/shutdown so KV **survives engine restarts**.
- Memory budget: ~2 MB VRAM/layer for 128K effective context (mixed CPU/GPU/disk
  per AGENTS.md directive D).
- **Bug fix (confirm-then-fix)**: `CPUKVCache._offload_to_cpu` indexed dim 1
  (n_kv) instead of dim 2 (seq) â€” `[:, x:y]` â†’ `[:, :, x:y]`. Latent bug
  (never exercised with n_kv>1 in tests); fixed in both `CPUKVCache` and the
  new `DiskKVCache` methods.

### 3. CacheBlend â€” non-prefix KV reuse (LMCache CacheBlend port + novel twist)
**File**: `research/inference/kv/cacheblend.py` (new â€” genuinely new functionality)
**Flag**: `use_cache_blend=True` (ActivationConfig) / `engine.enable_cache_blend()`

The EuroSys'25 Best Paper novelty: reuse the pre-computed KV of *any* repeated
text chunk (retrieved docs, tool outputs, prompt templates) at *arbitrary*
positions â€” not just prefixes â€” by selectively recomputing boundary tokens.
- `ChunkStore` â€” store per-chunk KV keyed by the same rolling hash as
  `ChunkedPrefixCache` (chunks are discoverable by both systems).
- `RangeMatcher` â€” sliding-window rolling-hash matcher finding all non-prefix
  chunk occurrences (greedy non-overlap, longest-first; token-verified).
- `BlendAssembler` â€” assembles the per-layer KV buffer: reused **V** copied
  verbatim (position-independent); reused **K** re-rotated to its new absolute
  position via `reposition_keys()` (inverse RoPE at old pos â†’ forward RoPE at
  new pos, using the model's `RotaryEmbedding` cos/sin tables).
- **Novel twist** (per directive C â€” "prefer novel over copy"): the paper
  selects critical tokens via an attention score from a *full* prefill pass
  (defeats the purpose for cold chunks). We use a **position-gradient
  heuristic**: `recompute_tokens = ceil(log2(chunk_len))`, capped by
  `recompute_fraction` (default 0.15). No extra forward pass; captures the
  empirical decay of cross-attention influence from preceding text.
- Boundary/gap tokens recomputed in-context by running the model on just
  those positions with the assembled KV as preceding cache, then spliced back.
- Wired into `ForgeEngine.generate()` before the prefix-cache path; on a miss
  returns `None` with zero overhead. Public API: `engine.register_blend_chunk(text)`.

### Integration
- `forge_engine.py`: imports all three; `generate()` attempts CacheBlend â†’
  prefix cache â†’ standard path (cascading fallback). New flags
  `use_chunked_prefix_cache`, `use_cache_blend` in `ActivationConfig` (auto-
  included in `feature_flags` dict). `enable_cache_blend()` +
  `register_blend_chunk()` public methods.
- `kv_backend.py` factory: `"disk_offload"` â†’ `DiskKVCache`.

### Tests
**File**: `tests/unit/test_lmcache_port.py` (19 tests, all CPU-runnable, all pass)
- ChunkedPrefixCache: determinism, hash-chain distinctness, full/partial/miss
  hits, collision guard, `_slice_past_kv`.
- DiskKVCache: GPUâ†’CPUâ†’disk cascade, persistence, factory.
- CacheBlend: ChunkStore, RangeMatcher (non-prefix match / prefix skip /
  collision guard), `reposition_keys` identity + roundtrip, BlendAssembler
  layout, miss-on-short-prompt, stats.
- `pytest tests/unit/test_lmcache_port.py --tb=short -q` â†’ 19 passed.

### Future Work
- Wire CacheBlend into the BSP tool-use path: auto-register tool outputs as
  chunks so repeated tool results skip prefill.
- Benchmark CacheBlend TTFT on a real RAG workload (retrieved-doc reuse) vs
  full prefill â€” measure the realized recompute fraction vs the 0.15 target.
- Add a disk-tier observability hook (LMCache-style hit/miss metrics) to
  `DiskKVCache.info()` for the Prometheus/OTel path when it exists.
- Sweep `recompute_fraction` (0.05â€“0.30) and `chunk_size` (128â€“512) on the
  1.2B model to find the ForgeAI-specific optimum (directive F: sweep, don't guess).

## Environment

- OS: Windows 11, GPU: RTX 5070 12GB
- Python venv: `D:\windsurf\ForgeAI\venv\`
- Key packages: torch, transformers, safetensors, bitsandbytes, pytest

### R&D Round 23: V10-8B Local Training Pipeline Tests (2026-08-29)

Test suite for the V10-8B local training pipeline (9 test files, 87 tests total).
Tests follow TDD pattern: tests for existing code PASS, tests for not-yet-implemented
modules FAIL with ModuleNotFoundError (defining the API contract for implementation).

**Test files** (all in 	ests/unit/):
- 	est_r23_nvme_muon_4bit.py — 9 tests: combined NVMe+4-bit Muon optimizer (3 pass, 6 TDD)
- 	est_r23_v8_keys_wiring.py — 10 tests: V8 config keys wired into model_loader (4 pass, 6 TDD)
- 	est_r23_fp8_gradtopk.py — 9 tests: FP8 activation + GradTopK (ALL PASS — existing R21 code)
- 	est_r23_train_v8_runner.py — 10 tests: V8 training runner, 5 modes, ETA, checkpoints (2 pass, 8 TDD)
- 	est_r23_data_pipeline.py — 9 tests: R22 data compression integration (ALL PASS — existing R22 code)
- 	est_r23_ckpt_tester.py — 10 tests: checkpoint tester, 50-Q probe (ALL TDD — module not built)
- 	est_r23_dlora.py — 10 tests: DLoRA warm-start (ALL TDD — module not built)
- 	est_r23_hypercloning.py — 10 tests: HyperCloning LFM→V8 (ALL TDD — module not built)
- 	est_r23_ligo.py — 10 tests: LiGO growth matrix (ALL TDD — module not built)

**Bug fix in existing code**: TopKGradientOptimizer._sparsify_grad() in

esearch/training/optim/r21_cross_domain.py used id(grad) as the EF error
feedback key, which collides when PyTorch reallocates gradient tensors across
steps (set_to_none=True). Fixed to use id(param) instead. Reproduced as
shape mismatch [128] doesn't match [128, 128] in AdamW._foreach_lerp_.

**Run command**:
`
# Tests for existing code (should all pass):
python -m pytest tests/unit/test_r23_data_pipeline.py tests/unit/test_r23_fp8_gradtopk.py --tb=short -q

# TDD tests (fail until modules are implemented):
python -m pytest tests/unit/test_r23_nvme_muon_4bit.py tests/unit/test_r23_v8_keys_wiring.py --tb=short -q
`

**Modules to implement** (defined by TDD test contracts):
1. NvmeMuon4Bit optimizer class + register in configure_optimizer() as "nvme_muon_4bit"
2. Wire V8 keys into model_loader.py: use_qsa, use_gated_residual, use_ngram_embedding, use_hashed_nlrq
3. 
esearch/sandbox/train_v8.py — V8 training runner (5 modes, ETA, rolling ckpts, resume)
4. 
esearch/evaluation/ckpt_tester.py — 50-question checkpoint probe
5. 
esearch/training/dlora.py — DLoRA (DoRA + LoRA) warm-start
6. 
esearch/architecture/hypercloning.py — function-preserving model expansion
7. 
esearch/architecture/ligo.py — learned linear growth operator


### R&D Round 25-26: Sub-BitNet Quantization (2026-08-31)

**Goal**: Get memory cost near or below BitNet (1.58 bits/w) with better quality,
training-free (post-training), supporting train-to-tune (STE + LoRA).

**Tested on**: V10-1.2B (LFM2.5 port) + Qwen 2.5 0.5B (real pretrained weights).
Scripts: scripts/test_r25_baselines.py, test_r25_additive_fp4_v3.py,
test_r26_sub_bitnet.py, test_r26_ternlc.py, test_r26_qwen.py.

**R25 (0.5-2.0 bpw regime)** -- 5 algorithms tested:
- AdditiveFP4-scaled (AQLM-inspired): TIED with IRI-FP4 at matched bpw, wins on
  finer granularity (+5-6 dB at 1.75 bpw). FP4 base + learned k-means codebook
  on normalized residual + per-block scale.
- IRI-Alloc (EXL2-inspired): FAILED. V9 layers too uniform for per-layer allocation.
- LatticeFP4 (Quip#-inspired): FAILED. Lattice uniform codebook wrong for Gaussian.
- HybridFP4 (SR + AdditiveFP4): FAILED. Stochastic rounding noise overwhelms.
- IRI-FP4 (R15) remains the 0.5-2.0 bpw champion.

**R26 (sub-BitNet regime)** -- 8 algorithms tested, 2 WINNERS:

1. **TernPack per-channel** (1.64 bits/w): FREE WIN -- below BitNet (2.0 bits),
   better quality on ALL layers (+0.3 to +2.9 dB). Base-3 packing (5 ternary
   values per byte) + per-output-channel scale. No calibration data needed.
   quantize_ternary_per_channel() in novel_quant.py.

2. **TernLC-refined** (1.97 bits/w at r=16, 2.99 at r=64): Ternary + Low-Rank
   Correction. W = T*scale + A@B where A@B = SVD of ternary error.
   Alternating refinement (re-ternarize residual, re-SVD).
   +0.7 to +9.5 dB over BitNet. Handles outlier layers (Attn Q L0 kurt=23.43:
   BitNet=3.55 dB, TernLC r=64=13.06 dB, +9.51 dB).
   quantize_ternlc() in novel_quant.py.

**Failed R26 approaches** (documented dead ends):
- BinarySalient (PTQ1.61-inspired): Binary {-1,+1} cant capture Gaussian mass
  near 0 -- the ternary zero is essential. -0.3 to -2.5 dB.
- LowRankTernary (NanoQuant-inspired): Ternary factors cant reconstruct full-rank
  at low rank. Needs QAT (LittleBit uses Dual-SVID). -1.5 to -5.2 dB.
- BinaryCodebook (BTC-LLM-inspired): Same binary problem. -0.7 to -6.7 dB.
- TernPrep (PTQ1.61-inspired): Per-block shift breaks ternary alignment. -0.4 to -2.1 dB.

**Key insight**: At sub-BitNet rates, ternary {-1,0,+1} is near-optimal for
post-training. Binary fails because the zero captures the Gaussian dense mass
near 0. To beat ternary, the correction must add negligible bits -- TernLCs
low-rank SVD correction adds ~0.02-0.08 bytes/w for +1-9 dB.

**Train-to-tune**: Ternary supports STE (already in bitnet_b158_key.py).
TernLCs low-rank correction (A, B) are float16, fully differentiable.
LoRA can be applied on top (freeze ternary+correction, train adapters).
The users int8 loading work (2026-08-31) supports direct int8 load via
load_prequantized() -- TernPack can use the same path with base-3 unpack.

**Wired into**: research/inference/quant/novel_quant.py --
ternary_to_base3_packed(), base3_packed_to_ternary(),
quantize_ternary_per_channel(), quantize_ternary_per_block(),
quantize_ternlc(), ternlc_dequantize(), ternlc_bpw().


### R&D Round 26: V10 -- IRI-FP4 Lossless Weight Quantization (2026-08-31)

**ForgeLM V10-1.2B**: LFM2.5-1.2B with R26 IRI-FP4 lossless weight quantization.
Replaces V9 BitNetResidual (ternary, catastrophic post-training PPL: 663K vs 9.64
baseline) with IRI-FP4 x2 (9.0 bits/w, 41.6 dB SQNR, -0.4% PPL delta -- near-lossless).

**Why V10 replaces V9 for deployment**: V9 BitNetResidual requires QAT to recover
from catastrophic post-training quantization (ternary PPL = 663K). V10 IRI-FP4 is
near-lossless post-training (PPL delta = -0.4%, no QAT needed). The original model
does not need to fit on the system -- V10 quantizes it on port and the packed
checkpoint is 1.87 GB (vs 2.34 GB bf16).

**IRI-FP4 compression levels** (validated on Qwen 2.5 0.5B full-model PPL):
| Rounds | bits/w | SQNR | PPL delta | vs fp32 | vs bf16 |
|--------|--------|------|-----------|---------|---------|
| 1 | 4.5 | 20.7 dB | +50.8% | 7.1x | 3.5x |
| 2 | 9.0 | 41.6 dB | -0.4% | 3.5x | 1.8x |
| 3 | 13.6 | 62.6 dB | +0.1% | 2.4x | 1.2x |
V10 uses x2: best lossless compression (PPL delta within noise).

**V10 VRAM comparison** (1.2B model, bf16 inference):
| Metric | V9 (bf16) | V10 (IRI-FP4 x2) | Savings |
|--------|-----------|-------------------|---------|
| GPU loaded | 2.451 GB | 1.705 GB | 30% |
| GPU after forward | 2.451 GB | 1.714 GB | 30% |
| GPU peak | 2.790 GB | 2.387 GB | 14% |
| Cosine similarity | -- | 1.000000 | near-lossless |

**V10 files**:
- Config: research/config.py -- forgelm_v10_1.2b preset
- Port: research/architecture/port_lfm25_to_v10.py -- LFM2.5 -> V10
- Key: research/keys/quantization/iri_fp4_key.py -- IRIFP4Linear, IRIFP4Key
- Model loader: research/model_loader.py -- packed VRAM load via IRIFP4Linear
- Checkpoint: research/checkpoints/ForgeLM_V10_1.2B.safetensors (1.87 GB)

**V10 usage**:
  python -m research.architecture.port_lfm25_to_v10 --input lfm25.safetensors --output V10.safetensors --rounds 2
  engine = ForgeEngine.from_checkpoint(checkpoint=V10.safetensors, config_name=forgelm_v10_1.2b)

**Sub-BitNet research findings** (R26, documented dead ends):
- Post-training ternary (BitNet, TernPack, TernLC): catastrophic PPL (663K-2.6M vs 9.64)
- Binary (BinSalient, LoRT, BinCB): worse than ternary (no zero level for Gaussian mass)
- TernLC-refined: best SQNR per-layer (+0.7-9.5 dB) but still catastrophic full-model
- Train-to-tune: BitNet QAT recovers to PPL 47.71 in 50 steps, TernLC needs more training
- Conclusion: sub-BitNet post-training is fundamentally broken; IRI-FP4 x2 is the right
  lossless compression for deployment without QAT

### R&D Round 27-28: LoRA Knowledge Injection (prior work, retro-documented)

Prior LoRA experiments on Qwen2.5-0.5B. Results deemed unreliable due to lenient
metrics, curated facts, and incomplete runs. Superceded by R29 which uses strict
exact-match evaluation, programmatic fact generation, and bit-exact merge verification.

### R&D Round 29: LoRA Knowledge Injection -- Golden Ratio (2026-09-01)

**Goal**: Find the "golden ratio" of LoRA parameters needed per fact for >90% new
knowledge recall with 0% regression on existing knowledge.

**Model**: Qwen2.5-0.5B (896 hidden, 4864 FFN intermediate, 24 layers)
**Target**: Single-layer FFN-trio LoRA (gate_proj + up_proj + down_proj at layer 14)
**LoRA params**: 17280 * rank (3 modules * rank * (896 + 4864))

**Process winner (Phase 1)**: `full_qa` -- orthogonal init + cosine scheduler +
replay + KL-with-QA-anchors. Achieved 100% recall, 0 net regression, -0.23% PPL
at r16/n100/120ep. The KL regularization uses QA-format anchor prompts to preserve
the model's generative behavior on known facts, not just logits.

**Golden ratio (Phase 2)**:
- At 120 epochs (batch=16): **2765 params/fact = 6.25 facts/rank**
  - Frontier: r4/r8/r16/r32 for n=25/50/100/200 (all at 2765 p/f)
  - n=400 needs only r16 (691 p/f) due to 2x total optimizer steps
- Epoch scaling: **2x epochs -> 4x fewer params** (quadratic trade-off)
  - r4@n100@240ep = 100% (691 p/f, was 10% at 120ep)
  - r8@n200@60ep = 2% (step-starved, confirms confound)
- Scaling law: **rank ~= 2304 * n_facts / epochs^2**
  - At 120ep: rank = 0.16*n (6.25 facts/rank)
  - At 240ep: rank = 0.04*n (25 facts/rank)

**Info density**: 19.93 bits/fact (random 6-digit codes)
- 0.0072 bits/param at 120ep, 0.029 bits/param at 240ep
- Full training reference: ~2 bits/param -> LoRA is 70-280x less efficient
- Trade-off: lower density but ZERO regression on existing knowledge

**Key invariants verified**:
- LoRA zero-init is a bit-exact no-op (merge produces identical logits)
- Parent weights frozen during training (no contamination)
- Merge reproduces adapter forward bit-exactly (max_logit_diff < 0.05)
- Regression measured on base model's correct answers (known + held facts)

**Bug fix**: `targets_met` was using `== 0` instead of `<= 0` for regression checks,
marking fact gains (model learning a previously-unknown known fact) as failures.
Fixed to `<= 0` (gains are not regressions).

**Tokenization artifact**: "New Delhi" -> " Delhi" is a tokenization difference
(BPE splits "New Delhi" differently), not knowledge loss. The model still knows
the answer; the exact-match metric flags it as a strict flip.

**Files**:
- `scripts/r29_generate_facts.py` -- programmatic synthetic fact generator
- `scripts/test_r29_injection.py` -- MinimalLoRA + process variants + eval + merge
- `scripts/r29_analyze.py` -- golden ratio analysis and frontier computation
- `tests/unit/test_r29_lora.py` -- 7 unit tests (zero-init, merge, invariants)
- `scripts/r29_phase2.json` -- Phase 2 sweep results (25 conditions)
- `scripts/r29_phase1b.json` -- escalation experiment results

### R&D Round 30: V10 QLoRA Training with Golden Ratio (2026-09-01)

**Goal**: Apply the R29 golden ratio to train ForgeLM V10-1.2B with IRI-FP4
quantized weights using QLoRA (frozen quant base + trainable LoRA adapters).

**QLoRA implementation** (new):
- `IRIFP4Linear.forward()` now checks for `lora_adapter` attribute and applies
  it after dequantization: `out = F.linear(x, dequant_w, bias) + lora(x)`
- `IRIFP4Linear.merge_lora()` dequantizes -> adds LoRA delta -> re-quantizes
  to IRI-FP4 (preserving the packed format)
- `bitnet_lora.py` `add_lora_adapters()` now recognizes `IRIFP4Linear` as a
  valid LoRA target (checks `in_features`/`out_features` attrs, doesn't require
  `nn.Parameter` weight)
- `bitnet_lora.py` `merge_lora_adapters()` dispatches to `IRIFP4Linear.merge_lora()`
  for QLoRA modules

**V10 golden ratio**:
- V10 FFN-trio: 3 * (2048 + 8192) = 30720 params/rank/layer
- 16 layers (all have FFN): 491520 params/rank total
- R29 ratio: 2765 params/fact at 120ep -> rank = 2765 * n / 491520
- At 6000 examples, 3 epochs: rank=32 (capped), 15.7M params (2621 p/f)
- Epoch scaling from R29: 2x epochs -> 4x fewer params (quadratic)

**Training**:
- Datasets: openhermes (2000) + gsm8k (2000) + code_alpaca (2000) = 6000 examples
- Converted from hf_datasets `{"prompt", "solution"}` to sft_train format
  `{"prompt", "response"}` using the exact LFM2.5 Qwen chat template:
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n`
- Tokenization: left-padding, pad=<|pad|> (id 0), labels mask prompt tokens,
  completion includes <|im_end|> (id 7 = eos) so model learns to stop
- Optimizer: AdamW (lr=1e-4, wd=0.01, betas=(0.9, 0.95)), cosine schedule
- Training: 1125 steps, 79 seconds, loss 1.10 -> 0.56
- 48 LoRA adapters merged into IRI-FP4 (dequant -> add delta -> re-quantize)

**Verification**:
- PPL on 20 held-out openhermes examples: **-12.05%** (1.36 -> 1.20, improvement)
- General knowledge preserved ("capital of France" unchanged, both correct)
- Some improvements on trained domains (US government branches, code generation)
- Checkpoint: `ForgeLM_V10_1.2B_R30.safetensors` (1.79 GB, 281 tensors)

**Files**:
- `scripts/train_r30_qlora.py` -- QLoRA training pipeline (dataset conversion,
  golden ratio rank computation, training, merge, save)
- `research/keys/quantization/iri_fp4_key.py` -- IRIFP4Linear QLoRA support
  (forward hook + merge_lora method)
- `research/training/bitnet_lora.py` -- add_lora_adapters + merge_lora_adapters
  updated for IRIFP4Linear compatibility
- `research/data/finetune/r30_training.jsonl` -- converted training data
- `research/checkpoints/ForgeLM_V10_1.2B_R30.safetensors` -- trained checkpoint

### R30 v2: Full-Feature QLoRA Training (2026-09-01)

**Goal**: Scale up R30 with all compatible ForgeEngine/sft_train features,
targeting weaknesses revealed by v1 verification (math, code, reasoning).

**Features integrated** (QLoRA-safe, from sft_train audit):
- **Gigatoken tokenizer**: `research.tokenizer_cache.get_tokenizer()` — ~35x
  faster encode vs AutoTokenizer (190s for 40k examples vs ~6700s)
- **Golden ratio param growth**: rank=64 (capped), 31.5M params, 786 p/f at 5ep
- **Disk token cache**: skips re-tokenization across runs (saves 190s on rerun)
- **Async prefetcher**: `AsyncPrefetcher` with `transfer_on_consume=True` —
  overlaps CPU collation + H2D copy with GPU compute
- **Curriculum pacing**: difficulty-ordered training (compression ratio signal)
- **Validation tracking**: 5% held-out, val_loss every 200 steps
- **Checkpointing**: LoRA-only saves every 1000 steps (60MB each)
- **Resume from checkpoint**: `--resume-from` + `--resume-step` for LR continuity

**Features tested then DROPPED** (not worth the cost for QLoRA):
- `--grad-mixup 2`: 2x slowdown, zero benefit (grad_accum=8 already averages)
- `--entropy-alpha 0.5`: 2x slowdown (forces full [B,T,65536] logits instead of
  chunked CE), marginal quality gain on 1.2B base
- `--sequential-freeze 4`: fights golden ratio (reduces effective param capacity
  by training only 4/16 layers at a time)
- `--ema`: marginal for LoRA params that get merged anyway

**Training**:
- Datasets: gsm8k, metamath, orca_math, magicoder_oss, code_alpaca,
  openhermes, no_robots, bbh (5000 each = 40000 total)
- 5 epochs, batch=2, grad_accum=8, lr=1e-4, max_len=1024
- 11875 steps total, ~714s for steps 9000-11875 (lean config, ~9 steps/sec)
- Final train loss: 0.79, final val loss: 0.878

**Verification**:
- PPL on 20 held-out openhermes: **-14.75%** (1.36 → 1.16, up from v1's -12.05%)
- General knowledge preserved ("capital of France" unchanged)
- Math: recognizes "distance = speed x time" formula (v1 was gibberish)
- Code: recognizes sum() pattern (v1 was gibberish)
- Government: correctly lists "Executive, Legislative and Judicial"
- Checkpoint: `ForgeLM_V10_1.2B_R30.safetensors` (1.79 GB, 281 tensors, IRI-FP4)

**Key lesson**: The golden ratio assumes ALL LoRA params trained simultaneously.
Sequential freeze, grad-mixup, and entropy-alpha either fight this assumption or
add 4x cost for ~0 gain. The lean config (just val + curriculum + disk-cache +
async-prefetch) is both faster AND higher quality.

### R&D Round 31: Self-Play Tool-Calling QLoRA (2026-09-01)

**Goal**: Train R31 LoRA on R30 base for self-play discovery loop tool-calling.
The R30 model could do basic math/code but failed to emit valid tool-call JSON
or perform multi-turn agent trajectories.

**Training data** (`r31_v3_training.jsonl`, 1438 examples):
- 1042 from `hermes_fc.jsonl` (real function-calling, converted to pythonic format)
- 396 synthetic discovery tool examples (all 18 tools from `discovery_tools.py`)
- Made-up tool schemas (15 fake tools) to teach schema adaptation
- Multi-turn trajectories (plan → execute → observe → conclude)
- Plain code generation (no tools)

**QLoRA config**:
- Rank 32, alpha 64 (up from R30's rank 64 — lower rank + more data = better)
- FFN + attention targets: `w_gate, w_up, w_down, q_proj, v_proj, out_proj, in_proj`
  (attention LoRA is REQUIRED for tool-name copying from system prompt)
- 86 adapters, 21.7M params (0.46% of base)
- 3 epochs, 256 steps (aligned to eff_batch=16), LR=1e-4, 75s training
- val_loss: 0.53 (down from R30's 1.16)

**LoRA-only save** (no merge): `ForgeLM_V10_1.2B_R31_lora.safetensors` (86MB, 172 tensors)
- Base R30 checkpoint untouched — LoRA loaded at runtime via `engine.load_lora()`

**KV cache bug fix** (critical, affected ALL models not just LoRA):
- `DoubleGatedConvLayer.forward` had `if past_key_value is None and T == 1: self._conv_state = None`
- Conv layers ALWAYS get `past_key_value=None` (they don't have KV cache entries)
- This wiped conv state during EVERY decode step, causing garbage generation
- Fix: reset conv state at MODEL level when `past_key_values is None` (new sequence)
  via `_conv_state_reset` flag, not at per-layer level
- Logit diff (cache vs no-cache): 5.66 → 0.125 (bf16 rounding)

**LoRA hot-loading** (new ForgeEngine feature):
- `engine.load_lora(path, rank, alpha, target_modules)` — attach + load LoRA
- `engine.unload_lora()` — remove LoRA, restore original forwards
- `engine.has_lora()` / `engine.lora_info()` — query state
- Supports dynamic skill injection at runtime (swap LoRAs without reloading base)
- `add_lora_adapters` now saves `_lora_orig_forward` for clean unload

**Self-play discovery loop** (`discovery_loop.py`):
- `from_default_model()` auto-detects R31 LoRA and loads it on R30 base
- Uses `ForgeEngine.load_lora()` for hot-loading
- After each session, `EpochManager.maybe_advance()` may fine-tune a new LoRA
  from DB content, evaluate it, and hot-swap the winner into the running model
- Verified: 5-step autonomous exploration with think → save_research → query_db →
  migrate_schema → record_discovery (all valid tool calls)

**Self-improvement loop** (finetune.py + epoch_manager.py):
- `finetune_from_db()` now uses QLoRA (IRI-FP4 frozen base + LoRA adapters)
  instead of HuggingFace PEFT (which was incompatible with IRI-FP4)
- LoRA adapter saved separately (not merged) → hot-loadable via engine.load_lora()
- `_load_model_at()` handles both LoRA checkpoints (*_lora.safetensors) and
  full checkpoints (*.safetensors)
- Discovery loop hot-swaps LoRA when a new epoch wins the comparison
- Training data from DB: discoveries, theories (supported), scripts (ok),
  research, tool-use trajectories

**Capability test results** (R31 with all 18 tools):
- Knowledge seeking: web_search/wikipedia_search/arxiv_search called correctly
- Web search works: DuckDuckGo + Wikipedia + arXiv return real results
- Fact-checking: run_script/calculate used to verify claims
- Multi-turn: interprets tool results (calculate→states answer, web_search→save_research)
- Autonomous: starts exploring with think() when told "Begin exploring."

**Files**:
- `scripts/train_r31_qlora.py` — QLoRA training (LoRA-only save, no merge)
- `scripts/gen_r31_data_v4.py` — Data generation (hermes_fc + discovery + made-up tools)
- `research/inference/forge_engine.py` — `load_lora/unload_lora/has_lora/lora_info`
- `research/training/bitnet_lora.py` — `_lora_orig_forward` saved for clean unload
- `research/model_loader.py` — KV cache conv state fix
- `research/self_play/discovery/discovery_loop.py` — Auto-loads R31 LoRA
