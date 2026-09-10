# ForgeAI — Agent Notes

## Agent Operating Directives (READ FIRST)

These directives govern how work is done in ForgeAI. They are non-negotiable
unless the user explicitly overrides them for a specific task.

### A. Model Versioning — Build On The Prior, Never Beside It
- **Every new custom model version MUST be derived from the immediately
  preceding version**, carrying forward all prior keys/architecture as the
  baseline, then adding or replacing only what's new. Example chain:
  `lfm25_1.2b` → `forgelm_v2_light` (V3/V4/V5/V7/V8/V9 presets were superseded by V10;
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

### B. Confirm-Then-Fix — Never Leave A Known Bug Sitting
- **When you find an issue, confirm it** (reproduce with a minimal script or
  test) **then fix it in the same session**. Do not log it and move on.
- If a fix would be large/risky, scope a minimal failing test first, then
  implement the smallest correct fix. Prefer targeted edits over rewrites.
- **Always add or update a test** for the fix so it cannot silently return.
  Tests live in `tests/unit/` and run on CPU where possible.
- If a fix is genuinely blocked (needs user input, env change, or a
  destructive op), say so explicitly and create a `todo` — do not pretend
  it's done.

### C. R&D Is The Default Mode — Push For Novel Improvements
- **No area is "solved"**. Every existing technique (attention, KV cache,
  quantization, decoding, optimizer, loss, scheduler) is fair game for a
  novel variation. The codebase already has 13 R&D rounds; round 14+ is
  expected, not exceptional.
- **Prefer novel over copy**: when implementing a known technique, always
  ask "what's the novel twist that could beat the paper's number on our
  specific hardware (RTX 5070, 12GB, SM120, Blackwell)?" Implement the
  baseline AND at least one novel variation in the same round.
- **Cross-domain combinations are the highest-value R&D** — see the
  expanded Novel Discovery Protocol in `docs/CHANGELOG.md`.
- **When stuck on a hard optimization, pivot don't quit**: if 2 iterations
  fail to beat the known best, shelf it in `.devin/scratchpad.md` with the
  failed approaches documented, then touch up a *different* area. Fresh
  context often surfaces the missing idea. Return to the hard problem later.
- **Record failures as carefully as successes** — a documented dead end
  saves the next session hours. Use `.devin/scratchpad.md`.

### D. GPU/VRAM — Mixed Approaches Are Mandatory To Consider
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
  this hardware — prefer them over "just use a smaller model".
- **Profile before assuming**: use `torch.cuda.memory_allocated()` /
  `torch.cuda.max_memory_allocated()` in test scripts. Guessing VRAM is
  how we get OOM at 3am.

### E. No Redundant Files — Search Before You Create
- **Before creating ANY new script or module, grep the codebase for an
  existing one that does the same thing.** The codebase has ~600 .py files
  after the aggressive refactor (was 908); the odds are high that a related
  implementation exists.
- **Prefer upgrading an existing file over spawning a new one.** If
  `forge/inference/kv/snapkv.py` exists and you want "smarter SnapKV",
  edit that file — do not create `snapkv_v2.py` or `smart_snapkv.py`.
  Versioned filenames fragment the codebase and hide the canonical path.
- **If two files end up doing the same thing, merge them** and delete the
  inferior one. Document the merge in the "Removed (consolidation)" section.
- **The canonical path wins**: when in doubt, the file already wired into
  `forge_engine.py` / `forge_server.py` / `sft_train.py` is canonical. New
  code hooks into those, not around them.

### F. Math Thinking + Script Testing — Find The True Optimum
- **Every optimization claim must be backed by a number from a script**,
  not a paper citation. Papers report numbers on different hardware/models;
  our numbers come from RTX 5070 + LFM2.5-1.2B.
- **Write the smallest possible test script first** (see Novel Discovery
  Protocol step 1 in `docs/CHANGELOG.md`). A 20-line script that runs in 5 seconds > a 200-line
  design doc.
- **Do the math by hand for small cases** before trusting a benchmark.
  If a KV compression scheme claims "15×", verify: `bytes_full /
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
  another area — fresh ideas surface indirectly.

### G. General Best Practices For This Codebase
- **Run the test suite before declaring done**: `pytest tests/ --tb=short -q`
  with the venv python and `PYTHONPATH=D:\windsurf\ForgeAI`.
- **Keep AGENTS.md current**: when you add a new file, feature, or R&D
  round, update the relevant section here in the SAME session. Stale
  AGENTS.md causes the next agent to duplicate your work.
- **Use `.devin/scratchpad.md` for long working notes**, not the main
  conversation. Keeps context clean and persists across sessions.
- **Debate the user's premise when warranted** — if a requested approach
  has a known-better alternative on this hardware, say so and propose the
  alternative. Don't silently implement a worse path.
- **Subagent delegation**: use `subagent_explore` for read-only research
  (file indexing, paper lookups) and `subagent_general` for parallel
  implementation tasks. Spawn 2-3 in parallel for independent work.
- **Skills**: invoke `.devin/skills/` skills (`log-bug`, `sync-memory`,
  `systemspecs`, `glm-supercharge`) when they match the task — they encode
  project-specific workflows.
- **IDE crash trigger — batch QA/training-data edits**: The IDE crashes when
  a single edit contains 25+ lines matching Question/Answer or similar
  training-data-like patterns. **Code edits do NOT need batching** — only
  QA pairs, fact lists, training samples, and other data-like content.
  When writing scripts with embedded fact sets or training data, split the
  file write into multiple smaller edits (≤20 QA lines per edit) or write
  the data to a separate `.json`/`.jsonl` file and load it at runtime.

### H. Commit Hygiene — One Logical Change Per Commit
- **One logical change per commit.** If the commit message needs "etc." or a "+"
  to list multiple features, split the commit.
- **Stop amending pushed commits.** Rewriting history breaks bisect and
  collaboration.
- **Commit message format**: imperative mood, focused on WHY not WHAT.
  Example: "Fix int8 dtype mismatch in FastINT8Linear dequantize path"
  NOT: "Add V12 architecture + ForgeQuant + tool-use SFT + GUI fixes"
- **Bisectability**: every commit must build and pass tests. If a commit
  breaks tests, fix it or revert before pushing.

## Current Canonical Layout

Production code: `forge/` (was `research/`, migrated commit f1a7542)
- `forge/config.py` — ModelConfig dataclass + presets
- `forge/model_loader.py` — ConfigurableResearchLLM, ModelLoader
- `forge/engine/forge_engine.py` — ForgeEngine (inference engine)
- `forge/engine/decoding.py` — Decoding strategies
- `forge/keys/` — KeyStack architecture keys (25 canonical)
- `forge/quant/` — Quantization implementations
- `forge/decoding/` — Decoding implementations
- `forge/training/` — Training runners + optimizers
- `forge/evolution/` — Evolutionary optimizer (ForgeEvolve); CLI: `python -m forge.evolution --domain <name> --steps <N>` (use `--list-domains` to see all domains)
- `forge/self_play/` — Self-play + discovery
- `forge_gui/` — PySide6 GUI
- `tests/unit/` — Unit tests (CPU-runnable where possible)
- `tests/integration/` — Integration tests (GPU required)
- `docs/` — Documentation + R&D round notes
- `scripts/` — Standalone scripts
- `research/` — Legacy paths (tokenizer cache, checkpoints only)

## Current Architecture: ForgeLM V2 Light-1.2B (SOLE BASE)

**Base model**: ForgeLM V2 Light-1.2B — lossless 1:1 port of LFM2.5-1.2B + V10 inference features.
- Same architecture as LFM2.5: 16 layers (10 conv + 6 GQA), d_model=2048, 32 heads, 8 KV heads
- V10 additions: IRI-FP4 weight quantization (9.0 bits/w, lossless, 3.5× vs fp32)
- 1304.6M params, 1.87 GB checkpoint (IRI-FP4 compressed)
- All prior ForgeLM models (V2/V4/V5/V7/V8/V9) deleted — V10 is the sole base

**Porting fix (2026-08-30)**: LFM2.5's `embedding_norm` is the FINAL norm (applied
after all layers, before head), NOT a post-embedding norm. The HF name is misleading.
Config uses `use_final_norm=True, use_embed_norm=False` to match. Port script:
`forge/architecture/port_lfm25_to_v10.py`.

**Base checkpoint**: `research/checkpoints/ForgeLM_V2_Light.safetensors`
**Tokenizer**: `research/checkpoints/lfm25_tokenizer/`
**Default config**: `forgelm_v2_light` (load_default_model() defaults to this)
**Path constant**: `research.paths.V10_CHECKPOINT` (LFM25_CHECKPOINT and V9_CHECKPOINT are backward-compat aliases to V10_CHECKPOINT)

### LFM2.5 original architecture (preserved in V10)
- 16 layers: 10 double-gated conv + 6 GQA attention (layers 2,5,8,10,12,14)
- d_model=2048, 32 heads, 8 KV heads (GQA 4x), head_dim=64
- SwiGLU FFN (intermediate=8192), RMSNorm, QK-layernorm on attention
- RoPE theta=1M, 128K context (32K for VRAM budget)
- Vocab=65536, tied embeddings

## Config Presets

Config presets (V3/V4/V5/V7/V8/V9 superseded and checkpoints deleted; V2-Light-1.2B is the sole base):
- `forgelm_v2_light` — **ForgeLM V2 Light-1.2B: THE DEFAULT AND SOLE BASE.** Lossless 1:1 port of LFM2.5-1.2B + V10 inference features (IRI-FP4 weight quantization, 9.0 bits/w, lossless, 3.5× vs fp32). Same architecture as LFM2.5 (d_model=2048, 16 layers, 1304.6M params). `load_default_model()` defaults to this. All tests run against this checkpoint.
- `lfm25_tiny` — 4-layer tiny model for fast testing (no checkpoint, config-only)
- Other presets (`forgelm_v7*`, `forgelm_v8_8b`, `forgelm_v9*`, `lfm25_1.2b`) have been DELETED from `forge/config.py`. Only `forgelm_v2_light`, `lfm25_tiny`, and `gen_model_tiny` remain.

### Evolution-Discovered Promotions (2026-08-24, from forge_evolve.db)
Promoted after validation against evolution data (39,631 discoveries scanned):
- **MTP**: n_heads=4, loss_weight=0.495 (was 2/0.3, score 27.69) ✓ validated
- **SpecDecode (PEAGLE)**: n_draft=7 (was 4, score 57.05) ✓ validated
- **BatchQueue**: batch_window=52ms, max_batch=15 (was 50ms/8, score 44.25) ✓ validated
- **SFT training**: grad_accum=5, grad_compression=int4 (score 30.00/11.10) ✓ validated
- **AirMoEKey**: cache_strategy=lfu, disk_cache_size=4096 (score 8.83, 0% miss rate) ✓ validated
- **ForgeEngine**: CREATIVE_SAMPLING preset (temp=1.98, top_p=0.989, top_k=69, score 10.45) ✓ validated
- Already applied prior: W8A8 fp8+alpha=0.999, PagedEvictKV page=64/LRU, ModConfig aux_loss=1e-8, CheckpointRecompute selective/block=512, FocalLoss gamma=4.93

### Reverted promotions (scoring artifacts — synthetic metrics didn't match real behavior)
- **MoE routing** (4/3/switch/6e-5): REVERTED to 8/2/aux_free/0.01. Evolution found top_k=1 scored higher (24.95 vs 24.87) but top_k=1 is a trivial solution (no ensemble). Scoring fixed: diversity penalty for top_k=1.
- **RoPE theta=10M**: REVERTED to 1M. Synthetic metric rewarded angle diversity, not attention quality. Scoring fixed: checkpoint compat penalty + frozen-dimension detection at long range.
- **Scheduler warmup=0**: REVERTED to warmup=500/20. Synthetic AUC rewarded no warmup, ignored training stability. Scoring fixed: stability penalty for zero warmup.
- **Label smoothing 0.29**: REVERTED to 0.1. Synthetic metric rewarded high smoothing for grad magnitude. Scoring fixed: smoothing penalty above 0.2 + focus ratio metric.

> Detailed evolution domain scoring fixes, DB rescores, focus profiles, and R&D round
> documentation have been moved to `docs/CHANGELOG.md`.

## Build & Test Commands

```powershell
# Run tests
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -m pytest tests/ --tb=short -q

# Verify model loads
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -c "from forge.model_loader import ConfigurableResearchLLM; print('OK')"

# Benchmark INT4 quantization
D:\windsurf\ForgeAI\venv\Scripts\python.exe D:\windsurf\ForgeAI\.devin\benchmark_int4.py
```

## Environment

- OS: Windows 11, GPU: RTX 5070 12GB
- Python venv: `D:\windsurf\ForgeAI\venv\`
- Key packages: torch, transformers, safetensors, bitsandbytes, pytest

## Historical Notes
R&D round documentation, refactor retrospectives, and historical notes
have been moved to `docs/CHANGELOG.md` (extracted 2026-09-09, critique F1).
