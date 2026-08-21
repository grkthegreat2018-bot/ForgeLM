# ForgeAI — Agent Notes

## Agent Operating Directives (READ FIRST)

These directives govern how work is done in ForgeAI. They are non-negotiable
unless the user explicitly overrides them for a specific task.

### A. Model Versioning — Build On The Prior, Never Beside It
- **Every new custom model version MUST be derived from the immediately
  preceding version**, carrying forward all prior keys/architecture as the
  baseline, then adding or replacing only what's new. Example chain:
  `lfm25_1.2b` → `forgelm_v3` → `forgelm_v4` → `forgelm_v5`.
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
  expanded Novel Discovery Protocol below.
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
  existing one that does the same thing.** The codebase has 340+ .py files
  across 13 R&D rounds; the odds are high that a related implementation
  exists.
- **Prefer upgrading an existing file over spawning a new one.** If
  `research/inference/kv/snapkv.py` exists and you want "smarter SnapKV",
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
  Protocol step 1). A 20-line script that runs in 5 seconds > a 200-line
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

## Current Architecture: LFM2.5-1.2B

**Base model**: Liquid AI LFM2.5-1.2B-Instruct, ported 100% lossless into ForgeAI framework.
- 16 layers: 10 double-gated conv + 6 GQA attention (layers 2,5,8,10,12,14)
- d_model=2048, 32 heads, 8 KV heads (GQA 4x), head_dim=64
- SwiGLU FFN (intermediate=8192), RMSNorm, QK-layernorm on attention
- RoPE theta=1M, 128K context (32K for VRAM budget)
- Vocab=65536, tied embeddings
- 1.17B params, 2.34GB bf16

**Base checkpoint**: `research/checkpoints/ForgeLM_V2_LFM25-1.2B.safetensors`
**Self-play starting checkpoint**: `research/checkpoints/ForgeLM_V2_BSP.safetensors` (sft4: 600 steps, 1657 examples, proper tool use + format)
**Tokenizer**: `research/checkpoints/lfm25_tokenizer/`

## Config Presets

Four configs:
- `forgelm_v4` — **ForgeLM V4: the new default.** Hardware-efficient evolution of V3. GTA (Grouped-Tied Attention, `attn_type="gta"`, V=K identity warm start, halves KV cache BW), Fused QKV + Gate-Up GEMM (`use_fused_gemm=True`, halves kernel launches), all V3 keys preserved (BitNet b1.58, TITAN memory rank 64, MoD keep 1.0, MHC rank=512 gate=0, AttnRes k=4 gates=0). Loads `ForgeLM_V2_BSP.safetensors` **bit-exact** via automatic GQA→GTA conversion (V=K, v_mix_gate=0). `load_default_model()` defaults to this. New inference features: W8A8 INT8 quantization (`quantize="w8a8"`), PagedEviction KV cache (`kv_cache="paged_eviction"`), XQuant rematerialization (`kv_cache="xquant"`), Megakernel decode (`acceleration="megakernel"`), Chunked prefill (`use_chunked_prefill=True`).
- `forgelm_v3` — ForgeLM V3: Differential Attention (`attn_type="diff"`, identity warm start), BitNet b1.58 QAT FFN, TITAN memory (rank 64), MoD router (keep 1.0), MHC hyper-connections (rank=512, gate=0), AttnRes cross-layer retrieval (k=4, gates=0). 1256M params. Loads `ForgeLM_V2_BSP.safetensors` **bit-exact** via automatic GQA→diff conversion. Benchmark: 165-328 tok/s, 2.69 GB VRAM (RTX 5070).
- `lfm25_1.2b` — Reference LFM2.5-1.2B port (plain GQA, no keys).
- `lfm25_tiny` — 4-layer tiny model for fast testing

## Key Files

### Core
- `research/config.py` — ModelConfig dataclass + presets
- `research/model_loader.py` — ConfigurableResearchLLM, ModelLoader, PreAllocatedKVCache
- `research/checkpoint_io.py` — Safetensors checkpoint I/O
- `research/paths.py` — Central path management
- `research/tokenizer_cache.py` — Tokenizer loading + caching

### Architecture (only MTP kept)
- `research/architecture/port_lfm25_to_forgeai.py` — Porting script (reference)
- `research/architecture/port_to_v3.py` — V3 porting script (bakes arch key conversions)
- `research/decoding/mtp.py` — MTP speculative decoding + training (independent heads design)

### Inference
- `research/inference/forge_engine.py` — Unified inference engine. **Hot-swap support**: `engine.hotswap.set_kv_cache()`, `set_decoding()`, `set_context_limit()`, `set_infinite_context()`, `set_generation_defaults()`, `set_feature()`, `update()`. `generate_batch()` for parallel multi-prompt generation. `generate_adaptive()` for RPO-trained adaptive thinking. Per-request `context_limit` override.
- `research/inference/forge_server.py` — **v3.2: OpenAI-compatible FastAPI server** with multi-model (ModelRegistry), tool calling, SSE streaming, sleep/wake, **concurrent task-based generation** (SessionManager + BatchQueue), request batching, **hot-swap API** (`/v1/engine/*`), **Library API** (`/v1/library/*`), **agentic tool API** (`/v1/chat/agent`, `/v1/tools`, `/v1/tools/execute`), and per-task config updates. Built-in tools give the LLM direct access to Library, hot-swap, batch gen, and engine introspection.
- `research/inference/session_manager.py` — **SessionManager + BatchQueue**: per-task conversation context (LRU eviction), concurrent request batching (50ms window, up to 8 requests per batch → single BatchedDecoding forward pass). 1005 tok/s on 8 concurrent tasks (3.5x vs serial)
- `research/inference/decoding.py` — Decoding strategies (standard, speculative, MTP, **batched**)
- `research/inference/batched_decoding.py` — BatchedDecoding: multiple prompts in single forward pass (GEMV→GEMM, 3-5x throughput)
- `research/inference/kv_backend.py` — KV cache strategies (standard, paged, rotorquant, hadamard, compressed, streaming, snapkv, **kvzip**)
- `research/inference/hotswap.py` — **HotSwapManager**: runtime config changes without restart. Hot-swap KV cache, decoding, quantization, context limits, generation defaults, feature flags, VRAM margin, batch config. Thread-safe, per-request overrides. Applied lazily on next generate() or forced via apply_pending().
- `research/inference/library.py` — **Library**: persistent knowledge base with lorebook-style injection. Pre-tokenizes content on save (token cache saved as .npy alongside .json). Auto-trims by relevance (LRU + priority + category retention). Model self-write (failures/wins/research). Tag + keyword indices for O(1) lookup. `inject()` scans prompt for triggers, injects matching entries up to token budget. `get_injection_tokens()` returns pre-tokenized IDs (no re-tokenization). Disk-backed at `research/data/library/`. API: `/v1/library/*`.
- `research/inference/engine_tools.py` — **EngineToolRegistry**: 38 built-in tools the LLM can call during generation. **Library** (9): `library_save/search/lookup/get/delete/update/stats/optimize/set_config`. **Hot-swap** (7): `engine_set_kv_cache/decoding/context_limit/infinite_context/generation_params/feature/apply_changes`. **Engine info** (3): `engine_get_settings/stats/pending`. **Generation** (2): `engine_batch_generate`, `engine_generate_adaptive`. **Math** (2): `math_eval` (safe AST-based expression eval with sqrt/sin/cos/log/etc.), `calc` (quick calculator). **Random** (2): `random_number` (int/float ranges), `chance` (coin_flip/dice/choice/shuffle/weighted). **Web** (3): `web_search` (Tavily API), `web_search_semantic` (Exa neural search), `web_scrape` (Firecrawl page scrape). **Files** (7): `file_read/write/edit/move/rename/list/delete` (workspace-scoped, path-validated, security-checked). **Security** (3): `scan_script` (pre-scan Python for dangerous imports/patterns), `security_get_config`, `security_get_pending`. Server-side execution via `execute(name, args)` / `execute_calls(calls)`. All file writes and web scrapes go through `ToolSecurityManager`. Used by `generate_with_tools()` agentic loop and `/v1/chat/agent` API endpoint. API keys in `.env`: `TAVILY_API_KEY`, `EXA_API_KEY`, `FIRECRAWL_API_KEY`.
- `research/inference/tool_security.py` — **ToolSecurityManager**: sandbox-file-based security for LLM tool execution. **Sandbox model**: `research/data/sandbox.json` defines all access rules. Everything OUTSIDE the sandbox is blacklisted for writes (defaults to read-only). **3 access levels**: `read_write` (full access), `read_only` (can read but not write/delete), `denied` (no access at all). **Reading is always allowed** within workspace unless path is `denied` — reading blacklisted files is OK. **File blacklist**: 17 patterns for files that can't be written/deleted (.env, secrets, keys, .pem, .ssh, credentials). **Protected engine files**: 14 core files that can never be written (but CAN be read). **Command blacklist**: 40+ risky patterns (os.system, subprocess, eval, exec, shutil.rmtree, socket, pickle, ctypes, winreg, directory traversal) → triggers `needs_permission` flag. **Hard refusal patterns**: attempts to disable security or hijack tool registry. **Website whitelist/blacklist**: domain filtering for web tools (empty whitelist = all allowed). **Script pre-scan**: AST-based scan of Python scripts for 17 dangerous imports + dynamic import detection. **Auto modes**: `allow`, `deny`, `ask` (default). **Permission flow**: risky ops create pending requests → user approves/denies via API. **Sandbox persistence**: all config changes auto-save to `sandbox.json`. API: `/v1/security/config` (GET/PATCH), `/v1/security/reload` (POST), `/v1/security/access/{path}` (GET), `/v1/security/pending` (GET), `/v1/security/pending/{id}/approve|deny` (POST), `/v1/security/scan` (POST).
- `research/data/sandbox.json` — **Sandbox config file**: defines access rules for LLM tool execution. Writable dirs: `research/data`, `research/output`, `research/sandbox`, `research/results`, `.devin`. Read-only dirs: `research/inference`, `research/training`, `research/self_play`, `research/distillation`, `research/evaluation`, `research/checkpoints`, `research/architecture`, `research/decoding`, `research/quantization`, `research/keys`, `research/moe`, `research/runtime`, `AGENTS.md`, `docs`. Denied: `.env`, `.git`, `venv`, `.venv`, `node_modules`, `__pycache__`. Website whitelist/blacklist empty by default (all sites allowed).
- `research/inference/quant/int4_quant.py` — INT4 weight-only quantization
- `research/inference/innovations.py` — Runtime innovations (MRL, QuaRot, V0, ProgressiveKV)

### Decoding implementations
- `research/decoding/dspark.py` — DSpark speculative decoding
- `research/decoding/eagle.py` — EAGLE-3 speculative decoding (feature-level, multi-layer fusion, TTT training)
- `research/decoding/medusa.py` — Medusa parallel prediction heads
- `research/decoding/mtp.py` — MTP speculative decoding

### Quantization
- `research/quantization/` — BitNet, FP8, RotorQuant, SpinQuant, WANDA, paged KV, KV compress

### Model Merging
- `research/merge_models.py` — SLERP, TIES, DARE, SVD, Task Arithmetic, Linear (model soup) on safetensors state dicts. CLI: `python -m research.merge_models --method <slerp|ties|dare|svd|task_arith|linear> ...`
- `research/inject_and_merge.py` — Unified pipeline: inject new params via KeyStack knowledge keys (facts, context patches, self-play, spectral, test-gated) then merge delta into target model. Auto-clones target as injection base for clean task vector. CLI: `python -m research.inject_and_merge --target <ckpt> --inject-type <facts|test_gated|context_patch|selfplay_patch|spectral> --merge-method <task_arith|ties|dare|svd|slerp|linear> ...`

### Keys (70+ files in research/keys/)
All keys preserved. Wired into V3:
- mHC (Manifold Hyper-Connections, DeepSeek-V4) — `use_mhc=True`, rank=512, gate=0 lossless
- AttnRes (Attention Residuals, Kimi K3) — `use_attn_residual=True`, k=4, gates=0 lossless
- PIT (Pseudo-Inverse Tying) — `use_pit=True`, L=I lossless (replaces weight tying)
- Differential Attention, BitNet b1.58, TITAN memory, MoD router — all lossless at load
Planned for further integration:
- MTP, Safety, LeRoPE, CSA, QK-Norm, SandwichNorm, LearnedSink, ValueResidual, SwiGLU Clamp, MRL

### Self-play & Training
- `research/training/optim/hybrid_offload.py` — **CPUAdamW**: ZeRO-Offload-style hybrid CPU-GPU optimizer. Keeps fp32 optimizer states + master weights on CPU pinned RAM, model bf16 on GPU. Eliminates 14.4GB fp32 AdamW VRAM cost for 1.2B models → full-precision training on 12GB GPU. Async grad offload + optional overlap mode (CPU math on background thread, GPU param sync on main thread). Use via `--optimizer cpu_offload` in any trainer. VRAM budget: ~6-7GB GPU (weights+grads+activations), ~19GB CPU (optimizer states).
- `research/training/runners/cpt_train.py` — **CPT with reasoning trace injection** (midtraining stage of LFM2.5-1.2B-Thinking recipe). Mixes reasoning traces (openthoughts, openr1_math, dolphin_r1) with general data (orca_math, metamath) at configurable ratio (default 60% reasoning). Full-sequence next-token prediction (no completion masking, unlike SFT). Sequence packing for efficiency. MixedDataSampler ensures every batch has the right reasoning/general ratio. CLI: `python -m research.training.runners.cpt_train --reasoning-data <files> --general-data <files> --checkpoint <ckpt> --save <out> --optimizer cpu_offload --reasoning-ratio 0.6 --lr 1e-4 --max-steps 5000`
- `research/training/runners/curriculum_sft.py` — **Curriculum SFT: Mix Distillation + two-stage curriculum** (SFT stage of LFM2.5-1.2B-Thinking recipe, informed by Small Model Learnability Gap research arXiv 2502.12143). Three subcommands: `prepare` (classifies data into short/long CoT, filters doom-loops via n-gram repetition, applies mix distillation blending), `train-stage1` (short CoT — builds internal solver, higher LR), `train-stage2` (long CoT + mix distillation — externalizes reasoning, lower LR), `full` (runs all three). Doom-loop filter: n-gram repetition ratio detector (n=8, threshold=0.3) removes training examples with excessive repetition (#1 failure mode of reasoning models). Mix distillation: blends long CoT (teacher) + short CoT (student) at target ratio to match small model intrinsic learning capacity. CLI: `python -m research.training.runners.curriculum_sft prepare --input <files> --output-dir <dir> --filter-doom-loops --mix-ratio 0.5`
- `research/training/runners/dpo_data_gen.py` — **DPO preference data generation with doom-loop mitigation** (DPO stage of LFM2.5-1.2B-Thinking recipe). Generates 5 temperature-sampled + 1 greedy candidate per prompt from SFT checkpoint, scores each with an LLM judge (teacher API model via distill_client), flags doom-loop candidates via n-gram repetition, constructs preference pairs where chosen=best non-looping, rejected=worst OR any looping (loops always rejected regardless of judge score). Reduces doom-loop rate from ~15% (SFT) to ~4% (DPO). CLI: `python -m research.training.runners.dpo_data_gen --prompts <jsonl> --checkpoint <sft_ckpt> --output <pairs.jsonl> --n-temp-samples 5 --judge-model qwen3-32b`. Then: `python -m research.training.runners.dpo_align --data <pairs.jsonl> --checkpoint <sft_ckpt> --save <dpo_ckpt> --optimizer cpu_offload`
- `research/training/runners/rlvr_train.py` — **RLVR (Reinforcement Learning with Verifiable Rewards)** (RL stage of LFM2.5-1.2B-Thinking recipe). GRPO-style RL on verifiable tasks with binary rewards (1.0 if verified correct, 0.0 otherwise). N-gram repetition penalty applied early in training (doom-loop mitigation, reduces rate from ~4% to ~0.36%). KL penalty against DPO checkpoint (reference model). Supports SPPO/PS-PPO/EVPO/GRPO-OR via --rl-algorithm. Math verification: extracts final answer (#### N, \boxed{}, "answer is X") and compares to gold. CLI: `python -m research.training.runners.rlvr_train --tasks <jsonl> --checkpoint <dpo_ckpt> --save <rlvr_ckpt> --use-repetition-penalty --rl-algorithm grpo --optimizer cpu_offload --max-steps 500`
- `research/self_play/grpo_trainer.py` — **Updated**: Added n-gram repetition penalty (LFM2.5-Thinking RLVR recipe) via `use_repetition_penalty`, `repetition_n`, `repetition_threshold`, `repetition_penalty`, `repetition_warmup_steps` config fields. Penalty applied in `train_step` after GRPO-λ length penalty, before advantage computation. Added CPUAdamW optimizer support via `config.optimizer='cpu_offload'`.
- `research/prequantize.py` — **Pre-quantization script**: Converts any ForgeLM checkpoint to BitNet b1.58 ternary int8 format. Ternary-quantizes all `.weight` tensors to {-1, 0, +1} and stores as int8 (1 byte/param vs 2 bytes bf16). Writes `_bitnet_prequant=1` metadata to safetensors for auto-detection by ForgeEngine. CLI: `python -m research.prequantize --checkpoint <input.safetensors> --output <output.safetensors>`. ForgeLM_V5_Base (14.1GB bf16) -> ForgeLM_V5_BitNet (7.08GB int8, 2x disk compression). ForgeEngine loads int8 directly to VRAM (no fp32 intermediate), then converts BitNetLinear layers to int8 buffer storage (4x weight VRAM cut vs fp32). V5 7.5B model loads in 5.8GB VRAM (vs 14.1GB bf16).
- `research/keys/quantization/bitnet_b158_key.py` — **Updated**: Added int8 weight storage mode for pre-quantized inference. `BitNetLinear.load_prequantized(int8_tensor, scale)` loads ternary weights directly as int8 buffer (bypasses fp32 parameter). `convert_model_to_int8(model)` converts all BitNetLinear layers. `BitNetLinear.forward()` uses int8 tensor-core GEMM directly when `_prequantized=True` (no runtime quantization cost). `_load_from_state_dict` handles int8 source tensors and meta-device loading.
- `research/inference/forge_engine.py` — **Updated**: Auto-detects pre-quantized BitNet checkpoints via safetensors metadata (`_bitnet_prequant=1`). Loads int8 weights directly to VRAM (no fp32 intermediate, no CPU RAM spike). Builds model on meta device, loads tensors one at a time to GPU, converts BitNetLinear to int8 buffer storage. `activate_optimal()` method: S4R KV cache (15x), torch.compile, fused QK-Norm+RoPE+Cache, Triton conv, prefix cache, chunked prefill, seq-aware split. V5 7.5B: 5.8GB VRAM inference.
- `research/moe/moe.py` — **Updated**: Dense bypass path handles BitNet int8 experts. When experts are BitNetLinear with `_prequantized=True`, calls each expert's forward individually (ternary GEMM kernel) instead of stacking weights for batched matmul.
- `research/training/runners/sft_train.py` — **Updated**: Optimal defaults for VRAM-efficient training: `--optimizer muon_sf` (2.39x vs AdamW), `--lora` default True (rank 32, alpha 64), `--bitnet-everywhere` default True, `--grad-checkpoint` default True. Use `--no-lora`, `--no-bitnet-everywhere`, `--no-grad-checkpoint` to disable.
- `research/self_play/infinite_loop.py` — **Updated**: Added `ThinkingPipeline` class + `ThinkingPipelineConfig` that orchestrates the full LFM2.5-1.2B-Thinking 4-stage pipeline (CPT -> Curriculum SFT -> DPO -> RLVR). Each stage runs as a subprocess calling the corresponding runner. Resumable: completed stages are skipped if output checkpoint exists. Accessible via `--self-play-mode thinking`. Also fixed pre-existing argparse `%%` escaping bug in `--saerl` help text. CLI: `python -m research.self_play.infinite_loop --checkpoint <base_ckpt> --self-play-mode thinking --config forgelm_v4 --ft-optimizer cpu_offload`
- `research/self_play/infinite_loop.py` — **Unified AZR self-play loop** (entry point). Propose → solve → verify → SFT → eval → promote. CLI: `python -m research.self_play.infinite_loop --checkpoint <ckpt> --epochs 50`
- `research/self_play/infinite_curriculum.py` — AZR curriculum engine (task proposal, validation, solving, ELO difficulty tracking)
- `research/self_play/` — Recursive self-play, sandbox, curriculum, GRPO
- `research/self_play/discovery/discovery_tools.py` — **Self-play discovery tool registry** (NOT for inference). Tools: `think`, `sudo_think`, `run_script`, `web_search` (DuckDuckGo, no API key), `wikipedia_search`, `arxiv_search`, `fetch_url`, `calculate`, `save_research`, `propose_theory`, `update_theory`, `record_discovery`, `query_db`, `migrate_schema`, `summarize_context`, `finish_session`, `set_goal`. Used by the discovery self-play loop and agentic distillation. For inference tool registry, see `research/inference/engine_tools.py`.
- `research/training/` — DPO alignment, training utils, chunked CE
- `research/evaluation/` — Prompt tests, reasoning benchmarks, LiveCodeBench

### Runtime
- `research/runtime/` — CUDA graphs, flex attention, VRAM manager, signal capture

## Training-Free Alignment (research/training_free/)

Forward-only adaptation — no gradients, no optimizer, no weight updates:
- `urial.py` — URIAL in-context alignment (`build_prompt`: system + 3 style examples).
- `reflexion.py` — `ReflexionBuffer`: bounded episodic memory rendered into the prompt.
- `steering.py` — `ActivationSteerer`: capture residual activations, extract task vectors (`positive - negative`), inject via pre-hooks.
- `rain.py` — `RAINGenerator`: self-eval + rewind-and-regenerate loop.
- `solver.py` — `TrainingFreeSolver`: frozen-solver adapter combining the above; `record(task, output, error, success)` collects activations, `build_task_vector()` + `apply_steering(alpha)` steer inference.
- `SelfPlaySandbox(...)` accepts `training_free=TrainingFreeSolver(...)` (or call `sandbox.enable_training_free()`): run_task styles prompts with ICA + memory and records every outcome — replaces GRPO weight updates for self-play adaptation.
- Tests: `tests/unit/test_training_free.py` (CPU, tiny model).

## V4 Hardware-Efficient Inference (2026-08)

Eight techniques built based on analysis of faster/more-efficient engines (mini-vllm,
mini-infer, Zerfoo, vLLM, DeepSeek-V3 MLA, PagedEviction, XQuant). All wired into
`forgelm_v4` preset and `load_default_model()`.

### Architecture Changes (active in V4 preset)
1. **GTA (Grouped-Tied Attention)** — `research/keys/attention/gta_key.py`
   - Ties V to K: `V = (1-gate)*K + gate*V_proj(x)`, gate=0 at init → V=K (lossless)
   - Halves KV cache write bandwidth (only K written, V=K derived)
   - `attn_type="gta"` in config; `GTAKey` converts GQA→GTA at checkpoint load
   - Training unties V from K (gate opens)
2. **GLA (Grouped Latent Attention)** — `research/keys/attention/gla_key.py`
   - Latent-compressed KV: projects K/V into compact latent, caches only latent
   - Identity warm start (latent_dim = full KV dim, gate=0 → lossless)
   - `attn_type="gla"`, `gla_latent_dim=0` (lossless) or >0 (compressed)
   - `GLAKey` converts GQA→GLA at checkpoint load (kv_down_proj=k_proj, identity up-projs)
3. **Fused QKV + Gate-Up GEMM** — `research/keys/quantization/fused_gemm_key.py`
   - `FusedQKVLinear`: single GEMM for Q/K/V projections (3→1 kernel launch)
   - `FusedGateUpLinear`: single GEMM for FFN gate+up (2→1 kernel launch)
   - `use_fused_gemm=True` in config; wired in `_maybe_fuse_qkv()` and `build_ffn()`
   - Lossless (same math, just fused weight matrix)

### Inference Optimizations (opt-in via ForgeEngine.activate())
4. **W8A8 INT8 Quantization** — `research/inference/quant/w8a8_quant.py`
   - `W8A8Linear`: weight + activation INT8 with `torch._int_mm` tensor-core GEMM
   - 2-3× decode speedup at batch=1 (shifts to compute-bound on tensor cores)
   - `quantize="w8a8"` in `activate()`; `FP8Linear` for Blackwell FP8 E4M3
5. **PagedEviction KV Cache** — `research/inference/kv/paged_eviction.py`
   - Block-wise eviction (entire 16-token blocks, not individual tokens)
   - Compatible with PagedAttention (no fragmentation), FlashAttention (no scores needed)
   - L2-norm-based scoring (attention-score-free); 3020 tok/s on LLaMA-1B (37% over full)
   - `kv_cache="paged_eviction"` in `activate()`
6. **XQuant KV Rematerialization** — `research/inference/kv/xquant_kv.py`
   - Caches layer-input activations X (INT4) instead of K/V; rematerializes K=W_K@X, V=W_V@X
   - 2× memory savings (INT4 X vs bf16 KV); up to 16× with SVD compression
   - `kv_cache="xquant"` in `activate()`
7. **Megakernel Decode** — `research/decoding/megakernel.py`
   - `CompiledMegakernelDecode`: CUDA graph capture of entire decode step + torch.compile
   - Single graph replay per token (1 API call vs dozens of kernel launches)
   - `acceleration="megakernel"` in `activate()`
8. **Chunked Prefill** — `research/inference/prefill/chunked_prefill.py`
   - `ChunkedPrefiller`: splits long prompts into 512-token chunks
   - Interleaves with decode steps (prevents long prompts from blocking decode queue)
   - `use_chunked_prefill=True` in `activate()`

## V4 R&D: Next-Gen Inference Techniques (2026-08)

Five additional techniques from latest 2026 research, all opt-in via `ForgeEngine.activate()`:

### FlexDecoding (research/inference/attention/flex_decoding.py)
- PyTorch 2.5+ FlexAttention with dedicated decode backend (FlashDecoding)
- Splits KV cache across all 192 SMs (vs 8 SMs with standard SDPA for 8 KV heads)
- `acceleration="flex_decoding"` — auto-switches to FlexDecoding kernel for q_len=1
- Expected: 1.5-3× decode speedup for batch=1 (SM utilization 4% → 100%)

### NVFP4 Quantization (research/inference/quant/nvfp4_quant.py)
- Native FP4 (E2M1) on Blackwell 5th-gen tensor cores with block scaling
- 2× throughput vs FP8, 4× memory vs FP16. ~99% quality with calibration
- RTX 5070 (SM120) supports `mxf8f6f4.block_scale` in MMA instructions
- `quantize="nvfp4"` — ForgeLM V4 (1.2B): 2.34GB → ~0.66GB (3.6× compression)

### ATFlash Wavelength Pruning (research/inference/attention/atflash.py)
- Per-RoPE-wavelength distance windows: each frequency pair prunes beyond its wavelength
- Prunes 37-48% of QK inner-product terms, 96-98% top-1 match rate (near-lossless)
- Input-independent (closed-form, no dynamic search), orthogonal to token-level sparsity
- `use_wavelength_pruning=True` — 1.29× at 128K context, grows with context length

### Adaptive n-gram + EAGLE-3 Speculative Decoding (research/decoding/adaptive_speculative.py)
- Combines n-gram lookup (free, optimal for code/RAG) with EAGLE-3 (model-based, novel gen)
- Adaptive selection: tracks per-request hit/acceptance rates, picks best drafter
- Up to 4.9× on code-editing, 2.89× average (MLSys 2026 production benchmarks)
- `use_adaptive_spec=True` — composes with existing EAGLE-3/MTP heads

### POD-Attention (research/inference/attention/pod_attention.py)
- Prefill-decode overlap: runs prefill (compute-bound) and decode (BW-bound) concurrently
- Uses CUDA streams to overlap on separate SMs (28% mean speedup for hybrid batches)
- `use_pod_attention=True` — pairs with BatchQueue's mixed prefill+decode batches

## V4 R&D Round 2: Long-Context + Speculative + VRAM (2026-08)

Five more techniques from latest 2026 research, all opt-in via `ForgeEngine.activate()`:

### CoSA Sparse Attention (research/inference/attention/cosa.py)
- Proxy-kernel co-designed sparse attention for long-context decode
- KAP scores KV blocks by key-norm × position-decay; OSK visits top-scored blocks
- 4.93× attention speedup at 128K, 2.53× TTFT, negligible quality loss
- `use_cosa=True` — activates for decode when KV > 2048 tokens

### Suffix Decoding (research/decoding/suffix_decoding.py)
- Training-free speculative decoding via suffix tree matching
- Matches longest repetitive patterns (code, RAG, summarization, multi-turn)
- Composes with n-gram (suffix for long, n-gram for short) and EAGLE-3
- `use_suffix_spec=True` — `ComboSpeculativeDecoder` combines all three drafters
- 1.5-2× on code/RAG workloads, zero training cost

### Sequence-Aware Split (research/inference/attention/seq_aware_split.py)
- Splits KV cache across SMs for low-head-count decode (8 KV heads → 192 SMs)
- Auto-computes optimal split count based on seq_len and SM count
- Split-merge attention with online softmax (log-sum-exp trick)
- `use_seq_split=True` — 21-24% decoder kernel efficiency improvement

### CPU KV Cache Offloading (research/inference/kv/cpu_kv_offload.py)
- Two-tier KV cache: GPU hot window (8K tokens) + CPU cold storage (rest)
- Extends effective KV cache to 128K+ within 12GB VRAM budget
- Async prefetch via CUDA stream for overlap with compute
- `kv_cache="cpu_offload"` — 32K hot window = 1.07GB GPU, 128K total = 4.3GB CPU

### CompactAttention (research/inference/attention/compact_attention.py)
- Block-union KV selection for efficient chunked prefill
- Unions selected KV blocks across Q positions and GQA groups
- In-place sparse attention on selected blocks (no KV compaction)
- `use_compact_attn=True` — 2.72× attention speedup at 128K under chunked prefill

## V4 R&D Round 3: Kernel Fusion + KV Compression + Compile (2026-08)

Five more techniques from latest 2026 research, all opt-in via `ForgeEngine.activate()`:

### Fused QK-Norm+RoPE+Cache-Write (research/inference/attention/fused_qk_norm_rope_cache.py)
- Fuses RMSNorm + RoPE + KV cache append into single kernel (vLLM #38621 pattern)
- Eliminates 2 kernel launches per attention layer (32 fewer launches per decode step)
- Warp-per-head design: V written first (fire-and-forget) while K norm runs
- Optional FP8/INT4 KV quantization fused into the same kernel
- `use_fused_qk_norm_rope_cache=True` — 5-10% decode speedup

### S4R Low-Rank KV (research/inference/kv/s4r_kv.py)
- Selective Sampling + Subspaces + Sparse Reconstruction for KV cache
- Builds low-rank basis from sampled prompt tokens (no external calibration)
- Sink tokens preserved full precision, rest stored as low-rank coefficients
- Sparse reconstruction during decode (only top-k relevant entries)
- `kv_cache="s4r"` — up to 15× KV compression with near full-cache accuracy

### HqeKV Hybrid Quant+Eviction (research/inference/kv/hqe_kv.py)
- Combines quantization AND eviction in a single KV cache (ACL Findings 2026)
- Three tiers: full precision (10%), INT8 (30%), INT4 (50%), evicted (10%)
- Joint K-V importance metric: ||K|| × ||V|| × recency_decay
- Integrated optimizer auto-selects compression action per token
- `kv_cache="hqe_kv"` — 7.9× KV memory reduction with minimal quality loss

### torch.compile Auto-Tuner (research/inference/graphs/compile_autotune.py)
- Benchmarks all torch.compile modes (default, reduce-overhead, max-autotune, etc.)
- Picks fastest mode per model, caches results to .devin/compile_mode_cache.json
- Heuristic fallback: get_recommended_mode() without benchmarking
- `use_compile_autotune=True` — avoids mode selection guesswork

### Block Fusion (research/inference/graphs/block_fusion.py)
- Per-block CUDA graph capture + torch.compile for full transformer block fusion
- Each block gets its own graph (supports MoD skip — skip entire block graph)
- CompiledBlockFusion: combines per-block graphs with torch.compile intra-block fusion
- `use_block_fusion=True` — 1.3-1.5× over eager (ClusterFusion++ approximation for SM120)

## V4 R&D Round 4: Prefix Caching + Scheduling (2026-08)

Four more techniques from latest 2026 research on prefix caching and batch scheduling:

### Feather Scheduler (research/inference/scheduler/feather_scheduler.py)
- Prefix-homogeneity-aware batch scheduling (2-10× throughput for prefix-sharing)
- Groups requests by shared prefix → smaller homogeneous batches outperform
  larger heterogeneous ones (better KV cache locality)
- Chunked Hash Tree (CHT): O(1) prefix detection vs O(depth) radix trees
- `BatchQueue(use_feather_scheduler=True)` — integrates with existing batching

### Learned Prefix Cache (research/inference/kv/learned_prefix_cache.py)
- ML-guided prefix cache eviction (NeurIPS 2025)
- Online logistic regression predicts continuation probability
- Features: recency, frequency, length, is_conversation
- 18-47% cache size reduction at equivalent hit ratios, 11% prefill throughput
- `use_learned_prefix_cache=True` — replaces LRU dict with learned policy

### Hybrid Chunked Prefill (research/inference/prefill/hybrid_prefill.py)
- Adaptive: chunk only when decode is active, continuous prefill otherwise
- Eliminates throughput tax of unconditional chunking (vLLM #26625)
- +2-5% total token throughput, 10-20% lower TTFT under low concurrency
- `use_hybrid_prefill=True` — extends ChunkedPrefiller with decode awareness

### HotPrefix (research/inference/scheduler/hotprefix.py)
- Hotness-aware GPU/CPU prefix KV promotion (SIGMOD 2026)
- Tracks prefix access frequency × recency × length → hotness score
- Hot prefixes → GPU (fast), cold → CPU (saves VRAM), periodic rebalancing
- `use_hotprefix=True` — system prompts stay on GPU, unique contexts offloaded

## V4 R&D Round 5: Training Optimizations (2026-08)

Six training-side techniques from latest 2026 research, all opt-in via `sft_train.py` CLI flags:

### FlashOptim 8-bit Optimizers (research/training/optim/flash_optim.py)
- `FlashAdamW`: 8-bit optimizer states with companding quantization (sqrt transform)
- `FlashLion`: 8-bit sign-momentum (single state, even more memory-efficient)
- 57% memory reduction vs standard AdamW (16→7 bytes/param, 5 with gradient release)
- Block-wise quantization isolates outliers, companding reduces small-value error
- `--optimizer flash_adamw` or `--optimizer flash_lion`

### FORGE Fused Gradient Elimination (research/training/optim/forge_optimizer.py)
- Folds optimizer step INTO backward pass via gradient hooks
- Each gradient tile consumed in registers the instant it's produced
- 53% peak memory reduction, 1.5× faster at small batch sizes
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
- 1.24× sustained TFLOPS, +1 batch size at each model scale
- `--lazy-train` or `--checkpoint-strategy lazy`; `--hybrid-clip` for fast clipping

### Streaming Parquet Prefetch Loader (research/training/data/streaming_loader.py)
- SlidingWindowCache: NVMe/disk cache with sliding window eviction
- BatchPlanProvider: precomputes epoch batch ordering (enables prefetch)
- PrefetchingLoader: wraps DataLoader with automatic prefetch triggers
- 10× data loading speedup (50→500 samples/s), zero worker crashes
- Integrates with existing ParquetDataset

### Optimal Checkpoint Planner (research/training/runners/optimal_checkpoint.py)
- Sliding-window Hirschberg knapsack for optimal checkpoint selection
- O(W) memory (vs O(nW) for dp_knapsack) — handles n=2000 (vs n=100)
- 25-28% runtime speedup over PyTorch's default solver
- `--checkpoint-strategy optimal`

## V4 R&D Round 6: Loss Functions + Quantization + Sparse Attention (2026-08)

Five more techniques from latest 2026 research:

### Improved Loss Functions (research/training/losses/improved_losses.py)
- `FocalCrossEntropy`: down-weights easy tokens, focuses on hard ones (γ=2.0)
- `LabelSmoothingCE`: prevents μ-singularity collapse (ε=0.1), better calibration
- `LovaszSoftmax`: directly optimizes Jaccard/exact-match (+36% EM on math/QA)
- `DynamicFocalCE`: curriculum-style focal weight (low→high γ over training)
- `MixtureLoss`: combines multiple losses with configurable/learnable weights
- `--loss-function focal|label_smoothing|lovasz|dynamic_focal|mixture`

### OffQ Activation Outlier Offsetting (research/quantization/offq.py)
- Top-1 PCA + rotation + offset absorption for W4A4KV4 quantization
- Concentrates outliers into 1 channel, absorbs as shared offset
- Enables uniform-grid W4A4KV4 without mixed precision
- `OffQQuantizer` — calibrates from sample inputs, applies to model

### AAAC Adaptive Codebooks (research/quantization/aaac.py)
- Two learned scalar codebooks per layer (64 bytes overhead, zero storage for selection)
- Activation-weighted k-means places levels where they matter most
- Outperforms AWQ, GPTQ, QuIP# at orders-of-magnitude less quantization time
- `AAACQuantizer` — 3-30 min calibration on single GPU

### MoSA Mixture of Sparse Attention (research/inference/kv/mosa.py)
- Expert-choice routing: each head selects k tokens to attend to
- Content-based, head-specific, perfectly balanced sparse attention
- 27% better perplexity at same compute, smaller KV cache
- `use_mosa=True` — activates for sequences > 512 tokens

### TriRoute Unified Routing (research/inference/scheduler/triroute.py)
- Single controller emits joint policy: attention mode (skip/local/full) + KV bits (4/8/16)
- Gumbel-Softmax + STE for categorical, load-balanced top-k for experts
- Better quality-compute tradeoff than independent MoD + KV quantization
- `use_triroute=True` — unifies MoD block skipping with KV precision selection

## V4 R&D Round 7: CUDA Graphs + Self-Play Improvements (2026-08)

Seven more techniques from latest 2026 research:

### Breakable CUDA Graph (research/inference/graphs/breakable_cuda_graph.py)
- SGLang BCG: segmented graph capture for dynamic shapes
- 1.70× faster prefill, 1.93× with full capture, 3.8-5.2× faster graph building
- Captures decode graphs for batch sizes [1, 2, 4, 8], pads at runtime
- `use_breakable_cuda_graph=True`

### CoRun Deterministic Inference (research/inference/scheduler/corun.py)
- Padding-based determinism: isolate prefill + fixed-shape batched decode
- Position-invariant kernels → pad to fixed shape → one CUDA graph
- 15-324% throughput over batch-invariant, -51.8% TTFT, -48.6% TPOT
- Per-request RNG state (reproducible sampling for RL/eval)
- `use_corun=True`

### Foundry Template-Based Cold Start (research/inference/graphs/foundry.py)
- Persists CUDA graph context (topology + kernel binaries + memory layout)
- Online materialization <1s vs 10-30s for fresh capture → 10× faster startup
- `use_foundry=True` — auto-captures or loads templates from `.devin/graph_templates/`

### SOAR Meta-RL Curriculum (research/self_play/soar.py)
- Teacher proposes stepping-stone problems, rewarded by student improvement
- Grounded rewards (measured progress) > intrinsic rewards (avoids collapse)
- Escapes learning plateaus on hard problems (0/128 success → improvement)
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
- 1.3× faster prefill, 1.6-1.9× faster decode (with FP8 KV cache)
- Software-emulated exponential, TMEM, 2-CTA MMA, ping-pong tiles
- Auto-detects sm_120 (RTX 5070), falls back to FA2/SDPA
- `use_fa4=True`

### SF-NorMuon (research/training/optim/sf_spectral_optimizers.py)
- Schedule-free spectral optimizer with per-neuron normalization
- Matches tuned AdamW across 1-8× Chinchilla horizons (no LR schedule)
- Weight decay at fast iterate Z, iterate averaging for anytime checkpoints
- `--optimizer sf_normuon`

### AMUSE (research/training/optim/sf_spectral_optimizers.py)
- Anytime Muon + Stable gradient Evaluation
- Time-varying interpolation: fast Muon → stable averaged (suppresses oscillations)
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
- 8× KV reduction at 32K context
- `use_kara_kv=True`

### MomentKV (research/inference/kv/moment_kv.py)
- Moment statistics over evicted tokens (count, key/value mean, VK covariance)
- Closes the directional gap: evicted tokens near-orthogonal to retained → big error
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
- High confidence → prune aggressively; low confidence → retain more
- Mixed FP16/INT8 storage (important → FP16, rest → INT8)
- Pyramidal per-layer budget (deeper layers get more)
- 91.4% retrieval on 32K NiH (vs 53.8% sliding window)
- `use_conf_kv=True`

## V4 R&D Round 9: Speculative Decoding + Tokenization + RoPE (2026-08)

Eight more techniques from latest 2026 research:

### P-EAGLE Parallel Speculative Decoding (research/decoding/peagle.py)
- Generates ALL K draft tokens in a SINGLE forward pass (vs K sequential)
- Sequence partition algorithm: maintains attention dependencies across chunks
- 1.05-1.69× speedup over vanilla EAGLE-3 on B200
- `use_peagle=True`

### Lookahead Quality Gate (research/decoding/lookahead_gate.py)
- Block-wise acceptance: accepts longest reliable prefix of k-token draft
- Geometry-based quality score from hidden states (no auxiliary heads)
- Quantile-calibrated threshold (estimated from unlabeled prompts)
- 2.6-7.9× faster generation + improved accuracy on math/science
- `use_lookahead_gate=True`

### Pruned BPE (research/tokenization/pruned_bpe.py)
- Post-training visibility pruning: low-exposure tokens → internal-only
- Reallocates freed vocabulary slots to better-exposed candidates
- 0.27-0.36% shorter encoding at same vocabulary size
- No model retraining (tokenizer post-processing only)

### ToaST Split-Tree Tokenization (research/tokenization/toast.py)
- Greedily splits pre-tokens into binary trees using n-gram counts
- Vocabulary selection via Integer Program (LP relaxation, near-optimal)
- 11% fewer tokens than BPE/WordPiece/UnigramLM at vocab ≥ 40K
- +2.6-7.6% CORE score on 1.5B models

### Jet-Long Dynamic Bifocal RoPE (research/inference/scheduler/jet_long.py)
- Local RoPE-faithful window + long-range dynamically rescaled window
- Parameter-free analytic schedule: recovers base at short, extrapolates at long
- +4.79 pp RULER at 128K (1.7B), 1.39× FA2 throughput, ≤4% overhead
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
- Proactive GPU↔host memory offloading for preempted requests
- 6.1× throughput improvement over vLLM
- `use_fastserve=True`

### Libra Micro-Request Partitioning (research/inference/scheduler/libra.py)
- Flexible partitioning: split requests at ANY token boundary into cooperating segments
- Two-level scheduling: global (split points) + local (SLO-aware batches)
- Chunked KV cache transfers for cross-instance execution
- 1.91× goodput, 1.15-3.07× serving capacity
- `use_libra=True`

### FASER Fine-Grained SD Phase Management (research/inference/attention/faser.py)
- Dynamic speculative length per-request (based on acceptance rate)
- Token-wise early exit: stop verification at first rejection
- Frontier execution: overlap verification chunks with next draft
- 53% higher throughput, 1.92× lower latency
- `use_faser=True`

### Kairos SLO-Aware Scheduling (research/inference/scheduler/kairos.py)
- Prefill: urgency-based priority (closest to missing TTFT deadline first)
- Decode: slack-guided adaptive batching (pack more when under TPOT SLO)
- +23.9% TTFT SLO, +27.1% TPOT SLO, +33.8% e2e SLO, +19.3% decode throughput
- `use_kairos=True`

### Curriculum Learning (research/training/data/curriculum_augment.py)
- Orders training data easy→hard using compression ratio, MTLD, Flesch readability
- Strategies: vanilla, pacing, interleaved, warmup
- 18-45% fewer steps to reach baseline, 3.5% sustained improvement as warmup
- `--curriculum pacing|vanilla|interleaved|warmup`

### Training-Time Data Augmentation (research/training/data/curriculum_augment.py)
- Three orthogonal categories: token-level noise, sequence permutations (FIM), target offset
- Regularizes against overfitting in multi-epoch data-constrained training
- `--augment`

### SYNPRO Synthetic Data Generation (research/training/data/curriculum_augment.py)
- Rephrasing + reformat operations (RL-optimized generators)
- 3.7-5.2× effective tokens from same organic data
- Surpasses non-data-bound oracle at 1.1B scale
- `--synpro`

### SeeDNorm Self-Rescaled Dynamic Normalization (research/training/optim/advanced_norm.py)
- Dynamically adjusts scaling coefficient based on input norm
- Preserves input norm information (RMSNorm discards it)
- Minimal params (1 per layer), negligible efficiency impact
- Consistently superior to RMSNorm and LayerNorm
- `--norm-type seednorm`

### Dynamic Tanh (DyT) (research/training/optim/advanced_norm.py)
- Bounded normalizer: γ * tanh(α * x) + β (no normalization needed)
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
- 1.15× prefill, 1.34× decode at half budget
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
- Importance-weighting correction → unbiased truncated gradient
- Large compute and memory savings for long reasoning traces
- `rl_algorithm="psppo"`

### EVPO Explained Variance PO (research/training/losses/advanced_rl.py)
- Monitors batch-level explained variance (EV) to adaptively switch
- Positive EV → critic-based (PPO); zero/negative EV → batch-mean (GRPO)
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
- Online N:M mask extracts outlier backbone → FP4, dense residual → FP4 GEMM
- Training-free: no calibration, retraining, or model-specific tuning
- 2.2-2.4× latency over FP16, 1.2-1.4× throughput over FP8 on RTX 5090
- `use_sharq=True`

### MosaicQuant Inlier-Outlier Disaggregation (research/quantization/adaptive_quant.py)
- Dense 4-bit base (inliers) + sparse 4-bit residual (outlier compensation)
- ZipperEngine: fuses sparse computation into dense GEMM pipeline
- Near-FP16 accuracy, 1.24× speedup over W16A16
- `use_mosaic_quant=True`

### AoH Autonomy-of-Heads (research/inference/kv/aoh_retmask.py)
- Data-free head classification from frozen QK geometry (effective rank of M_h)
- Low rank → retrieval head (full attention), high rank → streaming (sink+window)
- 50% sparsity: 96.5% of full attention, 66% decode latency reduction
- `use_aoh=True`

### RetMask Retrieval Head Optimization (research/inference/kv/aoh_retmask.py)
- Contrastive masking: train by contrasting normal vs retrieval-masked outputs
- +2.28 HELMET at 128K, +70% citation generation, +32% passage re-ranking
- Gains correlate with retrieval score sparsity

### Offline Top-K Logits + Chunked KL (research/training/runners/efficient_distillation.py)
- Cache teacher's top-K logits once → train against cache (no teacher in loop)
- 29% faster per iteration, 41% higher throughput
- Fused chunked KL: peak memory linear in seq length (4× context on same GPU)
- `--distill --teacher-checkpoint <path>`

### Sequence Truncation + Prefix OPD (research/training/runners/efficient_distillation.py)
- Train on first 50% of tokens → 91% of full-sequence performance
- On-policy prefix distillation: distill only reasoning prefixes
- 2-40× FLOP reduction, early-terminate sampling
- `--distill-truncate 0.5 --distill-prefix`

## V4 R&D Round 13: General Upkeep & Bug Fixes (2026-08)

Codebase-wide audit and fixes across all 12 previous R&D rounds:

### Critical Bug Fixes (4)
- **kvpop.py**: `KVpopScorer.__init__` missing `n_kv` parameter → added (was
  `AttributeError` at runtime). `v_sink_len` → `sink_len` (wrong attribute name).
- **conf_kv.py**: `_promote_from_window` referenced `self.positions` (doesn't
  exist on `ConfKVCache`) → fixed to `self.mixed_kv.positions` with empty-check guard.
- **moe_optim.py**: `AllocMoE.stats` referenced `self.total_budgets` (plural)
  → fixed to `self.total_budget` (singular).

### High-Severity Fixes (1)
- **aoh_retmask.py**: `classify_heads` assumed PyTorch Linear weights are
  `(d_model, n_heads * head_dim)` but they're actually `(out_features, in_features)`
  = `(n_heads * head_dim, d_model)` → fixed view + matmul to match real layout.

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

## 2025/2026 Architecture Keys (research/keys/)

All config-driven, dimension-generic (work at LFM2.5-1.2B scale, d_model=2048). **Main `lfm25_1.2b` preset now enables ALL of them losslessly** — verified bit-exact (max logit diff 0.0) vs the plain GQA model on the real BSP checkpoint, for both plain and KV-cached prefill+decode:
- `quantization/bitnet_b158_key.py` — BitNet b1.58 ternary QAT: `BitNetLinear` (STE, learned per-layer `qscale` re-anchored on checkpoint load, ternary ONLY in training; eval = full-precision master weights until `bitnet_force_quant`). **True BitNet integer kernels on CUDA**: default = int8 @ int8 tensor-core GEMM (`torch._int_mm`, a4.8-style activation quant); `FORGE_BITNET_KERNEL=triton` selects the b1.58 add-only Triton kernel (fp activations, zero-skip, no weight multiplies — verified bit-exact vs fp on small shapes). Applies to FFN + attention q/k/v/o projections. Enable: `use_bitnet=True` (main preset: on).
- `attention/differential_attn_key.py` — Diff-Transformer: dual-softmax subtraction, per-head λ, per-head RMSNorm+scale. **Identity warm start** (`lambda=0`): group-1 rows extracted contiguously (`_group1_weights`) so GEMM shapes match GQA exactly → bit-exact conversion; training moves λ off 0 to activate the real mechanism. `attn_type="diff"` (main preset: on; GQA checkpoints auto-convert at load). KV cache stores 2× head_dim.
- `attention/differential_attn_key.py` — Diff-Transformer: dual-softmax subtraction, per-head λ (paper init), per-head RMSNorm+scale. `attn_type="diff"`; KV cache stores 2× head_dim. `DifferentialAttentionKey` = GQA→diff weight transform (dup rows, warm start).
- `architecture/titan_memory_key.py` — TITAN neural memory: gated memory, zero-init gate => **lossless at start** (ported checkpoint loads identically). `TitanMemory.update()` = Hebbian surprise step (test-time training). Enable: `use_titan_memory=True`.
- `architecture/mod_router_key.py` — Mixture-of-Depths: per-block top-k token router (STE hard mask, soft grad). keep_fraction=1.0 => **lossless**. **TRUE skip in training** (no cache/mask): skipped tokens genuinely bypass attention+FFN (per-row gather/scatter, FLOPs scale with keep_fraction — verified: 0.5 fraction processes exactly 50% of tokens); router trained via aux loss (`ModRouter.aux_loss`). Inference keeps all tokens (KV alignment). Enable: `use_mod=True`.
- Main `lfm25_1.2b` preset: `attn_type="diff"`, `use_bitnet=True`, `use_titan_memory=True` (rank 64), `use_mod=True` (keep_fraction=1.0) — all lossless at load; training activates each mechanism. `get_config` returns FRESH copies (preset mutation no longer leaks).
- Tests: `tests/unit/test_arch_keys.py` (incl. main 1.2B build forward).

## AirMoE Expert Consolidation (research/training_free/expert_bake.py)

Static offline counterpart of AirMoE runtime hotswap: fold topic experts into dense FFN weights.
- `decompress_expert(state)` — decodes raw / SVD-only / SVD+INT4 expert files (mirrors `research/moe/airmoe_infinite.py` formats; LatentMoE up/down skipped).
- `bake_expert(target, expert_paths, alpha, layers, out)` — per-layer task arithmetic: `target += alpha * mean(expert − base_ffn)` per `w_gate/w_up/w_down`. Multiple experts per layer are averaged. Output is a normal dense .safetensors — no router, no disk I/O at inference.
- Layer parsed from `expert_l{layer}_{topic}.safetensors` filenames; override with `--layers`.
- CLI: `python -m research.training_free.expert_bake --target T --expert experts/expert_l0_math.safetensors --alpha 0.8 --out O.safetensors`
- Tests: `tests/unit/test_expert_bake.py`.

## Offline Weight Baking (research/training_free/bake.py)

Permanent weight modification without backpropagation:
- `bake_task_vector(target, finetuned, base, alpha, out)` — task arithmetic: adds `alpha*(finetuned - base)` onto an arbitrary target checkpoint (e.g. self-play checkpoint). Offline tensor math, output is a normal .safetensors.
- `extract_distill_dataset(packets_jsonl, out_jsonl, ...)` — context distillation: from self-play packet logs, keeps only (task, final correct code) pairs in sft_train JSONL format for a low-epoch SFT pass.
- `fuse_lora(base_ckpt, adapter_dir, out, alpha_override)` — offline PEFT LoRA fusion (`W' = W + (alpha/r)*B@A`), standalone output, no PEFT dependency at inference. (sft_train also fuses via `merge_and_unload` at save time.)
- CLI: `python -m research.training_free.bake {task-vector|distill|fuse-lora} ...`
- Tests: `tests/unit/test_bake.py` (CPU, tensor-level).

Constant-memory note: full SSM blocks (Mamba-2) were removed in the 2025-01 cleanup; the LFM2.5 arch already gets constant-memory O(1) state from its 10 conv layers (`DoubleGatedConvLayer`) — use `hybrid_offload` (conv→CPU, attention→GPU) to trade compute for VRAM instead of re-adding SSM code.

## I/O & VRAM Notes (2026-08)

- `StreamingDataLoader(ds, ..., num_workers=N, prefetch_factor=2)` — multi-worker path is picklable on Windows (module-level `_ParquetWorkerDataset`); `pin_memory=True` + device now does non-blocking H2D.
- `ParquetDataset` pickles by re-opening the file (workers re-open by path).
- `HadamardKVCache` pre-allocates its max buffer on first append (no per-token `torch.cat`).
- `StreamedGenerator` no longer double-syncs per decode step (`.item()` is the single sync).
- `load_checkpoint(..., map_location="cpu")` returns memory-mapped tensors (zero-copy).
- New config: `selective_gradient_checkpointing` — `"all"` (default), `"ffn"` (recompute only FFN, big VRAM save), `"attn"`, `"none"`. Enable with `use_gradient_checkpointing=True`.
- `model.cache_devices()` runs once on first forward (16+ `next(param).device` scans cached); `hybrid_offload` invalidates it.

## Recommended Improvements (2026-08)

Four improvements implemented based on architectural critique:

### 1. SnapKV + 4-bit KV Combined Cache (`research/inference/kv_backend.py`)
- `SnapKV4BitCache` — composes SnapKV eviction + Hadamard INT4 quantization.
- Total compression = eviction_ratio × bit_ratio (up to ~16× vs fp16 full cache).
- Strategy name: `"snapkv_4bit"` in `build_kv_cache()`.
- Evicts low-attention tokens first (SnapKV), then quantizes survivors (Hadamard INT4).

### 2. Golden Trajectory Injection (`research/self_play/grpo_trainer.py`)
- `GRPOTrainer` now accepts `replay_buffer` parameter (FOREVER-style `ReplayBuffer`).
- `GRPOConfig.replay_ratio` (default 0.15) — fraction of each batch from golden replays.
- `_inject_golden_replays()` — mixes previously successful trajectories into training batch.
- `_record_golden_trajectories()` — stores verified-successful completions for future replay.
- Prevents catastrophic forgetting in continual self-play (anti-regression).

### 3. ELO-Driven Curriculum (`research/self_play/elo_tracker.py`)
- `EloTracker` — ELO rating system for both model and individual prompts.
- Targets ~50% expected success (Goldilocks zone, max learning signal).
- `select_prompts()` — picks prompts closest to model's skill boundary.
- `select_mixed_prompts()` — Goldilocks-matched + exploration prompts.
- Integrated into `InfiniteCurriculum.record_result()` and `get_training_batch_elo()`.
- Zero-sum rating updates, K-factor decays with prompt attempts (stabilizes ratings).

### 4. Fused QK-Norm + RoPE Triton Kernel (`research/decoding/fused_rope_qknorm.py`)
- Fuses RMSNorm + RoPE into a single Triton kernel for Q and K preprocessing.
- Halves HBM traffic (1 load + 1 store vs 2+2 for separate ops).
- `fused_qk_norm_rope()` — public API, auto-falls back to PyTorch on CPU/compile-fail.
- Opt-in via `FORGE_FUSED_ROPE_QKNORM=1` env var in `GroupedQueryAttention.forward`.
- Attention itself stays on FlashAttention-2 (FA2) via SDPA — already fused.
- Tests: `tests/unit/test_recommended_improvements.py`.

## GRPO-λ Dynamic Length Penalty (2026-08)

Prevents the "CoT length penalty trap" (arXiv 2509.01155): static length penalties
cause accuracy collapse early in training when the model is still learning to reason.

### GRPOTrainer (`research/self_play/grpo_trainer.py`)
- `GRPOConfig.use_grpo_lambda` — enable dynamic length penalty (default False).
- `_group_correctness_ratio()` — fraction of completions with reward >= 0.99.
- `_apply_grpo_lambda_penalty()` — penalty only when correctness_ratio >= threshold.
  - Low correctness → NO penalty (pure 0/1 rewards, prioritize reasoning).
  - High correctness → penalty = -λ * n_tokens (encourage efficiency).
- `length_penalty_warmup` — delay penalty activation for first N steps.
- Stats track `correctness_ratios` and `length_penalty_active_count`.

### GoalScorer (`research/evaluation/goal_scorer.py`)
- `minimalism_active` flag (default True) — toggle minimalism/length penalty.
- `set_minimalism_active(False)` — redistributes minimalism weight to efficiency/diversity.
- Called by training loop based on GRPO-λ correctness ratio.
- Tests: `tests/unit/test_grpo_lambda.py` (17 tests).

## BitNet-Everywhere + Manual LoRA + Sequential Freeze (`research/training/bitnet_lora.py`)

Production training stack for V3 on 12GB RTX 5070. Validated on real 1.2B V3:
2.39x better convergence than AdamW, 6.32GB VRAM (53% of 12GB).

### CLI Flags (sft_train.py)
- `--bitnet-everywhere` — Convert ALL nn.Linear → BitNetLinear (ternary b1.58 QAT).
  No NF4/bnb needed — BitNet IS the quantization (1.58 bits). 41 layers converted,
  72 already BitNet = 113 total ternary layers.
- `--manual-lora` — Use manual LoRA adapters (BitNet-compatible, unlike PEFT which
  can't inject into BitNetLinear). Auto-enabled with `--bitnet-everywhere --lora`.
  97 adapters, 25.40M trainable params (2% of model) at rank=32.
- `--sequential-freeze N` — Train N layers at a time in phases. Full forward pass
  preserves MHC/AttnRes cross-layer connections; only gradients are scoped.
  Best with 100+ steps/phase. 0=disabled.
- `--final-finetune-steps N` — Reserve N steps at end to fine-tune ALL layers together.
- `--optimizer muon_sf_plain` — Muon (Newton-Schulz) + ScheduleFree AdamW, no blockwise
  sharpness. Optimal for V3 (blockwise conflicts with BitNet). 2.24x vs AdamW.
- `--grad-mixup 3` — 3-way gradient averaging. 1.25x better convergence, 3x compute/step.

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
- `convert_to_bitnet_everywhere(model)` — Replace all nn.Linear with BitNetLinear
- `add_lora_adapters(model, rank, alpha)` — Manual LoRA (works with BitNetLinear)
- `merge_lora_adapters(model)` — Merge LoRA into base weights for standalone save
- `freeze_unfreeze_lora(model, active_layers)` — Freeze/unfreeze by layer index
- `compute_phase_schedule(n_layers, n_phases, total_steps, final_finetune_steps)` — Phase schedule
- `build_muon_sf_lora_opt(lora_params, lr_muon, lr_adam)` — Muon-SF for LoRA params

### What NOT to use with V3
- `--optimizer muon_sf` (with blockwise sharpness) — conflicts with BitNet, 11% worse
- DiffusionBlocks — conflicts with MHC/AttnRes cross-layer connections
- PEFT LoRA (`--lora` without `--bitnet-everywhere`) — can't inject into BitNetLinear

## DiffusionBlocks (`research/diffusion_blocks.py`)

Block-wise training via diffusion interpretation of residual connections
(Sakana AI, ICLR 2026). First successful test on a >1B parameter model.

### How It Works
- Partitions 16 layers into B blocks (default B=4, 4 layers/block)
- Each training step trains ONE block independently as a denoising step
- B× memory reduction: only L/B layers need gradients per step
- AdaLN noise conditioning (shift/scale, zero-init = lossless at start)
- EDM-style loss weighting: w(σ) = (σ² + σ_data²) / (σ·σ_data)²

### V3 Benchmark Results (2026-08-18)
- **Standard training**: 5.74 GB, 8.14s/step (batch=2, seq=512)
- **DiffusionBlocks middle blocks**: 4.98 GB, 1.72-1.82s/step (13% less memory, 4x faster)
- **Batch scaling**: 4× larger batch (8 vs 2) fits in 8.70 GB
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
Same model served by multiple providers — client rotates through them:
- **gpt-oss-120b**: 7 providers (groq, nvidia, cerebras, sambanova, cloudflare, openrouter, huggingface)
- **gpt-oss-20b**: 5 providers (groq, cerebras, cloudflare, openrouter, huggingface)
- **deepseek-r1**: 4 providers (deepseek, nvidia, openrouter, huggingface)
- **deepseek-v3**: 3 providers (deepseek, nvidia, sambanova)
- **glm-4.7**: 3 providers (zai, cerebras, cloudflare)
- **qwen3-32b**: 2 providers (groq, cerebras)
- **qwen3-235b**: 2 providers (openrouter, cerebras)

### NVIDIA NIM Filter
`_nvidia_filter()` excludes NVIDIA's own models (Nemotron etc.) per Eval
Agreement §2.6. Only third-party MIT/Apache models on NIM are allowed.

### Key Features
- **Randomized model-per-goal**: shuffles model pool per goal for max quality diversity
- **Multi-distill**: different teacher models per sample → diverse training data
- **Auto-detects API keys**: only uses providers with credentials in env
- **Verification**: optional `verify_fn(solution, test_cases) -> bool` filters correct solutions
- **ReplayBuffer integration**: `distill_into_buffer()` stores verified results as golden trajectories
- **Temperature randomization**: (0.3, 1.0) range for GRPO group diversity
- **Reasoning traces**: captures CoT from R1/Qwen3 thinking mode for reasoning distillation

### Pipeline
```
DistillationClient → generate verified (prompt, solution) pairs
    ↓
ReplayBuffer.add() → store as golden trajectories
    ↓
GRPOTrainer._inject_golden_replays() → mix into training batches
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
- ✅ Apache 2.0 (Qwen3, gpt-oss, Mistral): "prepare Derivative Works" explicitly allowed
- ✅ MIT (DeepSeek R1, GLM, Phi-4): "distill & commercialize freely"
- ❌ Llama Community: "will not use output to improve any other LLM" (BANNED)
- ❌ Gemma Terms: distilled model becomes "Model Derivative" (BANNED)
- ❌ Gemini TOS: "may not use Services to develop models that compete" (BANNED)
- ❌ NVIDIA-own: Evaluation Agreement §2.6 prohibits (third-party on NIM = OK)
- ❌ OpenAI GPT: "may not use Output to develop models that compete" (BANNED)
- ❌ Anthropic: "may not use Outputs to train models that compete" (BANNED)
- ❌ Grok/xAI: "weights cannot be used to train other models" (BANNED)
- ❌ Cohere: non-commercial use only (BANNED for commercial distillation)

### BANNED providers (do not re-add)
- Google AI Studio — TOS prohibits competing model development
- GitHub Models — RETIRED July 30, 2026
- Hyperbolic/Together/Novita/Chutes — trial credits, not permanent

### Research doc
- `docs/GROQ_DISTILLATION_RESEARCH.md` — full provider analysis, rate limits, pricing
- Tests: `tests/unit/test_distill_client.py` (26 tests)

## Tool-Call + Code-Format Distillation (`research/distillation/distill_tool_calls.py`)

Cold-start SFT data generation for ForgeLM V3. Uses API teachers (gpt-oss-120b,
DeepSeek, etc.) to generate training data with **thinking tokens captured**:
1. **Direct answers**: Simple Qs ("5*10=50") — no tools, just answer from knowledge
2. **Code generation**: task -> Python function + test cases
3. **Tool-call trajectories**: Multi-turn with `[end]` marker — teacher writes code,
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
- **NOT applicable to ForgeLM V3** — calibration shows no saturation region:
  all 16 layers have low/negative cosine similarity (max +0.10, most negative)
  meaning the FFN is actively transforming representations in every layer
- FFN-SkipLLM requires 32+ layer models (LLaMa-2 7B/13B) where middle-layer
  FFNs become redundant (cosine similarity 0.95+)
- Infrastructure kept in codebase for future larger models
- Full analysis: `docs/FFN_RESEARCH.md`

### What DOES work for V3 inference acceleration:
- **Speculative decoding**: EAGLE-3, DSpark, MTP (already in `research/decoding/`)
- **Attention skipping in top layers**: "Attend First, Consolidate Later" (2024)
  — skip attention in top 30% for non-math tasks, keep all FFNs
- **middle_70 architecture**: for V4 — concentrate FFN in middle 70% of layers
  (+1.29% improvement at 1.2B scale per COLM 2025 paper)

## Agentic Distillation Client (`research/distillation/agentic_distill.py`)

Takes the self-play loop process from the former `tool_use_loop.py` (now removed — merged into the unified AZR loop) and applies it to the
distillation model router. Teacher API models call tools (run_script, web_search,
think, calculate, etc.) in an agentic loop to generate rich training trajectories.

**Fine-tuning is DISABLED** — pure data collection. Trajectories can later be
used for SFT or GRPO training of the local ForgeLM model.

### Architecture
```
Teacher API model (gpt-oss, DeepSeek, Qwen, etc.)
    ↓ receives task + tool definitions (OpenAI function-calling format)
    ↓ emits tool calls → we execute (run_script, web_search, think, etc.)
    ↓ results injected back → teacher continues
    ↓ loop until final answer or max_turns
    ↓
AgenticTrajectory (messages, tool_calls, reward, final_answer)
    ↓
save_trajectories() → JSONL for SFT training
```

### Key Components
- **`AgenticDistillClient`** — extends `DistillationClient` with agentic tool-use
- **`run_agentic_task()`** — runs the tool-use loop with a teacher model
- **`run_agentic_batch()`** — runs multiple tasks with multiple teachers per task
- **`generate_tasks()`** — teachers generate their own coding tasks with test cases
- **`save_trajectories()`** — saves full tool-use trajectories as JSONL
- **`compute_reward()`** — reuses the same multi-component reward from the former `tool_use_loop.py` (now legacy; import guarded)
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
- **MTP duplication**: `architecture/mtp.py` deleted (kept `decoding/mtp.py` — independent heads + trainer)
- **serving/ directory**: merged tool-calling support into `inference/forge_server.py`, deleted `serving/server.py`
- **Stale files**: `distillation/test_v3_inference.py`, `scripts/extract_vocab_packs.py`, `sample.csv`, `test.csv`, empty `.jsonl` data files
- **Doc consolidation**: `LLM_Research.md` merged into `research/COMPREHENSIVE_RESEARCH.md` (unique Blackwell/Triton/Gigatoken sections preserved as section 38)

## Novel Discovery Protocol (R&D methodology)

When asked to do R&D or develop novel systems, do NOT just design on paper.
Follow this iterative empirical loop:

1. **Isolated test script first** — write a minimal script with the smallest
   possible input that exercises the core idea (e.g. tiny model, 4 layers,
   100 steps). Run it BEFORE researching. Get a baseline number.
2. **Research + think long** — web_search the topic, read 3-5 papers, think
   hard about what's known vs unknown. Write findings to `.devin/scratchpad.md`.
   **Do the math by hand** for the small case before trusting any number.
3. **Apply novel ideas, compare to documented results** — implement 2-3 novel
   variations in the isolated script. Run them. Compare numbers to documented
   baselines from research AND to the current production baseline on this
   hardware. Most novel ideas will LOSE to known answers — that's expected
   and informative.
4. **Iterate once more before defaulting** — if novel ideas lost, adjust the
   angle (not just parameters). Try a different combination. Only after a
   second failed iteration should you default to the best known answer.
5. **Cross-domain risky combinations** — if still stuck, try combining
   something that barely relates to the topic but might have novel effects.
   E.g. diffusion scheduling ideas applied to optimizer step timing, or
   compression theory applied to gradient sparsity.
6. **Record what worked AND what failed** — failed novel ideas are still
   valuable; document them in scratchpad so the next session doesn't repeat.
   Include the numbers, the hypothesis, and why it likely failed.
7. **Sometimes, leave it to luch** - sometimes, leaving things to chance can be a good modivator. if you need more ideas, take all known and loosly related systems, throw them into a randomizer script, and see what it adds together. ie; ##### + #####. This could help you find new ideas that you wouldn't have thought of otherwise.
8. **Pivot don't quit** — if a hard optimization resists 2 iterations,
   shelf it (documented in scratchpad) and touch up a DIFFERENT area.
   Fresh context often surfaces the missing idea. Return to the hard
   problem in a later round. No area is truly solved, but not every area
   yields on the first session.
9. **Sweep before you ship** — any technique with hyperparameters gets a
   sweep script (5-10 values) before being declared optimal. The paper's
   default is rarely our optimum on RTX 5070 + 1.2B.
10. **Confirm-then-fix in R&D too** — if a test script reveals a bug in
    existing code (not your new code), confirm it and fix it in the same
    session per Directive B. R&D rounds that leave behind unfixed bugs
    are not complete.

Key principles:
- **Isolated scripts over integration** — test the core mechanism on a toy
  problem before touching the real training loop. Faster iteration, clearer
  signal.
- **Numbers over theory** — a 5-line script that runs in 10 seconds beats
  a 500-line design doc. Get a number, then explain it.
- **Lose fast** — most novel ideas don't work. Find out in 30 seconds with
  a toy script, not 30 hours of integration.
- **Cross-domain is where novelty lives** — the best novel discoveries
  combine techniques from fields that don't usually interact.
- **Mixed GPU/CPU is the default assumption** — never design a technique
  that only works if it fits in VRAM. Always have the CPU-offload fallback
  path designed from the start (Directive D).
- **Search before you build** — before writing a new test script, grep for
  an existing one that tests the same mechanism. Adapt it instead of
  starting fresh (Directive E).

## Fast Boot / Cold-Start Optimization (2026-08-19)

R&D round: boot-time (cold load → first token) optimization. 7 variations
tested against baseline. **Production change implemented in `model_loader.py`.**

### Results (forgelm_v3, ForgeLM_V3_Base.safetensors, RTX 5070, bf16)

| Variation | TOTAL (ms) | Speedup | Correct |
|-----------|-----------|---------|---------|
| baseline (traditional) | 11279 | 1.0x | ✓ |
| V1 skip_init | 10190 | 1.1x | ✓ |
| V2 parallel tokenizer | 7185 | 1.6x | ✓ |
| V3 OS prefetch | 6363 | 1.8x | ✓ |
| V4 PrefetchVirtualMemory | 6355 | 1.8x | ✗ (API failed) |
| V5 meta+assign | 4414 | 2.6x | ✓ |
| V6 V1+V2+V3 | 8209 | 1.4x | ✓ |
| **V7 V5+V2+V3 (production)** | **3316** | **3.4x** | **✓** |

### What Worked (V7 = production default)
1. **Meta device init + `load_state_dict(assign=True)`** — 6.1x on arch build
   - Build model on `torch.device("meta")` (zero storage, just shape metadata)
   - `assign=True` directly replaces meta params with state_dict tensors
   - Skips both init kernels AND the `.to(device)` copy
   - Requires: re-tie weights (`model.head.weight = model.embed.weight`),
     re-init RoPE non-persistent buffers (`_reset_non_persistent_buffers()`)
2. **Parallel tokenizer** — `ThreadPoolExecutor` hides 2.7s behind arch build
3. **OS page cache prefetch** — background thread reads 16MB blocks during arch build

### What Failed
- **V4 PrefetchVirtualMemory** — Windows API call failed (Python mmap buffer access)
- **V1 skip_init** — works but slower than V5 (`to_empty` materializes ALL params,
  then `load_state_dict` copies into them; `assign=True` skips the intermediate copy)

### Production API
- `ModelLoader.build_model_fast(..., fast_load=True)` (default) — meta init + assign + prefetch
- `load_default_model(..., fast_load=True)` (default) — also starts parallel tokenizer
- `fast_load=False` — traditional build path (for debugging / no checkpoint)
- `RotaryEmbedding` now stores `self.base`, `self.max_seq_len`, `self.rope_scaling`
  (needed for `_reset_non_persistent_buffers()` after meta init)
- `tokenizer_cache.get_tokenizer()` — fast path via `tokenizers` Rust library (223ms)
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
- **Lazy key instantiation** — 698ms (92% of meta init!) is TITAN/MoD/MHC/AttnRes/
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

Each task can specify independent **boot-time engine parameters** — settings
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
`hadamard_int4` for VRAM efficiency, Task C uses `cuda_graph` acceleration —
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
  single forward pass (GEMV→GEMM shift). 3-5x throughput on RTX 5070.

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

## Environment

- OS: Windows 11, GPU: RTX 5070 12GB
- Python venv: `D:\windsurf\ForgeAI\venv\`
- Key packages: torch, transformers, safetensors, bitsandbytes, pytest
