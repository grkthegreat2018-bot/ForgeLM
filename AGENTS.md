# ForgeAI — Agent Notes

## Agent Operating Directives (READ FIRST)

These directives govern how work is done in ForgeAI. They are non-negotiable
unless the user explicitly overrides them for a specific task.

### A. Model Versioning — Build On The Prior, Never Beside It
- **Every new custom model version MUST be derived from the immediately
  preceding version**, carrying forward all prior keys/architecture as the
  baseline, then adding or replacing only what's new. Current chain:
  `forgelm_v2` (Jamba base) → `forgelm_v12_jamba` (R37 keys).
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
  our numbers come from RTX 5070 + ForgeLM V2 (Jamba-3B).
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
- **Skills**: invoke user-level skills (`log-bug`, `sync-memory`,
  `systemspecs`, `glm-supercharge`) — they live in
  `~/.codeium/windsurf/skills/`, not `.devin/skills/` (which does not
  exist). They encode project-specific workflows.
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
- `forge/model_loader.py` — facade re-exporting `forge/model/` (kv_cache,
  attention_ops, layers, builders, llm/ConfigurableResearchLLM, loader/ModelLoader)
- `forge/checkpoint_io.py` — save/load_checkpoint (safetensors + .pt) and
  `load_safetensors_pipelined()` — the fast CUDA weight loader: N parallel
  file reads into per-thread pinned buffers + per-tensor async H2D on a
  side stream (0.53s vs 3.3s per-tensor safetensors on the 6.4GB V2 ckpt;
  fastsafetensors is broken on this Windows box — missing cudart DLL — so
  pipelined is the de-facto fast path; both have safetensors fallback).
  Wired into `ModelLoader._load_safetensors_mmap/_load_sharded_safetensors`
  and `load_checkpoint`. `from_checkpoint` also overlaps the tokenizer
  load on a daemon thread. Bench: `scripts/bench_weight_io.py` (I/O only),
  `scripts/bench_load_time.py` (full engine load phases — V2: ~2.4s total).
- `forge/engine/forge_engine.py` — ForgeEngine (inference engine; core init +
  8 mixins: engine_checkpoints, engine_activation, engine_generation,
  engine_diagnostics, engine_merging, engine_lora, engine_lifecycle,
  engine_sessions; shared helpers in engine_common.py)
- `forge/engine/decoding.py` — Decoding strategies (standard, speculative
  family, `dola` self-contrastive R50-2, `uno`; DRY n-gram penalty R50-1
  in the shared `_sample_from_logits` chain + all `engine_generation`
  paths, `dry_multiplier`/`dry_base`/`dry_allowed_length`/
  `dry_penalty_last_n` params, 0=off)
- `forge/engine/gated.py` — ForgeGate three-probe gated generation
  (route/doom/convergence); `ForgeEngine.load_gate_probes()` +
  `generate_gated()`; probe bundle `research/checkpoints/gate_probes.pt`
  (~44KB, version-guarded). VRAM overhead ~0 (logistic heads on hidden
  states the model already computes). Trained on the gate_r 1450-label
  self-verified corpus; retrain pipeline documented in
  `.devin/scratchpad.md` (R&D: ForgeGate probes). Held-out eval on
  ForgeLM V2: +16.4pts acc at −34% tokens vs always-think; numbers are
  corpus-dependent — recalibrate probes on new domains/checkpoints
  before trusting thresholds. GUI wiring:
  `EngineService._load_blocking` auto-loads the bundle when present;
  `chat_loop._route_p_easy` scores the route head once per turn on a
  TOOLS-FREE render of the conversation (the ~17-schema `<tools>` block
  collapses h_mean and drags p_easy ~0.25 below threshold — "Hello":
  0.355 tools-free vs 0.098 with tools; probe was trained tools-free)
  and switches to a closed-think render (`<think></think>` + "Answer:"
  anchor — bare </think> leaves the model musing) when p_easy ≥ 0.2.
  Bare greetings/acks bypass the probe entirely (`_is_trivial_turn`,
  gate_r has no chit-chat class). Emits a `gate` SSE event the Chat page
  shows as a `gate → mode (p=…)` chip. Routing + conv exit run ONLY on
  fresh user turns (last conv msg is `user`) — tool-continuation rounds
  go straight to think+budget: both probes are OOD on synthesis turns
  (observed: conv-exit forced `Answer:` → model restated its plan and
  re-called the same tool 5×). Think turns get TWO exits: the
  conv probe early-exit (`_conv_exit_observer` scores per-step hidden
  states via `generate_stream(hidden_observer=)`, K=2 consecutive
  >0.75 past 32 tokens → inject force-answer; emits a `conv-exit` gate
  event) AND the hard cap backstop (`_think_cap_processor` injects the
  gated.py force-answer suffix after `think_budget` think tokens,
  default 160, per-request via `ChatSendRequest.think_budget` /
  Chat settings "Think budget", 0=no forced exit) — plus a `<think>`
  (id 541) ban in generated text in every mode (a generated <think> can
  only re-open a reasoning pass — observed: model emitted `Answer:` then
  re-opened <think> and re-looped). Direct path passes `budget=None`
  (ban only). Two output guards: repeat-call detection (`_call_sig` —
  an identical name+args call drops tool defs so the next round must
  synthesize) and `_strip_direct_musing` (direct-path text before a
  stray `</think>` is reasoning voice — dropped; pre-call musing on
  direct tool turns is dropped entirely). Threshold calibrated via
  `scripts/bench_gate_route.py` — rerun it when the probes are
  retrained or the chat template changes.
- `forge/engine/decide.py` — SystemOneEvaluator: TypeSafe AI "System One"-
  compatible typed decisions (noul/choice/score questions → TypeSafe-style
  probability answers) via candidate-continuation scoring — single forward
  passes, no generation. Reasoning-model aware: primes
  `assistant\n<think>\n</think>\nAnswer:` so answer tokens land at the
  read position. `ForgeEngine.decide(state, questions)` wraps it.
  Server exposes `POST /v1/systemone` (typesafe_sdk drop-in: point
  `base_url` at Forge; `/v1/models` sniffs `X-TypeSafe-SDK` header and
  returns the TypeSafe `{"models": [...]}` shape). Without a scorer,
  probabilities are raw LM softmax — consistent within a question but
  NOT calibrated.
- `forge/engine/decision_head.py` — Tier-1 `DecisionScorer`: linear
  verifier head on last-token hidden states, `s(q,c) = w·h(prompt+" "+cand)`
  softmaxed per question; covers noul/choice/score with one head. Also
  `ProcessRewardHead` (R50-3): sigmoid step verifier on step-end hidden
  states — `fit_prm(model, tok, device, dataset)` trains on per-step or
  outcome-broadcast labels (BCE, frozen base), `score_steps` returns
  P(step correct); ~10KB weights, feeds GRPO advantage shaping.
  Train via `fit_decision_scorer(model, tok, device, dataset)` —
  dataset rows `(state, question_spec, correct_candidate_idx)`, group
  softmax CE (proper scoring rule) + LBFGS temperature scaling; ~10 KB
  weights, `scorer.save()`/`DecisionScorer.load()`. Attach with
  `engine.load_decision_scorer(path)` or `decide(..., scorer=...)`.
  Real-model check (340 mixed templated examples): ECE 0.067 vs 0.141
  raw at equal accuracy — calibrated-style, not production-calibrated;
  validate on non-templated data before confidence gating.
- `forge/keys/` — KeyStack architecture keys (25 canonical + `kda` R49-2
  side-path: `use_kda` config flag, gate=0 bit-exact, KDAKey BI port).
  `forge/keys/safety.py` is the production `safe_apply`/rollback harness
  (used by pit/lerope/attn_residual/mhc keys `safe=True` paths).
- `forge/quant/` — Quantization implementations
- `forge/decoding/` — Decoding implementations
- `forge/training/` — Training runners + optimizers; `training/data/`
  holds the dataset pipeline modules (`efficient_pipeline`,
  `parquet_dataset`, `curriculum_augment`, data-prep one-shots) — it is
  tracked source, exempted from the `data/` gitignore rule.
- `forge/evolution/` — Evolutionary optimizer (ForgeEvolve); CLI: `python -m forge.evolution --domain <name> --steps <N>` (use `--list-domains` to see all domains). Domain specs + run/focus profiles live in `forge/evolution/configs/` (canonical; the legacy `tests/evolution/configs/` copy was removed — the tests/evolution harness scripts now point at the canonical dir).
- `forge/self_play/` — Self-play + discovery (`infinite_loop.py` is the RSI
  loop; `live_status.py` writes live telemetry — status.json, heartbeat.json
  with progress-coupled stall detection, events.jsonl — to
  `research/checkpoints/self_play/` for the GUI Self-Play page + CLI polling;
  `--status-dir`/`--no-live-status` flags on the loop)
- `forge_gui/` — Qt-free domain layer (`api/` only; PySide6 shell removed).
  Web tools live in `forge_gui/api/web_tools.py` (defs + dispatch) backed by
  stdlib-only `forge/web_primitives.py`: `web_search` (DuckDuckGo HTML),
  `news_search` (Google News RSS — use for "the news"/current events; DDG
  returns portal homepages for those queries; empty query → top headlines),
  `web_fetch` (http/https only), `wikipedia_search`, `arxiv_search`. All
  keyless GETs; never gated by safety checks, agent approval modes, or
  read-only chat mode. `parse_ddg_html` drops DDG ad/tracker links (y.js
  redirectors carry u3= not uddg= and can't be unwrapped) and `ddg_search`
  falls back to Google News RSS when nothing parseable comes back. The
  agent-loop fallback harness also wires `WebTools` so web tools exist even
  if the deps harness factory fails.
- `forge_gui_server/` — FastAPI GUI backend (REST `/api` + WebSocket `/ws`)
  serving the React UI; ports `forge_gui/api/` to plain async services.
  Launch: `python -m forge_gui_server` (desktop window via pywebview),
  `--browser`, `--no-window`, or `--dev` (API only, Vite dev on :5173).
  EngineService preloads the resident model in the background at startup
  (`FORGE_GUI_NO_PRELOAD=1` to disable, `FORGE_GUI_PRELOAD` /
  `FORGE_GUI_PRELOAD_CONFIG` to pick a different checkpoint/config);
  a different-model request mid-load is queued (not dropped), a bare
  re-request for the resident model is a no-op, and unload() uses
  sleep(level=2) — no wasted GPU→CPU copy on discard.
- `forge_ui/` — React 19 + TS + Vite + Tailwind v4 frontend
  (`npm run dev` / `npm run build`; design tokens in `src/index.css`).
  Shared chat/agent primitives live in `src/components/chat/` (Markdown,
  blocks=ThinkCard/ToolActivityCard/SegmentedBody/UserBubble/CopyBtn,
  Composer, ScrollFeed) — reuse them instead of duplicating message cards.
  Backend endpoints the pages rely on: `POST /api/chats/{id}/truncate`
  (regenerate/edit-and-resend), `POST /api/chats/import` (paste a
  transcript → new conversation; `parse_transcript` in
  `forge_gui/api/chat_store.py` handles JSON/ChatML/role-marked/plain
  alternating-paragraph formats), `DELETE /api/agent/runs/{id}`,
  `GET /api/agent/tools` (real harness tool defs for the Agent picker —
  do not hardcode tool lists in the UI).
- `tests/unit/` — Unit tests (CPU-runnable where possible)
- `tests/integration/` — Integration tests (GPU required)
- `docs/` — Documentation + R&D round notes
- `scripts/` — Standalone scripts
- `research/` — Legacy paths (tokenizer cache, checkpoints only)

## Current Architecture: ForgeLM V2 Jamba-3B (SOLE BASE)

**Base model**: ForgeLM V2 — lossless port of AI21 Jamba-Reasoning-3B.
- 28-layer hybrid: 26 Mamba-2 SSM + 2 GQA attention layers
- d_model=2560, 20 heads, 1 KV head (MQA), vocab=65536, max_seq_len=262144
- No RoPE (Mamba handles position), untied embeddings, ~3.2B params
- bf16 checkpoint ~6.1 GB; server VRAM budget 8.0 GB

**Base checkpoint**: `research/checkpoints/ForgeLM_V2.safetensors`
**Tokenizer**: `research/checkpoints/forgelm_v2_tokenizer/`
**Default config**: `forgelm_v2` (load_default_model() defaults to this)
**Path constant**: `research.paths.V2_CHECKPOINT`, `research.paths.FORGE_TOKENIZER_DIR`
**Server model ID**: `forgelm-v2-jamba` (DEFAULT_MODELS in forge_server.py)

All prior checkpoints (Jamba_Reasoning_3B source port, ForgeLM_V2_Light*,
evolution R30/R31 artifacts, LFM2.5 GGUFs) were deleted 2026-09 — ForgeLM V2
Jamba is the sole base model.

## Config Presets

Config presets in `forge/config.py`:
- `forgelm_v2` — **ForgeLM V2 Jamba-3B: THE DEFAULT AND SOLE BASE.** 28 layers
  (26 Mamba + 2 GQA), d_model=2560, ~3.2B params.
- `forgelm_v12_jamba` — R37 research preset derived from `forgelm_v2`
  (parent declared; Mamba-3, Kronecker embed, PIT, OutRo, ForgeHybrid — all
  zero/identity-init for lossless warm start).
- `forgelm_tiny` — 4-layer tiny conv+attention model for fast tests
  (no checkpoint, config-only).
- `gen_model_tiny` — tiny generator config for evolution experiments.
- `qwen25_05b` — HF-compat preset for Qwen2.5-0.5B checkpoint detection.
- `qwen3_4b` — HF-compat preset for the Qwen3-4B-Instruct-2507 family
  (e.g. `0xA50C1A1/Qwen3-4B-Nymphaea-RP`). First preset to use the
  explicit `ModelConfig.head_dim` field (128; `d_model/n_heads`=80 would
  be wrong). GQA 32Q/8KV, QK-norm, no QKV bias, RoPE 5M, 256K ctx, tied
  embed. Auto-detected by `_detect_config_from_header` on
  (vocab=151936, d_model=2560, n_layers=36).

All legacy presets (`forgelm_v2_light`, `forgelm_v2_pro`, `forgelm_v12`,
`forgelm_v10_1.2b`, `forgelm_v11_3b_vl`, `lfm25_tiny`, `lfm25_1.2b`, V3–V9)
have been DELETED.

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
