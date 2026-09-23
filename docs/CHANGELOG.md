# ForgeAI Changelog

Historical notes, R&D round documentation, and refactor retrospectives.
Extracted from AGENTS.md on 2026-09-09 (critique F1).

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

#### Agent harness 2026-09-02: Web tools exposed to the model
- **Problem**: `ToolHarness` exposed coding/memory/LoRA/MCP/backup/sub-agent/
  time/library tools but NO web tools — the agent could not do real-time
  research. `engine_tools.py` had Tavily/Exa/Firecrawl web tools but those
  need API keys; `research/self_play/discovery/discovery_tools.py` had a
  proven KEYLESS DuckDuckGo-HTML/Wikipedia/arXiv/URL-fetch implementation
  (stdlib urllib, GET-only) but it was private to the self-play subsystem.
- **New module** `forge_gui/api/web_tools.py`: `WebTools` class +
  `web_tool_defs()` exposing 4 tools to the model: `web_search` (DuckDuckGo
  HTML, no key), `web_fetch` (http(s)-only GET, HTML→text strip, truncation),
  `wikipedia_search` (REST API), `arxiv_search` (Atom API). Primitives reuse
  the proven discovery_tools regex/parse logic (duplicated, not imported —
  discovery versions are `_`-private and coupled to the self-play DB emit
  pattern; `forge_gui/api/` must not depend on `research/self_play/`).
  Safety: `_is_safe_url()` rejects `javascript:`/`file:`/`data:`/`ftp:`
  before any network call; all requests GET-only with a hard 12s timeout;
  output capped (`MAX_FETCH_CHARS=4000`, snippets 400 chars) for the 12GB
  KV-cache budget; `n` clamped to [1,10], `max_chars` to [200,8000].
- **Wiring**: `ToolHarness.__init__` gained `web_tools: Optional[WebTools]`;
  `tool_defs()` extends with `web_tool_defs()`; `execute()` dispatches
  `WebTools.NAMES` to `web_tools.execute()` (normalizes falsy `error` key
  so the harness `"error" in result` ok-check works); `chat_tool_defs()`
  includes web tools (read-only GET → safe for chat). `app.py` constructs
  `WebTools(enabled=True)` and passes it to the shared harness. `agent.py`
  `HARNESS_EXTRA_TOOLS` now includes the 4 web tool names → always enabled
  (read-only, no approval prompt, not in `SIDE_EFFECT_TOOLS`).
- **Tests**: `tests/unit/test_gui_web_tools.py` (35) — defs shape, URL
  scheme validation (parametrized), DDG redirect unwrap, HTML/JSON/XML
  parsing, network-error handling, `WebTools.execute` dispatch + n/max_chars
  clamping + disabled flag, ToolHarness integration (defs include web,
  dispatch works, error marked not-ok, chat_defs include web, read-only
  mode keeps web). All network mocked via `unittest.mock.patch` on
  `urlopen` — no real HTTP, fast & deterministic. Suite: 290 GUI/tool/web
  tests pass.

#### Boot 2026-09-05: R37 GUI boot optimization — no more UI-thread freeze
- **Problem**: GUI window appeared then froze for 1-3 s on first refresh
  tick. Root cause: `GpuMonitor.snapshot()` calls `import torch` +
  `torch.cuda.*` queries **on the UI thread** — the 1-3 s CUDA runtime
  init blocked the event loop. The fast timer (500 ms) fired
  `DashboardPage.refresh()` → `gpu.snapshot()` before the window
  manager finished compositing, producing the "not responding" lockup.
  Additionally, all 16 shared backends were constructed eagerly in
  `MainWindow.__init__` even though only 6 are needed by the Dashboard
  (the first visible page).
- **Fix 1 — Background GPU poller** (`gpu_monitor.py`): new `GpuPoller`
  class (daemon thread, not QThread — no Qt dependency needed) calls
  `gpu.snapshot()` every 2 s in the background and updates a
  thread-safe cache (`_cached: GpuStats` + `_cache_lock`). New
  `GpuMonitor.cached_snapshot()` returns the cached stats instantly
  (zeroed `GpuStats` if no poll has completed yet). Dashboard, Compute,
  and `_refresh_slow` all call `cached_snapshot()` instead of
  `snapshot()` — the UI thread **never** imports torch or queries CUDA.
- **Fix 2 — Lazy backends** (`app.py`): 12 of 16 backends converted
  from eager attributes to lazy `@property` constructs: `chat_store`,
  `lora_mgr`, `lorebook`, `lora_harness`, `mcp_manager`, `lora_training`,
  `backup_manager`, `sub_agent_manager`, `time_manager`,
  `library_manager`, `web_tools`, `tool_harness`. Each constructs on
  first property access (triggered when the user visits the page that
  needs it). The 6 eager backends (`gpu`, `status_reader`, `models_index`,
  `log_tailer`, `proc_mgr`, `engine_runtime`) are either needed by the
  Dashboard or are cheap QObjects with no disk I/O.
- **Fix 3 — Delayed timer start** (`app.py`): the fast (500 ms) and
  slow (2000 ms) refresh timers are no longer started in `__init__`.
  Instead, `QTimer.singleShot(800, self._start_timers)` starts them
  after the window has been visible for ~800 ms, ensuring the first
  paint is never interrupted by a refresh tick.
- **Results**: MainWindow construction dropped from ~2-4 s (16 eager
  backends + torch CUDA init on first refresh) to **0.27 s** (6 eager
  backends, no torch on UI thread). `cached_snapshot()` returns in
  0.0000 s. Lazy backend construction (e.g. `chat_store`) is 0.006 s
  on first access. All 351 GUI-related tests pass.

#### GUI 2026-09-01: ForgeAI Control Center v2 (LM Studio + Agent + Train platform)
- **New shared backends** (`forge_gui/api/`):
  - `chat_store.py` — pure-python conversation persistence
    (`data/chats/conversations.json`) with per-message ratings
    (good/bad/toggle). `export_training_data()` writes good-rated turns as
    sft_train-compatible JSONL (`{"messages": [...]}` per line) to
    `data/sft/forge_chats_*.jsonl`. Export format is covered by a test that
    round-trips through `sft_train.load_examples`.
  - `engine_runtime.py` — ONE resident `ForgeEngine` shared by Chat/Agent/
    Engine pages. QThread loader, states idle→loading→ready/error,
    `acquire()` lease serializes generation across threads (12GB VRAM:
    only one model resident). `unload()` sleeps + empties CUDA cache.
  - `agent_tools.py` — sandboxed coding tools jailed to a workspace root
    (absolute paths + `..` escapes rejected), command allowlist
    (python/pip/pytest/git/...), timeouts, 4k output cap. Tools:
    list_dir/read_file/write_file/append_file/delete_file/run_python/
    run_cmd/grep_project. A returned `{"error": ...}` dict marks the call
    not-ok (execute() checks).
  - `agent_runner.py` — QThread agentic loop: qwen_render_messages →
    engine.generate → qwen_parse_tool_calls → ToolSandbox → feed results
    back, up to N rounds. Emits per-step Qt signals; optional approval gate
    (threading.Event) before side-effecting tools.
- **New pages** (`forge_gui/pages/`): `chat.py` (full rewrite: multi-chat
  sidebar, local-engine streaming via `generate_stream` OR OpenAI endpoint,
  system prompt, temp/top_p/top_k/max-tok, per-reply 👍/👎 rating → SFT
  export), `agent.py` (agentic coding: workspace picker, live round trace
  with tool-call cards, approval dialogs, tool toggles), `engine.py`
  (load/unload checkpoint, stats, benchmark/bottleneck/diagnose,
  sleep/wake), `finetune.py` (dataset multi-select from data/sft + data/,
  hyperparam form mirroring sft_train.py defaults, launches
  `research/training/runners/sft_train.py` via ProcessManager → Tasks page).
- **Wiring** (`app.py`): 13 pages — Dashboard, Chat Studio, Agent, Engine,
  Fine-Tune, Self-Play, Training Live, Generations, Models, Launch, Tasks,
  Compute, Logs. `EngineRuntime` + `ChatStore` created in MainWindow and
  passed to pages. Sidebar brand now shows `ForgeAI_Icon.png` pixmap.
- **Theme** (`theme.py`): added QSS for chat bubbles (user/assistant/
  system), rating buttons, agent round blocks + tool cards, engine console
  kv rows, QListWidget lists, tabs, checkboxes.
- **Tests**: `tests/unit/test_gui_chat_store.py` (8) +
  `tests/unit/test_gui_agent_tools.py` (14) — all pure-python, no Qt.
  Suite: 1247 passed, 0 failed. Smoke: `QT_QPA_PLATFORM=offscreen
  venv\Scripts\python.exe forge_gui\_smoke_test.py` (13 pages construct/
  switch/refresh); real-window check: `forge_gui\_launch_test.py`.

#### Engine fixes 2026-09-01: BOS tokenization + fused QK-RoPE + compile opt-out
Root-caused "model generates garbage" reported from the GUI. Chain of 3 bugs:
- **BOS stripped from prompts** (`forge_engine.py` — THE garbage-output bug):
  `_generate_impl` / `_tokenize` used `add_special_tokens=False`, but
  LFM2.5/V10 was trained with BOS `<|startoftext|>` (id 1). Without it the
  model repeats the last prompt token ("is is is…"). Fixed: special tokens
  now added in `_generate_impl`, `_tokenize` (default True), and the
  `generate_raw`/`generate_stream` call sites. Verified: raw greedy
  "The capital of France is" → " Paris. It is the most populous…", chat
  template → "Paris", code stream → real fibonacci.
- **fused_qk_norm_rope_cache.py shape/type bugs** (eager fallback was dead):
  (1) wrapper passed `position_ids` into the `eps` slot of
  `fused_qk_norm_rope` → triton kernel got a tensor as eps
  ("pointer<int64> + float32" compile error) and the py fallback did
  `.add(eps)` with a (1,T) tensor → shape corruption; (2) wrapper fallback
  `_py_qk_norm_rope` used half-dim NeoX math against the FULL-dim
  (duplicated-halves) `cos_cached` → 32-vs-64 dim error. Fixed: slice
  cos/sin to `[cache_position : cache_position+T]` inside
  `fused_qk_norm_rope_cache`, call `fused_qk_norm_rope(q, k, qw, kw,
  cos_chunk, sin_chunk)` (no position_ids), and rewrote `_py_qk_norm_rope`
  to full-dim rotate_half. Fused output now bit-matches standard path.
- **torch.compile broken on this stack** (triton/SM120): inductor
  mis-compiles the fused kernel → InductorError on every forward when
  compiled. `from_checkpoint` auto-activates the optimal preset
  (use_compile=True) so it crashed even when later `activate(use_compile=
  False)`. Fix: `_auto_activate_optimal` honors `FORGE_NO_COMPILE=1` env
  var; GUI fast-load sets it before importing the engine. Fast load
  ≈ 15-45s (was 161s with compile). Compile stays broken on SM120 until
  the triton kernel is fixed — GUI checkbox warns.
- **GUI logging**: `run()` now installs `logs/gui.log` (rotating, 2MB×3)
  + console handler, and forces UTF-8 stdio (`PYTHONUTF8=1` +
  `reconfigure`) — cp1252 consoles raised UnicodeEncodeError on engine
  prints (→/·) which silently skipped engine warmup.
- Diagnostics kept in `.devin/tmp/` (bisect_gen, diag_tok_logits,
  verify_bos_fix, verify_fused). Key lesson: two checkpoints (base + R30)
  produced IDENTICAL garbage → weights were fine, the decode path wasn't;
  always bisect features before blaming the checkpoint.

#### GUI memory fix 2026-09-02: one resident engine everywhere
Root-caused "GUI uses more VRAM than script boots": the GUI had THREE
independent engine-loading paths — EngineRuntime (Engine/Chat/Agent,
fast-load) plus `model_boot.py` (Models page) and `generation.py`
(Generations page), which each spawned their OWN engine with full
auto-activation (compile + CUDA graphs). Two resident engines + graph
pools = 12.82GB / 0 free → KV cap collapsed to 64 tokens → second load
fell into AirLLM meta-device streaming. Fixes:
- `model_boot.py` + `generation.py` now borrow the SHARED EngineRuntime
  (Models boot reloads only if checkpoint/config differs; Generations
  shows the resident model and errors with a hint if nothing is loaded).
  app.py passes runtime into ModelsPage/GenerationsPage.
- `EngineRuntime._LoadWorker`: VRAM pre-flight (free < checkpoint×2.5 →
  fail fast with "close other GPU apps / GUI instances" instead of the
  silent AirLLM meta-device fallback).
- LoRA adapters (`*lora*`) filtered out of Engine/Chat/Models boot combos
  — a 43MB LoRA file was loadable as a "base model" and produced the
  AirLLM meta-device mess in the field.
- `EngineRuntime.shutdown()` + MainWindow.closeEvent waits for in-flight
  loads (fixes "QThread: Destroyed while thread is still running").
- stdout/stderr teed into `logs/gui.log` (`_Tee` in app.py) — engine
  output is print()-based and never reached the logging file before.
- **triton kernel fixed** (`fused_rope_qknorm.py`): `tl.gather(x, idx)`
  requires `axis` on this triton (kernel never worked — every real
  generation used the py fallback). Replaced with gather-free shifted
  re-load + rotated weight + negate mask; verified vs py fallback
  (≤0.031 bf16 noise) standalone AND under torch.compile. NOTE: a test
  bug cost an iteration — build RoPE tables with `freqs.cos()`, not raw
  angles (cos[0] must be 1.0, not 0.0).
- **Compile mode still blocked** (different root): IRIFP4Linear
  `_dequantize_weight(cache=True)` caches a CUDA-graph-pool output →
  "tensor output of CUDAGraphs overwritten by subsequent run". Proper fix
  = pre-materialize ~2.4GB bf16 weights outside compiled region (defeats
  FP4 VRAM saving) — R&D item, not a quick fix. Fast load stays the
  default; Engine checkbox tooltip names the blocker.



#### R&D round 50 (2026-09-19): Missing-feature batch � DRY, DoLa, filler-KV, PRM, depth upscale + R49-2 KDA key
Survey of `indie_llm_research_scratchpad.md` vs the codebase found five gaps
worth implementing; all shipped CPU-tested this round plus the R49-2 KDA key:

- **R50-1 DRY repetition penalty** (`engine_common._dry_penalties`,
  llama.cpp `sampler_dry` semantics): longest repeated-suffix scan over the
  trailing `dry_penalty_last_n` tokens; continuations of repeats longer
  than `dry_allowed_length` get `dry_multiplier * dry_base**excess`
  subtracted in logit space. Wired end-to-end: `StandardDecoding`,
  `_sample_from_logits`, `_sample_next_token`, `_decode_with_kv`,
  `generate_raw`, `generate_stream`, `_finish_to_stop`, prefix-cache decode,
  `_validate_generation_params`, `forge_server` request models +
  `model_registry` propagation. Default `dry_multiplier=0` ? disabled,
  fully backward compatible.
- **R50-2 DoLa self-contrastive decoding** (`DoLaDecoding` in
  `decoding.py`, `build_decoding("dola")`): `log_softmax(final) -
  log_softmax(early)` restricted to the final distribution's
  `candidate_top_k` set; dynamic premature-layer selection by max JSD over
  `{n/4, n/2, 3n/4}` (or fixed `early_layer`). Uses the existing
  `return_hidden_states` path � no auxiliary model, no training. On the
  hybrid, contrasting a Mamba block also isolates the attention layers'
  recall contribution.
- **R50-3 stepwise Process Reward Model** (`ProcessRewardHead` +
  `fit_prm`/`score_steps` in `decision_head.py`): sigmoid step verifier on
  step-end hidden states; BCE on per-step labels or outcome-broadcast
  (Math-Shepherd weak supervision); base model frozen � external probe,
  ~10KB weights, `save`/`load` with version guard. Metrics: val accuracy
  + ECE. Feeds future GRPO advantage shaping / ForgeGate step routing.
- **R50-4 filler-token KV eviction** (`engine/kv/filler_kv.py` +
  `FillerKVCacheStrategy`, `build_kv_cache("filler")`, engine_activation
  fallback map): function words/punctuation flagged via `filler_ids`
  (`filler_token_ids(tokenizer)` builds the default English set) or
  `filler_pred`; evicts unprotected fillers first, then falls back to
  SnapKV-style attention-score eviction; sink prefix + observation window
  always kept; `filler_keep_ratio` retains top-scored fillers. Callers
  that never pass token_ids degrade to pure score eviction.
- **R50-5 depth upscaling / passthrough merge** (`research/merge_models.py`
  `depth_upscale` + `parse_layer_map` + `depth_layer_types`, CLI
  `--method depth --layers "0-15,8-23"`): renumbers `blocks.{i}.*` per a
  layer map, preserves non-block tensors, writes a `.depth.json` sidecar
  (layer_map + layer_types) for the loader. NOT lossless � duplicated
  Mamba/attention blocks change the function; warm start for continued
  training (SOLAR/mergekit semantics). Typed-layer caveat documented:
  the map only makes sense when duplicated source blocks share the type
  expected at their destination slot.
- **R49-2 KDA key** (`forge/keys/attention/kda_key.py` � Kimi Delta
  Attention / Gated DeltaNet, arXiv:2510.26692): `KDALayer` side-path with
  per-key-dim decay `S_t = Diag(a)S + �k(v - (Diag(a)S)?k)?`, GDN-style
  init (A_log=log U(0.01,16), dt_bias=softplus?� U(1e-3,0.1), depthwise
  causal conv k=4), fp32 sequential scan, sigmoid output gate, scalar
  `gate=0` ? bit-exact vs baseline. `KDAKey` (KeyClass.BI): deterministic
  seeded port adds `kda.*` per layer, reverse strips (lossless iff
  gate�0), `convert_model_state` for whole checkpoints. Config:
  `use_kda`, `kda_n_heads`, `kda_head_dim`, `kda_beta_gt1` (N3, off �
  destabilization risk), `kda_decay_floor`. Recurrent state +
  conv-boundary snapshots wired into the new-sequence reset, prefill
  snapshot, and prefix-cache save/restore paths � prefix hits continue
  the recurrence instead of restarting at zero state. VRAM: ~25M
  params/layer at 2560/20H (bf16 ~50MB/layer); fixed (H�d_k�d_v) state
  ~1.3MB fp32, NO KV cache � the future 3:1 hybrid cuts KV ~75%.
  Chunked-scan kernel remains a follow-up (naive scan is CPU-test speed).

Bug fixes confirmed+fixed in-session (see BUG_LOG):
- `SnapKVCache._evict`/`FillerKVCache._evict` bool-mask overflow: keep
  mask sized `total` was applied to the overgrown buffer axis � masked to
  `:total` slots in both.
- `_min_k_filter`/`_min_k_filter_logits`: dead `weighted_diffs >
  sensitivity*max_decay` expression � sensitivity ignored; now truncates
  at the rightmost exceeding cliff (argmax fallback).
- `fit_prm`: harvested features/labels were inference-mode tensors ?
  BCE loss couldn't backprop; post-harvest re-clone (fit_decision_scorer
  convention).

Tests: `tests/unit/test_r49_kda.py` (17) + `tests/unit/test_r50_rd_features.py`
(24) � hand-computed delta-rule check, prefill/decode parity, prefix-restore
consistency, block-level `torch.equal` bit-exactness, DRY suffix math,
DoLa JSD/contrast/top-k mask, filler eviction ordering, depth map parsing +
duplication + typed-layer validation, PRM learnability/ECE/save-load.
Full unit suite: **2607 passed**.
  - Smoke-verified all R50 features on the real ForgeLM V2 checkpoint
    (3.2B bf16 CUDA, `scripts/smoke_r50_gpu.py`, 9/9): DRY breaks a real
    repetition loop; DoLa generates richer continuations (after fixing an
    inference-tensor autograd crash in `_contrast_logits` � now
    `@torch.no_grad()`, BUG_LOG'd); KDA `convert_model_state` ports all
    28 blocks with strict=True load and max|dlogit| = 0.0 vs baseline.
    CPU smoke `scripts/smoke_r50_features.py` 23/23 on real tokenizer +
    tiny model (filler eviction 38?23 tokens, PRM step scores, depth
    upscale strict load + typed-layer negative check).

#### R&D round 51 (2026-09-22): FLUX � sparse online associative-memory LM

Non-transformer experiment (`forge/model/flux.py`): a sub-1GB, CPU-native
chat model where the "weights" are addressable memory cells written
live, not dense matrices trained offline.

- **R51-1 FluxLM architecture**: vocabulary = the existing tokenizer;
  context = a *sketch* � rolling polynomial suffix-hashes at orders
  1..32, an episodic ring buffer (tokens + prefix-hash ring, 1M-token
  capacity) for seed-match + backward-extension longest-suffix
  retrieval (effectively unbounded order), an EMA topic hypervector
  (Hebbian `A[x] += lr*c`, cosine readout on candidates), and a recency
  table.  No positional encoding ? no max context, no training needed
  for longer contexts.  Prediction is a geometric mixture over channels
  (`logit = log_uni + S w_c�(log p_c - log uni)`); channel weights w_c
  adapt online via Hedge multiplicative weights � meta-learning on top
  of the memory writes.
- **R51-2 Live learning + provenance**: every mutation is journaled
  (`FluxJournal`: op, channel, key, token, delta, tag) � "what is
  stored in which weights" is auditable (`audit()`, `writes_since()`)
  and revertible (`revert_since`, `revert_tag` � suffix-based un-learn;
  `snapshot`/`load` for coarse rollback).  Topic-row deltas stored fp16
  in a vec-delta ring; EMA inverted exactly on undo.  Decay sweeps are
  the only non-exact writes (clamped, counted as `inexact`).
- **R51-3 Engine support**: `ForgeEngine.from_flux(config=, checkpoint=,
  tokenizer_path=, device=)` explicit path (skips checkpoint detection,
  quant, KV activation); `unbounded_context=True` flag on the model makes
  `_generate_impl` skip tokenizer truncation.  Model honors the
  `(ids, past_key_values=None, use_cache) -> (logits, loss, None)`
  contract so StandardDecoding/DRY/min-p/top-k all work unchanged.
  Session learning is tagged `"live"` ? `revert_tag("live")` forgets
  everything learned in-session.
- **R51-5 GPU/mixed execution + generativity**: `FluxConfig.device="cuda"`
  (or `from_flux(device="cuda")`) puts the dense readout - topic matrix
  A, hypervector table R, context vector c, `proto` fingerprint, logit
  assembly - on GPU (65k x 192 matvec; measured 84 tok/s gen+learn,
  72MB VRAM) while sparse tables/journal stay host-resident (hash lookups
  are pointer-chasing - wrong for GPU).  `FluxLM.to_device()` migrates
  live.  New hedge channel `sem`: expected-next-token fingerprint
  (`proto` = EMA of follower A-rows) cosine-scored against learned token
  fingerprints - on CUDA the full vocab each step - so tokens
  distributionally similar to memorized followers gain lift without ever
  appearing in that context (recombination vs verbatim recall).
  `fast_ingest` skips deep/topic/sem rewards during bulk corpus ingest.
  Snapshots remap channel weights by name (checkpoint survives channel
  list changes).
- **R51-6 Anti-degeneration + selective distillation**: new hedged
  `fatigue` channel penalizes tokens emitted in the last
  `fatigue_span` (16) positions — intrinsic "you already said that";
  killed the greedy `<|im_start|>`/newline loops (hedge correctly gives
  it ~0.95 weight on non-repetitive text).  Hedge `w` revert is now
  exact: pre-update w vector snapshotted fp64 in `journal.w_ring`
  (multiplicative inverse was lossy through the min/max clip — revert
  drift was 0.89, now 0).  Gap distillation: `track_gaps` records
  positions where the probe-order table had < `gap_min_total` followers;
  `rescan_gaps()` rebuilds them for pre-tracking snapshots;
  `distill_from(score_fn)` batches 32-token gap windows through a
  teacher and writes top-k probs as fractional counts under
  `tag="distill"` (journaled ws/we block markers → `revert_tag` /
  `revert_writes` remove them without touching stream state).
  `scripts/distill_flux.py` drives ForgeLM V2 as the teacher.
  Measured: V2 scores 86 gaps/s — 5,000 gaps in 56s, ~400k cells —
  inside the few-minutes budget where full-corpus scoring (~350 tok/s
  prefill) is not.  Teacher knowledge stays recall-level (V2 top-k
  followers, not reasoning); explicit facts still prefer direct ingest.
- **R51-4 Tooling**: `scripts/train_flux.py` (txt/jsonl/json corpora,
  ChatML `messages` flattening, per-document `soft_reset`, `--resume`),
  `scripts/bench_flux.py` (ingest/gen tok/s + tracemalloc memory),
  `scripts/distill_flux.py` (selective teacher distillation driver).
- Measured (RTX 5070 box): ~32k tok/s ingest on real SFT text (CPU),
  ~250 tok/s predict+learn loop CPU, ~84 tok/s gen+learn on CUDA with
  72MB VRAM, 134 MB host RAM after 63k tokens; worst-case bounded by
  `max_cells_per_order`/`journal_cap`/`vec_delta_cap`.
  Smoke: trained on `data/sft/nontool_general.jsonl` (ChatML) ?
  "What is the capital of France?" answered "The capital of France is
  Paris." by memory retrieval.  Live-feeding Wikipedia/dictionary text
  raises recall quality directly (memory machine).  Greedy decoding
  loops remain (temperature ~0.6-0.8 + repetition_penalty/DRY help).
  (n-gram/episodic committee) � generalization is the known weak point;
  candidate upgrades: distilled count priors from ForgeLM, char-level
  fallback channel, learned sparse neural channel on hashed embeddings.
- Tests: `tests/unit/test_flux.py` (15 tests � forward contract, instant
  learning, episodic recall, cell locality, journal revert (exact w
  restore), snapshot roundtrip, Hedge adaptation, gap tracking +
  distill writes, CPU/CUDA engine integration, memory bound).

#### R&D round 52 (2026-10-03): FLUX live-learning upgrade — knn channel, archived episodic, hot-training

R51's known weak point was generalization (verbatim n-gram/episodic
committee) and episodic death at the ring horizon.  R52 attacks both
directly, keeping the associative-memory design (no gradients, no KV
cache, all writes journaled/reversible).

- **R52-1 `knn` channel — approximate-match context retrieval**:
  per-position bank of normalized context fingerprints
  (`normalize(c_prev @ Pcsem)`, 64-d) → successor token.  Prediction
  retrieves top-`knn_k` cosine neighbors and votes their successors
  weighted `relu(sim)**knn_gamma` (min-sim gated).  Exact `deep` only
  fires on verbatim repeats; knn generalizes to *similar* contexts
  across the whole stream — the memorizing-transformer pattern, fully
  online.  Bank capacity `knn_capacity` (default 1<<19) ring; every
  write journaled (`k` op) and revertible.  Hedged like every channel
  + `evidence_lift`.  VRAM: fp32 bank on GPU-primary ≈128MB @ 512k
  rows (capacity-configurable); CPU path stores an fp32 host bank.
- **R52-2 Archived episodic — memory past the ring horizon**: epi
  entries are now `(ctx_end, succ, cert_hashes)` tuples inserted
  *deferred* — when the successor arrives — so off-ring positions
  still vote via stored suffix-hash certificates (`cert_orders`,
  default 24/96-gram confidence grading).  Global FIFO cap
  (`epi_total_cap`) + per-key `epi_max_per_key`; evictions journaled
  (`eE`/`epi_ev`/`eve`) for exact revert.  Legacy bare-position
  entries still read (succ recovered from ring while bytes live).
- **R52-3 `teach(context, target)` — instant hot-training**:
  key-addressed write of a context→target association: multi-order
  count writes + csem fingerprint + knn bank row, journaled under a
  tag (`revert_tag`/`revert_writes` forget it).  Shares the
  `_write_counts_keys` refactor with `write_counts`/`distill_from`.
- **R52-4 VRAM fix — journal blocks on host**: `_ingest_bulk` block
  payloads (deltas, epi/knn snapshots, evictions) now store CPU
  tensors — bulk ingest no longer grows device memory with journal
  retention; `_undo_block` re-migrates on demand (`load()` keeps them
  host-side after deserialize).
- **Bug fixes found this round** (all confirmed + tested):
  - bulk `H_blk` closed form was off by one power of `P` (pre-existing
    R51 bug): every bulk-ingest suffix key was `correct·P⁻¹` —
    internally consistent, never matching live-era keys.  Episodic
    recall after `fit()` was silently broken on both backends.
  - `_pows` sized by max standard order but `cert_orders` reaches 96 —
    now sized over all hash orders.
  - GPU `used[row_epi]` counted per-entry not per-slot: duplicate keys
    in one block inflated it (592 for 225 slots); now unique-slot
    counted, revert returns to exact 0.
  - `_undo_block` epi victim restore resurrected block-era cells as
    zombie rows; now masks `pos >= base` cells and skips all-block
    rows (also avoids `_pick_slot` displacing live pre-block rows).
  - deferred epi `bout` flag: `dok1` bit packed into `bout[O+2]` —
    raw int64 keys are negative ~50% of the time, `>=0` was not a
    valid gate (half of deferred entries silently dropped).
  - `_knn_write`/`_fused_step` cursor sync (`_knn_cur`/`_knncur_g`)
    and `k`-undo cursor rewind.
  - `deep` channel was missing `evidence_lift` (hedge starved it at
    `hedge_min` during noise); GPU `_logits_dev` hardcoded channel
    indices replaced by `_chix` (knn insertion shifted them).
- **Known approximations** (documented, tested): bulk-ingest Hedge
  update is per-chunk constant-w (uni-fallback rewards) vs per-token
  live — weights converge slower on `fit()` corpora; GPU `index_add_`
  atomics make logits bit-non-deterministic (argmax stable);
  `_pick_slot` 2-choice victim restore is exact only when a
  free/dead candidate exists.
- Measured: CPU ingest ~1.4–3.4k tok/s (knn fp + write ≈ 2× R51
  per-token cost — the price of approximate retrieval; GPU bulk path
  unaffected), CUDA gen ~48 tok/s, 384MB host @ 200k tokens.
  Tests: 22 in `tests/unit/test_flux.py` — knn approximate recall,
  archived deep memory across ring wrap, teach+revert, snapshot
  roundtrip with knn/tuple-epi state, CPU/CUDA bulk-hash parity.
- **R52-5 Triton fusion + bulk-ingest correctness** (ForgeEngine
  HAS_TRITON convention, FluxConfig.use_triton, eager fallback
  everywhere):
  - knn bank scans sliced to the live ring prefix, power-of-2
    bucketed (knn_scan_min) so the shape stays static inside the
    captured CUDA graph; _graph_knn_b recaptures only on bucket
    crossing.  30k live rows: 32768 scanned vs 524288 padded —
    live CUDA step 470 → ~692 tok/s at V=4096 (profiled).
  - _flux_deep_votes_kernel — one launch evaluates all episodic
    cells (seed-hash check, backward extension, certificate grading,
    atomic vote scatter); replaces ~25 torch kernels, bit-exact.
  - _flux_logit_tail_kernel — sem+csem+knn mixture + recency/
    fatigue in one V-map pass, branchless device-scalar gates;
    max|Δ| 4.8e-7 vs eager, argmax identical.
  - _bout flag bit-packing vectorized (dot-product masks).
  - Bug fixes (reproduced + regression-tested):
    * _deep_votes_dev ignored the probe-found flag — a miss voted
      the default slot's unrelated row (ghost votes).
    * _Hg/_ringg bulk writes scattered with duplicate slots when
      a block exceeds 
ing_capacity — undefined order on CUDA,
      diverged from numpy last-wins; now chunked to unique-slot
      slices.
    * bulk episodic insert let colliding same-block keys write ghost
      cells under a foreign key; used counted entries not unique
      slots.  Now wholesale candidate-slot snapshots, found-gated
      cell writes, alternate-slot retry; _undo_block restores
      (key,raw,pos,cnt,succ,cert) per slot, legacy epi_gpu/epi_ev
      formats still readable.
  - Tests now 25 in 	ests/unit/test_flux.py.
- **R52-6 teacher graft + self-evolution primitives** (the V2-parent
  pipeline + math-driven self-improvement):
  - `graft_teacher(emb, method)` / `FluxConfig.teacher_embed` —
    replace random bipolar `_R` with a projected teacher embedding
    table (ForgeLM V2 `embed.weight`, [65536,2560] bf16).  Rows are
    unit-normed, **de-meaned** (removes the frequency/"commonness"
    axis — random-pair cosine baseline 0.116→0.002 while semantic
    pairs keep signal), then randproj→`topic_dim` (default; JL
    preserves pairwise structure — PCA collapses onto frequency axes,
    measured) and rescaled to sqrt(topic_dim) so Hebbian/delta-rule
    rates stay calibrated.  One transplant upgrades `c`/`_A`/proto/
    csem/knn fingerprints at once.  Snapshots persist the grafted
    matrix (`R_graft`); `load()` blanks the path so the teacher file
    isn't re-read.  `train_flux.py --graft`.
  - `distill_flux.py` loads the teacher via `ForgeEngine.from_checkpoint`
    (pipelined safetensors — 2.0s vs 57.7s raw build on this box).
  - `reinforce(tag, gain)` — outcome-weighted replay of a tag's
    journaled cell writes (self-play reward hook: gain>0 strengthens,
    gain<0 weakens with 0-floor clamps); journaled under
    `reinforce:<tag>`, revertible.
  - `consolidate(min_entries, min_agree, mass)` — self-distillation:
    episodic keys with agreeing modal successors promote into the
    order tables (`write_counts`, journaled) then free their cells
    (`eE`-journaled).  Off-ring cells stay archived.
  - Utility eviction: `_epi_hits` counts actual retrievals (CPU exact
    per-cell; GPU via a `_dhit_g` flag + 2 new `bout` slots read in
    `_fused_step`); `_epi_evict` scores the oldest `epi_evict_scan`
    fifo candidates by hits*epi_hit_w + recency instead of blind FIFO.
  - `_epi_drop`/`_epi_sync_row`: host evictions now resync the GPU
    episodic row — fixes a pre-existing divergence where host-evicted
    cells kept voting on CUDA.  `eE` undo resyncs the row too.
  - load() fix: GPU-snapshot → CPU-load wiped `_tables` to {} then
    triplet-rebuild KeyError'd; the per-order skeleton is kept when
    saved tables are empty.
  - Tests now 29 in `tests/unit/test_flux.py` (graft+snapshot,
    reinforce+revert, consolidate, utility-evict survival).
- **R52-7 ForgeEngine/GUI loadability + CPU→GPU snapshot fix**:
  - .flux snapshots are first-class checkpoints: models_index
    scans them (tagged config_name='flux'), EngineService._load_blocking
    routes them to ForgeEngine.from_flux (cuda+cuda_primary when the
    GPU is up), skipping the safetensors VRAM heuristic, activation
    presets, and gate probes (no hidden states to probe).
    config_name normalizes to 'flux' so dedupe/boot checks stay
    stable; DELETE /models accepts .flux; UI ModelEntry.is_flux.
  - rom_flux sets engine.checkpoint_path post-init (pre-init would
    run safetensors KeyStack detection on a pickle).
  - sleep(2)/wake() are flux-aware: FluxLM's weights ARE its live
    memory, so sleep(2) snapshots to a tempfile before discarding and
    wake() reloads via FluxLM.load — session learning survives the
    cycle (verified: stream+epi intact, identical generations).
    Sleep(1) stays a flag-only no-op for flux (no nn.Parameters —
    memory tables are the resident state; use level 2 to release).
  - Bug: CPU-snapshot → cuda_primary load wrote tok==vocab_size
    'sentinel' tot-cells into ttoks; _logits_dev indexed uni_pv[V]
    → device-side assert mid-decode.  Removed the sentinels (add()
    accumulates ttot/tocc itself) and hardened _logits_dev with an
    in-range clamp + mask so a stale cell id can never assert.
  - 	rain_flux.py --out default moved to research/checkpoints/flux/
    so snapshots appear in the GUI model index automatically.
- **R52-8 High-impact corpus filtering (`--topic`)**:
  - train_flux.py gained a domain scorer on title + intro (first 3k
    chars): preset packs `ai` / `code` / `science` (high-precision
    keyword sets — a false negative costs a page, a false positive
    pollutes memory) plus comma-separated custom keywords merged into
    the pack.  Title hit accepts outright; otherwise the intro needs
    >=min_hits (default 2) DISTINCT text hits so one repeated word
    can't carry a page in.
  - Applies to --hf streams (wikimedia/wikipedia rows), jsonl/ndjson/
    json records (row['title'] scored), and --dict-flatten kaikki
    entries (row['word'] as title).  .txt/.md files stay unfiltered —
    curated input.  [filter] progress lines report scanned/skipped.
  - Corpus-scale ingest: --hf streaming, --wiki-filter reuse of
    download_wikipedia.should_skip, --max-tokens/--max-docs,
    --ckpt-every periodic snapshots, per-doc soft_reset.
  - EngineService.learn(path, tag, max_tokens) + POST /engine/learn
    (+cancel): slice-wise (8k tok) ingest under the generation lease
    so chat interleaves; progress via engine_progress, result via
    learn_done hub events; FluxLM-resident only.
  - Tests: tests/unit/test_train_flux.py (12) — scorer accept/reject,
    distinct-hit rule, custom keywords, dict flatten, jsonl title
    scoring, txt passthrough.
- **R52-9 FluxLM RSI mode in InfiniteSelfPlayLoop (`--training-mode flux`)**:
  - Reuses the existing epoch skeleton (self-play -> train -> evaluate ->
    promote/demote, status writer, resumability) with flux primitives in
    place of the trainer: per-task journal-tagged generation attempts on
    _concise_qa_pairs verified tasks; failures revert_tag'd IMMEDIATELY
    (tag is the journal tail — deferred revert is a cascade bug, suffix
    rewind erases all later tags); successes reinforce(tag, gain) in
    Phase 2; failures teach_text()'d the verified answer in Q:/A:
    template with newline terminator; consolidate() each epoch; promote
    gates on a FIXED held-out QA set (flux_eval_seed — same questions
    every epoch); demote = full rollback via revert_since(epoch_mark).
  - Grader _flux_check: answer-first (startswith) or digit-aware
    word-boundary match ("42." ok, "42.5" not).
  - revert_since(seq_pos) fix: ws/we/g write-block markers are now
    boundaries — a position-floor revert can no longer eat teach/
    consolidate blocks below the mark (was silently draining journals).
    Regression test: test_revert_since_seq_pos_floor.
  - Verified run (6 epochs, ~60s/epoch, GPU): held-out acc 0 -> ~0.17-0.20
    with correct promote/demote gating; plateau matches the memory-model
    ceiling (finite-domain families memorize, arithmetic doesn't transfer).
  - R52-9b concurrent attempts + GUI launch: FluxLM.clone() (BytesIO
    snapshot round-trip — full-memory copy, zero shared state);
    generate_batch() routes FluxLM to cloned-worker threads
    (flux_batch_workers, default 4) with per-prompt temperature/seed/
    top-p/top-k/stops — exploration writes live on the clones and are
    rolled back per prompt, canonical model untouched; worker clones run
    _use_cuda_graph=False (concurrent CUDA-graph captures on one device
    invalidate each other — eager _step_dev instead).  LoopConfig gains
    flux_group_size (samples per task), flux_workers, flux_temp(_spread),
    flux_unique — cross-epoch prompt dedup via _flux_seen + eval-set
    exclusion (anti-overfit: no repeated goals, eval stays held-out).
    Attempts emit live task_start/task_done/flux_attempts_done/
    flux_eval_done events to status.json/events.jsonl.  GUI:
    /api/selfplay/start accepts mode="flux" + flux_checkpoint +
    flux_group_size/flux_workers; /api/selfplay/status exposes
    flux_models (checkpoints/flux/*.flux); SelfPlay page gains a
    mode picker + FLUX checkpoint select + worker knobs.
