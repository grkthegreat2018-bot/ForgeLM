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
