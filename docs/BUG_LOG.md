# Bug Log

## 2026-08-11 — Self-play instability: model corruption, crashes, stalls

### Symptom
Self-play training was unstable: crashed on bad samples/epochs, stalled with
no detection, and silently corrupted the model after the first fine-tune epoch
(cascading garbage outputs on all subsequent epochs/validation).

### Root Causes

1. **LoRA merge corruption (CRITICAL)** — `_finetune_expert` merge loop
   assigned `merged` only inside the `if hasattr(layer, 'merge_and_unload')`
   branch but called `setattr(target, attr_name, merged)` unconditionally.
   In dense_bypass MoE mode, `experts[0]` was never LoRA-wrapped (only
   `shared` was), so `merged` retained `shared`'s last merged tensor and
   overwrote `experts[0]` weights with stale data.
   File: `research/training/self_play_expert_training.py` (~L1134)

2. **No per-topic/per-epoch exception isolation** — `main()` topic loop and
   `train_topic()` fine-tune call had no try/except. One bad sample/epoch
   killed the entire run, skipping all remaining topics.
   File: `research/training/self_play_expert_training.py` (L1745, L582)

3. **Cooperative stop sentinel never checked** — GUI wrote `STOP_REQUESTED`
   sentinel but training loop never polled it. Only abort path was
   SIGINT/SIGTERM handler, unreliable on Windows (taskkill doesn't deliver
   catchable SIGTERM → no emergency checkpoint, process hangs).
   File: `research/training/self_play_expert_training.py`, `research/self_play/live_status.py`

4. **Heartbeat decoupled from training progress** — `LiveStatusWriter` daemon
   thread wrote `heartbeat.json` every interval regardless of training-loop
   progress. A hung loop still looked "alive" to the GUI's staleness check.
   File: `research/self_play/live_status.py`

### Resolution

1. **LoRA merge**: Guarded `setattr` — only call it when `merge_and_unload`
   was actually invoked. Unwrapped layers are left intact.

2. **Exception isolation**: Wrapped each topic in `main()` with try/except
   (log, cleanup VRAM, continue to next topic). Wrapped fine-tune call in
   `train_topic()` with try/except (log, `empty_cache`, `model.eval()`,
   skip to next epoch). Followed PyTorch forum guidance: clear graph
   references + `empty_cache` after exceptions to prevent OOM cascades.

3. **Cooperative stop**: Added `stop_requested()` method to
   `LiveStatusWriter` (checks `STOP_REQUESTED` sentinel next to
   `status.json`). Training loop polls at epoch boundaries, before each
   task batch, and at topic boundaries. TF Coordinator `should_stop()`-
   style pattern — reliable on Windows where signals aren't.

4. **Progress-coupled heartbeat**: Added `_last_progress_ts` to
   `LiveStatusWriter`, updated on every public API call
   (`task_started`, `round_done`, `task_done`, `set_phase`, `update`).
   Heartbeat thread checks stall age; if >120s since last progress,
   writes `{"ts": ..., "stalled": true, "stall_age_s": ...}`. GUI
   `EventsReader.heartbeat_stalled()` reads this flag; `SelfPlayPage`
   shows "STALLED" phase tag and warns the user the run is hung.

### Files Modified
- `research/training/self_play_expert_training.py` — Fixes 1, 2, 3
- `research/self_play/live_status.py` — Fixes 3, 4
- `forge_gui/api/events_reader.py` — Fix 4 (GUI-side reader)
- `forge_gui/pages/selfplay.py` — Fix 4 (GUI-side display)

### Research References
- PEFT `merge_and_unload` is not in-place; must assign result
  (huggingface/peft PR #2871, Issue #2032)
- PyTorch exception isolation: exceptions keep differentiable output alive
  in frame → must `del` refs + `empty_cache` (discuss.pytorch.org #108619)
- TF Coordinator `request_stop()`/`should_stop()` pattern for cooperative
  shutdown (tensorflow.org API docs)
- PyTorch NCCL watchdog heartbeat pattern: monitor thread detects stale
  progress, escalates (pytorch/pytorch PR #112518)

## 2026-08-24 — OptimizerConfig metadata JSON serialization failure

### Symptom
ForgeEvolve `boot` run reported `OptimizerConfig: Object of type Tensor is not JSON serializable` and skipped the domain (56/57 valid). The error occurred during ForgeEvolve runs when discoveries were written to the SQLite DB, which serializes metadata via `json.dumps`.

### Root Cause
In `research/evolution/domains/training_domains.py`, `OptimizerConfig.evaluate()` line 69 stored `losses[-1]` (a CUDA tensor) directly into the `metadata` dict instead of a Python float. All other domains in the file (SchedulerConfig, LossConfig, MuonConfig) correctly wrap tensor values with `float(...)` or `.item()`. The discoveries DB path serializes metadata via `json.dumps`, which raises `TypeError` on torch.Tensor.

### Resolution
Changed line 69 from `"final_loss": losses[-1]` to `"final_loss": float(losses[-1].item())` in `research/evolution/domains/training_domains.py`.

### Files Modified
- `research/evolution/domains/training_domains.py` — Fix (line 69)
- `tests/unit/test_evolution_domains.py` — Regression tests (new file)

### Regression Tests Added
- `test_optimizer_config_metadata_final_loss_is_float` — Direct regression check that final_loss is a float and metadata round-trips through json.dumps
- `test_metadata_is_json_serializable[name]` — Parametrized over all 57 registered domains, verifying each domain's evaluate() returns JSON-serializable metadata, score, and behavioral values

### Verification
- All 58 new tests pass
- Full unit suite (636 tests) passes
- Re-ran `run_evolve.py --profile boot --domains OptimizerConfig` → best=15.99, 29 discoveries, no error

## 2026-08-24 — CrossLayerKV negative recon_err + engine metadata mismatch

### Symptom
ForgeEvolve `boot` run showed two suspicious results:
1. `CrossLayerKV` scored 218,453 — wildly higher than any other domain.
   `recon_err` in metadata was **-233.1** (negative reconstruction error).
2. `XQuantKV` best config said `recomputation_ratio=1.0, quant_bits=4` but
   metadata said `recomputation_ratio=0.0, quant_bits=8` — config and
   metadata disagreed.

### Root Causes

1. **CrossLayerKV recon_err (kv_domains.py L485)**: The reconstruction
   error formula was:
   ``python
   (group[-n:] - shared.expand(...).norm()).mean()
   ``
   `.norm()` was applied to `shared.expand(...)` (a scalar), not to the
   difference `(group - shared)`. This subtracted a large scalar from each
   group element, producing large negative values. The score formula
   `param_reduction * 100 - recon_err * 500` then turned negative recon_err
   into a massive positive bonus (+116k).

2. **Engine metadata mismatch (engine.py L547)**: When saving discoveries
   to the DB, the engine used `self.all_results[-1].get("metadata", {})` —
   always the **last** result's metadata — for every discovery in the
   generation. If configs A and B both entered the archive, both got B's
   metadata.

### Resolution

1. **CrossLayerKV**: Replaced the broken formula with proper relative L2
   reconstruction error:
   ``python
   diff = target - recon
   recon_err += float(diff.norm().item() / (target.norm().item() + 1e-8))
   ``
   This is always non-negative and in [0, ~2] range.

2. **Engine**: Added `metadata_list` alongside the existing `scores` and
   `behavioral_list`, populated in the same `zip(configs, results)` loop.
   The discovery loop now zips `metadata_list` too, ensuring each
   discovery gets its own metadata.

### Files Modified
- `research/evolution/domains/kv_domains.py` — CrossLayerKV fix (L483-489)
- `research/evolution/engine.py` — metadata_list fix (L449, L508-518, L534-548)
- `tests/unit/test_evolution_domains.py` — regression tests added

### Regression Tests Added
- `test_cross_layer_kv_recon_err_non_negative` — parametrized over all
  mode/ratio/n_groups combinations, asserts recon_err >= 0 and score < 1000
- `test_xquant_kv_metadata_matches_config` — parametrized over all
  ratio/bits/interval combinations, asserts metadata matches config
- `test_engine_discovery_metadata_matches_config` — runs a full
  ForgeEvolve engine on SyntheticDomain, verifies all_results have
  consistent metadata
- `test_engine_metadata_list_aligned_with_configs` — verifies
  all_results ordering and metadata presence

### Verification
- All 160 domain tests pass
- Full unit suite (738 tests) passes
- CrossLayerKV: all 160 mode/ratio/groups combos produce recon_err >= 0
- XQuantKV: all 40 ratio/bits/interval combos produce matching metadata

## 2026-08-24 — JSON round-trip type coercion (4 domains)

### Symptom
After enabling canonical warm-start (loading past configs from the DB),
4 domains crashed with type errors:
- `GroupQuant`: `view(): argument 'size' failed to unpack`
- `KvZipKV`: `'<' not supported between instances of 'int' and 'str'`
- `RotorQuantKV`: `'str' object cannot be interpreted as an integer`
- `SyntheticDomain`: `ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U...'))`

### Root Cause
Configs saved to the DB via `json.dumps()` and loaded back via
`json.loads()` can lose type information — integer values become strings.
The domain `evaluate()` methods assumed correct types without coercion.
This only manifested when warm-started generators produced configs that
were saved to the DB and re-evaluated on the next run.

### Resolution
Added `int()` / `float()` / `np.array(..., dtype=np.float64)` coercion at
the point of use in each affected domain:
- `quant_domains.py` GroupQuant: `int(config["group_size"])`, `int(config["n_bits"])`
- `kv_domains.py` RotorQuantKV: `int(c["n_rotations"])`, `int(c["quant_bits"])`
- `kv_domains.py` KvZipKV: `int(c["codebook_size"])`, `int(c["n_iter"])`
- `synthetic.py` SyntheticDomain: `np.array(config["x"], dtype=np.float64)`

### Files Modified
- `research/evolution/domains/quant_domains.py` — GroupQuant fix
- `research/evolution/domains/kv_domains.py` — RotorQuantKV + KvZipKV fixes
- `research/evolution/domains/synthetic.py` — SyntheticDomain fix

### Verification
- All 4 domains now handle string-typed configs correctly
- Full unit suite (747 tests) passes

## 2026-09-02 — Agent/Chat tool calls rendered as plain text (never executed)

### Symptom
In the Forge GUI Agent page and Chat Studio agent loop, the model's tool
calls appeared as plain text in the output and were never executed. The
agent loop terminated after one round with no tools invoked.

### Root Cause
`ForgeEngine.generate()` defaults to `skip_special_tokens=True`, which
strips the `<|tool_call_start|>` (id 10) and `<|tool_call_end|>` (id 11)
special tokens from the decoded output. The agent runner and chat agent
loop both called `engine.generate(...)` without overriding this default,
so `qwen_parse_tool_calls` could not find the marker-wrapped tool calls
(its primary parse path). It fell back to bare-JSON detection, which
fails on a model trained to emit the marker-wrapped format, so the raw
text — including the JSON tool-call body — was returned as plain "musing"
text and no tools were executed.

The engine docstring documents the fix: *"Set to False for tool-call
parsing (preserves <|tool_call_start|>/<|tool_call_end|> markers)."* —
but the GUI call sites did not pass it.

### Resolution
- `forge_gui/api/agent_runner.py` (line ~150): added
  `skip_special_tokens=False` to the `engine.generate(...)` call.
- `forge_gui/pages/chat.py` (line ~150, agent-loop path): added
  `skip_special_tokens=False` to the `engine.generate(...)` call.
- `forge_gui/api/sub_agent.py` was checked — it does not parse tool
  calls (returns raw text), so no change needed.

### Verification
- `tests/unit/test_gui_agent_tools.py` (14) + `test_gui_chat_store.py`
  (8) pass: 22 passed, 0 failed.


## 2026-09-13 - ForgeLM V2 Jamba: decorrelated outputs (use_rope), bf16 SSM scan drift, EOS blind spot, Quamba2+LoRA dtype crashes

### Symptom
Booting ForgeLM V2 (Jamba-Reasoning-3B port) produced fluent but semantically
incoherent text: think-traces hallucinated alternate user prompts, never
converged to answers, and ran past <|im_end|> to the token cap. Full-model
parity vs HF JambaForCausalLM showed logit cosine ~0.0 (per-layer probe:
divergence jumped at attention blocks 7/21). Separately, Quamba2+LoRA crashed
in add_lora_adapters (no .weight on Quamba2Linear) and warmup hit
fp32-vs-bf16 matmul errors.

### Root Cause
- forgelm_v2 preset set use_rope=True, but Jamba attention layers carry NO
  positional encoding (HF JambaAttention has no rotary call - Mamba supplies
  position). RoPE-scrambled Q/K decorrelated the only 2 attention layers.
- MambaLayer._selective_scan_ref ran the SSM recurrence in bf16; HF keeps the
  state and discretization in fp32 (~8%/element/layer drift, compounded over
  26 layers). Same issue in Quamba2Block._selective_scan_ref and _rmsnorm.
- StandardDecoding hardcoded eos_set={7,151643,151645} (LFM/Qwen only),
  ignoring engine _DEFAULT_EOS_TOKEN_IDS - Jamba 2/519 never stopped decode.
  eos_set.add(list) would also crash on HF list-valued eos_token_id.
- add_lora_adapters only probed weight/weight_packed/dense_packed; Quamba2
  uses weight_int4_packed. Quamba2Linear.forward never applied lora_adapter.
- Quamba2Block cast the SSM core to fp16, promoting activations to fp32 and
  breaking downstream bf16 matmuls (out_proj/LoRA).

### Resolution
- forge/config.py: forgelm_v2 use_rope=False (Jamba attention is
  position-agnostic; rope_base kept for API shape only).
- mamba_probe.py + quamba2.py: fp32 SSM recurrence (h/A_bar/B_bar/x in fp32,
  per-step output cast to activation dtype like HF), fp32 RMSNorm.
- quamba2.py: Quamba2Linear.forward applies lora_adapter with dtype
  alignment; block output cast back to model dtype.
- bitnet_lora.py: device detection falls back to weight_int4_packed.
- decoding.py: eos_set now includes Jamba {2,519} at both StandardDecoding
  and speculative fallback sites; list-valued eos handled.
- forge_engine.py: AVMP/VirtualTensorPool log lines divided bytes by 1024^2
  labeled GB -> now GiB (printed 12226GB for a 12GB GPU).

### Files Modified
- forge/config.py, forge/keys/architecture/mamba_probe.py,
  forge/quant/quamba2.py, forge/training/bitnet_lora.py,
  forge/engine/decoding.py, forge/engine/forge_engine.py,
  tests/unit/test_forgelm_v2_jamba.py, tests/unit/test_quamba2.py

### Verification
- Layer parity vs HF Jamba (bf16, GPU): cos >= 0.9993 every block, logit
  max diff 0.44, identical argmax (was cos ~0.0).
- 61 unit tests pass (test_forgelm_v2_jamba incl. 3 new EOS/RoPE
  regressions, test_quamba2 incl. LoRA coverage, test_r39_sparse_grammar).
- End-to-end boot: correct rate arithmetic, correct Python-semantics answer
  ([36,81]), well-formed <tool_call> JSON, clean <|im_end|> stop.

## 2026-09-13 — GUI boot blocked: VRAM pre-flight demanded ~16GB for ForgeLM V2

### Symptom
Booting ForgeLM V2 from the GUI always failed with "not enough free VRAM:
~11.5 GB free, loading this checkpoint needs ~16.0 GB" on the 12GB RTX 5070.

### Root Cause
`_LoadWorker` in forge_gui/api/engine_runtime.py estimated the load
requirement as `ckpt_size * 2.5` (~16GB for the 6.39GB checkpoint) while the
engine's own fast-path gate in ForgeEngine.from_checkpoint uses
`ckpt_size * 1.3` (~8.3GB). The GUI gate could never pass on 12GB even though
the validated mamba_hybrid profile peaks at ~7.5GB.

### Resolution
- engine_common.py: new `_FAST_LOAD_VRAM_HEADROOM = 1.3` +
  `_fast_load_vram_required(ckpt_size)` shared by both checks so the GUI and
  engine thresholds can never drift apart again.
- engine_checkpoints.py + engine_runtime.py: both call the shared helper.

### Files Modified
- forge/engine/engine_common.py, forge/engine/engine_checkpoints.py,
  forge_gui/api/engine_runtime.py, tests/unit/test_engine_checkpoint_guard.py

### Verification
- 12/12 tests pass in test_engine_checkpoint_guard.py (3 new: headroom value,
  preflight blocks below gate, preflight passes and reaches from_checkpoint).
- End-to-end load via from_checkpoint: 80s, peak 7.37GB allocated, 4.0GB
  free; auto-activated rotorquant 4-bit KV, QuaRot-KV, Quamba2 W4A8 (26 SSM
  blocks), AVMP, VirtualTensorPool, prefix cache, CacheBlend, chunked
  prefill, adaptive+suffix spec decode, FASER, SeqSplit, ReplaySSM.

### Follow-up (same boot session): chat crash on `thinking` kwarg
- **Symptom:** every Chat Studio send failed with
  `TypeError: render_messages_for_config() got an unexpected keyword
  argument 'thinking'`.
- **Root cause:** the GUI wrapper `forge_gui/api/chat_render.py` never
  forwarded `thinking` to the engine renderer
  (`forge/self_play/discovery/qwen_adapter.py`, which accepts it).
- **Resolution:** wrapper now takes `thinking: bool = True` and forwards
  it. Regression test in tests/unit/test_gui_chat_stream.py
  (TestChatRenderWrapper).

## 2026-09-16 — Mixin split dropped heavy imports: NameError at class-def / call time

### Symptom
`ForgeEngine` was unimportable in the working tree: `NameError: name
'CacheBlend' is not defined` at class-definition time in
engine_generation.py. After that was repaired, runtime probing plus the
evolutionary-merge suite surfaced further `NameError`s at call time:
`build_kv_cache`, `build_decoding` (engine_activation.py),
`build_health_report` (engine_diagnostics.py), and
`unpack_output_with_kv` (engine_sessions.py).

### Root Cause
The uncommitted monolith-to-mixin split of forge_engine.py moved methods
into engine_*.py files but dropped the heavy imports the monolith relied
on (engine_common.py intentionally stays import-light). Annotations and
call sites referenced symbols that were never imported into the new
modules.

### Resolution
Restored per-mixin imports for every symbol each file actually uses:
- engine_generation.py: CacheBlend, StandardDecoding, and other
  generation-path deps.
- forge_engine.py: StandardDecoding and core-path deps.
- engine_activation.py: build_kv_cache (kv_backend), build_decoding
  (decoding), plus activation-path deps.
- engine_checkpoints.py / engine_diagnostics.py / engine_lifecycle.py /
  engine_merging.py: their respective call-site deps (incl.
  build_health_report in engine_diagnostics).
- engine_sessions.py: unpack_output_with_kv from forge.model_loader.
- SemanticKVAnchors imported from prefix_cache; GGUFInfo from
  forge_loader; CREATIVE_SAMPLING resolved as class attribute.

### Files Modified
- forge/engine/forge_engine.py, engine_activation.py,
  engine_checkpoints.py, engine_diagnostics.py, engine_generation.py,
  engine_lifecycle.py, engine_merging.py, engine_sessions.py

### Verification
- `import forge.engine.forge_engine` succeeds; end-to-end `decide()`
  smoke test runs on CPU.
- tests/unit/test_evolutionary_merge.py: 8 errors -> all pass.
- 157 passed across test_evolutionary_merge, test_decide,
  test_forge_engine_fixes, test_engine_concurrency_fixes,
  test_chat_features.

### Follow-up (same session): Windows MAX_PATH triton cache crash
- **Symptom:** `test_model_loader.py::test_compile_for_inference_gpu`
  failed with `InductorError: FileNotFoundError` writing
  `triton_poi_fused_..._10.source` into the torchinductor cache.
- **Root cause:** the triton cache `put()` path is 268 chars —
  `%TEMP%/torchinductor_tmk68/triton/0/<48-char hash>/tmp.<id>/<138-char
  kernel name>.source` exceeds the 260-char Windows MAX_PATH, so
  `open(temp_path)` fails even though `os.makedirs` succeeded.
- **Resolution:** `tests/conftest.py` sets
  `TORCHINDUCTOR_CACHE_DIR=%TEMP%/ti` (short base) on Windows before
  torch import. Test passes; no repo code change needed.

### Follow-up (System One work): silent random-weight load + reasoning-template prompt

- **Symptom:** `load_default_model("forgelm_v2")` produced a model that
  generated pure gibberish on every prompt (flat ~2.0 logits, nonsense
  tokens). `decide()` on it returned near-uniform probabilities.
- **Root cause (two bugs):**
  1. `load_default_model` documented "checkpoint_path defaults to config
     default" but passed `None` through — `build_model_fast` then skipped
     the weight-load branch and printed only a generic `Total:` line.
     **Silent random weights.** (The pending-bug list had flagged this.)
  2. `decide.py` built a plain ChatML prompt ending at `assistant\n` —
     ForgeLM V2 is a reasoning model: the canonical template (see
     `qwen_adapter.render_messages_for_config`) primes
     `assistant\n<think>\n`, so the first generated token is reasoning
     text and answer tokens sat ~15 logits below the top.
- **Resolution:**
  - `model_loader.py`: `checkpoint_path=None` now resolves to
    `V2_CHECKPOINT` when it exists; `build_model_fast` prints a loud
    "no checkpoint_path — RANDOM weights" warning when it doesn't.
  - `decide.py`: prompt now uses the canonical think-hint injection and
    ends `assistant\n<think>\n</think>\nAnswer:` (empty think block +
    explicit answer field); candidates are space-prefixed
    (" yes"/" no", " finance", " 3") matching what the model emits
    after `Answer:`.
- **Verification:** real ForgeLM V2 via `ForgeEngine.from_checkpoint`
  + `activate_optimal` (GUI path): sector choices 0.99/0.95 correct,
  sentiment/hype scores land on the right level, noul directions all
  correct (0.65-0.79 true / 0.13-0.15 false — direction right,
  calibration soft as expected pre-Tier-1). 31 decide tests pass.

## 2026-09-19 � KV-cache eviction mask overflow + min-k dead sensitivity

### Symptom
1. `SnapKVCache._evict` (and the new `FillerKVCache._evict` modeled on it)
   crashed with `IndexError: The shape of the mask [total] does not match
   the indexed tensor` whenever the K/V buffer had grown beyond the live
   sequence length � i.e. any single append with `T > 1` crossing the
   capacity boundary, or after a buffer-doubling growth.
2. `_min_k_filter` (`engine_common.py`) and `_min_k_filter_logits`
   (`decoding.py`) computed `weighted_diffs > sensitivity * max_decay` and
   discarded the result; the `sensitivity` argument had no effect � every
   call truncated at the global argmax regardless of the configured value.

### Root Cause
1. The eviction `keep` mask is sized `total` (= live seq_len) but was
   applied to the full buffer axis (capacity > total after growth slack).
   Trigger: `cache.append(k, v)` with T=20 on a 12-capacity cache
   allocated 32 slots; `k_cache[:, :, keep]` then fails to broadcast.
2. Dead expression � the boolean cliff-mask was computed and thrown away;
   `cliff_pos = weighted_diffs.argmax(...)` ignored the threshold.

### Resolution
1. Mask only the first `total` buffer slots:
   `k_cache[:, :, :total][:, :, keep]` (same for v/scores/filler flags) in
   both `snapkv.py::_evict` and `filler_kv.py::_evict`.
2. Truncate at the *rightmost* position whose weighted decay exceeds
   `sensitivity * max_decay` (higher sensitivity ? higher threshold ?
   earlier rightmost cliff ? more aggressive truncation, matching the
   docstring); fall back to the argmax when no position exceeds
   (sensitivity=1 edge). Applied identically in both copies.

### Verification
`test_snapkv_multi_token_overflow_regression` (T=20 single-shot append ?
seq_len==12, no crash) and `test_min_k_sensitivity_controls_truncation`
(sens 0.1 keeps 5 tokens vs sens 0.9 keeps 1 on a two-cliff distribution).
Pre-existing `test_min_k_*` cases still pass (single-cliff inputs give
identical results). Full unit suite: 2607 passed.

### Follow-up (R50 feature smoke): DoLa inference-tensor crash

- **Symptom:** `DoLaDecoding.generate` crashed on the first sampled step
  with `RuntimeError: Inference tensors cannot be saved for backward`
  inside `ln_f`/`head` � only on real models, not in unit tests (stubs
  had requires_grad=False params).
- **Root cause:** `_contrast_logits`/`_early_logits` ran outside any
  no-grad scope while their `hidden_list` inputs were inference tensors
  produced by the cached prefill/decode forwards. Model params
  (`requires_grad=True`) tried to save them for backward.
- **Resolution:** `@torch.no_grad()` on `_contrast_logits` � `no_grad`
  (not `inference_mode`) so the returned logits stay normal tensors the
  sampling chain can mutate in place.
- **Verification:** `test_dola_contrast_accepts_inference_tensors`
  (inference-mode inputs + requires_grad params + in-place mutation on
  the output); GPU smoke `scripts/smoke_r50_gpu.py` � DoLa generates on
  real ForgeLM V2 ("Paris and the currency the euro" vs greedy
  "Paris. The capital of Germany is Berlin").


## 2026-09-20 - Cleanup sweep: three broken import paths found + fixed

### Symptom
1. `engine_activation.block_reconstruct` checkpoint-reload branch was
   unreachable-by-accident: read `self._checkpoint_path` (never set -
   the real attr is `self.checkpoint_path`) AND imported
   `forge.engine.model_loader` (module does not exist) AND called
   `load_default_model` as a `ModelLoader` method (it is a module
   function in the `forge.model_loader` facade).
2. `build_decoding('speculative')` mapped to `SpeculativeDecoding`,
   whose `generate()` imported `research.speculative_decode` - deleted
   in the research->forge migration, so the strategy crashed on first call.
3. Four production keys (`pit_key`, `lerope_key`, `attn_residual_key`,
   `mhc_key`) import `forge.keys.safety.safe_apply` for their
   `safe=True` path, but `safety.py` had been moved to
   `tests/fixtures/keys/` - the safe path raised ImportError.

### Root Cause
Refactor leftovers: file moves (safety.py -> fixtures, sandbox ->
training/runners, research/ -> forge/) updated the movers but not all
import sites; dormant paths had no test coverage so the breakage stayed
latent.

### Resolution
1. `block_reconstruct` now uses `self.checkpoint_path` +
   `self.config` and calls the canonical
   `ModelLoader.build_model_fast(cfg, checkpoint_path=...)`.
2. Removed `SpeculativeDecoding`; `build_decoding('speculative')`
   now maps to `ExternalDraftSpeculativeDecoding` (same draft_model
   interface) - the strategy name works again instead of crashing.
3. `safety.py` moved back to `forge/keys/safety.py` (production code
   imported by production keys); the 4 `_load_module` test paths in
   `test_key_transforms.py` updated.
4. Deleted unwired `airllm_streamer.py` (its one dependency,
   `forge.keys.moe.airllm_key.AirLLMKey`, no longer exists; the real
   streaming fallback is `engine_checkpoints._load_streaming`) and
   pruned 19 stale entries from `vast_connector.CRITICAL_SOURCE_FILES`
   / `CRITICAL_INIT_FILES`.

### Verification
`git grep` confirms zero references to removed modules;
`vast_connector` manifest now resolves 73/73 paths; unit suite run in
the same cleanup commit (see git log).
