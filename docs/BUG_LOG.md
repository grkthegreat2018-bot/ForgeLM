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
   ```python
   (group[-n:] - shared.expand(...).norm()).mean()
   ```
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
   ```python
   diff = target - recon
   recon_err += float(diff.norm().item() / (target.norm().item() + 1e-8))
   ```
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
