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
