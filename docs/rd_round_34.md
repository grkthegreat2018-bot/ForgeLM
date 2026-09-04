# Round 34: Model Stability & Usability

**Date**: 2026-09-03
**Status**: COMPLETE — 32/32 tests passing
**Theme**: Hallucination detection, safety, self-correction, OOM resilience

## Features Implemented

### R34-1: TriLens — Per-Layer Logit-Lens Entropy
- **File**: `research/inference/safety/trilens.py` (408 lines)
- **Source**: arXiv 2606.01033
- **What**: At every layer, reads attention output + FFN output + residual
  through logit lens, records only entropy. 3L-dimensional trajectory.
  Single forward pass, no multiple samples.
- **Detection**: Late-layer entropy increase (0.6 weight) + trajectory
  variance (0.4 weight), sigmoid-squashed to [0,1]
- **VRAM**: 3L floats per token (~384 bytes for 32 layers)

### R34-2: PoP — Prediction-of-Prediction Inter-Layer Fusion
- **File**: `research/inference/safety/trilens.py` (same file)
- **Source**: arXiv 2608.27165
- **What**: Fuses intermediate hidden representations across depth.
  75.5% AUROC on TruthfulQA. <1.2% latency.
- **Detection**: Depth-weighted norm magnitude (0.6) + cross-depth
  coefficient of variation (0.4)
- **VRAM**: L floats per token (~128 bytes for 32 layers)
- **Ensemble**: `TriLensPoPEnsemble` combines both (0.6/0.4 weighting)

### R34-3: SyncThink — Reasoning Saturation Detection
- **File**: `research/inference/reasoning/syncthink.py` (206 lines)
- **Source**: OpenReview Hc9jAnIB3f
- **What**: Monitors attention transition signal — when answer tokens
  attend weakly to early reasoning and focus on boundary tokens,
  reasoning has saturated. 62% accuracy with 656 tokens vs 61.22%
  with 2141 tokens. +8.1 on GPQA.
- **Signal**: recent_mass / early_mass attention ratio, smoothed over
  window_size tokens
- **Parameters**: `window_size=32`, `transition_threshold=0.15`,
  `min_reasoning_tokens=128`

### R34-4: SPOC/MIRROR — Agent Self-Correction & Rollback
- **File**: `research/inference/safety/agent_corrector.py` (186 lines)
- **Sources**: SPOC (arXiv 2506.06923), MIRROR (arXiv 2505.20670)
- **What**: Detects tool-call errors, injects reflection prompts for
  self-correction (SPOC). Rollback to checkpoint on repeated errors
  (MIRROR). Inter-round error pattern tracking for escalation.
- **Classes**: `SPOCCorrector`, `MIRRORCorrector`, `AgentSelfCorrector`
- **Error detection**: Checks for error/exception/failed/ok=False keys
- **Rollback**: Deep-copies conversation state at checkpoints

### R34-5: VRAM Pressure Monitor — Proactive OOM Resilience
- **File**: `research/inference/safety/vram_monitor.py` (144 lines)
- **What**: Graduated degradation before OOM. 5 pressure levels:
  - 70%: compressed KV (s4r)
  - 80%: aggressive (snapkv_4bit)
  - 90%: CPU offload
  - 95%: critical (reduce tokens)
- **Callback system**: Register degradation callbacks per level
- **History tracking**: Records pressure over time for analysis
- **VRAM**: Negligible — stores only threshold config and float history

### R34-6: SIREN — Streaming Safety Guardrails
- **File**: `research/inference/safety/siren.py` (270 lines)
- **Sources**: SIREN (ACL 2026), StreamGuard (arXiv 2604.03962)
- **What**: Real-time harmful content detection using internal model
  representations. Lightweight linear probes (250x fewer params than
  guard models). 4 categories: harmful, deception, pii, violence.
- **Classes**: `SIRENGuard` (probe-based detection),
  `SIRENStreamWrapper` (wraps generate_stream iterator)
- **Permissive mode**: If no probes set, returns all zeros (no blocking)
- **VRAM**: ~64-128KB for probes (hidden_dim=8192)

## Test Results

```
tests/unit/test_round34_stability.py — 32 passed, 0 failed
```

Coverage:
- TriLens: init, record+trajectory, detect, reset (4)
- PoP: init, record+fuse, detect, ensemble (4)
- SyncThink: init, update+signal, should_terminate, min_tokens_gate, reset (5)
- SPOC/MIRROR: detect_error, reflection_prompt, checkpoint+rollback,
  escalation, agent_self_corrector, no_correction_on_success, stats (7)
- VRAM Monitor: init, get_level, callback, stats, reset, level_name (6)
- SIREN: init, evaluate_no_probes, evaluate_with_probes, check_token,
  reset, stream_wrapper (6)
