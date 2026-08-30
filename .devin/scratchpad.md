# ForgeAI Scratchpad

## R23 — V8-8B Local Training Pipeline (2026-08-29)

### Inventory (read-only exploration)

**V8 config preset** — `research/config.py:951-1058` (`forgelm_v8_8b`)
- 32-layer, d_model=4096, 64 heads, 16 KV heads, intermediate=16384
- BitNet b1.58 (int8 trainable), GTA attention, NLRQ FFN (rank=1024)
- R19 keys declared: `use_qsa`, `use_gated_residual`, `use_ngram_embedding`
- R20/R21 declared: `optimizer_type="nvme_muon_4bit"`, `nvme_streaming`,
  `use_fp8_activation`, `grad_topk_ratio=0.1`, `use_hashed_nlrq`
- Hyperparams: batch=1, seq=2048, 50K steps, warmup=2000, lr=3e-4→3e-5

**CRITICAL GAP**: V8 preset declares keys that are NOT wired into runtime:
- `configure_optimizer()` (`research/training/training_utils.py:174`) has NO
  handler for `"nvme_muon_4bit"`. The pieces exist separately in
  `research/training/optim/r20_memory_optimizers.py`:
  - `NVMeStreamedBAdam` (line 253) — mmap'd per-block optimizer states
  - `MuonBitNet4Bit` (line 460) — 4-bit Muon momentum + Newton-Schulz
  - But there is NO combined `nvme_muon_4bit` optimizer class.
- `use_qsa`, `use_gated_residual`, `use_ngram_embedding`, `use_hashed_nlrq`,
  `use_fp8_activation`, `grad_topk_ratio` — grep finds NO runtime consumer.
  These need model-loader wiring (R19 keys) + training-loop wiring (R21).

**Training scripts**
- `research/sandbox/train_8b_all.py` — V7-8B pretraining runner (BAdam +
  NLRQ STE factor training, resume bundle, VRAM preflight, async ckpt writer).
  Defaults to `forgelm_v7_8b_b`. This is the closest existing runner.
- `research/training/runners/sft_train.py` — SFT (LoRA, PEFT, manual LoRA).
- `research/training/runners/cpt_train.py` — continued pretraining.
- `research/training/runners/dpo_align.py` — has `--resume`.

**Checkpoint I/O** — `research/checkpoint_io.py`
- `save_checkpoint()` (line 68): atomic .tmp + verify + rename + .meta.json
- `load_checkpoint()` (line 171): fastsafetensors → safetensors → .pt
- `save_training_checkpoint()` (line 375): weights .safetensors + opt/EMA/RNG .train.pt
- `load_training_state()` (line 408): step, EMA, optimizer, RNG
- `step_checkpoint_path()`, `cleanup_step_keepers()` — rolling retention
- `train_8b_all.py` has its OWN resume bundle (`_resume.pt` with BAdam block
  idx, RNG, best_val, slim optimizer state) — separate from checkpoint_io.

**CPU/GPU mixed offload**
- `research/training/optim/hybrid_offload.py` — `CPUAdamW` (double_buffer,
  bandwidth_adaptive, grad_offload, grad_compression int4/int8, overlap)
- `research/training/optim/badam.py` — `BAdam` + `configure_badam()`
- `research/training/optim/r20_memory_optimizers.py` — `NVMeStreamedBAdam`,
  `MuonBitNet4Bit`, `AdamW4Bit`, `TernaryOptimizer`
- `research/training/optim/r22_training_speedups.py` — `CheckpointDelta`
  (8-bit quantized deltas, full_checkpoint_every=10), `MinHashDeduplicator`,
  `TokenImportanceSampling`, `ProgressiveLayerUnfreezing`,
  `GradientCompression` (4-bit + EF21), `AsyncDataPipeline`
- `research/inference/kv/cpu_kv_offload.py` — inference KV (not training)

**LoRA/DLoRA**
- `research/training/bitnet_lora.py` — `LoRAAdapter`, `add_lora_adapters()`,
  `merge_lora_adapters()`, `_MuonSFLoRA`. BitNet-compatible manual LoRA.
- `sft_train.py` CLI: `--lora`, `--lora-r`, `--lora-alpha`, `--manual-lora`,
  `--bitnet-everywhere`, `--sequential-freeze`.
- No DLoRA / DoRA file exists (DoRA import in scripts/train_expert.py is
  broken — `research/architecture/dora.py` missing).

**Checkpoint tester / knowledge regression**
- `train_8b_all.py --eval-only` — val loss / PPL only.
- `research/sandbox/test_v7_8b_generate.py` — single prompt smoke test.
- `research/evaluation/prompt_tests_auto.py` — 248 functions, 964 cases
  (code-task I/O pairs, NOT a checkpoint loader).
- `research/evaluation/reasoning_benchmarks.py` — ARC-AGI-2, NeoCoder, etc.
- `research/self_play/discovery/anti_regression.py` — DB fingerprint system.
- NO dedicated "load V8 checkpoint → run multi-question knowledge +
  regression suite" exists. This is a build target.

**Data**
- `research/sandbox/process_v7_data.py` — packs JSONL/Parquet → int32 .bin
  (train.bin, val.bin, manifest.json) under research/data/v7_train/.
- `research/data/generate_synthetic_data.py` — 70K verified synthetic examples.
- `research/data/download_gaps.py` — HF dataset downloader.
- `research/training/training_utils.py:BinaryDataset` (line 118) — flat uint32
  loader with pinned memory + async prefetch.
- `research/training/data/efficient_pipeline.py` — REFERENCED but MISSING
  (DiskTokenCache, AsyncPrefetcher, PackedSequenceDataset).
- `research/training/data/parquet_dataset.py` — REFERENCED but MISSING.
- No actual data/checkpoints on disk in this workspace.

**ETA / throughput estimates (existing)**
- `research/sandbox/calc_v8_8b_scratch_train.py`: 20.9 tok/s, 8.05B params,
  10.7B tokens × 3 epochs = 32.1B tokens → 289h (12 days), 1.96M steps,
  ~530ms/step. $12 electricity.
- `research/sandbox/calc_v8_8b_r22_speedup.py`: R22 combined 5.90× →
  3-epoch 289h→49h, 1-epoch 96h→16.3h. Breakdown: DataDedup 1.25×,
  TokenImportance 1.08×, ProgressiveUnfreeze 1.29×, GradCompression 1.94×,
  AsyncPipeline 1.74×, DeltaCheckpoint 6× (save/resume only).

### Plan (R23 build)

Goal: train ForgeLM V8-8B locally on RTX 5070 12GB + 32GB RAM with:
1. ETA/time tracking + live projection
2. Multiple rolling checkpoints + resume
3. Stable real training (VRAM preflight, spill detection, NaN guards)
4. Minimal memory (BAdam + NVMe-streamed 4-bit Muon + HashedNLRQ + FP8 act)
5. CPU/GPU mixed offload (BAdam block-wise + CPU master + NVMe optimizer)
6. Training data compression (packed int32 .bin + MinHash dedup + token importance)
7. In-depth checkpoint tester (multi-question knowledge + regression suite)
8. Minimal from-scratch run (LoRA warm-start from V7-8B-B checkpoint)
9. DLoRA research for from-scratch pretrain speedup

Build order:
1. Wire `nvme_muon_4bit` optimizer (combine NVMeStreamedBAdam + MuonBitNet4Bit)
2. Wire R19 keys into model_loader (QSA, GatedResidual, NgramEmbedding, HashedNLRQ)
3. Wire R21 training-time features (FP8 activation, GradTopK) into training loop
4. Build V8 training runner (fork train_8b_all.py → train_v8.py with V8 wiring)
5. Add ETA projection (tokens/sec EMA → wall-clock ETA + step ETA)
6. Add multi-checkpoint rolling retention + resume bundle (already in train_8b_all)
7. Add data compression pipeline (MinHash dedup + token importance sampling)
8. Build checkpoint tester (load ckpt → run knowledge + regression question suite)
9. DLoRA: research + implement DLoRA warm-start from V7-8B-B → V8

### HyperCloning + LiGO research (2026-08-29)

**HyperCloning** (Apple, NeurIPS 2024, arXiv:2409.12903)
- Function-preserving expansion: larger model's logits EXACTLY match smaller
  model's at init. Larger model inherits smaller model's accuracy at step 0.
- Supports embedding_dim_multiplier (int) + up_project_multiplier (int).
- Suggests: change # heads without changing head_size.
- Ref: https://github.com/apple/ml-hypercloning
- For ForgeAI: LFM2.5-1.2B (d=2048, L=16) → V8-8B (d=4096, L=32) = 2x width,
  2x depth. HyperCloning can do the 2x width (embedding_dim_multiplier=2).
  Depth doubling = layer stacking (duplicate each layer). Function-preserving
  if head_size stays 64 (V8 has 64 heads × 64 dim = 4096, LFM has 32 × 64 = 2048;
  2x heads = 2x width, head_size unchanged = HyperCloning-compatible).

**LiGO** (ICML 2023, arXiv:2303.00980)
- Learn linear map M: Θ_large = M · Θ_small. Factorized into width-growth
  (R_width) + depth-growth (L_depth) operators with Kronecker structure.
- Learned with ~100 steps of SGD on a small data subset.
- Saves up to 50% compute vs scratch. Outperforms stack-and-duplicate baselines.
- Ref: https://vita-group.github.io/LiGO/
- For ForgeAI: learn M from LFM2.5-1.2B → V8-8B. The Kronecker structure
  encodes the layer/neuron grouping. 100 steps of SGD on the growth matrix
  is cheap (~minutes on RTX 5070).

**V7-8B-B → V8-8B port** (same dimensions, new keys)
- V7-8B-B and V8-8B have IDENTICAL base dims (d=4096, L=32, heads=64,
  kv=16, intermediate=16384, NLRQ rank=1024).
- V8 adds: QSA, GatedResidual, NgramEmbedding, HashedNLRQ, FP8 act, GradTopK.
- Port = key-addition (identity/zero-init new modules), NOT dimension growth.
- This is the simplest port — lossless at init (new keys are no-ops at init).

### Five warm-start modes (user requested all three + HyperCloning + LiGO)

1. **scratch** — random init, full BAdam training. ~49-289h.
2. **lora-seed** — V7-8B-B init on attention (LoRA adapters), scratch on
   FFN/HashedNLRQ. Progressive unfreeze. ~49h for 3 epochs.
3. **dlora-warmstart** — Port V7-8B-B → V8 (lossless key-addition), then
   DLoRA (DoRA + LoRA) on V7-inherited layers + full-train V8-only keys.
   ~16h for 1 epoch.
4. **hypercloning** — LFM2.5-1.2B → V8-8B via HyperCloning (2x width, 2x depth).
   Function-preserving at init. Then full BAdam training. ~49h but starts
   at 1.2B accuracy, not random.
5. **ligo** — Learn growth matrix M from LFM2.5-1.2B → V8-8B (~100 steps SGD),
   then full BAdam training. Saves ~50% compute vs scratch. ~25h.
