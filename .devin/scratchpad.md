# ForgeAI Scratchpad

## R29 — LoRA knowledge injection + param growth (2026-09-01, COMPLETE)

**GOLDEN RATIO FOUND**: 2765 params/fact = 6.25 facts/rank at 120 epochs (batch=16,
FFN-trio single-layer LoRA on Qwen2.5-0.5B). Scaling law: rank ~= 2304 * n / epochs^2.
2x epochs → 4x fewer params (quadratic trade-off). Info density: 0.007-0.029 bits/param
(70-280x less efficient than full training, but ZERO regression on existing knowledge).

Phase 2 sweep: 25 conditions (5 sizes × 5 ranks) + 3 escalation experiments.
All results in scripts/r29_phase2.json + r29_phase1b.json. AGENTS.md documented.

### Root-cause find that unblocked everything
Qwen2.5-0.5B RESTATES the question after `"Question: {q}\nAnswer:"` — the
answer token sits at rank 3 (model emits " The capital of France is Paris").
With `"Q: {q}\nA:"` it answers directly (rank 1). R27/R28's "Question:/Answer:"
format + lenient top-3 metric papered over this; their strict exact-match runs
(truly_novel) silently scored 0 at baseline and were abandoned. Always use
`Q:/A:` for exact-match QA on this model.

### Facts (scripts/r29_facts.json, seeded, programmatic — no LLM data)
1000 inject: 400 6-digit access codes / 300 invented-word translations /
200 numeric specs / 100 serials. Codes tokenize as 7 tokens (space + 6 single
digits — Qwen splits every digit) → exact match = 7 exact greedy tokens =
hardest case, ~19.9 bits/fact true entropy. 10 known + 10 held real facts,
12 prose anchors + 8 QA-format anchors (added after phase-1 held-flip).

### Capacity ladder (probe2, 100 mixed facts, exact match)
- full FT lr1e-5 10ep: 88/100 (ceiling; no regression protection by design)
- r16 ffn-trio@L12 lr1e-3 30ep: 43 → 96 at 90ep (batched) — learnable
- r16 down_proj only 30ep: 35; r4 60ep: 29 → placement+rank+time all matter
- single down_proj / low rank learns the FORMAT (top1 52%) but not values

### Quant train-time matrix (scripts/r29_quant_speed.json) — user directive
| config | s/ep | exact@30ep | note |
|---|---|---|---|
| S0 fp32 b1 | 4.26 | 47 | old loop baseline |
| S2 bf16 b16 | 0.34 | 22 | batching = 12.7x lever |
| S2c bf16 b16 90ep | 0.34 | 96 | 31s total — WINNER |
| S3 torchao int8 | 0.43 | 17 | base PPL +3.5%, slower |
| S4 bnb NF4 | 0.59 | 32 | base PPL 27.5→42.8, slower |
| U1 unsloth 4bit+PEFT L12 | 0.54 | 95@90ep | 1.6x slower than own loop |
| U2 unsloth native 8.8M | 1.64 | 93@90ep | PPL 1872 catastrophic |

VERDICT: at 0.5B/12GB, base quantization does NOT minimize train time
(dequant overhead > memory-traffic savings); batching+bf16 does. bf16 LoRA
quantization is quality-neutral (27 vs 22 exact at 30ep). Unsloth native
placement at lr1e-3 wrecks general behavior.

### Phase 1 process matrix (r16 trio, b16, 90ep, 100 code facts)
plain 87% K0; replay 82% K10; kl 74% PPL+0.5% K3; orth 92% (worst PPL);
sched 94%; ema 100% K0; early==plain (never triggered); full 97% K10 Hreg1.
KEY INSIGHT: regression is FORMAT-specific, not general — replay protects
replayed facts, KL-prose protects fluency, NEITHER protects unseen QA →
held canary flips. Fix: QA-format anchors in the KL rotation (`full_qa`).

### WINNING PROCESS (validated twice, deterministic)
`full_qa` = orth init + warmup/cosine sched + 25% known-replay + KL(1.0,
prose+QA anchors) @ r16 ffn-trio@L12, b16, bf16 autocast, lr1e-3, 120ep:
- 100/100 inject exact, K 10/10 (one baseline-wrong known fact FIXED),
  H net 0 (single near-miss: "New Delhi"→" Delhi" — knowledge retained),
  PPL −0.23% (better than base). 276,480 params = 0.056% growth.
- 90ep variant: 87% recall (below target) → epochs matter with KL anchored.

### Bugs fixed this session (all confirmed-then-fixed)
1. r29_generate_facts gen_vocab_facts infinite loop (30-word pool vs n=300
   asked; 100% CPU spin) → multi-language product fix + assert len==1000.
2. Batched CE off-by-one (logits[i] vs labels[i]) → shift [..., :-1]/[..., 1:]
   + unit test test_ce_loss_shifts_by_one_position.
3. Merge bit-exact assert false-fails on trained deltas (fp32 GEMM
   reassociation ~1e-4 rel, amplified to ~0.03 logit) → 3-tier verification:
   bitwise weight reconstruction + layer rel <1e-3 + logit diff reported.
4. engine.model.layers → engine.model.model.layers (CausalLM wrapper).
5. bnb Params4bit.shape reports packed bytes (28672,1) — use
   in_features/out_features attrs; Params4bit must be built from CPU tensor
   then .cuda() (quantizes on move).
6. PEFT target_modules: list = literal names (matched all 24 layers, 6.6M
   params), single str = regex; unsloth get_peft_model rejects single-layer
   regex through its grouped-module patcher → vanilla PEFT with exact names.
7. torchao 0.17: int4_weight_only gone (needs unavailable 'mslk' lib, PyPI
   mslk is 0.0.0 placeholder) → Int8WeightOnlyConfig.
8. unsloth script: eval_all called before definition; stale `base = {t: 0}`
   rebinding destroyed eval results before JSON save.
9. KV-cached greedy eval asserted equivalent to slow path in phase_sanity
   (PASS) before adoption; cached eval ~5-7x faster for 400-fact sweeps.

### Cleanup pending
Delete scripts/_r29_debug.py, _r29_probe.py, _r29_probe2.py, _r29_read.py,
_r29_inspect.py, _r29_flips.py, _r29_run_n400.py (evidence recorded in JSONs +
AGENTS section). Keep test_r29_*.py and r29_analyze.py (canonical).

## R25 — FusedQKNormRopeCacheWrapper RoPE-convention bug (2026-08-31)

**Found while preparing infinite self-play for V10.** Two bugs in the
self-play → ForgeEngine wiring; first fixed, second shelved.

### Bug 1 (FIXED): cu_seqlens not accepted by fused_forward
- `FusedQKNormRopeCacheWrapper.fused_forward` (fused_qk_norm_rope_cache.py:206)
  did not accept `cu_seqlens`, but the block forward passes it for varlen
  packed sequences (model_loader.py:998+).
- **Fix**: added `cu_seqlens=None` param; when present, delegate to
  `original_forward` (which has the varlen_attention path). Fused path is
  inference-only; cu_seqlens is training-only — orthogonal, safe fallback.

### Bug 2 (SHELVED — needs bit-exact fix): RoPE convention mismatch
- The model uses **NeoX-style** RoPE: `cos_cached` shape `(max_seq, head_dim)`
  full-dim via `emb = cat((freqs, freqs), dim=-1)` (model_loader.py:443),
  applied as `(x * cos) + (rotate_half(x) * sin)`.
- The fused wrapper's `_py_qk_norm_rope` (fused_qk_norm_rope_cache.py:109)
  uses **GPT-J half-dim** convention: `q1,q2 = q[...,:half], q[...,half:]`
  with `cos` expected at `head_dim/2` → passes full-dim cos → 32-vs-64 crash.
- The imported `fused_qk_norm_rope` (decoding/fused_rope_qknorm.py) ALSO
  fails (272-vs-8 dim mismatch) — separate shape bug in that function.
- **Why shelved**: fixing RoPE convention wrong silently corrupts model
  outputs (RoPE is bit-sensitive). Per AGENTS.md directive A, needs a
  bit-exact forward-pass comparison against the original attention forward.
- **Workaround**: `LoopConfig.use_fused_qk_norm_rope_cache=False` (set in
  infinite_loop.py:75). Original attention forward handles RoPE correctly.
- **To fix later**: rewrite `_py_qk_norm_rope` to use NeoX convention
  (`rotate_half` + full-dim cos/sin), fix `fused_qk_norm_rope` shape bug,
  then bit-exact verify vs original forward on V10 checkpoint.

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

## R25 — LLM Quant Format R&D (2026-08-31)

### Baseline on real V9-1.2B weights (scripts/test_r25_baselines.py)
BitNet b1.58 post-training (NO QAT) = 6.61 dB SQNR — TERRIBLE. Its gold-standard
status comes ENTIRELY from from-scratch QAT. As post-training quant, it loses to
everything. This is the gap AQLM/EXL2/Quip# target.

| Algorithm | bpw | avg SQNR | avg out_err |
|-----------|-----|----------|-------------|
| BitNet b1.58 | 0.250 | 6.61 dB | 0.4675 |
| BitNetResidual 10% | 0.575 | 10.47 dB | 0.2992 |
| NVFP4 | 0.533 | 19.61 dB | 0.1044 |
| AS-FP4 | 0.533 | 20.36 dB | 0.0957 |
| SR-FP4 | 0.533 | 22.61 dB | 0.0740 |
| TSDS-FP4 | 0.720 | 20.42 dB | 0.0950 |
| HPR-FP4 | 0.533 | 20.74 dB | 0.0918 |
| IRI-FP4 x2 | 1.066 | 41.55 dB | 0.0084 |
| IRI-FP4 x3 | 1.600 | 62.60 dB | 0.0008 |
| IRI-FP4 x4 | 2.133 | 84.48 dB | — |

Real V9 weights are CLEAN: kurtosis 3.4-5.7 (near-Gaussian), no extreme outliers.
Hadamard rotation barely helps (+0.4 dB) — no outliers to spread.

### Algo 1: AdditiveFP4 (AQLM-inspired) — MODEST WIN
**v1 FAIL**: global k-means codebook (no per-block scale) lost to IRI-FP4 by 10-15 dB.
Root cause: IRI-FP4's per-block MSE-optimal scaling is the secret weapon.

**v2 WIN**: AdditiveFP4-scaled — FP4 base + learned codebook on NORMALIZED residual
+ per-block absmax/6 scale. Combines AQLM learned codebook with IRI per-block scaling.

Results (scripts/test_r25_additive_fp4_v3.py):
- 1cb(16) @ 1.064 bpw vs IRI x2 @ 1.066: TIED (±0.5 dB across layers)
- 2cb(16) @ 1.596 bpw vs IRI x3 @ 1.600: TIED (±1 dB across layers)
- 3cb(8) @ 1.752 bpw: +5-6 dB over IRI x3 (finer granularity)
- 4cb(8) @ 2.158 bpw: +0.5 dB over IRI x4 @ 2.133 (learned codebook marginally better)

Verdict: roughly tied at exact matched bpw, wins on FINER BPW GRANULARITY
(can hit 1.35, 1.75 vs IRI's discrete 0.533 steps). Real value: per-layer
allocation (Algo 2) where different layers get different n_codebooks.

PureAQLM (vector quantize original weights, no FP4 base) FAILED: 10-27 dB.
Clean near-Gaussian weights have little correlational structure for VQ to exploit.
FP4 base + learned residual codebook is the right approach.

Optimal config: n_codes=8, n_codebooks=3-4. Scale mode: absmax/6 (k-means
adapts, so absmax/4/3 identical). l2norm scale is worse (-4 dB).

### Algo 2: IRI-Alloc (EXL2-inspired) — FAILED
Per-layer IRI round allocation via greedy marginal SQNR gain. LOSES at all 3
targets (-1.9 to -4.8 dB). Root cause: V9 layers have nearly identical SQNR
sensitivity (R1=20.7, R2=41.6, R3=62.6, delta=21 dB/round uniformly). No
layer-to-layer variation to exploit. EXL2 wins on models with high sensitivity
variation; V9-1.2B doesn't have that structure.

### Algo 3: LatticeFP4 (Quip#-inspired) — FAILED
D4/E8 lattice codebook + Hadamard rotation. LatticeD4 = 9.5 dB (vs AS-FP4 20.7 dB).
Root cause: lattice codebook has only 24/240 points covering the ENTIRE space,
but FP4 has 8^4=4096 effective points per 4-element block. Lattice covering
density advantage is overwhelmed by FP4's sheer level count. Also: lattice is
uniform in angle/magnitude — wrong for Gaussian weights (wasted on tails,
insufficient near 0). FP4's non-uniform {0,0.5,1,1.5,2,3,4,6} matches Gaussian
mass distribution.

### Algo 4: Hessian-Weighted IRI — BLOCKED on model loading API
### Algo 5: HybridFP4 (SR + AdditiveFP4 + Hadamard) — FAILED
Stochastic rounding adds noise the learned codebook can't compensate for
(-7 to -20 dB vs IRI). SR-FP4's +2 dB on round 0 doesn't compound through
learned codebook residual rounds. Deterministic AdditiveFP4 remains the winner.

## R26 — SUB-BITNET Quantization (2026-08-31)

### Goal
Get memory cost near or below BitNet (1.58 bits/w = 0.1975 bytes/w) with
BETTER quality, training-free (post-training), supporting train-to-tune.

### SOTA Research (web search 2026-08-31)
- LittleBit (NeurIPS 2025): 0.1 BPW via low-rank binary factorization (QAT)
- PTQ1.61 (ACL 2025): 1.61-bit PTQ via salient channel mask + binary + scaling
- BTC-LLM (2025): 0.7-1.11 bits via binary codebook clustering (PTQ)
- NanoQuant (2025): Sub-1-bit PTQ via low-rank binary factorization + ADMM

### Algorithms Tested (scripts/test_r26_sub_bitnet.py, test_r26_ternlc.py, test_r26_qwen.py)

1. **TernaryPack (5-in-8 base-3)**: Pack 5 ternary {-1,0,+1} in 1 byte (3^5=243<256).
   1.6 bits/w. ZERO quality loss vs BitNet (same ternary values, better packing).
   Pure storage win — BitNet wastes int8 (8 bits for 1.58 bits of info).

2. **TernPack per-channel**: Ternary + per-OUTPUT-CHANNEL scale (vs BitNet's per-tensor).
   1.64 bits/w. BETTER quality than BitNet (+0.1 to +2.9 dB) at LESS memory.
   This is what BitNet QAT converges to, but we apply it post-training.

3. **TernLC (Ternary + Low-Rank Correction)**: W ≈ T*scale + A@B
   - T: ternary weights (0.2 bytes/w base-3 packed)
   - A@B: top-rank SVD of ternary error E = W - T*scale (float16)
   - Storage: 0.2 + (out*r + r*in)*2/n bytes/w
   - For 4864x896, r=16: 0.247 bytes/w = 1.97 bits/w (below BitNet!)
   - For r=64: 0.374 bytes/w = 2.99 bits/w (above BitNet, much better quality)

4. **TernLC-refined**: Alternating optimization (re-ternarize residual after
   correction, re-SVD error). +0.3-0.7 dB over basic TernLC at same bpw.

5. **BinarySalient (PTQ1.61-inspired)**: Binary non-salient + 4-bit salient.
   LOSES to BitNet (-0.3 to -2.5 dB). Binary {-1,+1} can't capture Gaussian
   mass near 0 — the ternary zero is essential.

6. **LowRankTernary (NanoQuant-inspired)**: W ≈ A@B, A and B ternary.
   LOSES badly (-1.5 to -5.2 dB). Ternary factors can't reconstruct full-rank
   weight matrix at low rank. Needs QAT (LittleBit uses Dual-SVID + residual
   compensation, not pure PTQ).

7. **BinaryCodebook (BTC-LLM-inspired)**: Cluster binary blocks into codebook.
   LOSES (-0.7 to -6.7 dB). Same binary problem — no zero level.

8. **TernPrep (PTQ1.61-inspired)**: Per-block (scale, shift) transform before
   ternary. LOSES (-0.4 to -2.1 dB). The shift breaks the ternary alignment.

### Results on Qwen 2.5 0.5B (real pretrained weights, 24 layers)

CRITICAL FINDING: Attn Q L0 has kurt=23.43 (extreme outliers). BitNet per-tensor
gets only 3.55 dB. TernPack per-channel gets 6.40 dB (+2.85). TernLC-refined r=64
gets 13.06 dB (+9.51 dB). Real models have outlier layers where BitNet's per-tensor
scale fails catastrophically. Per-channel scale + low-rank correction handles
outliers far better.

| Method | bits/w | vs BitNet | SQNR delta range |
|--------|--------|-----------|------------------|
| BitNet per-tensor | 2.00 | baseline | — |
| TernPack per-channel | 1.64 | BELOW | +0.3 to +2.9 dB |
| TernLC-refined r=16 | 1.97 | BELOW | +0.7 to +7.5 dB |
| TernLC-refined r=32 | 2.31 | above | +1.1 to +8.4 dB |
| TernLC-refined r=64 | 2.99 | above | +1.7 to +9.5 dB |

### Winners
1. **TernPack per-channel** (1.64 bits/w): FREE WIN — below BitNet, better quality
   on ALL layers. Pure post-training, no calibration data needed.
2. **TernLC-refined r=16** (1.97 bits/w): Below BitNet, +0.7-7.5 dB. Best
   near-BitNet option. SVD + 5 alternating iterations (~0.3s/layer).
3. **TernLC-refined r=64** (2.99 bits/w): Above BitNet but +1.7-9.5 dB. Best
   quality option for outlier layers.

### Train-to-Tune Compatibility
- Ternary weights support STE (straight-through estimator) — already in bitnet_b158_key.py
- Low-rank correction (A, B) are float16, fully differentiable
- TernLC is fully trainable: ternary via STE + low-rank via standard backprop
- LoRA can be applied on top (freeze ternary+correction, train adapters)
- The user's int8 loading work (2026-08-31) already supports direct int8 load
  via load_prequantized() — TernPack can use the same path with base-3 unpack

### CRITICAL: Full-Model Perplexity Test (2026-08-31)
End-to-end test on Qwen 2.5 0.5B (scripts/test_r26_qwen_ppl.py):
| Method | bits/w | PPL | vs baseline |
|--------|--------|-----|-------------|
| Original (fp32) | 32.0 | 9.64 | baseline |
| BitNet per-tensor | 2.00 | 663,925 | CATASTROPHIC |
| TernPack per-channel | 1.63 | 169,979 | CATASTROPHIC |
| TernLC r=16 | 2.02 | 2,639,566 | CATASTROPHIC |
| TernLC r=64 | 3.20 | 74,696 | CATASTROPHIC |

**Root cause**: Per-layer SQNR of 6-13 dB = 20-50% error per layer. These
errors COMPOUND across 24 layers. A model needs <1% error per layer (~30+ dB
SQNR) to function, which requires 4+ bits/w post-training.

**CONCLUSION**: Post-training sub-BitNet quantization is fundamentally broken
for full models. BitNet's success comes ENTIRELY from QAT — the model is
trained to be robust to ternary quantization. SQNR on individual layers is
MISLEADING for full-model quality.

**Path forward**: TernPack/TernLC as INITIALIZATION for QAT + fine-tuning
(train-to-tune). The per-channel scale + low-rank correction gives QAT a
better starting point than pure BitNet. The train-to-tune path:
1. Quantize model with TernLC (post-training initialization)
2. Fine-tune with STE (ternary gradients pass through as-is)
3. Low-rank correction (A, B) trains via standard backprop
4. LoRA adapters can be added on top for task-specific tuning

## R27 � Knowledge Injection Without Param Growth (2026-08-31)

### Problem
Two limitations identified when fine-tuning V10-1.2B without growing params:
1. **Net-new factual knowledge storage** is capacity-bound (can't pour new facts
   into fixed params without displacing old ones).
2. **Long-tail / world-model coverage** broadening is marginal at fixed param count.

### Capacity Math (the hard ceiling)
- **2 bits/param** for factual knowledge (ICLR 2025, Physics of LLMs 3.3) � even
  int8 quantized. V10 = 1304.6M params -> ~2.6B bits = ~325MB of facts.
- **3.6 bits/param** total memorization capacity (2025 study, 500K-1.5B models).
- English Wikipedia ~10GB compressed -> V10 can store ~3% of Wikipedia's facts.
- Subsequent facts OVERWRITE prior low-frequency facts (EMNLP 2024 scaling laws).
- This is the real ceiling. Fine-tuning repurposes; it does not create capacity.

### SOTA Fixes Researched (web search 2026-08-31)

**Issue 1 � Net-new knowledge (capacity-bound):**
- **MEMOIR (NeurIPS 2025)**: residual memory module Wm (zero-init copy of one
  FFN W_proj) + sparse sample-dependent masks M(a(x)). Each edit confined to
  k<<D columns of Wm. Base weights FROZEN. Scales to 1000s of edits, minimal
  forgetting. FFN_edited(a) = W0*a + Wm*(M(a) * a). Code: github.com/qym7/MEMOIR.
- **DyPRAG (2025)**: hypernetwork  parameter translator maps document ->
  LoRA deltas at test time. Plug-and-play parametric knowledge. Eliminates
  PRAG's per-document training cost.
- **PRAG (2025)**: parametric RAG � encode documents as LoRA modules, inject at
  FFN level. Deeper integration than in-context RAG.
- **MicroEdit (EMNLP 2025)**: SAE-based neuron disentanglement for precise
  minimal edits. Solves Edit Overshooting + Knowledge Entanglement.
- **MEMIT-Merge (ACL 2025)**: fixes same-subject batch editing conflicts (90%
  vs 50% success at large batches).
- **Lifelong Knowledge Editing regularization (EMNLP 2025)**: MPES (Most-
  Probable Early Stopping) + Frobenius norm constraint. Scales locate-then-edit
  to 10,000 edits, 42-61% faster.

**Issue 2 � Long-tail coverage (continual pretraining):**
- **Simple/Scalable Continual Pretraining (2024)**: LR re-warming + re-decaying
  + replay of old data MATCHES retraining-from-scratch at fraction of compute.
  Validated at 405M and 10B params.
- **TiC-LM (ACL 2025)**: web-scale benchmark (114 Common Crawl dumps).
  Autoregressive meta-schedule + fixed-ratio replay = 2.6x compute savings,
  matches retraining on general data.
- **GeRe (2025)**: small FIXED set of general replay samples sufficient to
  retain general capabilities across sequential domain tasks. TM-loss
  (threshold-based margin) maintains activation state consistency.
- **Gradient alignment + replay (ICML 2025)**: small replay rates > scaling
  model size at low replay budgets. MER (meta-experience replay) adds gradient
  alignment at negligible cost.
- **Domain-name prepending** (Physics of LLMs 3.3 result 12): prepending
  training data with domain names (e.g. wikipedia.org) SIGNIFICANTLY increases
  knowledge capacity. Model autonomously prioritizes knowledge-rich domains.

### R&D Goal: R27 � ForgeKM (Knowledge Module) for V10

**Objective**: Enable V10-1.2B to absorb net-new factual knowledge AND broaden
long-tail coverage WITHOUT growing base params, beating the 2 bits/param ceiling
via externalized + structured memory. Target: 10x effective knowledge capacity
at <0.5GB additional VRAM.

**Novel twists for RTX 5070 / 12GB / V10 architecture** (per AGENTS.md directive C):

#### Track A: IRI-FP4 Residual Memory (MEMOIR + V10 quantization)
- MEMOIR's Wm is a NEW param module (zero-init FFN copy). Paper used fp16.
- **Novel**: quantize Wm with IRI-FP4 x2 (9.0 bits/w, lossless, already in V10).
  -> ~1.8x more edits in same VRAM vs fp16 Wm. Base model stays frozen at IRI-FP4.
- V10 has 6 GQA layers (FFN at layers 2,5,8,10,12,14). Edit the MIDDLE GQA
  layer's FFN (layer 8 or 10) � standard MEMOIR placement.
- VRAM budget: Wm for one FFN = 8192x2048 = 16.8M params. IRI-FP4 x2 = 18.9MB.
  1000 edits x k=64 sparse cols = 64K params/edit touched. Trivial VRAM.
- **Implementation**: new key 
esearch/keys/memory/residual_memory_key.py
  (ResidualMemoryKey). Hooks into model_loader.py alongside IRI-FP4.

#### Track B: Conv-Layer Residual Memory (V10-specific novel variant)
- V10 has 10 CONV layers (unique to LFM2.5 architecture). MEMOIR only edits FFN.
- **Novel**: apply residual memory to conv layers with CHANNEL-wise sparse masks.
  Conv has natural channel sparsity (conv_kernel_size=3, d_model=2048 channels).
  Channel mask M(a) selects k<<2048 channels per edit -> each edit touches only
  k conv output channels. No paper has done this (MEMOIR is FFN-only).
- Conv layers are CHEAPER to edit than FFN (fewer params per layer) and may
  capture different knowledge type (local patterns vs key-value facts).
- **Risk**: conv layers may store less factual knowledge than FFN. Test this
  empirically � if conv edits don't take, shelf and document (directive C).

#### Track C: DyPRAG Parameter Translator (hypernetwork, CPU-offloaded)
- Train a tiny hypernetwork F'_phi (parameter translator) that maps document
  embeddings -> LoRA deltas for V10's GQA layers.
- **Novel**: translator trained on IRI-FP4-quantized LoRA targets (V10's native
  format). At inference: retrieve doc -> translator -> LoRA delta -> merge -> gen.
- **VRAM budget**: translator is small (1-5M params). Keep on CPU (pinned RAM),
  stream LoRA deltas to GPU. Fits hybrid_offload.py pattern.
- **Training data**: use V10 itself to generate (doc, LoRA-delta) pairs via
  PRAG-style offline parameterization on a synthetic fact corpus.
- **Implementation**: new module 
esearch/training/dyprag_translator.py.

#### Track D: Replay-Augmented CPT with Domain Tagging
- For broadening long-tail coverage (Issue 2), NOT single-fact injection.
- **Continual pretraining** with: LR re-warming/re-decaying + 5-10% replay of
  old data (GeRe finding: small fixed replay set is sufficient).
- **Novel**: V10 tokenizer has 65536 vocab with room. Add domain tokens
  (e.g. <|wiki|>, <|news|>, <|code|>) and PREPEND them to training data.
  Physics of LLMs 3.3 showed this increases knowledge capacity significantly.
- **Novel**: use SpectralKV (already in V10) to detect knowledge-dense tokens
  during CPT and upweight them in the replay buffer. SpectralKV's frequency
  analysis can identify high-information tokens for selective replay.
- **Implementation**: extend sft_train.py with --cpt mode (continual pretraining,
  full-sequence loss not completion-only) + replay buffer + domain tags.

### Success Metrics
1. **Edit reliability**: >90% of injected facts recalled correctly (MEMOIR baseline).
2. **Edit locality**: <5% degradation on unrelated facts after 1000 edits.
3. **Effective capacity**: store 10x more facts than 2 bits/param ceiling allows,
   via residual memory (Track A/B) + translator (Track C).
4. **VRAM overhead**: <0.5GB additional for the knowledge module.
5. **CPT forgetting**: <2% degradation on base benchmarks after 1B tokens of
   new-domain CPT with 5% replay (Track D).

### Implementation Order (port-first per directive A)
1. **Track A first** (lowest risk, highest value): ResidualMemoryKey + IRI-FP4
   Wm. Port MEMOIR's sparse mask logic. Bit-exact test: inject 100 facts,
   verify recall + locality vs base V10.
2. **Track D second** (independent, uses existing sft_train.py infra): CPT mode
   + replay + domain tags. Measure forgetting on held-out general set.
3. **Track B third** (novel, may fail): conv-layer residual memory. If conv
   edits don't take, document and shelf (directive C).
4. **Track C last** (highest complexity): DyPRAG translator. Needs synthetic
   (doc, LoRA) training data generation pipeline first.

### Test Plan
- 	ests/unit/test_residual_memory.py: inject N facts, verify recall + locality.
- scripts/test_r27_capacity.py: sweep edit count (10, 100, 1000, 10000),
  measure recall curve + VRAM.
- scripts/test_r27_cpt.py: CPT on synthetic domain, measure forgetting on
  general benchmark with/without replay + domain tags.
- All tests CPU-runnable where possible (directive B/G).

### Papers (full list)
- MEMOIR: arxiv.org/2506.07899 (NeurIPS 2025)
- DyPRAG: arxiv.org/2503.23895
- PRAG: arxiv.org/2501.15915
- MicroEdit: aclanthology.org/2025.emnlp-main.1719
- MEMIT-Merge: aclanthology.org/2025.findings-acl.415
- Lifelong Editing Reg: aclanthology.org/2025.findings-emnlp.1234
- Continual Pretraining: arxiv.org/2403.08763
- TiC-LM: aclanthology.org/2025.acl-long.1551
- GeRe: arxiv.org/2508.04676
- Gradient Alignment + Replay: proceedings.mlr.press/v330/abbes26a
- Knowledge Capacity 2 bits/param: arxiv.org/2404.05405 (ICLR 2025)
- Memorization 3.6 bits/param: arxiv.org/2505.24832
- Fact Memorization Scaling: aclanthology.org/2024.findings-emnlp.658


## R27 � Script-Based Research Results (2026-08-31)

### Setup
- Model: Qwen 2.5 0.5B (494M params, 24 layers, d_model=896, FFN=4864, GQA 14H/2KVH)
- Hardware: RTX 5070 12GB, ForgeEngine for model lifecycle
- Fact sets: 10 KNOWN (forgetting probe), 10 HELD (locality), 20 INJECT (new facts)
- Fix applied: unpack_output_with_kv() in model_loader.py now handles HF ModelOutput
  dataclasses (CausalLMOutputWithPast) � enables ForgeEngine to wrap any HF model.

### Experiment 1: Naive Full Fine-tune (Script 2)
- Train ALL params on INJECT facts, 3 epochs, LR=5e-5
- Results: Inject 12?19 correct (+7), Known 10?0 (-10), Held 9?0 (-9)
- PPL: 9.65 ? 163.54 (+153.89, 17x degradation)
- L2 drift: 11.93
- **VERDICT: Catastrophic forgetting. Model learns new facts but destroys ALL
  prior knowledge. PPL 17x worse. This is the baseline MEMOIR/replay must beat.**

### Experiment 2: MEMOIR Residual Memory (Script 3)
- Freeze ALL base params. Add Wm (zero-init, 4.36M params, 17.43MB VRAM) on
  layer 12 down_proj. Only Wm trains. k=64 sparse mask per edit.
- Results: Inject 12?16 (+4), Known 10?9 (-1), Held 9?9 (0)
- PPL: 9.65 ? 10.48 (+0.83, 8.6% degradation � negligible)
- Wm columns used: 1117/4864 (23%), theoretical capacity 76 edits at k=64
- **VERDICT: MEMOIR WINS. Near-zero forgetting (Known -1 vs -10 naive),
  PPL stable (+0.83 vs +153.89), learns 4 new facts. The 4.36M Wm module
  (0.9% of model params, 17MB VRAM) preserves base knowledge by construction
  (frozen base) while absorbing new facts via sparse residual.**

  Note: Inject recall is lower (16/20 vs 19/20 naive) � Wm has less capacity
  than full fine-tune. Trade-off: far less forgetting, slightly less recall.
  Could improve with: more epochs, higher LR, larger k, or multi-layer Wm.

  Interesting: Known/Held avg_prob INCREASED (0.05?0.60) � Wm's residual
  connections boost ALL fact confidence, not just injected ones. The sparse
  mask routes through frozen base, so old knowledge is reinforced, not erased.

### Experiment 3: CPT with Replay (Script 4)
- Full fine-tune (all params) with replay of KNOWN facts mixed into batches.
- Three conditions: 0% replay (naive), 10% replay, 50% replay.

| Condition   | Known ? | Held ? | Inject ? | PPL ?   | K-prob ? | I-prob ? |
|-------------|---------|--------|----------|---------|----------|----------|
| 0% (naive)  | -5      | -6     | +18      | +266.06 | -0.0437  | +0.7914  |
| 10% replay  | +3      | +0     | +18      | +153.91 | +0.4015  | +0.9372  |
| 50% replay  | +4      | +2     | +18      | +178.92 | +0.8489  | +0.9601  |

- **VERDICT: Replay works. 10% replay flips Known from -5 to +3 (net +8
  facts retained). 50% replay gets Known +4, Held +2. Inject recall stays
  at +18 (20/20) in all conditions � replay does NOT hurt new fact learning.**
- PPL still degrades (153-178 vs 266 naive) � replay helps but doesn't fully
  prevent language quality loss. MEMOIR''s PPL delta (+0.83) is far better.
- 10% replay is the sweet spot: +3 Known, +0 Held, +18 Inject, PPL +153.91.
  50% replay adds marginal Known gain (+1) but worse PPL (+178 vs +153).

### Head-to-Head Comparison

| Method          | Known ? | Held ? | Inject ? | PPL ?   | Extra VRAM |
|-----------------|---------|--------|----------|---------|------------|
| Naive FT        | -10     | -9     | +7       | +153.89 | 0          |
| MEMOIR (Wm)     | -1      | 0      | +4       | +0.83   | 17 MB      |
| CPT 10% replay  | +3      | +0     | +18      | +153.91 | 0          |
| CPT 50% replay  | +4      | +2     | +18      | +178.92 | 0          |

### Key Findings

1. **MEMOIR is best for locality** (zero forgetting, stable PPL) but has
   lower recall (4 new facts vs 18 for replay). Use when preserving base
   knowledge is critical and new facts are moderate in number.

2. **Replay is best for recall** (18-20 new facts learned) and still
   prevents most forgetting (Known +3-4 vs -10 naive). But PPL still
   degrades significantly. Use when learning many new facts and some
   PPL degradation is acceptable.

3. **Hybrid approach is the winner**: MEMOIR (locality) + replay (recall)
   could combine both strengths. Train Wm with replay of old facts in
   the same batch � Wm absorbs new facts sparsely, replay keeps base
   pathways active, PPL stays stable. This is the R27 Track A+D hybrid.

4. **The 2 bits/param ceiling is real but bypassable**: MEMOIR''s 4.36M Wm
   adds 8.72M bits of capacity (2 bits/param) OUTSIDE the frozen base.
   1000 edits � 64 cols � 896 out_dim = 57M bits touched. The residual
   memory is additive capacity, not repurposed capacity.

5. **ForgeEngine now supports HF models** (unpack_output_with_kv fix).
   This enables testing any HF model (Qwen, Llama, Mistral) through
   ForgeEngine''s inference stack (KV cache, decoding, generation).

### Next Steps (R27 Track A+D hybrid)
- Script 5: MEMOIR + replay � train Wm with replay of old facts mixed in
- Script 6: Multi-layer Wm � install on layers 8, 12, 16 for more capacity
- Script 7: Capacity sweep � 10, 50, 100, 500, 1000 edits, measure recall
  curve + interference
- Port to V10-1.2B: ResidualMemoryKey in research/keys/memory/



## R27 — QLoRA Full-Rank Delta Injection (User Idea, Script 5)

### The Idea
Freeze model, create full-shape zero-init delta params matching model's
weight pattern, train only delta on target info, merge delta into base,
unfreeze. Tested on Qwen 2.5 0.5B via ForgeEngine.

### Results (4 conditions, 3 epochs, LR=5e-5, facts from r27_facts.json)

| Condition           | K_base | K_fin | H_fin | I_fin | PPL_b | PPL_f | K_d  | I_d  |
|---------------------|--------|-------|-------|-------|-------|-------|------|------|
| naive_ft            | 5      | 0     | 0     | 20    | 8.05  | 201.6 | -5   | +18  |
| full_delta (USER)   | 5      | 10    | 10    | 20    | 8.05  | 16.51 | +5   | +18  |
| full_delta_unfreeze | 5      | 10    | 10    | 17    | 8.05  | 6.70  | +5   | +15  |
| lora_64             | 5      | 10    | 10    | 18    | 8.05  | 10.32 | +5   | +16  |

### Analysis

full_delta (user idea) is the WINNER:
- Learns ALL 20 new facts (20/20, same as naive FT)
- ZERO forgetting - Known actually IMPROVED (+5, from 5 to 10)
- PPL only 2x (16.5 vs 8.05), vs 25x for naive (201.6)
- Merge is lossless (MERGED = pre-merge exactly)
- 13.07M delta params (2.6% of model), 3 layers (8, 12, 16)

full_delta_unfreeze:
- PPL drops BELOW baseline (6.70 vs 8.05) - replay improved language quality
- But lost 3 injected facts (17 vs 20) - replay partially overwrote new knowledge
- Tradeoff: better PPL, worse recall

lora_64:
- 1.11M params (0.2% of model) - 12x fewer than full_delta
- K=+5, I=+16 (vs +18 full_delta) - slightly less recall
- PPL 10.32 (better than full_delta 16.51)
- Best param-efficiency: 92% of full_delta recall at 8.5% of the params

### Why It Works

1. Frozen base = zero forgetting by construction. Gradients only flow
   through delta_W, so W0 (base knowledge) is never touched. Same
   structural guarantee as MEMOIR, but without sparse masks.

2. Full-rank delta = full capacity. Unlike LoRA (rank-64 = 1.11M),
   full_delta has 13.07M params - same shape as W0. It can represent
   ANY update that naive FT could, just isolated in a separate tensor.

3. Zero-init = no-op start. delta_W starts at 0, model is identical to
   base at init. Delta only deviates where gradients push it - new facts
   create new patterns, old facts untouched (no gradient signal for them).

4. Merge is lossless. W_merged = W0 + delta_W. Merged model is
   bit-identical to the delta model. No information loss.

5. Known IMPROVED (+5) because: delta adds capacity that also reinforces
   existing pathways. QA training format teaches the QA pattern itself,
   which boosts recall of ALL facts including ones it already knew.

### Full R27 Comparison

| Method              | K_d  | I_d  | PPL_d   | Extra params |
|---------------------|------|------|---------|--------------|
| Naive FT            | -10  | +7   | +153.89 | 0            |
| MEMOIR (Wm, k=64)   | -1   | +4   | +0.83   | 4.36M        |
| CPT 10% replay      | +3   | +18  | +153.91 | 0            |
| CPT 50% replay      | +4   | +18  | +178.92 | 0            |
| full_delta (USER)   | +5   | +18  | +8.46   | 13.07M       |
| full_delta+unfreeze | +5   | +15  | -1.35   | 13.07M       |
| lora_64             | +5   | +16  | +2.27   | 1.11M        |

full_delta dominates: best recall (+18), best locality (+5), reasonable
PPL (+8.46). lora_64 is the param-efficient alternative (1.11M, +16 recall).

### Verdict
The user's idea works. Full-rank delta with frozen base is the best
knowledge injection method tested in R27. It combines MEMOIR's locality
guarantee (frozen base) + naive FT's full recall capacity (full-rank
delta) + lossless merge (W0 + delta_W) + optional unfreeze+replay for
PPL recovery (tradeoff: lose some new facts).

### Next Steps
- Test on V10-1.2B: full_delta on 6 GQA layers FFN
- Capacity sweep: 50, 100, 500, 1000 facts - when does full_delta saturate?
- Multi-layer sweep: 1 vs 3 vs 6 layers of delta
- Hybrid: full_delta + 10% replay (best of both?)
- Port: FullRankDeltaKey in research/keys/memory/ for V10


## R27 — Novelty Assessment: Full-Rank Delta Injection (User Idea)

### The idea's components
1. Full-rank delta_W (same shape as W0, NOT low-rank A@B like LoRA)
2. Zero-init delta (starts as no-op)
3. Frozen base (only delta trains)
4. Lossless merge (W_merged = W0 + delta_W)
5. Optional unfreeze + replay for PPL recovery
6. Application: knowledge injection (not task adaptation)

### Closest existing work

**LoRA (Hu et al., 2022)**: Low-rank A@B, frozen base, zero-init B, merge back.
- At rank r = min(d,k), LoRA IS full rank. But nobody uses it at full rank
  (2x params: A and B both d x d, defeats the purpose).
- User's idea uses a SINGLE delta_W tensor (1x params) vs LoRA's A@B (2x params).
- LoRA targets task adaptation, not knowledge injection.
- Merge-back concept is identical.

**Side-tuning (Zhang et al., ECCV 2020)**: Additive side network, frozen base.
- Side network is a SEPARATE network, not a same-shape delta.
- Vision-focused, not LLM knowledge injection.

**ParamDelta (ICLR 2025)**: Computes delta = post - base, applies to new base.
- POST-HOC delta extraction from already fine-tuned model.
- Does NOT train a delta from scratch. Different direction entirely.

**BitDelta (2024)** / **DARE (ICML 2024)**: Compress/sparsify delta post-hoc.
- Again post-hoc, not training a delta.

**DoRA (ICML 2024)**: Decomposes weight into magnitude + direction, LoRA for direction.
- Still low-rank. Merge back yes, but decomposition is different.

**MEMOIR (NeurIPS 2025)**: Residual memory Wm with sparse masks.
- Different mechanism: sparse masks vs full-rank delta.
- Same frozen-base anti-forgetting guarantee.

**Hybrid-LoRA (2025)**: Selectively full FT some modules + LoRA on others.
- Closest in spirit (some modules get full FT). But direct FT, not delta-merge.

**MiCA (2025)**: Adapts minor singular vector subspaces only.
- Targets specific subspaces, not full-rank delta.

**Model Merging for Knowledge Editing (ACL 2025)**: Full FT then merge with original.
- Full fine-tune THEN merge (recover general capabilities). Not train-a-delta.

**Delta Tuning survey (Ding et al., 2022)**: Categorizes PEFT into addition-based,
specification-based, reparameterization-based. Full-rank single-tensor delta
would fall under "addition-based" but at full rank — unusual since most
addition-based methods add small adapters or low-rank matrices.

### Novelty verdict

**NOT fully novel** — individual components exist:
- Frozen base + trainable delta: LoRA (2022)
- Zero-init: LoRA (B matrix zero-init)
- Merge back: LoRA
- Delta concept: Delta Tuning literature (2022)

**NOVEL combination and application**:
1. **Full-rank SINGLE-TENSOR delta** (not A@B decomposition): 2x more
   param-efficient than full-rank LoRA. Nobody uses full-rank LoRA because
   for task adaptation it's pointless (just full FT). The insight that it's
   valuable for KNOWLEDGE INJECTION (where frozen base = anti-forgetting)
   is novel.
2. **Knowledge injection application**: LoRA/delta tuning targets task
   adaptation. Using full-rank delta specifically for factual knowledge
   injection with measured recall + locality + PPL is a novel application.
3. **Freeze-train-merge-UNFREEZE-replay pipeline**: The unfreeze + replay
   step after merge is new. ParamDelta does merge, but no unfreeze+replay.
   Model Merging for Knowledge Editing does FT+merge, but not delta-train+merge+unfreeze.
4. **Empirical finding**: +5 Known (IMPROVEMENT, not just preservation) +
   +18 Inject with 2.6% params. This result — that frozen-base full-rank
   delta actually IMPROVES old knowledge while learning new — is not
   reported in any paper found.

**What would make it MORE novel (R&D directions)**:
- IRI-FP4 quantized delta_W (V10-native): compress the delta with V10's
  lossless FP4, reducing the 2.6% param overhead further
- Multi-delta composition: train separate deltas for separate knowledge
  domains, merge selectively (like ParamDelta but trained-from-scratch)
- Capacity scaling law: characterize how many facts per delta param before
  saturation (our 20 facts / 13M params = 1.5K bits per param — far below
  the 2 bits/param ceiling, suggesting headroom)
- Conv-layer delta (V10-specific): apply delta to conv layers, not just FFN

### Papers reviewed
- LoRA: arxiv.org/2106.09685 (Hu et al., 2022)
- Side-tuning: sidetuning.berkeley.edu (Zhang et al., ECCV 2020)
- ParamDelta: arxiv.org/2504.21023 (ICLR 2025)
- BitDelta: arxiv.org/2402.10193 (2024)
- DARE: proceedings.mlr.press/v235/yu24p.html (ICML 2024)
- DoRA: arxiv.org/2402.09353 (ICML 2024 Oral)
- MEMOIR: arxiv.org/2506.07899 (NeurIPS 2025)
- Hybrid-LoRA: arxiv.org/2605.18822 (2025)
- MiCA: arxiv.org/2604.01694 (2025)
- Model Merging for Knowledge Editing: aclanthology.org/2025.acl-industry.30
- Delta Tuning survey: arxiv.org/2203.06904 (Ding et al., 2022)
- IMPART: aclanthology.org/2025.acl-long.921 (2025)
- LoRA vs Full FT illusion: arxiv.org/2410.21228 (2024)
- LoRA knowledge packing: aclanthology.org/2025.findings-naacl.243 (2025)
- LoRA as knowledge memory: arxiv.org/2603.01097 (2025)


## R27 — Optimized Delta Injection Results (Script 6)

### Conditions tested (9 total)

| Condition        | K_d  | H_d  | I_d  | PPL_d  | dNorm | Params   | Time  |
|------------------|------|------|------|--------|-------|----------|-------|
| baseline_3L      | +5   | +4   | +18  | +8.46  | 2.62  | 13.07M   | 5.9s  |
| replay10_3L      | +5   | +4   | +18  | +3.55  | 2.65  | 13.07M   | 6.4s  |
| l2reg_3L         | +5   | +4   | +18  | +8.43  | 2.62  | 13.07M   | 5.7s  |
| replay_l2_3L     | +5   | +4   | +18  | +3.55  | 2.64  | 13.07M   | 5.8s  |
| 6layers          | +5   | +1   | +18  | +19.44 | 3.58  | 26.15M   | 6.6s  |
| 12layers         | +3   | +0   | +18  | +10.31 | 5.34  | 52.30M   | 24.2s |
| all_ffn_3L       | +2   | +1   | +18  | +12.50 | 4.21  | 39.22M   | 29.1s |
| replay10_6L      | +5   | +4   | +18  | +4.29  | 3.55  | 26.15M   | 31.6s |
| best_guess       | +5   | +4   | +18  | +4.28  | 3.55  | 26.15M   | 31.2s |

### Key findings

1. REPLAY IS THE PPL FIX. Adding 10% replay of old facts during delta
   training cut PPL delta from +8.46 to +3.55 (58% reduction) with ZERO
   impact on recall (still +18 inject) or locality (still +5 known).
   This is the single most effective improvement.

2. L2 REG ALONE DOES NOTHING. weight_decay=0.1 on delta_W had no measurable
   effect (PPL +8.43 vs +8.46). The delta norm barely changed (2.62 vs 2.62).
   L2 reg on delta is not the right lever - replay is.

3. MORE LAYERS HURTS (without replay). 6 layers: PPL +19.44 (2.3x worse
   than 3 layers), Held dropped from +4 to +1. 12 layers: PPL +10.31,
   Known dropped from +5 to +3. Spreading delta across more layers
   increases total disruption - each layer gets a smaller update but
   the cumulative PPL impact is worse.

4. MORE LAYERS + REPLAY RECOVERS. replay10_6L: PPL +4.29 (vs +19.44
   without replay), Held back to +4. Replay fixes the multi-layer
   problem. But 6L+replay (+4.29) is still worse than 3L+replay (+3.55)
   while using 2x params. 3 layers is the sweet spot.

5. ALL FFN MATRICES HURTS. gate_proj + up_proj + down_proj on 3 layers:
   PPL +12.50 (vs +8.46 down_proj only), Known dropped to +2 (vs +5),
   Held +1 (vs +4). 3x more params (39M vs 13M) for WORSE results.
   down_proj alone is the right target - it is the output projection
   where knowledge is read out, gate/up are input routing.

6. BEST CONFIG: replay10_3L (or equivalently replay_l2_3L).
   K=+5, H=+4, I=+18, PPL=+3.55, 13.07M params (2.6%), 6.4s train.
   This is the optimal operating point: 3 middle layers, down_proj only,
   10% replay, no L2 reg needed.

### Optimization summary

| Metric         | Original (Script 5) | Optimized (Script 6) | Improvement |
|----------------|---------------------|----------------------|-------------|
| Known delta    | +5                  | +5                   | same        |
| Held delta     | +4                  | +4                   | same        |
| Inject delta   | +18                 | +18                  | same        |
| PPL delta      | +8.46               | +3.55                | 58% better  |
| Params         | 13.07M              | 13.07M               | same        |
| Train time     | ~6s                 | 6.4s                 | same        |

The only change: add 10% replay of old facts during delta training.
Zero extra params, zero extra inference cost, 58% PPL improvement.

### What did NOT work
- L2 regularization on delta_W (no effect)
- More layers (6, 12) without replay (worse PPL, worse Held)
- All FFN matrices (gate+up+down) (worse on every metric, 3x params)
- 12 layers even with replay would likely still be worse (not tested
  but 6L+replay already worse than 3L+replay)

### Remaining PPL gap (+3.55)
The optimized method still has a small PPL increase. This is because
the delta adds new knowledge pathways that slightly shift the output
distribution. Options to close this gap:
- Higher replay ratio (20-30%) - may hurt inject recall
- Unfreeze + replay after merge (Script 5 showed PPL -1.35 but lost
  3 inject facts)
- Distillation: use base model as teacher during delta training
- Accept it: +3.55 PPL on a single test sentence is within noise

### Next steps
- Capacity sweep with optimized config (replay10_3L): 50, 100, 500, 1000 facts
- Distillation during delta training (teacher = base model)
- Port to V10-1.2B with IRI-FP4 quantized delta
- Multi-delta composition (separate deltas for separate domains)


## R28 — Fact-to-Param Relationship (Script 2, 2026-08-31)

### The Law
For rank-r LoRA on 1 layer down_proj with frozen base + 10% replay:
  params = r * (in_dim + out_dim)  [FIXED, independent of fact count]
  params_per_fact = params / n_facts  [decreases linearly with facts]

At rank 4 on Qwen 0.5B (in_dim=4864, out_dim=896):
  params = 23,040 (0.005% of 494M model)
  params_per_fact = 23,040 / n_facts

### Empirical Recall Curve (rank 4, 1 layer, 10% replay, 3 epochs)
  20 facts  -> 35% recall, 1152 p/f
  50 facts  -> 88% recall, 461 p/f
  100 facts -> 93% recall, 230 p/f
  150 facts -> 96% recall, 154 p/f

Recall INCREASES with more facts (shared subspace fills up).
No saturation observed at 150 facts.

### Rank Scaling (100 facts)
  rank 4  -> 93% recall, 0.023M params, 230 p/f  (OPTIMAL)
  rank 8  -> 93%, 0.046M, 461 p/f
  rank 16 -> 93%, 0.092M, 922 p/f
  rank 32 -> 93%, 0.184M, 1843 p/f
  rank 64 -> 95%, 0.369M, 3686 p/f
Rank 4 is optimal. Higher rank = wasted params.

### Per-Fact LoRA (shared A, separate B per fact)
  rank 1, 50 facts -> 96% recall, 993 p/f
  rank 2, 50 facts -> 96% recall, 1987 p/f
Prevents interference but less param-efficient at scale than shared LoRA.

### Orthogonal LoRA init
  Matches standard LoRA (94% vs 93% at 100 facts). No significant benefit.

### Distillation (KL div to base model, no replay)
  KILLS recall: 10% at 20 facts, 22% at 50 facts.
  Teacher pulls too hard toward base distribution.
  Replay is the correct PPL fix, not distillation.

### Optimal Recipe
  LoRA rank 4, alpha 8, 1 middle layer, down_proj only
  10% replay, 3 epochs, LR 1e-4
  Merge: W += (B @ A) * scale (lossless)
  Cost: 23K params, 93-96% recall, zero forgetting, PPL +0.41

### V10-1.2B Projection
  d_model=2048, FFN=8192, 6 GQA layers
  rank 4 on 1 layer: 4 * (8192 + 2048) = 40,960 params
  1000 facts: 41 p/f | 10000 facts: 4.1 p/f


## R28 — Param Floor Results (Script 3, 2026-08-31)

### The discovery: RANK 1 IS ENOUGH

| Rank | Params | P/F | Recall | Known Δ | PPL Δ |
|------|--------|-----|--------|---------|-------|
| **1** | **5,760** | **58** | **93%** | **+5** | **+0.03** |
| 2 | 11,520 | 115 | 94% | +5 | +0.36 |
| 3 | 17,280 | 173 | 93% | +5 | +0.37 |
| 4 | 23,040 | 230 | 93% | +5 | +0.41 |

Rank 1 on 100 facts: 93% recall, zero forgetting, PPL +0.03 (negligible),
with only 5,760 params (0.001% of model) and 58 params/fact.

Rank 1 is the floor. Going below rank 1 is impossible for LoRA (rank 0 = no update).

### Key findings

1. RANK 1 DOMINATES: 93% recall at 5,760 params. PPL +0.03 is essentially
   zero degradation. This is the optimal operating point.

2. RANK 1 ON 2 LAYERS: same recall (93%) but 2x params (11.5K) and worse
   PPL (+0.43). Spreading rank 1 across layers adds nothing.

3. SHARED-A ACROSS LAYERS: 93% recall, 26.6K params. Worse than single-layer
   rank 4 (23K). Sharing A saves nothing because A is already small (rank*in_dim).

4. SVD PRUNE rank4->rank2: 88% recall (dropped from 93%), PPL +1.26 (worse).
   Post-train pruning loses information. Training rank 2 directly is better
   (94% recall, PPL +0.36).

### The final law

For rank-r LoRA on 1 layer down_proj, frozen base + 10% replay:
  params = r * (in_dim + out_dim)
  At rank 1: params = in_dim + out_dim = 5,760 (Qwen 0.5B)
  
  This is THE FLOOR. One rank-1 update = one direction in weight space.
  100 facts share this single direction with 93% recall.
  
  params_per_fact = 5,760 / n_facts
  100 facts: 58 p/f | 1000 facts: 5.8 p/f | 10000 facts: 0.58 p/f

### Why rank 1 works
A single rank-1 update delta_W = B @ A (where B is [out,1] and A is [1,in])
adds one direction to the weight matrix. The QA training format creates a
shared subspace: all facts follow the pattern "Question: X Answer: Y" -
the model learns the PATTERN (one direction) and fills in the specific
values from the input. The rank-1 update shifts the model's QA-answering
capability; the specific answers come from the input context.

### V10-1.2B projection
  rank 1 on 1 layer: 8192 + 2048 = 10,240 params
  1000 facts: 10.2 p/f | 10000 facts: 1.02 p/f
  This is essentially FREE - 10K params on a 1.3B model.


## R28 — Rank-1 Saturation Ceiling (Script 4, 2026-08-31)

### Result: NO SATURATION up to 1000 facts

| Facts | Recall | Known Δ | PPL Δ | Params | P/F | Time |
|-------|--------|---------|-------|--------|-----|------|
| 50 | 100% | +4 | -0.01 | 5,760 | 115 | ~6s |
| 100 | 100% | +5 | +0.13 | 5,760 | 58 | ~12s |
| 250 | 100% | +4 | +0.06 | 5,760 | 23 | ~30s |
| 500 | 100% | +5 | +0.12 | 5,760 | 11.5 | ~60s |
| 1000 | 100% | +5 | +0.11 | 5,760 | 5.8 | ~120s |

### Key finding
Rank 1 LoRA has NO saturation ceiling. 100% recall at 1000 facts.
5,760 params total, 5.8 params/fact at 1000 facts.

The recall is 100% because the BASELINE already shows 100% — the model
already knows these facts (math is computable, elements are known, capitals
are common). The LoRA is not injecting new knowledge, it's reinforcing
existing knowledge.

### Critical realization
The test facts (math, elements, capitals) are things Qwen 0.5B ALREADY KNOWS.
Baseline shows I=100% for all fact counts. This means:
  - The LoRA is NOT being tested on knowledge injection
  - It's being tested on knowledge PRESERVATION (which frozen base guarantees)
  - We need facts the model DOESN'T know to test real injection

### Next step
Need facts that Qwen 0.5B gets WRONG at baseline. Options:
  - Obscure math (large primes, specific factorials)
  - Made-up facts (synthetic: "What is the capital of Zzzland? -> Quux")
  - Rare trivia the model likely hasn't seen
  - Custom domain knowledge (API docs, config values)

## 2026-09-01 Refactor session (b)
- Fixed pytest suite-killer: 11 script-style tests converted to pytest (import-time sys.exit/unlink crashed whole suite on Windows).
- Bit-exact parity restored: FlashOptimConfig (+strength_bonus), CrossLayerKV (spec weights 150/-100/-2), GlaAttention (QR/SVD proj + graduated penalty).
- engine.py: seeds persisted to DB (query_best_configs == archive best now).
- misc_sim.py: delegation simulators now build+cache legacy domain instances (was infinite recursion via JSONSpecDomain). Specs kara/hqe_kv/sparse_attn got real params.
- Deleted 7 dead scripts (V2/V7/V8/V9 refs), renamed test_sheet_v9->v10, merged _nullcontext x2 -> contextlib.nullcontext.
- Full suite: 1225 passed / 0 failed / 0 skipped.
- NOTE: two explore subagents stalled (b6eeba29, a6464224) - abandoned; survey done manually.
