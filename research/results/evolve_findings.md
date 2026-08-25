# ForgeEvolve Findings — Good Ideas List

Curated from the `boot` run (20 gen, 500 generators, 57 domains) on 2026-08-24.
Source: `research/results/evolve_boot_summary.json`, `evolve_boot_ideas.md`.

Each entry has a confidence tag based on how many top configs converge on
the same parameter region (high = 10+ configs agree, med = 3-9, low = 1-2).

---

## 1. W8A8 Quantization: fp8 + per-channel + high SmoothQuant alpha
- **Confidence**: HIGH (top 15 configs all fp8, all per-channel)
- **Best config**: `mode=fp8, per_channel=true, smoothquant_alpha=0.987, calib_samples=227`
- **Result**: SQNR=84.19 dB
- **Pattern**: fp8 dominates int8. per-channel consistently beats per-tensor.
  SmoothQuant alpha near 1.0 (aggressive activation smoothing) is optimal.
- **Action**: Already aligned with V7 production path (NVFP4/W8A8 first-class).
  Consider testing alpha > 0.99 in a deep run to see if it keeps improving.

## 2. PagedEvictKV: small pages (16-64) + LRU
- **Confidence**: HIGH (top 15 configs all LRU, page_size 16-64)
- **Best config**: `page_size=64, n_pages=64, eviction_policy=lru`
- **Result**: hit_rate=1.0, mem_eff=0.25
- **Pattern**: LRU dominates. Small page sizes (16-64) give best hit rates.
  Importance/LFU only competitive at larger page sizes (80+).
- **Action**: Wire page_size=64 + LRU as the default for PagedEvictKV in
  `kv_backend.py`. Test against current `s4r` production cache.

## 3. StreamingKV: small chunks + low overlap
- **Confidence**: HIGH (top configs all chunk_size <= 128, overlap <= 0.1)
- **Best config**: `chunk_size=128, n_sink=4, overlap=0.1`
- **Result**: coverage_err=0.0007
- **Pattern**: Counter-intuitive — smaller chunks with minimal overlap
  maximize coverage. The "bigger overlap = safer" heuristic is wrong for
  this domain. n_sink=4 is consistent (matches the attention sink pattern).
- **Action**: Challenge the current StreamingKV defaults. Test chunk=128,
  overlap=0.1 on real 32K context inference.

## 4. SpeculativeDecode: 7 draft tokens, tiny draft model, high threshold
- **Confidence**: HIGH (top configs converge on n_draft=7, ratio~0.125, thresh~0.95)
- **Best config**: `n_draft_tokens=7, draft_model_ratio=0.125, acceptance_threshold=0.948`
- **Result**: 3.33x speedup, 0.667 acceptance rate
- **Pattern**: 7 draft tokens is the sweet spot (not 4, not 8). A very small
  draft model (12.5% of main) with a high acceptance threshold (0.95) gives
  the best throughput. Temperature ~1.0 (near-greedy with slight randomness).
- **Action**: Update MTP speculative decoding defaults in `decoding/mtp.py`.
  Current defaults may be suboptimal. Sweep n_draft in [5,6,7,8] on real model.

## 5. KvRecompute: aggressive full recompute on 12GB
- **Confidence**: MED (best config is all 16 layers, but selective strategy)
- **Best config**: `recompute_layers=16, strategy=selective, threshold=0.9`
- **Result**: quality=0.92
- **Pattern**: On 12GB VRAM, recomputing all 16 layers wins. The selective
  strategy with a high threshold (0.9) preserves quality. This confirms the
  mixed-approach directive — trade compute for VRAM aggressively.
- **Action**: Already aligned with `CheckpointRecompute` finding (16 layers,
  selective, block_size=512). Consider as the default for 32K context.

## 6. FP8 Training: e5m2 + mu_scaling
- **Confidence**: MED (e5m2 appears 2x in top configs)
- **Best config**: `autocast_mode=e5m2, mu_scaling=true, loss_scale=3645`
- **Result**: overflow=0.0
- **Pattern**: e5m2 beats e4m3 for training (more exponent range = fewer
  overflows). mu_scaling (scaling the momentum update) helps stability.
  High loss_scale (3000+) handles small gradients.
- **Action**: Test e5m2 + mu_scaling in `Fp8TrainingConfig` against current
  bf16 training path. If stable, adopt for V7 training.

## 7. CPUAdamW: minimal offload + int4 compression + deep prefetch
- **Confidence**: MED (top configs agree on int4, prefetch_depth=5)
- **Best config**: `offload_ratio=0.005, prefetch_depth=5, compression=int4, update_freq=13`
- **Result**: throughput=2833 tok/s, latency=26ms
- **Pattern**: Almost no offload (0.5%) but int4 compression on the
  transferred gradients. Deep prefetch (5) hides latency. High update_freq
  (13) batches CPU updates.
- **Action**: Test int4 gradient compression in `cpu_offload.py` training
  path. Current path may use uncompressed transfers.

## 8. RoPE: very high theta + linear scaling
- **Confidence**: MED (top configs agree on theta ~10M, linear scaling)
- **Best config**: `theta=9997342, scaling_type=linear, scaling_factor=1.98`
- **Result**: scaling metric=3.47
- **Pattern**: theta near 10M (current production is 1M) with linear
  scaling factor ~2.0. This extends context without NTK-style interpolation.
- **Action**: Re-challenge the current RoPE theta=1M default. Test theta=10M
  + linear scaling on 32K context. May improve long-range attention.

## 9. MoE Routing: 5 experts, top-3, switch router, shared expert
- **Confidence**: MED (top configs agree on shared_expert=true, switch router)
- **Best config**: `n_experts=5, top_k=3, router_mode=switch, load_balance_weight=0.002, shared_expert=true`
- **Result**: balance=0.658, util=1.0
- **Pattern**: Small expert count (5) with high top_k (3) gives 100%
  utilization. Switch router with shared expert. Very low load balance
  weight (0.002) — don't fight the router too hard.
- **Action**: Relevant for `forgelm_v7_moe` preset. Test 5-expert top-3
  vs current MoE config.

## 10. Factorized Embeddings: rank=64, SVD init, high tie factor
- **Confidence**: LOW (single top config, but err is extremely low)
- **Best config**: `rank=64, init_mode=svd, tie_factor=0.153, vocab_size=47660`
- **Result**: reduction=0.80, err=8.96e-06
- **Pattern**: Rank-64 factorization with SVD init gives near-lossless
  embedding compression (err < 1e-5) at 80% parameter reduction.
- **Action**: Already in V7 (factorized embeddings). Confirm rank=64 is
  the production value; if not, test it.

---

## Domains needing more search (at floor score after 20 gen)
These domains scored negative — the surrogate hasn't found good regions yet.
A `deep` (100 gen) or `ultra` (500 gen) run with warm-start should help:
- `BitnetConfig` (-27.4) — ternary quant, search space may be too narrow
- `SharqQuant` (-18.9) — adaptive quant not finding good n_levels
- `OffqQuant` (-16.8) — offset quantization not converging
- `MixedPrecision` (-16.4) — bit allocation not finding good splits
- `SparseAttention` (-27.7) — compact strategy not working at these budgets
- `KVEviction` (-41.4) — paged strategy at low budget has high error
- `GLA` (-50.0) — latent_dim too compressed for d_model=2048

## Bugs fixed this session
1. **OptimizerConfig**: raw CUDA tensor in metadata → JSON serialization failure
2. **CrossLayerKV**: `.norm()` applied to wrong operand → negative recon_err
   → score inflated to 218k+ (polluting archive)
3. **Engine metadata mismatch**: `self.all_results[-1]` used for all
   discoveries → metadata didn't match the saved config
4. **JSON round-trip type coercion** (4 domains): GroupQuant, KvZipKV,
   RotorQuantKV, SyntheticDomain crashed when warm-started configs had
   string-typed values from JSON serialization

## Canonical knowledge system (new)
- **Permanent generator/surrogate persistence**: The best-ever generator
  and surrogate weights per domain are stored in `canonical_generators`
  and `canonical_surrogate` DB tables. Every new run loads these at
  startup (regardless of profile), so findings accumulate across ALL runs.
- **Shared DB**: All profiles (boot/deep/ultra) now use a single
  `forge_evolve.db` instead of per-profile DBs. Knowledge transfers
  freely between runs.
- **Migration**: Previous findings from `forge_evolve_boot.db`,
  `forge_evolve_deep.db`, and `forge_evolve_all.db` were migrated into
  the shared DB. 57 canonical generators + 57 canonical surrogates.

## Run 2 improvements (with canonical warm-start)
Cross-domain comparison (run 1 → run 2, both boot profile):
- SpeculativeDecode: 55.06 → 57.05 (improved from warm-started generators)
- Fp8TrainingConfig: 36.38 → 37.17
- MtpConfig: 23.58 → 27.69
- MoeRouting: 24.71 → 24.86
- OptimizerConfig: 15.99 → 24.52 (now working, was crashing before)
- CrossLayerKV: 218453 → -255.5 (bug fixed, score now correct)

---

## 10-Loop Run (2026-08-24, post-dedup cleanup)

**Setup**: 10 consecutive boot loops, canonical warm-start, 4 parallel domains,
deduped DB (0 duplicates). 19,436 unique discoveries, 53 canonical entries.

### Score progression (top 10 domains across 10 loops)

| Domain | L1 | L5 | L10 | Trend |
|--------|----|----|-----|-------|
| W8A8Quant | 177.88 | 177.69 | 177.56 | Stable (converged) |
| PagedEvictKV | 128.11 | 128.11 | 128.11 | Converged |
| StreamingKV | 97.13 | 97.13 | 97.13 | Converged |
| XQuantKV | 95.00 | 95.00 | 95.00 | Converged |
| KvRecompute | 74.55 | 74.55 | 74.55 | Converged |
| SpeculativeDecode | 57.05 | 57.05 | 57.05 | Converged |
| BatchedDecode | 44.21 | 44.25 | 44.25 | Improved |
| Fp8TrainingConfig | 36.72 | 37.62 | 36.20 | Oscillating |
| RopeConfig | 36.15 | 36.15 | 36.15 | Converged |
| CpuAdamwConfig | 30.00 | 30.00 | 30.00 | Converged |

**Key insight**: Most domains converge within 1-2 loops with canonical
warm-start. The remaining search space is explored but the best configs
are already locked in. This means the canonical system works — knowledge
accumulates and new runs start from the best-known point.

### Updated top configs (10-loop best)

#### 1. W8A8Quant (score=179.47) — CONFIDENCE: VERY HIGH
- `mode=fp8, calib_samples=65, per_channel=false, smoothquant_alpha=0.999`
- SQNR=86.73 dB
- **Change from run 1**: alpha moved from 0.987 → 0.999 (more aggressive)
- **Action**: Test smoothquant_alpha=0.999 in production W8A8 path

#### 2. PagedEvictKV (score=128.11) — CONFIDENCE: VERY HIGH
- `page_size=64, n_pages=64, eviction_policy=lru`
- hit_rate=1.0, mem_eff=0.25
- **Unchanged**: LRU + small pages confirmed across all 10 loops

#### 3. StreamingKV (score=97.13) — CONFIDENCE: VERY HIGH
- `chunk_size=128, n_sink=5, overlap=0.473`
- **Change**: overlap moved from 0.1 → 0.47 (more overlap helps with n_sink=5)
- **Action**: Re-test with overlap=0.47 on real 32K context

#### 4. XQuantKV (score=95.00) — CONFIDENCE: HIGH
- `recomputation_ratio=1.0, quant_bits=4, checkpoint_interval=16`
- Full recompute + 4-bit KV quant. Aggressive but high-scoring.
- **Action**: Wire into `kv_backend.py` as an option for 32K+ context

#### 5. SpeculativeDecode (score=57.05) — CONFIDENCE: VERY HIGH
- `n_draft_tokens=7, draft_model_ratio=0.10, acceptance_threshold=0.95`
- 5.37x speedup, 0.677 acceptance
- **Unchanged**: 7 draft tokens confirmed as optimal across all loops

#### 6. BatchedDecode (score=44.25) — CONFIDENCE: HIGH
- `max_batch_size=15, padding_strategy=left, merge_window_ms=52, max_seq_diff=0`
- **Change**: padding_strategy moved from "dynamic" → "left" with max_seq_diff=0
- **Action**: Test left-padding + zero seq diff in BatchedDecoding for
  equal-length prompt batches

#### 7. Fp8TrainingConfig (score=38.79) — CONFIDENCE: MED
- `autocast_mode=e5m2, smooth_swiglu=false, mu_scaling=true, loss_scale=3754`
- **Change**: smooth_swiglu=false (was true), higher loss_scale
- **Action**: Test e5m2 + mu_scaling + no SwiGLU smoothing in training

#### 8. RopeConfig (score=36.15) — CONFIDENCE: HIGH
- `theta=10000000, scaling_type=linear, scaling_factor=0.901`
- **Change**: scaling_factor moved from 1.98 → 0.901 (less aggressive)
- **Action**: Test theta=10M + linear scaling_factor=0.9 on 32K context

#### 9. CpuAdamwConfig (score=30.00) — CONFIDENCE: HIGH
- `offload_ratio=0.0, prefetch_depth=7, compression=int4, update_freq=15`
- **Change**: offload_ratio → 0 (no offload at all), deeper prefetch
- **Pattern**: int4 gradient compression + deep prefetch is the win, not offload
- **Action**: Add int4 gradient compression to CPU optimizer path

#### 10. MtpConfig (score=27.69) — CONFIDENCE: HIGH
- `n_heads=4, loss_weight=0.495, share_weights=true, depth_ratio=0.986`
- **Action**: Already in V7 MTP. Confirm share_weights=true is production.

#### 11. MoeRouting (score=24.87) — CONFIDENCE: HIGH
- `n_experts=4, top_k=3, router_mode=switch, load_balance_weight=6e-5, shared_expert=true`
- **Change**: n_experts 5→4, load_balance_weight 0.002→6e-5 (even lower)
- **Action**: Test 4-expert top-3 for V7-MoE preset

#### 12. OptimizerConfig (score=24.52) — CONFIDENCE: HIGH
- `opt_type=adamw, lr=0.00995, beta1=0.803, beta2=0.999, weight_decay=0.00025`
- **Pattern**: Low beta1 (0.8) + very high beta2 (0.999) + low wd
- **Action**: Test beta1=0.8 in SFT training (current default is likely 0.9)

#### 13. HadamardKV (score=19.70) — CONFIDENCE: MED
- Converged quickly. Hadamard rotation for KV quantization.
- **Action**: Already have `hadamard_kv.py`. Confirm production config.

#### 14. FactorizedEmbed (score=19.35) — CONFIDENCE: HIGH
- `rank=64, init_mode=svd, tie_factor=0.746, vocab_size=65486`
- **Change**: tie_factor 0.153→0.746 (much higher tying)
- **Action**: Test tie_factor=0.75 in V7 factorized embeddings

#### 15. RotorQuantKV (score=17.31) — CONFIDENCE: MED
- `rot_type=random, n_rotations=1, quant_bits=4`
- Single random rotation + 4-bit. Simple and effective.
- **Action**: Test random rotation (not DCT) in `rotorquant_kv.py`

### New ideas from 10-loop run (not in prior findings)

#### A. ExpertHotload (score=8.83, 78 discoveries)
- `n_hot_experts=4, prefetch_ahead=4, cache_strategy=lfu, disk_cache_size=4096`
- 0% miss rate with LFU + 4 hot experts + 4-step prefetch
- **Action**: Wire into AirMoE disk offload for V7-MoE. LFU cache with
  4 hot experts and 4-step prefetch should eliminate expert load stalls.

#### B. CheckpointRecompute (score=8.94, 36 discoveries)
- `n_checkpoint_layers=16, recompute_strategy=selective, block_size=512`
- All 16 layers + selective + block_size=512
- **Action**: Set as default for 32K context training. Block size 512
  matches typical training sequence length.

#### C. GradAccumConfig (score=11.10, 37 discoveries)
- `accum_steps=5, micro_batch=15, grad_clip=0.9999, sync_freq=15`
- Effective batch=75. High grad_clip (near 1.0). Sync every 15 steps.
- **Action**: Test grad_clip=1.0 (essentially no clipping) + sync_freq=15
  in SFT training. May be better than conservative clipping.

#### D. SamplingConfig (score=10.45, 27 discoveries)
- `temperature=1.98, top_p=0.989, top_k=69, repetition_penalty=1.011, frequency_penalty=0.014`
- High temperature + high top_p + moderate top_k. Very low penalties.
- **Action**: Test as generation default for creative/diverse tasks.

#### E. ModConfig (score=10.09, 12 discoveries)
- `keep_fraction=0.5, router_type=linear, aux_loss_weight=3e-8, n_skip_layers=15`
- Skip 15/16 layers with linear router. Near-zero aux loss weight.
- **Action**: Test in V7 MoD path. Near-zero aux loss may be better
  than fighting the router.

#### F. LossConfig (score=2.07, 5 discoveries)
- `loss_type=focal, label_smoothing=0.289, focal_gamma=4.93, temperature=1.95`
- Focal loss with high gamma (4.93) + high temperature (1.95)
- **Action**: Test focal loss in SFT training. Current path likely uses
  CE with label smoothing. Focal + high gamma may help on imbalanced data.

#### G. SchedulerConfig (score=10.03, 15 discoveries)
- `sched_type=cosine, warmup_steps=0, min_lr_ratio=0.379, decay_steps=101`
- No warmup + cosine + min_lr_ratio=0.38
- **Action**: Test zero-warmup cosine schedule. May work for short SFT runs.

### Domains still at floor (need deep/ultra run)
- `CrossLayerKV` (-255.5) — share_ratio=1.0 with avg mode has high recon_err
- `GlaAttention` (-50.0) — latent_dim too small for d_model=2048
- `KVEvictionDomain` (-45.1) — paged at low budget
- `BitnetConfig` (-27.5) — ternary quant error
- `SparseAttention` (-27.4) — compact strategy at low budget
- `SharqQuant` (-18.3) — adaptive quant not finding good levels
- `OffqQuant` (-17.2) — offset quantization
- `MixedPrecision` (-16.5) — bit allocation

### Dedup system verification
- **0 duplicate groups** in DB after 10 loops (was 529 before cleanup)
- **19,436 unique discoveries** (was 14,711 with 2,844 dupes)
- **53 canonical entries** (regenerated from clean state)
- DB dedup uses INSERT-or-UPDATE: same config with higher score updates
  the existing row, never inserts a duplicate
