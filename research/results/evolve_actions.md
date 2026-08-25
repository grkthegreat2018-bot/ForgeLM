# ForgeEvolve → ForgeEngine/Trainer Action List

Prioritized ideas from the 10-loop evolution run (19,436 unique discoveries).
Each item has a target file, config, and expected impact.

---

## Tier 1: High-confidence, high-impact (do first)

### 1. W8A8: smoothquant_alpha=0.999
- **Target**: `research/inference/quant/w8a8_quant.py`
- **Config**: `mode=fp8, per_channel=true, smoothquant_alpha=0.999, calib_samples=65`
- **Evidence**: 179.47 SQNR=86.73 dB, top 20 configs all fp8+alpha>0.99
- **Expected**: +2 dB SQNR over current alpha=0.987

### 2. PagedEvictKV: page_size=64 + LRU as default
- **Target**: `research/inference/kv_backend.py`, `research/inference/kv/paged_evict.py`
- **Config**: `page_size=64, n_pages=64, eviction_policy=lru`
- **Evidence**: 128.11 score, hit_rate=1.0, stable across all 10 loops
- **Expected**: Better than s4r for long context with memory pressure

### 3. SpeculativeDecode: n_draft=7, ratio=0.10, threshold=0.95
- **Target**: `research/decoding/mtp.py`, `research/inference/decoding.py`
- **Config**: `n_draft_tokens=7, draft_model_ratio=0.10, acceptance_threshold=0.95`
- **Evidence**: 57.05 score, 5.37x speedup, converged across all loops
- **Expected**: 5x+ decode speedup with 10% draft model size

### 4. XQuantKV: full recompute + 4-bit KV quant
- **Target**: `research/inference/kv/xquant_kv.py`
- **Config**: `recomputation_ratio=1.0, quant_bits=4, checkpoint_interval=16`
- **Evidence**: 95.00 score, stable
- **Expected**: 4x KV compression for 32K+ context

### 5. CPUAdamW: int4 gradient compression + deep prefetch
- **Target**: `research/training/cpu_offload.py`
- **Config**: `compression=int4, prefetch_depth=7, update_freq=15`
- **Evidence**: 30.00 score, pattern: compression is the win, not offload
- **Expected**: 2x CPU optimizer throughput via int4 gradient transfer

---

## Tier 2: Medium-confidence, test on real model

### 6. RoPE: theta=10M + linear scaling_factor=0.9
- **Target**: `research/config.py` (preset), `research/keys/rope_key.py`
- **Config**: `theta=10000000, scaling_type=linear, scaling_factor=0.901`
- **Evidence**: 36.15 score, converged
- **Expected**: Better long-range attention at 32K context

### 7. Fp8 Training: e5m2 + mu_scaling + no SwiGLU smoothing
- **Target**: `research/training/fp8_training.py` (if exists), or `sft_train.py`
- **Config**: `autocast_mode=e5m2, smooth_swiglu=false, mu_scaling=true, loss_scale=3754`
- **Evidence**: 38.79 score, zero overflows
- **Expected**: Stable FP8 training with 2x memory savings

### 8. OptimizerConfig: beta1=0.8 + beta2=0.999
- **Target**: `research/training/sft_train.py`
- **Config**: `lr=0.01, beta1=0.803, beta2=0.999, weight_decay=0.00025`
- **Evidence**: 24.52 score
- **Expected**: Faster convergence with low beta1 (less momentum lag)

### 9. MoE Routing: 4 experts, top-3, near-zero aux loss
- **Target**: `research/moe/` (V7-MoE preset)
- **Config**: `n_experts=4, top_k=3, router_mode=switch, load_balance_weight=6e-5, shared_expert=true`
- **Evidence**: 24.87 score, 99.1% balance, 100% utilization
- **Expected**: Better expert utilization than current MoE config

### 10. FactorizedEmbed: tie_factor=0.75
- **Target**: `research/keys/factorized_embed_key.py`
- **Config**: `rank=64, init_mode=svd, tie_factor=0.746`
- **Evidence**: 19.35 score, err=8.5e-06
- **Expected**: Better embedding quality with higher tying

---

## Tier 3: Novel ideas to explore

### 11. ExpertHotload: LFU cache + 4-step prefetch
- **Target**: `research/moe/airmoe.py` (disk offload)
- **Config**: `n_hot_experts=4, prefetch_ahead=4, cache_strategy=lfu, disk_cache_size=4096`
- **Evidence**: 8.83 score, 0% miss rate
- **Expected**: Eliminate expert load stalls in V7-MoE

### 12. CheckpointRecompute: all 16 layers + selective + block=512
- **Target**: `research/training/` (gradient checkpointing)
- **Config**: `n_checkpoint_layers=16, recompute_strategy=selective, block_size=512`
- **Evidence**: 8.94 score
- **Expected**: 50%+ VRAM savings for 32K context training

### 13. GradAccum: grad_clip=1.0 + sync_freq=15
- **Target**: `research/training/sft_train.py`
- **Config**: `accum_steps=5, micro_batch=15, grad_clip=1.0, sync_freq=15`
- **Evidence**: 11.10 score
- **Expected**: Less gradient clipping = faster convergence on clean data

### 14. Focal loss with high gamma
- **Target**: `research/training/` (loss function)
- **Config**: `loss_type=focal, label_smoothing=0.289, focal_gamma=4.93, temperature=1.95`
- **Evidence**: 2.07 score (low absolute, but novel direction)
- **Expected**: Better learning on imbalanced SFT data

### 15. MoD: skip 15/16 layers with near-zero aux loss
- **Target**: `research/keys/mod_key.py`
- **Config**: `keep_fraction=0.5, router_type=linear, aux_loss_weight=3e-8, n_skip_layers=15`
- **Evidence**: 10.09 score
- **Expected**: 2x inference speedup with minimal quality loss

---

## ForgeEvolve system improvements (for the evolution system itself)

### 16. Converged domains → switch to deep profile
- 10 of 57 domains are fully converged (same score across all 10 loops)
- These domains waste compute re-confirming the same configs
- **Action**: Add a "convergence detector" — if best score doesn't improve
  for 3 consecutive loops, switch that domain to a deeper profile (more
  generations, larger population) to escape the local optimum

### 17. Floor-score domains need different search strategy
- 8 domains still at negative scores (CrossLayerKV, GLA, BitnetConfig, etc.)
- The boot profile (10 gen, 300 generators) isn't enough for these
- **Action**: Run a targeted `deep` profile (100 gen) for just these 8
  domains, warm-started from current canonical

### 18. Cross-domain parameter patterns
- The ideas report shows patterns like `acceptance_threshold=0.95` appearing
  across multiple domains. These could be meta-parameters.
- **Action**: Add a meta-optimization layer that shares parameters across
  related domains (e.g., all quantization domains share `n_bits` and
  `group_size` priors)
