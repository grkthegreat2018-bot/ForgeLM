# ForgeEvolve 'deep' Run: Top Optimization Ideas
Generated: 2026-08-23 20:29:54
Domains: 56

## Tier 1: Top 50 Configurations

| Rank | Domain | Score | Config |
|------|--------|-------|--------|
| 1 | cross_layer_kv | 218453.47 | `{"share_ratio": 1.0, "n_share_groups": 1, "share_mode": "max"}` |
| 2 | cross_layer_kv | 204338.32 | `{"share_ratio": 0.8825158476829529, "n_share_groups": 1, "share_mode": "max"}` |
| 3 | cross_layer_kv | 189174.32 | `{"share_ratio": 0.75, "n_share_groups": 1, "share_mode": "max"}` |
| 4 | cross_layer_kv | 181117.47 | `{"share_ratio": 0.7066544890403748, "n_share_groups": 1, "share_mode": "max"}` |
| 5 | cross_layer_kv | 172685.57 | `{"share_ratio": 0.6465373635292053, "n_share_groups": 1, "share_mode": "max"}` |
| 6 | cross_layer_kv | 154448.52 | `{"share_ratio": 0.5, "n_share_groups": 1, "share_mode": "max"}` |
| 7 | cross_layer_kv | 133750.25 | `{"share_ratio": 0.4071280360221863, "n_share_groups": 1, "share_mode": "max"}` |
| 8 | cross_layer_kv | 109200.58 | `{"share_ratio": 0.25, "n_share_groups": 1, "share_mode": "max"}` |
| 9 | cross_layer_kv | 94567.27 | `{"share_ratio": 0.2118055522441864, "n_share_groups": 1, "share_mode": "max"}` |
| 10 | cross_layer_kv | 54592.85 | `{"share_ratio": 0.0, "n_share_groups": 1, "share_mode": "max"}` |
| 11 | cross_layer_kv | 52548.89 | `{"share_ratio": 0.6706153750419617, "n_share_groups": 5, "share_mode": "max"}` |
| 12 | cross_layer_kv | 47592.89 | `{"share_ratio": 0.0, "n_share_groups": 2, "share_mode": "max"}` |
| 13 | cross_layer_kv | 37144.01 | `{"share_ratio": 0.013628654181957245, "n_share_groups": 5, "share_mode": "max"}` |
| 14 | cross_layer_kv | 32789.22 | `{"share_ratio": 0.0014665212947875261, "n_share_groups": 7, "share_mode": "max"}` |
| 15 | cross_layer_kv | 25689.18 | `{"share_ratio": 1.0, "n_share_groups": 1, "share_mode": "avg"}` |
| 16 | cross_layer_kv | 24022.85 | `{"share_ratio": 0.8968042135238647, "n_share_groups": 1, "share_mode": "learn...` |
| 17 | cross_layer_kv | 24009.18 | `{"share_ratio": 0.9689676761627197, "n_share_groups": 2, "share_mode": "learn...` |
| 18 | cross_layer_kv | 23146.81 | `{"share_ratio": 0.840736985206604, "n_share_groups": 1, "share_mode": "avg"}` |
| 19 | cross_layer_kv | 21285.83 | `{"share_ratio": 0.7117084264755249, "n_share_groups": 1, "share_mode": "avg"}` |
| 20 | cross_layer_kv | 20962.47 | `{"share_ratio": 0.9959172606468201, "n_share_groups": 5, "share_mode": "avg"}` |
| 21 | w8a8_quant | 179.43 | `{"mode": "fp8", "calib_samples": 938, "per_channel": true, "smoothquant_alpha...` |
| 22 | w8a8_quant | 178.44 | `{"mode": "fp8", "calib_samples": 975, "per_channel": true, "smoothquant_alpha...` |
| 23 | w8a8_quant | 178.38 | `{"mode": "fp8", "calib_samples": 1020, "per_channel": true, "smoothquant_alph...` |
| 24 | w8a8_quant | 178.19 | `{"mode": "fp8", "calib_samples": 480, "per_channel": true, "smoothquant_alpha...` |
| 25 | w8a8_quant | 177.92 | `{"mode": "fp8", "calib_samples": 516, "per_channel": true, "smoothquant_alpha...` |
| 26 | w8a8_quant | 177.64 | `{"mode": "fp8", "calib_samples": 522, "per_channel": true, "smoothquant_alpha...` |
| 27 | w8a8_quant | 177.06 | `{"mode": "fp8", "calib_samples": 1008, "per_channel": true, "smoothquant_alph...` |
| 28 | w8a8_quant | 176.40 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 29 | w8a8_quant | 176.05 | `{"mode": "fp8", "calib_samples": 256, "per_channel": true, "smoothquant_alpha...` |
| 30 | w8a8_quant | 175.73 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 31 | w8a8_quant | 175.17 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 32 | w8a8_quant | 174.47 | `{"mode": "fp8", "calib_samples": 256, "per_channel": true, "smoothquant_alpha...` |
| 33 | w8a8_quant | 172.94 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 34 | paged_evict_kv | 128.11 | `{"page_size": 64, "n_pages": 64, "eviction_policy": "lru"}` |
| 35 | paged_evict_kv | 128.09 | `{"page_size": 32, "n_pages": 128, "eviction_policy": "lru"}` |
| 36 | paged_evict_kv | 127.13 | `{"page_size": 16, "n_pages": 256, "eviction_policy": "lru"}` |
| 37 | paged_evict_kv | 124.62 | `{"page_size": 72, "n_pages": 64, "eviction_policy": "lru"}` |
| 38 | paged_evict_kv | 121.46 | `{"page_size": 29, "n_pages": 178, "eviction_policy": "lru"}` |
| 39 | paged_evict_kv | 120.15 | `{"page_size": 16, "n_pages": 324, "eviction_policy": "lfu"}` |
| 40 | paged_evict_kv | 118.55 | `{"page_size": 56, "n_pages": 106, "eviction_policy": "lru"}` |
| 41 | paged_evict_kv | 118.20 | `{"page_size": 16, "n_pages": 349, "eviction_policy": "lru"}` |
| 42 | paged_evict_kv | 117.93 | `{"page_size": 52, "n_pages": 65, "eviction_policy": "lfu"}` |
| 43 | paged_evict_kv | 116.72 | `{"page_size": 50, "n_pages": 67, "eviction_policy": "lfu"}` |
| 44 | paged_evict_kv | 115.21 | `{"page_size": 107, "n_pages": 64, "eviction_policy": "importance"}` |
| 45 | paged_evict_kv | 114.90 | `{"page_size": 32, "n_pages": 217, "eviction_policy": "lru"}` |
| 46 | paged_evict_kv | 114.32 | `{"page_size": 92, "n_pages": 79, "eviction_policy": "importance"}` |
| 47 | paged_evict_kv | 114.15 | `{"page_size": 23, "n_pages": 303, "eviction_policy": "lru"}` |
| 48 | paged_evict_kv | 112.47 | `{"page_size": 64, "n_pages": 128, "eviction_policy": "lru"}` |
| 49 | paged_evict_kv | 112.26 | `{"page_size": 68, "n_pages": 122, "eviction_policy": "importance"}` |
| 50 | paged_evict_kv | 109.28 | `{"page_size": 17, "n_pages": 496, "eviction_policy": "importance"}` |

## Tier 2: Best Per Domain

### aaac_quant (score=9.98)
- **Config**: `{"n_codebooks": 4, "codebook_size": 482, "n_bits": 2}`
- **Metadata**: `{"n_codebooks": 5, "compression": 3.2}`

### activation_quant (score=9.63)
- **Config**: `{"calib_method": "percentile", "percentile": 0.9003667952804827, "smooth_alpha": 0.48856788873672485}`
- **Metadata**: `{"method": "minmax", "err": 0.11263978814323376}`

### attn_residual (score=3.25)
- **Config**: `{"k_layers": 1, "gate_init": 0.9997047781944275, "retrieval_dim": 227}`
- **Metadata**: `{"k": 1, "gate": 0.14032742381095886, "retrieval_dim": 496}`

### batched_decode (score=44.11)
- **Config**: `{"max_batch_size": 15, "padding_strategy": "left", "merge_window_ms": 94, "max_seq_diff": 7}`
- **Metadata**: `{"batch_size": 15, "throughput": 8.25}`

### beam_search (score=11.50)
- **Config**: `{"beam_width": 4, "length_penalty": 1.9984755218029022, "early_stopping": true, "diversity_penalty": 0.9994909763336182}`
- **Metadata**: `{"beam_width": 7, "accuracy": 0.7777777777777778}`

### bitnet_config (score=-27.38)
- **Config**: `{"learned_scale": false, "quant_mode": "ternary", "init_scale": 1.5255076587200165}`
- **Metadata**: `{"mode": "binary", "err": 0.9735062972120767}`

### checkpoint_recompute (score=8.94)
- **Config**: `{"n_checkpoint_layers": 16, "recompute_strategy": "selective", "block_size": 512}`
- **Metadata**: `{"mem_saved_gb": 0.40265315771102905, "recompute_overhead_ms": 3.0000001192092896}`

### conv_config (score=4.15)
- **Config**: `{"kernel_size": 5, "stride": 1, "dilation": 3, "groups": 7, "n_conv_layers": 5}`
- **Metadata**: `{"receptive_field": 70, "params": 16056320.0}`

### cpu_adamw_config (score=30.00)
- **Config**: `{"offload_ratio": 4.6751304694225837e-07, "prefetch_depth": 5, "compression": "int4", "update_freq": 9}`
- **Metadata**: `{"throughput": 562.1758103370667, "latency": 175.12967586517334}`

### cpu_kv_offload (score=0.04)
- **Config**: `{"offload_layers": 1, "offload_threshold": 0.9999842643737793, "prefetch_size": 2048, "async_copy": true}`
- **Metadata**: `{"kv_freed_gb": 9.642996883485466e-05, "decode_lat_ms": 1.699999988079071}`

### cross_layer_kv (score=218453.47)
- **Config**: `{"share_ratio": 1.0, "n_share_groups": 1, "share_mode": "max"}`
- **Metadata**: `{"share_ratio": 0.75, "n_share_groups": "2", "share_mode": "max", "param_reduction": 0.75, "recon_err": -233.10710906982422}`

### csa_attention (score=14.81)
- **Config**: `{"top_k": 64, "pattern_type": "csa_hca_hybrid", "block_size": 8}`
- **Metadata**: `{"top_k": 614, "block_size": 9}`

### diff_attn (score=10.00)
- **Config**: `{"lambda_init": 3.4092354894710297e-07, "n_heads": 31, "softmax_sep": 0.8985720872879028}`
- **Metadata**: `{"lambda": 0.015286394394934177, "n_heads": 31}`

### expert_hotload (score=8.83)
- **Config**: `{"n_hot_experts": 4, "prefetch_ahead": 4, "cache_strategy": "lfu", "disk_cache_size": 4096}`
- **Metadata**: `{"vram_used_gb": 1.025390625, "miss_rate": 0.0}`

### factorized_embed (score=19.35)
- **Config**: `{"rank": 64, "init_mode": "svd", "tie_factor": 0.2021797150373459, "vocab_size": 65304}`
- **Metadata**: `{"rank": 511, "reduction": 0.7419552596252192, "err": 1.4152672738677767}`

### ffn_skip (score=-6.18)
- **Config**: `{"skip_threshold": 0.0036844995338469744, "n_eval_layers": 15, "skip_strategy": "cosine", "min_keep": 0.5772349014878273}`
- **Metadata**: `{"compute_saved": 0.0, "deviation": 0.0}`

### fp8_training_config (score=37.89)
- **Config**: `{"autocast_mode": "e5m2", "smooth_swiglu": false, "mu_scaling": true, "loss_scale": 3117.2025299072266}`
- **Metadata**: `{"overflow": 0.0, "mode": "e5m2"}`

### gla_attention (score=-50.00)
- **Config**: `{"latent_dim": 509, "n_heads": 4, "compression_ratio": 7.978329360485077}`
- **Metadata**: `{"latent_dim": 481, "recon_err": 1.411698701703795}`

### grad_accum_config (score=11.10)
- **Config**: `{"accum_steps": 5, "micro_batch": 15, "grad_clip": 0.9998345375061035, "sync_freq": 10}`
- **Metadata**: `{"effective_batch": 90, "noise": 0.10540925533894598}`

### group_quant (score=9.11)
- **Config**: `{"group_size": 16, "n_bits": 8, "scheme": "asymmetric"}`
- **Metadata**: `{"group_size": 128, "bits": 8, "err": 0.009398426242696344}`

### gta_attention (score=5.00)
- **Config**: `{"v_k_mix": 2.485988375156012e-07, "n_kv_heads": 15, "tie_strength": 0.9667887091636658}`
- **Metadata**: `{"mix": 0.030498670414090157, "deviation": 0.043270198168042874}`

### hqe_kv (score=4.46)
- **Config**: `{"budget_full": 0.05, "budget_int8": 0.5, "budget_int4": 0.5, "group_size": 32, "recency_decay": 1.0}`
- **Metadata**: `{"fwd_err": 0.27108611182414827, "compression": 3.079120466077805, "quant_ms": 0.9237361398233415, "n_full": 409, "n_int8": 409, "n_int4": 2867, "n_evicted": 411, "group_size": 128, "recency_decay": 1.0}`

### hybrid_offload (score=4.18)
- **Config**: `{"offload_ratio": 0.9610263705253601, "prefetch_depth": 8, "pin_memory": true, "overlap_compute": true}`
- **Metadata**: `{"vram_saved_gb": 1.4237396717071533, "latency_ms": 1.9537549018859863}`

### kara (score=1.98)
- **Config**: `{"sink_size": 2, "window_size": 130, "target_budget": 3961, "chunk_expand_size": 15}`
- **Metadata**: `{"fwd_err": 0.5145441154375067, "compression": 1.263807466831225, "kara_ms": 0.6319037334156125, "sink_size": 2, "window_size": 966, "target_budget": 2273, "chunk_expand_size": 4, "comp_seq_len": 3241}`

### kv_eviction (score=-1.71)
- **Config**: `{"strategy": "streaming", "budget": 456, "observation_window": 307, "n_sinks": 2, "window_size": 1011, "block_size": 8}`
- **Metadata**: `{"fwd_err": 0.0, "compression": 1.0, "cache_ms": 0.3, "strategy": "paged", "budget": 639, "comp_seq_len": 1024}`

### kv_recompute (score=74.55)
- **Config**: `{"recompute_layers": 16, "recompute_strategy": "selective", "threshold": 0.9}`
- **Metadata**: `{"recompute_layers": "8", "strategy": "full", "threshold": 0.3, "n_actual_recomp": "8", "quality": 0.92}`

### kvzip_kv (score=28.24)
- **Config**: `{"compression_ratio": 2, "codebook_size": 256, "n_iter": 10}`
- **Metadata**: `{"compression_ratio": "16", "codebook_size": "128", "n_iter": "100", "recon_err": 0.6229643225669861, "actual_comp": 6.4}`

### local_global (score=13.86)
- **Config**: `{"local_window": 1689, "global_ratio": 0.9998893737792969, "n_global_heads": 13}`
- **Metadata**: `{"local_window": 1689, "global_ratio": 0.9998893737792969, "n_global_heads": 13}`

### loss_config (score=2.03)
- **Config**: `{"loss_type": "focal", "label_smoothing": 0.2635403037071228, "focal_gamma": 4.9942126870155334, "temperature": 1.9656695127487183}`
- **Metadata**: `{"loss": 10.518086433410645, "type": "ce"}`

### memory_budget (score=0.58)
- **Config**: `{"kv_budget": 0.5922980904579163, "weight_budget": 0.19752295315265656, "activation_budget": 0.10133626312017441, "reserve": 0.05818496644496918}`
- **Metadata**: `{"utilization": 0.10736706107854843, "oom_risk": 3.4118270874023438, "total_frac": 1.3411827087402344}`

### mhc_config (score=10.45)
- **Config**: `{"rank": 64, "gate_init": 0.7297391891479492, "n_connections": 1}`
- **Metadata**: `{"rank": 114, "n_connections": 2}`

### mixed_precision (score=-16.41)
- **Config**: `{"n_levels": 2, "assignment": "uniform", "bits_base": 6}`
- **Metadata**: `{"n_levels": 3, "avg_bits": 7.875}`

### mod_config (score=10.09)
- **Config**: `{"keep_fraction": 0.5001492883311585, "router_type": "linear", "aux_loss_weight": 3.052855674923194e-08, "n_skip_layers": 15}`
- **Metadata**: `{"keep_fraction": 0.832943856716156, "compute_saved": 0.167056143283844}`

### moe_routing (score=24.75)
- **Config**: `{"n_experts": 4, "top_k": 3, "router_mode": "aux_free", "load_balance_weight": 0.0035753905773162845, "shared_expert": true}`
- **Metadata**: `{"n_experts": 6, "balance": 0.8814145922846117, "util": 1.0}`

### mosaic_quant (score=4.21)
- **Config**: `{"n_tiles": 30, "tile_dim": 409, "mix_ratio": 0.0006063980981707573}`
- **Metadata**: `{"n_tiles": 7, "err": 0.15635676681995392}`

### mtp_config (score=23.62)
- **Config**: `{"n_heads": 3, "loss_weight": 0.49990136623382575, "share_weights": true, "depth_ratio": 0.9999881982803345}`
- **Metadata**: `{"n_heads": 2, "pred_acc": 0.6950157422285813}`

### muon_config (score=11.24)
- **Config**: `{"momentum": 0.9139960116147995, "nesterov": false, "weight_decay": 0.00034710675477981565, "ns_steps": 5}`
- **Metadata**: `{"final_loss": 15.070074081420898, "ns_steps": 3}`

### nvfp4_quant (score=14.32)
- **Config**: `{"block_size": 16, "w4a8": true, "scale_mode": "per_block"}`
- **Metadata**: `{"block_size": 16, "err": 0.09768078388317639}`

### offq_quant (score=-16.49)
- **Config**: `{"offset_init": 0.05462278425693512, "n_iter": 40, "learn_offset": false}`
- **Metadata**: `{"offset": -0.0032030893489718437, "err": 0.18585337325655701}`

### optimizer_config (score=24.43)
- **Config**: `{"opt_type": "adamw", "lr": 0.009891441650986672, "beta1": 0.8026680081151426, "beta2": 0.9989998072385788, "weight_decay": 6.282140384428203e-05}`
- **Metadata**: `{"final_loss": "tensor(2.2077, device='cuda:0')", "opt": "sgd"}`

### paged_evict_kv (score=128.11)
- **Config**: `{"page_size": 64, "n_pages": 64, "eviction_policy": "lru"}`
- **Metadata**: `{"page_size": "128", "n_pages": "128", "eviction_policy": "lfu", "capacity": "16384", "hit_rate": 1.0, "mem_eff": 0.25}`

### qk_norm (score=9.65)
- **Config**: `{"norm_type": "layernorm", "epsilon": 6.5553047601133585e-06, "scale_init": 0.5006855120736873}`
- **Metadata**: `{"epsilon": 0.000757039385020733, "scale": 1.8720133006572723}`

### quant (score=-4.95)
- **Config**: `{"block_size": 16, "scale_method": "absmax", "residual_ratio": 0.05, "global_scale_factor": 1.0, "scale_search_range": 0.3, "scale_search_steps": 5, "rounding_method": "rtn", "use_hadamard": false, "hadamard_dim": 16, "scale_clip_min": 0.001}`
- **Metadata**: `{"frob_err": 5.259815065870789, "fwd_err": 5.174951502350379, "compression": 2.438163267204256, "dequant_ms": 0.073728, "q_bytes": 107517, "block_size": 16, "scale_method": "absmax", "rounding_method": "rtn", "use_hadamard": false, "hadamard_dim": 16, "scale_clip_min": 0.1, "scale_search_range": 1.5, "scale_search_steps": 5}`

### rope_config (score=36.15)
- **Config**: `{"theta": 9999989.272236824, "scaling_type": "linear", "scaling_factor": 0.7994920015335083}`
- **Metadata**: `{"theta": 6012546.450257301, "scaling": 3.994786888360977}`

### rotor_quant_kv (score=2.55)
- **Config**: `{"rot_type": "random", "n_rotations": 1, "quant_bits": 4}`
- **Metadata**: `{"rot_type": "dct", "n_rotations": 7, "quant_bits": 4, "compute": 17.5}`

### sampling_config (score=10.38)
- **Config**: `{"temperature": 1.9811241388320924, "top_p": 0.988762378692627, "top_k": 69, "repetition_penalty": 1.0109057007357478, "frequency_penalty": 0.013882001861929893}`
- **Metadata**: `{"temp": 0.3075652211904526, "diversity": -0.0}`

### scheduler_config (score=10.03)
- **Config**: `{"sched_type": "cosine", "warmup_steps": 0, "min_lr_ratio": 0.37888941168785095, "decay_steps": 101}`
- **Metadata**: `{"sched_type": "constant", "auc": 0.07587336244541484}`

### sharq_quant (score=-18.18)
- **Config**: `{"n_levels": 30, "adaptive": false, "warmup_steps": 26}`
- **Metadata**: `{"n_levels": 4, "bits": 2.0}`

### sliding_window (score=8.79)
- **Config**: `{"window_size": 958, "stride": 175, "overlap_ratio": 0.0700632631778717}`
- **Metadata**: `{"window_size": 3677, "stride": 1515}`

### sparse_attn (score=-26.47)
- **Config**: `{"strategy": "compact", "budget_ratio": 0.8464348912239075, "block_size": 21, "min_seq_len": 941, "k_ratio": 0.7948082089424133}`
- **Metadata**: `{"fwd_err": 0.644821405688931, "speedup": 0.6610987186431885, "sparse_ms": 0.35126372698581376, "full_ms": 0.23221999981615227, "strategy": "compact", "activated": true, "budget_ratio": 0.6610987186431885, "block_size": 20, "k_ratio": 0.5401936173439026}`

### speculative_decode (score=57.03)
- **Config**: `{"n_draft_tokens": 7, "draft_model_ratio": 0.10004024206136819, "acceptance_threshold": 0.9499693155288695, "temperature": 0.9400233626365662}`
- **Metadata**: `{"n_draft": 7, "acceptance": 0.5926523742770817, "speedup": 3.902061475665263}`

### streaming_kv (score=97.13)
- **Config**: `{"chunk_size": 128, "n_sink": 5, "overlap": 0.4730985164642334}`
- **Metadata**: `{"chunk_size": 262, "n_sink": 31, "overlap": 0.07213012874126434, "n_chunks": 16, "coverage_err": 0.0010672807693481445}`

### synthetic (score=20.78)
- **Config**: `{"x": "[0.02276439 0.98795563 0.51748693 0.97020143 0.14724302 0.32582197\n 0.25636983 0.86014533]"}`
- **Metadata**: `{"rastrigin": 125.4662607499889, "deceptive": 3.501248598098755}`

### titan_memory (score=3.36)
- **Config**: `{"memory_rank": 64, "gate_init": 0.492929607629776, "n_memory_slots": 1, "update_freq": 4}`
- **Metadata**: `{"rank": 136, "capacity": 0.009601157693253048, "gate": 0.5003040432929993}`

### w8a8_quant (score=179.43)
- **Config**: `{"mode": "fp8", "calib_samples": 938, "per_channel": true, "smoothquant_alpha": 0.9989532232284546}`
- **Metadata**: `{"mode": "fp8", "sqnr": 84.70417785644531}`

### xquant_kv (score=95.00)
- **Config**: `{"recomputation_ratio": 1.0, "quant_bits": 4, "checkpoint_interval": 16}`
- **Metadata**: `{"recomputation_ratio": 0.0, "quant_bits": "8", "checkpoint_interval": "2", "n_recompute": 0, "mem_ratio": 0.25, "quant_err": 0.01069291215389967}`

## Tier 3: Cross-Domain Parameter Patterns

### `acceptance_threshold`
- `0.9499693155288695`: 1x, avg=57.0, domains=speculative_decode
- `0.9499999463558196`: 1x, avg=57.0, domains=speculative_decode
- `0.9499533832073211`: 1x, avg=57.0, domains=speculative_decode
### `accum_steps`
- `5`: 3x, avg=11.1, domains=grad_accum_config
### `activation_budget`
- `0.10133626312017441`: 1x, avg=0.6, domains=memory_budget
- `0.2967330515384674`: 1x, avg=0.1, domains=memory_budget
- `0.18823260068893433`: 1x, avg=-0.1, domains=memory_budget
### `adaptive`
- `False`: 3x, avg=-18.2, domains=sharq_quant
### `assignment`
- `uniform`: 3x, avg=-16.4, domains=mixed_precision
### `async_copy`
- `True`: 3x, avg=0.0, domains=cpu_kv_offload
### `autocast_mode`
- `e5m2`: 3x, avg=37.0, domains=fp8_training_config
### `aux_loss_weight`
- `3.052855674923194e-08`: 1x, avg=10.1, domains=mod_config
- `0.09996514320373535`: 1x, avg=10.1, domains=mod_config
- `0.09483858942985535`: 1x, avg=10.0, domains=mod_config
### `beam_width`
- `4`: 3x, avg=11.5, domains=beam_search
### `beta1`
- `0.8026680081151426`: 1x, avg=24.4, domains=optimizer_config
- `0.8005675473250449`: 1x, avg=24.4, domains=optimizer_config
- `0.8151963949203491`: 1x, avg=24.0, domains=optimizer_config
### `beta2`
- `0.9989998072385788`: 1x, avg=24.4, domains=optimizer_config
- `0.9989766992330551`: 1x, avg=24.4, domains=optimizer_config
- `0.9989865505099297`: 1x, avg=24.0, domains=optimizer_config
### `bits_base`
- `6`: 3x, avg=-16.4, domains=mixed_precision
### `block_size`
- `8`: 6x, avg=-6.1, domains=csa_attention,kv_eviction,sparse_attn
- `16`: 6x, avg=4.1, domains=nvfp4_quant,quant
- `512`: 1x, avg=8.9, domains=checkpoint_recompute
- `256`: 1x, avg=8.7, domains=checkpoint_recompute
- `128`: 1x, avg=8.3, domains=checkpoint_recompute
### `budget`
- `456`: 1x, avg=-1.7, domains=kv_eviction
- `129`: 1x, avg=-1.9, domains=kv_eviction
- `699`: 1x, avg=-2.6, domains=kv_eviction
### `budget_full`
- `0.05`: 3x, avg=4.4, domains=hqe_kv
### `budget_int4`
- `0.5`: 3x, avg=4.4, domains=hqe_kv
### `budget_int8`
- `0.5`: 3x, avg=4.4, domains=hqe_kv
### `budget_ratio`
- `0.8464348912239075`: 1x, avg=-26.5, domains=sparse_attn
- `0.75`: 1x, avg=-30.1, domains=sparse_attn
- `0.5`: 1x, avg=-32.6, domains=sparse_attn
### `cache_strategy`
- `lfu`: 1x, avg=8.8, domains=expert_hotload
- `priority`: 1x, avg=8.5, domains=expert_hotload
- `lru`: 1x, avg=8.5, domains=expert_hotload
### `calib_method`
- `percentile`: 3x, avg=9.6, domains=activation_quant
### `calib_samples`
- `938`: 1x, avg=179.4, domains=w8a8_quant
- `975`: 1x, avg=178.4, domains=w8a8_quant
- `1020`: 1x, avg=178.4, domains=w8a8_quant
### `checkpoint_interval`
- `16`: 2x, avg=91.4, domains=xquant_kv
- `8`: 1x, avg=84.1, domains=xquant_kv
### `chunk_expand_size`
- `5`: 2x, avg=1.7, domains=kara
- `15`: 1x, avg=2.0, domains=kara
### `chunk_size`
- `128`: 3x, avg=97.1, domains=streaming_kv
### `codebook_size`
- `254`: 2x, avg=8.1, domains=kvzip_kv
- `482`: 1x, avg=10.0, domains=aaac_quant
- `493`: 1x, avg=10.0, domains=aaac_quant
- `450`: 1x, avg=10.0, domains=aaac_quant
- `256`: 1x, avg=28.2, domains=kvzip_kv
### `compression`
- `int4`: 3x, avg=30.0, domains=cpu_adamw_config
### `compression_ratio`
- `7.978329360485077`: 1x, avg=-50.0, domains=gla_attention
- `5.7440425753593445`: 1x, avg=-100.1, domains=gla_attention
- `6.807766854763031`: 1x, avg=-100.2, domains=gla_attention
- `2`: 1x, avg=28.2, domains=kvzip_kv
- `3`: 1x, avg=8.3, domains=kvzip_kv
### `decay_steps`
- `101`: 1x, avg=10.0, domains=scheduler_config
- `548`: 1x, avg=10.0, domains=scheduler_config
- `929`: 1x, avg=10.0, domains=scheduler_config
### `depth_ratio`
- `0.9999881982803345`: 1x, avg=23.6, domains=mtp_config
- `0.999932050704956`: 1x, avg=23.6, domains=mtp_config
- `0.9999470114707947`: 1x, avg=23.6, domains=mtp_config
### `dilation`
- `3`: 3x, avg=3.6, domains=conv_config
### `disk_cache_size`
- `4096`: 3x, avg=8.6, domains=expert_hotload
### `diversity_penalty`
- `0.9994909763336182`: 1x, avg=11.5, domains=beam_search
- `0.996241569519043`: 1x, avg=11.5, domains=beam_search
- `0.9869310259819031`: 1x, avg=11.5, domains=beam_search
### `draft_model_ratio`
- `0.10004024206136819`: 1x, avg=57.0, domains=speculative_decode
- `0.10030796595383436`: 1x, avg=57.0, domains=speculative_decode
- `0.10000892783209565`: 1x, avg=57.0, domains=speculative_decode
### `early_stopping`
- `True`: 3x, avg=11.5, domains=beam_search
### `epsilon`
- `6.5553047601133585e-06`: 1x, avg=9.7, domains=qk_norm
- `1.0069217847330946e-06`: 1x, avg=9.7, domains=qk_norm
- `0.0009793123262524606`: 1x, avg=9.7, domains=qk_norm
### `eviction_policy`
- `lru`: 3x, avg=127.8, domains=paged_evict_kv
### `focal_gamma`
- `4.9942126870155334`: 1x, avg=2.0, domains=loss_config
- `4.095370173454285`: 1x, avg=2.0, domains=loss_config
- `4.994922578334808`: 1x, avg=2.0, domains=loss_config
### `frequency_penalty`
- `0.013882001861929893`: 1x, avg=10.4, domains=sampling_config
- `0.006513859145343304`: 1x, avg=10.2, domains=sampling_config
- `0.012407752685248852`: 1x, avg=10.2, domains=sampling_config
### `gate_init`
- `0.9997047781944275`: 1x, avg=3.2, domains=attn_residual
- `0.9990285634994507`: 1x, avg=3.2, domains=attn_residual
- `0.9997461438179016`: 1x, avg=3.2, domains=attn_residual
- `0.7297391891479492`: 1x, avg=10.5, domains=mhc_config
- `0.8714539408683777`: 1x, avg=10.4, domains=mhc_config
### `global_ratio`
- `0.9998893737792969`: 1x, avg=13.9, domains=local_global
- `0.9999821186065674`: 1x, avg=13.8, domains=local_global
- `0.9834641814231873`: 1x, avg=13.7, domains=local_global
### `global_scale_factor`
- `1.0`: 3x, avg=-6.1, domains=quant
### `grad_clip`
- `0.9998345375061035`: 1x, avg=11.1, domains=grad_accum_config
- `0.9995858073234558`: 1x, avg=11.1, domains=grad_accum_config
- `0.9988573789596558`: 1x, avg=11.1, domains=grad_accum_config
### `group_size`
- `32`: 3x, avg=4.4, domains=hqe_kv
- `16`: 2x, avg=9.1, domains=group_quant
- `128`: 1x, avg=9.1, domains=group_quant
### `groups`
- `7`: 3x, avg=3.6, domains=conv_config
### `hadamard_dim`
- `16`: 3x, avg=-6.1, domains=quant
### `init_mode`
- `svd`: 3x, avg=19.4, domains=factorized_embed
### `init_scale`
- `1.5255076587200165`: 1x, avg=-27.4, domains=bitnet_config
- `1.556933581829071`: 1x, avg=-27.5, domains=bitnet_config
- `1.5363288521766663`: 1x, avg=-27.5, domains=bitnet_config
### `k_layers`
- `1`: 3x, avg=3.2, domains=attn_residual
### `k_ratio`
- `0.8`: 2x, avg=-31.3, domains=sparse_attn
- `0.7948082089424133`: 1x, avg=-26.5, domains=sparse_attn
### `keep_fraction`
- `0.5001492883311585`: 1x, avg=10.1, domains=mod_config
- `0.5001025141100399`: 1x, avg=10.1, domains=mod_config
- `0.5007760632433929`: 1x, avg=10.0, domains=mod_config
### `kernel_size`
- `5`: 2x, avg=3.7, domains=conv_config
- `7`: 1x, avg=3.4, domains=conv_config
### `kv_budget`
- `0.5922980904579163`: 1x, avg=0.6, domains=memory_budget
- `0.518980085849762`: 1x, avg=0.1, domains=memory_budget
- `0.5890820026397705`: 1x, avg=-0.1, domains=memory_budget
### `label_smoothing`
- `0.2635403037071228`: 1x, avg=2.0, domains=loss_config
- `0.0963698387145996`: 1x, avg=2.0, domains=loss_config
- `0.05483957380056381`: 1x, avg=2.0, domains=loss_config
### `lambda_init`
- `3.4092354894710297e-07`: 1x, avg=10.0, domains=diff_attn
- `6.950302378072593e-08`: 1x, avg=10.0, domains=diff_attn
- `1.1462488913593916e-07`: 1x, avg=10.0, domains=diff_attn
### `latent_dim`
- `64`: 2x, avg=-100.2, domains=gla_attention
- `509`: 1x, avg=-50.0, domains=gla_attention
### `learn_offset`
- `False`: 3x, avg=-16.6, domains=offq_quant
### `learned_scale`
- `False`: 3x, avg=-27.4, domains=bitnet_config
### `length_penalty`
- `1.9984755218029022`: 1x, avg=11.5, domains=beam_search
- `1.998426079750061`: 1x, avg=11.5, domains=beam_search
- `1.999133825302124`: 1x, avg=11.5, domains=beam_search
### `load_balance_weight`
- `0.0035753905773162845`: 1x, avg=24.7, domains=moe_routing
- `0.002754231728613377`: 1x, avg=24.6, domains=moe_routing
- `0.0002464275108650327`: 1x, avg=24.5, domains=moe_routing
### `local_window`
- `1689`: 1x, avg=13.9, domains=local_global
- `1676`: 1x, avg=13.8, domains=local_global
- `1681`: 1x, avg=13.7, domains=local_global
### `loss_scale`
- `3117.2025299072266`: 1x, avg=37.9, domains=fp8_training_config
- `4075.6271591186523`: 1x, avg=37.1, domains=fp8_training_config
- `4091.6590728759766`: 1x, avg=36.0, domains=fp8_training_config
### `loss_type`
- `focal`: 3x, avg=2.0, domains=loss_config
### `loss_weight`
- `0.49990136623382575`: 1x, avg=23.6, domains=mtp_config
- `0.49998707771301276`: 1x, avg=23.6, domains=mtp_config
- `0.4997210025787354`: 1x, avg=23.6, domains=mtp_config
### `lr`
- `0.009891441650986672`: 1x, avg=24.4, domains=optimizer_config
- `0.00998863523364067`: 1x, avg=24.4, domains=optimizer_config
- `0.009859139657616615`: 1x, avg=24.0, domains=optimizer_config
### `max_batch_size`
- `15`: 2x, avg=43.7, domains=batched_decode
- `14`: 1x, avg=42.6, domains=batched_decode
### `max_seq_diff`
- `7`: 1x, avg=44.1, domains=batched_decode
- `158`: 1x, avg=43.2, domains=batched_decode
- `280`: 1x, avg=42.6, domains=batched_decode
### `memory_rank`
- `64`: 2x, avg=3.4, domains=titan_memory
- `66`: 1x, avg=3.4, domains=titan_memory
### `merge_window_ms`
- `94`: 1x, avg=44.1, domains=batched_decode
- `95`: 1x, avg=43.2, domains=batched_decode
- `60`: 1x, avg=42.6, domains=batched_decode
### `micro_batch`
- `15`: 3x, avg=11.1, domains=grad_accum_config
### `min_keep`
- `0.5772349014878273`: 1x, avg=-6.2, domains=ffn_skip
- `0.672686979174614`: 1x, avg=-6.2, domains=ffn_skip
- `0.5008852889295667`: 1x, avg=-6.2, domains=ffn_skip
### `min_lr_ratio`
- `0.37888941168785095`: 1x, avg=10.0, domains=scheduler_config
- `0.3281840980052948`: 1x, avg=10.0, domains=scheduler_config
- `0.1044170930981636`: 1x, avg=10.0, domains=scheduler_config
### `min_seq_len`
- `256`: 2x, avg=-31.3, domains=sparse_attn
- `941`: 1x, avg=-26.5, domains=sparse_attn
### `mix_ratio`
- `0.0006063980981707573`: 1x, avg=4.2, domains=mosaic_quant
- `0.0001755044359015301`: 1x, avg=4.2, domains=mosaic_quant
- `1.158172199211549e-05`: 1x, avg=4.2, domains=mosaic_quant
### `mode`
- `fp8`: 3x, avg=178.8, domains=w8a8_quant
### `mu_scaling`
- `True`: 3x, avg=37.0, domains=fp8_training_config
### `n_bits`
- `2`: 3x, avg=10.0, domains=aaac_quant
- `8`: 3x, avg=9.1, domains=group_quant
### `n_checkpoint_layers`
- `16`: 3x, avg=8.7, domains=checkpoint_recompute
### `n_codebooks`
- `4`: 3x, avg=10.0, domains=aaac_quant
### `n_connections`
- `1`: 3x, avg=10.4, domains=mhc_config
### `n_conv_layers`
- `5`: 2x, avg=3.8, domains=conv_config
- `4`: 1x, avg=3.3, domains=conv_config
### `n_draft_tokens`
- `7`: 3x, avg=57.0, domains=speculative_decode
### `n_eval_layers`
- `15`: 1x, avg=-6.2, domains=ffn_skip
- `6`: 1x, avg=-6.2, domains=ffn_skip
- `1`: 1x, avg=-6.2, domains=ffn_skip
### `n_experts`
- `4`: 3x, avg=24.6, domains=moe_routing
### `n_global_heads`
- `13`: 1x, avg=13.9, domains=local_global
- `14`: 1x, avg=13.8, domains=local_global
- `15`: 1x, avg=13.7, domains=local_global
### `n_heads`
- `31`: 3x, avg=-26.7, domains=diff_attn,gla_attention
- `3`: 3x, avg=23.6, domains=mtp_config
- `4`: 2x, avg=-75.1, domains=gla_attention
- `13`: 1x, avg=10.0, domains=diff_attn
### `n_hot_experts`
- `4`: 2x, avg=8.7, domains=expert_hotload
- `5`: 1x, avg=8.5, domains=expert_hotload
### `n_iter`
- `12`: 2x, avg=-4.4, domains=kvzip_kv,offq_quant
- `10`: 1x, avg=28.2, domains=kvzip_kv
- `11`: 1x, avg=8.3, domains=kvzip_kv
- `40`: 1x, avg=-16.5, domains=offq_quant
- `19`: 1x, avg=-16.7, domains=offq_quant
### `n_kv_heads`
- `15`: 2x, avg=5.0, domains=gta_attention
- `9`: 1x, avg=5.0, domains=gta_attention
### `n_levels`
- `2`: 3x, avg=-16.4, domains=mixed_precision
- `31`: 2x, avg=-18.3, domains=sharq_quant
- `30`: 1x, avg=-18.2, domains=sharq_quant
### `n_memory_slots`
- `1`: 3x, avg=3.4, domains=titan_memory
### `n_pages`
- `64`: 1x, avg=128.1, domains=paged_evict_kv
- `128`: 1x, avg=128.1, domains=paged_evict_kv
- `256`: 1x, avg=127.1, domains=paged_evict_kv
### `n_rotations`
- `1`: 2x, avg=2.3, domains=rotor_quant_kv
- `2`: 1x, avg=1.5, domains=rotor_quant_kv
### `n_share_groups`
- `1`: 3x, avg=203988.7, domains=cross_layer_kv
### `n_sink`
- `5`: 1x, avg=97.1, domains=streaming_kv
- `6`: 1x, avg=97.1, domains=streaming_kv
- `4`: 1x, avg=97.1, domains=streaming_kv
### `n_sinks`
- `2`: 1x, avg=-1.7, domains=kv_eviction
- `3`: 1x, avg=-1.9, domains=kv_eviction
- `4`: 1x, avg=-2.6, domains=kv_eviction
### `n_skip_layers`
- `15`: 1x, avg=10.1, domains=mod_config
- `2`: 1x, avg=10.1, domains=mod_config
- `6`: 1x, avg=10.0, domains=mod_config
### `n_tiles`
- `30`: 1x, avg=4.2, domains=mosaic_quant
- `11`: 1x, avg=4.2, domains=mosaic_quant
- `15`: 1x, avg=4.2, domains=mosaic_quant
### `norm_type`
- `layernorm`: 3x, avg=9.7, domains=qk_norm
### `observation_window`
- `307`: 1x, avg=-1.7, domains=kv_eviction
- `179`: 1x, avg=-1.9, domains=kv_eviction
- `44`: 1x, avg=-2.6, domains=kv_eviction
### `offload_layers`
- `1`: 3x, avg=0.0, domains=cpu_kv_offload
### `offload_ratio`
- `4.6751304694225837e-07`: 1x, avg=30.0, domains=cpu_adamw_config
- `7.136686690500937e-07`: 1x, avg=30.0, domains=cpu_adamw_config
- `4.516525223152712e-06`: 1x, avg=30.0, domains=cpu_adamw_config
- `0.9610263705253601`: 1x, avg=4.2, domains=hybrid_offload
- `0.9609100222587585`: 1x, avg=4.2, domains=hybrid_offload
### `offload_threshold`
- `0.9999842643737793`: 1x, avg=0.0, domains=cpu_kv_offload
- `0.999976396560669`: 1x, avg=0.0, domains=cpu_kv_offload
- `0.9999308586120605`: 1x, avg=0.0, domains=cpu_kv_offload
### `offset_init`
- `0.05462278425693512`: 1x, avg=-16.5, domains=offq_quant
- `0.07712052017450333`: 1x, avg=-16.7, domains=offq_quant
- `0.0038995745126158`: 1x, avg=-16.8, domains=offq_quant
### `opt_type`
- `adamw`: 3x, avg=24.3, domains=optimizer_config
### `overlap`
- `0.4730985164642334`: 1x, avg=97.1, domains=streaming_kv
- `0.16356708109378815`: 1x, avg=97.1, domains=streaming_kv
- `0.1`: 1x, avg=97.1, domains=streaming_kv
### `overlap_compute`
- `True`: 3x, avg=4.2, domains=hybrid_offload
### `overlap_ratio`
- `0.0700632631778717`: 1x, avg=8.8, domains=sliding_window
- `0.3254103660583496`: 1x, avg=8.8, domains=sliding_window
- `0.6614090800285339`: 1x, avg=8.8, domains=sliding_window
### `padding_strategy`
- `dynamic`: 2x, avg=42.9, domains=batched_decode
- `left`: 1x, avg=44.1, domains=batched_decode
### `page_size`
- `64`: 1x, avg=128.1, domains=paged_evict_kv
- `32`: 1x, avg=128.1, domains=paged_evict_kv
- `16`: 1x, avg=127.1, domains=paged_evict_kv
### `pattern_type`
- `csa_hca_hybrid`: 3x, avg=14.8, domains=csa_attention
### `per_channel`
- `True`: 3x, avg=178.8, domains=w8a8_quant
### `percentile`
- `0.9003667952804827`: 1x, avg=9.6, domains=activation_quant
- `0.9001695250299526`: 1x, avg=9.6, domains=activation_quant
- `0.9001529517275049`: 1x, avg=9.6, domains=activation_quant
### `pin_memory`
- `True`: 3x, avg=4.2, domains=hybrid_offload
### `prefetch_ahead`
- `4`: 2x, avg=8.7, domains=expert_hotload
- `2`: 1x, avg=8.5, domains=expert_hotload
### `prefetch_depth`
- `8`: 3x, avg=4.2, domains=hybrid_offload
- `5`: 1x, avg=30.0, domains=cpu_adamw_config
- `3`: 1x, avg=30.0, domains=cpu_adamw_config
- `7`: 1x, avg=30.0, domains=cpu_adamw_config
### `prefetch_size`
- `2048`: 3x, avg=0.0, domains=cpu_kv_offload
### `quant_bits`
- `4`: 4x, avg=25.3, domains=rotor_quant_kv,xquant_kv
- `8`: 2x, avg=86.0, domains=xquant_kv
### `quant_mode`
- `ternary`: 3x, avg=-27.4, domains=bitnet_config
### `rank`
- `64`: 4x, avg=17.1, domains=factorized_embed,mhc_config
- `66`: 1x, avg=10.4, domains=mhc_config
- `77`: 1x, avg=10.4, domains=mhc_config
### `recency_decay`
- `1.0`: 1x, avg=4.5, domains=hqe_kv
- `0.95`: 1x, avg=4.4, domains=hqe_kv
- `0.8`: 1x, avg=4.4, domains=hqe_kv
### `recomputation_ratio`
- `0.75`: 2x, avg=86.0, domains=xquant_kv
- `1.0`: 1x, avg=95.0, domains=xquant_kv
### `recompute_layers`
- `16`: 2x, avg=70.7, domains=kv_recompute
- `14`: 1x, avg=70.7, domains=kv_recompute
### `recompute_strategy`
- `selective`: 6x, avg=39.7, domains=checkpoint_recompute,kv_recompute
### `repetition_penalty`
- `1.0109057007357478`: 1x, avg=10.4, domains=sampling_config
- `1.0023982492275536`: 1x, avg=10.2, domains=sampling_config
- `1.0016888725804165`: 1x, avg=10.2, domains=sampling_config
### `reserve`
- `0.05818496644496918`: 1x, avg=0.6, domains=memory_budget
- `0.05693390592932701`: 1x, avg=0.1, domains=memory_budget
- `0.06022920459508896`: 1x, avg=-0.1, domains=memory_budget
### `residual_ratio`
- `0.05`: 1x, avg=-5.0, domains=quant
- `0.1`: 1x, avg=-6.1, domains=quant
- `0.2`: 1x, avg=-7.3, domains=quant
### `retrieval_dim`
- `227`: 1x, avg=3.2, domains=attn_residual
- `90`: 1x, avg=3.2, domains=attn_residual
- `88`: 1x, avg=3.2, domains=attn_residual
### `rot_type`
- `random`: 2x, avg=2.3, domains=rotor_quant_kv
- `hadamard`: 1x, avg=1.5, domains=rotor_quant_kv
### `rounding_method`
- `rtn`: 3x, avg=-6.1, domains=quant
### `router_mode`
- `aux_free`: 2x, avg=24.7, domains=moe_routing
- `switch`: 1x, avg=24.5, domains=moe_routing
### `router_type`
- `mlp`: 2x, avg=10.1, domains=mod_config
- `linear`: 1x, avg=10.1, domains=mod_config
### `scale_clip_min`
- `0.001`: 3x, avg=-6.1, domains=quant
### `scale_init`
- `0.5006855120736873`: 1x, avg=9.7, domains=qk_norm
- `0.5012165468069725`: 1x, avg=9.7, domains=qk_norm
- `0.5000340259812219`: 1x, avg=9.7, domains=qk_norm
### `scale_method`
- `absmax`: 3x, avg=-6.1, domains=quant
### `scale_mode`
- `per_block`: 2x, avg=14.3, domains=nvfp4_quant
- `per_channel`: 1x, avg=14.3, domains=nvfp4_quant
### `scale_search_range`
- `0.3`: 3x, avg=-6.1, domains=quant
### `scale_search_steps`
- `5`: 3x, avg=-6.1, domains=quant
### `scaling_factor`
- `0.7994920015335083`: 1x, avg=36.1, domains=rope_config
- `2.021500840783119`: 1x, avg=36.1, domains=rope_config
- `3.995861679315567`: 1x, avg=36.1, domains=rope_config
### `scaling_type`
- `linear`: 2x, avg=36.1, domains=rope_config
- `yarn`: 1x, avg=36.1, domains=rope_config
### `sched_type`
- `cosine`: 3x, avg=10.0, domains=scheduler_config
### `scheme`
- `asymmetric`: 3x, avg=9.1, domains=group_quant
### `share_mode`
- `max`: 3x, avg=203988.7, domains=cross_layer_kv
### `share_ratio`
- `1.0`: 1x, avg=218453.5, domains=cross_layer_kv
- `0.8825158476829529`: 1x, avg=204338.3, domains=cross_layer_kv
- `0.75`: 1x, avg=189174.3, domains=cross_layer_kv
### `share_weights`
- `True`: 3x, avg=23.6, domains=mtp_config
### `shared_expert`
- `True`: 2x, avg=24.7, domains=moe_routing
- `False`: 1x, avg=24.5, domains=moe_routing
### `sink_size`
- `2`: 3x, avg=1.8, domains=kara
### `skip_strategy`
- `cosine`: 3x, avg=-6.2, domains=ffn_skip
### `skip_threshold`
- `0.0036844995338469744`: 1x, avg=-6.2, domains=ffn_skip
- `0.0038752062246203423`: 1x, avg=-6.2, domains=ffn_skip
- `0.006381513085216284`: 1x, avg=-6.2, domains=ffn_skip
### `smooth_alpha`
- `0.48856788873672485`: 1x, avg=9.6, domains=activation_quant
- `0.15654350817203522`: 1x, avg=9.6, domains=activation_quant
- `0.13294455409049988`: 1x, avg=9.6, domains=activation_quant
### `smooth_swiglu`
- `False`: 3x, avg=37.0, domains=fp8_training_config
### `smoothquant_alpha`
- `0.9989532232284546`: 1x, avg=179.4, domains=w8a8_quant
- `0.999742329120636`: 1x, avg=178.4, domains=w8a8_quant
- `0.9999098777770996`: 1x, avg=178.4, domains=w8a8_quant
### `softmax_sep`
- `0.8985720872879028`: 1x, avg=10.0, domains=diff_attn
- `0.9569386839866638`: 1x, avg=10.0, domains=diff_attn
- `0.8618018627166748`: 1x, avg=10.0, domains=diff_attn
### `strategy`
- `streaming`: 3x, avg=-2.1, domains=kv_eviction
- `mosa`: 2x, avg=-31.3, domains=sparse_attn
- `compact`: 1x, avg=-26.5, domains=sparse_attn
### `stride`
- `1`: 3x, avg=3.6, domains=conv_config
- `175`: 1x, avg=8.8, domains=sliding_window
- `473`: 1x, avg=8.8, domains=sliding_window
- `445`: 1x, avg=8.8, domains=sliding_window
### `sync_freq`
- `15`: 2x, avg=11.1, domains=grad_accum_config
- `10`: 1x, avg=11.1, domains=grad_accum_config
### `target_budget`
- `3961`: 1x, avg=2.0, domains=kara
- `648`: 1x, avg=1.9, domains=kara
- `566`: 1x, avg=1.5, domains=kara
### `temperature`
- `1.9656695127487183`: 1x, avg=2.0, domains=loss_config
- `1.8956142365932465`: 1x, avg=2.0, domains=loss_config
- `1.9823136925697327`: 1x, avg=2.0, domains=loss_config
- `1.9811241388320924`: 1x, avg=10.4, domains=sampling_config
- `1.0289955705404281`: 1x, avg=10.2, domains=sampling_config
### `theta`
- `9999989.272236824`: 1x, avg=36.1, domains=rope_config
- `9999907.026052475`: 1x, avg=36.1, domains=rope_config
- `9992523.345053196`: 1x, avg=36.1, domains=rope_config
### `threshold`
- `0.9`: 1x, avg=74.5, domains=kv_recompute
- `0.8122888207435608`: 1x, avg=70.7, domains=kv_recompute
- `0.7`: 1x, avg=66.8, domains=kv_recompute
### `tie_factor`
- `0.2021797150373459`: 1x, avg=19.4, domains=factorized_embed
- `0.9994256496429443`: 1x, avg=19.4, domains=factorized_embed
- `0.7738939523696899`: 1x, avg=19.4, domains=factorized_embed
### `tie_strength`
- `0.9667887091636658`: 1x, avg=5.0, domains=gta_attention
- `0.964815080165863`: 1x, avg=5.0, domains=gta_attention
- `0.8207767605781555`: 1x, avg=5.0, domains=gta_attention
### `tile_dim`
- `409`: 1x, avg=4.2, domains=mosaic_quant
- `68`: 1x, avg=4.2, domains=mosaic_quant
- `106`: 1x, avg=4.2, domains=mosaic_quant
### `top_k`
- `3`: 3x, avg=24.6, domains=moe_routing
- `64`: 2x, avg=14.8, domains=csa_attention
- `71`: 1x, avg=14.8, domains=csa_attention
- `69`: 1x, avg=10.4, domains=sampling_config
- `99`: 1x, avg=10.2, domains=sampling_config
### `top_p`
- `0.988762378692627`: 1x, avg=10.4, domains=sampling_config
- `0.9995100498199463`: 1x, avg=10.2, domains=sampling_config
- `0.9999458193778992`: 1x, avg=10.2, domains=sampling_config
### `update_freq`
- `9`: 2x, avg=16.7, domains=cpu_adamw_config,titan_memory
- `8`: 2x, avg=30.0, domains=cpu_adamw_config
- `4`: 1x, avg=3.4, domains=titan_memory
- `6`: 1x, avg=3.4, domains=titan_memory
### `use_hadamard`
- `False`: 3x, avg=-6.1, domains=quant
### `v_k_mix`
- `2.485988375156012e-07`: 1x, avg=5.0, domains=gta_attention
- `1.6162712199729867e-06`: 1x, avg=5.0, domains=gta_attention
- `4.276202162145637e-05`: 1x, avg=5.0, domains=gta_attention
### `vocab_size`
- `65304`: 1x, avg=19.4, domains=factorized_embed
- `65385`: 1x, avg=19.4, domains=factorized_embed
- `63404`: 1x, avg=19.4, domains=factorized_embed
### `w4a8`
- `True`: 3x, avg=14.3, domains=nvfp4_quant
### `warmup_steps`
- `0`: 3x, avg=10.0, domains=scheduler_config
- `26`: 1x, avg=-18.2, domains=sharq_quant
- `73`: 1x, avg=-18.3, domains=sharq_quant
- `2`: 1x, avg=-18.3, domains=sharq_quant
### `weight_budget`
- `0.19752295315265656`: 1x, avg=0.6, domains=memory_budget
- `0.17770864069461823`: 1x, avg=0.1, domains=memory_budget
- `0.22392280399799347`: 1x, avg=-0.1, domains=memory_budget
### `weight_decay`
- `0.00034710675477981565`: 1x, avg=11.2, domains=muon_config
- `0.008010155558586122`: 1x, avg=11.2, domains=muon_config
- `6.282140384428203e-05`: 1x, avg=24.4, domains=optimizer_config
- `0.0005117942579090595`: 1x, avg=24.4, domains=optimizer_config
- `0.01755797117948532`: 1x, avg=24.0, domains=optimizer_config
### `window_size`
- `130`: 1x, avg=2.0, domains=kara
- `203`: 1x, avg=1.9, domains=kara
- `128`: 1x, avg=1.5, domains=kara
- `1011`: 1x, avg=-1.7, domains=kv_eviction
- `1010`: 1x, avg=-1.9, domains=kv_eviction
### `x`
- `[0.02276439 0.98795563 0.51748693 0.97020143 0.14724302 0.32582197
 0.25636983 0.86014533]`: 1x, avg=20.8, domains=synthetic
- `[0.01107647 0.98667496 0.99504447 0.40264595 0.17122655 0.13995731
 0.27633035 0.42861766]`: 1x, avg=8.3, domains=synthetic
- `[0.37006035 0.97863775 0.34690756 0.4034914  0.08614549 0.6004753
 0.24733509 0.25371113]`: 1x, avg=6.9, domains=synthetic
