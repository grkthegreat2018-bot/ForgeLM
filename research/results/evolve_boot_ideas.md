# ForgeEvolve 'boot' Run: Top Optimization Ideas
Generated: 2026-08-24 13:29:45
Domains: 56

## Tier 1: Top 50 Configurations

| Rank | Domain | Score | Config |
|------|--------|-------|--------|
| 1 | cross_layer_kv | 218453.47 | `{"share_ratio": 1.0, "n_share_groups": 1, "share_mode": "max"}` |
| 2 | cross_layer_kv | 204338.32 | `{"share_ratio": 0.8983016014099121, "n_share_groups": 1, "share_mode": "max"}` |
| 3 | cross_layer_kv | 189174.32 | `{"share_ratio": 0.75, "n_share_groups": 1, "share_mode": "max"}` |
| 4 | cross_layer_kv | 181117.47 | `{"share_ratio": 0.7484700679779053, "n_share_groups": 1, "share_mode": "max"}` |
| 5 | cross_layer_kv | 163820.76 | `{"share_ratio": 0.6212170124053955, "n_share_groups": 1, "share_mode": "max"}` |
| 6 | cross_layer_kv | 154448.52 | `{"share_ratio": 0.5, "n_share_groups": 1, "share_mode": "max"}` |
| 7 | cross_layer_kv | 144470.21 | `{"share_ratio": 0.4617895185947418, "n_share_groups": 1, "share_mode": "max"}` |
| 8 | cross_layer_kv | 133750.25 | `{"share_ratio": 0.39999449253082275, "n_share_groups": 1, "share_mode": "max"}` |
| 9 | cross_layer_kv | 125977.32 | `{"share_ratio": 0.9102497696876526, "n_share_groups": 2, "share_mode": "max"}` |
| 10 | cross_layer_kv | 122093.52 | `{"share_ratio": 0.3375462293624878, "n_share_groups": 1, "share_mode": "max"}` |
| 11 | cross_layer_kv | 109200.58 | `{"share_ratio": 0.25, "n_share_groups": 1, "share_mode": "max"}` |
| 12 | cross_layer_kv | 106458.45 | `{"share_ratio": 0.7127634882926941, "n_share_groups": 2, "share_mode": "max"}` |
| 13 | cross_layer_kv | 77210.59 | `{"share_ratio": 0.16445797681808472, "n_share_groups": 1, "share_mode": "max"}` |
| 14 | cross_layer_kv | 54592.85 | `{"share_ratio": 0.0, "n_share_groups": 1, "share_mode": "max"}` |
| 15 | cross_layer_kv | 52548.89 | `{"share_ratio": 0.9245825409889221, "n_share_groups": 5, "share_mode": "max"}` |
| 16 | cross_layer_kv | 47592.89 | `{"share_ratio": 0.0, "n_share_groups": 2, "share_mode": "max"}` |
| 17 | cross_layer_kv | 32789.22 | `{"share_ratio": 0.48218223452568054, "n_share_groups": 7, "share_mode": "max"}` |
| 18 | cross_layer_kv | 25689.18 | `{"share_ratio": 1.0, "n_share_groups": 1, "share_mode": "avg"}` |
| 19 | cross_layer_kv | 24023.85 | `{"share_ratio": 0.9297548532485962, "n_share_groups": 1, "share_mode": "avg"}` |
| 20 | cross_layer_kv | 24022.85 | `{"share_ratio": 0.9237259030342102, "n_share_groups": 1, "share_mode": "learn...` |
| 21 | w8a8_quant | 178.15 | `{"mode": "fp8", "calib_samples": 227, "per_channel": true, "smoothquant_alpha...` |
| 22 | w8a8_quant | 177.99 | `{"mode": "fp8", "calib_samples": 848, "per_channel": true, "smoothquant_alpha...` |
| 23 | w8a8_quant | 177.56 | `{"mode": "fp8", "calib_samples": 380, "per_channel": true, "smoothquant_alpha...` |
| 24 | w8a8_quant | 177.50 | `{"mode": "int8", "calib_samples": 990, "per_channel": false, "smoothquant_alp...` |
| 25 | w8a8_quant | 177.36 | `{"mode": "int8", "calib_samples": 692, "per_channel": true, "smoothquant_alph...` |
| 26 | w8a8_quant | 176.86 | `{"mode": "fp8", "calib_samples": 789, "per_channel": true, "smoothquant_alpha...` |
| 27 | w8a8_quant | 176.62 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 28 | w8a8_quant | 176.43 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 29 | w8a8_quant | 175.98 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 30 | w8a8_quant | 175.66 | `{"mode": "fp8", "calib_samples": 256, "per_channel": true, "smoothquant_alpha...` |
| 31 | w8a8_quant | 174.95 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 32 | w8a8_quant | 174.83 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 33 | w8a8_quant | 173.93 | `{"mode": "fp8", "calib_samples": 256, "per_channel": true, "smoothquant_alpha...` |
| 34 | w8a8_quant | 173.11 | `{"mode": "int8", "calib_samples": 256, "per_channel": true, "smoothquant_alph...` |
| 35 | paged_evict_kv | 128.11 | `{"page_size": 64, "n_pages": 64, "eviction_policy": "lru"}` |
| 36 | paged_evict_kv | 128.09 | `{"page_size": 32, "n_pages": 128, "eviction_policy": "lru"}` |
| 37 | paged_evict_kv | 127.13 | `{"page_size": 16, "n_pages": 256, "eviction_policy": "lru"}` |
| 38 | paged_evict_kv | 123.37 | `{"page_size": 16, "n_pages": 289, "eviction_policy": "lru"}` |
| 39 | paged_evict_kv | 123.30 | `{"page_size": 18, "n_pages": 260, "eviction_policy": "lfu"}` |
| 40 | paged_evict_kv | 120.97 | `{"page_size": 62, "n_pages": 86, "eviction_policy": "lru"}` |
| 41 | paged_evict_kv | 120.40 | `{"page_size": 65, "n_pages": 84, "eviction_policy": "importance"}` |
| 42 | paged_evict_kv | 118.43 | `{"page_size": 38, "n_pages": 156, "eviction_policy": "lru"}` |
| 43 | paged_evict_kv | 115.85 | `{"page_size": 83, "n_pages": 81, "eviction_policy": "lfu"}` |
| 44 | paged_evict_kv | 115.70 | `{"page_size": 16, "n_pages": 188, "eviction_policy": "importance"}` |
| 45 | paged_evict_kv | 114.75 | `{"page_size": 82, "n_pages": 87, "eviction_policy": "importance"}` |
| 46 | paged_evict_kv | 114.41 | `{"page_size": 25, "n_pages": 278, "eviction_policy": "lru"}` |
| 47 | paged_evict_kv | 113.91 | `{"page_size": 33, "n_pages": 222, "eviction_policy": "lfu"}` |
| 48 | paged_evict_kv | 112.47 | `{"page_size": 64, "n_pages": 128, "eviction_policy": "lru"}` |
| 49 | paged_evict_kv | 112.14 | `{"page_size": 95, "n_pages": 87, "eviction_policy": "importance"}` |
| 50 | paged_evict_kv | 111.53 | `{"page_size": 73, "n_pages": 119, "eviction_policy": "lru"}` |

## Tier 2: Best Per Domain

### aaac_quant (score=9.94)
- **Config**: `{"n_codebooks": 4, "codebook_size": 509, "n_bits": 2}`
- **Metadata**: `{"n_codebooks": 14, "compression": 0.7619047619047619}`

### activation_quant (score=9.63)
- **Config**: `{"calib_method": "percentile", "percentile": 0.9014481891123578, "smooth_alpha": 0.4332141578197479}`
- **Metadata**: `{"method": "entropy", "err": 0.0068390287544097894}`

### attn_residual (score=3.01)
- **Config**: `{"k_layers": 1, "gate_init": 0.990082323551178, "retrieval_dim": 252}`
- **Metadata**: `{"k": 6, "gate": 0.9535607099533081, "retrieval_dim": 335}`

### batched_decode (score=43.25)
- **Config**: `{"max_batch_size": 15, "padding_strategy": "dynamic", "merge_window_ms": 92, "max_seq_diff": 481}`
- **Metadata**: `{"batch_size": 4, "throughput": 3.52}`

### beam_search (score=11.41)
- **Config**: `{"beam_width": 4, "length_penalty": 1.920774519443512, "early_stopping": true, "diversity_penalty": 0.9949210286140442}`
- **Metadata**: `{"beam_width": 4, "accuracy": 0.6666666666666667}`

### bitnet_config (score=-27.43)
- **Config**: `{"learned_scale": false, "quant_mode": "ternary", "init_scale": 1.530042827129364}`
- **Metadata**: `{"mode": "ternary", "err": 0.4963875756375844}`

### checkpoint_recompute (score=8.94)
- **Config**: `{"n_checkpoint_layers": 16, "recompute_strategy": "selective", "block_size": 512}`
- **Metadata**: `{"mem_saved_gb": 0.40265315771102905, "recompute_overhead_ms": 3.0000001192092896}`

### conv_config (score=3.44)
- **Config**: `{"kernel_size": 7, "stride": 1, "dilation": 3, "groups": 7, "n_conv_layers": 5}`
- **Metadata**: `{"receptive_field": 27, "params": 758345.1428571428}`

### cpu_adamw_config (score=29.91)
- **Config**: `{"offload_ratio": 0.004509661812335253, "prefetch_depth": 5, "compression": "int4", "update_freq": 13}`
- **Metadata**: `{"throughput": 2832.7451795339584, "latency": 26.017416516939797}`

### cpu_kv_offload (score=0.03)
- **Config**: `{"offload_layers": 1, "offload_threshold": 0.6295778751373291, "prefetch_size": 2048, "async_copy": true}`
- **Metadata**: `{"kv_freed_gb": 0.03195071965456009, "decode_lat_ms": 4.527586194975623}`

### cross_layer_kv (score=218453.47)
- **Config**: `{"share_ratio": 1.0, "n_share_groups": 1, "share_mode": "max"}`
- **Metadata**: `{"share_ratio": 0.75, "n_share_groups": 2, "share_mode": "max", "param_reduction": 0.75, "recon_err": -233.10710906982422}`

### csa_attention (score=14.78)
- **Config**: `{"top_k": 64, "pattern_type": "csa_hca_hybrid", "block_size": 13}`
- **Metadata**: `{"top_k": 655, "block_size": 86}`

### diff_attn (score=9.99)
- **Config**: `{"lambda_init": 0.0027514046523720026, "n_heads": 23, "softmax_sep": 0.945634126663208}`
- **Metadata**: `{"lambda": 0.060547325760126114, "n_heads": 7}`

### expert_hotload (score=8.83)
- **Config**: `{"n_hot_experts": 4, "prefetch_ahead": 4, "cache_strategy": "lfu", "disk_cache_size": 4096}`
- **Metadata**: `{"vram_used_gb": 1.025390625, "miss_rate": 0.0}`

### factorized_embed (score=19.35)
- **Config**: `{"rank": 64, "init_mode": "svd", "tie_factor": 0.15296828746795654, "vocab_size": 47660}`
- **Metadata**: `{"rank": 396, "reduction": 0.8001778512297222, "err": 8.961730120150156e-06}`

### ffn_skip (score=-6.22)
- **Config**: `{"skip_threshold": 0.004109608940780163, "n_eval_layers": 15, "skip_strategy": "hybrid", "min_keep": 0.6856748014688492}`
- **Metadata**: `{"compute_saved": 0.0, "deviation": 0.0}`

### fp8_training_config (score=36.38)
- **Config**: `{"autocast_mode": "e5m2", "smooth_swiglu": false, "mu_scaling": true, "loss_scale": 3644.6247024536133}`
- **Metadata**: `{"overflow": 0.0, "mode": "e5m2"}`

### gla_attention (score=-50.00)
- **Config**: `{"latent_dim": 349, "n_heads": 4, "compression_ratio": 6.481169760227203}`
- **Metadata**: `{"latent_dim": 229, "recon_err": 1.4146760905982896}`

### grad_accum_config (score=11.09)
- **Config**: `{"accum_steps": 4, "micro_batch": 15, "grad_clip": 0.9958323836326599, "sync_freq": 15}`
- **Metadata**: `{"effective_batch": 3, "noise": 0.5773502691896258}`

### group_quant (score=9.09)
- **Config**: `{"group_size": 16, "n_bits": 8, "scheme": "asymmetric"}`
- **Metadata**: `{"group_size": 64, "bits": 2, "err": 0.8911506546137419}`

### gta_attention (score=5.00)
- **Config**: `{"v_k_mix": 2.7122376195620745e-05, "n_kv_heads": 4, "tie_strength": 0.6417432427406311}`
- **Metadata**: `{"mix": 0.007738437503576279, "deviation": 0.010918614784693264}`

### hqe_kv (score=4.46)
- **Config**: `{"budget_full": 0.05, "budget_int8": 0.5, "budget_int4": 0.5, "group_size": 32, "recency_decay": 1.0}`
- **Metadata**: `{"fwd_err": 0.27108611182414827, "compression": 3.079120466077805, "quant_ms": 0.9237361398233415, "n_full": 409, "n_int8": 409, "n_int4": 2867, "n_evicted": 411, "group_size": 128, "recency_decay": 1.0}`

### hybrid_offload (score=4.18)
- **Config**: `{"offload_ratio": 0.960943341255188, "prefetch_depth": 8, "pin_memory": true, "overlap_compute": true}`
- **Metadata**: `{"vram_saved_gb": 1.8323919773101807, "latency_ms": 3.097151517868042}`

### kara (score=1.25)
- **Config**: `{"sink_size": 2, "window_size": 649, "target_budget": 688, "chunk_expand_size": 5}`
- **Metadata**: `{"fwd_err": 0.5920983066016049, "compression": 1.2872407291011942, "kara_ms": 0.6436203645505971, "sink_size": 7, "window_size": 175, "target_budget": 1500, "chunk_expand_size": 6, "comp_seq_len": 3182}`

### kv_eviction (score=-41.38)
- **Config**: `{"strategy": "paged", "budget": 256, "observation_window": 512, "n_sinks": 2, "window_size": 128, "block_size": 8}`
- **Metadata**: `{"fwd_err": 0.5092571200916362, "compression": 1.3333333333333333, "cache_ms": 0.39999999999999997, "strategy": "paged", "budget": 256, "comp_seq_len": 768}`

### kv_recompute (score=74.55)
- **Config**: `{"recompute_layers": 16, "recompute_strategy": "selective", "threshold": 0.9}`
- **Metadata**: `{"recompute_layers": 8, "strategy": "full", "threshold": 0.3, "n_actual_recomp": 8, "quality": 0.92}`

### kvzip_kv (score=28.24)
- **Config**: `{"compression_ratio": 2, "codebook_size": 256, "n_iter": 10}`
- **Metadata**: `{"compression_ratio": 16, "codebook_size": 128, "n_iter": 100, "recon_err": 0.613283634185791, "actual_comp": 6.4}`

### local_global (score=13.86)
- **Config**: `{"local_window": 1716, "global_ratio": 0.9902746677398682, "n_global_heads": 15}`
- **Metadata**: `{"local_window": 303, "global_ratio": 0.7894176244735718, "n_global_heads": 2}`

### loss_config (score=2.05)
- **Config**: `{"loss_type": "focal", "label_smoothing": 0.15508339405059815, "focal_gamma": 4.430350065231323, "temperature": 1.9878134727478027}`
- **Metadata**: `{"loss": 5.323076248168945, "type": "focal"}`

### memory_budget (score=0.56)
- **Config**: `{"kv_budget": 0.584653913974762, "weight_budget": 0.18910351395606995, "activation_budget": 0.10464277118444443, "reserve": 0.05720803886651993}`
- **Metadata**: `{"utilization": 0.060088712722063065, "oom_risk": 0.0, "total_frac": 0.9037711024284363}`

### mhc_config (score=10.19)
- **Config**: `{"rank": 70, "gate_init": 0.26289334893226624, "n_connections": 1}`
- **Metadata**: `{"rank": 250, "n_connections": 4}`

### mixed_precision (score=-16.42)
- **Config**: `{"n_levels": 2, "assignment": "uniform", "bits_base": 6}`
- **Metadata**: `{"n_levels": 2, "avg_bits": 6.0}`

### mod_config (score=10.05)
- **Config**: `{"keep_fraction": 0.5001477687765146, "router_type": "mlp", "aux_loss_weight": 0.010835280269384386, "n_skip_layers": 4}`
- **Metadata**: `{"keep_fraction": 0.6801507025957108, "compute_saved": 0.31984929740428925}`

### moe_routing (score=24.71)
- **Config**: `{"n_experts": 5, "top_k": 3, "router_mode": "switch", "load_balance_weight": 0.0018776681274175644, "shared_expert": true}`
- **Metadata**: `{"n_experts": 5, "balance": 0.657673393954344, "util": 1.0}`

### mosaic_quant (score=4.15)
- **Config**: `{"n_tiles": 16, "tile_dim": 95, "mix_ratio": 0.0014527126913890243}`
- **Metadata**: `{"n_tiles": 13, "err": 0.009308667853474617}`

### mtp_config (score=23.58)
- **Config**: `{"n_heads": 3, "loss_weight": 0.4932160615921021, "share_weights": true, "depth_ratio": 0.9985441267490387}`
- **Metadata**: `{"n_heads": 1, "pred_acc": 0.7842553180197012}`

### muon_config (score=11.24)
- **Config**: `{"momentum": 0.9776730489730835, "nesterov": true, "weight_decay": 0.002937338948249817, "ns_steps": 2}`
- **Metadata**: `{"final_loss": 15.070074081420898, "ns_steps": 3}`

### nvfp4_quant (score=14.30)
- **Config**: `{"block_size": 16, "w4a8": true, "scale_mode": "per_channel"}`
- **Metadata**: `{"block_size": 32, "err": 0.10414015559203527}`

### offq_quant (score=-16.84)
- **Config**: `{"offset_init": 0.03407374396920204, "n_iter": 34, "learn_offset": false}`
- **Metadata**: `{"offset": 0.0023953847121447325, "err": 0.17855867319787577}`

### optimizer_config (score=15.99)
- **Config**: `{"opt_type": "adamw", "lr": 0.009001950696110726, "beta1": 0.9316969275474548, "beta2": 0.9989514474868775, "weight_decay": 0.08693135380744935}`
- **Metadata**: `{"final_loss": 12.981112480163574, "opt": "muon"}`

### paged_evict_kv (score=128.11)
- **Config**: `{"page_size": 64, "n_pages": 64, "eviction_policy": "lru"}`
- **Metadata**: `{"page_size": 128, "n_pages": 128, "eviction_policy": "lfu", "capacity": 16384, "hit_rate": 1.0, "mem_eff": 0.25}`

### qk_norm (score=9.65)
- **Config**: `{"norm_type": "layernorm", "epsilon": 0.0003313357861936093, "scale_init": 0.5054262886987999}`
- **Metadata**: `{"epsilon": 0.0009104777835011483, "scale": 1.965695083141327}`

### quant (score=-4.95)
- **Config**: `{"block_size": 16, "scale_method": "absmax", "residual_ratio": 0.05, "global_scale_factor": 1.0, "scale_search_range": 0.3, "scale_search_steps": 5, "rounding_method": "rtn", "use_hadamard": false, "hadamard_dim": 16, "scale_clip_min": 0.001}`
- **Metadata**: `{"frob_err": 5.259815065870789, "fwd_err": 5.174951502350379, "compression": 2.438163267204256, "dequant_ms": 0.073728, "q_bytes": 107517, "block_size": 16, "scale_method": "absmax", "rounding_method": "rtn", "use_hadamard": false, "hadamard_dim": 16, "scale_clip_min": 0.1, "scale_search_range": 1.5, "scale_search_steps": 5}`

### rope_config (score=36.15)
- **Config**: `{"theta": 9997342.494666576, "scaling_type": "linear", "scaling_factor": 1.9816195666790009}`
- **Metadata**: `{"theta": 9417653.507769108, "scaling": 3.4731838703155518}`

### rotor_quant_kv (score=1.49)
- **Config**: `{"rot_type": "hadamard", "n_rotations": 2, "quant_bits": 4}`
- **Metadata**: `{"rot_type": "hadamard", "n_rotations": 8, "quant_bits": 8, "compute": 8.0}`

### sampling_config (score=9.99)
- **Config**: `{"temperature": 1.673143845796585, "top_p": 0.9963274002075195, "top_k": 99, "repetition_penalty": 1.0291668307036161, "frequency_penalty": 0.025369761511683464}`
- **Metadata**: `{"temp": 0.738753044605255, "diversity": 0.1459840218944142}`

### scheduler_config (score=10.00)
- **Config**: `{"sched_type": "constant", "warmup_steps": 0, "min_lr_ratio": 0.3575887680053711, "decay_steps": 9981}`
- **Metadata**: `{"sched_type": "linear", "auc": 0.7015745668780079}`

### sharq_quant (score=-18.95)
- **Config**: `{"n_levels": 31, "adaptive": false, "warmup_steps": 551}`
- **Metadata**: `{"n_levels": 6, "bits": 2.584962500721156}`

### sliding_window (score=8.79)
- **Config**: `{"window_size": 958, "stride": 451, "overlap_ratio": 0.9517924785614014}`
- **Metadata**: `{"window_size": 3757, "stride": 1713}`

### sparse_attn (score=-27.75)
- **Config**: `{"strategy": "compact", "budget_ratio": 0.8444326519966125, "block_size": 12, "min_seq_len": 916, "k_ratio": 0.45655983686447144}`
- **Metadata**: `{"fwd_err": 1.4202334149266171, "speedup": 0.3679519891738891, "sparse_ms": 0.634308298090851, "full_ms": 0.23339500003203284, "strategy": "compact", "activated": true, "budget_ratio": 0.36795198917388916, "block_size": 31, "k_ratio": 0.37451601028442383}`

### speculative_decode (score=55.06)
- **Config**: `{"n_draft_tokens": 7, "draft_model_ratio": 0.12505258917808534, "acceptance_threshold": 0.947763842344284, "temperature": 0.9975244402885437}`
- **Metadata**: `{"n_draft": 4, "acceptance": 0.665941930703122, "speedup": 3.3276363858163998}`

### streaming_kv (score=97.10)
- **Config**: `{"chunk_size": 128, "n_sink": 4, "overlap": 0.1}`
- **Metadata**: `{"chunk_size": 512, "n_sink": 4, "overlap": 0.5, "n_chunks": 15, "coverage_err": 0.0007132887840270996}`

### synthetic (score=2.55)
- **Config**: `{"x": [0.3728865385055542, 0.8155550360679626, 0.9586705565452576, 0.7770828604698181, 0.37152254581451416, 0.14454662799835205, 0.11607314646244049, 0.8105693459510803]}`
- **Metadata**: `{"rastrigin": 153.07914404345235, "deceptive": 1.5248832702636719}`

### titan_memory (score=3.31)
- **Config**: `{"memory_rank": 69, "gate_init": 0.4769386053085327, "n_memory_slots": 1, "update_freq": 2}`
- **Metadata**: `{"rank": 69, "capacity": 0.0056478034993908905, "gate": 0.5925414562225342}`

### w8a8_quant (score=178.15)
- **Config**: `{"mode": "fp8", "calib_samples": 227, "per_channel": true, "smoothquant_alpha": 0.9866266846656799}`
- **Metadata**: `{"mode": "fp8", "sqnr": 84.19366455078125}`

### xquant_kv (score=95.00)
- **Config**: `{"recomputation_ratio": 1.0, "quant_bits": 4, "checkpoint_interval": 16}`
- **Metadata**: `{"recomputation_ratio": 0.0, "quant_bits": 8, "checkpoint_interval": 2, "n_recompute": 0, "mem_ratio": 0.25, "quant_err": 0.01069291215389967}`

## Tier 3: Cross-Domain Parameter Patterns

### `acceptance_threshold`
- `0.947763842344284`: 1x, avg=55.1, domains=speculative_decode
- `0.9453415662050246`: 1x, avg=54.8, domains=speculative_decode
- `0.9459685057401657`: 1x, avg=53.2, domains=speculative_decode
### `accum_steps`
- `4`: 2x, avg=11.1, domains=grad_accum_config
- `8`: 1x, avg=11.0, domains=grad_accum_config
### `activation_budget`
- `0.10464277118444443`: 1x, avg=0.6, domains=memory_budget
- `0.11374040693044662`: 1x, avg=0.4, domains=memory_budget
- `0.2655008137226105`: 1x, avg=-0.0, domains=memory_budget
### `adaptive`
- `False`: 3x, avg=-19.2, domains=sharq_quant
### `assignment`
- `importance`: 2x, avg=-16.5, domains=mixed_precision
- `uniform`: 1x, avg=-16.4, domains=mixed_precision
### `async_copy`
- `True`: 3x, avg=0.0, domains=cpu_kv_offload
### `autocast_mode`
- `e5m2`: 2x, avg=34.8, domains=fp8_training_config
- `e4m3`: 1x, avg=32.4, domains=fp8_training_config
### `aux_loss_weight`
- `0.010835280269384386`: 1x, avg=10.0, domains=mod_config
- `0.09993527531623841`: 1x, avg=10.0, domains=mod_config
- `0.0023413712158799173`: 1x, avg=10.0, domains=mod_config
### `beam_width`
- `4`: 3x, avg=11.4, domains=beam_search
### `beta1`
- `0.9316969275474548`: 1x, avg=16.0, domains=optimizer_config
- `0.9401863425970077`: 1x, avg=15.3, domains=optimizer_config
- `0.8179830491542817`: 1x, avg=14.4, domains=optimizer_config
### `beta2`
- `0.9989514474868775`: 1x, avg=16.0, domains=optimizer_config
- `0.998871717274189`: 1x, avg=15.3, domains=optimizer_config
- `0.9984131670594215`: 1x, avg=14.4, domains=optimizer_config
### `bits_base`
- `6`: 3x, avg=-16.5, domains=mixed_precision
### `block_size`
- `16`: 6x, avg=4.1, domains=nvfp4_quant,quant
- `8`: 5x, avg=-40.8, domains=kv_eviction,sparse_attn
- `512`: 1x, avg=8.9, domains=checkpoint_recompute
- `256`: 1x, avg=8.7, domains=checkpoint_recompute
- `128`: 1x, avg=8.3, domains=checkpoint_recompute
### `budget`
- `256`: 2x, avg=-43.0, domains=kv_eviction
- `128`: 1x, avg=-55.1, domains=kv_eviction
### `budget_full`
- `0.05`: 3x, avg=4.4, domains=hqe_kv
### `budget_int4`
- `0.5`: 3x, avg=4.4, domains=hqe_kv
### `budget_int8`
- `0.5`: 3x, avg=4.4, domains=hqe_kv
### `budget_ratio`
- `0.8444326519966125`: 1x, avg=-27.7, domains=sparse_attn
- `0.75`: 1x, avg=-30.1, domains=sparse_attn
- `0.5`: 1x, avg=-32.6, domains=sparse_attn
### `cache_strategy`
- `lfu`: 1x, avg=8.8, domains=expert_hotload
- `priority`: 1x, avg=8.5, domains=expert_hotload
- `lru`: 1x, avg=8.5, domains=expert_hotload
### `calib_method`
- `percentile`: 3x, avg=9.6, domains=activation_quant
### `calib_samples`
- `227`: 1x, avg=178.2, domains=w8a8_quant
- `848`: 1x, avg=178.0, domains=w8a8_quant
- `380`: 1x, avg=177.6, domains=w8a8_quant
### `checkpoint_interval`
- `16`: 2x, avg=91.4, domains=xquant_kv
- `8`: 1x, avg=84.1, domains=xquant_kv
### `chunk_expand_size`
- `5`: 2x, avg=0.8, domains=kara
- `4`: 1x, avg=0.4, domains=kara
### `chunk_size`
- `128`: 1x, avg=97.1, domains=streaming_kv
- `409`: 1x, avg=91.8, domains=streaming_kv
- `421`: 1x, avg=91.6, domains=streaming_kv
### `codebook_size`
- `509`: 1x, avg=9.9, domains=aaac_quant
- `217`: 1x, avg=9.9, domains=aaac_quant
- `498`: 1x, avg=9.8, domains=aaac_quant
- `256`: 1x, avg=28.2, domains=kvzip_kv
- `249`: 1x, avg=-12.3, domains=kvzip_kv
### `compression`
- `int4`: 3x, avg=29.9, domains=cpu_adamw_config
### `compression_ratio`
- `6.481169760227203`: 1x, avg=-50.0, domains=gla_attention
- `3.837277263402939`: 1x, avg=-100.7, domains=gla_attention
- `1.3251490630209446`: 1x, avg=-101.0, domains=gla_attention
- `2`: 1x, avg=28.2, domains=kvzip_kv
- `5`: 1x, avg=-12.3, domains=kvzip_kv
### `decay_steps`
- `9981`: 1x, avg=10.0, domains=scheduler_config
- `9918`: 1x, avg=10.0, domains=scheduler_config
- `9400`: 1x, avg=10.0, domains=scheduler_config
### `depth_ratio`
- `0.9985441267490387`: 1x, avg=23.6, domains=mtp_config
- `0.9959570169448853`: 1x, avg=23.6, domains=mtp_config
- `0.9954531788825989`: 1x, avg=23.6, domains=mtp_config
### `dilation`
- `3`: 3x, avg=3.4, domains=conv_config
### `disk_cache_size`
- `4096`: 3x, avg=8.6, domains=expert_hotload
### `diversity_penalty`
- `0.9949210286140442`: 1x, avg=11.4, domains=beam_search
- `0.9599438905715942`: 1x, avg=11.4, domains=beam_search
- `0.9963401556015015`: 1x, avg=11.4, domains=beam_search
### `draft_model_ratio`
- `0.12505258917808534`: 1x, avg=55.1, domains=speculative_decode
- `0.11303755193948746`: 1x, avg=54.8, domains=speculative_decode
- `0.152599835395813`: 1x, avg=53.2, domains=speculative_decode
### `early_stopping`
- `True`: 3x, avg=11.4, domains=beam_search
### `epsilon`
- `0.0003313357861936093`: 1x, avg=9.6, domains=qk_norm
- `0.00011757151114195586`: 1x, avg=9.6, domains=qk_norm
- `6.446889452636243e-05`: 1x, avg=9.5, domains=qk_norm
### `eviction_policy`
- `lru`: 3x, avg=127.8, domains=paged_evict_kv
### `focal_gamma`
- `4.430350065231323`: 1x, avg=2.0, domains=loss_config
- `4.946996867656708`: 1x, avg=2.0, domains=loss_config
- `4.782994985580444`: 1x, avg=2.0, domains=loss_config
### `frequency_penalty`
- `0.025369761511683464`: 1x, avg=10.0, domains=sampling_config
- `0.08130288869142532`: 1x, avg=9.8, domains=sampling_config
- `0.07301560789346695`: 1x, avg=9.5, domains=sampling_config
### `gate_init`
- `0.990082323551178`: 1x, avg=3.0, domains=attn_residual
- `0.9927080273628235`: 1x, avg=3.0, domains=attn_residual
- `0.9684485197067261`: 1x, avg=3.0, domains=attn_residual
- `0.26289334893226624`: 1x, avg=10.2, domains=mhc_config
- `0.4239192008972168`: 1x, avg=10.0, domains=mhc_config
### `global_ratio`
- `0.9902746677398682`: 1x, avg=13.9, domains=local_global
- `0.9881688952445984`: 1x, avg=13.8, domains=local_global
- `0.9716387391090393`: 1x, avg=13.8, domains=local_global
### `global_scale_factor`
- `1.0`: 3x, avg=-6.1, domains=quant
### `grad_clip`
- `0.9958323836326599`: 1x, avg=11.1, domains=grad_accum_config
- `0.9799154996871948`: 1x, avg=11.1, domains=grad_accum_config
- `0.9991569519042969`: 1x, avg=11.0, domains=grad_accum_config
### `group_size`
- `32`: 5x, avg=6.3, domains=group_quant,hqe_kv
- `16`: 1x, avg=9.1, domains=group_quant
### `groups`
- `7`: 3x, avg=3.4, domains=conv_config
### `hadamard_dim`
- `16`: 3x, avg=-6.1, domains=quant
### `init_mode`
- `svd`: 3x, avg=19.3, domains=factorized_embed
### `init_scale`
- `1.530042827129364`: 1x, avg=-27.4, domains=bitnet_config
- `1.580079972743988`: 1x, avg=-27.5, domains=bitnet_config
- `1.506555438041687`: 1x, avg=-27.6, domains=bitnet_config
### `k_layers`
- `1`: 3x, avg=3.0, domains=attn_residual
### `k_ratio`
- `0.8`: 2x, avg=-31.3, domains=sparse_attn
- `0.45655983686447144`: 1x, avg=-27.7, domains=sparse_attn
### `keep_fraction`
- `0.5001477687765146`: 1x, avg=10.0, domains=mod_config
- `0.5003404143790249`: 1x, avg=10.0, domains=mod_config
- `0.500279855041299`: 1x, avg=10.0, domains=mod_config
### `kernel_size`
- `7`: 1x, avg=3.4, domains=conv_config
- `5`: 1x, avg=3.3, domains=conv_config
- `3`: 1x, avg=3.3, domains=conv_config
### `kv_budget`
- `0.584653913974762`: 1x, avg=0.6, domains=memory_budget
- `0.473601758480072`: 1x, avg=0.4, domains=memory_budget
- `0.3848308324813843`: 1x, avg=-0.0, domains=memory_budget
### `label_smoothing`
- `0.15508339405059815`: 1x, avg=2.0, domains=loss_config
- `0.046536339819431304`: 1x, avg=2.0, domains=loss_config
- `0.011370994895696639`: 1x, avg=2.0, domains=loss_config
### `lambda_init`
- `0.0027514046523720026`: 1x, avg=10.0, domains=diff_attn
- `0.004787000361829996`: 1x, avg=10.0, domains=diff_attn
- `0.009016690775752068`: 1x, avg=10.0, domains=diff_attn
### `latent_dim`
- `64`: 2x, avg=-100.9, domains=gla_attention
- `349`: 1x, avg=-50.0, domains=gla_attention
### `learn_offset`
- `False`: 3x, avg=-17.0, domains=offq_quant
### `learned_scale`
- `False`: 3x, avg=-27.5, domains=bitnet_config
### `length_penalty`
- `1.920774519443512`: 1x, avg=11.4, domains=beam_search
- `1.9766989350318909`: 1x, avg=11.4, domains=beam_search
- `1.8904189765453339`: 1x, avg=11.4, domains=beam_search
### `load_balance_weight`
- `0.0018776681274175644`: 1x, avg=24.7, domains=moe_routing
- `0.019475299119949344`: 1x, avg=24.0, domains=moe_routing
- `0.010000000149011612`: 1x, avg=23.9, domains=moe_routing
### `local_window`
- `1716`: 1x, avg=13.9, domains=local_global
- `1709`: 1x, avg=13.8, domains=local_global
- `1753`: 1x, avg=13.8, domains=local_global
### `loss_scale`
- `3644.6247024536133`: 1x, avg=36.4, domains=fp8_training_config
- `2625.921401977539`: 1x, avg=33.2, domains=fp8_training_config
- `4000.2507934570312`: 1x, avg=32.4, domains=fp8_training_config
### `loss_type`
- `focal`: 2x, avg=2.0, domains=loss_config
- `kl`: 1x, avg=2.0, domains=loss_config
### `loss_weight`
- `0.4932160615921021`: 1x, avg=23.6, domains=mtp_config
- `0.49945943355560307`: 1x, avg=23.6, domains=mtp_config
- `0.4997420072555542`: 1x, avg=23.6, domains=mtp_config
### `lr`
- `0.009001950696110726`: 1x, avg=16.0, domains=optimizer_config
- `0.009539753757715225`: 1x, avg=15.3, domains=optimizer_config
- `0.007468916597366333`: 1x, avg=14.4, domains=optimizer_config
### `max_batch_size`
- `15`: 2x, avg=42.9, domains=batched_decode
- `13`: 1x, avg=41.7, domains=batched_decode
### `max_seq_diff`
- `481`: 1x, avg=43.2, domains=batched_decode
- `30`: 1x, avg=42.5, domains=batched_decode
- `47`: 1x, avg=41.7, domains=batched_decode
### `memory_rank`
- `69`: 2x, avg=3.3, domains=titan_memory
- `65`: 1x, avg=3.3, domains=titan_memory
### `merge_window_ms`
- `92`: 1x, avg=43.2, domains=batched_decode
- `31`: 1x, avg=42.5, domains=batched_decode
- `61`: 1x, avg=41.7, domains=batched_decode
### `micro_batch`
- `15`: 3x, avg=11.1, domains=grad_accum_config
### `min_keep`
- `0.6856748014688492`: 1x, avg=-6.2, domains=ffn_skip
- `0.7730254530906677`: 1x, avg=-8.0, domains=ffn_skip
- `0.8374631702899933`: 1x, avg=-8.1, domains=ffn_skip
### `min_lr_ratio`
- `0.3575887680053711`: 1x, avg=10.0, domains=scheduler_config
- `0.060555968433618546`: 1x, avg=10.0, domains=scheduler_config
- `0.014402310363948345`: 1x, avg=10.0, domains=scheduler_config
### `min_seq_len`
- `256`: 2x, avg=-31.3, domains=sparse_attn
- `916`: 1x, avg=-27.7, domains=sparse_attn
### `mix_ratio`
- `0.0014527126913890243`: 1x, avg=4.1, domains=mosaic_quant
- `0.0012940019369125366`: 1x, avg=4.1, domains=mosaic_quant
- `0.0022725164890289307`: 1x, avg=4.1, domains=mosaic_quant
### `mode`
- `fp8`: 3x, avg=177.9, domains=w8a8_quant
### `mu_scaling`
- `True`: 2x, avg=34.4, domains=fp8_training_config
- `False`: 1x, avg=33.2, domains=fp8_training_config
### `n_bits`
- `2`: 3x, avg=9.9, domains=aaac_quant
- `8`: 3x, avg=9.1, domains=group_quant
### `n_checkpoint_layers`
- `16`: 3x, avg=8.7, domains=checkpoint_recompute
### `n_codebooks`
- `4`: 3x, avg=9.9, domains=aaac_quant
### `n_connections`
- `1`: 3x, avg=10.1, domains=mhc_config
### `n_conv_layers`
- `5`: 2x, avg=3.4, domains=conv_config
- `4`: 1x, avg=3.3, domains=conv_config
### `n_draft_tokens`
- `7`: 3x, avg=54.3, domains=speculative_decode
### `n_eval_layers`
- `15`: 1x, avg=-6.2, domains=ffn_skip
- `1`: 1x, avg=-8.0, domains=ffn_skip
- `2`: 1x, avg=-8.1, domains=ffn_skip
### `n_experts`
- `5`: 1x, avg=24.7, domains=moe_routing
- `4`: 1x, avg=24.0, domains=moe_routing
- `8`: 1x, avg=23.9, domains=moe_routing
### `n_global_heads`
- `15`: 1x, avg=13.9, domains=local_global
- `1`: 1x, avg=13.8, domains=local_global
- `14`: 1x, avg=13.8, domains=local_global
### `n_heads`
- `3`: 3x, avg=23.6, domains=mtp_config
- `17`: 2x, avg=-45.5, domains=diff_attn,gla_attention
- `23`: 1x, avg=10.0, domains=diff_attn
- `7`: 1x, avg=10.0, domains=diff_attn
- `4`: 1x, avg=-50.0, domains=gla_attention
### `n_hot_experts`
- `4`: 2x, avg=8.7, domains=expert_hotload
- `5`: 1x, avg=8.5, domains=expert_hotload
### `n_iter`
- `10`: 1x, avg=28.2, domains=kvzip_kv
- `54`: 1x, avg=-12.3, domains=kvzip_kv
- `73`: 1x, avg=-20.2, domains=kvzip_kv
- `34`: 1x, avg=-16.8, domains=offq_quant
- `24`: 1x, avg=-17.0, domains=offq_quant
### `n_kv_heads`
- `4`: 1x, avg=5.0, domains=gta_attention
- `10`: 1x, avg=5.0, domains=gta_attention
- `5`: 1x, avg=4.6, domains=gta_attention
### `n_levels`
- `2`: 3x, avg=-16.5, domains=mixed_precision
- `31`: 2x, avg=-19.0, domains=sharq_quant
- `28`: 1x, avg=-19.7, domains=sharq_quant
### `n_memory_slots`
- `1`: 3x, avg=3.3, domains=titan_memory
### `n_pages`
- `64`: 1x, avg=128.1, domains=paged_evict_kv
- `128`: 1x, avg=128.1, domains=paged_evict_kv
- `256`: 1x, avg=127.1, domains=paged_evict_kv
### `n_rotations`
- `1`: 2x, avg=-1.7, domains=rotor_quant_kv
- `2`: 1x, avg=1.5, domains=rotor_quant_kv
### `n_share_groups`
- `1`: 3x, avg=203988.7, domains=cross_layer_kv
### `n_sink`
- `4`: 1x, avg=97.1, domains=streaming_kv
- `10`: 1x, avg=91.8, domains=streaming_kv
- `6`: 1x, avg=91.6, domains=streaming_kv
### `n_sinks`
- `2`: 2x, avg=-43.0, domains=kv_eviction
- `6`: 1x, avg=-55.1, domains=kv_eviction
### `n_skip_layers`
- `4`: 1x, avg=10.0, domains=mod_config
- `14`: 1x, avg=10.0, domains=mod_config
- `15`: 1x, avg=10.0, domains=mod_config
### `n_tiles`
- `16`: 1x, avg=4.1, domains=mosaic_quant
- `13`: 1x, avg=4.1, domains=mosaic_quant
- `5`: 1x, avg=4.1, domains=mosaic_quant
### `norm_type`
- `layernorm`: 2x, avg=9.6, domains=qk_norm
- `rmsnorm`: 1x, avg=9.5, domains=qk_norm
### `observation_window`
- `512`: 2x, avg=-43.0, domains=kv_eviction
- `32`: 1x, avg=-55.1, domains=kv_eviction
### `offload_layers`
- `1`: 3x, avg=0.0, domains=cpu_kv_offload
### `offload_ratio`
- `0.004509661812335253`: 1x, avg=29.9, domains=cpu_adamw_config
- `0.007042866665869951`: 1x, avg=29.9, domains=cpu_adamw_config
- `0.013318986631929874`: 1x, avg=29.8, domains=cpu_adamw_config
- `0.960943341255188`: 1x, avg=4.2, domains=hybrid_offload
- `0.959338903427124`: 1x, avg=4.2, domains=hybrid_offload
### `offload_threshold`
- `0.6295778751373291`: 1x, avg=0.0, domains=cpu_kv_offload
- `0.8856743574142456`: 1x, avg=0.0, domains=cpu_kv_offload
- `0.24549582600593567`: 1x, avg=0.0, domains=cpu_kv_offload
### `offset_init`
- `0.03407374396920204`: 1x, avg=-16.8, domains=offq_quant
- `0.10908336937427521`: 1x, avg=-17.0, domains=offq_quant
- `0.06451083719730377`: 1x, avg=-17.1, domains=offq_quant
### `opt_type`
- `adamw`: 3x, avg=15.2, domains=optimizer_config
### `overlap`
- `0.1`: 1x, avg=97.1, domains=streaming_kv
- `0.4617225229740143`: 1x, avg=91.8, domains=streaming_kv
- `0.29872429370880127`: 1x, avg=91.6, domains=streaming_kv
### `overlap_compute`
- `True`: 3x, avg=4.1, domains=hybrid_offload
### `overlap_ratio`
- `0.9517924785614014`: 1x, avg=8.8, domains=sliding_window
- `0.654741644859314`: 1x, avg=8.8, domains=sliding_window
- `0.925091028213501`: 1x, avg=8.8, domains=sliding_window
### `padding_strategy`
- `dynamic`: 1x, avg=43.2, domains=batched_decode
- `right`: 1x, avg=42.5, domains=batched_decode
- `left`: 1x, avg=41.7, domains=batched_decode
### `page_size`
- `64`: 1x, avg=128.1, domains=paged_evict_kv
- `32`: 1x, avg=128.1, domains=paged_evict_kv
- `16`: 1x, avg=127.1, domains=paged_evict_kv
### `pattern_type`
- `csa_hca_hybrid`: 3x, avg=14.6, domains=csa_attention
### `per_channel`
- `True`: 3x, avg=177.9, domains=w8a8_quant
### `percentile`
- `0.9014481891123578`: 1x, avg=9.6, domains=activation_quant
- `0.9025344122909009`: 1x, avg=9.6, domains=activation_quant
- `0.9186696035563946`: 1x, avg=9.6, domains=activation_quant
### `pin_memory`
- `True`: 3x, avg=4.1, domains=hybrid_offload
### `prefetch_ahead`
- `4`: 2x, avg=8.7, domains=expert_hotload
- `2`: 1x, avg=8.5, domains=expert_hotload
### `prefetch_depth`
- `8`: 3x, avg=4.1, domains=hybrid_offload
- `5`: 2x, avg=29.9, domains=cpu_adamw_config
- `7`: 1x, avg=29.8, domains=cpu_adamw_config
### `prefetch_size`
- `2048`: 2x, avg=0.0, domains=cpu_kv_offload
- `1024`: 1x, avg=0.0, domains=cpu_kv_offload
### `quant_bits`
- `4`: 4x, avg=23.3, domains=rotor_quant_kv,xquant_kv
- `8`: 2x, avg=86.0, domains=xquant_kv
### `quant_mode`
- `ternary`: 3x, avg=-27.5, domains=bitnet_config
### `rank`
- `64`: 1x, avg=19.3, domains=factorized_embed
- `65`: 1x, avg=19.3, domains=factorized_embed
- `67`: 1x, avg=19.3, domains=factorized_embed
- `70`: 1x, avg=10.2, domains=mhc_config
- `122`: 1x, avg=10.0, domains=mhc_config
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
- `1.0291668307036161`: 1x, avg=10.0, domains=sampling_config
- `1.007961212657392`: 1x, avg=9.8, domains=sampling_config
- `1.0728673562407494`: 1x, avg=9.5, domains=sampling_config
### `reserve`
- `0.05720803886651993`: 1x, avg=0.6, domains=memory_budget
- `0.15885864198207855`: 1x, avg=0.4, domains=memory_budget
- `0.18697446584701538`: 1x, avg=-0.0, domains=memory_budget
### `residual_ratio`
- `0.05`: 1x, avg=-5.0, domains=quant
- `0.1`: 1x, avg=-6.1, domains=quant
- `0.2`: 1x, avg=-7.3, domains=quant
### `retrieval_dim`
- `252`: 1x, avg=3.0, domains=attn_residual
- `493`: 1x, avg=3.0, domains=attn_residual
- `174`: 1x, avg=3.0, domains=attn_residual
### `rot_type`
- `random`: 2x, avg=-1.7, domains=rotor_quant_kv
- `hadamard`: 1x, avg=1.5, domains=rotor_quant_kv
### `rounding_method`
- `rtn`: 3x, avg=-6.1, domains=quant
### `router_mode`
- `aux_free`: 2x, avg=24.0, domains=moe_routing
- `switch`: 1x, avg=24.7, domains=moe_routing
### `router_type`
- `mlp`: 3x, avg=10.0, domains=mod_config
### `scale_clip_min`
- `0.001`: 3x, avg=-6.1, domains=quant
### `scale_init`
- `0.5054262886987999`: 1x, avg=9.6, domains=qk_norm
- `0.5239835530519485`: 1x, avg=9.6, domains=qk_norm
- `0.7491976022720337`: 1x, avg=9.5, domains=qk_norm
### `scale_method`
- `absmax`: 3x, avg=-6.1, domains=quant
### `scale_mode`
- `per_channel`: 2x, avg=14.3, domains=nvfp4_quant
- `per_block`: 1x, avg=14.3, domains=nvfp4_quant
### `scale_search_range`
- `0.3`: 3x, avg=-6.1, domains=quant
### `scale_search_steps`
- `5`: 3x, avg=-6.1, domains=quant
### `scaling_factor`
- `1.9816195666790009`: 1x, avg=36.1, domains=rope_config
- `3.9899935126304626`: 1x, avg=36.1, domains=rope_config
- `3.971285432577133`: 1x, avg=36.1, domains=rope_config
### `scaling_type`
- `linear`: 2x, avg=36.1, domains=rope_config
- `none`: 1x, avg=36.1, domains=rope_config
### `sched_type`
- `constant`: 1x, avg=10.0, domains=scheduler_config
- `cosine`: 1x, avg=10.0, domains=scheduler_config
- `linear`: 1x, avg=10.0, domains=scheduler_config
### `scheme`
- `asymmetric`: 3x, avg=9.1, domains=group_quant
### `share_mode`
- `max`: 3x, avg=203988.7, domains=cross_layer_kv
### `share_ratio`
- `1.0`: 1x, avg=218453.5, domains=cross_layer_kv
- `0.8983016014099121`: 1x, avg=204338.3, domains=cross_layer_kv
- `0.75`: 1x, avg=189174.3, domains=cross_layer_kv
### `share_weights`
- `True`: 3x, avg=23.6, domains=mtp_config
### `shared_expert`
- `True`: 2x, avg=24.3, domains=moe_routing
- `False`: 1x, avg=24.0, domains=moe_routing
### `sink_size`
- `5`: 2x, avg=0.4, domains=kara
- `2`: 1x, avg=1.3, domains=kara
### `skip_strategy`
- `cosine`: 2x, avg=-8.1, domains=ffn_skip
- `hybrid`: 1x, avg=-6.2, domains=ffn_skip
### `skip_threshold`
- `0.004109608940780163`: 1x, avg=-6.2, domains=ffn_skip
- `0.006224475335329771`: 1x, avg=-8.0, domains=ffn_skip
- `0.0037541568744927645`: 1x, avg=-8.1, domains=ffn_skip
### `smooth_alpha`
- `0.4332141578197479`: 1x, avg=9.6, domains=activation_quant
- `0.7158360481262207`: 1x, avg=9.6, domains=activation_quant
- `0.7532713413238525`: 1x, avg=9.6, domains=activation_quant
### `smooth_swiglu`
- `False`: 3x, avg=34.0, domains=fp8_training_config
### `smoothquant_alpha`
- `0.9866266846656799`: 1x, avg=178.2, domains=w8a8_quant
- `0.9965072274208069`: 1x, avg=178.0, domains=w8a8_quant
- `0.9502096176147461`: 1x, avg=177.6, domains=w8a8_quant
### `softmax_sep`
- `0.945634126663208`: 1x, avg=10.0, domains=diff_attn
- `0.019246352836489677`: 1x, avg=10.0, domains=diff_attn
- `0.004943231586366892`: 1x, avg=10.0, domains=diff_attn
### `strategy`
- `mosa`: 2x, avg=-31.3, domains=sparse_attn
- `paged`: 1x, avg=-41.4, domains=kv_eviction
- `snapkv`: 1x, avg=-44.6, domains=kv_eviction
- `streaming`: 1x, avg=-55.1, domains=kv_eviction
- `compact`: 1x, avg=-27.7, domains=sparse_attn
### `stride`
- `1`: 2x, avg=3.4, domains=conv_config
- `2`: 1x, avg=3.3, domains=conv_config
- `451`: 1x, avg=8.8, domains=sliding_window
- `454`: 1x, avg=8.8, domains=sliding_window
- `466`: 1x, avg=8.8, domains=sliding_window
### `sync_freq`
- `15`: 2x, avg=11.1, domains=grad_accum_config
- `14`: 1x, avg=11.0, domains=grad_accum_config
### `target_budget`
- `688`: 1x, avg=1.3, domains=kara
- `638`: 1x, avg=0.4, domains=kara
- `1538`: 1x, avg=0.4, domains=kara
### `temperature`
- `1.9878134727478027`: 1x, avg=2.0, domains=loss_config
- `1.9265738129615784`: 1x, avg=2.0, domains=loss_config
- `1.9639817774295807`: 1x, avg=2.0, domains=loss_config
- `1.673143845796585`: 1x, avg=10.0, domains=sampling_config
- `0.3173029638826847`: 1x, avg=9.8, domains=sampling_config
### `theta`
- `9997342.494666576`: 1x, avg=36.1, domains=rope_config
- `9988485.534191132`: 1x, avg=36.1, domains=rope_config
- `9976888.822197914`: 1x, avg=36.1, domains=rope_config
### `threshold`
- `0.9`: 1x, avg=74.5, domains=kv_recompute
- `0.9175277948379517`: 1x, avg=70.7, domains=kv_recompute
- `0.7`: 1x, avg=66.8, domains=kv_recompute
### `tie_factor`
- `0.15296828746795654`: 1x, avg=19.3, domains=factorized_embed
- `0.9766613841056824`: 1x, avg=19.3, domains=factorized_embed
- `0.10892506688833237`: 1x, avg=19.3, domains=factorized_embed
### `tie_strength`
- `0.6417432427406311`: 1x, avg=5.0, domains=gta_attention
- `0.9470784068107605`: 1x, avg=5.0, domains=gta_attention
- `0.7005675435066223`: 1x, avg=4.6, domains=gta_attention
### `tile_dim`
- `95`: 1x, avg=4.1, domains=mosaic_quant
- `159`: 1x, avg=4.1, domains=mosaic_quant
- `100`: 1x, avg=4.1, domains=mosaic_quant
### `top_k`
- `3`: 2x, avg=24.3, domains=moe_routing
- `64`: 1x, avg=14.8, domains=csa_attention
- `70`: 1x, avg=14.7, domains=csa_attention
- `194`: 1x, avg=14.5, domains=csa_attention
- `2`: 1x, avg=23.9, domains=moe_routing
### `top_p`
- `0.9963274002075195`: 1x, avg=10.0, domains=sampling_config
- `0.9124466776847839`: 1x, avg=9.8, domains=sampling_config
- `0.9068709313869476`: 1x, avg=9.5, domains=sampling_config
### `update_freq`
- `13`: 1x, avg=29.9, domains=cpu_adamw_config
- `14`: 1x, avg=29.9, domains=cpu_adamw_config
- `4`: 1x, avg=29.8, domains=cpu_adamw_config
- `2`: 1x, avg=3.3, domains=titan_memory
- `5`: 1x, avg=3.3, domains=titan_memory
### `use_hadamard`
- `False`: 3x, avg=-6.1, domains=quant
### `v_k_mix`
- `2.7122376195620745e-05`: 1x, avg=5.0, domains=gta_attention
- `0.00011373747838661075`: 1x, avg=5.0, domains=gta_attention
- `0.01458786241710186`: 1x, avg=4.6, domains=gta_attention
### `vocab_size`
- `47660`: 1x, avg=19.3, domains=factorized_embed
- `32945`: 1x, avg=19.3, domains=factorized_embed
- `33300`: 1x, avg=19.3, domains=factorized_embed
### `w4a8`
- `True`: 3x, avg=14.3, domains=nvfp4_quant
### `warmup_steps`
- `0`: 1x, avg=10.0, domains=scheduler_config
- `18`: 1x, avg=10.0, domains=scheduler_config
- `19`: 1x, avg=10.0, domains=scheduler_config
- `551`: 1x, avg=-18.9, domains=sharq_quant
- `121`: 1x, avg=-19.0, domains=sharq_quant
### `weight_budget`
- `0.18910351395606995`: 1x, avg=0.6, domains=memory_budget
- `0.21298974752426147`: 1x, avg=0.4, domains=memory_budget
- `0.21484771370887756`: 1x, avg=-0.0, domains=memory_budget
### `weight_decay`
- `0.002937338948249817`: 1x, avg=11.2, domains=muon_config
- `0.08693135380744935`: 1x, avg=16.0, domains=optimizer_config
- `0.09906891584396363`: 1x, avg=15.3, domains=optimizer_config
- `0.04970521628856659`: 1x, avg=14.4, domains=optimizer_config
### `window_size`
- `128`: 2x, avg=-43.0, domains=kv_eviction
- `649`: 1x, avg=1.3, domains=kara
- `894`: 1x, avg=0.4, domains=kara
- `1008`: 1x, avg=0.4, domains=kara
- `1024`: 1x, avg=-55.1, domains=kv_eviction
### `x`
- `[0.3728865385055542, 0.8155550360679626, 0.9586705565452576, 0.7770828604698181, 0.37152254581451416, 0.14454662799835205, 0.11607314646244049, 0.8105693459510803]`: 1x, avg=2.6, domains=synthetic
- `[0.9361981749534607, 0.7527074217796326, 0.9471004605293274, 0.6988914608955383, 0.5228747129440308, 0.31916314363479614, 0.2205742448568344, 0.6712738871574402]`: 1x, avg=-10.0, domains=synthetic
- `[0.9420604109764099, 0.17569594085216522, 0.9176703095436096, 0.0026985788717865944, 0.4038127660751343, 0.193594291806221, 0.05605176091194153, 0.6629123091697693]`: 1x, avg=-10.1, domains=synthetic
