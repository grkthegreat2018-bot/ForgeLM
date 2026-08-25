# ForgeEvolve 'boot' Run: Top Optimization Ideas
Generated: 2026-08-24 20:41:39
Domains: 146

## Tier 1: Top 50 Configurations

| Rank | Domain | Score | Config |
|------|--------|-------|--------|
| 1 | w8a8_quant | 179.47 | `{"mode": "fp8", "calib_samples": 65, "per_channel": false, "smoothquant_alpha...` |
| 2 | w8a8_quant | 179.43 | `{"mode": "fp8", "calib_samples": 938, "per_channel": true, "smoothquant_alpha...` |
| 3 | w8a8_quant | 178.95 | `{"mode": "int8", "calib_samples": 73, "per_channel": true, "smoothquant_alpha...` |
| 4 | w8a8_quant_refine_d1 | 178.75 | `{"mode": "int8", "calib_samples": 948, "per_channel": true, "smoothquant_alph...` |
| 5 | w8a8_quant | 178.59 | `{"mode": "int8", "calib_samples": 290, "per_channel": false, "smoothquant_alp...` |
| 6 | w8a8_quant | 178.44 | `{"mode": "fp8", "calib_samples": 975, "per_channel": true, "smoothquant_alpha...` |
| 7 | w8a8_quant | 178.43 | `{"mode": "fp8", "calib_samples": 1023, "per_channel": false, "smoothquant_alp...` |
| 8 | w8a8_quant | 178.38 | `{"mode": "fp8", "calib_samples": 1020, "per_channel": true, "smoothquant_alph...` |
| 9 | w8a8_quant | 178.35 | `{"mode": "fp8", "calib_samples": 371, "per_channel": false, "smoothquant_alph...` |
| 10 | w8a8_quant | 178.32 | `{"mode": "int8", "calib_samples": 89, "per_channel": true, "smoothquant_alpha...` |
| 11 | w8a8_quant | 178.30 | `{"mode": "fp8", "calib_samples": 141, "per_channel": true, "smoothquant_alpha...` |
| 12 | w8a8_quant_refine_d1 | 178.30 | `{"mode": "fp8", "calib_samples": 64, "per_channel": false, "smoothquant_alpha...` |
| 13 | w8a8_quant_refine_d1 | 178.27 | `{"mode": "fp8", "calib_samples": 64, "per_channel": false, "smoothquant_alpha...` |
| 14 | w8a8_quant | 178.21 | `{"mode": "fp8", "calib_samples": 1007, "per_channel": false, "smoothquant_alp...` |
| 15 | w8a8_quant | 178.19 | `{"mode": "fp8", "calib_samples": 480, "per_channel": true, "smoothquant_alpha...` |
| 16 | w8a8_quant | 178.16 | `{"mode": "fp8", "calib_samples": 1017, "per_channel": true, "smoothquant_alph...` |
| 17 | w8a8_quant | 178.13 | `{"mode": "fp8", "calib_samples": 75, "per_channel": true, "smoothquant_alpha"...` |
| 18 | w8a8_quant | 178.13 | `{"mode": "fp8", "calib_samples": 104, "per_channel": true, "smoothquant_alpha...` |
| 19 | w8a8_quant | 178.12 | `{"mode": "fp8", "calib_samples": 961, "per_channel": true, "smoothquant_alpha...` |
| 20 | w8a8_quant | 178.05 | `{"mode": "fp8", "calib_samples": 1015, "per_channel": true, "smoothquant_alph...` |
| 21 | w8a8_quant | 178.04 | `{"mode": "int8", "calib_samples": 1015, "per_channel": true, "smoothquant_alp...` |
| 22 | w8a8_quant | 178.01 | `{"mode": "fp8", "calib_samples": 219, "per_channel": true, "smoothquant_alpha...` |
| 23 | w8a8_quant | 178.00 | `{"mode": "fp8", "calib_samples": 1006, "per_channel": true, "smoothquant_alph...` |
| 24 | w8a8_quant_refine_d1 | 178.00 | `{"mode": "int8", "calib_samples": 1024, "per_channel": true, "smoothquant_alp...` |
| 25 | w8a8_quant_refine_d1 | 177.44 | `{"mode": "int8", "calib_samples": 1024, "per_channel": true, "smoothquant_alp...` |
| 26 | w8a8_quant_refine_d1 | 177.11 | `{"mode": "int8", "calib_samples": 1024, "per_channel": true, "smoothquant_alp...` |
| 27 | w8a8_quant_refine_d1 | 176.64 | `{"mode": "fp8", "calib_samples": 919, "per_channel": true, "smoothquant_alpha...` |
| 28 | paged_evict_kv_refine_d1 | 128.17 | `{"page_size": 41, "n_pages": 100, "eviction_policy": "lru"}` |
| 29 | paged_evict_kv | 128.11 | `{"page_size": 64, "n_pages": 64, "eviction_policy": "lru"}` |
| 30 | paged_evict_kv | 128.09 | `{"page_size": 32, "n_pages": 128, "eviction_policy": "lru"}` |
| 31 | paged_evict_kv | 127.13 | `{"page_size": 16, "n_pages": 256, "eviction_policy": "lru"}` |
| 32 | paged_evict_kv | 125.01 | `{"page_size": 55, "n_pages": 83, "eviction_policy": "lfu"}` |
| 33 | paged_evict_kv_refine_d1 | 124.87 | `{"page_size": 54, "n_pages": 85, "eviction_policy": "lru"}` |
| 34 | paged_evict_kv_refine_d1 | 124.77 | `{"page_size": 48, "n_pages": 96, "eviction_policy": "lru"}` |
| 35 | paged_evict_kv | 124.62 | `{"page_size": 72, "n_pages": 64, "eviction_policy": "lru"}` |
| 36 | paged_evict_kv_refine_d1 | 124.24 | `{"page_size": 73, "n_pages": 64, "eviction_policy": "lru"}` |
| 37 | paged_evict_kv_refine_d1 | 123.96 | `{"page_size": 47, "n_pages": 101, "eviction_policy": "lru"}` |
| 38 | paged_evict_kv_refine_d1 | 121.85 | `{"page_size": 50, "n_pages": 103, "eviction_policy": "lru"}` |
| 39 | paged_evict_kv_refine_d1 | 121.85 | `{"page_size": 51, "n_pages": 101, "eviction_policy": "lru"}` |
| 40 | paged_evict_kv | 121.64 | `{"page_size": 49, "n_pages": 106, "eviction_policy": "importance"}` |
| 41 | paged_evict_kv_refine_d1 | 121.56 | `{"page_size": 65, "n_pages": 80, "eviction_policy": "lru"}` |
| 42 | paged_evict_kv | 121.50 | `{"page_size": 45, "n_pages": 116, "eviction_policy": "lru"}` |
| 43 | paged_evict_kv_refine_d1 | 121.48 | `{"page_size": 81, "n_pages": 64, "eviction_policy": "lru"}` |
| 44 | paged_evict_kv | 121.46 | `{"page_size": 29, "n_pages": 178, "eviction_policy": "lru"}` |
| 45 | paged_evict_kv_refine_d1 | 120.94 | `{"page_size": 71, "n_pages": 75, "eviction_policy": "lru"}` |
| 46 | paged_evict_kv | 120.87 | `{"page_size": 83, "n_pages": 64, "eviction_policy": "lfu"}` |
| 47 | paged_evict_kv_refine_d1 | 120.69 | `{"page_size": 80, "n_pages": 67, "eviction_policy": "lru"}` |
| 48 | paged_evict_kv | 120.15 | `{"page_size": 16, "n_pages": 324, "eviction_policy": "lfu"}` |
| 49 | paged_evict_kv_refine_d1 | 118.73 | `{"page_size": 77, "n_pages": 76, "eviction_policy": "lru"}` |
| 50 | paged_evict_kv | 118.70 | `{"page_size": 62, "n_pages": 95, "eviction_policy": "lfu"}` |

## Tier 2: Best Per Domain

### Generic_use_adaptive_spec (score=64.93)
- **Config**: `{"use_adaptive_spec": 0.0032670486252754927, "enabled": 0.9984026551246643}`
- **Metadata**: `{"topic": "use_adaptive_spec", "generic": true}`

### Generic_use_adaptive_spec_refine_d1 (score=65.00)
- **Config**: `{"use_adaptive_spec": 0.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_adaptive_spec", "generic": true}`

### Generic_use_block_fusion (score=64.49)
- **Config**: `{"use_block_fusion": 0.9691361784934998, "enabled": 0.003460934152826667}`
- **Metadata**: `{"topic": "use_block_fusion", "generic": true}`

### Generic_use_block_fusion_refine_d1 (score=64.91)
- **Config**: `{"use_block_fusion": 0.9941515922546387, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_block_fusion", "generic": true}`

### Generic_use_breakable_cuda_graph (score=64.78)
- **Config**: `{"use_breakable_cuda_graph": 0.997200608253479, "enabled": 0.9881133437156677}`
- **Metadata**: `{"topic": "use_breakable_cuda_graph", "generic": true}`

### Generic_use_chunked_prefill (score=64.88)
- **Config**: `{"use_chunked_prefill": 0.005063542630523443, "enabled": 0.003023377386853099}`
- **Metadata**: `{"topic": "use_chunked_prefill", "generic": true}`

### Generic_use_chunked_prefill_refine_d1 (score=65.00)
- **Config**: `{"use_chunked_prefill": 1.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_chunked_prefill", "generic": true}`

### Generic_use_compact_attn (score=64.51)
- **Config**: `{"use_compact_attn": 0.993290901184082, "enabled": 0.9738672375679016}`
- **Metadata**: `{"topic": "use_compact_attn", "generic": true}`

### Generic_use_compact_attn_refine_d1 (score=65.00)
- **Config**: `{"use_compact_attn": 1.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_compact_attn", "generic": true}`

### Generic_use_compile (score=64.97)
- **Config**: `{"use_compile": 0.9983477592468262, "enabled": 0.00026365526719018817}`
- **Metadata**: `{"topic": "use_compile", "generic": true}`

### Generic_use_compile_autotune (score=64.81)
- **Config**: `{"use_compile_autotune": 0.990142822265625, "enabled": 0.9969174861907959}`
- **Metadata**: `{"topic": "use_compile_autotune", "generic": true}`

### Generic_use_compile_autotune_refine_d1 (score=65.00)
- **Config**: `{"use_compile_autotune": 0.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_compile_autotune", "generic": true}`

### Generic_use_compile_refine_d1 (score=65.00)
- **Config**: `{"use_compile": 1.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_compile", "generic": true}`

### Generic_use_corun (score=64.65)
- **Config**: `{"use_corun": 0.010578980669379234, "enabled": 0.9875099658966064}`
- **Metadata**: `{"topic": "use_corun", "generic": true}`

### Generic_use_corun_refine_d1 (score=65.00)
- **Config**: `{"use_corun": 0.0, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_corun", "generic": true}`

### Generic_use_cosa (score=64.82)
- **Config**: `{"use_cosa": 0.002034958219155669, "enabled": 0.9898774027824402}`
- **Metadata**: `{"topic": "use_cosa", "generic": true}`

### Generic_use_cosa_refine_d1 (score=65.00)
- **Config**: `{"use_cosa": 0.00019374489784240723, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_cosa", "generic": true}`

### Generic_use_foundry (score=64.58)
- **Config**: `{"use_foundry": 0.022415120154619217, "enabled": 0.9946938157081604}`
- **Metadata**: `{"topic": "use_foundry", "generic": true}`

### Generic_use_foundry_refine_d1 (score=65.00)
- **Config**: `{"use_foundry": 1.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_foundry", "generic": true}`

### Generic_use_fused_qk_norm_rope_cache (score=64.75)
- **Config**: `{"use_fused_qk_norm_rope_cache": 0.00599229522049427, "enabled": 0.010592309758067131}`
- **Metadata**: `{"topic": "use_fused_qk_norm_rope_cache", "generic": true}`

### Generic_use_fused_qk_norm_rope_cache_refine_d1 (score=65.00)
- **Config**: `{"use_fused_qk_norm_rope_cache": 0.0, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_fused_qk_norm_rope_cache", "generic": true}`

### Generic_use_hotprefix (score=64.72)
- **Config**: `{"use_hotprefix": 0.011793042533099651, "enabled": 0.006808698642998934}`
- **Metadata**: `{"topic": "use_hotprefix", "generic": true}`

### Generic_use_hotprefix_refine_d1 (score=65.00)
- **Config**: `{"use_hotprefix": 0.0, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_hotprefix", "generic": true}`

### Generic_use_hybrid_prefill (score=64.59)
- **Config**: `{"use_hybrid_prefill": 0.9811527729034424, "enabled": 0.008536428213119507}`
- **Metadata**: `{"topic": "use_hybrid_prefill", "generic": true}`

### Generic_use_hybrid_prefill_refine_d1 (score=65.00)
- **Config**: `{"use_hybrid_prefill": 1.0, "enabled": 9.740889072418213e-05}`
- **Metadata**: `{"topic": "use_hybrid_prefill", "generic": true}`

### Generic_use_learned_prefix_cache (score=64.55)
- **Config**: `{"use_learned_prefix_cache": 0.021552495658397675, "enabled": 0.008734255097806454}`
- **Metadata**: `{"topic": "use_learned_prefix_cache", "generic": true}`

### Generic_use_learned_prefix_cache_refine_d1 (score=65.00)
- **Config**: `{"use_learned_prefix_cache": 0.0, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_learned_prefix_cache", "generic": true}`

### Generic_use_mosa (score=64.32)
- **Config**: `{"use_mosa": 0.033636439591646194, "enabled": 0.011369445361196995}`
- **Metadata**: `{"topic": "use_mosa", "generic": true}`

### Generic_use_mosa_refine_d1 (score=65.00)
- **Config**: `{"use_mosa": 0.0, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_mosa", "generic": true}`

### Generic_use_pod_attention (score=64.82)
- **Config**: `{"use_pod_attention": 0.9900026321411133, "enabled": 0.001953455852344632}`
- **Metadata**: `{"topic": "use_pod_attention", "generic": true}`

### Generic_use_pod_attention_refine_d1 (score=65.00)
- **Config**: `{"use_pod_attention": 0.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_pod_attention", "generic": true}`

### Generic_use_prefix_cache (score=64.81)
- **Config**: `{"use_prefix_cache": 0.9933805465698242, "enabled": 0.9940292835235596}`
- **Metadata**: `{"topic": "use_prefix_cache", "generic": true}`

### Generic_use_prefix_cache_refine_d1 (score=65.00)
- **Config**: `{"use_prefix_cache": 1.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_prefix_cache", "generic": true}`

### Generic_use_progressive_kv (score=64.77)
- **Config**: `{"use_progressive_kv": 0.9936783909797668, "enabled": 0.9911388754844666}`
- **Metadata**: `{"topic": "use_progressive_kv", "generic": true}`

### Generic_use_progressive_kv_refine_d1 (score=65.00)
- **Config**: `{"use_progressive_kv": 0.0, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_progressive_kv", "generic": true}`

### Generic_use_seq_split (score=64.77)
- **Config**: `{"use_seq_split": 0.00903824158012867, "enabled": 0.006478232331573963}`
- **Metadata**: `{"topic": "use_seq_split", "generic": true}`

### Generic_use_seq_split_refine_d1 (score=65.00)
- **Config**: `{"use_seq_split": 1.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_seq_split", "generic": true}`

### Generic_use_spec_attn (score=64.72)
- **Config**: `{"use_spec_attn": 0.003011946566402912, "enabled": 0.9845836162567139}`
- **Metadata**: `{"topic": "use_spec_attn", "generic": true}`

### Generic_use_spec_attn_refine_d1 (score=65.00)
- **Config**: `{"use_spec_attn": 0.9997426271438599, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_spec_attn", "generic": true}`

### Generic_use_suffix_spec (score=64.75)
- **Config**: `{"use_suffix_spec": 0.013989963568747044, "enabled": 0.9970368146896362}`
- **Metadata**: `{"topic": "use_suffix_spec", "generic": true}`

### Generic_use_suffix_spec_refine_d1 (score=65.00)
- **Config**: `{"use_suffix_spec": 1.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_suffix_spec", "generic": true}`

### Generic_use_triroute (score=64.69)
- **Config**: `{"use_triroute": 0.0065451497212052345, "enabled": 0.9858196973800659}`
- **Metadata**: `{"topic": "use_triroute", "generic": true}`

### Generic_use_triroute_refine_d1 (score=65.00)
- **Config**: `{"use_triroute": 0.0, "enabled": 0.9999708533287048}`
- **Metadata**: `{"topic": "use_triroute", "generic": true}`

### Generic_use_triton_conv (score=64.85)
- **Config**: `{"use_triton_conv": 0.004687928128987551, "enabled": 0.9948917627334595}`
- **Metadata**: `{"topic": "use_triton_conv", "generic": true}`

### Generic_use_triton_conv_refine_d1 (score=65.00)
- **Config**: `{"use_triton_conv": 1.0, "enabled": 1.0}`
- **Metadata**: `{"topic": "use_triton_conv", "generic": true}`

### Generic_use_v0_warm (score=64.86)
- **Config**: `{"use_v0_warm": 0.9987877011299133, "enabled": 0.008416852913796902}`
- **Metadata**: `{"topic": "use_v0_warm", "generic": true}`

### Generic_use_v0_warm_refine_d1 (score=65.00)
- **Config**: `{"use_v0_warm": 1.0, "enabled": 0.000131167471408844}`
- **Metadata**: `{"topic": "use_v0_warm", "generic": true}`

### Generic_use_wavelength_pruning (score=64.90)
- **Config**: `{"use_wavelength_pruning": 0.0034131938591599464, "enabled": 0.003070124192163348}`
- **Metadata**: `{"topic": "use_wavelength_pruning", "generic": true}`

### Generic_use_wavelength_pruning_refine_d1 (score=65.00)
- **Config**: `{"use_wavelength_pruning": 0.0, "enabled": 0.0}`
- **Metadata**: `{"topic": "use_wavelength_pruning", "generic": true}`

### aaac_quant (score=9.96)
- **Config**: `{"n_codebooks": 4, "codebook_size": 401, "n_bits": 2}`
- **Metadata**: `{"n_codebooks": 4, "compression": 4.0}`

### activation_quant (score=9.63)
- **Config**: `{"calib_method": "percentile", "percentile": 0.9000000428387552, "smooth_alpha": 0.999768078327179}`
- **Metadata**: `{"method": "percentile", "err": 0.003701101072756236}`

### attn_residual (score=3.25)
- **Config**: `{"k_layers": 1, "gate_init": 0.9947109222412109, "retrieval_dim": 64}`
- **Metadata**: `{"k": 1, "gate": 0.9947109222412109, "retrieval_dim": 64}`

### attn_residual_refine_d1 (score=3.27)
- **Config**: `{"k_layers": 1, "gate_init": 1.0, "retrieval_dim": 64}`
- **Metadata**: `{"k": 1, "gate": 1.0, "retrieval_dim": 64}`

### batched_decode (score=44.15)
- **Config**: `{"max_batch_size": 15, "padding_strategy": "left", "merge_window_ms": 52, "max_seq_diff": 0}`
- **Metadata**: `{"batch_size": 15, "throughput": 8.25, "latency_penalty": 0.1}`

### batched_decode_refine_d1 (score=42.65)
- **Config**: `{"max_batch_size": 13, "padding_strategy": "left", "merge_window_ms": 56, "max_seq_diff": 0}`
- **Metadata**: `{"batch_size": 13, "throughput": 7.93}`

### beam_search (score=10.94)
- **Config**: `{"beam_width": 6, "length_penalty": 0.991542786359787, "early_stopping": true, "diversity_penalty": 0.6635481715202332}`
- **Metadata**: `{"beam_width": 6, "accuracy": 0.75, "beam_penalty": 0.0}`

### beam_search_refine_d1 (score=11.38)
- **Config**: `{"beam_width": 3, "length_penalty": 2.0, "early_stopping": true, "diversity_penalty": 1.0}`
- **Metadata**: `{"beam_width": 3, "accuracy": 0.6}`

### bitnet_config (score=-27.37)
- **Config**: `{"learned_scale": false, "quant_mode": "ternary", "init_scale": 1.5358287990093231}`
- **Metadata**: `{"mode": "ternary", "err": 0.4337293367049065}`

### bitnet_config_refine_d1 (score=-27.40)
- **Config**: `{"learned_scale": false, "quant_mode": "ternary", "init_scale": 1.5554749071598053}`
- **Metadata**: `{"mode": "ternary", "err": 0.4340482568961554}`

### checkpoint_recompute (score=8.94)
- **Config**: `{"n_checkpoint_layers": 16, "recompute_strategy": "selective", "block_size": 512}`
- **Metadata**: `{"mem_saved_gb": 0.40265315771102905, "recompute_overhead_ms": 3.0000001192092896}`

### checkpoint_recompute_refine_d1 (score=8.34)
- **Config**: `{"n_checkpoint_layers": 16, "recompute_strategy": "selective", "block_size": 128}`
- **Metadata**: `{"mem_saved_gb": 0.5368708968162537, "recompute_overhead_ms": 1.6000000635782878}`

### conv_config (score=4.79)
- **Config**: `{"kernel_size": 3, "stride": 1, "dilation": 4, "groups": 7, "n_conv_layers": 5}`
- **Metadata**: `{"receptive_field": 60, "params": 421302.85714285716}`

### cpu_adamw_config (score=30.00)
- **Config**: `{"offload_ratio": 3.946805104959594e-10, "prefetch_depth": 3, "compression": "int4", "update_freq": 15}`
- **Metadata**: `{"throughput": 2999.9999994079794, "latency": 3.2890042541329954e-08}`

### cpu_adamw_config_refine_d1 (score=27.85)
- **Config**: `{"offload_ratio": 0.0758352056145668, "prefetch_depth": 2, "compression": "int4", "update_freq": 16}`
- **Metadata**: `{"throughput": 2886.24719157815, "latency": 10.111360748608906}`

### cpu_kv_offload (score=0.04)
- **Config**: `{"offload_layers": 1, "offload_threshold": 0.9999639987945557, "prefetch_size": 2048, "async_copy": true}`
- **Metadata**: `{"kv_freed_gb": 0.008388306014239788, "decode_lat_ms": 0.0}`

### cpu_kv_offload_refine_d1 (score=-0.86)
- **Config**: `{"offload_layers": 4, "offload_threshold": 0.8999639749526978, "prefetch_size": 2048, "async_copy": true}`
- **Metadata**: `{"kv_freed_gb": 0.030197778716683388, "decode_lat_ms": 0.8428571309362138}`

### cross_layer_kv (score=-255.54)
- **Config**: `{"share_ratio": 1.0, "n_share_groups": 8, "share_mode": "avg"}`
- **Metadata**: `{"share_ratio": 1.0, "n_share_groups": 8, "share_mode": "avg", "param_reduction": 1.0, "recon_err": 0.7070745936959677}`

### cross_layer_kv_refine_d1 (score=-311.76)
- **Config**: `{"share_ratio": 0.936205267906189, "n_share_groups": 7, "share_mode": "avg"}`
- **Metadata**: `{"share_ratio": 0.936205267906189, "n_share_groups": 7, "share_mode": "avg", "param_reduction": 0.4375, "recon_err": 0.7070179362256279}`

### csa_attention (score=14.81)
- **Config**: `{"top_k": 64, "pattern_type": "csa_hca_hybrid", "block_size": 8}`
- **Metadata**: `{"top_k": 64, "block_size": 8}`

### diff_attn (score=10.00)
- **Config**: `{"lambda_init": 6.861171897298846e-08, "n_heads": 31, "softmax_sep": 0.008012239821255207}`
- **Metadata**: `{"lambda": 6.861171897298846e-08, "n_heads": 31}`

### diff_attn_refine_d1 (score=10.00)
- **Config**: `{"lambda_init": 0.0, "n_heads": 28, "softmax_sep": 0.0}`
- **Metadata**: `{"lambda": 0.0, "n_heads": 28}`

### expert_hotload (score=8.83)
- **Config**: `{"n_hot_experts": 4, "prefetch_ahead": 4, "cache_strategy": "lfu", "disk_cache_size": 4096}`
- **Metadata**: `{"vram_used_gb": 1.025390625, "miss_rate": 0.0}`

### expert_hotload_refine_d1 (score=8.27)
- **Config**: `{"n_hot_experts": 3, "prefetch_ahead": 4, "cache_strategy": "priority", "disk_cache_size": 4096}`
- **Metadata**: `{"vram_used_gb": 0.439453125, "miss_rate": 0.08500000089406967}`

### factorized_embed (score=18.95)
- **Config**: `{"rank": 72, "init_mode": "svd", "tie_factor": 0.02759479358792305, "vocab_size": 39902}`
- **Metadata**: `{"rank": 72, "reduction": 0.963064225479153, "err": 0.0031585302153561076, "tie_factor": 0.02759479358792305}`

### factorized_embed_refine_d1 (score=19.32)
- **Config**: `{"rank": 64, "init_mode": "svd", "tie_factor": 1.0, "vocab_size": 65536}`
- **Metadata**: `{"rank": 64, "reduction": 0.9677734375, "err": 0.00039168762300829425}`

### ffn_skip (score=-6.20)
- **Config**: `{"skip_threshold": 0.008037981577217579, "n_eval_layers": 13, "skip_strategy": "cosine", "min_keep": 0.5421887002885342}`
- **Metadata**: `{"compute_saved": 0.0625, "deviation": 0.24823034081322}`

### ffn_skip_refine_d1 (score=-6.20)
- **Config**: `{"skip_threshold": 0.0, "n_eval_layers": 4, "skip_strategy": "cosine", "min_keep": 0.945724755525589}`
- **Metadata**: `{"compute_saved": 0.0625, "deviation": 0.2483728633029396}`

### fp8_training_config (score=21.53)
- **Config**: `{"autocast_mode": "e5m2", "smooth_swiglu": false, "mu_scaling": true, "loss_scale": 3992.9418869018555}`
- **Metadata**: `{"overflow": 0.0, "mode": "e5m2", "quant_err": 5.53240909837481e-06, "mantissa_bits": 2}`

### fp8_training_config_refine_d1 (score=38.04)
- **Config**: `{"autocast_mode": "e4m3", "smooth_swiglu": false, "mu_scaling": true, "loss_scale": 3542.5012817382812}`
- **Metadata**: `{"overflow": 0.0, "mode": "e4m3"}`

### gla_attention (score=-50.00)
- **Config**: `{"latent_dim": 463, "n_heads": 28, "compression_ratio": 2.9543026983737946}`
- **Metadata**: `{"latent_dim": 463, "n_heads": 28, "recon_err": 1.5266957044037888}`

### grad_accum_config (score=11.10)
- **Config**: `{"accum_steps": 5, "micro_batch": 15, "grad_clip": 0.9999963045120239, "sync_freq": 15}`
- **Metadata**: `{"effective_batch": 75, "noise": 0.11547005383792514}`

### grad_accum_config_refine_d1 (score=11.14)
- **Config**: `{"accum_steps": 5, "micro_batch": 16, "grad_clip": 1.0, "sync_freq": 14}`
- **Metadata**: `{"effective_batch": 80, "noise": 0.11180339887498948}`

### group_quant (score=9.11)
- **Config**: `{"group_size": 16, "n_bits": 8, "scheme": "asymmetric"}`
- **Metadata**: `{"group_size": 16, "bits": 8, "err": 0.008853267317228148}`

### group_quant_refine_d1 (score=9.11)
- **Config**: `{"group_size": 16, "n_bits": 8, "scheme": "asymmetric"}`
- **Metadata**: `{"group_size": 16, "bits": 8, "err": 0.008853267317228148}`

### gta_attention (score=9.18)
- **Config**: `{"v_k_mix": 0.0033624120987951756, "n_kv_heads": 4, "tie_strength": 0.533734142780304}`
- **Metadata**: `{"mix": 0.0033624120987951756, "n_kv_heads": 4, "tie_strength": 0.533734142780304, "deviation": 0.0025244881816990283, "kv_reduction": 0.875}`

### gta_attention_refine_d1 (score=9.32)
- **Config**: `{"v_k_mix": 0.0, "n_kv_heads": 4, "tie_strength": 0.6292574405670166}`
- **Metadata**: `{"mix": 0.0, "n_kv_heads": 4, "tie_strength": 0.6292574405670166, "deviation": 0.0, "kv_reduction": 0.875}`

### hadamard_kv_refine_d1 (score=19.43)
- **Config**: `{"hadamard_dim": 64, "n_apply": 2, "quant_bits": 4}`
- **Metadata**: `{"hadamard_dim": 64, "n_apply": 2, "quant_bits": 4, "err_orig": 0.1958954781293869, "err_rot": 0.19583438336849213}`

### hqe_kv (score=4.46)
- **Config**: `{"budget_full": 0.05, "budget_int8": 0.5, "budget_int4": 0.5, "group_size": 32, "recency_decay": 1.0}`
- **Metadata**: `{"fwd_err": 0.27108611182414827, "compression": 3.079120466077805, "quant_ms": 0.9237361398233415, "n_full": 409, "n_int8": 409, "n_int4": 2867, "n_evicted": 411, "group_size": 128, "recency_decay": 1.0}`

### hybrid_offload (score=3.85)
- **Config**: `{"offload_ratio": 0.9999901056289673, "prefetch_depth": 5, "pin_memory": true, "overlap_compute": true}`
- **Metadata**: `{"vram_saved_gb": 2.3399767875671387, "latency_ms": 3.654475644839027, "prefetch_mem_cost": 0.25}`

### hybrid_offload_refine_d1 (score=4.19)
- **Config**: `{"offload_ratio": 0.9614534378051758, "prefetch_depth": 8, "pin_memory": true, "overlap_compute": true}`
- **Metadata**: `{"vram_saved_gb": 2.2498011589050293, "latency_ms": 3.2033748626708984}`

### kara (score=1.92)
- **Config**: `{"sink_size": 6, "window_size": 360, "target_budget": 3727, "chunk_expand_size": 5}`
- **Metadata**: `{"fwd_err": 0.030761755900857616, "compression": 1.0007329587099927, "kara_ms": 0.5003664793549963, "sink_size": 6, "window_size": 360, "target_budget": 3727, "chunk_expand_size": 5, "comp_seq_len": 4093}`

### kara_refine_d1 (score=-23.99)
- **Config**: `{"sink_size": 8, "window_size": 256, "target_budget": 512, "chunk_expand_size": 4}`
- **Metadata**: `{"fwd_err": 0.2931080695471184, "compression": 1.0644490644490645, "kara_ms": 0.5322245322245323, "sink_size": 8, "window_size": 256, "target_budget": 512, "chunk_expand_size": 4, "comp_seq_len": 3848}`

### kv_eviction (score=-4.00)
- **Config**: `{"strategy": "streaming", "budget": 362, "observation_window": 52, "n_sinks": 7, "window_size": 1002, "block_size": 10}`
- **Metadata**: `{"fwd_err": 0.09076120875579705, "compression": 1.0148662041625371, "cache_ms": 0.3044598612487611, "strategy": "streaming", "budget": 362, "comp_seq_len": 1009}`

### kv_eviction_refine_d1 (score=-50.00)
- **Config**: `{"strategy": "snapkv", "budget": 512, "observation_window": 512, "n_sinks": 2, "window_size": 128, "block_size": 8}`
- **Metadata**: `{"fwd_err": 0.0, "compression": 1.0, "cache_ms": 0.3, "strategy": "snapkv", "budget": 512, "comp_seq_len": 1024}`

### kv_recompute (score=52.67)
- **Config**: `{"recompute_layers": 16, "recompute_strategy": "selective", "threshold": 0.9}`
- **Metadata**: `{"recompute_layers": 16, "strategy": "selective", "threshold": 0.9, "n_actual_recomp": 15, "quality": 0.99, "inference_penalty": 21.875}`

### kv_recompute_refine_d1 (score=70.70)
- **Config**: `{"recompute_layers": 14, "recompute_strategy": "selective", "threshold": 0.925775945186615}`
- **Metadata**: `{"recompute_layers": 14, "strategy": "selective", "threshold": 0.925775945186615, "n_actual_recomp": 14, "quality": 0.98}`

### kvzip_kv (score=28.24)
- **Config**: `{"compression_ratio": 2, "codebook_size": 256, "n_iter": 10}`
- **Metadata**: `{"compression_ratio": 2, "codebook_size": 256, "n_iter": 10, "recon_err": 0.0, "actual_comp": 3.5555555555555554}`

### kvzip_kv_refine_d1 (score=13.18)
- **Config**: `{"compression_ratio": 3, "codebook_size": 255, "n_iter": 10}`
- **Metadata**: `{"compression_ratio": 3, "codebook_size": 255, "n_iter": 10, "recon_err": 0.05055888369679451, "actual_comp": 3.56794425087108}`

### local_global (score=14.04)
- **Config**: `{"local_window": 1754, "global_ratio": 0.9972541928291321, "n_global_heads": 15}`
- **Metadata**: `{"local_window": 1754, "global_ratio": 0.9972541928291321, "n_global_heads": 15}`

### local_global_refine_d1 (score=14.06)
- **Config**: `{"local_window": 1755, "global_ratio": 0.9993057250976562, "n_global_heads": 15}`
- **Metadata**: `{"local_window": 1755, "global_ratio": 0.9993057250976562, "n_global_heads": 15}`

### loss_config (score=4.40)
- **Config**: `{"loss_type": "kl", "label_smoothing": 0.07084664404392242, "focal_gamma": 2.189527302980423, "temperature": 1.496375560760498}`
- **Metadata**: `{"loss": 5.356783866882324, "type": "kl", "focus_ratio": 1.0, "smoothing_penalty": 0.0}`

### loss_config_refine_d1 (score=2.04)
- **Config**: `{"loss_type": "focal", "label_smoothing": 0.229548704624176, "focal_gamma": 4.815880358219147, "temperature": 1.9688957631587982}`
- **Metadata**: `{"loss": 4.416812896728516, "type": "focal"}`

### memory_budget (score=0.61)
- **Config**: `{"kv_budget": 0.42972683906555176, "weight_budget": 0.19435393810272217, "activation_budget": 0.2896943688392639, "reserve": 0.06700361520051956}`
- **Metadata**: `{"utilization": 0.0914919376373291, "oom_risk": 2.863624095916748, "total_frac": 1.2863624095916748}`

### memory_budget_refine_d1 (score=0.63)
- **Config**: `{"kv_budget": 0.47436875104904175, "weight_budget": 0.19535543024539948, "activation_budget": 0.2704744040966034, "reserve": 0.05370505154132843}`
- **Metadata**: `{"utilization": 0.07834988087415695, "oom_risk": 0.0, "total_frac": 0.993903636932373}`

### mhc_config (score=10.45)
- **Config**: `{"rank": 64, "gate_init": 0.7297391891479492, "n_connections": 1}`
- **Metadata**: `{"rank": 114, "n_connections": 2}`

### mhc_config_refine_d1 (score=10.44)
- **Config**: `{"rank": 75, "gate_init": 1.0, "n_connections": 1}`
- **Metadata**: `{"rank": 75, "n_connections": 1}`

### mixed_precision (score=-16.37)
- **Config**: `{"n_levels": 2, "assignment": "uniform", "bits_base": 6}`
- **Metadata**: `{"n_levels": 2, "avg_bits": 7.0}`

### mod_config (score=9.98)
- **Config**: `{"keep_fraction": 0.5000658252392896, "router_type": "mlp", "aux_loss_weight": 0.0003585296915844083, "n_skip_layers": 0}`
- **Metadata**: `{"keep_fraction": 0.5000658252392896, "compute_saved": 0.49993417476071045, "router_type": "mlp", "n_skip_layers": 0, "skip_penalty": 0.0, "aux_penalty": 0.0}`

### mod_config_refine_d1 (score=10.08)
- **Config**: `{"keep_fraction": 0.5, "router_type": "mlp", "aux_loss_weight": 0.03231801092624664, "n_skip_layers": 2}`
- **Metadata**: `{"keep_fraction": 0.5, "compute_saved": 0.5}`

### moe_routing (score=26.79)
- **Config**: `{"n_experts": 4, "top_k": 3, "router_mode": "switch", "load_balance_weight": 0.004044871404767037, "shared_expert": true}`
- **Metadata**: `{"n_experts": 4, "balance": 0.9914948269733705, "util": 1.0, "diversity_penalty": 0.0}`

### moe_routing_refine_d1 (score=26.87)
- **Config**: `{"n_experts": 4, "top_k": 3, "router_mode": "aux_free", "load_balance_weight": 0.0, "shared_expert": true}`
- **Metadata**: `{"n_experts": 4, "balance": 0.9914948269733705, "util": 1.0, "diversity_penalty": 0.0}`

### mosaic_quant (score=4.21)
- **Config**: `{"n_tiles": 30, "tile_dim": 409, "mix_ratio": 0.0006063980981707573}`
- **Metadata**: `{"n_tiles": 7, "err": 0.15635676681995392}`

### mosaic_quant_refine_d1 (score=4.20)
- **Config**: `{"n_tiles": 29, "tile_dim": 102, "mix_ratio": 0.0}`
- **Metadata**: `{"n_tiles": 29, "err": 0.007998540997505188}`

### mtp_config (score=25.71)
- **Config**: `{"n_heads": 4, "loss_weight": 0.4953504800796509, "share_weights": true, "depth_ratio": 0.9856214821338654}`
- **Metadata**: `{"n_heads": 4, "pred_acc": 0.6160134263336658, "lw_penalty": 0.0, "dr_latency": 1.9712429642677307}`

### mtp_config_refine_d1 (score=27.50)
- **Config**: `{"n_heads": 4, "loss_weight": 0.5, "share_weights": true, "depth_ratio": 0.9599757194519043}`
- **Metadata**: `{"n_heads": 4, "pred_acc": 0.5999848246574402}`

### muon_config (score=11.90)
- **Config**: `{"momentum": 0.9844147086143493, "nesterov": true, "weight_decay": 0.005212975144386292, "ns_steps": 1}`
- **Metadata**: `{"final_loss": 9.498847961425781, "ns_steps": 4}`

### nvfp4_quant (score=14.32)
- **Config**: `{"block_size": 16, "w4a8": true, "scale_mode": "per_channel"}`
- **Metadata**: `{"block_size": 16, "err": 0.09676586664985923}`

### offq_quant (score=-16.49)
- **Config**: `{"offset_init": 0.05462278425693512, "n_iter": 40, "learn_offset": false}`
- **Metadata**: `{"offset": -0.0032030893489718437, "err": 0.18585337325655701}`

### optimizer_config (score=24.52)
- **Config**: `{"opt_type": "adamw", "lr": 0.00994758250117302, "beta1": 0.8029370873235167, "beta2": 0.9989974006414414, "weight_decay": 0.0002499049296602607}`
- **Metadata**: `{"final_loss": 0.3056432604789734, "opt": "adamw"}`

### paged_evict_kv (score=128.11)
- **Config**: `{"page_size": 64, "n_pages": 64, "eviction_policy": "lru"}`
- **Metadata**: `{"page_size": "128", "n_pages": "128", "eviction_policy": "lfu", "capacity": "16384", "hit_rate": 1.0, "mem_eff": 0.25}`

### paged_evict_kv_refine_d1 (score=128.17)
- **Config**: `{"page_size": 41, "n_pages": 100, "eviction_policy": "lru"}`
- **Metadata**: `{"page_size": 41, "n_pages": 100, "eviction_policy": "lru", "capacity": 4100, "hit_rate": 1.0, "mem_eff": 0.9990243902439024}`

### qk_norm (score=9.65)
- **Config**: `{"norm_type": "layernorm", "epsilon": 6.5553047601133585e-06, "scale_init": 0.5006855120736873}`
- **Metadata**: `{"epsilon": 0.000757039385020733, "scale": 1.8720133006572723}`

### quant (score=-2.62)
- **Config**: `{"block_size": 16, "scale_method": "absmax", "residual_ratio": 0.0, "global_scale_factor": 1.0, "scale_search_range": 0.3, "scale_search_steps": 5, "rounding_method": "rtn", "use_hadamard": false, "hadamard_dim": 16, "scale_clip_min": 0.01}`
- **Metadata**: `{"frob_err": 0.09548715419543356, "fwd_err": 0.09629629359757037, "compression": 3.506849315068493, "dequant_ms": 0.073728, "q_bytes": 74752, "block_size": 16, "scale_method": "absmax", "rounding_method": "rtn", "use_hadamard": false, "hadamard_dim": 16, "scale_clip_min": 0.01, "scale_search_range": 0.3, "scale_search_steps": 5}`

### rope_config (score=34.64)
- **Config**: `{"theta": 8939256.656765938, "scaling_type": "linear", "scaling_factor": 0.546415823046118}`
- **Metadata**: `{"theta": 8939256.656765938, "scaling": 0.546415823046118, "frozen_frac": 0.125, "compat_penalty": 0.0}`

### rope_config_refine_d1 (score=34.44)
- **Config**: `{"theta": 10000000.0, "scaling_type": "linear", "scaling_factor": 0.578122635371983}`
- **Metadata**: `{"theta": 10000000.0, "scaling": 0.578122635371983, "frozen_frac": 0.15625, "compat_penalty": 0.0}`

### rotor_quant_kv (score=2.55)
- **Config**: `{"rot_type": "random", "n_rotations": 1, "quant_bits": 4}`
- **Metadata**: `{"rot_type": "dct", "n_rotations": 7, "quant_bits": 4, "compute": 17.5}`

### sampling_config (score=14.38)
- **Config**: `{"temperature": 0.2630424790084362, "top_p": 0.5126710869371891, "top_k": 70, "repetition_penalty": 1.266894280910492, "frequency_penalty": 0.8213245868682861}`
- **Metadata**: `{"temp": 0.2630424790084362, "diversity": -0.0, "rp_benefit": 2.1351542472839355, "fp_benefit": 2.4639737606048584}`

### sampling_config_refine_d1 (score=10.62)
- **Config**: `{"temperature": 1.7968702256679534, "top_p": 1.0, "top_k": 100, "repetition_penalty": 1.0, "frequency_penalty": 0.0}`
- **Metadata**: `{"temp": 1.7968702256679534, "diversity": 0.3820650366556218}`

### scheduler_config (score=10.00)
- **Config**: `{"sched_type": "cosine", "warmup_steps": 75, "min_lr_ratio": 0.003999027889221907, "decay_steps": 5156}`
- **Metadata**: `{"sched_type": "cosine", "auc": 0.5019700408918977, "stability_penalty": 0.0}`

### scheduler_config_refine_d1 (score=9.97)
- **Config**: `{"sched_type": "cosine", "warmup_steps": 15, "min_lr_ratio": 0.40436118841171265, "decay_steps": 930}`
- **Metadata**: `{"sched_type": "cosine", "auc": 0.698702218391562, "stability_penalty": 0.0}`

### sharq_quant (score=-18.13)
- **Config**: `{"n_levels": 30, "adaptive": false, "warmup_steps": 113}`
- **Metadata**: `{"n_levels": 30, "bits": 4.906890595608519}`

### sliding_window (score=9.65)
- **Config**: `{"window_size": 287, "stride": 78, "overlap_ratio": 0.8495826125144958}`
- **Metadata**: `{"window_size": 3227, "stride": 1219}`

### sliding_window_refine_d1 (score=8.60)
- **Config**: `{"window_size": 818, "stride": 140, "overlap_ratio": 0.0}`
- **Metadata**: `{"window_size": 818, "stride": 140}`

### sparse_attn (score=-25.38)
- **Config**: `{"strategy": "compact", "budget_ratio": 0.8477128744125366, "block_size": 17, "min_seq_len": 360, "k_ratio": 0.4541882276535034}`
- **Metadata**: `{"fwd_err": 0.3621227845867708, "speedup": 0.8477128744125367, "sparse_ms": 0.2781602203888968, "full_ms": 0.23579999997309642, "strategy": "compact", "activated": true, "budget_ratio": 0.8477128744125366, "block_size": 17, "k_ratio": 0.4541882276535034}`

### speculative_decode (score=41.66)
- **Config**: `{"n_draft_tokens": 7, "draft_model_ratio": 0.10759469568729402, "acceptance_threshold": 0.5017799103399738, "temperature": 0.45239147543907166}`
- **Metadata**: `{"n_draft": 7, "acceptance": 0.46170485255168175, "speedup": 3.9355248489192936, "temp": 0.45239147543907166, "temp_penalty": 0.0}`

### speculative_decode_refine_d1 (score=57.05)
- **Config**: `{"n_draft_tokens": 7, "draft_model_ratio": 0.1, "acceptance_threshold": 0.95, "temperature": 1.0}`
- **Metadata**: `{"n_draft": 7, "acceptance": 0.6773871772109372, "speedup": 5.366084336893982}`

### streaming_kv (score=96.85)
- **Config**: `{"chunk_size": 128, "n_sink": 4, "overlap": 0.1}`
- **Metadata**: `{"chunk_size": 128, "n_sink": 4, "overlap": 0.1, "n_chunks": 35, "coverage_err": 0.006494045257568359, "mem_with_overlap": 0.0353515625}`

### streaming_kv_refine_d1 (score=96.98)
- **Config**: `{"chunk_size": 128, "n_sink": 5, "overlap": 0.059021227061748505}`
- **Metadata**: `{"chunk_size": 128, "n_sink": 5, "overlap": 0.059021227061748505, "n_chunks": 34, "coverage_err": 0.005495965480804443, "mem_with_overlap": 0.03431511647067964}`

### synthetic (score=21.44)
- **Config**: `{"x": [0.21149232983589172, 0.9307464361190796, 0.9266494512557983, 0.9777611494064331, 0.35501474142074585, 0.764074444770813, 0.08462908864021301, 0.8343579173088074]}`
- **Metadata**: `{"rastrigin": 33.82922982748529, "deceptive": -5.272094562372881}`

### synthetic_refine_d1 (score=25.93)
- **Config**: `{"x": [0.12018084526062012, 0.9770234823226929, 0.548395574092865, 0.9852800369262695, 0.1675012856721878, 0.3275863230228424, 0.2662193179130554, 0.8825550675392151]}`
- **Metadata**: `{"rastrigin": 30.791598490892063, "deceptive": -6.725417026529974}`

### titan_memory (score=5.07)
- **Config**: `{"memory_rank": 64, "gate_init": 0.4873531460762024, "n_memory_slots": 1, "update_freq": 1}`
- **Metadata**: `{"rank": 64, "capacity": 0.01756790124675455, "gate": 0.619482696056366, "freshness": 1.0, "gate_interference": 1.1948269605636597}`

### titan_memory_refine_d1 (score=3.40)
- **Config**: `{"memory_rank": 64, "gate_init": 0.5, "n_memory_slots": 1, "update_freq": 1}`
- **Metadata**: `{"rank": 64, "capacity": 0.024540826848110993, "gate": 0.622459352016449}`

### w8a8_quant (score=179.47)
- **Config**: `{"mode": "fp8", "calib_samples": 65, "per_channel": false, "smoothquant_alpha": 0.9990673661231995}`
- **Metadata**: `{"mode": "fp8", "sqnr": 86.73324584960938}`

### w8a8_quant_refine_d1 (score=178.75)
- **Config**: `{"mode": "int8", "calib_samples": 948, "per_channel": true, "smoothquant_alpha": 1.0}`
- **Metadata**: `{"mode": "int8", "sqnr": 86.37299346923828}`

### xquant_kv (score=81.31)
- **Config**: `{"recomputation_ratio": 0.5119810104370117, "quant_bits": 8, "checkpoint_interval": 12}`
- **Metadata**: `{"recomputation_ratio": 0.5119810104370117, "quant_bits": 8, "checkpoint_interval": 12, "n_recompute": 8, "mem_ratio": 0.125, "quant_err": 0.01069291215389967, "inference_penalty": 0.7188606262207031}`

### xquant_kv_refine_d1 (score=77.86)
- **Config**: `{"recomputation_ratio": 0.5, "quant_bits": 8, "checkpoint_interval": 16}`
- **Metadata**: `{"recomputation_ratio": 0.5, "quant_bits": 8, "checkpoint_interval": 16, "n_recompute": 8, "mem_ratio": 0.125, "quant_err": 0.01069291215389967, "inference_penalty": 5.0}`

## Tier 3: Cross-Domain Parameter Patterns

### `acceptance_threshold`
- `0.95`: 2x, avg=55.2, domains=speculative_decode_refine_d1
- `0.5017799103399738`: 1x, avg=41.7, domains=speculative_decode
- `0.5056282434612512`: 1x, avg=40.4, domains=speculative_decode
- `0.5139564563520252`: 1x, avg=40.2, domains=speculative_decode
- `0.9337183475494384`: 1x, avg=51.4, domains=speculative_decode_refine_d1
### `accum_steps`
- `5`: 4x, avg=11.1, domains=grad_accum_config,grad_accum_config_refine_d1
- `4`: 1x, avg=11.1, domains=grad_accum_config_refine_d1
- `7`: 1x, avg=11.1, domains=grad_accum_config_refine_d1
### `activation_budget`
- `0.2896943688392639`: 1x, avg=0.6, domains=memory_budget
- `0.10133626312017441`: 1x, avg=0.6, domains=memory_budget
- `0.2043609321117401`: 1x, avg=0.6, domains=memory_budget
- `0.2704744040966034`: 1x, avg=0.6, domains=memory_budget_refine_d1
- `0.27027013897895813`: 1x, avg=0.6, domains=memory_budget_refine_d1
### `adaptive`
- `False`: 3x, avg=-18.2, domains=sharq_quant
### `assignment`
- `uniform`: 1x, avg=-16.4, domains=mixed_precision
- `importance`: 1x, avg=-16.4, domains=mixed_precision
- `random`: 1x, avg=-16.4, domains=mixed_precision
### `async_copy`
- `True`: 6x, avg=-0.5, domains=cpu_kv_offload,cpu_kv_offload_refine_d1
### `autocast_mode`
- `e5m2`: 3x, avg=21.5, domains=fp8_training_config
- `e4m3`: 3x, avg=37.1, domains=fp8_training_config_refine_d1
### `aux_loss_weight`
- `0.0003585296915844083`: 1x, avg=10.0, domains=mod_config
- `0.0004208577796816826`: 1x, avg=10.0, domains=mod_config
- `0.012472297996282578`: 1x, avg=9.8, domains=mod_config
- `0.03231801092624664`: 1x, avg=10.1, domains=mod_config_refine_d1
- `0.0`: 1x, avg=10.1, domains=mod_config_refine_d1
### `beam_width`
- `6`: 2x, avg=10.7, domains=beam_search
- `3`: 2x, avg=11.3, domains=beam_search_refine_d1
- `7`: 1x, avg=10.0, domains=beam_search
### `beta1`
- `0.8029370873235167`: 1x, avg=24.5, domains=optimizer_config
- `0.8013877812307328`: 1x, avg=24.4, domains=optimizer_config
- `0.8026680081151426`: 1x, avg=24.4, domains=optimizer_config
### `beta2`
- `0.9989974006414414`: 1x, avg=24.5, domains=optimizer_config
- `0.9989972078800201`: 1x, avg=24.4, domains=optimizer_config
- `0.9989998072385788`: 1x, avg=24.4, domains=optimizer_config
### `bits_base`
- `6`: 3x, avg=-16.4, domains=mixed_precision
### `block_size`
- `16`: 5x, avg=3.3, domains=nvfp4_quant,quant
- `8`: 4x, avg=-110.5, domains=csa_attention,kv_eviction_refine_d1
- `512`: 2x, avg=8.4, domains=checkpoint_recompute,checkpoint_recompute_refine_d1
- `128`: 2x, avg=8.3, domains=checkpoint_recompute,checkpoint_recompute_refine_d1
- `256`: 1x, avg=8.7, domains=checkpoint_recompute
### `budget`
- `128`: 2x, avg=-203.3, domains=kv_eviction_refine_d1
- `362`: 1x, avg=-4.0, domains=kv_eviction
- `458`: 1x, avg=-5.4, domains=kv_eviction
- `1964`: 1x, avg=-6.7, domains=kv_eviction
- `512`: 1x, avg=-50.0, domains=kv_eviction_refine_d1
### `budget_full`
- `0.05`: 3x, avg=4.4, domains=hqe_kv
### `budget_int4`
- `0.5`: 3x, avg=4.4, domains=hqe_kv
### `budget_int8`
- `0.5`: 3x, avg=4.4, domains=hqe_kv
### `budget_ratio`
- `0.8477128744125366`: 1x, avg=-25.4, domains=sparse_attn
- `0.8457093834877014`: 1x, avg=-25.8, domains=sparse_attn
- `0.8464348912239075`: 1x, avg=-26.5, domains=sparse_attn
### `cache_strategy`
- `lfu`: 2x, avg=8.5, domains=expert_hotload,expert_hotload_refine_d1
- `priority`: 2x, avg=8.4, domains=expert_hotload,expert_hotload_refine_d1
- `lru`: 2x, avg=8.3, domains=expert_hotload,expert_hotload_refine_d1
### `calib_method`
- `percentile`: 3x, avg=9.6, domains=activation_quant
### `calib_samples`
- `64`: 2x, avg=178.3, domains=w8a8_quant_refine_d1
- `65`: 1x, avg=179.5, domains=w8a8_quant
- `938`: 1x, avg=179.4, domains=w8a8_quant
- `73`: 1x, avg=178.9, domains=w8a8_quant
- `948`: 1x, avg=178.7, domains=w8a8_quant_refine_d1
### `checkpoint_interval`
- `12`: 2x, avg=80.3, domains=xquant_kv
- `16`: 2x, avg=76.6, domains=xquant_kv_refine_d1
- `14`: 1x, avg=80.1, domains=xquant_kv
- `8`: 1x, avg=75.4, domains=xquant_kv_refine_d1
### `chunk_expand_size`
- `4`: 3x, avg=-24.2, domains=kara_refine_d1
- `5`: 2x, avg=1.7, domains=kara
- `11`: 1x, avg=1.7, domains=kara
### `chunk_size`
- `128`: 6x, avg=96.7, domains=streaming_kv,streaming_kv_refine_d1
### `codebook_size`
- `255`: 2x, avg=12.4, domains=kvzip_kv,kvzip_kv_refine_d1
- `401`: 1x, avg=10.0, domains=aaac_quant
- `495`: 1x, avg=10.0, domains=aaac_quant
- `415`: 1x, avg=10.0, domains=aaac_quant
- `256`: 1x, avg=28.2, domains=kvzip_kv
### `compression`
- `int4`: 6x, avg=28.9, domains=cpu_adamw_config,cpu_adamw_config_refine_d1
### `compression_ratio`
- `2`: 3x, avg=2.8, domains=kvzip_kv,kvzip_kv_refine_d1
- `2.9543026983737946`: 1x, avg=-50.0, domains=gla_attention
- `3.055733799934387`: 1x, avg=-50.0, domains=gla_attention
- `1.509044997394085`: 1x, avg=-89.3, domains=gla_attention
- `5`: 1x, avg=11.6, domains=kvzip_kv
### `decay_steps`
- `5156`: 1x, avg=10.0, domains=scheduler_config
- `8322`: 1x, avg=10.0, domains=scheduler_config
- `2494`: 1x, avg=10.0, domains=scheduler_config
- `930`: 1x, avg=10.0, domains=scheduler_config_refine_d1
- `663`: 1x, avg=10.0, domains=scheduler_config_refine_d1
### `depth_ratio`
- `0.9856214821338654`: 1x, avg=25.7, domains=mtp_config
- `0.9999892115592957`: 1x, avg=22.1, domains=mtp_config
- `0.9999945759773254`: 1x, avg=22.1, domains=mtp_config
- `0.9599757194519043`: 1x, avg=27.5, domains=mtp_config_refine_d1
- `0.9599348604679108`: 1x, avg=27.5, domains=mtp_config_refine_d1
### `dilation`
- `3`: 2x, avg=3.8, domains=conv_config
- `4`: 1x, avg=4.8, domains=conv_config
### `disk_cache_size`
- `4096`: 6x, avg=8.4, domains=expert_hotload,expert_hotload_refine_d1
### `diversity_penalty`
- `0.6635481715202332`: 1x, avg=10.9, domains=beam_search
- `0.7342454195022583`: 1x, avg=10.4, domains=beam_search
- `0.7677754759788513`: 1x, avg=10.0, domains=beam_search
- `1.0`: 1x, avg=11.4, domains=beam_search_refine_d1
- `0.9629197716712952`: 1x, avg=11.3, domains=beam_search_refine_d1
### `draft_model_ratio`
- `0.1`: 2x, avg=55.2, domains=speculative_decode_refine_d1
- `0.10759469568729402`: 1x, avg=41.7, domains=speculative_decode
- `0.10669493973255158`: 1x, avg=40.4, domains=speculative_decode
- `0.10105040157213807`: 1x, avg=40.2, domains=speculative_decode
- `0.10552149713039399`: 1x, avg=51.4, domains=speculative_decode_refine_d1
### `early_stopping`
- `True`: 5x, avg=10.8, domains=beam_search,beam_search_refine_d1
### `enabled`
- `1.0`: 24x, avg=65.0, domains=Generic_use_adaptive_spec_refine_d1,Generic_use_chunked_prefill_refine_d1,Generic_use_compact_attn_refine_d1,Generic_use_compile_autotune_refine_d1,Generic_use_compile_refine_d1
- `0.0`: 19x, avg=65.0, domains=Generic_use_block_fusion_refine_d1,Generic_use_corun_refine_d1,Generic_use_cosa_refine_d1,Generic_use_fused_qk_norm_rope_cache_refine_d1,Generic_use_hotprefix_refine_d1
- `0.9984026551246643`: 1x, avg=64.9, domains=Generic_use_adaptive_spec
- `0.998700737953186`: 1x, avg=64.7, domains=Generic_use_adaptive_spec
- `0.9970566034317017`: 1x, avg=64.6, domains=Generic_use_adaptive_spec
### `epsilon`
- `6.5553047601133585e-06`: 1x, avg=9.7, domains=qk_norm
- `0.0009342671583890916`: 1x, avg=9.7, domains=qk_norm
- `1.0069217847330946e-06`: 1x, avg=9.7, domains=qk_norm
### `eviction_policy`
- `lru`: 6x, avg=126.9, domains=paged_evict_kv,paged_evict_kv_refine_d1
### `focal_gamma`
- `2.189527302980423`: 1x, avg=4.4, domains=loss_config
- `0.1541236974298954`: 1x, avg=4.4, domains=loss_config
- `2.0406222343444824`: 1x, avg=4.4, domains=loss_config
- `4.815880358219147`: 1x, avg=2.0, domains=loss_config_refine_d1
- `4.356221556663513`: 1x, avg=2.0, domains=loss_config_refine_d1
### `frequency_penalty`
- `0.0`: 3x, avg=10.6, domains=sampling_config_refine_d1
- `0.8213245868682861`: 1x, avg=14.4, domains=sampling_config
- `0.8024860620498657`: 1x, avg=13.8, domains=sampling_config
- `0.44739624857902527`: 1x, avg=13.5, domains=sampling_config
### `gate_init`
- `1.0`: 3x, avg=5.6, domains=attn_residual_refine_d1,mhc_config_refine_d1
- `0.5`: 2x, avg=3.4, domains=titan_memory_refine_d1
- `0.9947109222412109`: 1x, avg=3.3, domains=attn_residual
- `0.9991616010665894`: 1x, avg=3.2, domains=attn_residual
- `0.9931772947311401`: 1x, avg=3.2, domains=attn_residual
### `global_ratio`
- `1.0`: 2x, avg=14.1, domains=local_global_refine_d1
- `0.9972541928291321`: 1x, avg=14.0, domains=local_global
- `0.9984179735183716`: 1x, avg=13.9, domains=local_global
- `0.9998893737792969`: 1x, avg=13.9, domains=local_global
- `0.9993057250976562`: 1x, avg=14.1, domains=local_global_refine_d1
### `global_scale_factor`
- `1.0`: 3x, avg=-4.0, domains=quant
### `grad_clip`
- `1.0`: 3x, avg=11.1, domains=grad_accum_config_refine_d1
- `0.9999963045120239`: 1x, avg=11.1, domains=grad_accum_config
- `0.9999942779541016`: 1x, avg=11.1, domains=grad_accum_config
- `0.9999871253967285`: 1x, avg=11.1, domains=grad_accum_config
### `group_size`
- `32`: 4x, avg=5.6, domains=group_quant,hqe_kv
- `16`: 2x, avg=9.1, domains=group_quant,group_quant_refine_d1
- `128`: 2x, avg=9.1, domains=group_quant,group_quant_refine_d1
- `64`: 1x, avg=9.1, domains=group_quant_refine_d1
### `groups`
- `7`: 3x, avg=4.1, domains=conv_config
### `hadamard_dim`
- `16`: 3x, avg=-4.0, domains=quant
- `64`: 2x, avg=19.1, domains=hadamard_kv_refine_d1
### `init_mode`
- `svd`: 6x, avg=18.8, domains=factorized_embed,factorized_embed_refine_d1
### `init_scale`
- `1.5358287990093231`: 1x, avg=-27.4, domains=bitnet_config
- `1.5649056434631348`: 1x, avg=-27.4, domains=bitnet_config
- `1.563204675912857`: 1x, avg=-27.4, domains=bitnet_config
- `1.5554749071598053`: 1x, avg=-27.4, domains=bitnet_config_refine_d1
- `1.5556987822055817`: 1x, avg=-27.4, domains=bitnet_config_refine_d1
### `k_layers`
- `1`: 5x, avg=3.2, domains=attn_residual,attn_residual_refine_d1
- `2`: 1x, avg=3.2, domains=attn_residual_refine_d1
### `k_ratio`
- `0.4541882276535034`: 1x, avg=-25.4, domains=sparse_attn
- `0.6381897330284119`: 1x, avg=-25.8, domains=sparse_attn
- `0.7948082089424133`: 1x, avg=-26.5, domains=sparse_attn
### `keep_fraction`
- `0.5`: 3x, avg=10.1, domains=mod_config_refine_d1
- `0.5000658252392896`: 1x, avg=10.0, domains=mod_config
- `0.5002498800458852`: 1x, avg=10.0, domains=mod_config
- `0.5017301038606092`: 1x, avg=9.8, domains=mod_config
### `kernel_size`
- `3`: 1x, avg=4.8, domains=conv_config
- `5`: 1x, avg=4.2, domains=conv_config
- `7`: 1x, avg=3.4, domains=conv_config
### `kv_budget`
- `0.42972683906555176`: 1x, avg=0.6, domains=memory_budget
- `0.5922980904579163`: 1x, avg=0.6, domains=memory_budget
- `0.5224659442901611`: 1x, avg=0.6, domains=memory_budget
- `0.47436875104904175`: 1x, avg=0.6, domains=memory_budget_refine_d1
- `0.4827900528907776`: 1x, avg=0.6, domains=memory_budget_refine_d1
### `label_smoothing`
- `0.07084664404392242`: 1x, avg=4.4, domains=loss_config
- `0.02527545690536499`: 1x, avg=4.4, domains=loss_config
- `0.18622756004333496`: 1x, avg=4.4, domains=loss_config
- `0.229548704624176`: 1x, avg=2.0, domains=loss_config_refine_d1
- `0.26666311025619505`: 1x, avg=2.0, domains=loss_config_refine_d1
### `lambda_init`
- `0.0`: 3x, avg=10.0, domains=diff_attn_refine_d1
- `6.861171897298846e-08`: 1x, avg=10.0, domains=diff_attn
- `8.094455097307218e-07`: 1x, avg=10.0, domains=diff_attn
- `2.4777873477432877e-06`: 1x, avg=10.0, domains=diff_attn
### `latent_dim`
- `463`: 1x, avg=-50.0, domains=gla_attention
- `366`: 1x, avg=-50.0, domains=gla_attention
- `64`: 1x, avg=-89.3, domains=gla_attention
### `learn_offset`
- `False`: 2x, avg=-16.6, domains=offq_quant
- `True`: 1x, avg=-16.7, domains=offq_quant
### `learned_scale`
- `False`: 5x, avg=-27.4, domains=bitnet_config,bitnet_config_refine_d1
- `True`: 1x, avg=-27.4, domains=bitnet_config
### `length_penalty`
- `2.0`: 2x, avg=11.3, domains=beam_search_refine_d1
- `0.991542786359787`: 1x, avg=10.9, domains=beam_search
- `1.087046280503273`: 1x, avg=10.4, domains=beam_search
- `0.945908933877945`: 1x, avg=10.0, domains=beam_search
### `load_balance_weight`
- `0.0`: 2x, avg=26.1, domains=moe_routing_refine_d1
- `0.004044871404767037`: 1x, avg=26.8, domains=moe_routing
- `0.0035621780902147294`: 1x, avg=26.7, domains=moe_routing
- `0.0020832810550928116`: 1x, avg=26.6, domains=moe_routing
- `0.006238000094890595`: 1x, avg=26.7, domains=moe_routing_refine_d1
### `local_window`
- `1754`: 1x, avg=14.0, domains=local_global
- `1711`: 1x, avg=13.9, domains=local_global
- `1689`: 1x, avg=13.9, domains=local_global
- `1755`: 1x, avg=14.1, domains=local_global_refine_d1
- `1752`: 1x, avg=14.1, domains=local_global_refine_d1
### `loss_scale`
- `3992.9418869018555`: 1x, avg=21.5, domains=fp8_training_config
- `4012.030471801758`: 1x, avg=21.5, domains=fp8_training_config
- `3822.1218872070312`: 1x, avg=21.5, domains=fp8_training_config
- `3542.5012817382812`: 1x, avg=38.0, domains=fp8_training_config_refine_d1
- `3456.267868041992`: 1x, avg=37.4, domains=fp8_training_config_refine_d1
### `loss_type`
- `focal`: 4x, avg=2.6, domains=loss_config,loss_config_refine_d1
- `kl`: 2x, avg=4.4, domains=loss_config
### `loss_weight`
- `0.5`: 3x, avg=27.5, domains=mtp_config_refine_d1
- `0.4953504800796509`: 1x, avg=25.7, domains=mtp_config
- `0.4999894618988038`: 1x, avg=22.1, domains=mtp_config
- `0.4999691009521484`: 1x, avg=22.1, domains=mtp_config
### `lr`
- `0.00994758250117302`: 1x, avg=24.5, domains=optimizer_config
- `0.00999965344786644`: 1x, avg=24.4, domains=optimizer_config
- `0.009891441650986672`: 1x, avg=24.4, domains=optimizer_config
### `max_batch_size`
- `15`: 3x, avg=43.5, domains=batched_decode
- `13`: 3x, avg=42.3, domains=batched_decode_refine_d1
### `max_seq_diff`
- `0`: 3x, avg=43.1, domains=batched_decode,batched_decode_refine_d1
- `2`: 1x, avg=43.3, domains=batched_decode
- `36`: 1x, avg=42.9, domains=batched_decode
- `51`: 1x, avg=41.7, domains=batched_decode_refine_d1
### `memory_rank`
- `64`: 4x, avg=4.2, domains=titan_memory,titan_memory_refine_d1
- `66`: 2x, avg=4.2, domains=titan_memory,titan_memory_refine_d1
### `merge_window_ms`
- `52`: 1x, avg=44.1, domains=batched_decode
- `68`: 1x, avg=43.3, domains=batched_decode
- `62`: 1x, avg=42.9, domains=batched_decode
- `56`: 1x, avg=42.6, domains=batched_decode_refine_d1
- `47`: 1x, avg=42.5, domains=batched_decode_refine_d1
### `micro_batch`
- `15`: 3x, avg=11.1, domains=grad_accum_config
- `16`: 3x, avg=11.1, domains=grad_accum_config_refine_d1
### `min_keep`
- `0.5421887002885342`: 1x, avg=-6.2, domains=ffn_skip
- `0.6254786849021912`: 1x, avg=-6.2, domains=ffn_skip
- `0.9515098631381989`: 1x, avg=-6.2, domains=ffn_skip
- `0.945724755525589`: 1x, avg=-6.2, domains=ffn_skip_refine_d1
- `0.9997501075267792`: 1x, avg=-6.2, domains=ffn_skip_refine_d1
### `min_lr_ratio`
- `0.003999027889221907`: 1x, avg=10.0, domains=scheduler_config
- `0.0015946989879012108`: 1x, avg=10.0, domains=scheduler_config
- `0.02799869142472744`: 1x, avg=10.0, domains=scheduler_config
- `0.40436118841171265`: 1x, avg=10.0, domains=scheduler_config_refine_d1
- `0.40451323986053467`: 1x, avg=10.0, domains=scheduler_config_refine_d1
### `min_seq_len`
- `360`: 1x, avg=-25.4, domains=sparse_attn
- `968`: 1x, avg=-25.8, domains=sparse_attn
- `941`: 1x, avg=-26.5, domains=sparse_attn
### `mix_ratio`
- `0.0`: 3x, avg=4.2, domains=mosaic_quant_refine_d1
- `0.0006063980981707573`: 1x, avg=4.2, domains=mosaic_quant
- `0.0011922086123377085`: 1x, avg=4.2, domains=mosaic_quant
- `0.0014618440764024854`: 1x, avg=4.2, domains=mosaic_quant
### `mode`
- `fp8`: 4x, avg=178.9, domains=w8a8_quant,w8a8_quant_refine_d1
- `int8`: 2x, avg=178.8, domains=w8a8_quant,w8a8_quant_refine_d1
### `momentum`
- `0.9844147086143493`: 1x, avg=11.9, domains=muon_config
- `0.8807766097784042`: 1x, avg=11.9, domains=muon_config
- `0.9386885368824005`: 1x, avg=11.9, domains=muon_config
### `mu_scaling`
- `True`: 6x, avg=29.3, domains=fp8_training_config,fp8_training_config_refine_d1
### `n_bits`
- `8`: 6x, avg=9.1, domains=group_quant,group_quant_refine_d1
- `2`: 3x, avg=10.0, domains=aaac_quant
### `n_checkpoint_layers`
- `16`: 5x, avg=8.4, domains=checkpoint_recompute,checkpoint_recompute_refine_d1
- `14`: 1x, avg=7.8, domains=checkpoint_recompute_refine_d1
### `n_codebooks`
- `4`: 3x, avg=10.0, domains=aaac_quant
### `n_connections`
- `1`: 6x, avg=10.4, domains=mhc_config,mhc_config_refine_d1
### `n_conv_layers`
- `5`: 3x, avg=4.1, domains=conv_config
### `n_draft_tokens`
- `7`: 4x, avg=47.6, domains=speculative_decode,speculative_decode_refine_d1
- `6`: 2x, avg=46.8, domains=speculative_decode,speculative_decode_refine_d1
### `n_eval_layers`
- `14`: 2x, avg=-6.2, domains=ffn_skip_refine_d1
- `13`: 1x, avg=-6.2, domains=ffn_skip
- `9`: 1x, avg=-6.2, domains=ffn_skip
- `3`: 1x, avg=-6.2, domains=ffn_skip
- `4`: 1x, avg=-6.2, domains=ffn_skip_refine_d1
### `n_experts`
- `4`: 4x, avg=26.8, domains=moe_routing,moe_routing_refine_d1
- `5`: 2x, avg=26.0, domains=moe_routing,moe_routing_refine_d1
### `n_global_heads`
- `15`: 3x, avg=14.1, domains=local_global,local_global_refine_d1
- `13`: 2x, avg=13.9, domains=local_global
- `14`: 1x, avg=14.1, domains=local_global_refine_d1
### `n_heads`
- `31`: 4x, avg=-14.8, domains=diff_attn,diff_attn_refine_d1,gla_attention
- `4`: 4x, avg=27.0, domains=mtp_config,mtp_config_refine_d1
- `28`: 2x, avg=-20.0, domains=diff_attn_refine_d1,gla_attention
- `3`: 2x, avg=22.1, domains=mtp_config
- `22`: 1x, avg=10.0, domains=diff_attn
### `n_hot_experts`
- `4`: 2x, avg=8.7, domains=expert_hotload
- `6`: 2x, avg=8.1, domains=expert_hotload_refine_d1
- `5`: 1x, avg=8.5, domains=expert_hotload
- `3`: 1x, avg=8.3, domains=expert_hotload_refine_d1
### `n_iter`
- `10`: 4x, avg=5.4, domains=kvzip_kv,kvzip_kv_refine_d1
- `19`: 2x, avg=-2.5, domains=kvzip_kv,offq_quant
- `12`: 1x, avg=8.7, domains=kvzip_kv
- `40`: 1x, avg=-16.5, domains=offq_quant
- `54`: 1x, avg=-16.7, domains=offq_quant
### `n_kv_heads`
- `4`: 6x, avg=9.2, domains=gta_attention,gta_attention_refine_d1
### `n_levels`
- `2`: 3x, avg=-16.4, domains=mixed_precision
- `30`: 3x, avg=-18.2, domains=sharq_quant
### `n_memory_slots`
- `1`: 6x, avg=4.2, domains=titan_memory,titan_memory_refine_d1
### `n_pages`
- `64`: 1x, avg=128.1, domains=paged_evict_kv
- `128`: 1x, avg=128.1, domains=paged_evict_kv
- `256`: 1x, avg=127.1, domains=paged_evict_kv
- `100`: 1x, avg=128.2, domains=paged_evict_kv_refine_d1
- `85`: 1x, avg=124.9, domains=paged_evict_kv_refine_d1
### `n_rotations`
- `1`: 2x, avg=0.5, domains=rotor_quant_kv
- `2`: 1x, avg=1.5, domains=rotor_quant_kv
### `n_share_groups`
- `8`: 2x, avg=-280.5, domains=cross_layer_kv
- `7`: 2x, avg=-311.8, domains=cross_layer_kv,cross_layer_kv_refine_d1
- `4`: 1x, avg=-360.1, domains=cross_layer_kv_refine_d1
- `2`: 1x, avg=-394.8, domains=cross_layer_kv_refine_d1
### `n_sink`
- `4`: 2x, avg=96.9, domains=streaming_kv,streaming_kv_refine_d1
- `6`: 2x, avg=96.8, domains=streaming_kv,streaming_kv_refine_d1
- `5`: 2x, avg=96.5, domains=streaming_kv,streaming_kv_refine_d1
### `n_sinks`
- `2`: 3x, avg=-152.2, domains=kv_eviction_refine_d1
- `3`: 2x, avg=-6.0, domains=kv_eviction
- `7`: 1x, avg=-4.0, domains=kv_eviction
### `n_skip_layers`
- `0`: 2x, avg=9.9, domains=mod_config
- `3`: 1x, avg=10.0, domains=mod_config
- `2`: 1x, avg=10.1, domains=mod_config_refine_d1
- `16`: 1x, avg=10.1, domains=mod_config_refine_d1
- `15`: 1x, avg=10.1, domains=mod_config_refine_d1
### `n_tiles`
- `31`: 2x, avg=4.2, domains=mosaic_quant
- `30`: 1x, avg=4.2, domains=mosaic_quant
- `29`: 1x, avg=4.2, domains=mosaic_quant_refine_d1
- `26`: 1x, avg=4.2, domains=mosaic_quant_refine_d1
- `32`: 1x, avg=4.2, domains=mosaic_quant_refine_d1
### `nesterov`
- `True`: 2x, avg=11.9, domains=muon_config
- `False`: 1x, avg=11.9, domains=muon_config
### `norm_type`
- `layernorm`: 3x, avg=9.7, domains=qk_norm
### `ns_steps`
- `1`: 1x, avg=11.9, domains=muon_config
- `3`: 1x, avg=11.9, domains=muon_config
- `5`: 1x, avg=11.9, domains=muon_config
### `observation_window`
- `52`: 1x, avg=-4.0, domains=kv_eviction
- `60`: 1x, avg=-5.4, domains=kv_eviction
- `407`: 1x, avg=-6.7, domains=kv_eviction
- `512`: 1x, avg=-50.0, domains=kv_eviction_refine_d1
- `64`: 1x, avg=-197.8, domains=kv_eviction_refine_d1
### `offload_layers`
- `1`: 3x, avg=0.0, domains=cpu_kv_offload
- `4`: 3x, avg=-1.0, domains=cpu_kv_offload_refine_d1
### `offload_ratio`
- `3.946805104959594e-10`: 1x, avg=30.0, domains=cpu_adamw_config
- `9.943115486521492e-08`: 1x, avg=30.0, domains=cpu_adamw_config
- `4.590047865349334e-06`: 1x, avg=30.0, domains=cpu_adamw_config
- `0.0758352056145668`: 1x, avg=27.9, domains=cpu_adamw_config_refine_d1
- `0.07666758447885513`: 1x, avg=27.8, domains=cpu_adamw_config_refine_d1
### `offload_threshold`
- `0.8999639749526978`: 3x, avg=-1.0, domains=cpu_kv_offload_refine_d1
- `0.9999639987945557`: 1x, avg=0.0, domains=cpu_kv_offload
- `0.999956488609314`: 1x, avg=0.0, domains=cpu_kv_offload
- `0.9993625283241272`: 1x, avg=0.0, domains=cpu_kv_offload
### `offset_init`
- `0.05462278425693512`: 1x, avg=-16.5, domains=offq_quant
- `0.07712052017450333`: 1x, avg=-16.7, domains=offq_quant
- `0.18691237270832062`: 1x, avg=-16.7, domains=offq_quant
### `opt_type`
- `adamw`: 3x, avg=24.5, domains=optimizer_config
### `overlap`
- `0.1`: 1x, avg=96.8, domains=streaming_kv
- `0.16356708109378815`: 1x, avg=96.7, domains=streaming_kv
- `0.4730985164642334`: 1x, avg=95.9, domains=streaming_kv
- `0.059021227061748505`: 1x, avg=97.0, domains=streaming_kv_refine_d1
- `0.05060712248086929`: 1x, avg=97.0, domains=streaming_kv_refine_d1
### `overlap_compute`
- `True`: 6x, avg=4.0, domains=hybrid_offload,hybrid_offload_refine_d1
### `overlap_ratio`
- `0.0`: 2x, avg=8.6, domains=sliding_window_refine_d1
- `0.8495826125144958`: 1x, avg=9.6, domains=sliding_window
- `0.5944046378135681`: 1x, avg=9.6, domains=sliding_window
- `0.9705857038497925`: 1x, avg=9.5, domains=sliding_window
- `0.14549130201339722`: 1x, avg=8.6, domains=sliding_window_refine_d1
### `padding_strategy`
- `left`: 5x, avg=42.8, domains=batched_decode,batched_decode_refine_d1
- `right`: 1x, avg=43.3, domains=batched_decode
### `page_size`
- `64`: 1x, avg=128.1, domains=paged_evict_kv
- `32`: 1x, avg=128.1, domains=paged_evict_kv
- `16`: 1x, avg=127.1, domains=paged_evict_kv
- `41`: 1x, avg=128.2, domains=paged_evict_kv_refine_d1
- `54`: 1x, avg=124.9, domains=paged_evict_kv_refine_d1
### `pattern_type`
- `csa_hca_hybrid`: 3x, avg=14.8, domains=csa_attention
### `per_channel`
- `False`: 3x, avg=178.7, domains=w8a8_quant,w8a8_quant_refine_d1
- `True`: 3x, avg=179.0, domains=w8a8_quant,w8a8_quant_refine_d1
### `percentile`
- `0.9000000428387552`: 1x, avg=9.6, domains=activation_quant
- `0.9001394818278495`: 1x, avg=9.6, domains=activation_quant
- `0.900326121506514`: 1x, avg=9.6, domains=activation_quant
### `pin_memory`
- `True`: 6x, avg=4.0, domains=hybrid_offload,hybrid_offload_refine_d1
### `prefetch_ahead`
- `4`: 3x, avg=8.5, domains=expert_hotload,expert_hotload_refine_d1
- `1`: 2x, avg=8.1, domains=expert_hotload_refine_d1
- `2`: 1x, avg=8.5, domains=expert_hotload
### `prefetch_depth`
- `2`: 3x, avg=27.8, domains=cpu_adamw_config_refine_d1
- `5`: 3x, avg=3.8, domains=hybrid_offload
- `8`: 3x, avg=4.2, domains=hybrid_offload_refine_d1
- `3`: 1x, avg=30.0, domains=cpu_adamw_config
- `7`: 1x, avg=30.0, domains=cpu_adamw_config
### `prefetch_size`
- `2048`: 4x, avg=-0.2, domains=cpu_kv_offload,cpu_kv_offload_refine_d1
- `1024`: 1x, avg=-1.0, domains=cpu_kv_offload_refine_d1
- `512`: 1x, avg=-1.1, domains=cpu_kv_offload_refine_d1
### `quant_bits`
- `8`: 6x, avg=78.2, domains=xquant_kv,xquant_kv_refine_d1
- `4`: 5x, avg=8.1, domains=hadamard_kv_refine_d1,rotor_quant_kv
### `quant_mode`
- `ternary`: 6x, avg=-27.4, domains=bitnet_config,bitnet_config_refine_d1
### `rank`
- `64`: 5x, avg=15.8, domains=factorized_embed_refine_d1,mhc_config,mhc_config_refine_d1
- `68`: 2x, avg=14.4, domains=factorized_embed,mhc_config_refine_d1
- `72`: 1x, avg=18.9, domains=factorized_embed
- `215`: 1x, avg=17.3, domains=factorized_embed
- `65`: 1x, avg=10.4, domains=mhc_config
### `recency_decay`
- `1.0`: 1x, avg=4.5, domains=hqe_kv
- `0.95`: 1x, avg=4.4, domains=hqe_kv
- `0.8`: 1x, avg=4.4, domains=hqe_kv
### `recomputation_ratio`
- `0.5`: 2x, avg=76.6, domains=xquant_kv_refine_d1
- `0.5119810104370117`: 1x, avg=81.3, domains=xquant_kv
- `0.5397486686706543`: 1x, avg=80.1, domains=xquant_kv
- `0.5637326836585999`: 1x, avg=79.3, domains=xquant_kv
- `0.25`: 1x, avg=75.4, domains=xquant_kv_refine_d1
### `recompute_layers`
- `16`: 3x, avg=56.9, domains=kv_recompute,kv_recompute_refine_d1
- `14`: 2x, avg=61.3, domains=kv_recompute,kv_recompute_refine_d1
- `12`: 1x, avg=63.0, domains=kv_recompute_refine_d1
### `recompute_strategy`
- `selective`: 12x, avg=33.9, domains=checkpoint_recompute,checkpoint_recompute_refine_d1,kv_recompute,kv_recompute_refine_d1
### `repetition_penalty`
- `1.0`: 3x, avg=10.6, domains=sampling_config_refine_d1
- `1.266894280910492`: 1x, avg=14.4, domains=sampling_config
- `1.3984755277633667`: 1x, avg=13.8, domains=sampling_config
- `1.2824791967868805`: 1x, avg=13.5, domains=sampling_config
### `reserve`
- `0.06700361520051956`: 1x, avg=0.6, domains=memory_budget
- `0.05818496644496918`: 1x, avg=0.6, domains=memory_budget
- `0.07487867027521133`: 1x, avg=0.6, domains=memory_budget
- `0.05370505154132843`: 1x, avg=0.6, domains=memory_budget_refine_d1
- `0.053110286593437195`: 1x, avg=0.6, domains=memory_budget_refine_d1
### `residual_ratio`
- `0.05`: 2x, avg=-4.7, domains=quant
- `0.0`: 1x, avg=-2.6, domains=quant
### `retrieval_dim`
- `64`: 4x, avg=3.3, domains=attn_residual,attn_residual_refine_d1
- `93`: 1x, avg=3.2, domains=attn_residual
- `83`: 1x, avg=3.2, domains=attn_residual_refine_d1
### `rot_type`
- `random`: 2x, avg=0.5, domains=rotor_quant_kv
- `hadamard`: 1x, avg=1.5, domains=rotor_quant_kv
### `rounding_method`
- `rtn`: 3x, avg=-4.0, domains=quant
### `router_mode`
- `switch`: 3x, avg=26.7, domains=moe_routing
- `aux_free`: 3x, avg=26.3, domains=moe_routing_refine_d1
### `router_type`
- `mlp`: 5x, avg=10.0, domains=mod_config,mod_config_refine_d1
- `linear`: 1x, avg=10.1, domains=mod_config_refine_d1
### `scale_clip_min`
- `0.01`: 2x, avg=-3.7, domains=quant
- `0.001`: 1x, avg=-4.6, domains=quant
### `scale_init`
- `0.5006855120736873`: 1x, avg=9.7, domains=qk_norm
- `0.5033548106439412`: 1x, avg=9.7, domains=qk_norm
- `0.5012165468069725`: 1x, avg=9.7, domains=qk_norm
### `scale_method`
- `absmax`: 3x, avg=-4.0, domains=quant
### `scale_mode`
- `per_channel`: 2x, avg=14.0, domains=nvfp4_quant
- `per_block`: 1x, avg=14.3, domains=nvfp4_quant
### `scale_search_range`
- `0.3`: 3x, avg=-4.0, domains=quant
### `scale_search_steps`
- `5`: 3x, avg=-4.0, domains=quant
### `scaling_factor`
- `0.546415823046118`: 1x, avg=34.6, domains=rope_config
- `0.6064735520631075`: 1x, avg=34.5, domains=rope_config
- `0.7994920015335083`: 1x, avg=34.4, domains=rope_config
- `0.578122635371983`: 1x, avg=34.4, domains=rope_config_refine_d1
- `0.5588208986446261`: 1x, avg=34.4, domains=rope_config_refine_d1
### `scaling_type`
- `linear`: 6x, avg=34.5, domains=rope_config,rope_config_refine_d1
### `sched_type`
- `cosine`: 6x, avg=10.0, domains=scheduler_config,scheduler_config_refine_d1
### `scheme`
- `asymmetric`: 6x, avg=9.1, domains=group_quant,group_quant_refine_d1
### `share_mode`
- `avg`: 6x, avg=-323.2, domains=cross_layer_kv,cross_layer_kv_refine_d1
### `share_ratio`
- `0.75`: 2x, avg=-377.4, domains=cross_layer_kv_refine_d1
- `1.0`: 1x, avg=-255.5, domains=cross_layer_kv
- `0.0`: 1x, avg=-305.5, domains=cross_layer_kv
- `0.9562302231788635`: 1x, avg=-311.8, domains=cross_layer_kv
- `0.936205267906189`: 1x, avg=-311.8, domains=cross_layer_kv_refine_d1
### `share_weights`
- `True`: 6x, avg=25.4, domains=mtp_config,mtp_config_refine_d1
### `shared_expert`
- `True`: 6x, avg=26.5, domains=moe_routing,moe_routing_refine_d1
### `sink_size`
- `6`: 3x, avg=-6.9, domains=kara,kara_refine_d1
- `2`: 1x, avg=1.6, domains=kara
- `8`: 1x, avg=-24.0, domains=kara_refine_d1
- `4`: 1x, avg=-24.3, domains=kara_refine_d1
### `skip_strategy`
- `cosine`: 3x, avg=-6.2, domains=ffn_skip,ffn_skip_refine_d1
- `hybrid`: 3x, avg=-6.2, domains=ffn_skip,ffn_skip_refine_d1
### `skip_threshold`
- `0.0`: 2x, avg=-6.2, domains=ffn_skip_refine_d1
- `0.008037981577217579`: 1x, avg=-6.2, domains=ffn_skip
- `0.0054512484930455685`: 1x, avg=-6.2, domains=ffn_skip
- `0.003639993956312537`: 1x, avg=-6.2, domains=ffn_skip
- `0.003176305443048477`: 1x, avg=-6.2, domains=ffn_skip_refine_d1
### `smooth_alpha`
- `0.999768078327179`: 1x, avg=9.6, domains=activation_quant
- `0.36061403155326843`: 1x, avg=9.6, domains=activation_quant
- `0.7533970475196838`: 1x, avg=9.6, domains=activation_quant
### `smooth_swiglu`
- `False`: 6x, avg=29.3, domains=fp8_training_config,fp8_training_config_refine_d1
### `smoothquant_alpha`
- `1.0`: 2x, avg=178.5, domains=w8a8_quant_refine_d1
- `0.9990673661231995`: 1x, avg=179.5, domains=w8a8_quant
- `0.9989532232284546`: 1x, avg=179.4, domains=w8a8_quant
- `0.996192455291748`: 1x, avg=178.9, domains=w8a8_quant
- `0.9199916124343872`: 1x, avg=178.3, domains=w8a8_quant_refine_d1
### `softmax_sep`
- `0.0`: 3x, avg=10.0, domains=diff_attn_refine_d1
- `0.008012239821255207`: 1x, avg=10.0, domains=diff_attn
- `0.9963721036911011`: 1x, avg=10.0, domains=diff_attn
- `0.7972360849380493`: 1x, avg=10.0, domains=diff_attn
### `strategy`
- `streaming`: 3x, avg=-5.3, domains=kv_eviction
- `snapkv`: 3x, avg=-152.2, domains=kv_eviction_refine_d1
- `compact`: 3x, avg=-25.9, domains=sparse_attn
### `stride`
- `1`: 2x, avg=4.1, domains=conv_config
- `2`: 1x, avg=4.2, domains=conv_config
- `78`: 1x, avg=9.6, domains=sliding_window
- `105`: 1x, avg=9.6, domains=sliding_window
- `194`: 1x, avg=9.5, domains=sliding_window
### `sync_freq`
- `14`: 3x, avg=11.1, domains=grad_accum_config_refine_d1
- `15`: 1x, avg=11.1, domains=grad_accum_config
- `12`: 1x, avg=11.1, domains=grad_accum_config
- `6`: 1x, avg=11.1, domains=grad_accum_config
### `target_budget`
- `512`: 3x, avg=-24.2, domains=kara_refine_d1
- `3727`: 1x, avg=1.9, domains=kara
- `1282`: 1x, avg=1.7, domains=kara
- `1084`: 1x, avg=1.6, domains=kara
### `temperature`
- `1.0`: 2x, avg=55.2, domains=speculative_decode_refine_d1
- `1.496375560760498`: 1x, avg=4.4, domains=loss_config
- `1.4854052364826202`: 1x, avg=4.4, domains=loss_config
- `1.3274538815021515`: 1x, avg=4.4, domains=loss_config
- `1.9688957631587982`: 1x, avg=2.0, domains=loss_config_refine_d1
### `theta`
- `10000000.0`: 3x, avg=34.4, domains=rope_config_refine_d1
- `8939256.656765938`: 1x, avg=34.6, domains=rope_config
- `7695767.059803009`: 1x, avg=34.5, domains=rope_config
- `9999989.272236824`: 1x, avg=34.4, domains=rope_config
### `threshold`
- `0.7`: 3x, avg=60.4, domains=kv_recompute,kv_recompute_refine_d1
- `0.9`: 1x, avg=52.7, domains=kv_recompute
- `0.8437532186508179`: 1x, avg=52.0, domains=kv_recompute
- `0.925775945186615`: 1x, avg=70.7, domains=kv_recompute_refine_d1
### `tie_factor`
- `0.02759479358792305`: 1x, avg=18.9, domains=factorized_embed
- `0.08808882534503937`: 1x, avg=18.4, domains=factorized_embed
- `0.04103472828865051`: 1x, avg=17.3, domains=factorized_embed
- `1.0`: 1x, avg=19.3, domains=factorized_embed_refine_d1
- `0.9108207821846008`: 1x, avg=19.3, domains=factorized_embed_refine_d1
### `tie_strength`
- `0.533734142780304`: 1x, avg=9.2, domains=gta_attention
- `0.9152335524559021`: 1x, avg=9.0, domains=gta_attention
- `0.6125547885894775`: 1x, avg=8.9, domains=gta_attention
- `0.6292574405670166`: 1x, avg=9.3, domains=gta_attention_refine_d1
- `0.6255279779434204`: 1x, avg=9.3, domains=gta_attention_refine_d1
### `tile_dim`
- `409`: 1x, avg=4.2, domains=mosaic_quant
- `511`: 1x, avg=4.2, domains=mosaic_quant
- `415`: 1x, avg=4.2, domains=mosaic_quant
- `102`: 1x, avg=4.2, domains=mosaic_quant_refine_d1
- `64`: 1x, avg=4.2, domains=mosaic_quant_refine_d1
### `top_k`
- `3`: 6x, avg=26.5, domains=moe_routing,moe_routing_refine_d1
- `64`: 2x, avg=14.8, domains=csa_attention
- `66`: 1x, avg=14.8, domains=csa_attention
- `70`: 1x, avg=14.4, domains=sampling_config
- `5`: 1x, avg=13.8, domains=sampling_config
### `top_p`
- `1.0`: 3x, avg=10.6, domains=sampling_config_refine_d1
- `0.5126710869371891`: 1x, avg=14.4, domains=sampling_config
- `0.558130145072937`: 1x, avg=13.8, domains=sampling_config
- `0.5258882623165846`: 1x, avg=13.5, domains=sampling_config
### `update_freq`
- `1`: 6x, avg=8.7, domains=cpu_adamw_config,titan_memory,titan_memory_refine_d1
- `15`: 3x, avg=29.2, domains=cpu_adamw_config,cpu_adamw_config_refine_d1
- `16`: 2x, avg=27.8, domains=cpu_adamw_config_refine_d1
- `2`: 1x, avg=3.4, domains=titan_memory_refine_d1
### `use_adaptive_spec`
- `0.0032670486252754927`: 1x, avg=64.9, domains=Generic_use_adaptive_spec
- `0.9838629961013794`: 1x, avg=64.7, domains=Generic_use_adaptive_spec
- `0.02503044344484806`: 1x, avg=64.6, domains=Generic_use_adaptive_spec
- `0.0`: 1x, avg=65.0, domains=Generic_use_adaptive_spec_refine_d1
- `1.325458288192749e-05`: 1x, avg=65.0, domains=Generic_use_adaptive_spec_refine_d1
### `use_block_fusion`
- `0.9691361784934998`: 1x, avg=64.5, domains=Generic_use_block_fusion
- `0.9993190765380859`: 1x, avg=64.3, domains=Generic_use_block_fusion
- `0.9969172477722168`: 1x, avg=64.3, domains=Generic_use_block_fusion
- `0.9941515922546387`: 1x, avg=64.9, domains=Generic_use_block_fusion_refine_d1
- `1.0`: 1x, avg=64.9, domains=Generic_use_block_fusion_refine_d1
### `use_breakable_cuda_graph`
- `0.997200608253479`: 1x, avg=64.8, domains=Generic_use_breakable_cuda_graph
- `0.005650537554174662`: 1x, avg=64.8, domains=Generic_use_breakable_cuda_graph
- `0.007496565114706755`: 1x, avg=64.3, domains=Generic_use_breakable_cuda_graph
### `use_chunked_prefill`
- `0.005063542630523443`: 1x, avg=64.9, domains=Generic_use_chunked_prefill
- `0.009275319054722786`: 1x, avg=64.8, domains=Generic_use_chunked_prefill
- `0.004198264796286821`: 1x, avg=64.8, domains=Generic_use_chunked_prefill
- `1.0`: 1x, avg=65.0, domains=Generic_use_chunked_prefill_refine_d1
- `0.0`: 1x, avg=65.0, domains=Generic_use_chunked_prefill_refine_d1
### `use_compact_attn`
- `0.993290901184082`: 1x, avg=64.5, domains=Generic_use_compact_attn
- `0.006404716521501541`: 1x, avg=64.5, domains=Generic_use_compact_attn
- `0.9745955467224121`: 1x, avg=64.5, domains=Generic_use_compact_attn
- `1.0`: 1x, avg=65.0, domains=Generic_use_compact_attn_refine_d1
- `0.9996157884597778`: 1x, avg=65.0, domains=Generic_use_compact_attn_refine_d1
### `use_compile`
- `1.0`: 3x, avg=65.0, domains=Generic_use_compile_refine_d1
- `0.9983477592468262`: 1x, avg=65.0, domains=Generic_use_compile
- `0.9988700747489929`: 1x, avg=64.8, domains=Generic_use_compile
- `0.9956179857254028`: 1x, avg=64.8, domains=Generic_use_compile
### `use_compile_autotune`
- `0.0`: 2x, avg=65.0, domains=Generic_use_compile_autotune_refine_d1
- `0.990142822265625`: 1x, avg=64.8, domains=Generic_use_compile_autotune
- `0.007768276613205671`: 1x, avg=64.5, domains=Generic_use_compile_autotune
- `0.03320569917559624`: 1x, avg=64.2, domains=Generic_use_compile_autotune
- `1.0`: 1x, avg=65.0, domains=Generic_use_compile_autotune_refine_d1
### `use_corun`
- `0.010578980669379234`: 1x, avg=64.7, domains=Generic_use_corun
- `0.03681456670165062`: 1x, avg=64.2, domains=Generic_use_corun
- `0.0140072675421834`: 1x, avg=64.1, domains=Generic_use_corun
- `0.0`: 1x, avg=65.0, domains=Generic_use_corun_refine_d1
- `0.0008698254823684692`: 1x, avg=65.0, domains=Generic_use_corun_refine_d1
### `use_cosa`
- `0.002034958219155669`: 1x, avg=64.8, domains=Generic_use_cosa
- `0.9943264126777649`: 1x, avg=64.7, domains=Generic_use_cosa
- `0.9666350483894348`: 1x, avg=64.0, domains=Generic_use_cosa
- `0.00019374489784240723`: 1x, avg=65.0, domains=Generic_use_cosa_refine_d1
- `1.0`: 1x, avg=65.0, domains=Generic_use_cosa_refine_d1
### `use_foundry`
- `1.0`: 3x, avg=65.0, domains=Generic_use_foundry_refine_d1
- `0.022415120154619217`: 1x, avg=64.6, domains=Generic_use_foundry
- `0.9915881752967834`: 1x, avg=64.5, domains=Generic_use_foundry
- `0.02895944193005562`: 1x, avg=64.4, domains=Generic_use_foundry
### `use_fused_qk_norm_rope_cache`
- `0.0`: 3x, avg=65.0, domains=Generic_use_fused_qk_norm_rope_cache_refine_d1
- `0.00599229522049427`: 1x, avg=64.8, domains=Generic_use_fused_qk_norm_rope_cache
- `0.0205962136387825`: 1x, avg=64.6, domains=Generic_use_fused_qk_norm_rope_cache
- `0.02132292650640011`: 1x, avg=64.5, domains=Generic_use_fused_qk_norm_rope_cache
### `use_hadamard`
- `False`: 3x, avg=-4.0, domains=quant
### `use_hotprefix`
- `0.0`: 2x, avg=65.0, domains=Generic_use_hotprefix_refine_d1
- `0.011793042533099651`: 1x, avg=64.7, domains=Generic_use_hotprefix
- `0.9665820598602295`: 1x, avg=64.3, domains=Generic_use_hotprefix
- `0.030660094693303108`: 1x, avg=64.1, domains=Generic_use_hotprefix
### `use_hybrid_prefill`
- `0.9811527729034424`: 1x, avg=64.6, domains=Generic_use_hybrid_prefill
- `0.9856782555580139`: 1x, avg=64.5, domains=Generic_use_hybrid_prefill
- `0.9583883881568909`: 1x, avg=64.2, domains=Generic_use_hybrid_prefill
- `1.0`: 1x, avg=65.0, domains=Generic_use_hybrid_prefill_refine_d1
- `0.9978470802307129`: 1x, avg=65.0, domains=Generic_use_hybrid_prefill_refine_d1
### `use_learned_prefix_cache`
- `0.0`: 2x, avg=65.0, domains=Generic_use_learned_prefix_cache_refine_d1
- `0.021552495658397675`: 1x, avg=64.5, domains=Generic_use_learned_prefix_cache
- `0.026164527982473373`: 1x, avg=64.5, domains=Generic_use_learned_prefix_cache
- `0.04269755259156227`: 1x, avg=64.2, domains=Generic_use_learned_prefix_cache
- `0.0006692111492156982`: 1x, avg=65.0, domains=Generic_use_learned_prefix_cache_refine_d1
### `use_mosa`
- `0.0`: 2x, avg=65.0, domains=Generic_use_mosa_refine_d1
- `0.033636439591646194`: 1x, avg=64.3, domains=Generic_use_mosa
- `0.04110463336110115`: 1x, avg=64.1, domains=Generic_use_mosa
- `0.9727048873901367`: 1x, avg=64.1, domains=Generic_use_mosa
- `0.0036135204136371613`: 1x, avg=64.9, domains=Generic_use_mosa_refine_d1
### `use_pod_attention`
- `0.9900026321411133`: 1x, avg=64.8, domains=Generic_use_pod_attention
- `0.013253624550998211`: 1x, avg=64.7, domains=Generic_use_pod_attention
- `0.005085628479719162`: 1x, avg=64.7, domains=Generic_use_pod_attention
- `0.0`: 1x, avg=65.0, domains=Generic_use_pod_attention_refine_d1
- `0.00018574297428131104`: 1x, avg=65.0, domains=Generic_use_pod_attention_refine_d1
### `use_prefix_cache`
- `1.0`: 2x, avg=65.0, domains=Generic_use_prefix_cache_refine_d1
- `0.9933805465698242`: 1x, avg=64.8, domains=Generic_use_prefix_cache
- `0.9845463037490845`: 1x, avg=64.7, domains=Generic_use_prefix_cache
- `0.9828055500984192`: 1x, avg=64.7, domains=Generic_use_prefix_cache
- `0.9998519420623779`: 1x, avg=65.0, domains=Generic_use_prefix_cache_refine_d1
### `use_progressive_kv`
- `0.0`: 2x, avg=65.0, domains=Generic_use_progressive_kv_refine_d1
- `0.9936783909797668`: 1x, avg=64.8, domains=Generic_use_progressive_kv
- `0.9980587363243103`: 1x, avg=64.7, domains=Generic_use_progressive_kv
- `0.014395341277122498`: 1x, avg=64.7, domains=Generic_use_progressive_kv
- `4.354119300842285e-05`: 1x, avg=65.0, domains=Generic_use_progressive_kv_refine_d1
### `use_seq_split`
- `0.00903824158012867`: 1x, avg=64.8, domains=Generic_use_seq_split
- `0.011507074348628521`: 1x, avg=64.7, domains=Generic_use_seq_split
- `0.9887959957122803`: 1x, avg=64.6, domains=Generic_use_seq_split
- `1.0`: 1x, avg=65.0, domains=Generic_use_seq_split_refine_d1
- `0.9998842477798462`: 1x, avg=65.0, domains=Generic_use_seq_split_refine_d1
### `use_spec_attn`
- `0.003011946566402912`: 1x, avg=64.7, domains=Generic_use_spec_attn
- `0.9967620372772217`: 1x, avg=64.7, domains=Generic_use_spec_attn
- `0.007695632521063089`: 1x, avg=64.6, domains=Generic_use_spec_attn
- `0.9997426271438599`: 1x, avg=65.0, domains=Generic_use_spec_attn_refine_d1
- `0.9954344630241394`: 1x, avg=64.9, domains=Generic_use_spec_attn_refine_d1
### `use_suffix_spec`
- `1.0`: 2x, avg=65.0, domains=Generic_use_suffix_spec_refine_d1
- `0.013989963568747044`: 1x, avg=64.7, domains=Generic_use_suffix_spec
- `0.008543644100427628`: 1x, avg=64.7, domains=Generic_use_suffix_spec
- `0.9791184663772583`: 1x, avg=64.7, domains=Generic_use_suffix_spec
- `0.9981632232666016`: 1x, avg=65.0, domains=Generic_use_suffix_spec_refine_d1
### `use_triroute`
- `0.0`: 2x, avg=65.0, domains=Generic_use_triroute_refine_d1
- `0.0065451497212052345`: 1x, avg=64.7, domains=Generic_use_triroute
- `0.027177713811397552`: 1x, avg=64.2, domains=Generic_use_triroute
- `0.03177972882986069`: 1x, avg=63.8, domains=Generic_use_triroute
- `0.006047293543815613`: 1x, avg=64.9, domains=Generic_use_triroute_refine_d1
### `use_triton_conv`
- `1.0`: 2x, avg=65.0, domains=Generic_use_triton_conv_refine_d1
- `0.004687928128987551`: 1x, avg=64.9, domains=Generic_use_triton_conv
- `0.007595459930598736`: 1x, avg=64.8, domains=Generic_use_triton_conv
- `0.9957398176193237`: 1x, avg=64.7, domains=Generic_use_triton_conv
- `0.9954904317855835`: 1x, avg=64.9, domains=Generic_use_triton_conv_refine_d1
### `use_v0_warm`
- `1.0`: 3x, avg=65.0, domains=Generic_use_v0_warm_refine_d1
- `0.9987877011299133`: 1x, avg=64.9, domains=Generic_use_v0_warm
- `0.9921891689300537`: 1x, avg=64.8, domains=Generic_use_v0_warm
- `0.018126409500837326`: 1x, avg=64.5, domains=Generic_use_v0_warm
### `use_wavelength_pruning`
- `0.0034131938591599464`: 1x, avg=64.9, domains=Generic_use_wavelength_pruning
- `0.9943200349807739`: 1x, avg=64.8, domains=Generic_use_wavelength_pruning
- `0.003946754150092602`: 1x, avg=64.7, domains=Generic_use_wavelength_pruning
- `0.0`: 1x, avg=65.0, domains=Generic_use_wavelength_pruning_refine_d1
- `1.0`: 1x, avg=65.0, domains=Generic_use_wavelength_pruning_refine_d1
### `v_k_mix`
- `0.0`: 3x, avg=9.3, domains=gta_attention_refine_d1
- `0.0033624120987951756`: 1x, avg=9.2, domains=gta_attention
- `0.020175281912088394`: 1x, avg=9.0, domains=gta_attention
- `0.022901857271790504`: 1x, avg=8.9, domains=gta_attention
### `vocab_size`
- `65536`: 2x, avg=19.3, domains=factorized_embed_refine_d1
- `39902`: 1x, avg=18.9, domains=factorized_embed
- `46569`: 1x, avg=18.4, domains=factorized_embed
- `40110`: 1x, avg=17.3, domains=factorized_embed
- `64911`: 1x, avg=19.3, domains=factorized_embed_refine_d1
### `w4a8`
- `True`: 3x, avg=14.1, domains=nvfp4_quant
### `warmup_steps`
- `75`: 1x, avg=10.0, domains=scheduler_config
- `749`: 1x, avg=10.0, domains=scheduler_config
- `39`: 1x, avg=10.0, domains=scheduler_config
- `15`: 1x, avg=10.0, domains=scheduler_config_refine_d1
- `13`: 1x, avg=10.0, domains=scheduler_config_refine_d1
### `weight_budget`
- `0.19435393810272217`: 1x, avg=0.6, domains=memory_budget
- `0.19752295315265656`: 1x, avg=0.6, domains=memory_budget
- `0.20053376257419586`: 1x, avg=0.6, domains=memory_budget
- `0.19535543024539948`: 1x, avg=0.6, domains=memory_budget_refine_d1
- `0.19492179155349731`: 1x, avg=0.6, domains=memory_budget_refine_d1
### `weight_decay`
- `0.005212975144386292`: 1x, avg=11.9, domains=muon_config
- `0.0019092297554016114`: 1x, avg=11.9, domains=muon_config
- `0.008289214372634888`: 1x, avg=11.9, domains=muon_config
- `0.0002499049296602607`: 1x, avg=24.5, domains=optimizer_config
- `0.05111624598503113`: 1x, avg=24.4, domains=optimizer_config
### `window_size`
- `256`: 3x, avg=-24.2, domains=kara_refine_d1
- `128`: 3x, avg=-152.2, domains=kv_eviction_refine_d1
- `360`: 1x, avg=1.9, domains=kara
- `241`: 1x, avg=1.7, domains=kara
- `838`: 1x, avg=1.6, domains=kara
### `x`
- `[0.21149232983589172, 0.9307464361190796, 0.9266494512557983, 0.9777611494064331, 0.35501474142074585, 0.764074444770813, 0.08462908864021301, 0.8343579173088074]`: 1x, avg=21.4, domains=synthetic
- `[0.02276439 0.98795563 0.51748693 0.97020143 0.14724302 0.32582197
 0.25636983 0.86014533]`: 1x, avg=20.8, domains=synthetic
- `[0.5232633948326111, 0.9593720436096191, 0.9440037608146667, 0.41904279589653015, 0.10955667495727539, 0.14226679503917694, 0.23010389506816864, 0.6533341407775879]`: 1x, avg=20.1, domains=synthetic
- `[0.12018084526062012, 0.9770234823226929, 0.548395574092865, 0.9852800369262695, 0.1675012856721878, 0.3275863230228424, 0.2662193179130554, 0.8825550675392151]`: 1x, avg=25.9, domains=synthetic_refine_d1
- `[0.0, 0.940140426158905, 0.49625641107559204, 1.0, 0.10090510547161102, 0.3667561411857605, 0.22193539142608643, 0.8438556790351868]`: 1x, avg=13.3, domains=synthetic_refine_d1
