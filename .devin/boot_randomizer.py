"""Randomizer: throw all known + loosely related systems into a randomizer
to generate unexpected combinations for boot-time reduction.

Step 7 of the Novel Discovery Protocol.
"""
import random
random.seed(42)

techniques = [
    "meta_device_init", "parallel_tokenizer", "os_page_prefetch",
    "prefetch_virtual_memory", "skip_init", "fastsafetensors_pipeline",
    "cuda_stream_weight_copy", "torch_compile_warm_cache",
    "kv_cache_lazy_alloc", "mmap_map_populate", "weight_quantization_at_load",
    "module_prelink", "shared_memory_tokenizer", "lru_arch_cache",
    "buffer_reuse", "async_diff_convert", "lazy_key_instantiation",
    "persistent_compile_cache", "gigatoken_warm",
    "weight_tying_after_assign", "rope_buffer_reset",
    "cache_devices_during_load", "flash_attention_warmup",
    "empty_cache_before_load", "pin_memory_weights",
]

cross_domain = [
    "db_index_prefetch", "diffusion_noise_schedule",
    "gradient_sparsity_pattern", "compression_dict_lookup",
    "spectral_decomposition_load", "moe_expert_bake",
    "titant_memory_update", "mod_router_skip",
]

print("=== RANDOMIZER: 5 unexpected combinations ===\n")
for i in range(5):
    n_known = random.randint(2, 3)
    n_cross = random.randint(0, 1)
    combo = random.sample(techniques, n_known) + random.sample(cross_domain, n_cross)
    random.shuffle(combo)
    print(f"  Combo {i+1}: {' + '.join(combo)}")

print("\n=== Most promising combos to try next ===")
print("  A: meta_device_init + parallel_tokenizer + os_page_prefetch + cache_devices_during_load")
print("     (V7 + pre-scan devices during weight load instead of first forward)")
print("  B: meta_device_init + fastsafetensors_pipeline + lazy_key_instantiation")
print("     (skip TITAN/MHC/AttnRes init until first forward — they are zero-init=lossless)")
print("  C: meta_device_init + kv_cache_lazy_alloc + flash_attention_warmup")
print("     (overlap KV alloc + FA2 kernel compile with weight load)")
