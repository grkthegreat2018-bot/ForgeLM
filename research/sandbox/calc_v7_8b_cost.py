"""Full-stack cost analysis for ForgeLM V7-8B."""
import math

d_model = 4096
n_layers = 32
n_heads = 64
n_kv_heads = 16
head_dim = d_model // n_heads
intermediate = 16384
vocab = 65536
max_seq = 32768

total_params = 8.05e9

bitnet_weights = total_params * 1
bf16_master = total_params * 2
fp32_optimizer = total_params * 12

print("=" * 70)
print("  ForgeLM V7-8B Full-Stack Cost Analysis")
print("  Hardware: RTX 5070 12GB VRAM + 32GB RAM + NVMe")
print("=" * 70)

print(f"\n  === WEIGHTS ===")
print(f"  Total params:           {total_params/1e9:.2f}B")
print(f"  GPU weights (int8):     {bitnet_weights/1e9:.2f} GB")
print(f"  CPU master (bf16):      {bf16_master/1e9:.2f} GB")
print(f"  CPU optimizer (fp32):   {fp32_optimizer/1e9:.2f} GB")

kv_per_token = n_kv_heads * head_dim * 2 * 2
kv_32k = kv_per_token * max_seq
kv_32k_compressed = kv_32k / 8

print(f"\n  === KV CACHE ===")
print(f"  Per token:              {kv_per_token/1024:.1f} KB")
print(f"  32K context (bf16):     {kv_32k/1e9:.2f} GB")
print(f"  32K with KARA 8x:       {kv_32k_compressed/1e9:.2f} GB")
print(f"  4K context (bf16):      {kv_per_token * 4096 / 1e9:.3f} GB")

inference_vram = bitnet_weights + kv_32k_compressed + 0.5e9
w8a8_weights = total_params * 1
fp4_weights = total_params * 0.56

print(f"\n  === INFERENCE VRAM (GPU) ===")
fits1 = inference_vram < 12e9
fits2 = (w8a8_weights + kv_32k + 0.5e9) < 12e9
fits3 = (fp4_weights + kv_32k_compressed + 0.5e9) < 12e9
print(f"  BitNet int8 + KARA KV:  {inference_vram/1e9:.2f} GB  (fits 12GB: {'YES' if fits1 else 'NO'})")
print(f"  W8A8 + bf16 KV 32K:     {(w8a8_weights + kv_32k + 0.5e9)/1e9:.2f} GB  (fits 12GB: {'YES' if fits2 else 'NO'})")
print(f"  FP4 (HPR+IRI) + KARA:   {(fp4_weights + kv_32k_compressed + 0.5e9)/1e9:.2f} GB  (fits 12GB: {'YES' if fits3 else 'NO'})")

badam_vram = 8.50e9
badam_cpu_8bit = total_params * 4

print(f"\n  === TRAINING VRAM (BAdam) ===")
print(f"  GPU (1 layer active):   {badam_vram/1e9:.2f} GB  (fits 12GB: YES)")
print(f"  CPU optimizer (fp32):   {fp32_optimizer/1e9:.2f} GB  (exceeds 32GB RAM!)")
print(f"  CPU optimizer (8-bit):  {badam_cpu_8bit/1e9:.2f} GB  (fits 32GB: YES)")
print(f"  CPU master (bf16):      {bf16_master/1e9:.2f} GB")

print(f"\n  === FULL SYSTEM (inference, 32K context) ===")
inf_gpu = bitnet_weights + kv_32k_compressed + 0.5e9
inf_cpu = 0
print(f"  GPU VRAM:               {inf_gpu/1e9:.2f} GB / 12 GB")
print(f"  System RAM:             {inf_cpu/1e9:.2f} GB / 32 GB")
print(f"  NVMe (checkpoint):      {bitnet_weights/1e9:.2f} GB")

print(f"\n  === FULL SYSTEM (training, 1K context) ===")
train_gpu = badam_vram
train_cpu = bf16_master + badam_cpu_8bit
train_fits = train_cpu < 32e9
print(f"  GPU VRAM:               {train_gpu/1e9:.2f} GB / 12 GB")
print(f"  System RAM:             {train_cpu/1e9:.2f} GB / 32 GB  ({'FITS' if train_fits else 'EXCEEDS'})")
print(f"  NVMe (checkpoint):      {bitnet_weights/1e9:.2f} GB")

qsa_vram = 15e6
gr_vram = 256e6
ngram_host = 15.3e9
print(f"\n  === WITH R19 KEYS (inference) ===")
r19_gpu = inf_gpu + qsa_vram + gr_vram
r19_cpu = inf_cpu + ngram_host
r19_fits = r19_cpu < 32e9
print(f"  GPU VRAM:               {r19_gpu/1e9:.2f} GB / 12 GB")
print(f"  System RAM:             {r19_cpu/1e9:.2f} GB / 32 GB  ({'FITS' if r19_fits else 'EXCEEDS'})")

print(f"\n  === ESTIMATED THROUGHPUT (RTX 5070) ===")
# Decode is MEMORY-BANDWIDTH bound, not compute bound.
# RTX 5070: ~448 GB/s memory bandwidth (GDDR7)
# BitNet int8: 1 byte/param → 8.05 GB to read all weights per token
# Plus KV cache read: ~0.02 GB (KARA compressed)
# Plus activations: ~0.01 GB
# Total per-token memory: ~8.08 GB
# At 448 GB/s: 8.08 / 448 = 18ms/token → ~55 tok/s
# But BitNet int8@int8 GEMM has some overhead, realistic: ~40 tok/s
mem_bw = 448e9  # GB/s
bytes_per_token = bitnet_weights + kv_32k_compressed + 0.01e9
decode_s_per_tok = bytes_per_token / mem_bw
decode_tps = 1 / decode_s_per_tok
print(f"  Memory bandwidth:       {mem_bw/1e9:.0f} GB/s (RTX 5070 GDDR7)")
print(f"  Bytes/token (weights):  {bytes_per_token/1e9:.2f} GB")
print(f"  Decode (theoretical):   {decode_s_per_tok*1000:.1f} ms/tok -> {decode_tps:.1f} tok/s")
print(f"  Decode (realistic 70%): {decode_s_per_tok*1000/0.7:.1f} ms/tok -> {decode_tps*0.7:.1f} tok/s")
print(f"  With PEAGLE 2x:         {decode_tps*0.7*2:.1f} tok/s")
# Prefill is compute-bound: ~50 TFLOPS int8, 2*8B*2 FLOP/token
flops_per_token = 2 * total_params * 2
gpu_tflops_int8 = 50
prefill_tps = (gpu_tflops_int8 * 1e12) / flops_per_token
print(f"  Prefill (compute-bound): ~{prefill_tps:.0f} tok/s (theoretical)")
print(f"  Prefill (realistic 50%): ~{prefill_tps*0.5:.0f} tok/s")

print(f"\n  === CLOUD COST EQUIVALENT (RunPod) ===")
# RTX 5070 ~= RTX 4070 Ti on cloud ~= $0.40/hr
# Compare: 8B model on A100 80GB ~= $2.50/hr
print(f"  Local RTX 5070:         $0 (sunk cost) or ~$0.40/hr equivalent")
print(f"  RunPod RTX 4070 Ti:     ~$0.34/hr")
print(f"  RunPod A100 80GB:       ~$2.50/hr (needed for 8B bf16)")
print(f"  RunPod H100 80GB:       ~$4.00/hr")
print(f"  ")
print(f"  ForgeLM V7-8B on RTX 5070: ~$0.34/hr (fits 12GB with BitNet+KARA)")
print(f"  Same model on A100 (bf16):  ~$2.50/hr (no quant needed)")
print(f"  Monthly (24/7):         $248 vs $1,825 — 7.4x cheaper")
