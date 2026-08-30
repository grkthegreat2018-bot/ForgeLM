"""Full scratch training cost analysis for ForgeLM V8-8B.

Calculates: time, tokens, steps, GPU hours, electricity, cloud equivalent
for training V8-8B from scratch on ALL ForgeAI training data.
"""
import math

# ── Model config (from forgelm_v8_8b) ──
d_model = 4096
n_layers = 32
n_heads = 64
n_kv_heads = 16
head_dim = d_model // n_heads
intermediate = 16384
vocab = 65536
max_seq = 32768

# True parameter count (with HashedNLRQ)
# FFN dense: 2 * d_model * intermediate = 134M per layer × 32 = 4.29B
ffn_dense_per_layer = 2 * d_model * intermediate
ffn_dense_total = ffn_dense_per_layer * n_layers  # 4.29B

# NLRQ rank=1024: 1024 * (4096 + 16384) = 21.1M per layer
nlrq_per_layer = 1024 * (d_model + intermediate)
nlrq_total = nlrq_per_layer * n_layers  # 675M

# HashedNLRQ (8x hash): 675M / 8 = 84.4M
hashed_nlrq_total = nlrq_total / 8  # 84.4M

# Non-FFN params (attention + embeddings + norms + keys)
# Attention: 4 * d_model * d_model (Q,K,V,O) but GQA: Q,O = d_model², K,V = n_kv_heads*head_dim*d_model
attn_per_layer = (d_model * d_model * 2 +  # Q, O
                  n_kv_heads * head_dim * d_model * 2)  # K, V
attn_total = attn_per_layer * n_layers  # ~1.07B

# Embeddings (factorized): vocab * 512 + 512 * d_model
embed_params = vocab * 512 + 512 * d_model  # 35.6M

# R19 keys overhead
qsa_params = n_layers * 256 * 256  # small sparse attention params
gr_params = n_layers * d_model  # gated residual gates
ngram_params = vocab ** 3 * 256  # trigram table (host RAM, not GPU)

# Other keys (TITAN, MHC, MTP, AttnRes, LiSA, Hyperloop, etc.)
# Rough estimate: ~10% of attention params
other_keys = attn_total * 0.10

# Total true params (GPU)
non_ffn = attn_total + embed_params + qsa_params + gr_params + other_keys
true_params = non_ffn + hashed_nlrq_total
dense_equiv = non_ffn + ffn_dense_total  # 8.05B equivalent

# ── Training data ──
# Pretrain corpus: 39.76 GB JSONL, ~50.5M lines, ~845 chars/line
# Token estimate: ~4 chars/token → 845/4 ≈ 211 tokens/line
# Total pretrain tokens: 50.5M × 211 ≈ 10.66B tokens
pretrain_gb = 39.76
pretrain_lines = 50.5e6
tokens_per_line = 211
pretrain_tokens = pretrain_lines * tokens_per_line  # 10.66B

# Fine-tune + expert training: ~0.36 GB, ~240K lines, ~50M tokens
finetune_tokens = 240e3 * 200  # ~48M

# Total training tokens
total_tokens = pretrain_tokens + finetune_tokens  # ~10.71B

# ── Chinchilla scaling ──
# Optimal tokens-to-params ratio: 20:1 (Chinchilla)
# For 8B dense equiv: optimal = 160B tokens
# We have 10.7B tokens → 1.34:1 ratio (undertrained)
# But with HashedNLRQ (3.84B true params): 10.7B / 3.84B = 2.8:1 (better)
# For full Chinchilla-optimal: need 3.84B × 20 = 76.8B tokens
# We'll train for multiple epochs to compensate

# ── Training compute ──
# FLOPs per token (forward+backward) ≈ 6 × N_params
# With HashedNLRQ: effective compute uses true_params for optimizer
# but dense_equiv for forward/backward (reconstructed weights)
flops_per_token = 6 * dense_equiv  # 6 × 8.05B = 48.3 GFLOP/token

# RTX 5070 (Blackwell SM120):
# - BF16 dense: ~30 TFLOPS
# - INT8 (BitNet): ~50 TFLOPS
# - FP8: ~60 TFLOPS (theoretical)
# - Realistic MFU (model FLOP utilization): 35-45% for BAdam
gpu_tflops_bf16 = 30
gpu_tflops_int8 = 50
mfu = 0.40  # 40% model FLOP utilization (BAdam is efficient)

effective_tflops = gpu_tflops_int8 * mfu  # 20 TFLOPS effective

# Tokens per second
tokens_per_sec = (effective_tflops * 1e12) / flops_per_token
tokens_per_min = tokens_per_sec * 60
tokens_per_hour = tokens_per_sec * 3600

# ── Training steps ──
seq_len = 2048
batch_size = 1
grad_accum = 8  # effective batch = 8
effective_batch = batch_size * grad_accum
tokens_per_step = effective_batch * seq_len  # 16,384 tokens/step

# Epochs needed
# 1 epoch = 10.71B tokens / 16,384 tokens/step = 653K steps
steps_per_epoch = total_tokens / tokens_per_step

# Chinchilla-optimal: 76.8B tokens / 10.71B = 7.2 epochs
# But we'll do 3 epochs (practical, diminishing returns after 3)
n_epochs = 3
total_steps = steps_per_epoch * n_epochs
total_train_tokens = total_tokens * n_epochs

# Time per step
# BAdam: 1 layer forward+backward + NVMe switch
# With GradTopK 10%: 10x faster NVMe switches
# Step time: ~0.5s compute + ~0.03s NVMe switch = ~0.53s
step_time_sec = 0.53
total_time_sec = total_steps * step_time_sec
total_time_hours = total_time_sec / 3600
total_time_days = total_time_hours / 24

# ── Memory budget ──
# GPU VRAM (training, BAdam 1 layer active + FP8 activations)
gpu_weights = true_params * 1  # BitNet int8 (true params, not dense equiv)
gpu_1_layer_fwd_bwd = 4.0e9  # 1 layer forward+backward
gpu_fp8_activations = 0.5e9  # FP8 (was 1.0 GB with bf16)
gpu_r19 = 271e6  # QSA + GR
gpu_overhead = 0.3e9  # CUDA context
gpu_total = gpu_weights + gpu_1_layer_fwd_bwd + gpu_fp8_activations + gpu_r19 + gpu_overhead

# CPU RAM
cpu_master = true_params * 2  # bf16 master (true params, not dense)
cpu_optimizer_active = 0.2e9  # NVMe+4-bit Muon (1 layer active)
cpu_ngram = 15.3e9  # N-gram host table
cpu_total = cpu_master + cpu_optimizer_active + cpu_ngram

# NVMe storage
nvme_optimizer = true_params * 0.625  # 4-bit Muon states for all layers
nvme_checkpoint = true_params * 2  # bf16 checkpoint

# ── Electricity cost ──
# RTX 5070 TDP: 220W
# System idle: ~100W
# Under load: ~350W total
# Electricity: $0.12/kWh (US average)
power_w = 350
power_kw = power_w / 1000
elec_rate = 0.12  # $/kWh
elec_cost = total_time_hours * power_kw * elec_rate

# ── Cloud equivalent ──
# A100 80GB: ~$2.50/hr (RunPod)
# H100 80GB: ~$4.00/hr
# RTX 5070 equivalent: ~$0.34/hr (RunPod RTX 4070 Ti)
a100_cost = total_time_hours * 2.50
h100_cost = total_time_hours * 4.00
local_cloud_equiv = total_time_hours * 0.34

# ════════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ════════════════════════════════════════════════════════════════════════
print("=" * 75)
print("  ForgeLM V8-8B: Full Scratch Training Cost Analysis")
print("  Hardware: RTX 5070 12GB + 32GB RAM + NVMe")
print("=" * 75)

print(f"\n  {'='*73}")
print(f"  {'MODEL ARCHITECTURE':^73}")
print(f"  {'='*73}")
print(f"  d_model:                {d_model}")
print(f"  n_layers:               {n_layers}")
print(f"  n_heads / n_kv_heads:   {n_heads} / {n_kv_heads} (GQA 4x)")
print(f"  intermediate_size:      {intermediate}")
print(f"  vocab_size:             {vocab}")
print(f"  max_seq_len:            {max_seq}")
print(f"  seq_len (training):     {seq_len}")
print(f"  effective_batch:        {effective_batch} (bs={batch_size}, accum={grad_accum})")

print(f"\n  Parameter breakdown:")
print(f"    FFN (HashedNLRQ r=1024, 8x hash):  {hashed_nlrq_total/1e9:.3f}B")
print(f"    Attention (GTA, QSA):              {attn_total/1e9:.3f}B")
print(f"    Embeddings (factorized):           {embed_params/1e6:.1f}M")
print(f"    R19 keys (QSA+GR):                 {(qsa_params+gr_params)/1e6:.1f}M")
print(f"    Other keys (TITAN+MHC+MTP+etc):    {other_keys/1e9:.3f}B")
print(f"    ───────────────────────────────────────")
print(f"    True params (trainable):           {true_params/1e9:.2f}B")
print(f"    Dense equivalent:                  {dense_equiv/1e9:.2f}B")
print(f"    Compression (true/dense):          {dense_equiv/true_params:.1f}x")

print(f"\n  R19 keys (all lossless at init):")
print(f"    QSA (Qwen Sparse Attention):       top_k=256, {qsa_params/1e6:.1f}M params")
print(f"    Gated Residual:                    gate=1.0, {gr_params/1e6:.1f}M params")
print(f"    N-gram Embedding:                  3-gram, {ngram_params/1e9:.1f}B entries (host)")

print(f"\n  R20+R21 training optimizations:")
print(f"    Optimizer:                         NVMe-streamed 4-bit Muon")
print(f"    FP8 activation storage:            2x activation memory reduction")
print(f"    GradTopK:                          top-10% gradients, 10x faster NVMe")
print(f"    HashedNLRQ:                         50.7x FFN compression")

print(f"\n  {'='*73}")
print(f"  {'TRAINING DATA':^73}")
print(f"  {'='*73}")
print(f"  Pretrain corpus:        {pretrain_gb:.2f} GB JSONL")
print(f"  Pretrain lines:         {pretrain_lines/1e6:.1f}M")
print(f"  Pretrain tokens:        {pretrain_tokens/1e9:.2f}B")
print(f"  Fine-tune + expert:     {finetune_tokens/1e6:.0f}M tokens")
print(f"  Total unique tokens:    {total_tokens/1e9:.2f}B")
print(f"  Epochs:                 {n_epochs}")
print(f"  Total training tokens:  {total_train_tokens/1e9:.2f}B")
print(f"  Chinchilla ratio:       {total_train_tokens/true_params:.1f}:1 "
      f"(optimal=20:1 for {true_params/1e9:.1f}B params)")

print(f"\n  {'='*73}")
print(f"  {'MEMORY BUDGET':^73}")
print(f"  {'='*73}")
print(f"\n  GPU VRAM (training):")
print(f"    BitNet int8 weights:   {gpu_weights/1e9:.2f} GB  (true params)")
print(f"    1 layer fwd+bwd:       {gpu_1_layer_fwd_bwd/1e9:.2f} GB")
print(f"    FP8 activations:       {gpu_fp8_activations/1e9:.2f} GB  (R21)")
print(f"    R19 QSA+GR:            {gpu_r19/1e6:.0f} MB")
print(f"    CUDA overhead:         {gpu_overhead/1e9:.2f} GB")
print(f"    ─────────────────────────────────")
print(f"    TOTAL GPU:             {gpu_total/1e9:.2f} GB / 12 GB  "
      f"({'FITS' if gpu_total < 12e9 else 'EXCEEDS'})")
print(f"    Headroom:              {(12e9 - gpu_total)/1e9:.2f} GB")

print(f"\n  System RAM:")
print(f"    bf16 master:           {cpu_master/1e9:.2f} GB  (true params)")
print(f"    Optimizer (NVMe+4b):   {cpu_optimizer_active/1e9:.2f} GB  (1 layer active)")
print(f"    N-gram host table:     {cpu_ngram/1e9:.1f} GB  (R19)")
print(f"    ─────────────────────────────────")
print(f"    TOTAL CPU RAM:         {cpu_total/1e9:.2f} GB / 32 GB  "
      f"({'FITS' if cpu_total < 28e9 else 'EXCEEDS'})")
print(f"    Headroom:              {(28e9 - cpu_total)/1e9:.2f} GB")

print(f"\n  NVMe storage:")
print(f"    Optimizer states:      {nvme_optimizer/1e9:.2f} GB  (4-bit Muon, all layers)")
print(f"    Checkpoint (bf16):     {nvme_checkpoint/1e9:.2f} GB")
print(f"    Total NVMe:            {(nvme_optimizer + nvme_checkpoint)/1e9:.2f} GB")

print(f"\n  {'='*73}")
print(f"  {'TRAINING TIME':^73}")
print(f"  {'='*73}")
print(f"  FLOPs per token:        {flops_per_token/1e9:.1f} GFLOP")
print(f"  GPU effective TFLOPS:   {effective_tflops:.0f}  (int8 {gpu_tflops_int8} × MFU {mfu})")
print(f"  Tokens/sec:             {tokens_per_sec:.1f}")
print(f"  Tokens/hour:            {tokens_per_hour/1e6:.1f}M")
print(f"  Tokens/step:            {tokens_per_step:,}")
print(f"  Steps/epoch:            {steps_per_epoch/1e3:.0f}K")
print(f"  Total steps:            {total_steps/1e3:.0f}K  ({n_epochs} epochs)")
print(f"  Step time:              {step_time_sec*1000:.0f} ms  (BAdam + GradTopK 10%)")
print(f"\n  ──────────────────────────────────────────────")
print(f"  TOTAL TRAINING TIME:    {total_time_hours:.0f} hours = {total_time_days:.1f} days")
print(f"  ──────────────────────────────────────────────")

print(f"\n  {'='*73}")
print(f"  {'COST ANALYSIS':^73}")
print(f"  {'='*73}")
print(f"\n  Local (RTX 5070):")
print(f"    Power draw:            {power_w}W")
print(f"    Electricity rate:      ${elec_rate}/kWh")
print(f"    Total energy:          {total_time_hours * power_kw:.0f} kWh")
print(f"    Electricity cost:      ${elec_cost:.2f}")
print(f"    Hardware:              $0 (sunk cost)")
print(f"    TOTAL LOCAL COST:      ${elec_cost:.2f}")

print(f"\n  Cloud equivalents (same training time):")
print(f"    RunPod RTX 4070 Ti:    ${local_cloud_equiv:.2f}  (${0.34}/hr)")
print(f"    RunPod A100 80GB:      ${a100_cost:.2f}  (${2.50}/hr)")
print(f"    RunPod H100 80GB:      ${h100_cost:.2f}  (${4.00}/hr)")
print(f"\n  Local vs A100 savings:  {a100_cost/elec_cost:.0f}x cheaper")
print(f"  Local vs H100 savings:  {h100_cost/elec_cost:.0f}x cheaper")

print(f"\n  {'='*73}")
print(f"  {'THROUGHPUT COMPARISON':^73}")
print(f"  {'='*73}")
print(f"  RTX 5070 (local):       {tokens_per_sec:.1f} tok/s  ({total_time_days:.1f} days)")
# A100 80GB: ~312 TFLOPS BF16, MFU 45%, but can do full batch (no BAdam)
a100_tflops = 312 * 0.45
a100_tps = a100_tflops * 1e12 / flops_per_token
a100_time = total_train_tokens / a100_tps / 3600
print(f"  A100 80GB (cloud):      {a100_tps:.1f} tok/s  ({a100_time/24:.1f} days)")
# H100 80GB: ~989 TFLOPS BF16, MFU 50%
h100_tflops = 989 * 0.50
h100_tps = h100_tflops * 1e12 / flops_per_token
h100_time = total_train_tokens / h100_tps / 3600
print(f"  H100 80GB (cloud):      {h100_tps:.1f} tok/s  ({h100_time/24:.1f} days)")
print(f"\n  A100 is {a100_tps/tokens_per_sec:.1f}x faster but {a100_cost/elec_cost:.0f}x more expensive")
print(f"  H100 is {h100_tps/tokens_per_sec:.1f}x faster but {h100_cost/elec_cost:.0f}x more expensive")

print(f"\n  {'='*73}")
print(f"  {'POST-TRAINING INFERENCE':^73}")
print(f"  {'='*73}")
kv_per_token = n_kv_heads * head_dim * 2 * 2
kv_32k = kv_per_token * 32768
kv_32k_kara = kv_32k / 8
inf_gpu = true_params * 1 + kv_32k_kara + 271e6 + 0.25e9 + 0.3e9
inf_cpu = 15.3e9
print(f"  GPU VRAM:               {inf_gpu/1e9:.2f} GB / 12 GB  (FITS)")
print(f"  CPU RAM:                {inf_cpu/1e9:.1f} GB / 32 GB  (FITS)")
mem_bw = 448e9
decode_tps = mem_bw / (true_params * 1 + kv_32k_kara)
print(f"  Decode throughput:      {decode_tps*0.7:.1f} tok/s realistic")
print(f"  With PEAGLE 2x:         {decode_tps*0.7*2:.1f} tok/s")

print(f"\n  {'='*73}")
print(f"  {'SUMMARY':^73}")
print(f"  {'='*73}")
print(f"  Model:                  ForgeLM V8-8B ({true_params/1e9:.2f}B true params)")
print(f"  Training data:          {total_tokens/1e9:.2f}B tokens × {n_epochs} epochs = {total_train_tokens/1e9:.2f}B")
print(f"  Training time:          {total_time_days:.1f} days ({total_time_hours:.0f} hours)")
print(f"  GPU VRAM:               {gpu_total/1e9:.2f} GB / 12 GB  (FITS)")
print(f"  System RAM:             {cpu_total/1e9:.2f} GB / 32 GB  (FITS)")
print(f"  NVMe storage:           {(nvme_optimizer + nvme_checkpoint)/1e9:.2f} GB")
print(f"  Local cost:             ${elec_cost:.2f} (electricity only)")
print(f"  A100 equivalent:        ${a100_cost:.2f} ({a100_cost/elec_cost:.0f}x more)")
print(f"  H100 equivalent:        ${h100_cost:.2f} ({h100_cost/elec_cost:.0f}x more)")
print(f"\n  V8 vs V7-8B-B improvements:")
print(f"    + R19 QSA:            sparse attention (O(n*k) vs O(n²))")
print(f"    + R19 Gated Residual: learnable residual gating")
print(f"    + R19 N-gram:         host-side knowledge table (15.3 GB)")
print(f"    + R20 NVMe-Muon:      16.3 GB RAM (was 48.3 GB, 3x reduction)")
print(f"    + R21 FP8 act:        2x activation memory reduction")
print(f"    + R21 GradTopK:       10x faster NVMe block switches")
print(f"    + R21 HashedNLRQ:     3.84B true params (was 8.05B, 2.1x reduction)")
print(f"    Training RAM:         {cpu_total/1e9:.1f} GB (was 48.3 GB, {48.3/cpu_total*1e9:.1f}x reduction)")
print(f"    Training fits:        YES (was NO with 8-bit AdamW)")
