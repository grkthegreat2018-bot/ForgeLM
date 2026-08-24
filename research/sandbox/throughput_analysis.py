"""Analyze training throughput bottlenecks and compute theoretical limits.

Current: ~67-97 tok/s on V7-8B (2.8B params, BAdam, seq_len=512, bf16)
Goal: Identify what's limiting us and what would maximize tok/s
"""
import math

# ── Hardware ──
# RTX 5070 (Blackwell SM120)
# Source: NVIDIA specs — 4608 CUDA cores, ~2.5 GHz boost
# BF16 tensor cores: ~96 TFLOPS (with sparsity ~192 TFLOPS)
# FP8 tensor cores: ~193 TFLOPS (with sparsity ~386 TFLOPS)
# FP4 tensor cores: ~386 TFLOPS (with sparsity ~772 TFLOPS)
# Memory bandwidth: 288 GB/s (GDDR6, 12GB)
BF16_TFLOPS = 96.0
FP8_TFLOPS = 193.0
FP4_TFLOPS = 386.0
MEM_BW_GBPS = 288.0

# ── Model ──
d_model = 4096
n_layers = 32
intermediate = 16384
rank = 1024  # NLRQ rank for 8B-B
vocab = 65536
seq_len = 512
batch = 1

# ── FLOP analysis ──
# Per token, per layer:
# 1. Attention QKV projection: 4 * d_model * d_model = 4 * 4096^2 = 67M FLOP
# 2. Attention computation: seq_len * d_model * 2 (QK^T + softmax * V) — but this is per-token
#    Per token: 2 * d_model * head_dim * n_heads = 2 * d_model^2 / n_heads * n_heads = 2 * d_model^2 = 33.5M
#    Actually: per token attention = 2 * seq_len * d_model = 2 * 512 * 4096 = 4.2M (but this is per-token average)
#    More precisely: QK^T = seq * d_model, softmax * V = seq * d_model → 2 * seq * d_model per token
#    With GQA (16 KV heads): KV projection = 2 * d_model * (d_model * 16/64) = 2 * d_model * 1024 = 8.4M
# 3. FFN (NLRQ factorized): 
#    Dense SwiGLU: 3 * d_model * intermediate = 3 * 4096 * 16384 = 201M FLOP
#    NLRQ factorized: 3 * (d_model * rank + rank * intermediate) = 3 * (4096*1024 + 1024*16384)
#                   = 3 * (4.2M + 16.8M) = 3 * 21M = 63M FLOP
# 4. LayerNorm: ~2 * d_model = 8K (negligible)

# Attention per token (amortized over seq_len):
# QKV proj: 4 * d_model^2 / token (Q,K,V,O) but GQA: Q=d^2, K,V = 2*d*d_kv, O=d^2
# With GQA 4x: Q = d^2 = 16.8M, KV = 2 * d * (d/4) = 8.4M, O = d^2 = 16.8M → total 42M
attn_proj_flop = d_model * d_model + 2 * d_model * (d_model * 16 // 64) + d_model * d_model
# Attention score + context: 2 * seq_len * d_model per token
attn_score_flop = 2 * seq_len * d_model

# FFN NLRQ: 3 projections (gate, up, down), each factorized
# gate/up: (d_model → rank → intermediate) = d_model*rank + rank*intermediate
# down: (intermediate → rank → d_model) = intermediate*rank + rank*d_model
ffn_factorized_flop = 3 * (d_model * rank + rank * intermediate)
ffn_dense_flop = 3 * d_model * intermediate  # for comparison

# Embedding: vocab * d_model (only first token, amortized ~0)
# Head: d_model * vocab (per token)
head_flop = d_model * vocab

per_token_flop = (attn_proj_flop + attn_score_flop + ffn_factorized_flop + head_flop) * n_layers
per_token_dense_flop = (attn_proj_flop + attn_score_flop + ffn_dense_flop + head_flop) * n_layers

print("=" * 60)
print("  TRAINING THROUGHPUT ANALYSIS — V7-8B on RTX 5070")
print("=" * 60)

print(f"\n── Per-Token FLOPs (per forward pass) ──")
print(f"  Attention proj (GQA 4x): {attn_proj_flop/1e6:.1f}M")
print(f"  Attention scores:        {attn_score_flop/1e6:.1f}M")
print(f"  FFN (NLRQ factorized):   {ffn_factorized_flop/1e6:.1f}M")
print(f"  FFN (dense, for ref):    {ffn_dense_flop/1e6:.1f}M")
print(f"  LM head:                 {head_flop/1e6:.1f}M")
print(f"  Per layer total:         {(attn_proj_flop + attn_score_flop + ffn_factorized_flop + head_flop)/1e6:.1f}M")
print(f"  Full model ({n_layers} layers): {per_token_flop/1e9:.2f} GFLOP/token (NLRQ)")
print(f"  Full model (dense):      {per_token_dense_flop/1e9:.2f} GFLOP/token")
print(f"  NLRQ saves:              {per_token_dense_flop/per_token_flop:.1f}x FLOPs")

# Training = 3x forward (forward + backward)
# With gradient checkpointing: +1x forward recompute = 4x total
# With BAdam: backward still flows through all layers (grad w.r.t. inputs needed)
train_flop_per_token = per_token_flop * 4  # fwd + bwd + checkpoint recompute

print(f"\n── Training FLOPs (fwd + bwd + checkpoint recompute) ──")
print(f"  Per token: {train_flop_per_token/1e9:.2f} GFLOP")
print(f"  Per step ({seq_len} tokens): {train_flop_per_token * seq_len / 1e12:.2f} TFLOP")

# ── Theoretical throughput ──
print(f"\n── Theoretical Throughput ──")
for precision, tflops in [("bf16", BF16_TFLOPS), ("fp8", FP8_TFLOPS), ("fp4", FP4_TFLOPS)]:
    # Assume 50% MFU (model FLOP utilization) — typical for small models on consumer GPUs
    # Small seq_len=512 means poor tensor core utilization
    mfu = 0.30 if precision == "bf16" else 0.20  # fp8/fp4 have overhead
    achievable_tflops = tflops * mfu
    tok_s = achievable_tflops * 1e12 / train_flop_per_token
    time_per_step = seq_len / tok_s
    print(f"  {precision.upper()} ({tflops:.0f} TFLOPS, {mfu*100:.0f}% MFU): "
          f"{tok_s:.0f} tok/s → {time_per_step:.1f}s/step")

# ── Current vs theoretical ──
current_tok_s = 80  # observed ~67-97
print(f"\n── Current vs Theoretical ──")
print(f"  Current: {current_tok_s} tok/s")
print(f"  BF16 theoretical (30% MFU): {BF16_TFLOPS * 0.3 * 1e12 / train_flop_per_token:.0f} tok/s")
print(f"  Gap: {BF16_TFLOPS * 0.3 * 1e12 / train_flop_per_token / current_tok_s:.1f}x")

# ── Memory bandwidth analysis ──
print(f"\n── Memory Bandwidth Analysis ──")
# Per step, we need to:
# 1. Read model weights: 8 GB (NLRQ INT8 + bf16 params)
# 2. Read/write gradients for active block: ~0.2 GB
# 3. Read/write optimizer states for active block: ~0.4 GB (m + v for 1 layer)
# 4. Activations: with checkpointing, ~seq_len * d_model * 2 bytes per checkpoint
# 5. Read input tokens: negligible
weights_gb = 7.98
active_block_gb = (d_model * d_model * 4 + 3 * (d_model * rank + rank * intermediate)) * 12 / 1e9
# With checkpointing, we recompute → read weights twice
total_read_gb = weights_gb * 2  # forward + checkpoint recompute
total_write_gb = active_block_gb * 2  # grads + optimizer
total_mem_gb = total_read_gb + total_write_gb

bw_limited_tok_s = MEM_BW_GBPS / (total_mem_gb / seq_len)
print(f"  Weights (read 2x for checkpoint): {total_read_gb:.1f} GB")
print(f"  Active block (grads+opt):         {total_write_gb:.1f} GB")
print(f"  Total mem per step:               {total_mem_gb:.1f} GB")
print(f"  BW-limited throughput:            {bw_limited_tok_s:.0f} tok/s")
print(f"  Current:                          {current_tok_s} tok/s")
print(f"  → {'BW-bound' if bw_limited_tok_s < BF16_TFLOPS * 0.3 * 1e12 / train_flop_per_token else 'compute-bound'}")

# ── Bottleneck analysis ──
print(f"\n── Bottleneck Analysis ──")
compute_limit = BF16_TFLOPS * 0.3 * 1e12 / train_flop_per_token
bw_limit = bw_limited_tok_s
print(f"  Compute limit (bf16, 30% MFU): {compute_limit:.0f} tok/s")
print(f"  Memory BW limit:               {bw_limit:.0f} tok/s")
print(f"  Current:                       {current_tok_s} tok/s")
print(f"  → We are {min(compute_limit, bw_limit) / current_tok_s:.1f}x below the theoretical limit")

# ── Optimization opportunities ──
print(f"\n{'='*60}")
print(f"  OPTIMIZATION OPPORTUNITIES (ranked by impact)")
print(f"{'='*60}")

opts = [
    ("Larger seq_len (512→2048)", "3.5x better tensor core utilization, same FLOP/token", "3.5x", "Need activation offload or more VRAM"),
    ("Disable gradient checkpointing", "Eliminates 1x forward recompute (4x→3x FLOP)", "1.33x", "Need more VRAM for activations"),
    ("FP8 training (Blackwell)", "2x tensor core throughput", "2x", "Need fp8 autocast + scaling"),
    ("Triton fused NLRQ kernel", "1 kernel instead of 2 matmuls + dequant", "1.2x", "Need Triton implementation"),
    ("BAdam: skip backward for frozen layers", "Backward only through active block", "1.5-2x", "Custom autograd needed"),
    ("torch.compile", "Fuses elementwise ops, reduces launch overhead", "1.3x", "First-call compile time"),
    ("CPU-GPU overlap (BAdam)", "Optimizer step on CPU while GPU does next fwd", "1.2x", "Need async CPU AdamW"),
    ("SwiGLU gate+up fusion", "1 matmul instead of 2 for SwiGLU", "1.1x", "Already have use_fused_gemm?"),
    ("Activation CPU offload", "Enables larger seq_len without OOM", "enabler", "PCIe bandwidth overhead"),
    ("FP4 training (Blackwell)", "4x tensor core throughput", "4x", "Quality loss, needs research"),
]

for i, (name, benefit, speedup, blocker) in enumerate(opts, 1):
    print(f"  {i}. {name}")
    print(f"     Benefit: {benefit} → ~{speedup} speedup")
    print(f"     Blocker: {blocker}")
    print()

# ── Combined theoretical max ──
print(f"── Combined Optimization Stack ──")
# Start from current
combined = current_tok_s
factors = [
    ("seq_len 512→2048", 3.5),
    ("No checkpointing", 1.33),
    ("FP8 training", 2.0),
    ("Triton NLRQ fusion", 1.2),
    ("torch.compile", 1.3),
    ("CPU-GPU overlap", 1.2),
]
for name, factor in factors:
    combined *= factor
    print(f"  After {name}: {combined:.0f} tok/s")
print(f"\n  Theoretical max with all opts: {combined:.0f} tok/s ({combined/current_tok_s:.0f}x current)")
print(f"  But: some opts are mutually exclusive or have diminishing returns")
print(f"  Realistic target: {current_tok_s * 5:.0f}-{current_tok_s * 10:.0f} tok/s")
