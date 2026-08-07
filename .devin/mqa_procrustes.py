"""Procrustes-aligned GQA→MQA conversion.

Instead of naive mean-pooling (which destroys information when heads are
orthogonal), we:
1. Run calibration data through the original model
2. Collect K/V cache outputs for each KV head
3. Find orthogonal Q that aligns head 1 → head 0 (Procrustes: Q = UV^T from SVD)
4. Transform head 1's weights: W1' = Q @ W1
5. Mean-pool the aligned heads: W_mqa = (W0 + W1') / 2

For V: the transformation is computationally invariant — we also rotate the
corresponding O projection rows by Q^T, preserving the output exactly.

For K: the transformation changes attention scores, but makes the pooled K
more representative of both original heads. RoPE compatibility requires Q
to be block-diagonal (2x2 rotation blocks), but we use the full Procrustes
solution and accept minor RoPE distortion.

Reference: "Align Attention Heads Before Merging Them" (Jin et al., 2024)
"""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from safetensors import safe_open
from safetensors.torch import save_file
from research.config import get_config
from research.model_loader import ModelLoader

SRC = 'research/checkpoints/qwen25_coder_1.5b_ported.safetensors'
OUT = 'research/checkpoints/xp_mqa_aligned.safetensors'

# --- Step 1: Collect KV caches from calibration data ---
print("Step 1: Collecting KV caches from calibration data...")
import numpy as np
train_np = np.memmap("research/data/train.bin", dtype=np.uint16, mode="r")

m_orig = ModelLoader.build_model(
    get_config('qwen25_coder_1.5b', device='cuda'),
    checkpoint_path=SRC
).to('cuda', dtype=torch.bfloat16).eval()

N_CALIB = 64
SEQ_LEN = 128
n_layers = 28
n_kv = 2
head_dim = 128
d_model = 1536

# Collect KV caches per layer per KV head
# k_cache[layer] = [n_calib, n_kv, seq_len, head_dim]
k_caches = {l: [] for l in range(n_layers)}
v_caches = {l: [] for l in range(n_layers)}

with torch.inference_mode():
    for i in range(N_CALIB):
        idx = np.random.randint(0, len(train_np) - SEQ_LEN)
        tokens = train_np[idx:idx + SEQ_LEN].astype(np.int64)
        ids = torch.tensor(tokens, device='cuda').unsqueeze(0)

        # Run through model, collecting KV per layer
        x = m_orig.embed(ids)
        for li, block in enumerate(m_orig.blocks):
            attn = block.attn
            B, T, C = x.shape
            q = attn.q_proj(x).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)

            # Apply RoPE (same as model does)
            q = attn.rope(q)
            k = attn.rope(k)

            # k, v shape: [B, n_kv, T, head_dim]
            k_caches[li].append(k[0].cpu().float())  # [n_kv, T, head_dim]
            v_caches[li].append(v[0].cpu().float())

            # Forward through full block
            x = block(x)
            if isinstance(x, tuple):
                x = x[0]

        if (i + 1) % 16 == 0:
            print(f"  Calibrated {i+1}/{N_CALIB}")

# Stack: [n_calib, n_kv, T, head_dim] -> [n_kv, n_calib*T, head_dim]
k_stacked = {}
v_stacked = {}
for li in range(n_layers):
    k_stacked[li] = torch.cat(k_caches[li], dim=1)  # [n_kv, n_calib*T, head_dim]
    v_stacked[li] = torch.cat(v_caches[li], dim=1)
    print(f"  Layer {li}: K cache {k_stacked[li].shape}, V cache {v_stacked[li].shape}")

del m_orig
torch.cuda.empty_cache()

# --- Step 2: Compute Procrustes alignment per layer ---
print("\nStep 2: Computing Procrustes alignment...")
# For each layer, find Q_k and Q_v such that head1 @ Q ≈ head0
# Procrustes: Q = UV^T where U Σ V^T = SVD(head0^T @ head1)

Q_k = {}  # Q_k[layer] = [head_dim, head_dim] orthogonal
Q_v = {}

for li in range(n_layers):
    # K alignment: want K1 @ Q_k ≈ K0
    K0 = k_stacked[li][0]  # [n_calib*T, head_dim]
    K1 = k_stacked[li][1]
    M_k = K0.T @ K1  # [head_dim, head_dim]
    U_k, S_k, Vt_k = torch.linalg.svd(M_k)
    Q_k[li] = U_k @ Vt_k  # orthogonal

    # Check alignment quality
    K1_aligned = K1 @ Q_k[li]
    cos_before = F.cosine_similarity(K0.flatten().unsqueeze(0), K1.flatten().unsqueeze(0)).item()
    cos_after = F.cosine_similarity(K0.flatten().unsqueeze(0), K1_aligned.flatten().unsqueeze(0)).item()
    rel_err_before = (K0 - K1).norm() / K0.norm()
    rel_err_after = (K0 - K1_aligned).norm() / K0.norm()

    # V alignment
    V0 = v_stacked[li][0]
    V1 = v_stacked[li][1]
    M_v = V0.T @ V1
    U_v, S_v, Vt_v = torch.linalg.svd(M_v)
    Q_v[li] = U_v @ Vt_v

    V1_aligned = V1 @ Q_v[li]
    v_cos_before = F.cosine_similarity(V0.flatten().unsqueeze(0), V1.flatten().unsqueeze(0)).item()
    v_cos_after = F.cosine_similarity(V0.flatten().unsqueeze(0), V1_aligned.flatten().unsqueeze(0)).item()

    if li < 5 or li == n_layers - 1:
        print(f"  Layer {li}: K cos {cos_before:.3f}→{cos_after:.3f}, err {rel_err_before:.3f}→{rel_err_after:.3f} | "
              f"V cos {v_cos_before:.3f}→{v_cos_after:.3f}")

# --- Step 3: Apply aligned conversion to weights ---
print("\nStep 3: Applying aligned GQA→MQA conversion...")
state = {}
with safe_open(SRC, framework='pt') as f:
    for k in f.keys():
        t = f.get_tensor(k)
        state[k] = t

        if 'k_proj' in k and 'weight' in k:
            # Parse layer index
            parts = k.split('.')
            li = int(parts[1])
            # t shape: [n_kv * head_dim, d_model] = [256, 1536]
            W = t.view(n_kv, head_dim, d_model)  # [2, 128, 1536]
            W0 = W[0]  # [128, 1536]
            W1 = W[1]  # [128, 1536]
            # Align W1 to W0: W1' = Q_k @ W1 (so K1' = Q_k @ K1 ≈ K0)
            W1_aligned = Q_k[li].to(W1.dtype) @ W1
            # Mean-pool aligned heads
            W_mqa = (W0 + W1_aligned) / 2
            state[k] = W_mqa.contiguous()

        elif 'k_proj' in k and 'bias' in k:
            parts = k.split('.')
            li = int(parts[1])
            b = t.view(n_kv, head_dim)
            b0 = b[0]
            b1 = b[1]
            b1_aligned = Q_k[li].to(b1.dtype) @ b1
            state[k] = ((b0 + b1_aligned) / 2).contiguous()

        elif 'v_proj' in k and 'weight' in k:
            parts = k.split('.')
            li = int(parts[1])
            W = t.view(n_kv, head_dim, d_model)
            W0 = W[0]
            W1 = W[1]
            W1_aligned = Q_v[li].to(W1.dtype) @ W1
            W_mqa = (W0 + W1_aligned) / 2
            state[k] = W_mqa.contiguous()

        elif 'v_proj' in k and 'bias' in k:
            parts = k.split('.')
            li = int(parts[1])
            b = t.view(n_kv, head_dim)
            b0 = b[0]
            b1 = b[1]
            b1_aligned = Q_v[li].to(b1.dtype) @ b1
            state[k] = ((b0 + b1_aligned) / 2).contiguous()

# --- Step 4: Adjust O projection for V alignment ---
# V was rotated by Q_v, so O projection rows for query heads 6-11
# (which attended to V head 1) need Q_v^T applied
# O proj weight: [d_model, d_model] = [n_heads * head_dim, d_model]
# Rows 0-5*head_dim correspond to Q heads 0-5 (attended to V head 0)
# Rows 6*head_dim-12*head_dim correspond to Q heads 6-11 (attended to V head 1)
print("  Adjusting O projection for V alignment...")
for li in range(n_layers):
    o_key = f'blocks.{li}.attn.out_proj.weight'
    if o_key in state:
        W_o = state[o_key]  # [d_model, d_model] = [1536, 1536]
        # Reshape rows: [n_heads, head_dim, d_model]
        W_o_reshaped = W_o.view(12, head_dim, d_model)
        # Apply Q_v^T to rows of heads 6-11
        Qv_T = Q_v[li].to(W_o.dtype).T
        W_o_reshaped[6:12] = Qv_T @ W_o_reshaped[6:12]
        state[o_key] = W_o_reshaped.view(d_model, d_model).contiguous()

save_file(state, OUT)
print(f"\nSaved {len(state)} tensors to {OUT}")
print("Done! Aligned MQA model ready for testing.")
