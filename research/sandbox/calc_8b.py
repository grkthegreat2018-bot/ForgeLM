"""Calculate VRAM for 8B V7 variants."""
import sys

d_model = 4096
n_layers = 32
intermediate = 16384
rank = 768
vocab = 65536

# Per-layer NLRQ FFN: 3 projections (gate, up, down)
ffn_per_proj = (intermediate * rank + rank * d_model) * 1  # INT8 = 1 byte
ffn_total = ffn_per_proj * 3
ffn_all_layers = ffn_total * n_layers

attn_per_layer = d_model * d_model * 4 * 2  # bf16
attn_all_layers = attn_per_layer * n_layers

embed = vocab * 512 * 2 + 512 * d_model * 2
other = n_layers * (d_model * 4 * 2 + 128 * d_model * 2 + 1024 * d_model * 2)

total_bytes = ffn_all_layers + attn_all_layers + embed + other

print("=== Current V7 (2.8B) ===")
print(f"FFN (NLRQ INT8): {ffn_all_layers/1e9:.2f} GB")
print(f"Attention (bf16): {attn_all_layers/1e9:.2f} GB")
print(f"Embedding: {embed/1e9:.2f} GB")
print(f"Other: {other/1e9:.2f} GB")
print(f"Total storage: {total_bytes/1e9:.2f} GB")
print()

configs = [
    ("8B-A: d=6144, L=48, r=768", 6144, 48, 768),
    ("8B-B: d=4096, L=32, r=1024", 4096, 32, 1024),
    ("8B-C: d=5120, L=40, r=768", 5120, 40, 768),
    ("8B-D: d=4096, L=48, r=768", 4096, 48, 768),
    ("8B-E: d=4096, L=64, r=512", 4096, 64, 512),
]

for label, d, nl, r in configs:
    inter = d * 4
    ffn_p = (inter * r + r * d) * 3 * nl  # INT8
    attn_p = d * d * 4 * nl * 2  # bf16
    emb = vocab * 512 * 2 + 512 * d * 2
    oth = nl * (d * 4 * 2 + 128 * d * 2 + (d // 4) * d * 2)
    tot = ffn_p + attn_p + emb + oth

    # Training memory with BAdam (1 layer active)
    active_opt = ((inter * r + r * d) * 3 + d * d * 4) * 12  # fp32 optimizer 1 layer
    active_grads = ((inter * r + r * d) * 3 + d * d * 4) * 2  # bf16 grads 1 layer
    gpu_train = tot + active_opt + active_grads

    # With CPUAdamW + grad_offload (all optimizer on CPU, grads streamed)
    gpu_offload = tot + 0.5  # model + overhead, grads+optimizer on CPU

    # Inference only
    fits_12 = "YES" if tot / 1e9 < 11 else "NO"
    fits_train_badam = "YES" if gpu_train / 1e9 < 11 else "NO"
    fits_train_offload = "YES" if gpu_offload / 1e9 < 11 else "NO"

    print(f"{label}")
    print(f"  Storage (inference):      {tot/1e9:.2f} GB  -> fits 12GB? {fits_12}")
    print(f"  BAdam training:           {gpu_train/1e9:.2f} GB  -> fits 12GB? {fits_train_badam}")
    print(f"  CPUAdamW+grad_offload:    {gpu_offload/1e9:.2f} GB  -> fits 12GB? {fits_train_offload}")
    print(f"  CPU RAM needed (offload): {(active_opt * nl + active_grads * nl)/1e9:.1f} GB")
    print()
