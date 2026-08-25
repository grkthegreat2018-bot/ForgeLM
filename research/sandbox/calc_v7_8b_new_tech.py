"""V7-8B-B memory stats: current vs new improvements.
Shows both 'training mode' (bf16 attention, BitNet fp32) and 'inference mode' (BitNet int8).
"""
import math

D = 4096; L = 32; V = 65536; H = 64; KV = 16; HD = D // H; INT = 16384

def fmt(b):
    if b >= 1e9: return f"{b/1e9:.2f} GB"
    if b >= 1e6: return f"{b/1e6:.1f} MB"
    return f"{b:.0f} B"

def nlrq_proj(out_f, in_f, rank, bits, residual, hadamard):
    bpf = bits / 8
    t = int(out_f * rank * bpf) + int(rank * in_f * bpf)
    t += rank * 2 + out_f * 2 + in_f * 2  # S + scales
    if hadamard: t += rank * rank * 4  # H_U + H_V (bf16)
    if residual:
        t += int(out_f * in_f * 0.5) + out_f * (in_f // 128) * 2
    return t

def ffn_bytes(rank, bits, residual, hadamard):
    per = (nlrq_proj(INT, D, rank, bits, residual, hadamard) +
           nlrq_proj(INT, D, rank, bits, residual, hadamard) +
           nlrq_proj(D, INT, rank, bits, residual, hadamard))
    return per * L

def attn_bytes(mode):
    """mode: 'bf16_nogqa' (old calc), 'bf16_gqa', 'int8_gqa' (BitNet)"""
    if mode == 'bf16_nogqa':
        return D * D * 4 * L * 2  # Q,K,V,O all d×d, bf16
    bpp = 2 if 'bf16' in mode else 1
    per = D * D + D * KV * HD + D * KV * HD + D * D  # Q, K, V, O with GQA
    return per * L * bpp

def embed_bytes(int8=False):
    bpp = 1 if int8 else 2
    return (V * 512 + 512 * D) * bpp  # factorized, tied

def other_bytes(int8=False):
    bpp = 1 if int8 else 2
    return L * (D * 4 + 128 * D + 1024 * D + D * 4) * bpp  # norms, TITAN, mHC, AttnRes

def peagle_bytes(tied=True):
    if tied: return 7 * (32 * D + D * 32) * 2  # LoRA only
    return 7 * D * V * 2

def kv_bytes(seq=32768, comp="rotorquant"):
    base = 2 * seq * KV * HD * L * 2
    return base / {"rotorquant": 15, "hadamard_int4": 4, "paged_evict": 2, "none": 1}[comp]

def badam_bytes(model, n_blocks=32):
    block = model / n_blocks
    return model + block * 4 + block * 2  # model + 1 block opt (fp32 m+v) + 1 block grads (bf16)

def badam_int4(model, n_blocks=32):
    block = model / n_blocks
    return model + block * 4 + block * 1  # model + opt + int4 grads (0.5x)

print("=" * 78)
print("ForgeLM V7-8B-B — Memory Stats with New R&D Improvements")
print(f"d={D}, L={L}, V={V}, H={H}, KV={KV}, INT={INT}, head_dim={HD}")
print("=" * 78)

scenarios = [
    ("CURRENT: INT8 NLRQ r=1024", 1024, 8, False, False),
    ("NEW: HINT4 r=1024 (no res)", 1024, 4, False, True),
    ("NEW: HINT4+Res r=1024", 1024, 4, True, True),
    ("NEW: HINT4+Res r=768", 768, 4, True, True),
    ("NEW: HINT4 r=768 (no res)", 768, 4, False, True),
]

# Dense baseline
dense = ffn_bytes(1024, 16, False, False) + attn_bytes('bf16_gqa') + embed_bytes() + other_bytes()
print(f"\nDense baseline (bf16, GQA): {fmt(dense)}")
print(f"Logical params: ~{(3 * L * (INT * 1024 + 1024 * D) + L * (2*D*D + 2*D*KV*HD) + V*512 + 512*D) / 1e9:.1f}B (FFN+attn+embed, rank-1024 basis)")

for name, rank, bits, res, had in scenarios:
    print(f"\n{'─' * 78}")
    print(f"  {name}")
    print(f"{'─' * 78}")

    for mode_name, attn_mode, attn_int8 in [("Inference (BitNet int8 + GQA)", 'int8_gqa', True),
                                              ("Training (bf16, BitNet fp32)", 'bf16_gqa', False)]:
        ffn = ffn_bytes(rank, bits, res, had)
        attn = attn_bytes(attn_mode)
        emb = embed_bytes(attn_int8)
        oth = other_bytes(attn_int8)
        model = ffn + attn + emb + oth

        if "Inference" in mode_name:
            peagle = peagle_bytes(tied=True)
            kv = kv_bytes(32768, "rotorquant")
            total = model + kv + peagle
            print(f"  [{mode_name}]")
            print(f"    FFN:          {fmt(ffn)}  (r={rank}, {bits}bit, Had={had}, Res={res})")
            print(f"    Attention:    {fmt(attn)}")
            print(f"    Embed+Other:  {fmt(emb + oth)}")
            print(f"    Model total:  {fmt(model)}")
            print(f"    KV (32K):     {fmt(kv)}  (RotorQuant 15x)")
            print(f"    PEAGLE:       {fmt(peagle)}  (tied LoRA)")
            print(f"    INFERENCE:    {fmt(total)}  → {'FITS 12GB' if total < 12e9 else 'OOM'}")
        else:
            train = badam_int4(model)
            print(f"  [{mode_name}]")
            print(f"    Model:        {fmt(model)}")
            print(f"    BAdam+int4:   {fmt(train)}  → {'FITS 12GB' if train < 12e9 else 'OOM'}")

    # Checkpoint file size (inference mode: smallest)
    ckpt = ffn_bytes(rank, bits, res, had) + attn_bytes('int8_gqa') + embed_bytes(True) + other_bytes(True)
    print(f"  Checkpoint:     {fmt(ckpt)}")
    cr = dense / (ffn_bytes(rank, bits, res, had) + attn_bytes('int8_gqa') + embed_bytes(True) + other_bytes(True))
    print(f"  Compression:    {cr:.1f}x vs dense")

# Error comparison
print(f"\n{'═' * 78}")
print("RECONSTRUCTION ERROR (from tests):")
print("  INT8 NLRQ (current):     ~1.3%")
print("  HINT4 (no residual):     ~18.7%")
print("  HINT4 + INT4 residual:   ~2.2%  (8.4x better than pure HINT4)")
print()
print("RECOMMENDATION:")
print("  • HINT4 r=1024 (no res):  2.96 GB model, 3.25 GB infer — BEST memory, 18% error")
print("  • HINT4+Res r=768:        5.86 GB model, 6.15 GB infer — BEST quality, 2.2% error")
print("  • HINT4+Res r=1024:       6.28 GB model, 6.57 GB infer — max quality, still <12GB")
print("  • Current INT8 r=1024:    3.57 GB model, 3.86 GB infer — baseline, 1.3% error")
print()
print("  For V7-8B-B on RTX 5070: HINT4 r=1024 (no res) saves 0.61 GB with 18% FFN error.")
print("  If quality matters: HINT4+Res r=768 gives 2.2% error at 5.86 GB (still fits 12GB).")
