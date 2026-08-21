"""Parameter-Reduction Calculator for LFM2.5-1.2B base.

For each compression / parameter-reduction technique, computes:
  - exact baseline param breakdown of LFM2.5-1.2B
  - parameter savings (absolute + %)
  - final param count + bf16 disk size
  - theoretical quality-preservation rating (LOSSLESS / NEAR-LOSSLESS / LOSSY-RECOVERABLE / LOSSY)
  - notes on recovery mechanism (distillation, QAT, finetune)

Run:  python -m research.sandbox.param_reduction_calc
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

# ============================================================================
# LFM2.5-1.2B base architecture (from research/config.py preset "lfm25_1.2b")
# ============================================================================
VOCAB = 65536
D_MODEL = 2048
N_LAYERS = 16
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS          # 64
INTER = 8192                            # SwiGLU intermediate
CONV_K = 3                              # double-gated conv kernel
N_ATTN_LAYERS = 6                       # layers 2,5,8,10,12,14
N_CONV_LAYERS = N_LAYERS - N_ATTN_LAYERS  # 10
BF16 = 2                                # bytes per param
TIE_EMBED = True                        # embed == head (tied)

# Per-layer param counts ------------------------------------------------------
def conv_params() -> int:
    """Double-gated conv block (LFM2.5 style).

    Approximate: depthwise conv (D*CONV_K) + 2 gates (D*D each, in/out proj)
    + 2 RMSNorm (D) + FFN (SwiGLU 3*D*INTER/2... but conv blocks in LFM2.5
    have a smaller FFN; we use the same SwiGLU for parity with checkpoint).
    Conservative: treat conv block as 2*D*D (gated proj) + conv kernel + FFN.
    """
    # LFM2.5 conv block: gated linear unit + depthwise conv + FFN
    # We approximate using the actual checkpoint tensor inventory:
    #   in_proj (D, 2*D), out_proj (D, D), conv weight (D, 1, K), 2 norms, FFN
    in_proj = D_MODEL * (2 * D_MODEL)         # gate + up
    out_proj = D_MODEL * D_MODEL
    conv = D_MODEL * CONV_K
    norms = 2 * D_MODEL
    ffn = 3 * D_MODEL * INTER                 # SwiGLU gate/up/down (gate,up share via fused)
    return in_proj + out_proj + conv + norms + ffn

def attn_params() -> int:
    """GQA attention block + SwiGLU FFN + 2 RMSNorm + QK-norm."""
    q = D_MODEL * (N_HEADS * HEAD_DIM)        # = D*D
    k = D_MODEL * (N_KV_HEADS * HEAD_DIM)     # = D * (D/4) = D*D/4
    v = D_MODEL * (N_KV_HEADS * HEAD_DIM)
    o = (N_HEADS * HEAD_DIM) * D_MODEL        # = D*D
    qk_norm = 2 * D_MODEL                     # q_norm + k_norm (RMSNorm, per-head shared)
    norms = 2 * D_MODEL                       # post_attn + post_ffn
    ffn = 3 * D_MODEL * INTER                 # SwiGLU
    return q + k + v + o + qk_norm + norms + ffn

def embed_params(tied: bool = TIE_EMBED) -> int:
    return VOCAB * D_MODEL if tied else 2 * VOCAB * D_MODEL

# ============================================================================
# Baseline
# ============================================================================
def baseline_breakdown() -> dict:
    cp = conv_params()
    ap = attn_params()
    embed = embed_params()
    total = N_CONV_LAYERS * cp + N_ATTN_LAYERS * ap + embed
    return {
        "conv_per_layer": cp,
        "attn_per_layer": ap,
        "embedding": embed,
        "conv_total": N_CONV_LAYERS * cp,
        "attn_total": N_ATTN_LAYERS * ap,
        "total": total,
        "bf16_mb": total * BF16 / 1e6,
    }

BASE = baseline_breakdown()
print("=" * 78)
print("LFM2.5-1.2B BASELINE PARAMETER BREAKDOWN")
print("=" * 78)
for k, v in BASE.items():
    if k == "total":
        print(f"  {k:20s}: {v:>14,d}  ({v/1e6:.3f}B)")
    elif k == "bf16_mb":
        print(f"  {k:20s}: {v:>14.2f} MB")
    else:
        print(f"  {k:20s}: {v:>14,d}")
print()

# ============================================================================
# Technique registry
# ============================================================================
@dataclass
class Technique:
    name: str
    category: str
    quality: str            # LOSSLESS / NEAR-LOSSLESS / LOSSY-RECOVERABLE / LOSSY
    fn: Callable[[int, dict], dict]   # (baseline_total, breakdown) -> result
    notes: str = ""

def fmt(n: int) -> str:
    return f"{n:,d} ({n/1e6:.3f}B)"

# ----------------------------------------------------------------------------
# 1. Factorized Embedding (ALBERT) — already in codebase
# ----------------------------------------------------------------------------
def t_factorized_embed(base, b):
    rank = 256
    orig = b["embedding"]
    new = VOCAB * rank + rank * D_MODEL
    saved = orig - new
    return dict(saved=saved, final=base - saved,
                detail=f"rank={rank}: {VOCAB}x{D_MODEL} -> {VOCAB}x{rank} + {rank}x{D_MODEL}")

# ----------------------------------------------------------------------------
# 2. BitNet b1.58 ternary (int8 storage) — already in codebase
#    All linear weights -> 1 byte/param (vs 2 bf16). Param COUNT unchanged but
#    STORAGE halved. We report effective-param-equivalent for fair comparison.
# ----------------------------------------------------------------------------
def t_bitnet_storage(base, b):
    # All weights except norms/embed become 1 byte. Effective param-equiv:
    # storage_bytes / 2 (bf16 baseline). So 2x storage reduction = 0.5x equiv.
    non_embed = base - b["embedding"]
    # norms stay bf16 (tiny), embed stays bf16 unless bitnet_embedding
    new_storage_bytes = non_embed * 1 + b["embedding"] * BF16
    equiv = new_storage_bytes // BF16
    saved = base - equiv
    return dict(saved=saved, final=equiv,
                detail="linear weights -> int8 ternary (1 byte); embed stays bf16")

def t_bitnet_full(base, b):
    # BitNet on embeddings too
    new_storage_bytes = base * 1
    equiv = new_storage_bytes // BF16
    saved = base - equiv
    return dict(saved=saved, final=equiv,
                detail="ALL weights (incl embed) -> int8 ternary; 2x storage cut")

# ----------------------------------------------------------------------------
# 3. Tensor decomposition: SVD low-rank on FFN
#    W_gate (D x INTER) ≈ U (D x r) @ V (r x INTER). Params: r*(D+INTER).
# ----------------------------------------------------------------------------
def t_ffn_svd(base, b, rank_frac=0.25):
    # FFN per layer = 3 * D * INTER. SVD all 3 matrices at rank r.
    # SVD of (D x INTER) at rank r costs r*(D+INTER). Saves only when
    # r < D*INTER/(D+INTER) = 1638 here. So rank_frac is fraction of 1638.
    max_rank = (D_MODEL * INTER) // (D_MODEL + INTER)  # 1638
    r = int(max_rank * rank_frac)
    ffn_per_layer = 3 * D_MODEL * INTER
    svd_per_layer = 3 * r * (D_MODEL + INTER)
    saved_per_layer = ffn_per_layer - svd_per_layer
    saved = N_LAYERS * saved_per_layer
    return dict(saved=saved, final=base - saved,
                detail=f"FFN SVD rank={r} (frac={rank_frac} of max {max_rank}): "
                       f"{ffn_per_layer:,d} -> {svd_per_layer:,d} per layer")

# ----------------------------------------------------------------------------
# 4. Kronecker-factored FFN (KronBERT)
#    W (m x n) ≈ A (a x b) ⊗ B (c x d) where a*c=m, b*d=n.
#    Params: a*b + c*d vs m*n. For D=2048, INTER=8192: factor as
#    (32 x 64) ⊗ (64 x 128) -> 2048 x 8192. Params: 32*64 + 64*128 = 10240 vs 16.7M.
# ----------------------------------------------------------------------------
def t_kronecker_ffn(base, b):
    # W_gate: (D, INTER) = (2048, 8192). Factor (32,64) ⊗ (64,128): 32*64 + 64*128 = 10240
    # vs 2048*8192 = 16,777,216. ~1636x reduction on each weight (extreme).
    # Realistic: use (64,32) ⊗ (32,256): 64*32 + 32*256 = 10240
    # We use a moderate factorization preserving rank.
    a, bb = 64, 32      # A is (64,32)
    c, d = 32, 256      # B is (32,256)  -> product (2048, 8192)
    assert a * c == D_MODEL and bb * d == INTER
    kron_params = a * bb + c * d
    full = D_MODEL * INTER
    saved_per_matrix = full - kron_params
    saved = N_LAYERS * 3 * saved_per_matrix
    return dict(saved=saved, final=base - saved,
                detail=f"Kronecker ({a}x{bb})x({c}x{d}): {full:,d} -> {kron_params:,d} per matrix")

# ----------------------------------------------------------------------------
# 5. Tensor-Train (TT) decomposition of FFN
#    W (d1 x d2) reshaped to 4D core chain. TT-rank=4 typically.
# ----------------------------------------------------------------------------
def t_tensor_train_ffn(base, b, tt_rank=4):
    # Reshape (2048, 8192) -> (8,8,8,4) x (8,8,8,16) = 4 cores
    # Cores: (1,8,r) (r,8,r) (r,8,r) (r,16,1) with r=tt_rank
    r = tt_rank
    # Approximate: 4 cores of shape (r, d_i, r) -> 4 * r * d * r ~ 4*r^2*d
    # For 2048=8^3*4, 8192=8^3*16: cores (1,8,r)(r,8,r)(r,8,r)(r,4,1) * (1,8,r)(r,8,r)(r,8,r)(r,16,1)
    # Simplified: total TT params ~ 2 * (3 * r * r * 8 + r * 8 * 1 + r * 16 * 1) per matrix
    tt_per_matrix = 2 * (3 * r * r * 8 + r * 8 + r * 16)  # very rough
    full = D_MODEL * INTER
    saved_per_matrix = max(0, full - tt_per_matrix)
    saved = N_LAYERS * 3 * saved_per_matrix
    return dict(saved=saved, final=base - saved,
                detail=f"TT-rank={r}: ~{tt_per_matrix:,d} per matrix vs {full:,d}")

# ----------------------------------------------------------------------------
# 6. Monarch matrices (Dao et al.) — O(n sqrt(n)) param, near-dense expressivity
#    W = L @ R @ P (L,R block-diagonal, P permutation). For n=2048: blocks 32x32.
# ----------------------------------------------------------------------------
def t_monarch_ffn(base, b, block=32):
    # W (D, INTER): L is (D, D) block-diag with D/block blocks of (block,block)
    # = D*block params. R is (INTER, INTER) block-diag = INTER*block.
    # Total: D*block + INTER*block = block*(D+INTER)
    monarch_per_matrix = block * (D_MODEL + INTER)
    full = D_MODEL * INTER
    saved_per_matrix = full - monarch_per_matrix
    saved = N_LAYERS * 3 * saved_per_matrix
    return dict(saved=saved, final=base - saved,
                detail=f"Monarch block={block}: {monarch_per_matrix:,d} vs {full:,d} per matrix")

# ----------------------------------------------------------------------------
# 7. FastFood transforms for FFN (Le et al.)
#    Replaces (D, INTER) with structured Hadamard+diagonal: 2*D params per row.
# ----------------------------------------------------------------------------
def t_fastfood_ffn(base, b):
    # FastFood: ~2*D params per output dim instead of D. So per matrix: 2*D*INTER/sqrt(INTER)
    # Actually FastFood maps D->D with O(D log D) params; for D->INTER we tile.
    # Conservative: per matrix params = 2 * D_MODEL (diagonal + Hadamard scaling)
    # applied INTER/D_MODEL times -> 2 * INTER
    ff_per_matrix = 2 * INTER  # very approximate
    full = D_MODEL * INTER
    saved_per_matrix = full - ff_per_matrix
    saved = N_LAYERS * 3 * saved_per_matrix
    return dict(saved=saved, final=base - saved,
                detail=f"FastFood: ~{ff_per_matrix:,d} vs {full:,d} per matrix (aggressive)")

# ----------------------------------------------------------------------------
# 8. Structured pruning + distillation (e.g., LLM-Pruner, Sheared LLaMA)
#    Remove ~20-30% of channels/heads, recover via CPT/distillation.
# ----------------------------------------------------------------------------
def t_structured_prune(base, b, frac=0.25):
    # Prune 25% of FFN intermediate + 25% of attention heads (KV heads 8->6)
    new_inter = int(INTER * (1 - frac))
    new_kv = max(2, int(N_KV_HEADS * (1 - frac)))
    ffn_old = 3 * D_MODEL * INTER
    ffn_new = 3 * D_MODEL * new_inter
    # attention K/V shrink
    kv_old = 2 * D_MODEL * (N_KV_HEADS * HEAD_DIM)
    kv_new = 2 * D_MODEL * (new_kv * HEAD_DIM)
    saved_per_layer = (ffn_old - ffn_new) + (kv_old - kv_new)
    saved = N_LAYERS * saved_per_layer
    return dict(saved=saved, final=base - saved,
                detail=f"prune {frac*100:.0f}%: INTER {INTER}->{new_inter}, KV heads {N_KV_HEADS}->{new_kv}")

# ----------------------------------------------------------------------------
# 9. Layer dropping / progressive depth (ShortGPT, LaCoLlama)
#    Drop ~25% of layers, recover via distillation.
# ----------------------------------------------------------------------------
def t_layer_drop(base, b, drop_frac=0.25):
    n_drop = int(N_LAYERS * drop_frac)
    per_layer = (BASE["conv_per_layer"] + BASE["attn_per_layer"]) / 2  # avg
    saved = int(n_drop * per_layer)
    return dict(saved=saved, final=base - saved,
                detail=f"drop {n_drop}/{N_LAYERS} layers (~{drop_frac*100:.0f}%)")

# ----------------------------------------------------------------------------
# 10. Cross-layer weight sharing (ALBERT / Hyperloop / LiSA)
#     Share FFN+attention across all layers; keep per-layer adapters.
# ----------------------------------------------------------------------------
def t_cross_layer_share(base, b, n_unique=4):
    # Keep n_unique layer templates, share across N_LAYERS. Add per-layer
    # low-rank adapter (rank=64) for differentiation: 2 * 64 * (D + INTER) per layer.
    per_layer = (BASE["conv_per_layer"] + BASE["attn_per_layer"]) / 2
    shared = n_unique * per_layer
    adapter_per_layer = 2 * 64 * (D_MODEL + INTER)  # rough
    adapters = N_LAYERS * adapter_per_layer
    new_total = shared + adapters + b["embedding"]
    saved = base - new_total
    return dict(saved=saved, final=new_total,
                detail=f"{n_unique} shared templates + rank-64 adapter per layer")

# ----------------------------------------------------------------------------
# 11. Hyperloop Transformers — begin+middle(looped)+end, 50% fewer params
# ----------------------------------------------------------------------------
def t_hyperloop(base, b, n_begin=2, n_end=2, n_loop=4):
    # begin (unique) + middle (shared, looped n_loop times) + end (unique)
    per_layer = (BASE["conv_per_layer"] + BASE["attn_per_layer"]) / 2
    n_middle_actual = N_LAYERS - n_begin - n_end
    n_loop_iters = n_middle_actual // n_loop
    used = n_begin + n_loop + n_end  # unique layers stored
    new_total = used * per_layer + b["embedding"]
    saved = base - new_total
    return dict(saved=saved, final=new_total,
                detail=f"begin={n_begin} + loop={n_loop} (x{n_loop_iters}) + end={n_end}: "
                       f"{used} unique layers")

# ----------------------------------------------------------------------------
# 12. LiSA / cross-layer Q/K sharing — 6x Q/K compression
# ----------------------------------------------------------------------------
def t_lisa_qk(base, b, compress=6):
    # Q (D*D) and K (D*D/4) per attn layer. Share across attn layers with
    # tiny alignment FFN. 6x compression on Q+K.
    q = D_MODEL * (N_HEADS * HEAD_DIM)
    k = D_MODEL * (N_KV_HEADS * HEAD_DIM)
    per_attn = q + k
    saved_per_attn = per_attn * (1 - 1 / compress)
    # alignment FFN: ~D*D/4 per attn layer (small)
    align = D_MODEL * (D_MODEL // 4)
    saved = N_ATTN_LAYERS * (saved_per_attn - align)
    return dict(saved=saved, final=base - saved,
                detail=f"Q/K shared 6x across {N_ATTN_LAYERS} attn layers + alignment FFN")

# ----------------------------------------------------------------------------
# 13. MoE with shared expert + fine segmentation (DeepSeekMoE)
#     Active params unchanged; TOTAL params grow but per-token cost drops.
#     For "param cost" we report ACTIVE params (what matters for inference).
# ----------------------------------------------------------------------------
def t_moe_active(base, b, n_experts=8, top_k=2, expert_inter=None):
    # Active = top_k routed experts (each FFN-sized). To REDUCE active cost vs
    # dense FFN, each expert must be smaller than INTER/top_k.
    ei = expert_inter or INTER // 4  # smaller experts (2048)
    dense_ffn = 3 * D_MODEL * INTER
    routed_active = top_k * 3 * D_MODEL * ei
    saved_active_per_layer = dense_ffn - routed_active
    saved = N_LAYERS * saved_active_per_layer
    return dict(saved=saved, final=base - saved,
                detail=f"MoE top-{top_k} of {n_experts}, expert_inter={ei}: "
                       f"active FFN {dense_ffn:,d} -> {routed_active:,d}/layer")

# ----------------------------------------------------------------------------
# 14. Expert Tying (EPFL 2026) — share experts across consecutive layer groups
# ----------------------------------------------------------------------------
def t_expert_tying(base, b, group=2):
    # Applies to MoE: divides expert params by `group`. We model on a hypothetical
    # V5 MoE with 8 experts * 3 * D * (INTER//2) per layer.
    ei = INTER // 2
    expert_per_layer = 8 * 3 * D_MODEL * ei
    total_expert = N_LAYERS * expert_per_layer
    saved = total_expert * (1 - 1 / group)
    # This is savings on the EXPERT portion only (additive to base).
    # Report as: if you had a V5 MoE, this is the expert savings.
    return dict(saved=saved, final=base,  # base unchanged; savings on expert pool
                detail=f"Expert tying g={group}: saves {saved/1e6:.1f}M on a V5-MoE expert pool "
                       f"(base unchanged, expert storage {total_expert/1e6:.0f}M -> "
                       f"{(total_expert-saved)/1e6:.0f}M)")

# ----------------------------------------------------------------------------
# 15. LorExperts — experts as low-rank corrections to cluster dominant
# ----------------------------------------------------------------------------
def t_lorexperts(base, b, rank=64, n_clusters=4):
    # Hypothetical V5 MoE: 8 experts per layer. Cluster into 4 dominants +
    # 4 low-rank (rank=64) corrections. Expert FFN: 3*D*ei, ei=INTER//2.
    ei = INTER // 2
    full_expert = 3 * D_MODEL * ei
    dominant = 4 * full_expert
    lorank = 4 * 2 * rank * (D_MODEL + ei)  # A,B low-rank
    orig = 8 * full_expert
    saved_per_layer = orig - (dominant + lorank)
    saved = N_LAYERS * saved_per_layer
    return dict(saved=saved, final=base,
                detail=f"LorExperts rank={rank} clusters={n_clusters}: "
                       f"expert pool {orig/1e6:.1f}M -> {(orig-saved_per_layer)/1e6:.1f}M/layer")

# ----------------------------------------------------------------------------
# 16. SharVeT — similarity-aware sharing across layers
# ----------------------------------------------------------------------------
def t_sharvet(base, b, share_frac=0.5):
    # Share 50% of FFN+attn weights across similar layers, keep 50% unique.
    per_layer = (BASE["conv_per_layer"] + BASE["attn_per_layer"]) / 2
    ffn_attn_per_layer = 3 * D_MODEL * INTER + D_MODEL * D_MODEL  # rough
    shareable = ffn_attn_per_layer * share_frac
    # Group N_LAYERS into ~4 groups, share within group
    n_groups = 4
    saved = (N_LAYERS - n_groups) * shareable
    return dict(saved=saved, final=base - saved,
                detail=f"share {share_frac*100:.0f}% of FFN/attn across {n_groups} groups")

# ----------------------------------------------------------------------------
# 17. SLlama RRHP — reduce hidden size + projection
# ----------------------------------------------------------------------------
def t_rrhp(base, b, hidden_frac=0.5):
    # Reduce effective hidden dim by 50% with projection matrices.
    # Approximate: half the D*D matrices become D*(D/2).
    new_d = int(D_MODEL * hidden_frac)
    # Attention Q/O shrink, FFN stays INTER but gate/up use new_d
    saved_per_layer = 2 * D_MODEL * (D_MODEL - new_d)  # Q and O shrink
    saved = N_LAYERS * saved_per_layer
    return dict(saved=saved, final=base - saved,
                detail=f"RRHP hidden {D_MODEL}->{new_d}: Q/O shrink/layer")

# ----------------------------------------------------------------------------
# 18. INT4 weight-only quant (already in codebase) — storage, not param count
# ----------------------------------------------------------------------------
def t_int4_storage(base, b):
    new_bytes = base * 0.5 + b["embedding"] * 0.5  # 4 bits = 0.5 byte
    equiv = new_bytes // BF16
    saved = base - equiv
    return dict(saved=saved, final=equiv,
                detail="INT4 weight-only: 4 bits/param (vs 16 bf16) = 4x storage")

# ----------------------------------------------------------------------------
# 19. Combined: factorized embed + BitNet + MoE-active + cross-layer share
# ----------------------------------------------------------------------------
def t_combined_aggressive(base, b):
    # Stack: factorized embed (rank 256) + BitNet storage + MoE active (top-2 small)
    # + cross-layer share (4 templates) + LiSA Q/K
    r1 = t_factorized_embed(base, b)
    r2 = t_bitnet_full(r1["final"], {**b, "embedding": VOCAB * 256 + 256 * D_MODEL})
    r3 = t_moe_active(r2["final"], b)
    r4 = t_cross_layer_share(r3["final"], b, n_unique=4)
    r5 = t_lisa_qk(r4["final"], b)
    return dict(saved=base - r5["final"], final=r5["final"],
                detail="factorized_embed + bitnet_full + moe_active + cross_layer(4) + lisa")

# ============================================================================
# Run all techniques
# ============================================================================
TECHNIQUES = [
    Technique("Factorized Embedding (r=256)", "embedding", "LOSSLESS-at-init",
              t_factorized_embed, "SVD init = lossless; needs finetune to recover"),
    Technique("BitNet b1.58 (linear only)", "quant-storage", "NEAR-LOSSLESS",
              t_bitnet_storage, "QAT recovers; ternary {-1,0,1}+scale"),
    Technique("BitNet b1.58 (full, incl embed)", "quant-storage", "LOSSY-RECOVERABLE",
              t_bitnet_full, "Embed ternary is more lossy; needs longer QAT"),
    Technique("INT4 weight-only", "quant-storage", "NEAR-LOSSLESS",
              t_int4_storage, "GPTQ/AWQ recover ~1% perplexity"),
    Technique("FFN SVD (rank=25% INTER)", "decomposition", "LOSSY-RECOVERABLE",
              lambda b, bb: t_ffn_svd(b, bb, 0.25), "Distillation recovers most"),
    Technique("FFN SVD (rank=50% INTER)", "decomposition", "LOSSY-RECOVERABLE",
              lambda b, bb: t_ffn_svd(b, bb, 0.50), "Higher rank = less loss"),
    Technique("Kronecker FFN", "decomposition", "LOSSY-RECOVERABLE",
              t_kronecker_ffn, "KronBERT shows <1% drop with finetune"),
    Technique("Tensor-Train FFN (rank=4)", "decomposition", "LOSSY-RECOVERABLE",
              lambda b, bb: t_tensor_train_ffn(b, bb, 4), "TT-BERT: 10x compress, ~2% drop"),
    Technique("Monarch FFN (block=32)", "decomposition", "NEAR-LOSSLESS",
              lambda b, bb: t_monarch_ffn(b, bb, 32), "Monarch matches dense expressivity"),
    Technique("FastFood FFN", "decomposition", "LOSSY",
              t_fastfood_ffn, "Aggressive; quality drops noticeably"),
    Technique("Structured Prune (25%)", "pruning", "LOSSY-RECOVERABLE",
              lambda b, bb: t_structured_prune(b, bb, 0.25), "CPT/distillation recovers"),
    Technique("Layer Drop (25%)", "pruning", "LOSSY-RECOVERABLE",
              lambda b, bb: t_layer_drop(b, bb, 0.25), "ShortGPT: drop redundant layers"),
    Technique("Cross-Layer Share (4 templates)", "sharing", "LOSSY-RECOVERABLE",
              lambda b, bb: t_cross_layer_share(b, bb, 4), "ALBERT-style + adapters"),
    Technique("Hyperloop (begin2+loop4+end2)", "sharing", "NEAR-LOSSLESS",
              lambda b, bb: t_hyperloop(b, bb, 2, 2, 4), "Paper: 50% fewer, BETTER quality"),
    Technique("LiSA Q/K sharing (6x)", "sharing", "NEAR-LOSSLESS",
              lambda b, bb: t_lisa_qk(b, bb, 6), "6x Q/K compress, 19-40% throughput"),
    Technique("MoE active (top-2 small experts)", "moe", "LOSSLESS-at-init",
              t_moe_active, "dense_bypass=True = lossless start"),
    Technique("Expert Tying (g=2)", "moe", "NEAR-LOSSLESS",
              lambda b, bb: t_expert_tying(b, bb, 2), "EPFL: ~0 perplexity loss"),
    Technique("LorExperts (r=64, 4 clusters)", "moe", "NEAR-LOSSLESS",
              lambda b, bb: t_lorexperts(b, bb, 64, 4), "~50% expert compress, no retrain"),
    Technique("SharVeT (50% shared)", "sharing", "LOSSY-RECOVERABLE",
              lambda b, bb: t_sharvet(b, bb, 0.5), "32% better than naive sharing"),
    Technique("SLlama RRHP (hidden 50%)", "sharing", "LOSSY-RECOVERABLE",
              lambda b, bb: t_rrhp(b, bb, 0.5), "Projection recovers capacity"),
    Technique("COMBINED aggressive stack", "combined", "LOSSY-RECOVERABLE",
              t_combined_aggressive, "Stack of 5 techniques; needs full retrain"),
]

print("=" * 78)
print(f"{'Technique':<38} {'Category':<16} {'Quality':<20}")
print("=" * 78)
results = []
for t in TECHNIQUES:
    r = t.fn(BASE["total"], BASE)
    saved_pct = 100 * r["saved"] / BASE["total"] if BASE["total"] else 0
    final_mb = r["final"] * BF16 / 1e6
    results.append((t, r, saved_pct, final_mb))

# Sort by final param count (smallest = most compressed)
results.sort(key=lambda x: x[1]["final"])

print()
print("=" * 78)
print("RESULTS — sorted by final param count (most compressed first)")
print("=" * 78)
print(f"{'Technique':<38} {'Saved':>14} {'Final':>14} {'%':>7} {'MB':>9}  Quality")
print("-" * 110)
for t, r, pct, mb in results:
    print(f"{t.name:<38} {int(r['saved']):>14,d} {int(r['final']):>14,d} {pct:>6.1f}% {mb:>8.1f}M  {t.quality}")

print()
print("=" * 78)
print("DETAILS")
print("=" * 78)
for t, r, pct, mb in results:
    print(f"\n[{t.name}]  ({t.category}, {t.quality})")
    print(f"  saved: {int(r['saved']):,d}  final: {int(r['final']):,d}  ({pct:.1f}% reduction, {mb:.1f} MB bf16-equiv)")
    print(f"  detail: {r['detail']}")
    print(f"  recovery: {t.notes}")

# ============================================================================
# Quality-preservation ranking (theoretical)
# ============================================================================
print()
print("=" * 78)
print("QUALITY-PRESERVATION RANKING (theoretical, no retrain needed -> heavy retrain)")
print("=" * 78)
order = {"LOSSLESS-at-init": 0, "LOSSLESS": 1, "NEAR-LOSSLESS": 2,
         "LOSSY-RECOVERABLE": 3, "LOSSY": 4}
ranked = sorted(results, key=lambda x: (order.get(x[0].quality, 5), -x[2]))
for t, r, pct, mb in ranked:
    print(f"  {t.quality:<22} {pct:>6.1f}%  {t.name}")

print()
print("Done. See .devin/scratchpad.md for full analysis.")
