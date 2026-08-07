"""Dissect the two lossy keys (GQA→MQA, ValueResidual) to find analytical fixes.

GQA→MQA (cos=0.77): We mean-pool 2 KV heads into 1. Alternatives:
  1. SVD: find the rank-1 approximation that best preserves the 2-head output
  2. Weighted average: weight by head importance (norm, activation energy)
  3. Keep both heads but share (actually MQA = 1 head, so this doesn't apply)
  4. Optimal projection: find the 1-head W that minimizes ||W_1-head @ x - W_2-head @ x||^2

ValueResidual (cos=0.92): V_i += V_0. The issue is that this changes attention
output for all layers. Alternatives:
  1. Scale V_0 by a learnable coefficient (init=0, so it starts as identity)
  2. Only add V_0 to later layers (not all)
  3. Use a gated residual: V_i + gate_i * V_0 where gate_i is initialized to 0
  4. Don't apply at all — it's a training-time technique, not a weight transform

Let's test each alternative analytically (no training).
"""
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors import safe_open
from safetensors.torch import save_file
from research.config import get_config
from research.model_loader import ModelLoader
import tempfile

SRC = "research/checkpoints/qwen25_coder_1.5b_ported.safetensors"
CFG = "qwen25_coder_1.5b"

TEST_IDS = [
    ("Hello world", [151643, 9707, 1917]),
    ("def fibonacci", [151643, 644, 8436, 23706, 1620]),
    ("The meaning of", [151643, 464, 7434, 315]),
    ("import torch", [151643, 1364, 10631, 4288]),
]

GPU = torch.device("cuda")

def get_logits(model, ids):
    input_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(input_ids)
        return out[0] if isinstance(out, tuple) else out

def compare(logits_a, logits_b):
    a = logits_a.float().flatten()
    b = logits_b.float().flatten()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    top5_a = set(logits_a[0, -1].topk(5).indices.tolist())
    top5_b = set(logits_b[0, -1].topk(5).indices.tolist())
    return cos, len(top5_a & top5_b)

def get_baseline_logits():
    """Get original model logits as reference."""
    model = ModelLoader.build_model(
        get_config(CFG, device="cuda"), checkpoint_path=SRC
    ).to("cuda", dtype=torch.bfloat16).eval()
    logits = {name: get_logits(model, ids) for name, ids in TEST_IDS}
    del model
    torch.cuda.empty_cache()
    return logits

def build_and_test(transform_fn, overrides=None, label=""):
    """Apply transform_fn to all tensors, build model, test accuracy."""
    output = {}
    with safe_open(SRC, framework="pt") as f:
        for kn in sorted(f.keys()):
            tensor = f.get_tensor(kn)
            tensor = transform_fn(kn, tensor)
            output[kn] = tensor.contiguous().to(torch.bfloat16)

    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp.close()
    save_file(output, tmp.name)

    model = ModelLoader.build_model(
        get_config(CFG, device="cuda", **(overrides or {})),
        checkpoint_path=tmp.name
    ).to("cuda", dtype=torch.bfloat16).eval()

    baseline = get_baseline_logits()
    cos_scores = []
    top5_scores = []
    for name, ids in TEST_IDS:
        logits = get_logits(model, ids)
        bl = baseline[name]
        if logits.shape != bl.shape:
            min_v = min(logits.shape[-1], bl.shape[-1])
            logits = logits[..., :min_v]
            bl = bl[..., :min_v]
        cos, top5 = compare(bl, logits)
        cos_scores.append(cos)
        top5_scores.append(top5)

    avg_cos = sum(cos_scores) / len(cos_scores)
    avg_top5 = sum(top5_scores) / len(top5_scores)
    print(f"  {label:<50} cos={avg_cos:.4f}  top5={avg_top5:.1f}/5")

    del model
    torch.cuda.empty_cache()
    os.unlink(tmp.name)
    return avg_cos, avg_top5

# ─── GQA→MQA alternatives ──────────────────────────────────────────────────

def gqa_mqa_mean(name, tensor):
    """Current: naive mean pooling."""
    if "k_proj" not in name and "v_proj" not in name:
        return tensor
    n_kv = 2
    if tensor.dim() == 2 and "bias" not in name:
        head_dim = tensor.shape[0] // n_kv
        d_model = tensor.shape[1]
        return tensor.view(n_kv, head_dim, d_model).mean(dim=0).reshape(head_dim, d_model)
    if tensor.dim() == 1 and "bias" in name:
        head_dim = tensor.shape[0] // n_kv
        return tensor.view(n_kv, head_dim).mean(dim=0)
    return tensor

def gqa_mqa_svd(name, tensor):
    """SVD: find rank-1 approximation of the 2-head stack.

    For weight W of shape (n_kv * head_dim, d_model), reshape to
    (n_kv, head_dim, d_model), then find the best rank-1 approximation
    across the n_kv dimension. This finds the single head that best
    represents both heads.
    """
    if "k_proj" not in name and "v_proj" not in name:
        return tensor
    n_kv = 2
    if tensor.dim() == 2 and "bias" not in name:
        head_dim = tensor.shape[0] // n_kv
        d_model = tensor.shape[1]
        # Stack: (n_kv, head_dim * d_model) → SVD → rank-1
        stacked = tensor.view(n_kv, head_dim * d_model).to(GPU, dtype=torch.float32)
        # SVD: U (n_kv, n_kv), S (n_kv,), V (n_kv, head_dim*d_model)
        U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
        # Rank-1: S[0] * U[:, 0] * Vh[0, :]
        # But we want (head_dim, d_model), so take Vh[0] reshaped
        rank1 = S[0] * Vh[0]  # (head_dim * d_model,)
        result = rank1.reshape(head_dim, d_model)
        return result.cpu().to(tensor.dtype)
    if tensor.dim() == 1 and "bias" in name:
        head_dim = tensor.shape[0] // n_kv
        # SVD on biases too
        stacked = tensor.view(n_kv, head_dim).to(GPU, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(stacked, full_matrices=False)
        rank1 = S[0] * Vh[0]
        return rank1.cpu().to(tensor.dtype)
    return tensor

def gqa_mqa_optimal(name, tensor, state={}):
    """Optimal projection: find W_single that minimizes ||W_single @ x - W_double @ x||^2.

    For each input x, the 2-head output is [W_1; W_2] @ x = concat(W_1 @ x, W_2 @ x).
    We want W_single @ x ≈ mean([W_1 @ x, W_2 @ x]) (which is what mean pooling does),
    but we can also find W_single that minimizes the error w.r.t. the FULL output
    (not just the mean).

    Actually, the optimal W_single for minimizing ||W_single @ x - W_1 @ x||^2 + ||W_single @ x - W_2 @ x||^2
    is exactly the mean: W_single = (W_1 + W_2) / 2. So mean pooling IS optimal
    for this objective.

    But the real objective is: minimize ||O @ concat(W_single @ x, W_single @ x) - O @ concat(W_1 @ x, W_2 @ x)||^2
    where O is the output projection. This is different! The O projection mixes
    the two heads, so we need to account for that.

    For simplicity, let's try: W_single = (W_1 + W_2) / 2 but also adjust O.
    Actually that requires changing O too. Let's try a different approach:
    weighted average where weights are determined by the O projection.
    """
    if "k_proj" not in name and "v_proj" not in name:
        return tensor
    n_kv = 2
    if tensor.dim() == 2 and "bias" not in name:
        head_dim = tensor.shape[0] // n_kv
        d_model = tensor.shape[1]
        # Weight by row norms (heads with larger weights contribute more)
        w1 = tensor[:head_dim]
        w2 = tensor[head_dim:]
        norm1 = w1.norm().item()
        norm2 = w2.norm().item()
        total = norm1 + norm2
        # Weighted average
        result = (norm1 * w1 + norm2 * w2) / total
        return result
    if tensor.dim() == 1 and "bias" in name:
        head_dim = tensor.shape[0] // n_kv
        b1 = tensor[:head_dim]
        b2 = tensor[head_dim:]
        norm1 = b1.norm().item()
        norm2 = b2.norm().item()
        total = norm1 + norm2
        return (norm1 * b1 + norm2 * b2) / total
    return tensor

def gqa_mqa_first_head(name, tensor):
    """Just keep the first head (drop second). Simplest baseline."""
    if "k_proj" not in name and "v_proj" not in name:
        return tensor
    n_kv = 2
    if tensor.dim() == 2 and "bias" not in name:
        head_dim = tensor.shape[0] // n_kv
        return tensor[:head_dim]
    if tensor.dim() == 1 and "bias" in name:
        head_dim = tensor.shape[0] // n_kv
        return tensor[:head_dim]
    return tensor

# ─── ValueResidual alternatives ────────────────────────────────────────────

def vr_naive(name, tensor, state={}):
    """Current: V_i += V_0 for all layers i > 0."""
    if "v_proj" not in name or "weight" not in name:
        return tensor
    if tensor.dim() != 2:
        return tensor
    parts = name.split(".")
    try:
        layer_idx = int(parts[1])
    except (IndexError, ValueError):
        return tensor
    if layer_idx == 0:
        state["v0"] = tensor.clone()
        return tensor
    v0 = state.get("v0")
    if v0 is None:
        return tensor
    return tensor + v0

def vr_gated_zero(name, tensor, state={}):
    """Gated residual with gate=0 (identity at init, learn gate later).

    V_i = V_i + 0 * V_0  →  same as original model at init.
    The gate is a learnable scalar per layer, initialized to 0.
    This is a NO-OP weight transform — the benefit comes from training.
    """
    return tensor  # no change, gate=0 means identity

def vr_scaled(name, tensor, state={}):
    """Scaled residual: V_i += alpha * V_0 where alpha = 1/n_layers.

    Smaller contribution from V_0 to reduce disruption.
    """
    if "v_proj" not in name or "weight" not in name:
        return tensor
    if tensor.dim() != 2:
        return tensor
    parts = name.split(".")
    try:
        layer_idx = int(parts[1])
    except (IndexError, ValueError):
        return tensor
    if layer_idx == 0:
        state["v0"] = tensor.clone()
        return tensor
    v0 = state.get("v0")
    if v0 is None:
        return tensor
    alpha = 1.0 / 28  # 1/n_layers
    return tensor + alpha * v0

def vr_only_late(name, tensor, state={}):
    """Only add V_0 to layers >= n_layers/2 (later layers only)."""
    if "v_proj" not in name or "weight" not in name:
        return tensor
    if tensor.dim() != 2:
        return tensor
    parts = name.split(".")
    try:
        layer_idx = int(parts[1])
    except (IndexError, ValueError):
        return tensor
    if layer_idx == 0:
        state["v0"] = tensor.clone()
        return tensor
    if layer_idx < 14:  # only later half
        return tensor
    v0 = state.get("v0")
    if v0 is None:
        return tensor
    return tensor + v0

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ANALYTICAL DISSECTION OF LOSSY KEYS")
    print("=" * 70)

    print("\n--- GQA→MQA alternatives ---")
    print("(Original: 2 KV heads → 1 KV head via mean pooling)")
    build_and_test(gqa_mqa_mean, {"n_kv_heads": 1}, "Mean pooling (current)")
    build_and_test(gqa_mqa_svd, {"n_kv_heads": 1}, "SVD rank-1 approximation")
    build_and_test(gqa_mqa_optimal, {"n_kv_heads": 1}, "Norm-weighted average")
    build_and_test(gqa_mqa_first_head, {"n_kv_heads": 1}, "Keep first head only")

    print("\n--- ValueResidual alternatives ---")
    print("(Original: V_i += V_0 for all layers > 0)")
    build_and_test(vr_naive, {}, "Naive V_i += V_0 (current)")
    build_and_test(vr_gated_zero, {}, "Gated, gate=0 (identity/no-op)")
    build_and_test(vr_scaled, {}, "Scaled: V_i += (1/28) * V_0")
    build_and_test(vr_only_late, {}, "Only layers >= 14: V_i += V_0")

    print("\n--- Combined best ---")
    # Will be filled based on results above

if __name__ == "__main__":
    main()
