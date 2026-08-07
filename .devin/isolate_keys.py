"""Isolate per-key accuracy cost: apply ONE key at a time, measure cosine sim.

Applies each weight-transform key individually to the original Qwen model,
then compares forward-pass logits against the original. This tells us which
keys are lossless (should be ~1.0) and which are lossy.

Uses streaming (one tensor at a time) to avoid stutter.
"""
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors import safe_open
from safetensors.torch import save_file
from research.config import get_config
from research.model_loader import ModelLoader
from research.forge_keystack import (
    transform_mrl, transform_quarot, transform_spinquant,
    transform_value_residual, transform_gqa_to_mqa, transform_wanda,
    GPU, hadamard_matrix_gpu, gpu_matmul, gpu_index,
)
import tempfile

SRC = "research/checkpoints/qwen25_coder_1.5b_ported.safetensors"
CFG = "qwen25_coder_1.5b"

# Test prompts (token IDs)
TEST_IDS = [
    ("Hello world", [151643, 9707, 1917]),
    ("def fibonacci", [151643, 644, 8436, 23706, 1620]),
    ("The meaning of", [151643, 464, 7434, 315]),
    ("import torch", [151643, 1364, 10631, 4288]),
]

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

def apply_single_key(src_path, key_name, cfg):
    """Apply one key to the source model, return path to temp safetensors."""
    cfg_obj = get_config(CFG)
    state = {
        "n_heads": cfg_obj.n_heads,
        "n_kv_heads": cfg_obj.n_kv_heads or cfg_obj.n_heads,
        "calibration_acts": None,
        "wanda_sparsity": 0.2,
    }

    # MRL needs reorder indices computed first
    if key_name == "mrl":
        with safe_open(src_path, framework="pt") as f:
            emb_w = f.get_tensor("embed.weight")
            emb_gpu = emb_w.to(GPU, dtype=torch.float32)
            importance = emb_gpu.norm(dim=0)
            state["mrl_reorder"] = importance.argsort(descending=True).cpu()
            del emb_gpu, emb_w, importance
            torch.cuda.empty_cache()

    # Wanda needs calibration data
    if key_name == "wanda":
        # Simple: use random activations as proxy
        state["calibration_acts"] = torch.randn(128, cfg_obj.d_model)

    # SpinQuant needs explicit enable flag (no-op without quantization)
    if key_name == "spinquant_hadamard":
        state["enable_spinquant"] = True

    # QuaRot needs v0 buffered
    if key_name == "value_residual":
        pass  # transform handles v0 internally

    output = {}
    with safe_open(src_path, framework="pt") as f:
        all_keys = sorted(f.keys())
        for kn in all_keys:
            tensor = f.get_tensor(kn)
            if key_name == "mrl":
                tensor = transform_mrl(kn, tensor, state)
            elif key_name == "quarot_r2":
                tensor = transform_quarot(kn, tensor, state)
                # Handle buffered v_transformed
                if "v_proj" in kn and "weight" in kn:
                    parts = kn.split(".")
                    layer_idx = parts[1] if len(parts) > 1 else "0"
                    qs = state.get("quarot", {}).get(layer_idx, {})
                    if "v_transformed" in qs:
                        output[kn] = qs["v_transformed"].contiguous().to(torch.bfloat16)
                        del qs["v_transformed"]
                        continue
            elif key_name == "spinquant_hadamard":
                tensor = transform_spinquant(kn, tensor, state)
            elif key_name == "value_residual":
                tensor = transform_value_residual(kn, tensor, state)
            elif key_name == "gqa_to_mqa":
                tensor = transform_gqa_to_mqa(kn, tensor, state)
            elif key_name == "wanda":
                tensor = transform_wanda(kn, tensor, state)
            output[kn] = tensor.contiguous().to(torch.bfloat16)

    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp.close()
    save_file(output, tmp.name)
    return tmp.name

def main():
    print("=" * 70)
    print("PER-KEY ACCURACY ISOLATION")
    print("=" * 70)

    cfg = get_config(CFG)

    # Get baseline logits
    print("\nLoading original model for baseline...")
    model_orig = ModelLoader.build_model(
        get_config(CFG, device="cuda"), checkpoint_path=SRC
    ).to("cuda", dtype=torch.bfloat16).eval()
    baseline_logits = {name: get_logits(model_orig, ids) for name, ids in TEST_IDS}
    del model_orig
    torch.cuda.empty_cache()

    # Test each key
    keys_to_test = [
        ("mrl", "MRL reorder (lossless if inverse applied)"),
        ("spinquant_hadamard", "SpinQuant Hadamard (lossless if inverse applied)"),
        ("quarot_r2", "QuaRot R2 (lossless if inverse applied)"),
        ("value_residual", "ValueResidual (ResFormer, needs fine-tune)"),
        ("gqa_to_mqa", "GQA→MQA (lossy, pools KV heads)"),
        ("wanda", "Wanda 20% prune (lossy)"),
    ]

    print(f"\n{'Key':<25} {'Description':<45} {'AvgCos':>8} {'Top5':>6}")
    print(f"{'='*90}")

    results = []
    for key_name, desc in keys_to_test:
        try:
            tmp_path = apply_single_key(SRC, key_name, cfg)
            # Build model with appropriate config
            overrides = {}
            if key_name == "gqa_to_mqa":
                overrides["n_kv_heads"] = 1
            model_keyed = ModelLoader.build_model(
                get_config(CFG, device="cuda", **overrides),
                checkpoint_path=tmp_path
            ).to("cuda", dtype=torch.bfloat16).eval()

            cos_scores = []
            top5_scores = []
            for name, ids in TEST_IDS:
                logits = get_logits(model_keyed, ids)
                # Handle shape mismatch
                bl = baseline_logits[name]
                if logits.shape != bl.shape:
                    min_v = min(logits.shape[-1], bl.shape[-1])
                    logits = logits[..., :min_v]
                    bl = bl[..., :min_v]
                cos, top5 = compare(bl, logits)
                cos_scores.append(cos)
                top5_scores.append(top5)
            avg_cos = sum(cos_scores) / len(cos_scores)
            avg_top5 = sum(top5_scores) / len(top5_scores)
            print(f"  {key_name:<25} {desc:<45} {avg_cos:>8.4f} {avg_top5:>5.1f}/5")
            results.append((key_name, avg_cos, avg_top5))
            del model_keyed
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {key_name:<25} {desc:<45} ERROR: {e}")
            results.append((key_name, 0, 0))
        finally:
            try: os.unlink(tmp_path)
            except: pass

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (sorted by cosine similarity, highest = least damage)")
    print(f"{'='*70}")
    for name, cos, top5 in sorted(results, key=lambda x: -x[1]):
        status = "LOSSLESS" if cos > 0.95 else ("LOW LOSS" if cos > 0.7 else "LOSSY")
        print(f"  {name:<25} cos={cos:.4f}  top5={top5:.1f}/5  [{status}]")

if __name__ == "__main__":
    main()
