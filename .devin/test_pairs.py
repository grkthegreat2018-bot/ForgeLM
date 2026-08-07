"""Test pairs of keys to find destructive interactions."""
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors import safe_open
from safetensors.torch import save_file
from research.config import get_config
from research.model_loader import ModelLoader
from research.forge_keystack import (
    transform_mrl, transform_quarot, transform_spinquant,
    transform_value_residual, transform_gqa_to_mqa, transform_wanda,
    GPU,
)
import tempfile

SRC = "research/checkpoints/qwen25_coder_1.5b_ported.safetensors"
CFG = "qwen25_coder_1.5b"

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

def apply_keys(src_path, key_list, cfg):
    cfg_obj = get_config(CFG)
    state = {
        "n_heads": cfg_obj.n_heads,
        "n_kv_heads": cfg_obj.n_kv_heads or cfg_obj.n_heads,
        "calibration_acts": None,
        "wanda_sparsity": 0.2,
    }

    # MRL needs reorder indices
    if "mrl" in key_list:
        with safe_open(src_path, framework="pt") as f:
            emb_w = f.get_tensor("embed.weight")
            emb_gpu = emb_w.to(GPU, dtype=torch.float32)
            importance = emb_gpu.norm(dim=0)
            state["mrl_reorder"] = importance.argsort(descending=True).cpu()
            del emb_gpu, emb_w, importance
            torch.cuda.empty_cache()

    # Wanda needs calibration data (permuted if MRL is applied)
    if "wanda" in key_list:
        state["calibration_acts"] = torch.randn(128, cfg_obj.d_model)
        if "mrl" in key_list:
            state["calibration_acts"] = state["calibration_acts"][:, state["mrl_reorder"]]

    output = {}
    with safe_open(src_path, framework="pt") as f:
        for kn in sorted(f.keys()):
            tensor = f.get_tensor(kn)
            for key_name in key_list:
                if key_name == "mrl":
                    tensor = transform_mrl(kn, tensor, state)
                elif key_name == "quarot_r2":
                    tensor = transform_quarot(kn, tensor, state)
                    if "v_proj" in kn and "weight" in kn:
                        parts = kn.split(".")
                        layer_idx = parts[1] if len(parts) > 1 else "0"
                        qs = state.get("quarot", {}).get(layer_idx, {})
                        if "v_transformed" in qs:
                            output[kn] = qs["v_transformed"].contiguous().to(torch.bfloat16)
                            del qs["v_transformed"]
                            tensor = None
                            break
                elif key_name == "spinquant_hadamard":
                    state["enable_spinquant"] = True
                    tensor = transform_spinquant(kn, tensor, state)
                elif key_name == "value_residual":
                    tensor = transform_value_residual(kn, tensor, state)
                elif key_name == "gqa_to_mqa":
                    tensor = transform_gqa_to_mqa(kn, tensor, state)
                elif key_name == "wanda":
                    tensor = transform_wanda(kn, tensor, state)
            if tensor is not None:
                output[kn] = tensor.contiguous().to(torch.bfloat16)

    tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp.close()
    save_file(output, tmp.name)

    overrides = {}
    if "gqa_to_mqa" in key_list:
        overrides["n_kv_heads"] = 1

    model = ModelLoader.build_model(
        get_config(CFG, device="cuda", **overrides),
        checkpoint_path=tmp.name
    ).to("cuda", dtype=torch.bfloat16).eval()

    # Get baseline
    model_orig = ModelLoader.build_model(
        get_config(CFG, device="cuda"), checkpoint_path=SRC
    ).to("cuda", dtype=torch.bfloat16).eval()

    cos_scores = []
    top5_scores = []
    for name, ids in TEST_IDS:
        logits = get_logits(model, ids)
        bl = get_logits(model_orig, ids)
        if logits.shape != bl.shape:
            min_v = min(logits.shape[-1], bl.shape[-1])
            logits = logits[..., :min_v]
            bl = bl[..., :min_v]
        cos, top5 = compare(bl, logits)
        cos_scores.append(cos)
        top5_scores.append(top5)

    del model, model_orig
    torch.cuda.empty_cache()
    os.unlink(tmp.name)
    return sum(cos_scores)/len(cos_scores), sum(top5_scores)/len(top5_scores)

def main():
    print("=" * 70)
    print("KEY PAIR INTERACTION TEST")
    print("=" * 70)

    # Single keys (baseline)
    singles = [
        ("mrl",),
        ("gqa_to_mqa",),
        ("wanda",),
        ("value_residual",),
    ]

    # Pairs
    pairs = [
        ("mrl", "gqa_to_mqa"),
        ("mrl", "wanda"),
        ("mrl", "value_residual"),
        ("gqa_to_mqa", "wanda"),
        ("gqa_to_mqa", "value_residual"),
        ("wanda", "value_residual"),
    ]

    # Triples
    triples = [
        ("mrl", "gqa_to_mqa", "wanda"),
        ("mrl", "gqa_to_mqa", "value_residual"),
        ("mrl", "wanda", "value_residual"),
        ("gqa_to_mqa", "wanda", "value_residual"),
    ]

    # All
    all_keys = ("mrl", "gqa_to_mqa", "wanda", "value_residual")

    print(f"\n{'Keys':<45} {'AvgCos':>8} {'Top5':>6} {'Expected':>10}")
    print(f"{'='*75}")

    # Singles
    for keys in singles:
        cos, top5 = apply_keys(SRC, list(keys), get_config(CFG))
        print(f"  {str(keys):<45} {cos:>8.4f} {top5:>5.1f}/5 {cos:>10.4f}")

    # Pairs
    print()
    for keys in pairs:
        cos, top5 = apply_keys(SRC, list(keys), get_config(CFG))
        # Expected = product of singles
        singles_cos = []
        for k in keys:
            sc, _ = apply_keys(SRC, [k], get_config(CFG))
            singles_cos.append(sc)
        expected = 1.0
        for c in singles_cos:
            expected *= c
        ratio = cos / expected if expected > 0 else 0
        print(f"  {str(keys):<45} {cos:>8.4f} {top5:>5.1f}/5 {expected:>10.4f}  ratio={ratio:.2f}")

    # Triples
    print()
    for keys in triples:
        cos, top5 = apply_keys(SRC, list(keys), get_config(CFG))
        print(f"  {str(keys):<45} {cos:>8.4f} {top5:>5.1f}/5")

    # All
    print()
    cos, top5 = apply_keys(SRC, list(all_keys), get_config(CFG))
    print(f"  {str(all_keys):<45} {cos:>8.4f} {top5:>5.1f}/5")

if __name__ == "__main__":
    main()
