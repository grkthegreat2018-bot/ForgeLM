"""Test original Qwen weights (pre-transform) vs ForgeLM v2 (post-transform)."""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.config import get_config
from research.model_loader import ModelLoader
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

prompt = (
    "def sum_list(lst):\n"
    '    """Compute the sum of all elements in a list"""\n'
    "    "
)

ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

def test_model(label, config_name, ckpt_path, moe_topk=None):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    cfg = get_config(config_name, device="cuda")
    kwargs = {}
    if moe_topk:
        kwargs['moe_top_k'] = moe_topk
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=ckpt_path, **kwargs)
    model.to("cuda").eval()

    with torch.inference_mode():
        cur_ids = ids.clone()
        for step in range(50):
            logits, _ = model(cur_ids)
            next_token = logits[0, -1].argmax()
            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token.item() == tok.eos_token_id:
                break

    gen = tok.decode(cur_ids[0], skip_special_tokens=True)
    print(f"Generated: {repr(gen[len(prompt):])}")

    # Clean up
    del model
    torch.cuda.empty_cache()

# Test 1: Original Qwen ported (should work perfectly)
test_model("Qwen2.5-Coder-1.5B (original ported)",
           "qwen25_coder_1.5b", "research/checkpoints/qwen25_coder_1.5b_ported.safetensors")

# Test 2: ForgeLM v2 (all transforms applied)
test_model("ForgeLM v2 (MLA+MoE+MRL+QuaRot+RotorQuant+MTP+...)",
           "forgelm_v2", "research/checkpoints/forgelm_v2.safetensors", moe_topk=2)
