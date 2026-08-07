"""Bisect: which transform breaks ForgeLM generation?

Tests generation quality after each transform:
  0. Original Qwen (baseline) 
  1. V1 checkpoint (all transforms: MLA + MoE + MRL + QuaRot + RotorQuant + MTP + ValueResidual)
  2. V2 checkpoint (v1 + QK-Norm + DenseFormer + SandwichNorm + LogitCap + SwiGLUClamp)

If v1 is broken, we need to rebuild step by step (no intermediate checkpoints saved).
But first, let's check if v1 itself works or if v2's transforms broke it.
"""
import torch, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.config import get_config
from research.model_loader import ModelLoader
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

# Simple code completion prompt
prompt = (
    "def sum_list(lst):\n"
    '    """Compute the sum of all elements in a list"""\n'
    "    "
)

ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

def test_gen(label, config_name, ckpt_path, moe_topk=None):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  config={config_name}, ckpt={os.path.basename(ckpt_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(ckpt_path):
        print(f"  SKIP: checkpoint not found")
        return None
    
    cfg = get_config(config_name, device="cuda")
    kwargs = {}
    if moe_topk:
        kwargs['moe_top_k'] = moe_topk
    
    try:
        model = ModelLoader.build_model_fast(cfg, checkpoint_path=ckpt_path, **kwargs)
        model.to("cuda").eval()
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        return None
    
    # Generate 30 tokens with KV cache
    try:
        with torch.inference_mode():
            cur_ids = ids.clone()
            past_kv = None
            for step in range(30):
                if past_kv is not None:
                    logits, _, past_kv = model(
                        cur_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    logits, _, past_kv = model(cur_ids, use_cache=True)
                next_token = logits[0, -1].argmax()
                cur_ids = torch.cat([cur_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                if next_token.item() == tok.eos_token_id:
                    break
        
        gen = tok.decode(cur_ids[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"  Generated: {repr(gen[:100])}")
        
        # Check if output is coherent (not garbage)
        is_coherent = all(ord(c) < 128 for c in gen[:50])  # No Cyrillic/etc
        has_code = any(kw in gen for kw in ['return', 'sum', 'for', 'def', ' ', '='])
        is_garbage = '<|fim_middle|>' in gen or 'грани' in gen or 'accounts' in gen[:20]
        
        if is_garbage:
            verdict = "GARBAGE"
        elif is_coherent and has_code:
            verdict = "OK"
        else:
            verdict = "SUSPICIOUS"
        
        print(f"  Verdict: {verdict}")
        return verdict
    except Exception as e:
        print(f"  GEN ERROR: {e}")
        return "ERROR"
    finally:
        del model
        torch.cuda.empty_cache()


results = {}

# Test 0: Original Qwen (baseline — should be OK)
results['qwen_original'] = test_gen(
    "Qwen2.5-Coder-1.5B (original ported)",
    "qwen25_coder_1.5b",
    "research/checkpoints/qwen25_coder_1.5b_ported.safetensors")

# Test 1: ForgeLM v1 (MLA + MoE + MRL + QuaRot + RotorQuant + MTP + ValueResidual)
results['forgelm_v1'] = test_gen(
    "ForgeLM v1 (MLA+MoE+MRL+QuaRot+RotorQuant+MTP+ValueResidual)",
    "forgelm_v1",
    "research/checkpoints/forgelm_v1.safetensors",
    moe_topk=4)  # top-4 = all experts (lossless)

# Test 1b: V1 with top-2 (in case top-4 routing is the issue)
results['forgelm_v1_top2'] = test_gen(
    "ForgeLM v1 (top-2 MoE routing)",
    "forgelm_v1",
    "research/checkpoints/forgelm_v1.safetensors",
    moe_topk=2)

# Test 2: ForgeLM v2 (v1 + QK-Norm + DenseFormer + SandwichNorm + LogitCap + SwiGLUClamp)
results['forgelm_v2'] = test_gen(
    "ForgeLM v2 (v1 + QK-Norm + DenseFormer + SandwichNorm + ...)",
    "forgelm_v2",
    "research/checkpoints/forgelm_v2.safetensors",
    moe_topk=4)

# Test 2b: V2 with top-2
results['forgelm_v2_top2'] = test_gen(
    "ForgeLM v2 (top-2 MoE routing)",
    "forgelm_v2",
    "research/checkpoints/forgelm_v2.safetensors",
    moe_topk=2)

# Summary
print(f"\n{'='*60}")
print("BISECTION SUMMARY")
print(f"{'='*60}")
for name, verdict in results.items():
    if verdict is None:
        print(f"  {name:30s} SKIP (no checkpoint)")
    else:
        print(f"  {name:30s} {verdict}")

# Analysis
print(f"\nANALYSIS:")
if results.get('qwen_original') == 'OK' and results.get('forgelm_v1') != 'OK':
    print("  → V1 transforms break the model (MLA/MoE/MRL/QuaRot/RotorQuant/MTP/ValueResidual)")
    print("  → Need to bisect V1's pipeline step by step")
elif results.get('forgelm_v1') == 'OK' and results.get('forgelm_v2') != 'OK':
    print("  → V2 transforms break the model (QK-Norm/DenseFormer/SandwichNorm/LogitCap/SwiGLUClamp)")
elif results.get('forgelm_v1_top2') == 'OK' and results.get('forgelm_v1') != 'OK':
    print("  → top-4 MoE routing is broken, top-2 works")
else:
    print("  → Unexpected pattern — manual investigation needed")
