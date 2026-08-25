"""Quick generation test — is the model coherent at all?"""
import sys, os
sys.path.insert(0, "D:\\windsurf\\ForgeAI")
os.environ["PYTHONPATH"] = "D:\\windsurf\\ForgeAI"

import torch
torch.set_grad_enabled(False)

from research.config import get_config
from research.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer

cfg = get_config("forgelm_v7_8b_b", device="cuda")
cfg.mtp_n_heads = 2
cfg.use_mtp = False
cfg.use_chunked_ce = True
cfg.ce_chunk_size = 128

# Try BOTH checkpoints
for ckpt_name, ckpt_path in [
    ("V7_8B_final", "research/checkpoints/ForgeLM_V7_8B_final.safetensors"),
    ("V7_best", "research/checkpoints/ForgeLM_V7_best.safetensors"),
]:
    print(f"\n{'='*60}")
    print(f"=== Testing {ckpt_name} ===")
    print(f"{'='*60}")
    try:
        model = ModelLoader.build_model_fast(
            cfg, checkpoint_path=ckpt_path,
            dtype=torch.bfloat16,
        )
        tok = get_tokenizer()
        model.eval()

        # Test 1: Simple prompt
        prompt = "The capital of France is"
        ids = tok.encode(prompt)
        input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
        
        with torch.no_grad():
            out = model(input_ids, targets=None)
            logits = out[0] if isinstance(out, tuple) else out
        
        # Check next token prediction
        probs = torch.softmax(logits[0, -1], dim=-1)
        top5 = torch.topk(probs, 5)
        print(f"Prompt: '{prompt}'")
        print(f"Top-5 next tokens:")
        for idx, prob in zip(top5.indices, top5.values):
            t = tok.decode([idx.item()])
            print(f"  token {idx.item()}: '{t}' (p={prob.item():.4f})")

        # Test 2: Another prompt
        prompt2 = "def fibonacci(n):"
        ids2 = tok.encode(prompt2)
        input_ids2 = torch.tensor([ids2], device="cuda", dtype=torch.long)
        
        with torch.no_grad():
            out2 = model(input_ids2, targets=None)
            logits2 = out2[0] if isinstance(out2, tuple) else out2
        
        probs2 = torch.softmax(logits2[0, -1], dim=-1)
        top5_2 = torch.topk(probs2, 5)
        print(f"\nPrompt: '{prompt2}'")
        print(f"Top-5 next tokens:")
        for idx, prob in zip(top5_2.indices, top5_2.values):
            t = tok.decode([idx.item()])
            print(f"  token {idx.item()}: '{t}' (p={prob.item():.4f})")

        # Test 3: Generate 20 tokens
        print(f"\nGenerating 20 tokens from '{prompt}':")
        gen_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(20):
                out = model(gen_ids, targets=None)
                logits = out[0] if isinstance(out, tuple) else out
                next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen_ids = torch.cat([gen_ids, next_token], dim=1)
        
        generated_text = tok.decode(gen_ids[0].tolist())
        print(f"  '{generated_text}'")

        # Test 4: Check weight statistics
        print(f"\nWeight stats:")
        print(f"  embed.embed.weight: norm={model.embed.embed.weight.norm().item():.4f}, "
              f"std={model.embed.embed.weight.std().item():.6f}")
        print(f"  embed.project.weight: norm={model.embed.project.weight.norm().item():.4f}, "
              f"std={model.embed.project.weight.std().item():.6f}")
        
        # Check a few NLRQ FFN params
        blk2 = model.blocks[2]
        ffn = blk2.ffn
        for name in ['w_gate.U_q', 'w_gate.V_q', 'w_gate.S', 'w_gate.U_scale', 'w_gate.V_scale']:
            if hasattr(ffn, 'w_gate'):
                param = getattr(ffn.w_gate, name.replace('w_gate.', ''), None)
                if param is not None:
                    print(f"  block2 ffn.{name}: norm={param.norm().item():.4f}, "
                          f"shape={param.shape}")
        
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n=== Done ===")
