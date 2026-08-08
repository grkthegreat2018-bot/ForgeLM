"""Isolate norm folding bug — compare layer-by-layer outputs."""
import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from safetensors.torch import load_file
from research.config import get_config
from research.model_loader import ModelLoader

cfg = get_config("forgelm_v2", device="cuda")

# Load both models
print("Loading teacher (V2)...")
teacher = ModelLoader.build_model_fast(cfg, "research/checkpoints/forgelm_v2.safetensors",
                                       moe_top_k=0, dtype=torch.bfloat16).to("cuda").eval()
print("Loading student (V2-opt)...")
student = ModelLoader.build_model_fast(cfg, "research/checkpoints/forgelm_v2_opt.safetensors",
                                       moe_top_k=0, dtype=torch.bfloat16).to("cuda").eval()

# Check ln1 weights in both
print("\n--- ln1 weights comparison ---")
for i in [0, 1, 14, 27]:
    t_w = teacher.blocks[i].ln1.weight
    s_w = student.blocks[i].ln1.weight
    print(f"Layer {i} ln1: teacher max={t_w.abs().max():.4f}, student max={s_w.abs().max():.4f}, "
          f"teacher identity={((t_w-1).abs().max() < 1e-6).item()}, "
          f"student identity={((s_w-1).abs().max() < 1e-6).item()}")

# Check q_proj weights
print("\n--- q_proj weight comparison ---")
for i in [0, 1, 14, 27]:
    t_w = teacher.blocks[i].attn.q_proj.weight
    s_w = student.blocks[i].attn.q_proj.weight
    diff = (t_w.float() - s_w.float()).abs().max().item()
    cos = F.cosine_similarity(t_w.float().flatten().unsqueeze(0),
                              s_w.float().flatten().unsqueeze(0)).item()
    print(f"Layer {i} q_proj: diff={diff:.6f}, cos={cos:.8f}")

# Check head weights
print("\n--- head/embed comparison ---")
t_head = teacher.head.weight
s_head = student.head.weight
t_embed = teacher.embed.weight
s_embed = student.embed.weight
print(f"head: diff={((t_head-s_head).float().abs().max()):.6f}, "
      f"cos={F.cosine_similarity(t_head.float().flatten().unsqueeze(0), s_head.float().flatten().unsqueeze(0)).item():.8f}")
print(f"embed: diff={((t_embed-s_embed).float().abs().max()):.6f}, "
      f"cos={F.cosine_similarity(t_embed.float().flatten().unsqueeze(0), s_embed.float().flatten().unsqueeze(0)).item():.8f}")

# Check ln_f
print(f"\nln_f: teacher={teacher.ln_f.weight[:5].tolist()}, student={student.ln_f.weight[:5].tolist()}")
print(f"ln_f teacher identity={((teacher.ln_f.weight-1).abs().max() < 1e-6).item()}, "
      f"student identity={((student.ln_f.weight-1).abs().max() < 1e-6).item()}")

# Run a forward pass and compare intermediate outputs
print("\n--- Forward pass comparison ---")
ids = torch.tensor([[198, 198, 198, 198, 198]]).to("cuda")  # simple input

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # Teacher
    t_x = teacher.embed(ids)
    for i in range(28):
        t_ln1 = teacher.blocks[i].ln1(t_x)
        t_attn_out, _ = teacher.blocks[i].attn(t_ln1)
        t_x = t_x + t_attn_out
        t_ln2 = teacher.blocks[i].ln2(t_x)
        t_ffn_out, _ = teacher.blocks[i].ffn(t_ln2)
        t_x = t_x + t_ffn_out
        if i in [0, 1, 14, 27]:
            print(f"  Layer {i}: t_x norm={t_x.float().norm():.4f}")

    # Student
    s_x = student.embed(ids)
    for i in range(28):
        s_ln1 = student.blocks[i].ln1(s_x)
        s_attn_out, _ = student.blocks[i].attn(s_ln1)
        s_x = s_x + s_attn_out
        s_ln2 = student.blocks[i].ln2(s_x)
        s_ffn_out, _ = student.blocks[i].ffn(s_ln2)
        s_x = s_x + s_ffn_out
        if i in [0, 1, 14, 27]:
            cos = F.cosine_similarity(t_x.float().flatten().unsqueeze(0),
                                      s_x.float().flatten().unsqueeze(0)).item()
            diff = (t_x.float() - s_x.float()).abs().max().item()
            print(f"  Layer {i}: s_x norm={s_x.float().norm():.4f}, cos_vs_teacher={cos:.8f}, diff={diff:.6f}")

    # Final logits
    t_logits = teacher.head(teacher.ln_f(t_x))
    s_logits = student.head(student.ln_f(s_x))
    cos = F.cosine_similarity(t_logits[0,-1].float().unsqueeze(0),
                              s_logits[0,-1].float().unsqueeze(0)).item()
    print(f"\n  Final logit cos: {cos:.8f}")
