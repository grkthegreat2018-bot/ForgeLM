"""Diagnose which params remain on meta after build_model_fast load."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from research.config import get_config
from research.model_loader import ModelLoader

cfg = get_config("forgelm_v7_8b_b", device="cuda")
ckpt = "research/checkpoints/ForgeLM_V7_8B_B_ported.safetensors"

# Patch load_state_dict to intercept and list meta params before the crash
import torch.nn as nn
_orig_load = nn.Module.load_state_dict

model = ModelLoader.build_model_fast(cfg, checkpoint_path=ckpt, dtype=torch.bfloat16, fast_load=True)
print("Build completed successfully")
