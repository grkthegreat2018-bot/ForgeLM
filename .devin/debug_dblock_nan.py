"""Debug NaN in DiffusionBlocks train_step."""
import os, torch, random, numpy as np
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from research.config import get_config
from research.model_loader import ConfigurableResearchLLM
from research.diffusion_blocks import DiffusionBlocks, DiffusionBlockConfig
import torch.nn.functional as F

torch.manual_seed(42); random.seed(42); np.random.seed(42)

config = get_config("lfm25_tiny")
model = ConfigurableResearchLLM(config).to("cuda").to(torch.bfloat16)
model.train()

db_config = DiffusionBlockConfig(num_blocks=2, use_noise_conditioning=True, cond_dim=128,
                                  sigma_min=0.001, sigma_max=1.0)
dblock = DiffusionBlocks(model, db_config, config.d_model, len(model.blocks))

train_seq = torch.tensor([[i % 4 for i in range(32)] for _ in range(4)], device="cuda")
train_labels = torch.cat([train_seq[:, 1:], train_seq[:, 0:1]], dim=1)

all_params = list(model.parameters()) + list(dblock.timestep_embedder.parameters())
for adaln in dblock.adalns:
    all_params += list(adaln.parameters())
optimizer = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=0.01)

for step in range(60):
    block_idx = step % 2
    for p in model.parameters():
        p.requires_grad = True

    w_nan = any(torch.isnan(p).any().item() for p in model.parameters())
    result = dblock.train_step(train_seq, train_labels, optimizer, block_idx=block_idx)
    ce = result["ce_loss"]
    is_nan = ce != ce

    if step % 10 == 0 or is_nan:
        print(f"Step {step:3d}: ce={ce:.4f}, w_nan={w_nan}, dropped={result['noise_dropped']}")

    if is_nan:
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f"  NaN in {name}: shape={p.shape}")
                break
        # Also check timestep embedder
        for name, p in dblock.timestep_embedder.named_parameters():
            if torch.isnan(p).any():
                print(f"  NaN in timestep.{name}")
        for i, adaln in enumerate(dblock.adalns):
            for name, p in adaln.named_parameters():
                if torch.isnan(p).any():
                    print(f"  NaN in adaln[{i}].{name}")
        break
