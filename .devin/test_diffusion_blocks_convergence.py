"""Test DiffusionBlocks training stability and convergence (AR-adapted)."""
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

# ── Build tiny model ──
print("=== Building Tiny Model ===")
config = get_config("lfm25_tiny")
model = ConfigurableResearchLLM(config).to("cuda").to(torch.bfloat16)
model.train()

db_config = DiffusionBlockConfig(
    num_blocks=2, use_noise_conditioning=True, cond_dim=128,
    sigma_min=0.001, sigma_max=1.0, gamma=0.1,
)
dblock = DiffusionBlocks(model, db_config, config.d_model, len(model.blocks))

# ── Training task: predict next token in 0,1,2,3,0,1,2,3,... ──
train_seq = torch.tensor([[i % 4 for i in range(32)] for _ in range(4)], device="cuda")
train_labels = torch.cat([train_seq[:, 1:], train_seq[:, 0:1]], dim=1)
print(f"Task: learn 0,1,2,3,0,1,2,3,... (next token prediction)")

# ── Standard training (100 steps) ──
print(f"\n=== Standard Training (100 steps) ===")
model_std = ConfigurableResearchLLM(config).to("cuda").to(torch.bfloat16)
model_std.train()
opt_std = torch.optim.AdamW(model_std.parameters(), lr=3e-4, weight_decay=0.01)
for step in range(100):
    opt_std.zero_grad()
    out = model_std(train_seq, targets=train_labels)
    loss = out[1] if isinstance(out, tuple) and len(out) > 1 else None
    if loss is None:
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), train_labels.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_std.parameters(), 1.0)
    opt_std.step()
    if step % 20 == 0 or step == 99:
        print(f"  Step {step:3d}: loss={loss.item():.4f}")
with torch.no_grad():
    out = model_std(train_seq[:1])
    logits = out[0] if isinstance(out, tuple) else out
    std_acc = (logits[0].argmax(-1) == train_labels[0]).float().mean().item()
print(f"  Final accuracy: {std_acc:.2%}")

# ── DiffusionBlocks training (300 steps, cycling 2 blocks) ──
print(f"\n=== DiffusionBlocks Training (300 steps) ===")
# Single optimizer with all params (simpler than per-block)
all_params = list(model.parameters())
if dblock.timestep_embedder is not None:
    all_params += list(dblock.timestep_embedder.parameters())
if dblock.adalns is not None:
    for adaln in dblock.adalns:
        all_params += list(adaln.parameters())
optimizer = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=0.01)

db_losses = []
for step in range(300):
    block_idx = step % 2
    # Unfreeze all (simpler test — train all params, cycle blocks for layer_indices)
    for p in model.parameters():
        p.requires_grad = True

    result = dblock.train_step(train_seq, train_labels, optimizer, block_idx=block_idx)
    db_losses.append(result["ce_loss"])

    if step % 50 == 0 or step == 299:
        model.eval()
        with torch.no_grad():
            out = model(train_seq[:1])
            logits = out[0] if isinstance(out, tuple) else out
            acc = (logits[0].argmax(-1) == train_labels[0]).float().mean().item()
        model.train()
        adaln_norm = dblock.adalns[block_idx].linear.weight.abs().max().item()
        print(f"  Step {step:3d} [B{block_idx}]: ce={result['ce_loss']:.4f}, "
              f"acc={acc:.2%}, adaln={adaln_norm:.6f}, "
              f"noise_dropped={result['noise_dropped']}")

# ── Analysis ──
print(f"\n=== Analysis ===")
first_10 = sum(db_losses[:10]) / 10
last_10 = sum(db_losses[-10:]) / 10
print(f"CE Loss: {first_10:.4f} → {last_10:.4f} (Δ={last_10-first_10:.4f})")
has_nan = any(l != l for l in db_losses)
print(f"Stability: NaN={has_nan}")

# Final eval
model.eval()
with torch.no_grad():
    out = model(train_seq[:1])
    logits = out[0] if isinstance(out, tuple) else out
    db_acc = (logits[0].argmax(-1) == train_labels[0]).float().mean().item()
    preds = logits[0].argmax(-1)

print(f"\n=== Final Comparison ===")
print(f"Standard:     {std_acc:.2%}  (100 steps)")
print(f"DiffusionBlocks: {db_acc:.2%}  (300 steps = 150/block)")
print(f"Predicted: {preds[:10].tolist()}")
print(f"Expected:  {train_labels[0][:10].tolist()}")

if db_acc > 0.5:
    print("\n✅ DiffusionBlocks model learned the pattern (>50% accuracy)")
elif last_10 < first_10:
    print("\n⚠️  Loss decreased but accuracy <50% — needs more training or tuning")
else:
    print("\n❌ DiffusionBlocks model did not learn")
