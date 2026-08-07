"""Train DSparkLite head for ForgeLM v1 — fast version (~5 min)."""
import sys, os, torch
sys.path.insert(0, '.')

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from research.config import get_config
from research.model_loader import ModelLoader
from research.dspark_lite import DSparkLite, train_dspark_lite

DEVICE = "cuda"
SAVE_PATH = "research/checkpoints/dspark_lite_forgelm_v1.safetensors"

print("=" * 60)
print("DSparkLite Training — ForgeLM v1")
print("=" * 60)

# Load model (frozen)
print("\n[1/2] Loading ForgeLM v1...")
cfg = get_config("forgelm_v1", device=DEVICE)
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/forgelm_v1.safetensors")
model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params (frozen)")

# Create DSparkLite with shared LM head
print("\n[2/2] Creating + training DSparkLite...")
head = DSparkLite(
    d_model=1536,
    vocab_size=151936,
    n_predict=4,
    lm_head=model.head,  # share the model's LM head (not trained)
).to(DEVICE)

n_params = sum(p.numel() for p in head.adapters.parameters())
print(f"  DSparkLite adapters: {n_params/1e6:.1f}M params (vs 1013M full DSpark)")

# Train
head = train_dspark_lite(
    model, head,
    steps=500,
    lr=5e-4,
    seq_len=256,
    batch_size=2,
    warmup=50,
    grad_accum=2,  # effective batch 4
    save_path=SAVE_PATH,
    device=DEVICE,
)

print(f"\nCheckpoint: {SAVE_PATH}")
