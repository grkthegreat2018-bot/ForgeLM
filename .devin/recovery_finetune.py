"""Targeted recovery fine-tune for XP model after KeyStack transforms.

Only trains the layers that were actually modified by lossy keys:
  - GQA→MQA: K/V projections (mean-pooled, need to relearn)
  - ValueResidual: V projections (V_i += V_0, needs adjustment)
  - Wanda: pruned weights (need to redistribute)

Freezes everything else (embedding, Q, O, FFN, norms, LM head) to save VRAM.
Uses gradient checkpointing + tiny seq_len to fit in 12GB.
"""
import sys, os, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.config import get_config
from research.model_loader import ModelLoader
from research.checkpoint_io import save_checkpoint

CKPT = "research/checkpoints/xp_model_keystack.safetensors"
OUT = "research/checkpoints/xp_model_recovered.safetensors"
TRAIN_BIN = "research/data/train.bin"
VAL_BIN = "research/data/val.bin"
STEPS = 200
SEQ_LEN = 128
LR = 5e-5

def main():
    print("=" * 70)
    print("TARGETED RECOVERY FINE-TUNE (frozen layers, KV+V only)")
    print("=" * 70)

    cfg = get_config("xp_1.5b_mqa", device="cuda")
    model = ModelLoader.build_model(cfg, checkpoint_path=CKPT)
    model = model.to("cuda", dtype=torch.bfloat16)

    # Freeze everything, then unfreeze only K/V projections
    n_trainable = 0
    n_frozen = 0
    for name, param in model.named_parameters():
        # Only train k_proj, v_proj (GQA→MQA + ValueResidual recovery)
        should_train = any(k in name for k in ["k_proj", "v_proj"])
        param.requires_grad_(should_train)
        if should_train:
            n_trainable += param.numel()
        else:
            n_frozen += param.numel()

    print(f"  Trainable: {n_trainable/1e6:.1f}M params (K/V projections only)")
    print(f"  Frozen:    {n_frozen/1e6:.1f}M params")
    print(f"  Steps: {STEPS}, seq_len: {SEQ_LEN}, lr: {LR}")

    # Optimizer only on trainable params (saves VRAM)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

    # Load training data (raw uint16 tokens, Qwen vocab)
    import numpy as np
    train_np = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_np = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    train_data = torch.from_numpy(np.array(train_np)).to(torch.int64)
    val_data = torch.from_numpy(np.array(val_np)).to(torch.int64)
    print(f"  Train tokens: {len(train_data)/1e6:.1f}M, Val tokens: {len(val_data)/1e6:.1f}M")

    # Gradient checkpointing
    model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") else None

    model.train()
    print(f"\n{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}")

    for step in range(STEPS):
        # Random batch
        idx = torch.randint(0, len(train_data) - SEQ_LEN - 1, (1,)).item()
        x = train_data[idx:idx + SEQ_LEN].unsqueeze(0).to("cuda")
        y = train_data[idx + 1:idx + SEQ_LEN + 1].unsqueeze(0).to("cuda")

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            # Chunked CE to save VRAM
            loss = 0
            chunk = 64
            for i in range(0, SEQ_LEN, chunk):
                end = min(i + chunk, SEQ_LEN)
                l = nn.functional.cross_entropy(
                    logits[:, i:end].reshape(-1, logits.shape[-1]),
                    y[:, i:end].reshape(-1)
                )
                loss = loss + l
            loss = loss / ((SEQ_LEN + chunk - 1) // chunk)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        if step % 20 == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"  Step {step:3d}/{STEPS} | loss={loss.item():.4f} | VRAM={vram:.2f} GB")

        if step % 100 == 0 and step > 0:
            # Quick val
            model.eval()
            with torch.inference_mode():
                v_idx = torch.randint(0, len(val_data) - SEQ_LEN - 1, (1,)).item()
                vx = val_data[v_idx:v_idx + SEQ_LEN].unsqueeze(0).to("cuda")
                vy = val_data[v_idx + 1:v_idx + SEQ_LEN + 1].unsqueeze(0).to("cuda")
                vout = model(vx)
                vlogits = vout[0] if isinstance(vout, tuple) else vout
                vloss = nn.functional.cross_entropy(
                    vlogits.reshape(-1, vlogits.shape[-1]),
                    vy.reshape(-1)
                )
                print(f"  → val loss={vloss.item():.4f} (ppl={torch.exp(vloss).item():.1f})")
            model.train()

    # Save
    print(f"\n{'='*70}")
    print(f"SAVING RECOVERED MODEL")
    print(f"{'='*70}")
    from safetensors.torch import save_file
    state = {}
    for k, v in model.state_dict().items():
        state[k] = v.detach().cpu().contiguous().to(torch.bfloat16).clone()
    save_file(state, OUT)
    print(f"  Saved {len(state)} tensors to {OUT}")

    # Print final VRAM
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

if __name__ == "__main__":
    main()
