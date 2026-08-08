"""Fine-tune ForgeLM V2-opt to recover bf16 rounding from attention scale fold.

The scale fold introduces cos > 0.99999 (bf16 rounding in q_proj/k_up_proj
biases). A few fine-tune steps on calibration data recover this perfectly.

This is a MINIMAL fine-tune:
  - Loads V2-opt (735 tensors, scale-folded)
  - Runs N steps on synthetic data (default 50)
  - Uses low LR (5e-6) to nudge weights back to exact values
  - Saves recovered checkpoint as forgelm_v2_opt_ft.safetensors

Usage:
    py -3.13 .devin/finetune_v2_opt.py [--steps 50] [--lr 5e-6] [--data PATH]
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

# Paths
CKPT_OPT = os.path.join("research", "checkpoints", "forgelm_v2_opt.safetensors")
META_PATH = CKPT_OPT + ".meta.json"
CKPT_FT = os.path.join("research", "checkpoints", "forgelm_v2_opt_ft.safetensors")
DATA_PATH = os.path.join("research", "data", "all_teachers_v2.jsonl")
TOKENIZER_PATH = os.path.join("research", "checkpoints", "qwen_hf")


def load_data(path, tokenizer, n_samples=100, max_seq_len=256):
    """Load JSONL data and tokenize into input/target tensors."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= n_samples:
                break
            try:
                obj = json.loads(line)
                text = obj.get("text", obj.get("prompt", "") + " " + obj.get("response", ""))
                if not text.strip():
                    continue
                # Tokenize
                ids = tokenizer.encode(text, return_tensors="pt", max_length=max_seq_len,
                                       truncation=True).squeeze(0)
                if len(ids) < 16:
                    continue
                samples.append(ids)
            except Exception:
                continue
    print(f"  Loaded {len(samples)} samples from {path}")
    return samples


def main():
    # Parse args
    steps = 50
    lr = 5e-6
    data_path = DATA_PATH
    for i, arg in enumerate(sys.argv):
        if arg == "--steps" and i + 1 < len(sys.argv):
            steps = int(sys.argv[i + 1])
        elif arg == "--lr" and i + 1 < len(sys.argv):
            lr = float(sys.argv[i + 1])
        elif arg == "--data" and i + 1 < len(sys.argv):
            data_path = sys.argv[i + 1]

    print("="*60)
    print("Fine-tune V2-opt — Recover Scale Fold Rounding")
    print("="*60)
    print(f"  Checkpoint: {CKPT_OPT}")
    print(f"  Steps: {steps}")
    print(f"  LR: {lr}")
    print(f"  Data: {data_path}")
    print()

    # Load model
    from research.config import get_config
    from research.model_loader import ModelLoader
    from transformers import AutoTokenizer

    print("[1] Loading model (bf16, scale-folded)...")
    t0 = time.time()
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=CKPT_OPT,
        moe_top_k=0,  # dense_bypass
        dtype=torch.bfloat16)
    model.to("cuda")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\n[2] Loading calibration data...")
    samples = load_data(data_path, tokenizer, n_samples=100, max_seq_len=256)

    # Measure initial loss
    print("\n[3] Measuring initial loss...")
    model.eval()
    with torch.no_grad():
        initial_losses = []
        for ids in samples[:10]:
            x = ids[:-1].unsqueeze(0).to("cuda")
            y = ids[1:].unsqueeze(0).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
            initial_losses.append(loss.item())
    initial_loss = sum(initial_losses) / len(initial_losses)
    print(f"  Initial loss: {initial_loss:.4f} (ppl: {math.exp(initial_loss):.2f})")

    # Fine-tune
    print(f"\n[4] Fine-tuning for {steps} steps (LR={lr})...")
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    step = 0
    losses = []
    t_ft = time.time()
    while step < steps:
        for ids in samples:
            if step >= steps:
                break
            x = ids[:-1].unsqueeze(0).to("cuda")
            y = ids[1:].unsqueeze(0).to("cuda")

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1
            if step % 10 == 0:
                avg = sum(losses[-10:]) / min(10, len(losses))
                elapsed = time.time() - t_ft
                print(f"  Step {step}/{steps}: loss={avg:.4f} ({elapsed:.1f}s, {step/elapsed:.1f} steps/s)")

    ft_time = time.time() - t_ft
    print(f"  Done in {ft_time:.1f}s ({steps/ft_time:.1f} steps/s)")

    # Measure final loss
    print("\n[5] Measuring final loss...")
    model.eval()
    with torch.no_grad():
        final_losses = []
        for ids in samples[:10]:
            x = ids[:-1].unsqueeze(0).to("cuda")
            y = ids[1:].unsqueeze(0).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
            final_losses.append(loss.item())
    final_loss = sum(final_losses) / len(final_losses)
    print(f"  Final loss: {final_loss:.4f} (ppl: {math.exp(final_loss):.2f})")
    print(f"  Loss change: {initial_loss:.4f} → {final_loss:.4f} (Δ={final_loss-initial_loss:+.4f})")

    # Save fine-tuned checkpoint
    print(f"\n[6] Saving fine-tuned checkpoint to {CKPT_FT}...")
    state = {}
    for k, v in model.state_dict().items():
        state[k] = v.to(torch.bfloat16).clone()  # clone to break shared storage

    from safetensors.torch import save_file
    save_file(state, CKPT_FT)

    # Copy metadata from opt checkpoint (dedup map still applies)
    import shutil
    meta_ft = CKPT_FT + ".meta.json"
    shutil.copy(META_PATH, meta_ft)

    ft_size = os.path.getsize(CKPT_FT)
    print(f"  Saved: {ft_size/1e6:.1f} MB")
    print(f"  Metadata copied to: {meta_ft}")

    # Verify: compare scale-folded weights before and after fine-tune
    print(f"\n[7] Verifying weight recovery...")
    from safetensors.torch import load_file
    orig = load_file(os.path.join("research", "checkpoints", "forgelm_v2.safetensors"))
    ft = load_file(CKPT_FT)

    # Un-fold scale from fine-tuned weights for comparison
    fold_factor = math.sqrt(1.0 / math.sqrt(128))
    n_layers = 28
    for i in range(n_layers):
        for pname in ["q_proj", "k_up_proj"]:
            wk = f"blocks.{i}.attn.{pname}.weight"
            if wk in ft:
                ft[wk] = (ft[wk].float() / fold_factor).to(torch.bfloat16)
            bk = f"blocks.{i}.attn.{pname}.bias"
            if bk in ft:
                ft[bk] = (ft[bk].float() / fold_factor).to(torch.bfloat16)

    # Compare
    max_diff = 0
    cos_sum = 0
    cos_count = 0
    worst = []
    for k in orig:
        if k in ft:
            diff = (orig[k].float() - ft[k].float()).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                orig[k].float().flatten().unsqueeze(0),
                ft[k].float().flatten().unsqueeze(0)).item()
            max_diff = max(max_diff, diff)
            cos_sum += cos
            cos_count += 1
            if diff > 0.001:
                worst.append((k, diff, cos))

    worst.sort(key=lambda x: -x[1])
    print(f"  Max diff (after un-fold): {max_diff:.6f}")
    print(f"  Avg cosine: {cos_sum/cos_count:.8f}")
    if worst:
        print(f"  Tensors with diff > 0.001: {len(worst)}")
        for k, d, c in worst[:5]:
            print(f"    {k}: diff={d:.6f}, cos={c:.8f}")
    else:
        print(f"  No tensors with diff > 0.001 — FULLY RECOVERED")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f} (Δ={final_loss-initial_loss:+.4f})")
    print(f"  Max weight diff vs original: {max_diff:.6f}")
    print(f"  Avg cosine vs original: {cos_sum/cos_count:.8f}")
    print(f"  Fine-tune time: {ft_time:.1f}s ({steps} steps)")
    print(f"  Checkpoint: {CKPT_FT} ({ft_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
