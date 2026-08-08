"""Teacher-Child distillation: train V2-opt (norm-folded) to match V2 original.

Teacher: ForgeLM V2 original (928 tensors, with real norm γ)
Student: ForgeLM V2-opt (678 tensors, norm-folded, γ absorbed into weights)

The norm folding is mathematically lossless but bf16 rounding causes small diffs.
Teacher-child distillation recovers this perfectly by training the student to
match the teacher's logits on calibration data.

Loss: KL(teacher_logits || student_logits) + CE(student, targets)
LR: 1e-5 (very low — just nudging bf16 rounding)
Steps: 100-200 (converges fast since diff is tiny)

Usage:
    py -3.13 .devin\teacher_child_distill.py [--steps 100] [--lr 1e-5]
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

CKPT_TEACHER = "research/checkpoints/forgelm_v2.safetensors"
CKPT_STUDENT = "research/checkpoints/forgelm_v2_opt.safetensors"
CKPT_OUT = "research/checkpoints/forgelm_v2_opt_distilled.safetensors"
META_OUT = CKPT_OUT + ".meta.json"
DATA_PATH = "research/data/all_teachers_v2.jsonl"
TOKENIZER_PATH = "research/checkpoints/qwen_hf"


def load_data(path, tokenizer, n_samples=200, max_seq_len=256):
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
    steps = 100
    lr = 1e-5
    for i, arg in enumerate(sys.argv):
        if arg == "--steps" and i + 1 < len(sys.argv):
            steps = int(sys.argv[i + 1])
        elif arg == "--lr" and i + 1 < len(sys.argv):
            lr = float(sys.argv[i + 1])

    print("="*60)
    print("Teacher-Child Distillation — V2 → V2-opt")
    print("="*60)
    print(f"  Teacher: {CKPT_TEACHER}")
    print(f"  Student: {CKPT_STUDENT}")
    print(f"  Output:  {CKPT_OUT}")
    print(f"  Steps: {steps}, LR: {lr}")
    print()

    from research.config import get_config
    from research.model_loader import ModelLoader
    from transformers import AutoTokenizer

    cfg = get_config("forgelm_v2", device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load student FIRST — so model cache has identity norms (not teacher's)
    # The model loader caches architecture via deepcopy; if teacher loads first,
    # missing norm keys (deduped) keep teacher's γ values instead of identity.
    print("[1] Loading student (V2-opt, norm-folded)...")
    t0 = time.time()
    student = ModelLoader.build_model_fast(
        cfg, checkpoint_path=CKPT_STUDENT,
        moe_top_k=0, dtype=torch.bfloat16)
    student.to("cuda").train()
    print(f"  Loaded in {time.time()-t0:.1f}s (trainable)")

    # Load teacher (frozen)
    print("\n[2] Loading teacher (V2 original)...")
    t0 = time.time()
    teacher = ModelLoader.build_model_fast(
        cfg, checkpoint_path=CKPT_TEACHER,
        moe_top_k=0, dtype=torch.bfloat16)
    teacher.to("cuda").eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s (frozen)")

    # Load data
    print("\n[3] Loading calibration data...")
    samples = load_data(DATA_PATH, tokenizer, n_samples=200, max_seq_len=256)

    # Measure initial divergence
    print("\n[4] Measuring initial divergence (teacher vs student)...")
    cos_scores = []
    kl_scores = []
    with torch.no_grad():
        for ids in samples[:10]:
            x = ids[:-1].unsqueeze(0).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                t_out = teacher(x)
                s_out = student(x)
                t_logits = t_out[0] if isinstance(t_out, tuple) else t_out
                s_logits = s_out[0] if isinstance(s_out, tuple) else s_out
                t_flat = t_logits[0, -1].float()
                s_flat = s_logits[0, -1].float()
                cos = F.cosine_similarity(t_flat.unsqueeze(0), s_flat.unsqueeze(0)).item()
                kl = F.kl_div(F.log_softmax(s_flat, -1), F.softmax(t_flat, -1),
                             reduction="sum").item()
                cos_scores.append(cos)
                kl_scores.append(kl)
    init_cos = sum(cos_scores) / len(cos_scores)
    init_kl = sum(kl_scores) / len(kl_scores)
    print(f"  Initial logit cos: {init_cos:.8f}")
    print(f"  Initial KL divergence: {init_kl:.6f}")

    # Distillation training
    print(f"\n[5] Distilling for {steps} steps (LR={lr})...")
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.0)

    step = 0
    losses = []
    cos_track = []
    t_train = time.time()
    while step < steps:
        for ids in samples:
            if step >= steps:
                break
            x = ids[:-1].unsqueeze(0).to("cuda")
            y = ids[1:].unsqueeze(0).to("cuda")

            # Teacher forward (no grad)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                t_out = teacher(x)
                t_logits = t_out[0] if isinstance(t_out, tuple) else t_out
                t_probs = F.softmax(t_logits.float(), dim=-1)

            # Student forward
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                s_out = student(x)
                s_logits = s_out[0] if isinstance(s_out, tuple) else s_out

                # Distillation loss: KL(teacher || student)
                kl_loss = F.kl_div(
                    F.log_softmax(s_logits.float(), dim=-1),
                    t_probs, reduction="batchmean")

                # Also CE with ground truth (small weight)
                ce_loss = F.cross_entropy(
                    s_logits.view(-1, s_logits.size(-1)).float(), y.view(-1))

                loss = kl_loss + 0.1 * ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1
            if step % 10 == 0:
                avg = sum(losses[-10:]) / min(10, len(losses))
                elapsed = time.time() - t_train
                # Quick cos check
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    s_check = student(x)
                    s_l = s_check[0] if isinstance(s_check, tuple) else s_check
                    c = F.cosine_similarity(
                        t_logits[0, -1].float().unsqueeze(0),
                        s_l[0, -1].float().unsqueeze(0)).item()
                    cos_track.append(c)
                print(f"  Step {step}/{steps}: loss={avg:.6f}, cos={c:.8f} "
                      f"({elapsed:.1f}s, {step/elapsed:.1f} steps/s)")

    train_time = time.time() - t_train
    print(f"  Done in {train_time:.1f}s")

    # Measure final divergence
    print("\n[6] Measuring final divergence...")
    student.eval()
    cos_scores = []
    kl_scores = []
    t1_match = 0
    with torch.no_grad():
        for ids in samples[:20]:
            x = ids[:-1].unsqueeze(0).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                t_out = teacher(x)
                s_out = student(x)
                t_logits = t_out[0] if isinstance(t_out, tuple) else t_out
                s_logits = s_out[0] if isinstance(s_out, tuple) else s_out
                t_flat = t_logits[0, -1].float()
                s_flat = s_logits[0, -1].float()
                cos = F.cosine_similarity(t_flat.unsqueeze(0), s_flat.unsqueeze(0)).item()
                kl = F.kl_div(F.log_softmax(s_flat, -1), F.softmax(t_flat, -1),
                             reduction="sum").item()
                cos_scores.append(cos)
                kl_scores.append(kl)
                if t_flat.argmax() == s_flat.argmax():
                    t1_match += 1
    final_cos = sum(cos_scores) / len(cos_scores)
    final_kl = sum(kl_scores) / len(kl_scores)
    t1_rate = t1_match / 20
    print(f"  Final logit cos: {final_cos:.8f}")
    print(f"  Final KL divergence: {final_kl:.8f}")
    print(f"  Top-1 match: {t1_rate:.0%}")
    print(f"  Cos improvement: {init_cos:.6f} → {final_cos:.6f}")

    # Save distilled checkpoint
    print(f"\n[7] Saving distilled checkpoint to {CKPT_OUT}...")
    state = {}
    for k, v in student.state_dict().items():
        state[k] = v.to(torch.bfloat16).clone()
    from safetensors.torch import save_file
    save_file(state, CKPT_OUT)

    # Copy metadata from opt checkpoint
    import shutil
    shutil.copy(CKPT_STUDENT + ".meta.json", META_OUT)

    out_size = os.path.getsize(CKPT_OUT)
    print(f"  Saved: {out_size/1e6:.1f} MB")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Initial cos: {init_cos:.8f}")
    print(f"  Final cos:   {final_cos:.8f}")
    print(f"  Initial KL:  {init_kl:.6f}")
    print(f"  Final KL:    {final_kl:.8f}")
    print(f"  Top-1 match: {t1_rate:.0%}")
    print(f"  Steps: {steps}, Time: {train_time:.1f}s")
    print(f"  Checkpoint: {CKPT_OUT} ({out_size/1e6:.1f} MB)")
    if final_cos > 0.9999:
        print(f"  STATUS: NEAR-PERFECT RECOVERY (cos > 0.9999)")
    elif final_cos > init_cos:
        print(f"  STATUS: IMPROVED (cos +{final_cos - init_cos:.6f})")
    else:
        print(f"  STATUS: NO IMPROVEMENT")


if __name__ == "__main__":
    main()
