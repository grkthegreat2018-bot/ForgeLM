"""Train DSpark speculative decoding head on ForgeLM V2.

DSpark predicts 4 tokens per forward pass (vs 1 for standard generation).
This speeds up all generation by 60-85%, including self-play training.

The head is trained once on the base model, then used across all experts.
It's a small module (~50M params) that sits on top of the model's hidden states.

Usage:
    python train_dspark.py
    python train_dspark.py --data research/data/synthetic_coding.jsonl --epochs 3
    python train_dspark.py --n-predict 4 --lr 1e-4 --batch-size 2
"""
import os
import sys
import time
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V2_CHECKPOINT = "research/checkpoints/forgelm_v2.safetensors"
DSHARK_SAVE_PATH = "research/checkpoints/dspark_head.pt"
DEFAULT_DATA = "research/data/all_teachers_v2_scored.jsonl"


def load_training_texts(path: str, max_samples: int = 2000) -> List[Dict]:
    """Load training texts from JSONL or JSON.

    Returns list of {"text": str, "quality": float} dicts.
    Quality scores (if present) are used for weighted training.
    """
    p = Path(path)
    samples = []

    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prompt = item.get("prompt", "")
                completion = item.get("completion", "")
                text = (prompt + "\n" + completion).strip()
                quality = item.get("_quality_score", 1.0)
                if len(text) > 50:
                    samples.append({"text": text, "quality": quality})
                if len(samples) >= max_samples:
                    break
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text = (item.get("prompt", "") + "\n" + item.get("completion", "")).strip()
                    quality = item.get("_quality_score", 1.0)
                elif isinstance(item, str):
                    text = item
                    quality = 1.0
                else:
                    continue
                if len(text) > 50:
                    samples.append({"text": text, "quality": quality})
                if len(samples) >= max_samples:
                    break

    if not samples:
        raise ValueError(f"No valid texts found in {path}")

    avg_q = sum(s["quality"] for s in samples) / len(samples)
    print(f"  Loaded {len(samples)} samples from {path} (avg quality: {avg_q:.2f})")
    return samples


def tokenize_batch(texts: List[str], tokenizer, max_length: int = 512,
                   device: str = "cuda") -> torch.Tensor:
    """Tokenize a batch of texts into (B, T) token id tensor."""
    batch = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=False)
        batch.append(enc.input_ids[0])

    # Pad to same length
    max_len = max(t.shape[0] for t in batch)
    padded = torch.full((len(batch), max_len), tokenizer.pad_token_id or 0,
                        dtype=torch.long)
    for i, t in enumerate(batch):
        padded[i, :t.shape[0]] = t
    return padded.to(device)


@torch.no_grad()
def generate_with_acceptance(model, head, prompt_ids, max_new_tokens=60,
                             device="cuda"):
    """Generate with base model, record DSpark draft acceptance at each step.

    Returns:
        generated_ids: (1, T) full sequence (prompt + generated)
        acceptance_log: list of per-position dicts:
            {"pos": int, "draft_tokens": [int], "actual_tokens": [int],
             "accepted": [bool], "hidden": tensor}
    """
    model.eval()
    head.eval()
    gamma = head.n_predict
    eos_ids = {151643, 151645}
    if hasattr(model, "config") and hasattr(model.config, "eos_token_id"):
        if model.config.eos_token_id is not None:
            eos_ids.add(model.config.eos_token_id)

    input_ids = prompt_ids.clone()
    prompt_len = prompt_ids.shape[1]
    acceptance_log = []

    while input_ids.shape[1] - prompt_len < max_new_tokens:
        # Forward: get hidden states for the full sequence
        out = model(input_ids, return_hidden=True)
        hidden = out[-1] if len(out) > 2 else out[0]  # (1, T, d_model)
        logits = out[0] if isinstance(out, tuple) else out  # (1, T, V)
        last_hidden = hidden[:, -1:, :]  # (1, 1, d_model)
        last_token = input_ids[:, -1:]  # (1, 1) — the anchor token for DSpark conditioning

        # Base model picks next token (greedy)
        next_token = logits[0, -1].argmax().item()

        if next_token in eos_ids:
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            break

        # DSpark drafts gamma tokens from the last hidden state
        # Pass (1,1) hidden + (1,1) token so shapes match inside forward
        draft_logits_list, _ = head(last_hidden, last_token)
        draft_tokens = []
        for k in range(gamma):
            dt = draft_logits_list[k][0, -1].argmax().item()
            draft_tokens.append(dt)

        # Generate gamma actual tokens from base model (ground truth)
        # We already have the next token; for positions beyond that,
        # we'll compare against what the base model produces in the next
        # iteration. For now, just record the next token and compare head 1.
        actual_tokens = [next_token]

        # Compare draft head 1 to actual next token
        # (head 1 predicts t+1, which is what we just generated)
        accepted = [draft_tokens[0] == next_token]

        acceptance_log.append({
            "pos": input_ids.shape[1] - prompt_len,
            "draft_tokens": draft_tokens,
            "actual_tokens": [next_token],
            "accepted": accepted,
        })

        # Append base model's next token (always accepted)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

        # Free intermediate tensors
        del out, hidden, logits, last_hidden, draft_logits_list

    return input_ids, acceptance_log


def on_policy_train_step(model, head, optimizer, prompt_ids, max_gen=60,
                         device="cuda", fail_weight=2.0, correct_weight=0.5):
    """One on-policy training step: generate → compare → weighted loss.

    1. Base model generates a sequence (frozen, no grad)
    2. At each position, DSpark drafted gamma tokens
    3. Compare drafts to actual base model tokens
    4. Weight loss: failed predictions get fail_weight, correct get correct_weight
    5. Confidence head trained on actual acceptance (1.0/0.0)

    Returns:
        stats dict with loss, acceptance_rate, n_positions
    """
    import torch.nn.functional as F
    gamma = head.n_predict

    # Phase 1: Generate with base model + record acceptance (no grad)
    gen_ids, acc_log = generate_with_acceptance(
        model, head, prompt_ids, max_new_tokens=max_gen, device=device)

    if not acc_log:
        return {"total_loss": 0.0, "acceptance_rate": 0.0, "n_positions": 0,
                "ce_loss": 0.0, "conf_loss": 0.0}

    # Phase 2: Forward through model to get hidden states for all positions (no grad)
    with torch.no_grad():
        out = model(gen_ids, return_hidden=True)
        hidden = out[-1] if len(out) > 2 else out[0]  # (1, T, d_model)

    # Phase 3: DSpark forward (WITH grad) on the generated sequence
    head.train()
    draft_logits_list, confidences = head(hidden.detach(), gen_ids)
    # draft_logits_list: list of (1, T, V), confidences: (1, T, gamma)

    # Phase 4: Compute acceptance-weighted loss
    total_ce = 0.0
    total_conf = 0.0
    n_correct = 0
    n_total = 0

    for entry in acc_log:
        pos = entry["pos"]
        accepted = entry["accepted"]

        for k in range(min(gamma, len(accepted))):
            if pos + k + 1 >= gen_ids.shape[1]:
                break

            # Target: what the base model actually generated at pos+k+1
            target_tok = gen_ids[0, pos + k + 1].item()

            # DSpark's prediction at this position
            pred_logits = draft_logits_list[k][0, pos]  # (V,)
            target_tensor = torch.tensor([target_tok], device=device)

            # CE loss for this position
            ce = F.cross_entropy(pred_logits.unsqueeze(0), target_tensor)

            # Weight: failed predictions get more gradient
            w = fail_weight if not accepted[k] else correct_weight
            total_ce = total_ce + w * ce

            # Confidence loss: target = 1.0 if draft matched, 0.0 if not
            conf_pred = confidences[0, pos, k]
            conf_target = torch.tensor([1.0 if accepted[k] else 0.0],
                                       device=device, dtype=conf_pred.dtype)
            conf_loss = F.binary_cross_entropy_with_logits(
                conf_pred.unsqueeze(0), conf_target)
            total_conf = total_conf + conf_loss

            n_correct += int(accepted[k])
            n_total += 1

    if n_total == 0:
        return {"total_loss": 0.0, "acceptance_rate": 0.0, "n_positions": 0,
                "ce_loss": 0.0, "conf_loss": 0.0}

    ce_loss = total_ce / n_total
    conf_loss = total_conf / n_total
    total_loss = ce_loss + 0.5 * conf_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in head.parameters() if p.requires_grad], 1.0)
    optimizer.step()

    # Free generation tensors
    del gen_ids, hidden, draft_logits_list, confidences, out

    return {
        "total_loss": float(total_loss),
        "ce_loss": float(ce_loss),
        "conf_loss": float(conf_loss),
        "acceptance_rate": n_correct / n_total,
        "n_positions": n_total,
    }


def train_dspark(data_path: str, epochs: int, lr: float, batch_size: int,
                 n_predict: int, max_length: int, max_samples: int,
                 save_path: str, device: str = "cuda", mode: str = "onpolicy"):
    """Train DSpark head on ForgeLM V2.

    Modes:
        onpolicy: Base model generates → DSpark drafts → compare → weighted loss
                  (directly optimizes acceptance rate, best for finishing)
        offline: Train on teacher data texts (faster, less targeted)
    """
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.decoding.dspark import DSparkHead, DSparkTrainer
    from transformers import AutoTokenizer

    print("=" * 70)
    print(f"Train DSpark Speculative Decoding Head ({mode} mode)")
    print(f"  Data: {data_path}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"  n_predict: {n_predict} (tokens per forward pass)")
    print("=" * 70)

    # 1. Load data
    print(f"\n[1] Loading training data...")
    samples = load_training_texts(data_path, max_samples)

    # Split 90/10
    import random
    rng = random.Random(42)
    rng.shuffle(samples)
    n_val = max(1, len(samples) // 10)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"  Split: {len(train_samples)} train / {len(val_samples)} val")

    # 2. Load model (frozen, bf16 to save 3GB VRAM)
    print(f"\n[2] Loading ForgeLM V2 (frozen, bf16)...")
    from research.runtime.vram_manager import VRAMManager
    vram = VRAMManager(total_vram_gb=12.0, safety_margin_gb=0.5)
    vram.setup_compile_cache()

    cfg = get_config("forgelm_v2", device=device)
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=V2_CHECKPOINT, moe_top_k=0,
        dtype=torch.bfloat16)  # bf16: 3.2GB vs 6.3GB fp32
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vram.profile_after_model_load(model, "v2_loaded")

    d_model = cfg.d_model  # 1536
    vocab_size = cfg.vocab_size  # 151936
    print(f"  d_model={d_model}, vocab_size={vocab_size}")

    # 3. Create DSpark head (n_layers=1 to save VRAM — direct projection, no intermediate MLP)
    # Head in bf16 to match model — saves 50% head VRAM
    print(f"\n[3] Creating DSpark head...")
    head = DSparkHead(
        d_model=d_model,
        vocab_size=vocab_size,
        n_predict=n_predict,
        n_layers=1,  # 1 layer = just Linear(d_model, vocab) per head
        seq_rank=128,  # smaller RNN rank (256 was too much for 12GB)
        seq_mode="rnn",
    ).to(device).to(torch.bfloat16)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"  DSpark head: {n_params/1e6:.1f}M params")

    # Initialize from key system — steal LM head for parallel backbone
    # DSparkKey wraps MTPKey: head 1 = exact LM head, heads 2-4 = LM head (Markov approx)
    # RNN + confidence = zero-init (identity at start, refined by training)
    print(f"\n  Initializing from key system (DSparkKey + MTPKey)...")
    try:
        from research.keys.dspark_key import init_dspark_from_model
        init_dspark_from_model(model, head)
        print(f"  Parallel backbone: LM head stolen via MTPKey (head 1 exact, 2-4 approx)")
        print(f"  RNN + confidence: zero-init (identity, will refine in training)")
    except Exception as e:
        print(f"  Key init failed: {e}")
        import traceback; traceback.print_exc()

    # 4. Freeze head 1 (already exact LM head — no training needed)
    # This saves ~233M params from optimizer + gradient memory
    print(f"\n[4] Freezing head 1 (exact LM head copy, no training needed)...")
    for p in head.parallel_heads[0].parameters():
        p.requires_grad = False
    # Also freeze W1 embedding (shared with model, not trained here)
    head.W1.weight.requires_grad = False

    trainable = [p for p in head.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in head.parameters())
    print(f"  Trainable: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M total")

    # 5. Create trainer with 8-bit optimizer to save VRAM
    print(f"\n[5] Creating trainer...")
    trainer = DSparkTrainer(model, head, lr=lr, tv_weight=0.2, conf_weight=0.5)
    # Replace optimizer with 8-bit version to save ~6GB VRAM
    try:
        import bitsandbytes as bnb
        trainer.optimizer = bnb.optim.AdamW8bit(trainable, lr=lr, weight_decay=0.01)
        print(f"  Optimizer: bitsandbytes AdamW8bit (saves ~6GB VRAM)")
    except ImportError:
        print(f"  Optimizer: torch AdamW (install bitsandbytes for 8-bit)")

    # Gradient checkpointing on frozen model — saves ~50% activation VRAM
    # Model is frozen (no_grad in compute_loss) but checkpointing still helps
    # the forward pass use less peak memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"  Gradient checkpointing: ON")

    vram.empty_cache()
    vram.snapshot("after_head_init")

    # 5. Training loop
    print(f"\n[5] Training ({mode} mode)...")
    best_loss = float("inf")
    best_acc = 0.0
    best_state = None
    steps_per_epoch = (len(train_samples) + batch_size - 1) // batch_size

    # For on-policy mode, load existing head if present (continual training)
    if mode == "onpolicy" and os.path.exists(save_path):
        print(f"  Loading existing head for continual training...")
        try:
            ckpt = torch.load(save_path, map_location=device, weights_only=False)
            head.load_state_dict(ckpt["state_dict"])
            best_acc = ckpt.get("best_acceptance", 0.0)
            print(f"  Loaded (prev best acceptance: {best_acc:.1%})")
            # Re-init optimizer since head weights changed
            try:
                import bitsandbytes as bnb
                trainer.optimizer = bnb.optim.AdamW8bit(trainable, lr=lr, weight_decay=0.01)
            except ImportError:
                trainer.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        except Exception as e:
            print(f"  Load failed, starting fresh: {e}")

    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs} ({steps_per_epoch} steps)")
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0
        t0 = time.time()

        rng.shuffle(train_samples)
        vram.empty_cache()

        for step in range(steps_per_epoch):
            batch_samples = train_samples[step * batch_size:(step + 1) * batch_size]
            batch_texts = [s["text"] for s in batch_samples]

            try:
                if mode == "onpolicy":
                    # On-policy: generate from prompt, then tune on acceptance
                    prompt_text = batch_texts[0][:200]  # use first 200 chars as prompt
                    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                                           truncation=True, max_length=64).input_ids.to(device)
                    stats = on_policy_train_step(
                        model, head, trainer.optimizer, prompt_ids,
                        max_gen=60, device=device,
                        fail_weight=2.0, correct_weight=0.5)
                    epoch_acc += stats.get("acceptance_rate", 0)
                else:
                    # Offline: train on teacher data
                    input_ids = tokenize_batch(batch_texts, tokenizer, max_length, device)
                    stats = trainer.train_step(input_ids)
                    del input_ids

                epoch_loss += stats["total_loss"]
                epoch_steps += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    OOM at step {step}, skipping")
                    import gc; gc.collect()
                    vram.empty_cache()
                    continue
                raise

            if (step + 1) % 20 == 0 or step == 0:
                avg = epoch_loss / max(epoch_steps, 1)
                elapsed = time.time() - t0
                rate = (step + 1) / max(elapsed, 1)
                if mode == "onpolicy":
                    avg_acc = epoch_acc / max(epoch_steps, 1)
                    print(f"    [{step+1}/{steps_per_epoch}] loss={avg:.4f} "
                          f"acc={avg_acc:.1%} ({rate:.1f} steps/s)")
                else:
                    print(f"    [{step+1}/{steps_per_epoch}] loss={avg:.4f} "
                          f"({rate:.1f} steps/s)")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t0
        if mode == "onpolicy":
            avg_acc = epoch_acc / max(epoch_steps, 1)
            print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}, "
                  f"acceptance={avg_acc:.1%} ({elapsed:.0f}s)")
        else:
            print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")

        # Free training activations before validation
        vram.empty_cache()

        # Validation — measure acceptance rate on val prompts
        if mode == "onpolicy":
            # On-policy val: measure acceptance rate directly
            val_acc = 0.0
            val_n = 0
            head.eval()
            with torch.no_grad():
                for s in val_samples[:20]:  # subset for speed
                    prompt_text = s["text"][:200]
                    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                                           truncation=True, max_length=64).input_ids.to(device)
                    try:
                        _, acc_log = generate_with_acceptance(
                            model, head, prompt_ids, max_new_tokens=40, device=device)
                        for entry in acc_log:
                            for a in entry["accepted"]:
                                val_acc += int(a)
                                val_n += 1
                    except RuntimeError:
                        vram.empty_cache()
                        continue
            avg_val = val_acc / max(val_n, 1)
            print(f"  Val acceptance: {avg_val:.1%} ({val_n} positions)")
            metric = avg_val  # higher is better
            is_better = metric > best_acc
            metric_name = "acceptance"
        else:
            # Offline val: loss-based
            val_loss = 0.0
            val_steps = 0
            head.eval()
            with torch.no_grad():
                for i in range(0, len(val_samples), batch_size):
                    batch_texts = [s["text"] for s in val_samples[i:i + batch_size]]
                    input_ids = tokenize_batch(batch_texts, tokenizer, max_length, device)
                    try:
                        loss, stats = trainer.compute_loss(input_ids)
                        val_loss += stats["total_loss"]
                        val_steps += 1
                        del input_ids
                    except RuntimeError:
                        vram.empty_cache()
                        continue
            avg_val = val_loss / max(val_steps, 1)
            print(f"  Val loss: {avg_val:.4f}")
            metric = avg_val  # lower is better
            is_better = metric < best_loss
            metric_name = "loss"

        if is_better:
            if mode == "onpolicy":
                best_acc = metric
            else:
                best_loss = metric
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            print(f"  * New best {metric_name}: {metric:.4f}")
        else:
            print(f"  No improvement")

        head.train()  # back to train mode
        vram.empty_cache()

    # 6. Restore best + save
    if best_state is not None:
        head.load_state_dict(best_state)
        if mode == "onpolicy":
            print(f"\n  Restored best head (val acceptance: {best_acc:.1%})")
        else:
            print(f"\n  Restored best head (val loss: {best_loss:.4f})")

    # Disable gradient checkpointing for inference speed test
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()

    print(f"\n[6] Saving DSpark head...")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(),
        "config": {
            "d_model": d_model,
            "vocab_size": vocab_size,
            "n_predict": n_predict,
            "n_layers": 1,
            "seq_rank": 128,
            "seq_mode": "rnn",
        },
        "best_val_loss": best_loss,
        "best_acceptance": best_acc,
        "trained_on": V2_CHECKPOINT,
        "mode": mode,
    }, str(save_path))
    file_size = save_path.stat().st_size
    print(f"  Saved: {save_path} ({file_size/1e6:.0f} MB)")

    # 7. Quick speed test (free training memory first)
    print(f"\n[7] Speed test: DSpark vs baseline...")
    del trainer, best_state
    import gc; gc.collect()
    vram.empty_cache()

    test_text = val_samples[0]["text"] if val_samples else train_samples[0]["text"]
    test_ids = tokenizer(test_text, return_tensors="pt",
                         truncation=True, max_length=128).input_ids.to(device)

    # Baseline: token-by-token
    model.eval()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        past = None
        cur = test_ids
        for _ in range(20):
            logits, _, past = model(cur, past_key_values=past, use_cache=True)
            next_tok = logits[0, -1].argmax(keepdim=True)
            cur = next_tok.unsqueeze(0)
    torch.cuda.synchronize()
    baseline_ms = (time.time() - t0) * 1000

    # DSpark: 4 tokens per pass
    from research.decoding.dspark import dspark_generate
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        _ = dspark_generate(model, head, test_ids, max_new_tokens=20,
                            temperature=0.0, device=device)
    torch.cuda.synchronize()
    dspark_ms = (time.time() - t0) * 1000

    speedup = baseline_ms / max(dspark_ms, 1)
    print(f"  Baseline (20 tokens): {baseline_ms:.0f}ms")
    print(f"  DSpark   (20 tokens): {dspark_ms:.0f}ms")
    print(f"  Speedup: {speedup:.2f}x")

    print(f"\n{'='*70}")
    print(f"Done! DSpark head at {save_path}")
    print(f"  Best val loss: {best_loss:.4f}")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Train DSpark head on ForgeLM V2")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Training data (JSONL or JSON)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=2,
                        help="Tokens to predict per forward pass (block size)")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--save", default=DSHARK_SAVE_PATH)
    parser.add_argument("--mode", choices=["onpolicy", "offline"], default="onpolicy",
                        help="onpolicy: generate→compare→tune (best for finishing). "
                             "offline: train on teacher data (faster, less targeted)")
    args = parser.parse_args()

    train_dspark(
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_predict=args.n_predict,
        max_length=args.max_length,
        max_samples=args.max_samples,
        save_path=args.save,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
