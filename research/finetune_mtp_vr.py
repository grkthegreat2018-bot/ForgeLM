"""Quick fine-tune for MTP heads + ValueResidual gates.

Trains ONLY:
  - MTP heads (shared trunk + 4 prediction heads) — for real speculative decoding
  - ValueResidual gates (28 scalars) — for V0 warm-start contribution

Everything else is frozen. This is a very small parameter count (~50M for MTP
heads + 28 scalars), so it trains in minutes on a single GPU.

Usage:
  python -m research.finetune_mtp_vr \
    --checkpoint research/checkpoints/xp_full_no_mqa.safetensors \
    --config qwen25_coder_1.5b \
    --steps 500 --lr 1e-3 --batch-size 2 --seq-len 256
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.config import get_config
from research.model_loader import ModelLoader
from research.mtp import MTPHead, MTPTrainer
from research.checkpoint_io import load_checkpoint, save_checkpoint


def load_training_data(tokenizer, seq_len=256, n_samples=500):
    """Load a small set of training data from pre-tokenized bins."""
    # Try to load pre-tokenized data
    data_path = "research/data/train_tokens.bin"
    if os.path.exists(data_path):
        data = torch.frombuffer(open(data_path, "rb").read(), dtype=torch.uint16)
        # Extract random sequences
        samples = []
        for _ in range(n_samples):
            start = torch.randint(0, len(data) - seq_len - 8, (1,)).item()
            chunk = data[start:start + seq_len + 8].long()
            samples.append(chunk)
        return samples

    # Fallback: generate simple code snippets as training data
    prompts = [
        "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True\n",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid+1\n        else:\n            hi = mid-1\n    return -1\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n",
        "def merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        return self.items.pop()\n    def is_empty(self):\n        return len(self.items) == 0\n",
        "class Queue:\n    def __init__(self):\n        self.items = []\n    def enqueue(self, item):\n        self.items.append(item)\n    def dequeue(self):\n        return self.items.pop(0)\n    def is_empty(self):\n        return len(self.items) == 0\n",
        "def bfs(graph, start):\n    visited = set()\n    queue = [start]\n    while queue:\n        node = queue.pop(0)\n        if node not in visited:\n            visited.add(node)\n            queue.extend(graph[node] - visited)\n    return visited\n",
    ]

    samples = []
    for prompt in prompts * (n_samples // len(prompts) + 1):
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        if len(ids) > seq_len + 8:
            ids = ids[:seq_len + 8]
        samples.append(ids)
    return samples[:n_samples]


def finetune_mtp_vr(checkpoint, config_name, steps=500, lr=1e-3,
                    batch_size=2, seq_len=256, device="cuda",
                    save_path=None):
    """Fine-tune MTP heads + ValueResidual gates."""
    print(f"=== MTP + VR Fine-tune ===")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Config: {config_name}")
    print(f"  Steps: {steps}, LR: {lr}, Batch: {batch_size}, SeqLen: {seq_len}")

    # Load model
    cfg = get_config(config_name, device=device)
    model = ModelLoader.build_model(cfg, checkpoint_path=checkpoint)
    model.eval()

    # Load checkpoint state for VR gates
    state = load_checkpoint(checkpoint, map_location=device)
    has_vr = "value_residual_gates" in state
    has_v0 = "value_residual_v0" in state

    # Setup MTP head
    n_predict = 4
    mtp_head = MTPHead(
        d_model=cfg.d_model, vocab_size=cfg.vocab_size,
        n_predict=n_predict,
    ).to(device)

    # Initialize MTP heads from LM head
    if hasattr(model, "head"):
        lm_weight = model.head.weight.data
        for h in mtp_head.heads:
            h.weight.data.copy_(lm_weight)
        # Initialize shared trunk as identity
        mtp_head.trunk[0].weight.data.copy_(torch.eye(cfg.d_model, device=device))
        if mtp_head.trunk[0].bias is not None:
            mtp_head.trunk[0].bias.data.zero_()
        print(f"  MTP heads initialized from LM head")

    # Freeze MTP heads (already initialized from LM head), only train trunk
    for h in mtp_head.heads:
        for p in h.parameters():
            p.requires_grad = False
    print(f"  MTP heads frozen (initialized from LM head)")
    print(f"  Only training shared trunk + LayerNorm")

    # Setup ValueResidual gates as trainable parameter
    vr_gates = None
    if has_vr:
        vr_gates = nn.Parameter(state["value_residual_gates"].clone().to(device))
        print(f"  VR gates loaded: shape={vr_gates.shape}, initial={vr_gates.data.tolist()[:5]}...")
    else:
        print(f"  VR gates: not found in checkpoint, skipping")

    # Freeze ALL model parameters
    for param in model.parameters():
        param.requires_grad = False
    print(f"  Frozen {sum(p.numel() for p in model.parameters())/1e6:.1f}M model params")

    # Trainable params: MTP trunk only (heads frozen) + VR gates
    mtp_params = [p for p in mtp_head.trunk.parameters() if p.requires_grad]
    trainable_params = mtp_params
    if vr_gates is not None:
        trainable_params.append(vr_gates)

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Trainable: {n_trainable/1e6:.2f}M params ({len(trainable_params)} tensors)")

    # Optimizer — use 8-bit Adam to save VRAM
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=0.01)
        print(f"  Using 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # Load training data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")
    samples = load_training_data(tokenizer, seq_len=seq_len, n_samples=steps * batch_size)
    print(f"  Training data: {len(samples)} samples")

    # MTP trainer for loss computation
    mtp_trainer = MTPTrainer(model, mtp_head, n_predict=n_predict,
                             curriculum=True, curriculum_steps=steps // 2,
                             mtp_weight=0.5)

    # Training loop
    model.train()  # Enable dropout etc for MTP trunk
    losses = []
    t0 = time.time()

    for step in range(steps):
        # Sample batch
        batch_indices = torch.randint(0, len(samples), (batch_size,))
        batch = [samples[i] for i in batch_indices]

        # Pad to same length
        max_len = max(len(s) for s in batch)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, s in enumerate(batch):
            padded[i, :len(s)] = s

        # Forward pass
        x = padded[:, :-1]  # input
        targets = padded[:, 1:]  # targets

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Get hidden states from model (frozen)
            with torch.no_grad():
                result = model(x, return_hidden=True)
                # result = (logits, loss, hidden) when return_hidden=True and no use_cache
                hidden = result[2] if len(result) > 2 else result[0]

            # MTP loss
            mtp_logits = mtp_head(hidden.detach())
            mtp_loss = torch.tensor(0.0, device=device)
            n_active = mtp_trainer.get_n_active(step) if hasattr(mtp_trainer, 'get_n_active') else n_predict

            for k in range(min(n_active, n_predict)):
                offset = k + 2
                if offset >= targets.shape[1]:
                    continue
                mtp_targets = targets[:, offset:]
                mtp_pred = mtp_logits[k][:, :-offset] if mtp_logits[k].shape[1] > offset else mtp_logits[k]
                # Align sizes
                min_t = min(mtp_pred.shape[1], mtp_targets.shape[1])
                mtp_loss = mtp_loss + F.cross_entropy(
                    mtp_pred[:, :min_t].reshape(-1, mtp_pred.size(-1)),
                    mtp_targets[:, :min_t].reshape(-1),
                    ignore_index=0,
                )
            if n_active > 0:
                mtp_loss = mtp_loss / n_active

            # VR gate regularization (encourage small positive gates)
            vr_loss = torch.tensor(0.0, device=device)
            if vr_gates is not None:
                # L2 regularization on gates (keep them small)
                vr_loss = (vr_gates ** 2).mean() * 0.01

            total_loss = mtp_loss + vr_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(total_loss.item())
        if step % 50 == 0 or step == steps - 1:
            avg_loss = sum(losses[-50:]) / min(50, len(losses))
            elapsed = time.time() - t0
            gate_vals = vr_gates.data.tolist()[:5] if vr_gates is not None else None
            print(f"  Step {step:4d}/{steps} | loss={avg_loss:.4f} | "
                  f"mtp={mtp_loss.item():.4f} | "
                  f"gates={gate_vals} | {elapsed:.0f}s")

    # Save
    if save_path:
        # Save MTP head
        mtp_path = save_path.replace(".safetensors", "_mtp.safetensors")
        from safetensors.torch import save_file as save_safetensors
        mtp_state = {k: v.cpu() for k, v in mtp_head.state_dict().items()}
        save_safetensors(mtp_state, mtp_path)
        print(f"  Saved MTP head to {mtp_path}")

        # Save VR gates back to checkpoint
        if vr_gates is not None:
            state["value_residual_gates"] = vr_gates.data.cpu()
            save_safetensors({k: v.cpu() for k, v in state.items()
                             if isinstance(v, torch.Tensor)}, save_path)
            print(f"  Saved VR gates to {save_path}")

    print(f"\nDone in {time.time()-t0:.0f}s")
    return mtp_head, vr_gates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="research/checkpoints/xp_full_no_mqa.safetensors")
    parser.add_argument("--config", default="qwen25_coder_1.5b")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", default="research/checkpoints/xp_finetuned.safetensors")
    args = parser.parse_args()

    finetune_mtp_vr(
        checkpoint=args.checkpoint,
        config_name=args.config,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        save_path=args.save,
    )
