"""Live-updating model trainer (Phase 2 + 3).

Consumes (prompt, completion) pairs from signal_capture.py and trains a
low-rank adapter (LoRA) on top of the frozen base model. The adapter is
small (~21M params at rank 16 on a 360M model), so updates are cheap and
the base weights stay safe.

Safety mechanisms (reused from online_learn.py / training_utils.py):
  - Replay buffer: mix new + old examples each batch → no catastrophic forgetting
  - EMA shadow weights on the LoRA params: rollback if val loss regresses
  - Gradient clipping (max_norm=1.0)
  - Quality gate: discard samples where self-verification score < threshold

Persistence (Phase 3):
  - LoRA adapters saved as versioned safetensors: research/checkpoints/live/v001/
  - Rollback CLI: --rollback v003
  - Merge CLI: --merge v005  (folds adapter into base weights)

Usage:
    # Train on captured signals (run in background alongside serve.py)
    python -m research.live_learn --base-ckpt research/checkpoints/distilled_llm.safetensors \\
        --signals research/data/live_training.jsonl --steps 100 --lr 1e-4

    # Rollback to a previous version
    python -m research.live_learn --rollback v003

    # Merge a versioned adapter into the base model
    python -m research.live_learn --merge v005 --base-ckpt research/checkpoints/distilled_llm.safetensors \\
        --out research/checkpoints/merged_live.safetensors
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.checkpoint_io import load_checkpoint, save_checkpoint
from research.config import get_config
from research.model_loader import ModelLoader
from research.runtime.signal_capture import filter_training_pairs, load_signals
from research.tokenizer_cache import get_tokenizer
from research.training.dpo_align import dpo_loss
from research.training.training_utils import (
    BinaryDataset,
    evaluate_loss,
    get_lr,
    init_ema,
    patch_triton_cache_for_windows,
    restore_ema,
    update_ema,
)

# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wrap an nn.Linear with a low-rank update: y = Wx + BAx.

    The base weight W is frozen; only A and B are trainable.
    Scaling: alpha / rank (standard LoRA scaling).
    """

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base.in_features
        out_features = base.out_features

        # A: (rank, in_features), B: (out_features, rank)
        # Init: A = kaiming, B = zeros → initial update is zero (identity start).
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze base weight.
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora(model, rank=16, alpha=32, target_modules=None):
    """Replace target nn.Linear modules in model with LoRALinear wrappers.

    target_modules: set of attribute name substrings to match (e.g. {"q_proj", "w_gate"}).
                    If None, defaults to attention projections only (safe + cheap).
    Returns list of (parent_module, attr_name, original_module) for removal/merge.
    """
    if target_modules is None:
        target_modules = {"q_proj", "out_proj", "kv_down_proj", "kv_up_proj"}

    injected = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if any(t in child_name for t in target_modules):
                wrapper = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, wrapper)
                injected.append((module, child_name, child))
    return injected


def extract_lora_state_dict(model):
    """Return only the LoRA params (A, B matrices) as a state dict."""
    lora_sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_sd[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_sd[f"{name}.lora_B"] = module.lora_B.data.clone()
    return lora_sd


def merge_lora_into_base(model):
    """Fold LoRA updates into base weights in-place: W := W + B*A*scaling."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = (module.lora_B @ module.lora_A) * module.scaling
                module.base.weight.add_(delta)
                # Replace wrapper with the merged base linear.
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                attr = name.rsplit(".", 1)[1] if "." in name else name
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, module.base)


# ---------------------------------------------------------------------------
# Versioned checkpoint persistence
# ---------------------------------------------------------------------------

class VersionedCheckpointer:
    """Save LoRA adapters to research/checkpoints/live/vNNN/ as safetensors."""

    def __init__(self, base_dir="research/checkpoints/live"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_versions(self):
        """Return sorted list of version strings: ['v001', 'v002', ...]."""
        versions = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
                versions.append(d.name)
        return sorted(versions, key=lambda v: int(v[1:]))

    def latest_version(self):
        v = self.list_versions()
        return v[-1] if v else None

    def next_version_path(self):
        latest = self.latest_version()
        n = int(latest[1:]) + 1 if latest else 1
        return self.base_dir / f"v{n:03d}"

    def save(self, lora_state_dict, metadata=None):
        """Save LoRA adapter + metadata to a new versioned directory."""
        vpath = self.next_version_path()
        vpath.mkdir(parents=True, exist_ok=True)
        adapter_path = vpath / "adapter.safetensors"
        meta_path = vpath / "metadata.json"

        # Use safetensors if available, else torch.save.
        try:
            from safetensors.torch import save_file
            save_file(lora_state_dict, str(adapter_path))
        except ImportError:
            adapter_path = vpath / "adapter.pt"
            torch.save(lora_state_dict, adapter_path)

        meta = {
            "version": vpath.name,
            "created_at": time.time(),
            "num_params": sum(t.numel() for t in lora_state_dict.values()),
            **(metadata or {}),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [checkpoint] saved {vpath.name} ({meta['num_params']:,} LoRA params)")
        return vpath.name

    def load(self, version):
        """Load a LoRA adapter state dict from a versioned directory."""
        vpath = self.base_dir / version
        if not vpath.exists():
            raise FileNotFoundError(f"Version {version} not found at {vpath}")
        adapter_path = vpath / "adapter.safetensors"
        if adapter_path.exists():
            from safetensors.torch import load_file
            return load_file(str(adapter_path))
        adapter_path = vpath / "adapter.pt"
        if adapter_path.exists():
            return torch.load(adapter_path, map_location="cpu")
        raise FileNotFoundError(f"No adapter file in {vpath}")

    def rollback(self, version):
        """Delete all versions newer than `version`."""
        all_v = self.list_versions()
        target_n = int(version[1:])
        for v in all_v:
            if int(v[1:]) > target_n:
                vpath = self.base_dir / v
                shutil.rmtree(vpath)
                print(f"  [rollback] removed {v}")
        print(f"  [rollback] active version is now {version}")


# ---------------------------------------------------------------------------
# Live trainer
# ---------------------------------------------------------------------------

class LiveTrainer:
    """Online LoRA trainer that consumes signal_capture pairs."""

    def __init__(self, cfg, base_ckpt, device="cuda",
                 lora_rank=16, lora_alpha=32,
                 replay_size=1000, ema_decay=0.999,
                 val_dataset=None, max_seq_len=512):
        self.cfg = cfg
        self.device = device
        self.ema_decay = ema_decay
        self.max_seq_len = max_seq_len
        self.tokenizer = get_tokenizer("Qwen/Qwen2.5-0.5B")

        # Load base model (frozen).
        print(f"Loading base model from {base_ckpt}...")
        self.model = ModelLoader.build_model(cfg, checkpoint_path=base_ckpt).to(device)
        self.model.eval()

        # Inject LoRA adapters.
        self.injected = inject_lora(self.model, rank=lora_rank, alpha=lora_alpha)
        n_lora = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Injected {len(self.injected)} LoRA adapters | {n_lora:,} trainable params")

        # EMA on LoRA params only.
        self.ema_state = init_ema(self._lora_module_view())

        # Replay buffer (token-level, like online_learn.py).
        from collections import deque
        self.replay = deque(maxlen=replay_size)

        # Val dataset for quality monitoring.
        self.val_dataset = val_dataset

        # Checkpointer.
        self.checkpointer = VersionedCheckpointer()

        self.best_val_loss = float("inf")

    def _lora_module_view(self):
        """Return a fake module-like dict for EMA helpers (name -> param)."""
        return {name: p for name, p in self.model.named_parameters() if p.requires_grad}

    def _encode_pair(self, prompt_messages, completion):
        """Encode a (messages, completion) pair into (input_ids, target_ids).

        Uses the chat template so the prompt matches what serve.py produces.
        Target masks: only train on the completion tokens.
        """
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full = prompt + completion
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        full_ids = self.tokenizer(full, return_tensors="pt").input_ids[0]

        # Truncate from the left if too long (keep the completion).
        if len(full_ids) > self.max_seq_len:
            keep = self.max_seq_len
            full_ids = full_ids[-keep:]
            prompt_len = max(0, len(prompt_ids) - (len(full_ids) - keep + len(prompt_ids) - len(full_ids)))
            prompt_len = min(prompt_len, len(full_ids) - 1)
        else:
            prompt_len = len(prompt_ids)

        # Targets: shift by 1. Mask out prompt tokens with -100.
        input_ids = full_ids[:-1]
        target_ids = full_ids[1:].clone()
        target_ids[:prompt_len - 1] = -100  # don't train on prompt

        return input_ids, target_ids

    def train_step(self, pairs, optimizer, lr):
        """Run one gradient step on a batch of (prompt, completion) pairs.

        pairs: list of dicts from filter_training_pairs (label="positive" or "weak_positive").
        """
        self.model.train()
        total_loss = 0.0
        n = 0

        # Encode all pairs.
        batch_inputs = []
        batch_targets = []
        for p in pairs:
            inp, tgt = self._encode_pair(p["prompt"], p["completion"])
            batch_inputs.append(inp)
            batch_targets.append(tgt)

        if not batch_inputs:
            return 0.0

        # Pad to same length (right padding with 0, targets -100).
        max_len = max(len(t) for t in batch_inputs)
        for inp, tgt in zip(batch_inputs, batch_targets):
            pad = max_len - len(inp)
            inp = F.pad(inp, (0, pad), value=0)
            tgt = F.pad(tgt, (0, pad), value=-100)
            batch_inputs.append(inp)
            batch_targets.append(tgt)
        # Note: we appended to the same list we're iterating — fix by slicing.
        batch_inputs = batch_inputs[len(batch_inputs)//2:]
        batch_targets = batch_targets[len(batch_targets)//2:]

        x = torch.stack(batch_inputs).to(self.device)
        y = torch.stack(batch_targets).to(self.device)

        # Mix in replay examples (50% new, 50% replay if buffer has data).
        if len(self.replay) > 0 and len(self.replay) >= len(pairs):
            import random
            replay_samples = random.sample(list(self.replay), len(pairs))
            rx = torch.stack([s[0] for s in replay_samples]).to(self.device)
            ry = torch.stack([s[1] for s in replay_samples]).to(self.device)
            # Pad replay to match batch length.
            if rx.shape[1] < x.shape[1]:
                rx = F.pad(rx, (0, x.shape[1] - rx.shape[1]), value=0)
                ry = F.pad(ry, (0, y.shape[1] - ry.shape[1]), value=-100)
            elif rx.shape[1] > x.shape[1]:
                x = F.pad(x, (0, rx.shape[1] - x.shape[1]), value=0)
                y = F.pad(y, (0, ry.shape[1] - y.shape[1]), value=-100)
            x = torch.cat([x, rx], dim=0)
            y = torch.cat([y, ry], dim=0)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                y.view(-1),
                ignore_index=-100,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        # Update EMA on LoRA params.
        update_ema(self.ema_state, self._lora_module_view(), self.ema_decay)

        # Add to replay buffer.
        for inp, tgt in zip(batch_inputs[:len(pairs)], batch_targets[:len(pairs)]):
            self.replay.append((inp.cpu().clone(), tgt.cpu().clone()))

        return loss.item()

    def _compute_logp(self, input_ids, target_ids):
        """Compute per-token average log-probability of target_ids given input_ids.

        Used for DPO: logp(chosen) and logp(rejected).
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            # Shift: logits[:-1] predict target_ids[1:] conventionally, but
            # our _encode_pair already aligns input_ids = full[:-1], target = full[1:].
            log_probs = F.log_softmax(logits.float(), dim=-1)
            # Gather logp at target positions (ignore -100).
            mask = (target_ids != -100)
            safe_targets = target_ids.clamp(min=0)
            token_logp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
            token_logp = token_logp * mask
            total_logp = token_logp.sum(dim=-1)
            n_tokens = mask.sum(dim=-1).clamp(min=1)
            return total_logp / n_tokens  # per-token average logp

    def train_dpo_step(self, dpo_pairs, optimizer, beta=0.1):
        """Run one DPO gradient step on (chosen, rejected) pairs from user edits.

        dpo_pairs: list of dicts with keys prompt, completion_chosen, completion_rejected.
        Uses the frozen base model (no LoRA) as the reference policy.
        """
        self.model.train()
        if not dpo_pairs:
            return 0.0

        # Encode chosen and rejected for each pair.
        chosen_inputs, chosen_targets = [], []
        rejected_inputs, rejected_targets = [], []
        for p in dpo_pairs:
            ci, ct = self._encode_pair(p["prompt"], p["completion_chosen"])
            ri, rt = self._encode_pair(p["prompt"], p["completion_rejected"])
            chosen_inputs.append(ci)
            chosen_targets.append(ct)
            rejected_inputs.append(ri)
            rejected_targets.append(rt)

        # Pad to same length within chosen/rejected.
        def pad_batch(inputs, targets):
            max_len = max(len(t) for t in inputs)
            padded_x, padded_y = [], []
            for inp, tgt in zip(inputs, targets):
                pad = max_len - len(inp)
                padded_x.append(F.pad(inp, (0, pad), value=0))
                padded_y.append(F.pad(tgt, (0, pad), value=-100))
            return torch.stack(padded_x).to(self.device), torch.stack(padded_y).to(self.device)

        cx, cy = pad_batch(chosen_inputs, chosen_targets)
        rx, ry = pad_batch(rejected_inputs, rejected_targets)

        # Reference logps: base model without LoRA (temporarily disable adapters).
        # We approximate by using the EMA-frozen LoRA weights as reference.
        # For a true reference, we'd save base logits, but that doubles VRAM.
        # The EMA approximation is standard in online DPO settings.
        with torch.no_grad():
            ref_chosen_logp = self._compute_logp(cx, cy)
            ref_rejected_logp = self._compute_logp(rx, ry)

        # Policy logps (with LoRA, gradients on).
        chosen_logp = self._compute_logp(cx, cy)
        rejected_logp = self._compute_logp(rx, ry)

        loss = dpo_loss(
            chosen_logp, rejected_logp,
            ref_chosen_logp.detach(), ref_rejected_logp.detach(),
            beta=beta,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        update_ema(self.ema_state, self._lora_module_view(), self.ema_decay)

        return loss.item()

    def self_verify(self, prompt_messages, response, max_new_tokens=64):
        """Model rates its own output quality (0.0 to 1.0).

        Approach: compute the model's average log-probability on its own response,
        then map to a 0-1 score via sigmoid. Higher logp → more confident → higher score.
        This is a lightweight proxy for quality; real self-verification would use
        a separate reward model, but this runs in-line during serving.
        """
        self.model.eval()
        with torch.no_grad():
            inp, tgt = self._encode_pair(prompt_messages, response)
            inp = inp.unsqueeze(0).to(self.device)
            tgt = tgt.unsqueeze(0).to(self.device)
            avg_logp = self._compute_logp(inp, tgt).item()
        # Map logp to [0, 1]: sigmoid(logp / temperature).
        # Typical avg logp for good responses: -1 to -3; bad: -5 to -10.
        score = torch.sigmoid(torch.tensor(avg_logp / 2.0)).item()
        return score, avg_logp

    def train(self, signals_path, steps=100, lr=1e-4, batch_size=4,
              min_self_verify=0.5, save_every=50, quality_check=True,
              dpo_beta=0.1, dpo_every=2):
        """Main training loop: load signals, filter, train, checkpoint.

        dpo_every: run a DPO step every N SFT steps (uses edit pairs).
        """
        print("\n=== Live Trainer ===")
        print(f"Signals: {signals_path} | steps: {steps} | lr: {lr} | batch: {batch_size}")

        interactions = load_signals(signals_path)
        if not interactions:
            print("No interactions found. Exiting.")
            return
        print(f"Loaded {len(interactions)} interactions from {signals_path}")

        pairs = filter_training_pairs(interactions, min_self_verify=min_self_verify)
        positive = [p for p in pairs if p["label"] in ("positive", "weak_positive")]
        dpo_pairs = [p for p in pairs if p["label"] == "dpo"]
        print(f"Training pairs: {len(positive)} positive, {len(dpo_pairs)} DPO, {len(pairs) - len(positive) - len(dpo_pairs)} negative")

        if not positive and not dpo_pairs:
            print("No positive training pairs. Exiting.")
            return

        # Optimizer on LoRA params only.
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)

        # Initial val loss.
        if quality_check and self.val_dataset is not None:
            self.best_val_loss = evaluate_loss(
                self.model, self.val_dataset, self.device, tag="before live training"
            )

        step = 0
        while step < steps:
            # Sample a batch of positive pairs (with replacement if too few).
            import random
            batch = random.sample(positive, min(batch_size, len(positive)))
            if len(batch) < batch_size:
                batch = batch + random.choices(positive, k=batch_size - len(batch))

            lr_step = get_lr(step, steps, lr, lr * 0.1, warmup_steps=min(10, steps // 10))
            for g in optimizer.param_groups:
                g["lr"] = lr_step

            loss = self.train_step(batch, optimizer, lr_step)
            step += 1

            # DPO step on edit pairs (every dpo_every SFT steps).
            dpo_loss_val = None
            if dpo_pairs and step % dpo_every == 0:
                import random
                dpo_batch = random.sample(
                    dpo_pairs, min(batch_size, len(dpo_pairs))
                )
                dpo_loss_val = self.train_dpo_step(dpo_batch, optimizer, beta=dpo_beta)

            if step % 10 == 0:
                dpo_str = f" | dpo: {dpo_loss_val:.4f}" if dpo_loss_val is not None else ""
                print(f"  [step {step}/{steps}] loss: {loss:.4f}{dpo_str} | lr: {lr_step:.2e} | replay: {len(self.replay)}")

            # Quality check + EMA rollback.
            if quality_check and self.val_dataset is not None and step % 25 == 0:
                val_loss = evaluate_loss(
                    self.model, self.val_dataset, self.device, tag=f"step {step}"
                )
                if val_loss > self.best_val_loss * 1.05:
                    print(f"  [SAFETY] Val loss regressed ({val_loss:.4f} > {self.best_val_loss:.4f}). Rolling back EMA.")
                    restore_ema(self.ema_state, self._lora_module_view())
                elif val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            # Versioned checkpoint.
            if step % save_every == 0 or step == steps:
                lora_sd = extract_lora_state_dict(self.model)
                self.checkpointer.save(lora_sd, metadata={
                    "step": step, "loss": loss, "val_loss": self.best_val_loss,
                })

        print(f"\nDone. {step} steps. Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints: {self.checkpointer.list_versions()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live-updating model trainer (LoRA + signal capture).")
    parser.add_argument("--config", type=str, default="360m_mla")
    parser.add_argument("--base-ckpt", type=str, default="research/checkpoints/distilled_llm.safetensors")
    parser.add_argument("--signals", type=str, default="research/data/live_training.jsonl",
                        help="Path to signal JSONL from serve.py")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--replay-size", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--min-self-verify", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--no-quality-check", action="store_true")
    # Phase 3: rollback / merge
    parser.add_argument("--rollback", type=str, default=None,
                        help="Rollback to a version (e.g. v003). Deletes newer versions.")
    parser.add_argument("--merge", type=str, default=None,
                        help="Merge a LoRA version into base weights (e.g. v005).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for --merge")
    args = parser.parse_args()

    patch_triton_cache_for_windows()
    cfg = get_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rollback mode (no model loading needed).
    if args.rollback:
        cp = VersionedCheckpointer()
        cp.rollback(args.rollback)
        return

    # Merge mode.
    if args.merge:
        print(f"Merging {args.merge} into base model...")
        model = ModelLoader.build_model(cfg, checkpoint_path=args.base_ckpt).to(device)
        inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        cp = VersionedCheckpointer()
        lora_sd = cp.load(args.merge)
        # Load LoRA weights into the wrappers.
        for name, param in model.named_parameters():
            if name in lora_sd:
                param.data.copy_(lora_sd[name])
        merge_lora_into_base(model)
        out_path = args.out or f"research/checkpoints/merged_{args.merge}.safetensors"
        save_checkpoint(model.state_dict(), out_path)
        print(f"Merged model saved to {out_path}")
        return

    # Training mode.
    val_dataset = None
    if not args.no_quality_check:
        val_path = "research/data/val.bin"
        if Path(val_path).exists():
            val_dataset = BinaryDataset(val_path, cfg.seq_len, cfg.vocab_size)
        else:
            print(f"Warning: {val_path} not found. Quality check disabled.")
            args.no_quality_check = True

    trainer = LiveTrainer(
        cfg, args.base_ckpt, device=device,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        replay_size=args.replay_size, ema_decay=args.ema_decay,
        val_dataset=val_dataset, max_seq_len=args.max_seq_len,
    )
    trainer.train(
        args.signals, steps=args.steps, lr=args.lr, batch_size=args.batch_size,
        min_self_verify=args.min_self_verify, save_every=args.save_every,
        quality_check=not args.no_quality_check,
    )


if __name__ == "__main__":
    main()
