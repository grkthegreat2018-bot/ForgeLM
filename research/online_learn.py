"""Full online learning: main model adapts weights during serving.

EXTENDS speculative decoding with main-model weight updates. The main model
learns from:
  1. User interaction signals (which responses are good/bad via implicit feedback)
  2. Self-generated training data (the model generates text, then trains on it)
  3. Teacher corrections (when the draft is rejected, the main model's own
     verification logits become a training signal)

SAFETY MECHANISMS:
  - Replay buffer: mix in old training data to prevent catastrophic forgetting
  - EMA shadow weights: if online learning degrades quality, roll back to EMA
  - Low learning rate (1e-5 to 1e-6): gentle updates, no sharp divergence
  - Gradient clipping: prevent outlier gradients from destabilizing weights
  - Quality monitor: track val loss every N steps; halt if it regresses > threshold

Usage:
    python -m research.online_learn --model research/checkpoints/distilled_llm.safetensors \
        --lr 1e-5 --replay-buffer-size 1000 --max-steps 10000
"""
import argparse
import math
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from research.config import get_config
from research.model_loader import ModelLoader
from research.training_utils import (
    BinaryDataset,
    add_safeguard_args,
    configure_optimizer,
    has_nan_params,
    init_ema,
    restore_ema,
    update_ema,
    get_lr,
    patch_triton_cache_for_windows,
    vram_exceeded,
    write_heartbeat,
    write_status_json,
)
from research.checkpoint_io import (
    cleanup_step_checkpoints,
    emergency_save,
    load_training_state,
    save_checkpoint,
    save_training_checkpoint,
    step_checkpoint_path,
)


class ReplayBuffer:
    """Circular buffer of recent training examples to prevent catastrophic forgetting.

    Stores (input_ids, target_ids) pairs. During online learning, each batch
    is mixed with replay examples to ensure the model doesn't forget old knowledge.
    """

    def __init__(self, max_size=1000, device="cuda"):
        self.buffer = deque(maxlen=max_size)
        self.device = device

    def add(self, x, y):
        """Add a training example (x: input, y: targets)."""
        self.buffer.append((x.cpu().clone(), y.cpu().clone()))

    def sample(self, n):
        """Sample n replay examples."""
        if len(self.buffer) == 0 or n <= 0:
            return None, None
        samples = random.sample(list(self.buffer), min(n, len(self.buffer)))
        xs = torch.cat([s[0] for s in samples], dim=0).to(self.device)
        ys = torch.cat([s[1] for s in samples], dim=0).to(self.device)
        return xs, ys

    def __len__(self):
        return len(self.buffer)


class OnlineLearner:
    """Online learning system for the main model during serving.

    The model updates its weights in real-time based on:
    1. Self-supervised loss (next-token prediction on its own outputs)
    2. Replay buffer (old data to prevent forgetting)
    3. EMA shadow weights (safety net for quality regression)

    Quality is monitored via val loss. If val loss regresses beyond a threshold,
    learning is paused and EMA weights are restored.
    """

    def __init__(self, model, cfg, lr=1e-5, ema_decay=0.9999,
                 replay_size=1000, quality_threshold=0.5, device="cuda"):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.lr = lr
        self.ema_decay = ema_decay
        self.quality_threshold = quality_threshold  # max allowed val loss increase

        # Optimizer for online updates.
        self.optimizer = configure_optimizer(model, lr, cfg.weight_decay, "bnb")

        # EMA shadow weights (safety net).
        self.ema_state = init_ema(model)

        # Replay buffer.
        self.replay = ReplayBuffer(max_size=replay_size, device=device)

        # Quality tracking.
        self.val_dataset = BinaryDataset("research/data/val.bin", cfg.seq_len, cfg.vocab_size)
        self.best_val_loss = float("inf")
        self.steps_without_improvement = 0
        self.learning_paused = False

        # Stats.
        self.total_steps = 0
        self.total_replay_used = 0

    def _update_ema(self):
        """Update EMA shadow weights."""
        update_ema(self.ema_state, self.model, self.ema_decay)

    def _restore_ema(self):
        """Restore EMA weights (rollback if quality degrades)."""
        print("  [SAFETY] Restoring EMA weights due to quality regression.")
        restore_ema(self.ema_state, self.model)

    def _compute_val_loss(self):
        """Quick val loss on a few batches."""
        self.model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(5):
                x, y = self.val_dataset.get_batch(2, self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self.model(x)
                    logits = out[0] if isinstance(out, tuple) else out
                    losses.append(F.cross_entropy(
                        logits.view(-1, logits.size(-1)).float(), y.view(-1)).item())
        self.model.train()
        return sum(losses) / len(losses)

    def _check_quality(self):
        """Check if online learning is degrading quality. Pause if so."""
        val_loss = self._compute_val_loss()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.steps_without_improvement = 0
            return val_loss, False
        else:
            self.steps_without_improvement += 1
            regression = val_loss - self.best_val_loss
            if regression > self.quality_threshold:
                self._restore_ema()
                self.learning_paused = True
                print(f"  [SAFETY] Learning PAUSED. Val loss {val_loss:.4f} > best {self.best_val_loss:.4f} + {self.quality_threshold}")
            return val_loss, True

    def learn_step(self, x, y, use_replay=True, replay_ratio=0.25):
        """One online learning step.

        Args:
            x: input tokens [B, T]
            y: target tokens [B, T]
            use_replay: mix in replay buffer examples
            replay_ratio: fraction of batch from replay buffer

        Returns: loss value
        """
        if self.learning_paused:
            return 0.0

        # Add to replay buffer.
        self.replay.add(x, y)

        # Mix with replay examples to prevent forgetting.
        if use_replay and len(self.replay) > 4:
            n_replay = max(1, int(x.shape[0] * replay_ratio))
            replay_x, replay_y = self.replay.sample(n_replay)
            if replay_x is not None and replay_x.shape[0] > 0:
                x = torch.cat([x, replay_x], dim=0)
                y = torch.cat([y, replay_y], dim=0)
                self.total_replay_used += replay_x.shape[0]

        # Forward + backward.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self._update_ema()

        self.total_steps += 1
        return loss.item()

    def save(self, path, use_ema=True, step=None):
        """Save model (optionally EMA weights) + full training state (resumable)."""
        if use_ema:
            # Swap EMA weights in, save, then swap the live weights back.
            raw = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            restore_ema(self.ema_state, self.model)
            save_training_checkpoint(self.model, path, optimizer=self.optimizer,
                                     ema_state=self.ema_state, step=step)
            with torch.no_grad():
                for k, v in self.model.state_dict().items():
                    v.copy_(raw[k])
        else:
            save_training_checkpoint(self.model, path, optimizer=self.optimizer,
                                     ema_state=self.ema_state, step=step)
        print(f"Saved {'EMA ' if use_ema else ''}model to {path}")


def main():
    p = argparse.ArgumentParser(description="Online learning for the main model")
    p.add_argument("--model", default="research/checkpoints/distilled_llm.safetensors")
    p.add_argument("--config", default="360m_mla")
    p.add_argument("--lr", type=float, default=1e-5, help="Online learning rate (very low)")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--replay-buffer-size", type=int, default=1000)
    p.add_argument("--quality-threshold", type=float, default=0.5, help="Max val loss increase before pause")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--check-every", type=int, default=50, help="Check val loss every N steps")
    p.add_argument("--save", default="research/checkpoints/online_learned.safetensors")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume weights + training state from this checkpoint")
    add_safeguard_args(p)
    args = p.parse_args()

    patch_triton_cache_for_windows()
    device = torch.device("cuda")

    cfg = get_config(args.config)
    cfg.seq_len = args.seq_len
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    weights_path = args.resume or args.model
    print(f"Loading model from {weights_path}...")
    model = ModelLoader.build_model(cfg, checkpoint_path=weights_path).to(device)
    model.train()

    learner = OnlineLearner(
        model, cfg, lr=args.lr, ema_decay=args.ema_decay,
        replay_size=args.replay_buffer_size, quality_threshold=args.quality_threshold,
        device=device,
    )

    # Resume training state (optimizer + EMA + RNG + step) if requested.
    start_step = 0
    if args.resume:
        ts = load_training_state(args.resume, optimizer=learner.optimizer)
        if ts["ema"] is not None:
            learner.ema_state = {k: v.to(device) for k, v in ts["ema"].items()}
            print("Restored EMA shadow weights from checkpoint.")
        if ts["step"] is not None:
            start_step = ts["step"] + 1
            learner.total_steps = ts["step"]
            print(f"Resuming from {args.resume} at step {start_step}")

    # Ctrl-C -> emergency checkpoint before dying.
    current = {"step": start_step}

    def _sigint_handler(sig, frame):
        emergency_save(model, args.save, "interrupt", current["step"],
                       optimizer=learner.optimizer, ema_state=learner.ema_state)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    # Use training data as the "stream" of new examples.
    train_ds = BinaryDataset("research/data/train.bin", args.seq_len, cfg.vocab_size)

    # Initial val loss.
    initial_val = learner._compute_val_loss()
    learner.best_val_loss = initial_val
    print(f"\nInitial val loss: {initial_val:.4f} | ppl: {math.exp(initial_val):.2f}")
    print(f"Online learning: lr={args.lr}, ema={args.ema_decay}, replay={args.replay_buffer_size}")
    print(f"Quality threshold: {args.quality_threshold} (will pause + restore EMA if exceeded)\n")

    aborted = False
    t0 = time.time()
    for step in range(start_step + 1, args.max_steps + 1):
        if vram_exceeded(args.vram_limit_gb, device):
            print("VRAM limit exceeded; emergency save + abort.")
            emergency_save(model, args.save, "emergency", step,
                           optimizer=learner.optimizer, ema_state=learner.ema_state)
            aborted = True
            break
        current["step"] = step

        # Get a new batch from the "stream".
        x, y = train_ds.get_batch(2, device)

        # Learn.
        loss = learner.learn_step(x, y, use_replay=True, replay_ratio=0.25)

        if step % 50 == 0 and has_nan_params(model):
            print("NaN/Inf detected in parameters; emergency save + abort.")
            emergency_save(model, args.save, "nan", step,
                           optimizer=learner.optimizer, ema_state=learner.ema_state)
            aborted = True
            break

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt = step_checkpoint_path(args.save, step)
            save_training_checkpoint(model, ckpt, optimizer=learner.optimizer,
                                     ema_state=learner.ema_state, step=step)
            cleanup_step_checkpoints(args.save, args.keep_checkpoints)

        if step % 20 == 0:
            elapsed = time.perf_counter() - t0
            tok_s = step * 2 * args.seq_len / elapsed
            print(f"Step {step:4d}/{args.max_steps} | loss {loss:.4f} | {tok_s:.0f} tok/s | replay_used {learner.total_replay_used}")
            write_status_json(args.status_file, {
                "step": step, "max_steps": args.max_steps, "loss": loss,
                "replay_used": learner.total_replay_used,
                "learning_paused": learner.learning_paused,
            })
            write_heartbeat(args.heartbeat_file)

        # Quality check.
        if step % args.check_every == 0:
            val_loss, regressed = learner._check_quality()
            status = "REGRESSED" if regressed else "OK"
            print(f"  [Quality check] val {val_loss:.4f} | best {learner.best_val_loss:.4f} | {status}")

        if learner.learning_paused:
            print("Learning paused due to quality regression. Saving EMA weights.")
            break

    # Final save.
    if not aborted:
        learner.save(args.save, use_ema=True, step=learner.total_steps)
    final_val = learner._compute_val_loss()
    print(f"\nFinal val loss: {final_val:.4f} | ppl: {math.exp(final_val):.2f}")
    print(f"Change: {initial_val:.4f} -> {final_val:.4f} ({(math.exp(final_val)/math.exp(initial_val)-1)*100:+.1f}% ppl)")


if __name__ == "__main__":
    main()
