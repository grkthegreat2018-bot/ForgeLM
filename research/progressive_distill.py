"""Progressive Distillation — train student from successive teacher checkpoints.

Instead of distilling from a single final teacher, the student learns from
a sequence of teacher checkpoints saved during the teacher's own training.
This creates an implicit curriculum: early teacher checkpoints are simpler,
later ones are more complex. The student trains at the same speed as the
larger model but converges to better quality (ICLR 2025).

Two modes:
1. from_checkpoint_series: distill from a list of teacher checkpoints
2. from_training_run: save teacher checkpoints during training, distill after

Usage:
    from research.progressive_distill import ProgressiveDistiller

    # Mode 1: from existing checkpoint series
    pd = ProgressiveDistiller(student_model, teacher_checkpoints, tokenizer)
    pd.train(steps=5000)

    # Mode 2: during teacher training, save checkpoints periodically
    # Then distill from the series
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional


class ProgressiveDistiller:
    """Progressive distillation from a series of teacher checkpoints.

    Args:
        student_model: the student model to train
        teacher_checkpoints: list of paths to teacher checkpoints (in order)
        tokenizer: tokenizer
        temperature: distillation temperature
        alpha: weight for KD loss (1-alpha for CE loss)
        device: cuda or cpu
    """

    def __init__(self, student_model, teacher_checkpoints: List[str],
                 tokenizer, temperature=2.0, alpha=0.5, device="cuda"):
        self.student = student_model
        self.teacher_checkpoints = teacher_checkpoints
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.current_teacher = None
        self.current_ckpt_idx = 0

    def _load_teacher(self, ckpt_path: str, model_builder):
        """Load a teacher checkpoint into a model instance.

        Args:
            ckpt_path: path to checkpoint
            model_builder: callable that returns an empty model instance
        """
        from research.checkpoint_io import load_checkpoint
        if self.current_teacher is None:
            self.current_teacher = model_builder()
            self.current_teacher = self.current_teacher.to(self.device)

        state = load_checkpoint(ckpt_path, map_location=self.device)
        self.current_teacher.load_state_dict(state, strict=False)
        self.current_teacher.eval()
        for p in self.current_teacher.parameters():
            p.requires_grad_(False)

    def train(self, train_data, steps=5000, lr=1e-4,
              steps_per_checkpoint=None, model_builder=None):
        """Train the student progressively through teacher checkpoints.

        Args:
            train_data: list of (input_ids, target_ids) or DataLoader
            steps: total training steps
            lr: learning rate
            steps_per_checkpoint: steps before switching to next teacher
                                  (default: steps / len(teacher_checkpoints))
            model_builder: callable returning empty teacher model instance
        """
        if model_builder is None:
            raise ValueError("model_builder (callable returning empty teacher) is required")

        if steps_per_checkpoint is None:
            steps_per_checkpoint = steps // len(self.teacher_checkpoints)

        optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad], lr=lr
        )

        # Cosine LR schedule across the full training.
        def get_lr(step):
            progress = step / steps
            return lr * 0.5 * (1 + math.cos(progress * 3.14159))

        import math
        import random

        step = 0
        ckpt_idx = 0
        self._load_teacher(self.teacher_checkpoints[0], model_builder)
        print(f"[Progressive Distill] Stage 1/{len(self.teacher_checkpoints)}: "
              f"{self.teacher_checkpoints[0]}")

        self.student.train()
        while step < steps:
            # Check if we should switch to next teacher checkpoint.
            target_idx = min(step // steps_per_checkpoint,
                            len(self.teacher_checkpoints) - 1)
            if target_idx != ckpt_idx:
                ckpt_idx = target_idx
                self._load_teacher(self.teacher_checkpoints[ckpt_idx], model_builder)
                print(f"[Progressive Distill] Stage {ckpt_idx+1}/{len(self.teacher_checkpoints)}: "
                      f"{self.teacher_checkpoints[ckpt_idx]} (step {step})")

            # Get a batch.
            if isinstance(train_data, list):
                input_ids, target_ids = random.choice(train_data)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    target_ids = target_ids.unsqueeze(0)
            else:
                input_ids, target_ids = next(iter(train_data))

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Teacher forward (no grad).
            with torch.no_grad():
                t_out = self.current_teacher(input_ids)
                t_logits = t_out[0] if isinstance(t_out, tuple) else t_out
                t_soft = F.softmax(t_logits / self.temperature, dim=-1)

            # Student forward.
            s_out = self.student(input_ids)
            s_logits = s_out[0] if isinstance(s_out, tuple) else s_out

            # KD loss: KL divergence with teacher soft targets.
            s_log_soft = F.log_softmax(s_logits / self.temperature, dim=-1)
            kd_loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean") * (self.temperature ** 2)

            # CE loss: standard next-token.
            ce_loss = F.cross_entropy(
                s_logits.view(-1, s_logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-100,
            )

            # Combined loss.
            loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

            # Update.
            for g in optimizer.param_groups:
                g["lr"] = get_lr(step)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            if step % 100 == 0:
                print(f"  [PD] step {step}/{steps} | stage {ckpt_idx+1} | "
                      f"kd={kd_loss.item():.4f} ce={ce_loss.item():.4f} | lr={get_lr(step):.2e}")

        print(f"[Progressive Distill] complete: {step} steps, {len(self.teacher_checkpoints)} stages")


def save_teacher_series(model, train_fn, save_dir, n_checkpoints=5,
                        steps_between=1000):
    """Run teacher training and save checkpoints periodically.

    Args:
        model: the teacher model
        train_fn: callable(model, steps) that trains the model
        save_dir: directory to save checkpoints
        n_checkpoints: number of checkpoints to save
        steps_between: training steps between checkpoints

    Returns:
        list of checkpoint paths (in order)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from research.checkpoint_io import save_checkpoint

    paths = []
    for i in range(n_checkpoints):
        print(f"[Teacher Series] Training stage {i+1}/{n_checkpoints} ({steps_between} steps)...")
        train_fn(model, steps_between)
        path = save_dir / f"teacher_stage_{i:03d}.safetensors"
        save_checkpoint(model.state_dict(), str(path))
        paths.append(str(path))
        print(f"[Teacher Series] Saved {path}")

    return paths
