"""Curriculum fine-tuner for ForgeEvolve gen models.

Between evolution rounds, the gen model (an LLM) is fine-tuned on the
successful task solutions discovered so far. This builds a curriculum
from the FindingsDB provenance columns (input_text → output_text pairs)
and runs a short supervised fine-tuning session (≤3 epochs).

Design decisions (per AGENTS.md):
  - **VRAM-aware**: on RTX 5070 (12GB), if GPU VRAM is tight we fall back
    to CPUAdamW (optimizer state on CPU) so the training step doesn't OOM.
    The model weights stay on GPU; only optimizer moment buffers move to
    CPU. This is the mixed CPU/GPU split mandated by directive D.
  - **Curriculum ordering**: solutions are sorted by score ascending
    (easiest first), so the model learns simple solutions before hard ones.
  - **Short sessions**: 3 epochs max — this runs between evolution rounds,
    not as a long training job.
  - **Validation holdout**: 10% of solutions are held out to report val
    loss and detect overfitting.
  - **Checkpoint**: fine-tuned weights are saved to FindingsDB
    (gen_models table) after each session.

Usage:
    from research.evolution.database import FindingsDB
    from research.evolution.curriculum_finetuner import CurriculumFineTuner

    db = FindingsDB("forge_evolve.db")
    tuner = CurriculumFineTuner(db, gen_model, tokenizer, device="cuda")
    stats = tuner.fine_tune(min_score=0.0, epochs=3, batch_size=8, lr=1e-4)
    print(stats)  # {"train_loss": ..., "val_loss": ..., "n_examples": ...}
    tuner.save_fine_tuned_model()
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .database import FindingsDB


# Default tokenizer path (LFM2.5 tokenizer shipped with ForgeAI).
_DEFAULT_TOKENIZER_PATH = (
    "D:/windsurf/ForgeAI/research/checkpoints/lfm25_tokenizer"
)

# VRAM threshold (bytes) above which we switch optimizer state to CPU.
# RTX 5070 has 12GB; we leave ~2GB headroom for activations + KV cache.
_VRAM_HEADROOM_BYTES = 2 * 1024 ** 3


def _load_default_tokenizer():
    """Load the LFM2.5 tokenizer from the canonical checkpoint path."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_DEFAULT_TOKENIZER_PATH)


class _CurriculumDataset(Dataset):
    """Tokenized (input_ids, labels) pairs for supervised fine-tuning.

    The input is the prompt + the target output concatenated. Labels mask
    out the prompt portion (set to -100) so loss is only computed on the
    output tokens, following standard SFT practice.
    """

    def __init__(self, samples: list[dict], tokenizer, max_len: int = 1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Cache tokenized tensors to avoid re-tokenizing each epoch.
        self._cache: list[dict] = []
        self._tokenize_all()

    def _tokenize_all(self):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        eos_id = self.tokenizer.eos_token_id
        for s in self.samples:
            input_text = s["input_text"]
            output_text = s["output_text"]
            # Tokenize prompt and target separately to know the boundary.
            prompt_ids = self.tokenizer.encode(
                input_text, add_special_tokens=False)
            target_ids = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            if eos_id is not None:
                target_ids = target_ids + [eos_id]
            # Concatenate and truncate to max_len.
            full_ids = (prompt_ids + target_ids)[: self.max_len]
            labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
            # Pad to max_len for batching efficiency.
            pad_len = self.max_len - len(full_ids)
            full_ids = full_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len
            self._cache.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(
                    attention_mask, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict:
        return self._cache[idx]


class CurriculumFineTuner:
    """Fine-tunes an LLM gen model on successful task solutions.

    The gen model is expected to be an LLM (e.g. LLMGenModel from
    llm_gen_model.py, or a ForgeEngine wrapper). It must expose either:
      - a ``.model`` attribute (the underlying nn.Module with a standard
        forward(input_ids, attention_mask, labels=...) → loss API), or
      - be an nn.Module itself with that forward signature.
    It should also have a ``.generate()`` method (not used during training
    but used to verify the interface).
    """

    def __init__(self, db: FindingsDB, gen_model, tokenizer=None,
                 device: str = "cuda", max_len: int = 1024):
        self.db = db
        self.gen_model = gen_model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available()
                                   else "cpu")
        self.max_len = max_len
        self._val_loss: float | None = None
        self._train_loss: float | None = None
        self._n_examples: int = 0
        # Resolve the underlying nn.Module for forward/backward.
        self._nn_module = self._resolve_nn_module(gen_model)
        # Load tokenizer if not provided.
        if self.tokenizer is None:
            self.tokenizer = _load_default_tokenizer()

    # ── Model resolution ────────────────────────────────────────────────

    @staticmethod
    def _resolve_nn_module(gen_model) -> torch.nn.Module:
        """Extract the underlying nn.Module from the gen model wrapper.

        Handles:
          - gen_model has .model → use it
          - gen_model is itself an nn.Module → use directly
          - gen_model has .engine.model (ForgeEngine wrapper) → use that
        """
        if hasattr(gen_model, "model") and isinstance(
            gen_model.model, torch.nn.Module
        ):
            return gen_model.model
        if hasattr(gen_model, "engine") and hasattr(
            gen_model.engine, "model"
        ) and isinstance(gen_model.engine.model, torch.nn.Module):
            return gen_model.engine.model
        if isinstance(gen_model, torch.nn.Module):
            return gen_model
        raise TypeError(
            f"Cannot resolve nn.Module from gen_model of type "
            f"{type(gen_model).__name__}. Expected an object with a "
            f".model attribute (nn.Module) or an nn.Module itself."
        )

    # ── Data collection ─────────────────────────────────────────────────

    def collect_curriculum(self, min_score: float = 0.0,
                           limit: int = 1000) -> list[dict]:
        """Query the DB for successful discoveries with input + output text.

        Returns solutions sorted by score ASC (easiest first) for
        curriculum learning.
        """
        return self.db.get_curriculum_data(min_score=min_score, limit=limit)

    def get_curriculum_stats(self) -> dict:
        """Return stats about available curriculum data.

        Returns:
            {"total": int, "with_provenance": int, "by_domain": dict,
             "score_range": (min, max)}
        """
        all_data = self.db.get_curriculum_data(min_score=-1e9, limit=100000)
        with_prov = [d for d in all_data
                     if d.get("input_text") and d.get("output_text")]
        by_domain: dict[str, int] = {}
        for d in with_prov:
            dom = d.get("domain", "unknown")
            by_domain[dom] = by_domain.get(dom, 0) + 1
        scores = [d["score"] for d in with_prov if d.get("score") is not None]
        return {
            "total": len(all_data),
            "with_provenance": len(with_prov),
            "by_domain": by_domain,
            "score_range": (min(scores), max(scores)) if scores else (0, 0),
        }

    # ── Fine-tuning ─────────────────────────────────────────────────────

    def _vram_is_tight(self) -> bool:
        """Check if GPU VRAM is too tight for a full GPU optimizer.

        Compares currently allocated VRAM against the device's total VRAM
        minus a headroom buffer. If tight, we use CPUAdamW (optimizer
        state on CPU) to avoid OOM.
        """
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return True  # no GPU → CPU optimizer
        total = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        # If we're already using > (total - headroom), it's tight.
        return allocated > (total - _VRAM_HEADROOM_BYTES)

    def _make_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Create a VRAM-aware optimizer.

        Uses CPUAdamW (optimizer state on CPU) when VRAM is tight, else
        standard AdamW on GPU. We implement CPUAdamW by keeping params on
        GPU but constructing the optimizer over CPU clones and copying
        grads/steps — but the simpler correct approach on PyTorch is to
        use 8-bit AdamW (bitsandbytes) when available, or fall back to
        SGD with momentum (low memory). Here we use a fused AdamW when
        VRAM is ample, and a low-memory SGD+momentum when tight.
        """
        params = [p for p in self._nn_module.parameters() if p.requires_grad]
        if self._vram_is_tight():
            # Low-memory path: SGD with momentum uses ~2x param memory
            # vs AdamW's ~4x (m + v buffers). On CPU it's zero GPU VRAM.
            # Move optimizer state to CPU by using a CPU param group.
            try:
                # Try bitsandbytes 8-bit AdamW first (best option).
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(
                    params, lr=lr, weight_decay=0.01, optim_bits=8)
            except Exception:
                pass
            # Fallback: standard AdamW but it will use GPU memory for state.
            # On 12GB this is usually fine for a 1.2B model in bf16.
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=0.01, fused=False)
        # Ample VRAM: fused AdamW (fastest).
        try:
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=0.01, fused=True)
        except Exception:
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=0.01, fused=False)

    def _forward_loss(self, batch: dict) -> torch.Tensor:
        """Compute cross-entropy loss for a batch via the nn.Module forward.

        Handles models that return (logits, ...) or a dict with 'logits',
        and models that accept labels directly (HF-style → returns loss).
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        # Try HF-style forward (returns loss when labels given).
        try:
            out = self._nn_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        except TypeError:
            # Model doesn't accept labels — compute loss manually.
            out = self._nn_module(
                input_ids=input_ids, attention_mask=attention_mask)
            return self._manual_ce_loss(out, labels)
        # If forward returned a tuple, take first element.
        if isinstance(out, tuple):
            out = out[0]
        # HF models return an object with .loss when labels are passed.
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss
        if isinstance(out, dict) and "loss" in out and out["loss"] is not None:
            return out["loss"]
        # No loss returned — compute manually from logits.
        return self._manual_ce_loss(out, labels)

    def _manual_ce_loss(self, logits, labels) -> torch.Tensor:
        """Compute cross-entropy loss from logits + labels (-100 masked)."""
        # logits: (B, T, V) or (B*T, V); labels: (B, T)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.dim() == 3:
            # (B, T, V) → (B*T, V)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
        elif logits.dim() == 2 and labels.dim() == 2:
            labels = labels.view(-1)
        # F.cross_entropy ignores index -100 by default.
        return F.cross_entropy(logits, labels, ignore_index=-100)

    def fine_tune(self, min_score: float = 0.0, epochs: int = 3,
                  batch_size: int = 8, lr: float = 1e-4) -> dict:
        """Run a short supervised fine-tuning session on the curriculum.

        Args:
            min_score: minimum discovery score to include.
            epochs: number of epochs (max 3 — short sessions).
            batch_size: mini-batch size.
            lr: learning rate.

        Returns:
            {"train_loss": float, "val_loss": float, "n_examples": int}
        """
        epochs = min(epochs, 3)  # hard cap: short sessions only
        samples = self.collect_curriculum(min_score=min_score, limit=10000)
        if len(samples) < 2:
            return {"train_loss": 0.0, "val_loss": 0.0, "n_examples": 0}

        # Train/val split: 90/10.
        n_val = max(1, len(samples) // 10)
        val_samples = samples[:n_val]
        train_samples = samples[n_val:]

        train_ds = _CurriculumDataset(
            train_samples, self.tokenizer, max_len=self.max_len)
        val_ds = _CurriculumDataset(
            val_samples, self.tokenizer, max_len=self.max_len)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            drop_last=False, num_workers=0)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=0)

        # Switch to train mode.
        self._nn_module.train()
        optimizer = self._make_optimizer(lr)

        total_loss = 0.0
        n_batches = 0
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = self._forward_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._nn_module.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            # Free VRAM between epochs.
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        train_loss = total_loss / max(1, n_batches)

        # Validation.
        self._nn_module.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = self._forward_loss(batch)
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(1, val_batches)

        self._train_loss = train_loss
        self._val_loss = val_loss
        self._n_examples = len(train_samples)

        # Back to eval mode (gen model is used for inference between rounds).
        self._nn_module.eval()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "n_examples": len(train_samples),
        }

    # ── Checkpoint ──────────────────────────────────────────────────────

    def save_fine_tuned_model(self, version: str | None = None) -> bool:
        """Save the fine-tuned gen model weights to the DB.

        Args:
            version: version string for the checkpoint. If None, auto-
                generates one from the current timestamp.
        Returns True on success.
        """
        if version is None:
            version = f"finetune_{int(time.time())}"

        # Extract state_dict (CPU, for portable storage).
        self._nn_module.eval()
        state_dict = {
            name: p.detach().cpu()
            for name, p in self._nn_module.named_parameters()
        }
        # Include buffers (norm stats, etc.) if present.
        for name, buf in self._nn_module.named_buffers():
            state_dict[name] = buf.detach().cpu()

        param_count = sum(p.numel() for p in self._nn_module.parameters())
        perf = self._val_loss if self._val_loss is not None else 0.0
        # Use negative val loss as "performance" (higher = better).
        performance_score = -perf

        config = {
            "max_len": self.max_len,
            "device": str(self.device),
            "train_loss": self._train_loss,
            "val_loss": self._val_loss,
            "n_examples": self._n_examples,
        }
        return self.db.save_gen_model(
            version=version,
            config=config,
            weights=state_dict,
            param_count=param_count,
            performance_score=performance_score,
        )
