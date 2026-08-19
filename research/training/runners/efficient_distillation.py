"""Efficient distillation: offline top-K logits + chunked KL + truncation + prefix OPD.

Based on four 2026 papers:
  1. Offline Top-K Logits + Chunked KL (arXiv 2608.03796):
     - Cache teacher's top-K logits once → train student against cache
     - 29% faster per iteration, 41% higher throughput
     - Fused chunked KL loss: peak memory linear in seq length (4× context)
  2. Distilling the Essence (ACL 2026 Findings):
     - Train on first 50% of tokens → 91% of full-sequence performance
     - 50% reduction in training time, memory, FLOPs
  3. On-Policy Prefix Distillation (ACL 2026 Findings):
     - Apply distillation only to prefixes of student-generated outputs
     - Early-terminate sampling → 2-40× FLOP reduction
  4. SODA (arXiv 2604.03873): semi on-policy black-box distillation.
     - Static student snapshot for contrastive signal
     - No dynamic rollouts or adversarial training

For our self-play → SFT pipeline:
  - Offline top-K: cache teacher (larger model) logits for training data
  - Chunked KL: enable 32K context distillation on 12GB GPU
  - Truncation: train on first 50% of CoT → 2× faster
  - Prefix OPD: distill only reasoning prefixes → massive FLOP savings
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional
import math


class OfflineTopKLogits:
    """Offline top-K logits caching for efficient distillation.

    Instead of running the teacher model online (expensive), cache the
    teacher's top-K logits for each token once, then train the student
    against the cache.

    Benefits:
      - 29% faster per iteration (no teacher forward pass)
      - 41% higher throughput (teacher not in memory)
      - Can run hundreds of ablations against same cached targets
    """

    def __init__(self, k: int = 50, cache_dir: str | None = None):
        self.k = k  # number of top logits to cache
        self.cache_dir = cache_dir
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def cache_teacher_logits(self, teacher_model: torch.nn.Module,
                              input_ids: torch.Tensor,
                              batch_idx: int = 0):
        """Run teacher model and cache top-K logits.

        Args:
            teacher_model: the teacher model (frozen)
            input_ids: (B, T) input token IDs
            batch_idx: cache key
        """
        with torch.inference_mode():
            outputs = teacher_model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            # (B, T, V)

        # Get top-K logits and indices
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)
        # (B, T, K)

        self._cache[batch_idx] = (topk_vals.cpu(), topk_idx.cpu())

    def get_cached_logits(self, batch_idx: int,
                          device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached top-K logits.

        Returns:
            topk_vals: (B, T, K) top-K logit values
            topk_idx: (B, T, K) top-K token indices
        """
        if batch_idx not in self._cache:
            raise KeyError(f"Batch {batch_idx} not cached")
        vals, idx = self._cache[batch_idx]
        return vals.to(device), idx.to(device)

    def compute_distill_loss(self, student_logits: torch.Tensor,
                             batch_idx: int,
                             temperature: float = 2.0) -> torch.Tensor:
        """Compute distillation loss against cached teacher top-K logits.

        Uses KL divergence on the top-K subspace (much cheaper than full V).

        Args:
            student_logits: (B, T, V) student model logits
            batch_idx: cache key
            temperature: softmax temperature

        Returns:
            loss: KL divergence loss
        """
        teacher_vals, teacher_idx = self.get_cached_logits(batch_idx,
                                                            student_logits.device)

        # Teacher: softmax over top-K
        teacher_probs = F.softmax(teacher_vals / temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_vals / temperature, dim=-1)

        # Student: gather logits at the same top-K positions
        student_topk = torch.gather(
            student_logits / temperature, -1, teacher_idx)  # (B, T, K)
        student_log_probs = F.log_softmax(student_topk, dim=-1)

        # KL divergence: teacher || student
        kl = teacher_probs * (teacher_log_probs - student_log_probs)
        loss = kl.sum(dim=-1).mean() * (temperature ** 2)

        return loss


class ChunkedKLLoss:
    """Fused chunked KL loss: peak memory linear in sequence length.

    Standard KL loss materializes the full (B, T, V) logit tensor → memory
    spike that caps context length. Chunked KL processes the sequence in
    chunks, never materializing the full tensor.

    Result: 4× longer context on same GPU (32K → 128K on single H200).
    """

    def __init__(self, chunk_size: int = 512, temperature: float = 2.0):
        self.chunk_size = chunk_size
        self.temperature = temperature

    def compute(self, student_logits_fn, teacher_logits_fn,
                T: int, vocab_size: int,
                device: str = "cuda") -> torch.Tensor:
        """Compute chunked KL loss.

        Args:
            student_logits_fn: function(chunk_start, chunk_end) → (B, chunk_T, V)
            teacher_logits_fn: function(chunk_start, chunk_end) → (B, chunk_T, V)
            T: total sequence length
            vocab_size: vocabulary size
            device: torch device

        Returns:
            loss: KL divergence loss (scalar)
        """
        total_loss = 0.0
        n_chunks = 0

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)

            # Get logits for this chunk only (no full materialization)
            student_chunk = student_logits_fn(start, end)
            teacher_chunk = teacher_logits_fn(start, end)

            # KL divergence for this chunk
            student_log_probs = F.log_softmax(
                student_chunk / self.temperature, dim=-1)
            teacher_probs = F.softmax(
                teacher_chunk / self.temperature, dim=-1)

            kl = teacher_probs * (
                F.log_softmax(teacher_chunk / self.temperature, dim=-1) -
                student_log_probs
            )
            total_loss += kl.sum(dim=-1).mean()
            n_chunks += 1

        return total_loss / max(n_chunks, 1) * (self.temperature ** 2)


class SequenceTruncationDistillation:
    """Sequence truncation for efficient reasoning distillation.

    Train on only the first 50% of tokens of each training sequence:
      - 91% of full-sequence performance on math benchmarks
      - 50% reduction in training time, memory, FLOPs

    Insight: beyond a specific length, longer sequences provide marginal
    returns but require substantially more resources.
    """

    def __init__(self, truncation_ratio: float = 0.5,
                 truncate_from: str = "end"):
        """
        Args:
            truncation_ratio: fraction of tokens to keep (0.5 = first half)
            truncate_from: 'end' (keep prefix) or 'start' (keep suffix)
        """
        self.ratio = truncation_ratio
        self.from_side = truncate_from

    def truncate(self, input_ids: torch.Tensor,
                 labels: torch.Tensor,
                 attention_mask: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Truncate sequences to the specified ratio.

        Args:
            input_ids: (B, T) token IDs
            labels: (B, T) labels
            attention_mask: (B, T) attention mask

        Returns:
            truncated input_ids, labels, attention_mask
        """
        T = input_ids.shape[1]
        keep_len = int(T * self.ratio)

        if self.from_side == "end":
            # Keep prefix (first keep_len tokens)
            trunc_ids = input_ids[:, :keep_len]
            trunc_labels = labels[:, :keep_len]
            trunc_mask = attention_mask[:, :keep_len] if attention_mask is not None else None
        else:
            # Keep suffix (last keep_len tokens)
            trunc_ids = input_ids[:, T - keep_len:]
            trunc_labels = labels[:, T - keep_len:]
            trunc_mask = attention_mask[:, T - keep_len:] if attention_mask is not None else None

        return trunc_ids, trunc_labels, trunc_mask


class OnPolicyPrefixDistillation:
    """On-policy prefix distillation: distill only reasoning prefixes.

    Key insight: training signals are stronger in the PREFIX of each output
    reasoning trace. Even a short teacher-generated prefix can significantly
    help the student produce the correct answer.

    Method:
      1. Sample trajectories from the student model
      2. Apply distillation objective only to prefixes
      3. Terminate sampling early (don't need full trajectory)

    Result: 2-40× FLOP reduction while matching full OPD performance.
    """

    def __init__(self, prefix_ratio: float = 0.3,
                 temperature: float = 2.0,
                 early_terminate: bool = True):
        self.prefix_ratio = prefix_ratio
        self.temperature = temperature
        self.early_terminate = early_terminate

    def compute_loss(self, student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute prefix distillation loss.

        Args:
            student_logits: (B, T, V) student logits for sampled trajectory
            teacher_logits: (B, T, V) teacher logits for same trajectory
            mask: (B, T) validity mask

        Returns:
            loss: KL divergence loss on prefix only
        """
        T = student_logits.shape[1]
        prefix_len = int(T * self.prefix_ratio)

        # Only use prefix
        student_prefix = student_logits[:, :prefix_len]
        teacher_prefix = teacher_logits[:, :prefix_len]

        # KL divergence
        student_log_probs = F.log_softmax(
            student_prefix / self.temperature, dim=-1)
        teacher_probs = F.softmax(
            teacher_prefix / self.temperature, dim=-1)

        kl = teacher_probs * (
            F.log_softmax(teacher_prefix / self.temperature, dim=-1) -
            student_log_probs
        )
        loss = kl.sum(dim=-1).mean() * (self.temperature ** 2)

        if mask is not None:
            prefix_mask = mask[:, :prefix_len]
            loss = loss * prefix_mask.mean()

        return loss

    def should_terminate(self, current_len: int, max_len: int) -> bool:
        """Check if early termination should happen."""
        if not self.early_terminate:
            return current_len >= max_len
        return current_len >= int(max_len * self.prefix_ratio)


class SODADistillation:
    """SODA: Semi On-Policy Distillation with Alignment.

    Uses a static snapshot of the student's responses as a contrastive
    signal against the teacher's superior responses. No dynamic rollouts
    or adversarial training needed.
    """

    def __init__(self, alignment_weight: float = 0.3,
                 temperature: float = 2.0):
        self.alignment_weight = alignment_weight
        self.temperature = temperature
        self._student_snapshot_logits: torch.Tensor | None = None

    def snapshot_student(self, student_model: torch.nn.Module,
                          input_ids: torch.Tensor):
        """Take a one-time static snapshot of student's responses."""
        with torch.inference_mode():
            outputs = student_model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        self._student_snapshot_logits = logits.cpu()

    def compute_loss(self, student_logits: torch.Tensor,
                     teacher_logits: torch.Tensor,
                     input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Compute SODA contrastive distillation loss.

        loss = KL(teacher || student) - λ * KL(snapshot || student)

        The first term aligns student to teacher (standard KD).
        The second term pushes student AWAY from its inferior snapshot
        (contrastive alignment).
        """
        # Standard KD: align to teacher
        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=-1)
        kd_loss = (teacher_probs * (
            F.log_softmax(teacher_logits / self.temperature, dim=-1) -
            student_log_probs
        )).sum(dim=-1).mean()

        # Contrastive: push away from snapshot
        if self._student_snapshot_logits is not None:
            snapshot = self._student_snapshot_logits.to(student_logits.device)
            snapshot_probs = F.softmax(snapshot / self.temperature, dim=-1)
            # Maximize KL(student || snapshot) → push student away from snapshot
            contrast_loss = (student_log_probs.exp() * (
                student_log_probs -
                F.log_softmax(snapshot / self.temperature, dim=-1)
            )).sum(dim=-1).mean()
        else:
            contrast_loss = torch.tensor(0.0, device=student_logits.device)

        loss = kd_loss - self.alignment_weight * contrast_loss
        return loss * (self.temperature ** 2)
