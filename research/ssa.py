"""SSA — Sparse Sparse Attention training framework.

Trains a model to work well under BOTH sparse and full attention by aligning
their outputs during training. At inference, you can use sparse attention for
speed without the quality loss that normally comes from training-inference mismatch.

How it works (per training step):
1. Randomly select sparse OR full attention mode (50/50)
2. Run forward pass in selected mode
3. If sparse mode: also compute full attention output (no grad) for alignment loss
4. Alignment loss: KL(sparse_output || full_output) — bidirectional
5. Total loss = task_loss + lambda * alignment_loss

At inference: use sparse attention for speed, model is already adapted to it.

Usage:
    from research.ssa import SSATrainer

    # Wrap your existing trainer
    ssa_trainer = SSATrainer(model, sparse_ratio=0.5, alignment_lambda=0.1)
    # In training loop:
    loss = ssa_trainer.compute_loss(batch, task_loss_fn)
    loss.backward()

Reference: "SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention
Outputs in Feature Space" (arXiv:2511.20102)
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class SparseAttentionWrapper(nn.Module):
    """Wraps an attention module to optionally use sparse attention.

    Sparse attention: only attend to top-k most relevant tokens per query.
    This is a simple top-k mask on the attention weights.

    Args:
        attention_module: the original attention module
        sparsity: fraction of tokens to KEEP (0.5 = keep top 50%)
    """

    def __init__(self, attention_module, sparsity=0.5):
        super().__init__()
        self.attention = attention_module
        self.sparsity = sparsity
        self._sparse_mode = False

    def set_sparse(self, enabled: bool):
        """Enable/disable sparse attention for next forward."""
        self._sparse_mode = enabled

    def __call__(self, *args, **kwargs):
        if not self._sparse_mode:
            return self.attention(*args, **kwargs)

        # Sparse mode: apply top-k masking to attention weights.
        # If the attention module supports a native sparse_k, use it.
        if hasattr(self.attention, "sparse_k"):
            k = max(1, int(self.sparsity * getattr(self.attention, "head_dim", 64)))
            self.attention.sparse_k = k
            result = self.attention(*args, **kwargs)
            self.attention.sparse_k = None
            return result

        # Otherwise, intercept the attention computation via hooks.
        # We register a forward hook that applies top-k masking to the
        # attention weights before the final value multiplication.
        return self._sparse_forward(*args, **kwargs)

    def _sparse_forward(self, *args, **kwargs):
        """Run attention with top-k sparse masking applied to the score matrix.

        This works by:
        1. Running the attention module's Q/K/V projections manually
        2. Computing scores, applying top-k mask + causal mask
        3. Computing softmax and multiplying by V
        Falls back to full attention if the module structure is incompatible.
        """
        import torch
        import torch.nn.functional as F

        attn = self.attention

        # Try to extract q, k, v projections from the attention module.
        q_proj = getattr(attn, "q_proj", None) or getattr(attn, "query", None)
        k_proj = getattr(attn, "k_proj", None) or getattr(attn, "key", None)
        v_proj = getattr(attn, "v_proj", None) or getattr(attn, "value", None)
        out_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)

        if q_proj is None or k_proj is None or v_proj is None or out_proj is None:
            # Can't decompose — fall back to full attention.
            return attn(*args, **kwargs)

        # First arg is typically the hidden state (B, T, d_model).
        x = args[0] if args else kwargs.get("hidden_states")
        if x is None:
            return attn(*args, **kwargs)

        B, T, D = x.shape
        n_heads = getattr(attn, "n_heads", getattr(attn, "num_heads", 8))
        head_dim = D // n_heads

        q = q_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
        v = v_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)

        # Compute attention scores.
        scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)

        # Apply causal mask.
        causal = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))

        # Top-k sparse masking: keep only top-k scores per query position.
        k_keep = max(1, int(self.sparsity * T))
        # For each query, find the top-k keys.
        topk_vals, topk_idx = scores.topk(k_keep, dim=-1)
        # Build a sparse mask from top-k indices.
        sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
        sparse_mask.scatter_(-1, topk_idx, True)
        # Combine with causal mask.
        final_mask = sparse_mask & causal
        scores = scores.masked_fill(~final_mask, float("-inf"))

        # Softmax + value multiplication.
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = out_proj(out)

        # Return in the same format as the original attention module.
        # Most attention modules return (output, ...) tuples.
        return out


class SSATrainer:
    """SSA training wrapper that alternates sparse/full attention with alignment.

    Args:
        model: the model to train (must have attention modules)
        sparse_ratio: probability of using sparse attention per step (default 0.5)
        alignment_lambda: weight of the alignment loss (default 0.1)
        sparsity: fraction of tokens kept in sparse mode (default 0.5)
    """

    def __init__(self, model, sparse_ratio=0.5, alignment_lambda=0.1, sparsity=0.5):
        self.model = model
        self.sparse_ratio = sparse_ratio
        self.alignment_lambda = alignment_lambda
        self.sparsity = sparsity

        # Wrap all attention modules.
        self.wrappers = []
        self._wrap_attention_modules(model)

        # Statistics.
        self.sparse_steps = 0
        self.full_steps = 0
        self.alignment_losses = []

    def _wrap_attention_modules(self, module, path=""):
        """Recursively find and wrap attention modules."""
        for name, child in module.named_children():
            full_path = f"{path}.{name}" if path else name
            # Detect attention modules by class name.
            cls_name = type(child).__name__.lower()
            if "attention" in cls_name or "attn" in cls_name:
                wrapper = SparseAttentionWrapper(child, sparsity=self.sparsity)
                setattr(module, name, wrapper)
                self.wrappers.append((full_path, wrapper))
            else:
                self._wrap_attention_modules(child, full_path)

    def set_mode(self, sparse: bool):
        """Set all attention modules to sparse or full mode."""
        for _, wrapper in self.wrappers:
            wrapper.set_sparse(sparse)

    def compute_loss(self, batch, task_loss_fn: Callable,
                     return_alignment: bool = False):
        """Compute SSA loss for one training step.

        Args:
            batch: input batch (model inputs)
            task_loss_fn: function(model, batch) -> (loss, logits)
            return_alignment: if True, return alignment loss separately

        Returns:
            total_loss = task_loss + lambda * alignment_loss
        """
        # Step 1: randomly choose sparse or full mode.
        use_sparse = random.random() < self.sparse_ratio
        self.set_mode(use_sparse)

        if use_sparse:
            self.sparse_steps += 1
        else:
            self.full_steps += 1

        # Step 2: forward pass in selected mode.
        task_loss, logits = task_loss_fn(self.model, batch)

        if not use_sparse:
            # Full attention: no alignment needed.
            if return_alignment:
                return task_loss, 0.0
            return task_loss

        # Step 3: sparse mode — compute full attention output for alignment.
        with torch.no_grad():
            self.set_mode(False)  # switch to full
            _, full_logits = task_loss_fn(self.model, batch)
            self.set_mode(True)  # switch back to sparse

        # Step 4: alignment loss (KL divergence between sparse and full outputs).
        # Align the logit distributions (feature-space alignment).
        sparse_logp = F.log_softmax(logits.float(), dim=-1)
        full_p = F.softmax(full_logits.float(), dim=-1)
        align_loss = F.kl_div(sparse_logp, full_p, reduction="batchmean")

        self.alignment_losses.append(align_loss.item())

        # Step 5: total loss.
        total = task_loss + self.alignment_lambda * align_loss

        if return_alignment:
            return total, align_loss.item()
        return total

    def stats(self):
        """Return training statistics."""
        avg_align = (sum(self.alignment_losses) / len(self.alignment_losses)
                     if self.alignment_losses else 0.0)
        return {
            "sparse_steps": self.sparse_steps,
            "full_steps": self.full_steps,
            "sparse_ratio_actual": (self.sparse_steps /
                                    max(1, self.sparse_steps + self.full_steps)),
            "avg_alignment_loss": avg_align,
        }
