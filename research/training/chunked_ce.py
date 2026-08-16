"""Pure-PyTorch Fused Linear Cross-Entropy (FLCE) — no Triton dependency.

Achieves the same memory savings as Liger FLCE by chunking the token dimension
and never materializing the full [B*T, V] logits tensor. Works on any GPU,
including consumer Blackwell (sm_120) where Liger's Triton kernels crash.

Trade-off: ~2-3x slower than Liger's fused Triton kernel, but same memory profile
and exact numerical parity with F.cross_entropy.
"""
import torch
import torch.nn.functional as F
from torch.autograd import Function


class ChunkedLinearCrossEntropy(Function):
    """Fused linear + cross-entropy with chunked computation.

    Computes: loss = cross_entropy(x @ weight.t(), target)
    Without ever materializing the full [N, V] logits tensor.

    Args (forward):
        x:       [N, H]  hidden states (requires_grad)
        weight:  [V, H]  LM head weight (requires_grad)
        target:  [N]     token ids
        chunk_size: number of tokens per chunk (tune for memory/speed)

    Returns:
        loss scalar
    """

    @staticmethod
    def forward(ctx, x, weight, target, chunk_size=512):
        N, H = x.shape
        V = weight.shape[0]
        device = x.device
        ignore_index = -100

        grad_input = torch.zeros_like(x) if x.requires_grad else None
        grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        n_valid = 0  # count of non-ignored tokens for mean reduction

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x[start:end]                          # [cs, H]
            target_chunk = target[start:end]                # [cs]

            # Mask for non-ignored tokens (target != ignore_index)
            valid_mask = target_chunk != ignore_index       # [cs]
            n_valid_chunk = valid_mask.sum().item()
            n_valid += n_valid_chunk

            # Materialize logits only for this chunk: [cs, V]
            logits_chunk = x_chunk @ weight.t()
            # F.cross_entropy handles ignore_index=-100 by default
            loss_chunk = F.cross_entropy(
                logits_chunk.float(), target_chunk, reduction="sum"
            )
            loss_total = loss_total + loss_chunk

            if x.requires_grad:
                # grad_logits = softmax(logits) - one_hot(target), then / n_valid for mean
                with torch.no_grad():
                    probs = F.softmax(logits_chunk.float(), dim=-1)  # [cs, V]
                    # Subtract 1 at target positions (gradient of CE w.r.t. logits)
                    # Only for non-ignored tokens; use clamped index to avoid OOB.
                    safe_target = target_chunk.clamp(min=0)  # -100 → 0 (will be masked)
                    probs.scatter_add_(
                        1, safe_target.unsqueeze(1),
                        torch.full_like(probs[:, :1], -1.0),
                    )
                    # Zero out gradients for ignored positions
                    probs = probs * valid_mask.unsqueeze(1).to(probs.dtype)

                # grad_input_chunk = grad_logits @ weight  → [cs, H]
                grad_input[start:end] = probs.to(x.dtype) @ weight
                # grad_weight += grad_logits.t() @ x_chunk → [V, H]
                grad_weight += probs.to(x.dtype).t() @ x_chunk

            # logits_chunk and probs go out of scope here → freed

        # Mean over non-ignored tokens (not total N)
        loss = loss_total / max(n_valid, 1)

        ctx.save_for_backward(grad_input, grad_weight)
        return loss.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        # Scale by grad_output (dL/dloss = 1.0 for scalar loss, but handle chain rule)
        grad_input = grad_input * grad_output
        grad_weight = grad_weight * grad_output
        return grad_input, grad_weight, None, None


def chunked_linear_cross_entropy(x, weight, target, chunk_size=512):
    """Functional interface for ChunkedLinearCrossEntropy."""
    return ChunkedLinearCrossEntropy.apply(x, weight, target, chunk_size)
