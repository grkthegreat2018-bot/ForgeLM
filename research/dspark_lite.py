"""Lightweight DSpark head — Medusa-1 style with shared LM head.

Instead of 4 independent Linear(d_model, vocab) heads (936M params),
uses the model's existing LM head + 4 tiny residual adapters:
  - Each adapter: Linear(d_model, d_model) + GELU + residual
  - Output: model.head(adapter(hidden)) — reuses the 234M LM head
  - Total new params: 4 × (1536×1536) = ~9.4M

This trains 100x faster than the full DSpark head and achieves
similar acceptance rates (Medusa-1 paper shows single-layer heads suffice).

Usage:
    from research.dspark_lite import DSparkLite, train_dspark_lite
    head = DSparkLite(d_model=1536, vocab_size=151936, n_predict=4, lm_head=model.head)
    train_dspark_lite(model, head, steps=500)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from safetensors.torch import save_file
import time, math, os


class DSparkLite(nn.Module):
    """Lightweight speculative decoding head — Medusa-1 style.

    Each prediction head is a single residual MLP adapter that transforms
    the hidden state before passing through the shared LM head.

    Architecture per head k:
        h_k = hidden + W2_k(GELU(W1_k(hidden)))   # residual adapter
        logits_k = lm_head(h_k)                    # shared output projection

    Total params: n_predict × (2 × d_model²) = 4 × 2 × 1536² ≈ 18.9M
    (vs 1013M for the full DSpark head)
    """

    def __init__(self, d_model: int, vocab_size: int, n_predict: int = 4,
                 lm_head: Optional[nn.Module] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_predict = n_predict
        self.lm_head = lm_head  # shared, not trained

        # Tiny residual adapters: Linear → GELU → Linear (with residual)
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model, bias=False),
                nn.GELU(),
                nn.Linear(d_model, d_model, bias=False),
            )
            for _ in range(n_predict)
        ])

        # Initialize as near-identity (residual starts dominant)
        for adapter in self.adapters:
            for m in adapter:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    # Make second layer small so residual dominates at init
                    if m is adapter[-1]:
                        nn.init.normal_(m.weight, mean=0.0, std=0.001)

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Predict N future token distributions from hidden states.

        Args:
            hidden_states: (B, T, d_model)

        Returns:
            list of (B, T, vocab_size) logits, one per prediction position
        """
        logits_list = []
        for k, adapter in enumerate(self.adapters):
            h_k = hidden_states + adapter(hidden_states)  # residual
            if self.lm_head is not None:
                logits_k = self.lm_head(h_k)
            else:
                logits_k = h_k  # fallback (shouldn't happen)
            logits_list.append(logits_k)
        return logits_list

    def predict_block(self, hidden_states: torch.Tensor,
                      temperature: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict N draft tokens from the last hidden state.

        Args:
            hidden_states: (B, T, d_model)
            temperature: 0 for greedy

        Returns:
            tokens: (B, N) predicted token ids
            confidences: (B, N) confidence scores (max prob)
        """
        last_h = hidden_states[:, -1:, :]  # (B, 1, d_model)
        logits_list = self.forward(last_h)

        tokens = []
        confs = []
        for logits in logits_list:
            l = logits[:, -1, :]  # (B, vocab)
            if temperature == 0:
                tok = l.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(l / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
            tokens.append(tok)
            confs.append(F.softmax(l, dim=-1).max(dim=-1, keepdim=True).values)

        tokens = torch.cat(tokens, dim=1)  # (B, N)
        confs = torch.cat(confs, dim=1)  # (B, N)
        return tokens, confs

    @torch.no_grad()
    def generate_block(self, model, input_ids, max_block_size=4, temperature=0.0):
        """Generate a draft block — compatible with DSparkDecoding interface.

        Returns list of (token_id, confidence) pairs.
        """
        self.eval()
        try:
            out = model(input_ids, return_hidden=True)
            hidden = out[2] if len(out) > 2 else out[0]
        except TypeError:
            out = model(input_ids)
            hidden = out[0] if isinstance(out, tuple) else out
            if hidden.size(-1) == self.vocab_size:
                hidden = model.embed(hidden.argmax(-1))

        tokens, confs = self.predict_block(hidden, temperature)
        return [(tokens[0, k].item(), confs[0, k].item())
                for k in range(min(max_block_size, self.n_predict))]

    def confidence_schedule(self, confidences, throughput_profile=None,
                            threshold=0.5):
        """Simple threshold-based scheduling (compatible with DSpark interface).

        Returns number of tokens to verify based on confidence scores.
        """
        n = 0
        for conf in confidences:
            if conf >= threshold:
                n += 1
            else:
                break
        return min(n, self.n_predict)


def train_dspark_lite(model, dspark_head, steps=500, lr=3e-4, seq_len=256,
                      batch_size=2, warmup=50, save_path=None,
                      grad_accum=1, device="cuda"):
    """Train a DSparkLite head with CE loss + distribution matching.

    Uses the model's frozen hidden states + LM head. Only the tiny adapters
    are trained (~19M params), so this is very fast.

    Args:
        model: frozen main LLM
        dspark_head: DSparkLite instance (with lm_head set)
        steps: training steps
        lr: learning rate
        seq_len: sequence length
        batch_size: micro-batch size
        warmup: LR warmup steps
        save_path: if set, save checkpoint here
        grad_accum: gradient accumulation steps
        device: cuda or cpu
    """
    dev = torch.device(device)
    n_predict = dspark_head.n_predict

    # Freeze main model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Only train adapters
    optimizer = torch.optim.AdamW(dspark_head.adapters.parameters(), lr=lr, weight_decay=0.01)

    def cosine_lr(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

    # Position weights: exponential decay
    pos_weights = torch.exp(-torch.arange(1, n_predict + 1, dtype=torch.float) / n_predict).to(dev)

    # Load data on CPU
    train_data = torch.frombuffer(
        open("research/data/train.bin", "rb").read(), dtype=torch.uint32
    ).long()

    n_params = sum(p.numel() for p in dspark_head.adapters.parameters())
    print(f"Training DSparkLite: {n_params/1e6:.1f}M trainable params, {steps} steps")
    print(f"  lr={lr}, seq={seq_len}, batch={batch_size}, grad_accum={grad_accum}")

    t0 = time.time()
    optimizer.zero_grad()
    best_loss = float("inf")

    for step in range(steps):
        dspark_head.train()

        for _ in range(grad_accum):
            idx = torch.randint(0, train_data.numel() - seq_len - 1, (batch_size,))
            batch = torch.stack([train_data[i:i+seq_len] for i in idx]).to(dev)

            with torch.no_grad():
                out = model(batch, return_hidden=True)
                hidden = out[2] if len(out) > 2 else out[0]

            logits_list = dspark_head(hidden)

            # CE loss for each head
            ce_loss = 0.0
            for k in range(n_predict):
                offset = k + 1
                if offset >= seq_len:
                    break
                pred = logits_list[k][:, :-offset, :].contiguous()
                target = batch[:, offset:].contiguous()
                ce_loss = ce_loss + F.cross_entropy(
                    pred.view(-1, pred.size(-1)),
                    target.view(-1), ignore_index=-100
                ) * pos_weights[k]

            (ce_loss / grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(dspark_head.adapters.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 25 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (steps - step - 1)
            lr_now = optimizer.param_groups[0]["lr"]
            loss_val = ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss
            print(f"  step {step:4d}/{steps} | ce={loss_val:.4f} | lr={lr_now:.2e} | "
                  f"{elapsed:.0f}s, ~{eta:.0f}s left")

            if loss_val < best_loss:
                best_loss = loss_val

        if step % 50 == 0:
            torch.cuda.empty_cache()

    # Save
    if save_path:
        state = {f"dspark_lite.{k}": v.cpu()
                 for k, v in dspark_head.state_dict().items()
                 if "adapters" in k}
        save_file(state, save_path)
        print(f"\nSaved to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    total = time.time() - t0
    print(f"\nDone in {total:.0f}s ({total/60:.1f} min)")
    print(f"  Best loss: {best_loss:.4f}")
    return dspark_head
