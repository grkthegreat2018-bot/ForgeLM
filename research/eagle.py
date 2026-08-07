"""EAGLE-3 style speculative decoding head.

Upgrades our basic draft model approach to EAGLE-3:
- Lightweight autoregressive head attached to target model's hidden states
- No separate draft model needed (saves VRAM)
- Higher acceptance rates than Medusa (sequential dependence)
- Trained on (hidden_state, next_token) pairs from the target model

Architecture:
    EAGLEHead(
        input_proj: Linear(d_model * 2, d_model)  # concat hidden + embedding
        layers: 2x TransformerBlock (lightweight)
        lm_head: Linear(d_model, vocab_size)  # shared with target or separate
    )

Training:
    1. Run target model on training data, cache hidden states
    2. Train EAGLE head to predict next token from (hidden_t, embed_t)
    3. At inference: head generates k draft tokens autoregressively,
       target model verifies in one forward pass

Usage:
    from research.eagle import EAGLEHead, train_eagle_head, eagle_speculative_generate

    # Train
    head = train_eagle_head(target_model, tokenizer, train_data, steps=1000)

    # Inference (replaces speculative_generate)
    output = eagle_speculative_generate(target_model, head, tokenizer, prompt)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EAGLEHead(nn.Module):
    """EAGLE-3 speculative decoding head.

    Attaches to target model's hidden states and autoregressively predicts
    draft tokens. Lighter than a separate draft model.

    Args:
        d_model: target model hidden dimension
        vocab_size: vocabulary size
        n_layers: number of lightweight transformer layers in head (default 2)
        n_heads: attention heads in head (default 8)
    """

    def __init__(self, d_model, vocab_size, n_layers=2, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads

        # Input: concat(target_hidden, token_embedding) -> d_model
        self.input_proj = nn.Linear(d_model * 2, d_model)

        # Lightweight transformer layers (can reuse model_loader's ModularBlock
        # but for simplicity, use basic attention + FFN).
        self.layers = nn.ModuleList([
            self._make_layer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection to vocab (can be shared with target lm_head).
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _make_layer(self, d_model, n_heads):
        """Create a lightweight transformer layer."""
        head_dim = d_model // n_heads
        return nn.ModuleDict({
            "ln1": nn.LayerNorm(d_model),
            "attn_q": nn.Linear(d_model, d_model, bias=False),
            "attn_k": nn.Linear(d_model, d_model, bias=False),
            "attn_v": nn.Linear(d_model, d_model, bias=False),
            "attn_o": nn.Linear(d_model, d_model, bias=False),
            "ln2": nn.LayerNorm(d_model),
            "ffn_up": nn.Linear(d_model, d_model * 4),
            "ffn_down": nn.Linear(d_model * 4, d_model),
            "n_heads": nn.ConstantPad1d((0, 0), n_heads) if False else nn.Identity(),
        })

    def forward(self, target_hidden: torch.Tensor, token_embeds: torch.Tensor,
                past_kv: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        """Forward pass of EAGLE head.

        Args:
            target_hidden: (B, T, d_model) hidden states from target model
            token_embeds: (B, T, d_model) embeddings of current tokens
            past_kv: past key-value cache for autoregressive generation

        Returns:
            logits: (B, T, vocab_size)
            new_kv: updated key-value cache
        """
        B, T, D = target_hidden.shape

        # Concat hidden + embedding, project.
        x = self.input_proj(torch.cat([target_hidden, token_embeds], dim=-1))

        # Transformer layers with causal attention.
        new_kv = []
        for i, layer in enumerate(self.layers):
            x_ln = layer["ln1"](x)
            q = layer["attn_q"](x_ln)
            k = layer["attn_k"](x_ln)
            v = layer["attn_v"](x_ln)

            # Reshape for multi-head attention.
            n_h = self.n_heads
            hd = D // n_h
            q = q.view(B, T, n_h, hd).transpose(1, 2)
            k = k.view(B, T, n_h, hd).transpose(1, 2)
            v = v.view(B, T, n_h, hd).transpose(1, 2)

            # Append to past KV if available.
            if past_kv is not None and i < len(past_kv):
                pk, pv = past_kv[i]
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            new_kv.append((k, v))

            # Causal attention.
            T_k = k.shape[2]
            mask = torch.tril(torch.ones(T, T_k, device=x.device, dtype=torch.bool))
            attn = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
            attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            out = attn @ v  # (B, n_h, T, hd)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            x = x + layer["attn_o"](out)

            # FFN.
            x_ln2 = layer["ln2"](x)
            x = x + layer["ffn_down"](F.gelu(layer["ffn_up"](x_ln2)))

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_kv

    def generate_draft(self, target_hidden, token_embeds, k=4,
                       temperature=0.0, past_kv=None, target_embedding=None):
        """Generate k draft tokens autoregressively.

        Args:
            target_hidden: (B, 1, d_model) hidden state from target model
            token_embeds: (B, 1, d_model) embedding of last token
            k: number of draft tokens to generate
            temperature: 0 for greedy, >0 for sampling
            target_embedding: optional nn.Embedding from target model for
                              proper token→embedding lookup (instead of soft approx)

        Returns:
            draft_tokens: (B, k) generated token ids
            draft_logits: (B, k, vocab_size) logits for each draft token
        """
        B = target_hidden.shape[0]
        device = target_hidden.device

        draft_tokens = []
        draft_logits = []
        cur_hidden = target_hidden
        cur_embed = token_embeds
        cur_kv = past_kv

        for _ in range(k):
            logits, cur_kv = self.forward(cur_hidden, cur_embed, past_kv=cur_kv)
            last_logits = logits[:, -1, :]  # (B, vocab)

            if temperature == 0:
                token = last_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)

            draft_tokens.append(token)
            draft_logits.append(last_logits)

            # Next step: look up the embedding of the generated token.
            # Use target model's embedding table if provided (proper EAGLE),
            # otherwise fall back to soft embedding via lm_head weight projection.
            if target_embedding is not None:
                cur_embed = target_embedding(token)  # (B, 1, d_model)
            else:
                cur_embed = F.softmax(last_logits, dim=-1) @ self.lm_head.weight
                cur_embed = cur_embed.unsqueeze(1)  # (B, 1, d_model)
            cur_hidden = cur_hidden  # keep the same (EAGLE uses static context)

        draft_tokens = torch.cat(draft_tokens, dim=1)  # (B, k)
        draft_logits = torch.stack(draft_logits, dim=1)  # (B, k, vocab)
        return draft_tokens, draft_logits


def train_eagle_head(target_model, tokenizer, train_texts, steps=1000,
                     lr=1e-4, batch_size=4, max_seq_len=128, device="cuda"):
    """Train an EAGLE head on target model's hidden states.

    Args:
        target_model: the main model (frozen during training)
        tokenizer: tokenizer
        train_texts: list of training strings
        steps: training steps
        lr: learning rate
        batch_size: samples per batch
        max_seq_len: max sequence length
        device: cuda or cpu

    Returns:
        trained EAGLEHead
    """
    d_model = target_model.config.d_model if hasattr(target_model, "config") else 1024
    vocab_size = tokenizer.vocab_size

    head = EAGLEHead(d_model, vocab_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    import random
    for step in range(steps):
        # Sample a batch of texts.
        batch_texts = random.sample(train_texts, min(batch_size, len(train_texts)))

        # Tokenize.
        all_input_ids = []
        for text in batch_texts:
            ids = tokenizer(text, return_tensors="pt", max_length=max_seq_len,
                           truncation=True).input_ids[0]
            all_input_ids.append(ids)

        # Pad to same length.
        max_len = max(len(ids) for ids in all_input_ids)
        input_ids = torch.full((len(all_input_ids), max_len), tokenizer.pad_token_id or 0,
                              dtype=torch.long, device=device)
        for i, ids in enumerate(all_input_ids):
            input_ids[i, :len(ids)] = ids

        # Get target model hidden states.
        with torch.no_grad():
            target_out = target_model(input_ids)
            target_hidden = target_out[0] if isinstance(target_out, tuple) else target_out
            # Get target's logits for distillation.
            target_logits = target_model.lm_head(target_hidden) if hasattr(target_model, "lm_head") else target_out

        # Get token embeddings (for EAGLE input).
        if hasattr(target_model, "wte"):
            token_embeds = target_model.wte(input_ids)
        elif hasattr(target_model, "embed_tokens"):
            token_embeds = target_model.embed_tokens(input_ids)
        else:
            token_embeds = target_hidden  # fallback

        # Train EAGLE head to predict next token.
        eagle_logits, _ = head(target_hidden, token_embeds)

        # Loss: next-token prediction (distill from target).
        # Target: shift input_ids by 1.
        targets = input_ids[:, 1:].contiguous()
        eagle_logits_shifted = eagle_logits[:, :-1, :].contiguous()

        loss = F.cross_entropy(
            eagle_logits_shifted.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=tokenizer.pad_token_id or 0,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"  [EAGLE train] step {step+1}/{steps} | loss: {loss.item():.4f}")

    return head


def eagle_speculative_generate(target_model, eagle_head, tokenizer,
                               prompt, max_new_tokens=50, k=4,
                               temperature=0.0, device="cuda"):
    """Speculative decoding using EAGLE head.

    Args:
        target_model: main model
        eagle_head: trained EAGLE head
        tokenizer: tokenizer
        prompt: input prompt string
        max_new_tokens: max tokens to generate
        k: draft tokens per speculation round
        temperature: sampling temperature

    Returns:
        generated text
    """
    target_model.eval()
    eagle_head.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    for _ in range(max_new_tokens // k + 1):
        # 1. Target model forward to get hidden states.
        with torch.no_grad():
            target_out = target_model(generated)
            target_hidden = target_out[0] if isinstance(target_out, tuple) else target_out
            target_logits = target_out if not isinstance(target_out, tuple) else target_out[0]
            if hasattr(target_model, "lm_head"):
                target_logits = target_model.lm_head(target_hidden)

        # 2. EAGLE head generates k draft tokens.
        last_hidden = target_hidden[:, -1:, :]
        target_emb = None
        if hasattr(target_model, "wte"):
            last_embed = target_model.wte(generated[:, -1:])
            target_emb = target_model.wte
        elif hasattr(target_model, "embed_tokens"):
            last_embed = target_model.embed_tokens(generated[:, -1:])
            target_emb = target_model.embed_tokens
        else:
            last_embed = last_hidden

        draft_tokens, draft_logits = eagle_head.generate_draft(
            last_hidden, last_embed, k=k, temperature=temperature,
            target_embedding=target_emb,
        )

        # 3. Target model verifies draft tokens in one forward pass.
        draft_seq = torch.cat([generated, draft_tokens], dim=1)
        with torch.no_grad():
            verify_out = target_model(draft_seq)
            verify_hidden = verify_out[0] if isinstance(verify_out, tuple) else verify_out
            if hasattr(target_model, "lm_head"):
                verify_logits = target_model.lm_head(verify_hidden)
            else:
                verify_logits = verify_out

        # 4. Accept tokens that match (greedy verification).
        n_accepted = 0
        for i in range(k):
            draft_token = draft_tokens[0, i].item()
            # Target's prediction at this position.
            target_pred = verify_logits[0, generated.shape[1] + i - 1, :].argmax().item()
            if draft_token == target_pred:
                n_accepted += 1
            else:
                break

        # 5. Append accepted tokens + 1 target token.
        if n_accepted > 0:
            accepted = draft_tokens[:, :n_accepted]
            generated = torch.cat([generated, accepted], dim=1)
        # Always add at least the target's next token.
        next_token = verify_logits[0, generated.shape[1] - 1, :].argmax().unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

        if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
            break

        # Check EOS.
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode.
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
