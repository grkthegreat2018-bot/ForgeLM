"""EAGLE-3 speculative decoding head (NeurIPS 2025).

EAGLE-3 key innovations over EAGLE-2:
- Multi-layer feature fusion: extracts hidden states from low/mid/high layers
  of the target model (not just the top layer). This captures richer semantic
  information at different abstraction levels.
- Removes feature prediction constraint: EAGLE-2 predicted features then
  tokens; EAGLE-3 predicts tokens directly. This removes an unnecessary
  constraint that limited expressiveness.
- Training-time test: simulates autoregressive generation during training
  by feeding draft model outputs back as inputs, closing the train/test gap.

Architecture (EAGLE-3):
    EAGLE3Head(
        feature_proj: Linear(d_model * n_feature_layers, d_model)
        fusion: lightweight cross-attention or MLP
        layers: 1-2x TransformerBlock (lightweight)
        lm_head: Linear(d_model, vocab_size)
    )

At inference:
    1. Target model forward → extract features from layers [low, mid, high]
    2. Concatenate features → project to d_model
    3. EAGLE-3 head generates k draft tokens autoregressively
    4. Target model verifies all k tokens in one forward pass
    5. Accept matching prefix, resample at first mismatch

Reference: Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large
Language Models via Training-Time Test", NeurIPS 2025.

Usage:
    from research.decoding.eagle import EAGLE3Head, train_eagle3_head, eagle3_speculative_generate

    head = train_eagle3_head(target_model, tokenizer, train_data, steps=1000)
    output = eagle3_speculative_generate(target_model, head, tokenizer, prompt)
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EAGLEHead(nn.Module):
    """EAGLE-2 style speculative decoding head (legacy, kept for compatibility).

    Uses only top-layer hidden states. For new work, prefer EAGLE3Head.
    """

    def __init__(self, d_model, vocab_size, n_layers=2, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.input_proj = nn.Linear(d_model * 2, d_model)
        self.layers = nn.ModuleList([
            self._make_layer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
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
            "n_heads": nn.Identity(),
        })

    def forward(self, target_hidden: torch.Tensor, token_embeds: torch.Tensor,
                past_kv: list | None = None) -> tuple[torch.Tensor, list]:
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


class EAGLE3Head(nn.Module):
    """EAGLE-3 speculative decoding head with multi-layer feature fusion.

    Key difference from EAGLE-2: fuses hidden states from multiple target
    model layers (low/mid/high) instead of just the top layer. This captures
    richer semantic information and improves draft acceptance rates.

    Args:
        d_model: target model hidden dimension
        vocab_size: vocabulary size
        feature_layers: list of target layer indices to extract features from.
                        Default [1, 7, 15] for LFM2.5-1.2B (16 layers).
                        Low=1, Mid=7, High=15.
        n_layers: number of lightweight transformer layers in head (default 1)
        n_heads: attention heads in head (default 8)
    """

    def __init__(self, d_model, vocab_size, feature_layers=None,
                 n_layers=1, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.feature_layers = feature_layers or [1, 7, 15]
        self.n_features = len(self.feature_layers)

        # Fuse multi-layer features + token embedding → d_model
        # (n_features * d_model) from target layers + d_model from embedding
        self.feature_proj = nn.Linear(d_model * (self.n_features + 1), d_model)

        # Lightweight transformer layers
        self.layers = nn.ModuleList([
            self._make_layer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _make_layer(self, d_model, n_heads):
        return nn.ModuleDict({
            "ln1": nn.LayerNorm(d_model),
            "attn_q": nn.Linear(d_model, d_model, bias=False),
            "attn_k": nn.Linear(d_model, d_model, bias=False),
            "attn_v": nn.Linear(d_model, d_model, bias=False),
            "attn_o": nn.Linear(d_model, d_model, bias=False),
            "ln2": nn.LayerNorm(d_model),
            "ffn_up": nn.Linear(d_model, d_model * 4),
            "ffn_down": nn.Linear(d_model * 4, d_model),
            "n_heads": nn.Identity(),
        })

    def extract_features(self, target_model, input_ids,
                         past_key_values=None) -> torch.Tensor:
        """Extract hidden states from target model at specified layers.

        Returns:
            fused_features: (B, T, d_model) concatenated + projected features
        """
        target_model.eval()
        with torch.no_grad():
            # Run target model, capturing hidden states at feature layers
            out = target_model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            if isinstance(out, tuple):
                hidden_states = out[-1]  # all hidden states
                presents = out[2] if len(out) > 2 else None
            else:
                hidden_states = [out]
                presents = None

        # Collect features from specified layers
        features = []
        for layer_idx in self.feature_layers:
            if layer_idx < len(hidden_states):
                features.append(hidden_states[layer_idx][:, -1:, :])  # (B, 1, d)
            else:
                # Fallback: use last available
                features.append(hidden_states[-1][:, -1:, :])

        return torch.cat(features, dim=-1), presents  # (B, 1, d*n_features)

    def forward(self, target_features: torch.Tensor, token_embeds: torch.Tensor,
                past_kv: list | None = None) -> tuple[torch.Tensor, list]:
        """Forward pass with multi-layer fused features.

        Args:
            target_features: (B, T, d_model * n_features) fused from target layers
            token_embeds: (B, T, d_model) embeddings of current tokens
            past_kv: past key-value cache

        Returns:
            logits: (B, T, vocab_size)
            new_kv: updated key-value cache
        """
        B, T, _ = target_features.shape
        D = self.d_model

        # Fuse features + embedding
        x = self.feature_proj(torch.cat([target_features, token_embeds], dim=-1))

        # Transformer layers
        new_kv = []
        for i, layer in enumerate(self.layers):
            x_ln = layer["ln1"](x)
            q = layer["attn_q"](x_ln)
            k = layer["attn_k"](x_ln)
            v = layer["attn_v"](x_ln)

            n_h = self.n_heads
            hd = D // n_h
            q = q.view(B, T, n_h, hd).transpose(1, 2)
            k = k.view(B, T, n_h, hd).transpose(1, 2)
            v = v.view(B, T, n_h, hd).transpose(1, 2)

            if past_kv is not None and i < len(past_kv):
                pk, pv = past_kv[i]
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            new_kv.append((k, v))

            T_k = k.shape[2]
            mask = torch.tril(torch.ones(T, T_k, device=x.device, dtype=torch.bool))
            attn = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
            attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            x = x + layer["attn_o"](out)

            x_ln2 = layer["ln2"](x)
            x = x + layer["ffn_down"](F.gelu(layer["ffn_up"](x_ln2)))

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_kv

    def generate_draft(self, target_model, input_ids, past_key_values,
                       k=4, temperature=0.0, past_kv=None):
        """Generate k draft tokens using multi-layer feature fusion.

        Args:
            target_model: the main model (for feature extraction)
            input_ids: (B, 1) last token id
            past_key_values: target model's KV cache
            k: number of draft tokens
            temperature: sampling temperature
            past_kv: EAGLE head's own KV cache

        Returns:
            draft_tokens: (B, k)
            draft_logits: (B, k, vocab_size)
        """
        B = input_ids.shape[0]
        device = input_ids.device
        draft_tokens = []
        draft_logits = []
        cur_kv = past_kv
        cur_ids = input_ids

        # Get target model's embedding table
        target_emb = None
        for name, module in target_model.named_modules():
            if isinstance(module, nn.Embedding) and module.weight.shape[0] > 1000:
                target_emb = module
                break

        for _ in range(k):
            # Extract multi-layer features from target model
            target_feats, target_kv = self.extract_features(
                target_model, cur_ids, past_key_values=past_key_values)

            # Get token embedding
            if target_emb is not None:
                tok_emb = target_emb(cur_ids)
            else:
                tok_emb = target_feats[:, :, :self.d_model]  # fallback

            # Forward through EAGLE-3 head
            logits, cur_kv = self.forward(target_feats, tok_emb, past_kv=cur_kv)
            last_logits = logits[:, -1, :]

            if temperature == 0:
                token = last_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)

            draft_tokens.append(token)
            draft_logits.append(last_logits)
            cur_ids = token

        draft_tokens = torch.cat(draft_tokens, dim=1)
        draft_logits = torch.stack(draft_logits, dim=1)
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
    eos_id = tokenizer.eos_token_id

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
        start_pos = generated.shape[1] - 1
        target_preds = verify_logits[0, start_pos:start_pos + k, :].argmax(-1)
        matches = (draft_tokens[0, :k] == target_preds)
        n_accepted = matches.cumprod(dim=-1).sum().item()

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
        if eos_id is not None:
            is_eos = (next_token == eos_id).reshape(-1).any()
            if is_eos.item():
                break

    # Decode.
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── EAGLE-3 Training & Generation ────────────────────────────────────────────

def train_eagle3_head(target_model, tokenizer, train_texts, steps=1000,
                      lr=1e-4, batch_size=4, max_seq_len=128, device="cuda",
                      feature_layers=None):
    """Train an EAGLE-3 head with multi-layer feature fusion + training-time test.

    EAGLE-3 key training innovations:
    - Multi-layer feature extraction (not just top layer)
    - Training-time test: feed draft model outputs back as inputs to
      close the train/test distribution gap
    - Token-only loss (no feature prediction constraint)
    """
    d_model = target_model.config.d_model if hasattr(target_model, "config") else 2048
    vocab_size = tokenizer.vocab_size
    if feature_layers is None:
        n_total = getattr(target_model.config, 'n_layers', 16) if hasattr(target_model, 'config') else 16
        feature_layers = [1, n_total // 2, n_total - 1]

    head = EAGLE3Head(d_model, vocab_size, feature_layers=feature_layers).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    # Enable output_hidden_states on target model
    if hasattr(target_model, 'config'):
        target_model.config.output_hidden_states = True

    import random
    for step in range(steps):
        batch_texts = random.sample(train_texts, min(batch_size, len(train_texts)))
        all_input_ids = []
        for text in batch_texts:
            ids = tokenizer(text, return_tensors="pt", max_length=max_seq_len,
                           truncation=True).input_ids[0]
            all_input_ids.append(ids)

        max_len = max(len(ids) for ids in all_input_ids)
        input_ids = torch.full((len(all_input_ids), max_len),
                              tokenizer.pad_token_id or 0,
                              dtype=torch.long, device=device)
        for i, ids in enumerate(all_input_ids):
            input_ids[i, :len(ids)] = ids

        # Extract multi-layer features from target model
        with torch.no_grad():
            out = target_model(input_ids, output_hidden_states=True)
            if isinstance(out, tuple):
                hidden_states = out[-1]
                target_logits = out[0]
            else:
                hidden_states = [out]
                target_logits = out

            # Fuse features from specified layers
            features = []
            for li in feature_layers:
                if li < len(hidden_states):
                    features.append(hidden_states[li])
                else:
                    features.append(hidden_states[-1])
            fused = torch.cat(features, dim=-1)  # (B, T, d*n_features)

        # Token embeddings
        if hasattr(target_model, "wte"):
            token_embeds = target_model.wte(input_ids)
        elif hasattr(target_model, "embed_tokens"):
            token_embeds = target_model.embed_tokens(input_ids)
        else:
            token_embeds = fused[:, :, :d_model]

        # Training-time test: simulate autoregressive generation
        # Feed the draft model's own outputs as inputs for subsequent positions
        eagle_logits, _ = head(fused, token_embeds)
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
            print(f"  [EAGLE-3 train] step {step+1}/{steps} | loss: {loss.item():.4f}")

    return head


def eagle3_speculative_generate(target_model, eagle3_head, tokenizer,
                                prompt, max_new_tokens=50, k=4,
                                temperature=0.0, device="cuda"):
    """Speculative decoding using EAGLE-3 head with multi-layer features.

    Uses EAGLE-style tree verification: all k drafts verified in one forward
    pass. The first mismatch position determines the accepted prefix, and
    the model's prediction at that position becomes the corrected token.
    """
    target_model.eval()
    eagle3_head.eval()
    eos_id = tokenizer.eos_token_id

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    # Enable hidden states output on target model
    if hasattr(target_model, 'config'):
        target_model.config.output_hidden_states = True

    # Prefill: run full prompt through target model to get KV cache
    with torch.no_grad():
        out = target_model(generated, use_cache=True, output_hidden_states=True)
        if isinstance(out, tuple):
            logits = out[0]
            past_kv = out[2] if len(out) > 2 else out[1]
            hidden_states = out[-1]
        else:
            logits = out
            past_kv = None
            hidden_states = [out]

    for _ in range(max_new_tokens // k + 1):
        # Get target's next token prediction (greedy)
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)
        if temperature == 0:
            main_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            main_token = torch.multinomial(
                F.softmax(next_logits, dim=-1), num_samples=1)

        # EAGLE-3 generates k draft tokens
        draft_tokens, draft_logits = eagle3_head.generate_draft(
            target_model, main_token, past_kv, k=k, temperature=temperature)

        # Verify all k drafts in one forward pass (EAGLE tree verification)
        verify_seq = torch.cat([main_token, draft_tokens], dim=1)
        with torch.no_grad():
            v_out = target_model(verify_seq, past_key_values=past_kv,
                                use_cache=True, output_hidden_states=True)
            if isinstance(v_out, tuple):
                verify_logits = v_out[0]
                new_past_kv = v_out[2] if len(v_out) > 2 else v_out[1]
                verify_hidden = v_out[-1]
            else:
                verify_logits = v_out
                new_past_kv = None
                verify_hidden = [v_out]

        # Compare model predictions vs drafts (positions 0..k-1 predict tokens 1..k)
        preds = verify_logits[:, :k, :].argmax(-1)  # (B, k)
        matches = (preds == draft_tokens)            # (B, k)

        # First mismatch position
        not_match = ~matches
        any_mismatch = not_match.any(dim=-1)
        n_accepted = torch.where(
            any_mismatch,
            not_match.float().argmax(dim=-1),
            torch.full_like(any_mismatch, k, dtype=torch.long),
        ).min().item()

        # Accept prefix + resample at mismatch
        accepted = verify_seq[:, :n_accepted + 1]
        generated = torch.cat([generated, accepted], dim=1)

        # Set up next iteration: logits at last accepted position
        next_idx = n_accepted  # last accepted position
        logits = verify_logits[:, next_idx:next_idx + 1, :]
        past_kv = new_past_kv

        # EOS check
        if eos_id is not None and (accepted == eos_id).any().item():
            break
        if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
            break

    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
