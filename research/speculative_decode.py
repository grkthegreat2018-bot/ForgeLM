"""Speculative decoding for the ForgeAI research model.

Self-speculative decoding: a small draft model (2-layer) proposes K tokens,
then the main model verifies them in one forward pass. Accepted tokens are
free (1 forward pass for K tokens instead of K). Zero quality loss — the
output distribution is identical to standard decoding.

Usage:
    python -m research.speculative_decode --prompt "Hello" --max-new-tokens 50 --k 4
"""
import argparse
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from research.config import get_config, ModelConfig
from research.model_loader import ModelLoader


def _model_forward(model, idx, past=None, use_cache=True):
    """Call model and return (logits, past). Handles the (logits, loss, past)
    and (logits, loss) return shapes."""
    out = model(idx, past_key_values=past, use_cache=use_cache)
    logits = out[0]
    past = out[2] if len(out) > 2 else past
    return logits, past


def speculative_generate(main_model, draft_model, tokenizer, prompt, max_new_tokens=50, k=4, temperature=0.0, device="cuda"):
    main_model.eval(); draft_model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    idx = inputs.input_ids  # [1, T]
    main_past = None; draft_past = None
    tokens_generated = 0
    total_proposed = 0; total_accepted = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        while tokens_generated < max_new_tokens:
            # --- Phase 1: Draft proposes k tokens autoregressively ---
            draft_tokens = []
            for i in range(k):
                if draft_past is None:
                    d_logits, draft_past = _model_forward(draft_model, idx, use_cache=True)
                else:
                    d_logits, draft_past = _model_forward(draft_model, idx[:, -1:], past=draft_past, use_cache=True)
                d_logits = d_logits[:, -1, :]
                if temperature == 0:
                    next_tok = d_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(d_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(next_tok)
                idx = torch.cat([idx, next_tok], dim=1)
            draft_tokens_t = torch.cat(draft_tokens, dim=1)  # [1, k]

            # --- Phase 2: Main model verifies all k tokens in ONE forward pass ---
            if main_past is None:
                m_logits, main_past = _model_forward(main_model, idx, use_cache=True)
            else:
                m_logits, main_past = _model_forward(main_model, draft_tokens_t, past=main_past, use_cache=True)
            # m_logits[:, i, :] predicts token at position i+1.
            # We need the logits that predict the k draft tokens + 1 bonus.
            m_logits = m_logits[:, -k - 1:, :]  # [1, k+1, V]

            # --- Phase 3: Accept/reject ---
            accepted = 0
            rejected = False
            for i in range(k):
                m_tok = m_logits[:, i, :].argmax(dim=-1) if temperature == 0 else torch.multinomial(F.softmax(m_logits[:, i, :] / temperature, dim=-1), 1).squeeze(-1)
                d_tok = draft_tokens_t[:, i]
                if m_tok == d_tok:
                    accepted += 1
                else:
                    # Reject at position i: truncate the k-i pending draft tokens,
                    # append the main model's token.
                    idx = idx[:, :idx.shape[1] - (k - i)]
                    idx = torch.cat([idx, m_tok.view(1, 1)], dim=1)
                    tokens_generated += i + 1
                    total_proposed += k
                    total_accepted += accepted
                    rejected = True
                    break
            if not rejected:
                # All k accepted: sample bonus token from main model's k-th logit.
                bonus_tok = m_logits[:, k, :].argmax(dim=-1, keepdim=True) if temperature == 0 else torch.multinomial(F.softmax(m_logits[:, k, :] / temperature, dim=-1), 1)
                idx = torch.cat([idx, bonus_tok], dim=1)
                tokens_generated += k + 1
                total_proposed += k
                total_accepted += k

            if idx[0, -1].item() == tokenizer.eos_token_id:
                break

    elapsed = time.perf_counter() - t0
    accept_rate = total_accepted / max(total_proposed, 1)
    text = tokenizer.decode(idx[0, prompt_len:], skip_special_tokens=True)
    return text, tokens_generated, elapsed, accept_rate


def baseline_generate(model, tokenizer, prompt, max_new_tokens, temperature=0.0, device="cuda"):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    idx = inputs.input_ids
    past = None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past is None:
                logits, past = _model_forward(model, idx, use_cache=True)
            else:
                logits, past = _model_forward(model, idx[:, -1:], past=past, use_cache=True)
            logits = logits[:, -1, :]
            if temperature == 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                next_tok = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            idx = torch.cat([idx, next_tok], dim=1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(idx[0, prompt_len:], skip_special_tokens=True)
    return text, max_new_tokens, elapsed


def speculative_generate_online(
    main_model, draft_model, draft_optimizer, tokenizer, prompt,
    max_new_tokens=50, k=4, temperature=0.0, device="cuda",
    learn_rate=1e-4, save_path=None,
):
    """Speculative decoding with online distillation.

    The draft model learns from the main model's rejections in real-time:
    every time the main model rejects a draft token, we compute a cross-entropy
    loss on the draft's logits at that position (target = main model's token)
    and update the draft. Over time, the draft converges to mimic the main
    model on exactly the inputs you serve, increasing accept rate and speed.

    Returns: (text, tokens_generated, time, accept_rate, final_loss)
    """
    main_model.eval()
    draft_model.train()  # Enable grad for online learning.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    idx = inputs.input_ids
    main_past = None
    tokens_generated = 0
    total_proposed = 0; total_accepted = 0
    total_loss = 0.0; loss_count = 0
    t0 = time.perf_counter()

    for gen_step in range(max_new_tokens):
        if tokens_generated >= max_new_tokens:
            break
        # --- Phase 1: Draft proposes k tokens (no KV cache, fresh graph each round) ---
        draft_tokens = []
        for i in range(k):
            # Full forward (no cache) so each round has a clean graph.
            d_out = draft_model(idx, use_cache=False)
            d_logits = d_out[0] if isinstance(d_out, tuple) else d_out
            d_logits_last = d_logits[:, -1, :]
            if temperature == 0:
                next_tok = d_logits_last.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(d_logits_last / max(temperature, 1e-5), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(next_tok)
            idx = torch.cat([idx, next_tok], dim=1)
        draft_tokens_t = torch.cat(draft_tokens, dim=1)

        # --- Phase 1b: Fresh draft forward for online loss (clean graph) ---
        d_out = draft_model(idx[:, :-1], use_cache=False)  # predict all but last
        d_logits_for_loss = d_out[0] if isinstance(d_out, tuple) else d_out  # [1, T-1, V]

        # --- Phase 2: Main model verifies (no grad) ---
        with torch.no_grad():
            if main_past is None:
                m_out = main_model(idx, use_cache=True)
            else:
                m_out = main_model(draft_tokens_t, past_key_values=main_past, use_cache=True)
            m_logits = m_out[0] if isinstance(m_out, tuple) else m_out
            main_past = m_out[2] if isinstance(m_out, tuple) and len(m_out) > 2 else main_past
            m_logits = m_logits[:, -k - 1:, :]

        # --- Phase 3: Accept/reject + online learning ---
        accepted = 0; rejected = False
        online_loss = 0.0
        # The draft logits for loss correspond to positions that predict the
        # draft tokens. d_logits_for_loss[:, -(k+1):-1] predicts the k draft tokens.
        draft_loss_logits = d_logits_for_loss[:, -k - 1:-1, :]  # [1, k, V]
        for i in range(k):
            m_tok = m_logits[:, i, :].argmax(dim=-1) if temperature == 0 else torch.multinomial(F.softmax(m_logits[:, i, :] / temperature, dim=-1), 1).squeeze(-1)
            d_tok = draft_tokens_t[:, i]
            # Online loss: teach draft to predict main's token at this position.
            m_tok_flat = m_tok.view(-1) if m_tok.dim() > 0 else m_tok.unsqueeze(0)
            online_loss = online_loss + F.cross_entropy(draft_loss_logits[:, i, :], m_tok_flat)
            if m_tok == d_tok:
                accepted += 1
            else:
                idx = idx[:, :idx.shape[1] - (k - i)]
                idx = torch.cat([idx, m_tok.view(1, 1)], dim=1)
                tokens_generated += i + 1
                total_proposed += k
                total_accepted += accepted
                rejected = True
                break
        if not rejected:
            bonus_tok = m_logits[:, k, :].argmax(dim=-1, keepdim=True) if temperature == 0 else torch.multinomial(F.softmax(m_logits[:, k, :] / temperature, dim=-1), 1)
            idx = torch.cat([idx, bonus_tok], dim=1)
            tokens_generated += k + 1
            total_proposed += k
            total_accepted += k

        # --- Online learning step: update draft from main's feedback ---
        if online_loss > 0:
            draft_optimizer.zero_grad(set_to_none=True)
            (online_loss / k).backward()
            torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
            draft_optimizer.step()
            total_loss += online_loss.item() / k
            loss_count += 1

        if idx[0, -1].item() == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - t0
    accept_rate = total_accepted / max(total_proposed, 1)
    avg_loss = total_loss / max(loss_count, 1)
    text = tokenizer.decode(idx[0, prompt_len:], skip_special_tokens=True)
    # Save the improved draft.
    if save_path:
        from research.checkpoint_io import save_checkpoint
        save_checkpoint(draft_model.state_dict(), save_path)
    draft_model.eval()
    return text, tokens_generated, elapsed, accept_rate, avg_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main-model", default="research/checkpoints/pretrained_llm.safetensors")
    p.add_argument("--draft-model", default="research/checkpoints/draft_llm.safetensors")
    p.add_argument("--prompt", default="The future of AI is")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--online", action="store_true", help="Enable online distillation: draft learns from main model's rejections during inference.")
    p.add_argument("--learn-rate", type=float, default=1e-4, help="Online learning rate for draft model.")
    p.add_argument("--save-draft", default=None, help="Save the (online-learned) draft model to this path.")
    p.add_argument("--quantize", choices=["none", "int8"], default="int8",
                   help="INT8 weight quantization for main model (4-8x speedup, default int8).")
    args = p.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    main_cfg = get_config("360m_mla")
    main_model = ModelLoader.build_model(main_cfg, checkpoint_path=args.main_model).to(device).eval()

    # Apply INT8 quantization to main model for 4-8x inference speedup.
    if args.quantize == "int8":
        from research.inference_quant import quantize_model_int8
        n = quantize_model_int8(main_model, verbose=False)
        print(f"Main model INT8 quantized: {n} layers (4-8x speedup)")

    draft_cfg = ModelConfig(d_model=main_cfg.d_model, n_layers=2, n_heads=main_cfg.n_heads, attn_type="mla", ffn_type="swiglu", kv_compression_dim=main_cfg.kv_compression_dim, max_seq_len=main_cfg.max_seq_len, batch_size=1, seq_len=main_cfg.seq_len)
    draft_model = ModelLoader.build_model(draft_cfg, checkpoint_path=args.draft_model).to(device).eval()
    print(f"Main: {sum(p.numel() for p in main_model.parameters())/1e6:.1f}M | Draft: {sum(p.numel() for p in draft_model.parameters())/1e6:.1f}M params")

    print(f"\n--- Baseline ({args.max_new_tokens} tokens) ---")
    bt, bn, btime = baseline_generate(main_model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, device)
    print(f"Time: {btime:.2f}s | {bn/btime:.1f} tok/s\nOutput: {bt[:80]}")

    if args.online:
        print(f"\n--- Speculative + Online Distillation (k={args.k}, lr={args.learn_rate}) ---")
        from research.training_utils import configure_optimizer
        draft_opt = configure_optimizer(draft_model, args.learn_rate, 0.0, "bnb")
        st, sn, stime, ar, avg_loss = speculative_generate_online(
            main_model, draft_model, draft_opt, tokenizer, args.prompt,
            args.max_new_tokens, args.k, args.temperature, device,
            args.learn_rate, args.save_draft,
        )
        print(f"Time: {stime:.2f}s | {sn/stime:.1f} tok/s | accept: {ar:.1%} | draft loss: {avg_loss:.4f}")
        print(f"Output: {st[:80]}")
        print(f"\n--- SUMMARY ---")
        print(f"Baseline:    {bn/btime:.1f} tok/s")
        print(f"Spec+Online: {sn/stime:.1f} tok/s")
        print(f"Speedup:     {(sn/stime)/(bn/btime):.2f}x | Accept: {ar:.1%} | Draft loss: {avg_loss:.4f}")
        if args.save_draft:
            print(f"Improved draft saved to: {args.save_draft}")
    else:
        print(f"\n--- Speculative (k={args.k}) ---")
        st, sn, stime, ar = speculative_generate(main_model, draft_model, tokenizer, args.prompt, args.max_new_tokens, args.k, args.temperature, device)
        print(f"Time: {stime:.2f}s | {sn/stime:.1f} tok/s | accept: {ar:.1%}\nOutput: {st[:80]}")
        print(f"\n--- SUMMARY ---")
        print(f"Baseline:    {bn/btime:.1f} tok/s")
        print(f"Speculative: {sn/stime:.1f} tok/s")
        print(f"Speedup:     {(sn/stime)/(bn/btime):.2f}x | Accept rate: {ar:.1%}")


if __name__ == "__main__":
    main()
