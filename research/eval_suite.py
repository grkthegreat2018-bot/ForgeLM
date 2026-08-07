"""Evaluation suite: perplexity, JSON tool-call syntax accuracy, throughput."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows terminals.
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer

from research.config import get_config
from research.model_loader import ModelLoader
from research.training_utils import BinaryDataset


SAMPLE_EVAL_PROMPTS = [
    'Call tool to fetch weather: <functions>[{"name": "get_weather", "parameters": {"city": "London"}}]</functions>',
    'Call tool to check stock price: <functions>[{"name": "get_stock", "parameters": {"symbol": "NVDA"}}]</functions>',
    'Calculate the value of (25 + 17) * 3 and then call the calculate tool.',
    'What files are in the project root? Use list_directory.',
    "Use the search_files tool to find where 'train.py' is mentioned.",
]


def evaluate_perplexity(model, val_dataset, batch_size, seq_len, device, eval_batches=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = val_dataset.get_batch(batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
    avg = sum(losses) / len(losses)
    ppl = math.exp(avg)
    return avg, ppl


def evaluate_json_compliance(model, tokenizer, prompts, max_new_tokens=128):
    model.eval()
    valid = 0
    total_tokens = 0
    start = time.time()
    results = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            idx = inputs.input_ids

            for _ in range(max_new_tokens):
                idx_cond = idx[:, -model.config.max_seq_len :]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / 0.1  # low temperature for deterministic eval
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                idx = torch.cat((idx, next_token), dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

            generated = tokenizer.decode(idx[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            total_tokens += len(idx[0]) - inputs.input_ids.shape[1]

            try:
                if "<functioncall>" in generated:
                    json_str = generated.split("<functioncall>")[1].split("</functioncall>")[0].strip()
                else:
                    json_str = generated.strip()
                json.loads(json_str)
                valid += 1
                status = "VALID"
            except Exception:
                status = "INVALID"

            results.append({"prompt": prompt, "status": status, "output": generated})
            print(f"[{i + 1}/{len(prompts)}] {status} | {generated[:80]}...")

    elapsed = time.time() - start
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    accuracy = valid / len(prompts) * 100
    return accuracy, tok_per_sec, results


def throughput_benchmark(model, tokenizer, prompt="The key to local LLM pre-training is", max_new_tokens=256):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    idx = inputs.input_ids

    start = time.time()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.max_seq_len :]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.7
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    elapsed = time.time() - start
    generated = len(idx[0]) - inputs.input_ids.shape[1]
    return generated, generated / elapsed, tokenizer.decode(idx[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ForgeAI research checkpoint.")
    parser.add_argument("--config", type=str, default="360m_mla")
    parser.add_argument("--checkpoint", type=str, default="research/checkpoints/sft_llm.pt")
    parser.add_argument("--val-bin", type=str, default="research/data/val.bin")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--quantize", choices=["none", "int8"], default="none",
                        help="INT8 weight quantization for faster inference (default: none for eval accuracy).")
    args = parser.parse_args()

    cfg = get_config(args.config)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.seq_len:
        cfg.seq_len = args.seq_len

    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    model = ModelLoader.build_model(cfg, checkpoint_path=args.checkpoint, compile=False).to(device)

    if args.quantize == "int8":
        from research.inference_quant import quantize_model_int8
        n = quantize_model_int8(model, verbose=False)
        print(f"INT8 quantization: {n} layers (faster inference, ~1% accuracy loss)")

    print("\n=== Perplexity Evaluation ===")
    val_dataset = BinaryDataset(args.val_bin, cfg.seq_len, cfg.vocab_size)
    val_loss, ppl = evaluate_perplexity(model, val_dataset, cfg.batch_size, cfg.seq_len, device, args.eval_batches)
    print(f"Val loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

    print("\n=== JSON Tool-Call Compliance ===")
    accuracy, tok_s, _ = evaluate_json_compliance(model, tokenizer, SAMPLE_EVAL_PROMPTS, args.max_new_tokens)
    print(f"JSON syntax accuracy: {accuracy:.1f}%")
    print(f"Generation speed: {tok_s:.2f} tok/s")

    print("\n=== Throughput Benchmark ===")
    gen_len, gen_tok_s, sample_text = throughput_benchmark(model, tokenizer, max_new_tokens=256)
    print(f"Generated {gen_len} tokens at {gen_tok_s:.2f} tok/s")
    print(f"Sample: {sample_text[:200]}...")


if __name__ == "__main__":
    main()
