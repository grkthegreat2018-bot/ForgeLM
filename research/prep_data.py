"""Offline tokenization pipeline for ForgeAI pre-training.

Streams the FineWeb-Edu / Cosmopedia-v2 / Stack-Edu-Python mix and writes
memory-mapped uint32 binary files (train.bin / val.bin) for zero-overhead
disk-to-VRAM transfers.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import interleave_datasets, load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

from research.task_logger import task_scope

sys.stdout.reconfigure(encoding="utf-8")

# Load .env (HF_TOKEN, etc.) and give Hugging Face downloads a longer leash.
load_dotenv()
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


DEFAULT_TOKENIZER = "Qwen/Qwen2.5-0.5B"
DATA_DIR = Path(__file__).resolve().parent / "data"

# How many streamed samples are accumulated before invoking the encoder.
# For the native gigatoken backend this is the unit of parallelism handed to
# Rust; for hf/compat it is just a loop bound (no extra memory cost).
ENCODE_BATCH = 512


def build_encoder(backend, model_name, task=None):
    """Return (encode_fn, eos_id, vocab_size) for the requested backend.

    encode_fn(texts: list[str]) -> list[list[int]] is the uniform hot path used
    by build_binary_file. Special tokens are never added by the encoder; the
    caller appends eos_id itself so behaviour is identical across backends.
    """
    # Always load the HF tokenizer first: it is the source of truth for
    # eos_token_id and vocab_size metadata, and is the encoder for `hf`.
    if task:
        task.log(f"Loading tokenizer: {model_name} (backend={backend})")
    hf_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    eos_id = hf_tok.eos_token_id
    vocab_size = len(hf_tok)

    if backend == "hf":
        def encode_fn(texts):
            return [
                hf_tok.encode(t, add_special_tokens=False, truncation=False)
                for t in texts
            ]
        return encode_fn, eos_id, vocab_size

    # Lazy-import gigatoken so the `hf` backend works without it installed.
    import gigatoken as gt

    if backend == "gigatoken-compat":
        # Drop-in HF-compatible wrapper. Per-sample encode, exact output parity
        # with HF, ~30-40x faster on a typical desktop CPU.
        compat = gt.Tokenizer(hf_tok).as_hf()

        def encode_fn(texts):
            return [compat.encode(t, add_special_tokens=False) for t in texts]
        return encode_fn, eos_id, vocab_size

    if backend == "gigatoken-native":
        # Native gigatoken tokenizer. encode_batch_list parallelizes the whole
        # batch in Rust and returns list[list[int]] preserving per-document
        # boundaries (so per-doc EOS insertion stays exact).
        # NOTE: gt.encode_files (disk-spill) is intentionally NOT used here:
        # it returns a flat token stream with no document boundaries, which
        # would make per-document EOS insertion impossible without fragile
        # delimiter-recovery logic.
        native = gt.Tokenizer(model_name)

        def encode_fn(texts):
            return native.encode_batch_list(texts)
        return encode_fn, eos_id, vocab_size

    raise ValueError(f"Unknown tokenizer backend: {backend}")


def build_binary_file(encode_fn, eos_id, filename, target_tokens, stream_iter, task=None, buffer_size=1_000_000, encode_batch=ENCODE_BATCH):
    """Write `target_tokens` token ids to `filename` as uint32 flat binary.

    Samples are pulled from `stream_iter` in batches of `encode_batch`, encoded
    via `encode_fn`, and an `eos_id` is appended after every document. Output
    is truncated to exactly `target_tokens` once the buffer fills past it.
    """
    token_buffer = []
    total_written = 0
    msg = f"Building {filename} (target {target_tokens / 1e6:.1f}M tokens)"
    print(msg)
    if task:
        task.log(msg)

    def flush_to_file(f):
        """Drain whole buffer_size chunks; return True if target reached."""
        nonlocal total_written, token_buffer
        while len(token_buffer) >= buffer_size:
            chunk = token_buffer[:buffer_size]
            arr = np.array(chunk, dtype=np.uint32)
            f.write(arr.tobytes())
            total_written += len(chunk)
            token_buffer = token_buffer[buffer_size:]
            progress_msg = f"Written {total_written / 1e6:.1f}M tokens"
            print(progress_msg, end="\r")
            if task:
                task.update(progress={"written_m": round(total_written / 1e6, 2), "target_m": round(target_tokens / 1e6, 1)})

    with open(filename, "wb") as f:
        text_batch = []
        for sample in stream_iter:
            text = sample.get("text") or sample.get("content") or ""
            if len(text) < 100:
                continue
            text_batch.append(text)
            if len(text_batch) < encode_batch:
                continue

            # Encode the accumulated batch and fold per-document EOS into the buffer.
            for ids in encode_fn(text_batch):
                ids.append(eos_id)
                token_buffer.extend(ids)
            text_batch = []

            # If we have enough tokens to finish, write exactly the needed amount and stop.
            if total_written + len(token_buffer) >= target_tokens:
                need = target_tokens - total_written
                chunk = token_buffer[:need]
                arr = np.array(chunk, dtype=np.uint32)
                f.write(arr.tobytes())
                total_written += len(chunk)
                if task:
                    task.update(progress={"written_m": round(total_written / 1e6, 2), "target_m": round(target_tokens / 1e6, 1)})
                    task.log(f"Written {total_written / 1e6:.2f}M tokens")
                break

            flush_to_file(f)

        # Flush any trailing partial batch.
        if text_batch:
            for ids in encode_fn(text_batch):
                ids.append(eos_id)
                token_buffer.extend(ids)
            text_batch = []

        # Flush any remainder if target wasn't reached.
        if token_buffer and total_written < target_tokens:
            arr = np.array(token_buffer, dtype=np.uint32)
            f.write(arr.tobytes())
            total_written += len(token_buffer)

    print(f"\nCompleted {filename}: {total_written / 1e6:.2f}M tokens")
    if task:
        task.log(f"Completed {filename}: {total_written / 1e6:.2f}M tokens")
    return total_written


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize a 100M-token pre-training mix.")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--tokenizer-backend",
        type=str,
        default="hf",
        choices=["hf", "gigatoken-compat", "gigatoken-native"],
        help=(
            "Tokenizer backend for bulk encode. 'hf' = HuggingFace AutoTokenizer "
            "(default, no extra deps). 'gigatoken-compat' = gigatoken HFCompat "
            "wrapper, exact HF parity, ~30-40x faster on desktop CPUs. "
            "'gigatoken-native' = gigatoken native encode_batch_list, batched in "
            "Rust, largest speedup on many-core CPUs."
        ),
    )
    parser.add_argument("--encode-batch", type=int, default=ENCODE_BATCH, help="Samples accumulated per encode call (native backend).")
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--local-jsonl", type=str, default=None,
                        help="Path to a local JSONL file with a 'text' field. When set, tokenizes from this file instead of streaming HF datasets.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with task_scope("prep_data") as task:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        train_file = data_dir / "train.bin"
        val_file = data_dir / "val.bin"

        encode_fn, eos_id, vocab_size = build_encoder(args.tokenizer_backend, args.tokenizer, task=task)
        task.log(f"Vocab size: {vocab_size} | EOS id: {eos_id} | backend: {args.tokenizer_backend}")

        if args.local_jsonl:
            # Local JSONL mode: stream from a downloaded file instead of HF.
            task.log(f"Streaming from local JSONL: {args.local_jsonl}")

            def local_stream():
                with open(args.local_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        yield {"text": row.get("text", "")}

            stream_iter = local_stream()
        else:
            task.log("Streaming datasets...")
            ds_web = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
            # Cosmopedia v2 lives under the HuggingFaceTB/cosmopedia-v2 namespace and needs its config name.
            ds_synth = load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True)
            # python-finest-pretrain has the actual `text` content ready to stream. The original
            # Stack-Edu Python dataset only contains SWHIDs and requires an S3 download step, so we
            # use this functionally equivalent high-quality Python corpus for the offline pipeline.
            ds_code = load_dataset("Yxanul/python-finest-pretrain", split="train", streaming=True)

            mixed_stream = interleave_datasets(
                [ds_web, ds_synth, ds_code],
                probabilities=[0.50, 0.30, 0.20],
                seed=args.seed,
                stopping_strategy="first_exhausted",
            )

            stream_iter = iter(mixed_stream)

        # Validation split is tokenized first from the stream so it has a fixed seed window.
        val_written = build_binary_file(encode_fn, eos_id, val_file, args.val_tokens, stream_iter, task=task, encode_batch=args.encode_batch)
        train_written = build_binary_file(encode_fn, eos_id, train_file, args.train_tokens, stream_iter, task=task, encode_batch=args.encode_batch)

        metadata = {
            "tokenizer": args.tokenizer,
            "tokenizer_backend": args.tokenizer_backend,
            "vocab_size": vocab_size,
            "eos_token_id": int(eos_id),
            "dtype": "uint32",
            "train_tokens": int(train_written),
            "val_tokens": int(val_written),
            "seed": args.seed,
        }
        with open(data_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        task.log("Data pipeline complete.")
        task.update(metrics={"train_gb": round(train_file.stat().st_size / 1e9, 2), "val_gb": round(val_file.stat().st_size / 1e9, 2)})
        print(f"  train.bin: {train_file.stat().st_size / 1e9:.2f} GB")
        print(f"  val.bin:   {val_file.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
