"""Extract vocab packs from trained experts.

For each trained expert, we:
1. Run the base model on domain-specific prompts → get base logits
2. Load the expert → run same prompts → get expert logits
3. Find tokens where the probability distribution shifted significantly
4. Extract the embedding delta that would cause this shift
5. Save as a vocab pack (portable, injectable)

The vocab pack captures what the expert "knows" about domain-specific tokens.
At inference, injecting the pack into the base model's embedding makes the
base model recognize domain tokens even without the expert loaded.

Usage:
    python extract_vocab_packs.py
    python extract_vocab_packs.py --topic python_math --n-prompts 20
"""
import os
import sys
import json
import torch
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V2_CHECKPOINT = "research/checkpoints/forgelm_v2.safetensors"
V4_DIR = "research/checkpoints/forgelm_v4"
VOCAB_PACK_DIR = "research/checkpoints/vocab_packs"
TOKENIZER_PATH = "research/checkpoints/qwen_hf"

# Domain-specific prompts per topic (for measuring logit shifts)
DOMAIN_PROMPTS = {
    "python_math": [
        "def factorial(n):",
        "def is_prime(n):",
        "def gcd(a, b):",
        "def fibonacci(n):",
        "def power(base, exp):",
        "def sqrt_approx(n):",
        "def sum_range(a, b):",
        "def modulo_check(a, b):",
    ],
    "python_algorithms": [
        "def binary_search(arr, target):",
        "def merge_sort(arr):",
        "def quicksort(arr):",
        "def bfs(graph, start):",
        "def dfs(graph, start):",
        "def dijkstra(graph, src):",
        "def knapsack(weights, values, capacity):",
        "def lcs(s1, s2):",
    ],
    "python_strings": [
        "def reverse_string(s):",
        "def count_vowels(s):",
        "def is_palindrome(s):",
        "def split_words(text):",
        "def replace_chars(s, old, new):",
        "def format_string(template, **kwargs):",
        "def strip_whitespace(s):",
        "def camel_to_snake(s):",
    ],
    "python_oop": [
        "class Animal:",
        "class Stack:",
        "class Queue:",
        "class LinkedList:",
        "class BinaryTree:",
        "class Singleton:",
        "class AbstractFactory:",
        "class Observer:",
    ],
    "python_file_io": [
        "def read_file(path):",
        "def write_file(path, content):",
        "def append_log(path, message):",
        "def read_csv(path):",
        "def write_json(path, data):",
        "def list_dir(path):",
        "def copy_file(src, dst):",
        "def file_exists(path):",
    ],
    "python_general": [
        "def hello_world():",
        "def print_list(items):",
        "def dict_to_list(d):",
        "def list_to_dict(pairs):",
        "def flatten(nested):",
        "def unique(items):",
        "def group_by(items, key):",
        "def chunk(items, size):",
    ],
    "test_arithmetic": [
        "def add(a, b):",
        "def subtract(a, b):",
        "def multiply(a, b):",
        "def divide(a, b):",
        "def square(n):",
        "def cube(n):",
        "def double(n):",
        "def halve(n):",
    ],
}


@torch.no_grad()
def get_logit_distribution(model, tokenizer, prompt, device="cuda", top_k=50):
    """Get the top-K logit distribution for the next token after prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    logits, _ = model(input_ids, use_cache=False)
    next_logits = logits[0, -1]
    probs = torch.softmax(next_logits, dim=-1)
    topk_probs, topk_idx = probs.topk(top_k)
    return {idx.item(): prob.item() for idx, prob in zip(topk_idx, topk_probs)}


def extract_vocab_pack_for_topic(topic, model, tokenizer, expert_loader,
                                 device="cuda", n_prompts=8, top_k=50):
    """Extract vocab pack for one topic.

    Compares base model logit distribution vs expert-loaded distribution
    on domain prompts. Tokens that shift significantly are the domain vocab.

    Returns:
        {"topic": str, "token_ids": list, "prob_deltas": tensor,
         "embedding_deltas": tensor or None}
    """
    prompts = DOMAIN_PROMPTS.get(topic, [])[:n_prompts]
    if not prompts:
        print(f"  No prompts for topic {topic}, skipping")
        return None

    # Phase 1: Get base model distributions
    base_dists = []
    for prompt in prompts:
        dist = get_logit_distribution(model, tokenizer, prompt, device, top_k)
        base_dists.append(dist)

    # Phase 2: Load expert and get expert distributions
    expert_loaded = False
    try:
        expert_loader(topic)
        expert_loaded = True
    except Exception as e:
        print(f"  Could not load expert for {topic}: {e}")
        return None

    expert_dists = []
    for prompt in prompts:
        dist = get_logit_distribution(model, tokenizer, prompt, device, top_k)
        expert_dists.append(dist)

    # Phase 3: Find tokens with significant probability shifts
    token_shifts = Counter()
    for base, expert in zip(base_dists, expert_dists):
        all_tokens = set(base.keys()) | set(expert.keys())
        for tok in all_tokens:
            b = base.get(tok, 0.0)
            e = expert.get(tok, 0.0)
            shift = abs(e - b)
            if shift > 0.005:  # 0.5% probability shift threshold
                token_shifts[tok] += shift

    # Sort by total shift, keep top tokens
    sorted_tokens = token_shifts.most_common(100)
    token_ids = [t for t, _ in sorted_tokens]
    prob_deltas = torch.tensor([s for _, s in sorted_tokens], dtype=torch.float32)

    # Phase 4: Extract embedding deltas
    # The embedding is tied to the LM head: model.embed.weight == model.head.weight
    # The expert changes FFN weights, not embeddings directly. But the effective
    # vocabulary shift can be approximated by the logit shift direction.
    # We store the token IDs + probability shifts as the pack.
    # At injection time, we bias the logits for these tokens by the delta.

    # Decode token IDs to text for inspection
    token_texts = []
    for tid in token_ids[:20]:
        try:
            text = tokenizer.decode([tid])
            token_texts.append(text)
        except Exception:
            token_texts.append(f"<{tid}>")

    print(f"  {topic}: {len(token_ids)} shifted tokens, top: {token_texts[:10]}")

    return {
        "topic": topic,
        "token_ids": token_ids,
        "prob_deltas": prob_deltas,
        "n_prompts": len(prompts),
        "top_tokens": token_texts,
    }


def save_vocab_pack(pack, output_dir):
    """Save vocab pack to disk."""
    topic = pack["topic"]
    save_path = Path(output_dir) / f"vocab_pack_{topic}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "topic": topic,
        "token_ids": pack["token_ids"],
        "prob_deltas": pack["prob_deltas"],
        "n_prompts": pack["n_prompts"],
        "top_tokens": pack["top_tokens"],
    }, str(save_path))
    print(f"  Saved: {save_path} ({save_path.stat().st_size / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Extract vocab packs from trained experts")
    parser.add_argument("--topic", default=None, help="Specific topic (default: all)")
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--output", default=VOCAB_PACK_DIR)
    args = parser.parse_args()

    print("=" * 70)
    print("Extract Vocab Packs from Trained Experts")
    print("=" * 70)

    from research.config import get_config
    from research.model_loader import ModelLoader
    from transformers import AutoTokenizer

    device = "cuda"
    cfg = get_config("forgelm_v2", device=device)
    model = ModelLoader.build_model_fast(
        cfg, checkpoint_path=V2_CHECKPOINT, moe_top_k=0,
        dtype=torch.bfloat16)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Expert loader: loads expert weights into model
    from safetensors.torch import load_file

    def make_expert_loader(topic):
        def load_expert(topic_name=topic):
            expert_dir = Path(V4_DIR) / "experts"
            model_dtype = next(model.parameters()).dtype
            loaded = 0
            for layer in range(len(model.blocks)):
                shard_path = expert_dir / f"expert_l{layer}_{topic}.safetensors"
                if not shard_path.exists():
                    continue
                state = load_file(str(shard_path))
                # Navigate to the routed expert (expert index 0 in dense-bypass MoE)
                block = model.blocks[layer]
                if not hasattr(block.ffn, "experts") or len(block.ffn.experts) == 0:
                    continue
                expert = block.ffn.experts[0]
                for part in ["w1", "w2", "w3"]:
                    U = state.get(f"{part}_U")
                    S = state.get(f"{part}_S")
                    Vh = state.get(f"{part}_Vh")
                    if U is not None and S is not None and Vh is not None:
                        # Reconstruct full weight from SVD: W = U @ diag(S) @ Vh
                        W = (U.float() * S.float().unsqueeze(0)) @ Vh.float()
                        getattr(expert, part).weight.data = W.to(device, model_dtype)
                        loaded += 1
            return loaded > 0
        return load_expert

    # Determine topics
    if args.topic:
        topics = [args.topic]
    else:
        # Scan for trained experts
        expert_dir = Path(V4_DIR) / "experts"
        topics = []
        if expert_dir.exists():
            for f in expert_dir.iterdir():
                if f.name.startswith(".trained_"):
                    topics.append(f.name[len(".trained_"):])
        if not topics:
            # Fall back to manifest topics
            manifest_path = Path(V4_DIR) / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                topics = list(manifest.get("topics", {}).keys())

    print(f"\nTopics: {topics}")

    # Extract vocab pack for each topic
    for topic in topics:
        print(f"\n[{topic}]")
        pack = extract_vocab_pack_for_topic(
            topic, model, tokenizer, make_expert_loader(topic),
            device=device, n_prompts=args.n_prompts)
        if pack is not None:
            save_vocab_pack(pack, args.output)

    print(f"\n{'=' * 70}")
    print(f"Done! Vocab packs in {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
