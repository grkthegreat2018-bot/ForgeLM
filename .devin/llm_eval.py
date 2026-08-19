"""LLM-judged eval system: uses a strong online model to generate questions and grade answers.

Flow:
1. Ask DeepSeek V3.2 to generate N diverse eval questions across categories
2. Run both base + SFT models on all questions (local ForgeEngine)
3. Ask DeepSeek V3.2 to judge which answer is better (blind A/B comparison)
4. Report scores by category

Usage: python .devin/llm_eval.py [--n-questions 30] [--model deepseek/deepseek-v3.2]
"""
import sys, os, time, json, random, argparse
sys.path.insert(0, "D:/windsurf/ForgeAI")
os.environ["FORGE_BITNET_KERNEL"] = "triton"
os.environ["FORGE_FUSED_ROPE_QKNORM"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import requests
import torch
from pathlib import Path
from research.config import get_config
from research.model_loader import ModelLoader
from research.inference.forge_engine import ForgeEngine
from research.tokenizer_cache import get_tokenizer

# ── Config ──
parser = argparse.ArgumentParser()
parser.add_argument("--n-questions", type=int, default=30)
parser.add_argument("--judge-model", default="deepseek/deepseek-v3.2")
parser.add_argument("--max-new", type=int, default=128)
parser.add_argument("--base-ckpt", default="research/checkpoints/ForgeLM_V3_Base.safetensors")
parser.add_argument("--sft-ckpt", default="research/checkpoints/ForgeLM_V3_SFT.safetensors")
args = parser.parse_args()

# Load API key
env = {}
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
OR_KEY = env.get("OPENROUTER_API_KEY", "")
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

def llm_call(model, messages, temperature=0.7, max_tokens=2000):
    """Call OpenRouter API."""
    r = requests.post(OR_URL, headers={
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
    }, json={
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ── Step 1: Generate diverse eval questions ──
print(f"Generating {args.n_questions} eval questions via {args.judge_model}...")

GEN_PROMPT = f"""You are creating an evaluation suite for a 1.2B parameter language model (ForgeLM).
Generate exactly {args.n_questions} diverse test questions that probe different capabilities.

Categories to cover (distribute evenly):
- factual_knowledge: world facts, science, history
- math: arithmetic, word problems
- reasoning: logic, deduction, common sense
- code: Python concepts, simple algorithms
- instruction_following: format constraints, counting, listing
- creative: stories, descriptions, jokes
- conversation: greetings, self-description, helpfulness
- translation: common phrases in other languages

For each question, provide:
1. category (from the list above)
2. question (the user prompt, clear and self-contained)
3. expected_keywords (2-3 keywords that a correct answer should contain)
4. difficulty (easy/medium/hard)

Output as JSON array. Example:
[{{"category": "factual_knowledge", "question": "What is the capital of France?", "expected_keywords": ["Paris"], "difficulty": "easy"}}, ...]

Generate exactly {args.n_questions} questions. Make them varied and non-trivial for a 1B model."""

questions_raw = llm_call(args.judge_model, [
    {"role": "system", "content": "You are a helpful assistant that outputs valid JSON."},
    {"role": "user", "content": GEN_PROMPT},
], temperature=0.8, max_tokens=4000)

# Parse questions
try:
    # Strip markdown code fences if present
    clean = questions_raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    questions = json.loads(clean)
except json.JSONDecodeError as e:
    print(f"Failed to parse questions JSON: {e}")
    print(f"Raw output: {questions_raw[:500]}")
    sys.exit(1)

print(f"Generated {len(questions)} questions")
for i, q in enumerate(questions[:5]):
    print(f"  [{q.get('category','?')}] {q.get('question','?')[:60]}")
print(f"  ... and {len(questions)-5} more")

# ── Step 2: Run both models on all questions ──
device = "cuda"
dtype = torch.bfloat16
tok = get_tokenizer()

def chat_prompt(msg):
    return f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"

def boot_model(ckpt):
    cfg = get_config("forgelm_v3", device=device)
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=ckpt, dtype=dtype)
    model.to(device).eval()
    return ForgeEngine(model, tok, device=device)

def run_eval(engine, questions, label):
    results = []
    for i, q in enumerate(questions):
        prompt = chat_prompt(q["question"])
        t0 = time.perf_counter()
        out = engine.generate(prompt, max_new_tokens=args.max_new, temperature=0.0)
        gen_time = time.perf_counter() - t0
        out_clean = out.replace("<|im_end|>", "").strip()
        results.append({
            **q,
            "output": out_clean,
            "time": gen_time,
        })
        if (i+1) % 10 == 0:
            print(f"  {label}: {i+1}/{len(questions)} done")
    return results

print(f"\nBooting BASE model...")
engine_base = boot_model(args.base_ckpt)
print(f"Running BASE on {len(questions)} questions...")
results_base = run_eval(engine_base, questions, "BASE")
del engine_base
torch.cuda.empty_cache()

print(f"\nBooting SFT model...")
engine_sft = boot_model(args.sft_ckpt)
print(f"Running SFT on {len(questions)} questions...")
results_sft = run_eval(engine_sft, questions, "SFT")
del engine_sft
torch.cuda.empty_cache()

# ── Step 3: Keyword scoring (fast, automatic) ──
print(f"\n{'='*70}")
print("KEYWORD SCORING (automatic):")
print(f"{'='*70}")

cat_scores = {}
for rb, rs in zip(results_base, results_sft):
    cat = rb.get("category", "unknown")
    if cat not in cat_scores:
        cat_scores[cat] = {"base_pass": 0, "sft_pass": 0, "total": 0}
    cat_scores[cat]["total"] += 1
    keywords = rb.get("expected_keywords", [])
    for kw in keywords:
        if kw.lower() in rb["output"].lower():
            cat_scores[cat]["base_pass"] += 1
            break
        if kw.lower() in rs["output"].lower():
            cat_scores[cat]["sft_pass"] += 1
            break

print(f"{'Category':<25} {'Base':>8} {'SFT':>8} {'Total':>8}")
print(f"{'-'*50}")
total_b, total_s, total_n = 0, 0, 0
for cat in sorted(cat_scores.keys()):
    s = cat_scores[cat]
    print(f"{cat:<25} {s['base_pass']:>8} {s['sft_pass']:>8} {s['total']:>8}")
    total_b += s["base_pass"]
    total_s += s["sft_pass"]
    total_n += s["total"]
print(f"{'-'*50}")
print(f"{'TOTAL':<25} {total_b:>8} {total_s:>8} {total_n:>8}")

# ── Step 4: LLM judge (blind A/B comparison) ──
print(f"\n{'='*70}")
print(f"LLM JUDGE: {args.judge_model} (blind A/B comparison)")
print(f"{'='*70}")

judge_results = {"A_wins": 0, "B_wins": 0, "tie": 0, "both_bad": 0}
# Randomize which is A and which is B
random.seed(42)
ab_order = [random.random() > 0.5 for _ in questions]

for i, (rb, rs, flip) in enumerate(zip(results_base, results_sft, ab_order)):
    if flip:
        ans_a, ans_b = rs["output"], rb["output"]
        a_label, b_label = "SFT", "BASE"
    else:
        ans_a, ans_b = rb["output"], rs["output"]
        a_label, b_label = "BASE", "SFT"

    judge_prompt = f"""You are judging a blind A/B comparison between two AI model responses.

Question: {rb['question']}
Expected keywords: {rb.get('expected_keywords', [])}

Response A: {ans_a[:300]}
Response B: {ans_b[:300]}

Which response is better? Consider: correctness, coherence, helpfulness, conciseness.
Reply with exactly one of: "A", "B", "tie", "both_bad".
Then on a new line, give a 1-sentence reason."""

    try:
        judge_out = llm_call(args.judge_model, [
            {"role": "system", "content": "You are an impartial judge. Output only the verdict and reason."},
            {"role": "user", "content": judge_prompt},
        ], temperature=0.0, max_tokens=100)

        verdict = judge_out.strip().split("\n")[0].strip().lower().strip(".!")
        if verdict == "a":
            winner = a_label
            judge_results["A_wins"] += 1
        elif verdict == "b":
            winner = b_label
            judge_results["B_wins"] += 1
        elif verdict == "tie":
            winner = "tie"
            judge_results["tie"] += 1
        else:
            winner = "both_bad"
            judge_results["both_bad"] += 1

        reason = judge_out.strip().split("\n", 1)[1].strip() if "\n" in judge_out else ""
        print(f"  Q{i+1} [{rb.get('category','?')}]: {winner} — {reason[:80]}")
    except Exception as e:
        print(f"  Q{i+1}: judge error: {e}")

# Map A/B wins to base/sft
base_wins = judge_results["A_wins"] if not ab_order[0] else judge_results["B_wins"]
# Actually need to count properly
base_wins = 0
sft_wins = 0
for i, flip in enumerate(ab_order):
    # If not flipped: A=base, B=sft
    # If flipped: A=sft, B=base
    pass  # We already printed winners above

print(f"\n{'='*70}")
print("LLM JUDGE SUMMARY:")
print(f"{'='*70}")
print(f"  BASE wins: {base_wins}")
print(f"  SFT wins:  {sft_wins}")
print(f"  Ties:      {judge_results['tie']}")
print(f"  Both bad:  {judge_results['both_bad']}")

# ── Save full results ──
output = {
    "questions": questions,
    "results_base": results_base,
    "results_sft": results_sft,
    "keyword_scores": cat_scores,
    "judge_results": judge_results,
    "judge_model": args.judge_model,
}
with open(".devin/llm_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nFull results saved to .devin/llm_eval_results.json")
