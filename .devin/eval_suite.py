"""Comprehensive model comparison: base vs SFT across 30+ diverse prompts.

Categories: knowledge, reasoning, math, code, instruction-following, chat, format,
translation, creative, safety, multi-turn, edge cases.

Scoring: exact-match for factual, keyword-match for open-ended, manual review for quality.
"""
import sys, os, time, json
sys.path.insert(0, "D:/windsurf/ForgeAI")
os.environ["FORGE_BITNET_KERNEL"] = "triton"
os.environ["FORGE_FUSED_ROPE_QKNORM"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from research.config import get_config
from research.model_loader import ModelLoader
from research.inference.forge_engine import ForgeEngine
from research.tokenizer_cache import get_tokenizer

device = "cuda"
dtype = torch.bfloat16
tok = get_tokenizer()

# ── 30+ diverse test prompts ──
TESTS = [
    # Knowledge (exact/keyword match)
    {"cat": "knowledge", "q": "What is the capital of France?", "expect": "Paris", "type": "keyword"},
    {"cat": "knowledge", "q": "What is the capital of Japan?", "expect": "Tokyo", "type": "keyword"},
    {"cat": "knowledge", "q": "What is the largest planet in our solar system?", "expect": "Jupiter", "type": "keyword"},
    {"cat": "knowledge", "q": "Who wrote Romeo and Juliet?", "expect": "Shakespeare", "type": "keyword"},
    {"cat": "knowledge", "q": "What is the chemical symbol for water?", "expect": "H2O", "type": "keyword"},
    {"cat": "knowledge", "q": "What is the boiling point of water in Celsius?", "expect": "100", "type": "keyword"},
    {"cat": "knowledge", "q": "What country has the most population?", "expect": "China", "type": "keyword"},
    {"cat": "knowledge", "q": "What is the currency of the United Kingdom?", "expect": "pound", "type": "keyword"},

    # Math
    {"cat": "math", "q": "What is 2+2?", "expect": "4", "type": "keyword"},
    {"cat": "math", "q": "What is 7 times 8?", "expect": "56", "type": "keyword"},
    {"cat": "math", "q": "What is 100 minus 37?", "expect": "63", "type": "keyword"},
    {"cat": "math", "q": "What is 15 divided by 3?", "expect": "5", "type": "keyword"},
    {"cat": "math", "q": "What is the square root of 144?", "expect": "12", "type": "keyword"},

    # Reasoning
    {"cat": "reasoning", "q": "If I have 3 apples and eat 1, how many do I have left?", "expect": "2", "type": "keyword"},
    {"cat": "reasoning", "q": "If today is Monday, what day is it in 3 days?", "expect": "Thursday", "type": "keyword"},
    {"cat": "reasoning", "q": "Which is heavier: a pound of feathers or a pound of lead?", "expect": "same", "type": "keyword"},

    # Code
    {"cat": "code", "q": "Write a Python one-liner to reverse a string.", "expect": "[::-1]", "type": "keyword"},
    {"cat": "code", "q": "What does len([1,2,3]) return in Python?", "expect": "3", "type": "keyword"},
    {"cat": "code", "q": "What is the time complexity of binary search?", "expect": "log", "type": "keyword"},

    # Instruction following
    {"cat": "instruction", "q": "Say hello in exactly 3 words.", "expect": "hello", "type": "keyword"},
    {"cat": "instruction", "q": "List three colors.", "expect": "red", "type": "keyword"},

    # Chat / conversational
    {"cat": "chat", "q": "Hello, how are you?", "expect": "", "type": "manual"},
    {"cat": "chat", "q": "What's your name?", "expect": "", "type": "manual"},
    {"cat": "chat", "q": "Tell me a joke.", "expect": "", "type": "manual"},

    # Format
    {"cat": "format", "q": "Write a haiku about the ocean (5-7-5 syllables).", "expect": "", "type": "manual"},
    {"cat": "format", "q": "Write a numbered list of 3 fruits.", "expect": "", "type": "manual"},

    # Translation
    {"cat": "translation", "q": "How do you say 'hello' in Spanish?", "expect": "hola", "type": "keyword"},
    {"cat": "translation", "q": "How do you say 'thank you' in French?", "expect": "merci", "type": "keyword"},

    # Creative
    {"cat": "creative", "q": "Write a one-sentence story about a cat.", "expect": "", "type": "manual"},
    {"cat": "creative", "q": "Describe the color blue to someone who can't see.", "expect": "", "type": "manual"},

    # Edge cases / robustness
    {"cat": "edge", "q": "What is the meaning of life?", "expect": "", "type": "manual"},
    {"cat": "edge", "q": "Repeat after me: The quick brown fox jumps.", "expect": "quick brown fox", "type": "keyword"},
]

def chat_prompt(msg):
    return f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"

def run_eval(engine, label, max_new=64):
    results = []
    for t in TESTS:
        prompt = chat_prompt(t["q"])
        t0 = time.perf_counter()
        out = engine.generate(prompt, max_new_tokens=max_new, temperature=0.0)
        gen_time = time.perf_counter() - t0
        out_clean = out.replace("<|im_end|>", "").strip()
        results.append({**t, "output": out_clean[:200], "time": gen_time})
    return results

def score(results):
    """Score keyword tests automatically, flag manual ones."""
    scores = {"pass": 0, "fail": 0, "manual": 0, "by_cat": {}}
    for r in results:
        cat = r["cat"]
        if cat not in scores["by_cat"]:
            scores["by_cat"][cat] = {"pass": 0, "fail": 0, "manual": 0}
        if r["type"] == "manual":
            scores["manual"] += 1
            scores["by_cat"][cat]["manual"] += 1
        else:
            passed = r["expect"].lower() in r["output"].lower()
            if passed:
                scores["pass"] += 1
                scores["by_cat"][cat]["pass"] += 1
            else:
                scores["fail"] += 1
                scores["by_cat"][cat]["fail"] += 1
    return scores

def boot_model(ckpt):
    cfg = get_config("forgelm_v3", device=device)
    model = ModelLoader.build_model_fast(cfg, checkpoint_path=ckpt, dtype=dtype)
    model.to(device).eval()
    return ForgeEngine(model, tok, device=device)

# ── Run evals ──
print(f"Running {len(TESTS)} prompts on base vs SFT...\n")

print("Booting BASE model...")
engine_base = boot_model("research/checkpoints/ForgeLM_V3_Base.safetensors")
results_base = run_eval(engine_base, "BASE")
del engine_base
torch.cuda.empty_cache()

print("Booting SFT model...")
engine_sft = boot_model("research/checkpoints/ForgeLM_V3_SFT.safetensors")
results_sft = run_eval(engine_sft, "SFT")
del engine_sft
torch.cuda.empty_cache()

# ── Score ──
scores_base = score(results_base)
scores_sft = score(results_sft)

print(f"\n{'='*70}")
print(f"RESULTS: {len(TESTS)} prompts ({scores_base['pass']+scores_base['fail']} auto-scored, {scores_base['manual']} manual)")
print(f"{'='*70}")
print(f"{'Category':<15} {'Base pass':>10} {'SFT pass':>10} {'Base fail':>10} {'SFT fail':>10}")
print(f"{'-'*55}")
for cat in sorted(scores_base["by_cat"].keys()):
    b = scores_base["by_cat"][cat]
    s = scores_sft["by_cat"][cat]
    print(f"{cat:<15} {b['pass']:>10} {s['pass']:>10} {b['fail']:>10} {s['fail']:>10}")

total_auto = scores_base["pass"] + scores_base["fail"]
print(f"{'-'*55}")
print(f"{'TOTAL':<15} {scores_base['pass']:>10}/{total_auto} {scores_sft['pass']:>10}/{total_auto} {scores_base['fail']:>10} {scores_sft['fail']:>10}")
print(f"{'Manual':<15} {scores_base['manual']:>10} {scores_sft['manual']:>10}")

# ── Detailed comparison for failed tests ──
print(f"\n{'='*70}")
print("FAILED TESTS (auto-scored):")
print(f"{'='*70}")
for i, (rb, rs) in enumerate(zip(results_base, results_sft)):
    if rb["type"] == "manual":
        continue
    b_pass = rb["expect"].lower() in rb["output"].lower()
    s_pass = rs["expect"].lower() in rs["output"].lower()
    if not b_pass or not s_pass:
        status_b = "PASS" if b_pass else "FAIL"
        status_s = "PASS" if s_pass else "FAIL"
        print(f"\n[{rb['cat']}] {rb['q']}")
        print(f"  Expected: {rb['expect']}")
        print(f"  Base ({status_b}): {rb['output'][:100]}")
        print(f"  SFT  ({status_s}): {rs['output'][:100]}")

# ── Manual review samples ──
print(f"\n{'='*70}")
print("MANUAL REVIEW (chat/creative/format):")
print(f"{'='*70}")
for i, (rb, rs) in enumerate(zip(results_base, results_sft)):
    if rb["type"] != "manual":
        continue
    print(f"\n[{rb['cat']}] {rb['q']}")
    print(f"  Base: {rb['output'][:150]}")
    print(f"  SFT:  {rs['output'][:150]}")

# ── Save full results ──
with open(".devin/eval_results.json", "w", encoding="utf-8") as f:
    json.dump({"base": results_base, "sft": results_sft,
               "scores_base": scores_base, "scores_sft": scores_sft}, f, indent=2, ensure_ascii=False)
print(f"\nFull results saved to .devin/eval_results.json")
