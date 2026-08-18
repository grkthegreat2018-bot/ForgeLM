"""Debug: see what the model generates when trying to solve an API task."""
import os, sys, torch
from pathlib import Path
env_path = Path("D:/windsurf/ForgeAI/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from research.distillation.distill_client import DistillationClient, DistillModel, MODEL_POOL
from research.model_loader import load_default_model
from research.tokenizer_cache import get_tokenizer
from research.self_play.infinite_curriculum import InfiniteCurriculum, ProposedTask
import ast

# Use a known-good model from Groq
groq_models = [m for m in MODEL_POOL if m.provider == "groq"]
print(f"Groq models: {[(m.model_id, m.provider) for m in groq_models]}")

client = DistillationClient()
# Generate tasks directly with a specific model
model_info = groq_models[0]  # gpt-oss-120b on groq
print(f"Using: {model_info.model_id} ({model_info.provider})")

api_client = client._get_client(model_info)
if api_client is None:
    print("No client!")
    sys.exit(1)

prompt = """Generate 5 diverse Python coding tasks. Each task should:
1. Require writing a Python function
2. Have 2-5 test cases with specific inputs and expected outputs
3. Range from easy to hard
4. Cover diverse topics: strings, math, data structures, algorithms, etc.

Output one task per line as JSON:
{"task": "...", "test_cases": [{"input": "...", "output": "..."}]}

Example:
{"task": "Write a Python function is_palindrome(s) that returns True if s is a palindrome", "test_cases": [{"input": "racecar", "output": "True"}, {"input": "hello", "output": "False"}]}"""

messages = [
    {"role": "system", "content": "You are a task generator for an AI training pipeline."},
    {"role": "user", "content": prompt},
]

response = api_client.chat.completions.create(
    model=model_info.model_id, messages=messages,
    temperature=0.7, max_completion_tokens=4096, timeout=60,
)
content = response.choices[0].message.content or ""
print(f"\n--- API Response (first 1000 chars) ---\n{content[:1000]}\n")

# Parse tasks
import json
tasks = []
for line in content.split("\n"):
    line = line.strip()
    if not line or not line.startswith("{"):
        continue
    try:
        task = json.loads(line)
        if "task" in task:
            tasks.append(task)
    except json.JSONDecodeError:
        continue

print(f"Parsed {len(tasks)} tasks")
for t in tasks[:3]:
    print(f"  task: {t['task'][:80]}")
    print(f"  tests: {t.get('test_cases', [])[:2]}")

if not tasks:
    sys.exit(0)

# Now load the local model and try to solve
print("\n--- Loading local model ---")
model, tokenizer = load_default_model("forgelm_v3",
    checkpoint_path="research/checkpoints/ForgeLM_V3_SFT.safetensors",
    device="cuda", dtype=torch.bfloat16)
model.eval()

curr = InfiniteCurriculum(model=model, tokenizer=tokenizer, device="cuda",
                          max_gen_tokens=256, temperature=0.7, top_k=80, top_p=0.95)

# Manually create a ProposedTask from the first API task
t = tasks[0]
desc = t["task"]
raw_tests = t.get("test_cases", [])

# Convert test cases
test_cases = []
for tc in raw_tests:
    if isinstance(tc, dict):
        inp = tc.get("input", "")
        out = tc.get("output", "")
    elif isinstance(tc, (list, tuple)) and len(tc) >= 2:
        inp, out = tc[0], tc[1]
    else:
        continue
    try:
        if isinstance(inp, str):
            try:
                inp_parsed = ast.literal_eval(inp.strip())
            except (ValueError, SyntaxError):
                inp_parsed = inp.strip()
        else:
            inp_parsed = inp
        if isinstance(out, str):
            try:
                out_parsed = ast.literal_eval(out.strip())
            except (ValueError, SyntaxError):
                out_parsed = out.strip()
        else:
            out_parsed = out
        args = (inp_parsed,) if not isinstance(inp_parsed, tuple) else inp_parsed
        test_cases.append({"args": args, "expected": out_parsed})
    except Exception as e:
        print(f"  test parse error: {e}")
        continue

if len(test_cases) < 2:
    print("Not enough test cases")
    sys.exit(0)

# Infer signature
first_args = test_cases[0]["args"]
type_hints = []
for a in first_args:
    if isinstance(a, int): type_hints.append("int")
    elif isinstance(a, str): type_hints.append("str")
    elif isinstance(a, list): type_hints.append("list")
    else: type_hints.append("any")
params = ", ".join(f"arg{i}: {t}" for i, t in enumerate(type_hints))
ret = test_cases[0]["expected"]
if isinstance(ret, int): ret_type = "int"
elif isinstance(ret, str): ret_type = "str"
elif isinstance(ret, list): ret_type = "list"
else: ret_type = "any"
signature = f"({params}) -> {ret_type}"

ptask = ProposedTask(
    id="test_1", domain="algorithms", difficulty="easy",
    description=desc, signature=signature, solve_name="solve",
    test_cases=test_cases, archetype="api_generated",
    raw_output="", generation_mode="api",
)

print(f"\nTask: {desc[:100]}")
print(f"Signature: {signature}")
print(f"Test cases: {test_cases[:3]}")

# Solve it
result = curr.solve_task(ptask)
print(f"\nSuccess: {result.get('final_success', False)}")
attempts = result.get("attempts", [])
print(f"Attempts: {len(attempts)}")
if attempts:
    att = attempts[0]
    code = att.get("code", "")
    print(f"\n--- Generated code (first 800 chars) ---")
    print(code[:800])
    print(f"\n--- Error ---")
    print(att.get("error", "")[:500])
