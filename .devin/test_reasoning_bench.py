"""Test data loading for all reasoning benchmarks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("Reasoning Benchmark Data Loading Test")
print("=" * 60)

# Test 1: ARC-AGI-2
print("\n[1] ARC-AGI-2...")
from research.reasoning_benchmarks import load_arc_agi2, build_arc_agi2_prompt
tasks = load_arc_agi2(max_tasks=2)
assert len(tasks) >= 1, "No ARC-AGI-2 tasks loaded"
t = tasks[0]
assert "train" in t, "Missing train examples"
assert "test" in t, "Missing test input"
assert len(t["train"]) >= 1, "No training examples"
prompt = build_arc_agi2_prompt(t)
assert "Training Examples" in prompt
assert "Test Input" in prompt
print(f"  OK: {len(tasks)} tasks, first task has {len(t['train'])} examples")
print(f"  Prompt length: {len(prompt)} chars")

# Test 2: NeoCoder
print("\n[2] NeoCoder...")
from research.reasoning_benchmarks import load_neocoder, _build_neocoder_prompt, _extract_technique
problems, human_sols, test_cases = load_neocoder(max_problems=2)
assert len(problems) >= 1, "No NeoCoder problems loaded"
p = problems[0]
prompt = _build_neocoder_prompt(p)
assert "Problem" in prompt or "statement" in prompt.lower() or len(prompt) > 50
print(f"  OK: {len(problems)} problems, {len(human_sols)} human solutions, {len(test_cases)} test cases")
# Test technique extraction
tech = _extract_technique("def solve():\n    x = sorted(arr)\n    dp = [0]*n\n    return dp[-1]")
assert "sorting" in tech or "dynamic programming" in tech
print(f"  OK: technique extraction: {tech}")

# Test 3: FineReason
print("\n[3] FineReason...")
from research.reasoning_benchmarks import load_finereason, build_finereason_prompt
puzzles = load_finereason(max_problems=3)
assert len(puzzles) >= 1, "No FineReason puzzles loaded"
p = puzzles[0]
prompt = build_finereason_prompt(p)
assert "step by step" in prompt.lower()
print(f"  OK: {len(puzzles)} puzzles loaded")
print(f"  First puzzle keys: {list(p.keys())[:5]}")

# Test 4: ThinkBench
print("\n[4] ThinkBench...")
from research.reasoning_benchmarks import load_thinkbench, build_thinkbench_prompt, _evaluate_thinkbench
problems = load_thinkbench(max_problems=3)
assert len(problems) >= 1, "No ThinkBench problems loaded"
p = problems[0]
prompt = build_thinkbench_prompt(p)
assert "Solution" in prompt
print(f"  OK: {len(problems)} problems loaded")
print(f"  First problem keys: {list(p.keys())[:5]}")

# Test evaluation with a fake response
test_resp = "The answer is \\boxed{42}"
correct, detail = _evaluate_thinkbench(test_resp, {"answer": "42"})
assert correct, f"Should be correct: {detail}"
print(f"  OK: evaluation works ({detail})")

# Test 5: Suite
print("\n[5] Suite import...")
from research.reasoning_benchmarks import ReasoningBenchmarkSuite
print(f"  OK: ReasoningBenchmarkSuite imports")

print(f"\n{'='*60}")
print("All data loading tests passed!")
print(f"{'='*60}")
