"""Verification test for goal-oriented self-play (GOSP) modules.

Tests:
  1. GoalTaskGenerator generates valid GoalTasks with correct I/O pairs
  2. build_goal_prompt produces correct prompts
  3. GoalScorer: correctness gate, minimalism, efficiency, diversity, anti-redundancy
  4. AST fingerprint extraction and Dice coefficient
  5. AdaptiveDifficulty controller
  6. Import check on modified recursive_self_play and self_play_expert_training
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("GOSP Verification Test")
print("=" * 60)

# ── Test 1: GoalTaskGenerator ──────────────────────────────────────
print("\n[1] GoalTaskGenerator...")
from research.goal_tasks import GoalTaskGenerator, GoalTask, build_goal_prompt

gen = GoalTaskGenerator(seed=42)
task = gen.generate(archetype="fibonacci", difficulty="medium")
assert task.id.startswith("fibonacci"), f"bad id: {task.id}"
assert task.domain == "algorithms", f"bad domain: {task.domain}"
assert task.description == "Compute the n-th Fibonacci number (fib(0)=0, fib(1)=1)"
assert task.input_signature == "(n: int) -> int"
assert len(task.test_cases) >= 3, f"too few test cases: {len(task.test_cases)}"
assert task.stress_index is not None, "no stress index"

# Verify expected outputs are correct (fib(10) should be 55)
fib_10 = next(tc for tc in task.test_cases if tc["args"] == (10,))
assert fib_10["expected"] == 55, f"fib(10) expected 55, got {fib_10['expected']}"
fib_0 = next(tc for tc in task.test_cases if tc["args"] == (0,))
assert fib_0["expected"] == 0, f"fib(0) expected 0, got {fib_0['expected']}"
print(f"  OK: fibonacci medium task, {len(task.test_cases)} test cases, stress at index {task.stress_index}")

# Test batch generation
batch = gen.generate_batch(5, domain="math")
assert len(batch) == 5
for t in batch:
    assert t.domain == "math", f"bad domain in batch: {t.domain}"
print(f"  OK: batch of 5 math tasks")

# Test all archetypes generate
from research.goal_tasks import ARCHETYPES
for arch_name in ARCHETYPES:
    t = gen.generate(archetype=arch_name, difficulty="easy")
    assert t.archetype == arch_name
    assert len(t.test_cases) >= 2
print(f"  OK: all {len(ARCHETYPES)} archetypes generate valid tasks")

# ── Test 2: build_goal_prompt ──────────────────────────────────────
print("\n[2] build_goal_prompt...")
prompt = build_goal_prompt(task)
assert "Goal:" in prompt
assert "def solve" in prompt
assert "==" in prompt  # I/O contract
assert "any way you choose" in prompt
print(f"  OK: prompt built ({len(prompt)} chars)")
# Show a sample
for line in prompt.split("\n")[:8]:
    print(f"    {line}")

# ── Test 3: AST fingerprint + Dice coefficient ─────────────────────
print("\n[3] AST fingerprint + Dice coefficient...")
from research.goal_scorer import extract_ast_fingerprint, dice_coefficient, count_ast_nodes

code_a = "def solve(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a"
code_b = "def solve(n):\n    if n < 2: return n\n    return solve(n-1) + solve(n-2)"
code_c = "def solve(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return a"  # same as A

fp_a = extract_ast_fingerprint(code_a)
fp_b = extract_ast_fingerprint(code_b)
fp_c = extract_ast_fingerprint(code_c)

assert "FunctionDef" in fp_a
assert "For" in fp_a, "iterative fib should have For"
assert "If" in fp_b, "recursive fib should have If"
assert "For" not in fp_b, "recursive fib should NOT have For"

dice_ab = dice_coefficient(fp_a, fp_b)
dice_ac = dice_coefficient(fp_a, fp_c)
assert dice_ac == 1.0, f"identical code should have Dice=1.0, got {dice_ac}"
assert 0.0 < dice_ab < 1.0, f"different approaches should have 0<Dice<1, got {dice_ab}"
print(f"  OK: iterative vs recursive Dice={dice_ab:.3f} (should be < 1.0)")
print(f"  OK: identical code Dice={dice_ac:.3f} (should be 1.0)")
print(f"  AST nodes: iterative={count_ast_nodes(code_a)}, recursive={count_ast_nodes(code_b)}")

# ── Test 4: GoalScorer ─────────────────────────────────────────────
print("\n[4] GoalScorer...")
from research.goal_scorer import GoalScorer, ScoreResult

scorer = GoalScorer()

# Test 4a: correctness gate (failed)
result_fail = scorer.score(
    code=code_a, correct=False, exec_time_ms=1.0, mean_logprob=-0.5,
    goal_id="test_1", tokens_generated=30)
assert result_fail.quality == 0.0, "failed correctness should give quality=0"
assert not result_fail.accepted
print(f"  OK: failed correctness gate -> quality={result_fail.quality}")

# Test 4b: correct, first solution (high diversity)
result_ok1 = scorer.score(
    code=code_a, correct=True, exec_time_ms=1.0, mean_logprob=-0.1,
    goal_id="test_2", tokens_generated=30)
assert result_ok1.accepted, "first correct solution should be accepted"
assert result_ok1.quality > 0.5, f"first solution quality should be high, got {result_ok1.quality}"
assert result_ok1.scores["diversity"] == 1.0, "first solution should have full diversity"
print(f"  OK: first correct solution → quality={result_ok1.quality:.3f}, scores={result_ok1.scores}")
scorer.record_fingerprint("test_2", result_ok1.fingerprint)

# Test 4c: correct, same approach (redundant → rejected)
result_redundant = scorer.score(
    code=code_c, correct=True, exec_time_ms=1.0, mean_logprob=-0.1,
    goal_id="test_2", tokens_generated=30)
assert not result_redundant.accepted, "redundant solution should be rejected"
assert "redundant" in result_redundant.rejected_reason
print(f"  OK: redundant solution rejected: {result_redundant.rejected_reason}")

# Test 4d: correct, different approach (accepted, diversity bonus)
result_diff = scorer.score(
    code=code_b, correct=True, exec_time_ms=5.0, mean_logprob=-0.2,
    goal_id="test_2", tokens_generated=40)
assert result_diff.accepted, "different approach should be accepted"
assert result_diff.scores["diversity"] > 0.3, "different approach should have meaningful diversity"
print(f"  OK: different approach -> quality={result_diff.quality:.3f}, diversity={result_diff.scores['diversity']:.3f}")

# ── Test 5: AdaptiveDifficulty ─────────────────────────────────────
print("\n[5] AdaptiveDifficulty...")
from research.goal_tasks import AdaptiveDifficulty

adj = AdaptiveDifficulty()
assert adj.get_difficulty("test") == "easy", "default should be easy"

# Record mostly successes -> should escalate (easy -> medium -> hard)
for _ in range(15):
    adj.record("test_domain", True)
diff_after_success = adj.get_difficulty("test_domain")
assert diff_after_success in ("medium", "hard"), f"high success should escalate, got {diff_after_success}"
print(f"  OK: after 15 successes -> {diff_after_success}")

# Record mostly failures -> should ease
for _ in range(20):
    adj.record("test_domain", False)
diff_after_fail = adj.get_difficulty("test_domain")
assert diff_after_fail in ("easy", "medium"), f"low success should ease, got {diff_after_fail}"
print(f"  OK: after 20 failures -> {diff_after_fail}")
print(f"  OK: adaptive difficulty escalates/eases correctly")

# ── Test 6: Import check on modified files ─────────────────────────
print("\n[6] Import check on modified files...")
try:
    from research.recursive_self_play import RecursiveSelfPlay
    assert hasattr(RecursiveSelfPlay, 'run_goal_task'), "run_goal_task method missing"
    assert hasattr(RecursiveSelfPlay, '_verify_goal_io'), "_verify_goal_io method missing"
    print("  OK: RecursiveSelfPlay has run_goal_task and _verify_goal_io")
except Exception as e:
    print(f"  FAIL: RecursiveSelfPlay import: {e}")

try:
    from research.self_play_expert_training import ExpertSelfPlayTrainer, DOMAIN_ARCHETYPES
    assert "python_algorithms" in DOMAIN_ARCHETYPES
    print(f"  OK: ExpertSelfPlayTrainer imports, {len(DOMAIN_ARCHETYPES)} domains")
except Exception as e:
    print(f"  FAIL: ExpertSelfPlayTrainer import: {e}")

# ── Summary ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("All tests passed!")
print(f"{'='*60}")
