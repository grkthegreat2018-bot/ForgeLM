"""Quick test for LiveCodeBench data loading."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.livecodebench_eval import load_livecodebench

problems = load_livecodebench(start_date='2024-09-01', max_problems=3)
print(f"\nLoaded {len(problems)} problems:")
for p in problems[:3]:
    title = p.get('question_title', '?')[:50]
    diff = p.get('difficulty', '?')
    date = str(p.get('contest_date', ''))[:10]
    print(f"  {title} | {diff} | {date}")

# Check test case format
if problems:
    p = problems[0]
    print(f"\nFirst problem keys: {list(p.keys())}")
    raw_tc = p.get('public_test_cases', '')
    print(f"public_test_cases type: {type(raw_tc).__name__}")
    print(f"public_test_cases (first 200 chars): {str(raw_tc)[:200]}")
