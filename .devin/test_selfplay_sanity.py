"""Quick import + sanity check for the updated self-play modules."""
import sys
sys.path.insert(0, r"D:\windsurf\ForgeAI")

from research.self_play.discovery.tool_use_loop import (
    ToolUseSelfPlay, TaskCurriculum, SelfPlayConfig,
    compute_reward, ToolUseReward,
)
from research.self_play.discovery.infinite_tool_loop import (
    InfiniteToolLoop, LoopConfig,
)

c = TaskCurriculum()
s = c.sample(5)
print(f"Tiers: {c.tiers}")
print(f"Tasks: {len(c.tasks['easy'])} easy, {len(c.tasks['medium'])} medium, "
      f"{len(c.tasks['hard'])} hard, {len(c.tasks['explore'])} explore")
print(f"Sample: {[(t, task[:40]) for t, task in s]}")

# Test reward computation with a fake trajectory
r = compute_reward(
    task="Calculate the factorial of 10.",
    tool_calls=[{
        "name": "run_script",
        "args": {"code": "import math; print(math.factorial(10))"},
        "result": {"stdout": "3628800", "stderr": "", "returncode": 0, "ok": True},
        "success": True,
    }],
    final_answer="The factorial of 10 is 3628800.",
    stopped_after_tools=True,
    stopped_after_answer=True,
)
print(f"\nReward test: {r.to_dict()}")
print(f"Total: {r.total:.3f}")
print("OK")
