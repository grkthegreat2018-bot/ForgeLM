"""Quick import + sanity check for the unified AZR self-play loop."""
import sys
sys.path.insert(0, r"D:\windsurf\ForgeAI")

from research.self_play.infinite_loop import (
    InfiniteSelfPlayLoop, LoopConfig,
)
from research.self_play.infinite_curriculum import (
    InfiniteCurriculum, ProposedTask, CurriculumStats,
)

# Test config creation
c = LoopConfig()
print(f"Config: tasks={c.tasks_per_epoch}, ft_steps={c.ft_max_steps}, "
      f"epochs={c.max_epochs}, lora={c.ft_lora}")
print(f"Domains: {c.domains}")

# Test loop construction (no model loaded — just config)
loop = InfiniteSelfPlayLoop(
    checkpoint="research/checkpoints/ForgeLM_V3_SFT.safetensors",
    config=c,
)
print(f"Loop: epoch={loop.epoch}, best={loop.best_checkpoint}")

# Test curriculum stats
stats = CurriculumStats()
print(f"Stats: proposed={stats.total_proposed}, validated={stats.total_validated}, "
      f"solved={stats.total_solved}")
print(f"Diversity: {stats.diversity_score:.2f}")
print(f"Proposer reward: {stats.mean_proposer_reward:.2f}")

print("OK")
