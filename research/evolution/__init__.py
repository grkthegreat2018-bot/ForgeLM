"""ForgeEvolve: evolutionary candidate discovery via tiny generators + surrogate filtering.

Architecture:
  Phase 1: GENERATE  — N tiny MLP generators produce candidate configs (CPU, ~1s)
  Phase 2: FILTER    — trained surrogate predicts scores, only top-K evaluated (CPU, ~1s)
  Phase 3: SCORE     — real evaluation of K candidates (GPU/domain-specific, ~minutes)
  Phase 4: TRAIN     — update generators (REINFORCE) + surrogate (online) + archive (CPU, ~10s)
  Phase 5: REPEAT

Key insight: 1000 candidates generated, only 50 evaluated → 20:1 compression.
Surrogate learns which configs tend to work, so its top-50 predictions improve over time.
"""
from .engine import ForgeEvolve, ForgeEvolveConfig
from .generators import BatchedGenerator, TemplateGenerator, GeneratorPopulation
from .surrogate import SurrogateModel
from .archive import MapElitesArchive
from .trainer import GeneratorTrainer
from .database import FindingsDB
from .checker_model import SharedCheckerModel, HeuristicChecker, get_checker, reset_checker
from .curriculum_finetuner import CurriculumFineTuner
from .llm_gen_model import LLMGenModel
from .gen_model_manager import GenModelManager

__all__ = [
    "ForgeEvolve", "ForgeEvolveConfig",
    "BatchedGenerator", "TemplateGenerator", "GeneratorPopulation",
    "SurrogateModel", "MapElitesArchive", "GeneratorTrainer",
    "FindingsDB",
    "SharedCheckerModel", "HeuristicChecker", "get_checker", "reset_checker",
    "CurriculumFineTuner",
    "LLMGenModel", "GenModelManager",
]
