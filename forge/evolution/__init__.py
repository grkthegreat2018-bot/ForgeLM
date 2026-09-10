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
from .archive import MapElitesArchive
from .checker_model import HeuristicChecker, SharedCheckerModel, get_checker, reset_checker
from .curriculum_finetuner import CurriculumFineTuner
from .database import FindingsDB
from .engine import ForgeEvolve, ForgeEvolveConfig
from .gen_model_manager import GenModelManager
from .generators import BatchedGenerator, GeneratorPopulation, TemplateGenerator
from .llm_gen_model import LLMGenModel
from .surrogate import SurrogateModel
from .trainer import GeneratorTrainer

__all__ = [
    "BatchedGenerator",
    "CurriculumFineTuner",
    "FindingsDB",
    "ForgeEvolve",
    "ForgeEvolveConfig",
    "GenModelManager",
    "GeneratorPopulation",
    "GeneratorTrainer",
    "HeuristicChecker",
    "LLMGenModel",
    "MapElitesArchive",
    "SharedCheckerModel",
    "SurrogateModel",
    "TemplateGenerator",
    "get_checker",
    "reset_checker",
]
