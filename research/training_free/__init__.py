"""Training-free alignment & adaptation — inference-only behavior control.

All techniques run strictly forward-pass: no gradients, no optimizer, no
parameter updates. VRAM stays at the inference footprint.

Modules:
  - urial:   URIAL in-context alignment (styling via system prompt + 3 examples)
  - reflexion: episodic memory of past attempts rendered into the prompt
  - steering: activation steering / task vectors (residual stream hooks)
  - rain:    RAIN rewindable autoregressive inference (self-eval + rewind)
  - solver:  TrainingFreeSolver — frozen-solver adapter combining the above
             for self-play loops (replaces GRPO weight updates)
"""
from research.training_free.expert_bake import bake_expert, decompress_expert
from research.training_free.rain import RAINGenerator
from research.training_free.reflexion import ReflexionBuffer
from research.training_free.solver import TrainingFreeSolver
from research.training_free.steering import ActivationSteerer
from research.training_free.urial import STYLE_EXAMPLES, SYSTEM_PROMPT, build_prompt

__all__ = [
    "RAINGenerator",
    "ReflexionBuffer",
    "TrainingFreeSolver",
    "ActivationSteerer",
    "bake_expert",
    "decompress_expert",
    "build_prompt",
    "SYSTEM_PROMPT",
    "STYLE_EXAMPLES",
]
