"""Reasoning-time optimization modules.

Training-free inference-time techniques for efficient chain-of-thought:
  - ShortcutDecoder: Dual-signal convergence detection (ACL 2026, long.1330).
    Detects reasoning convergence via entropy + hidden-state probe to skip
    redundant CoT steps. 35% token reduction, maintains accuracy.
  - SyncThinkTerminator: Training-free reasoning saturation detection
    (OpenReview Hc9jAnIB3f). Monitors answer-token attention patterns to
    terminate reasoning when the model has converged. 62% accuracy with
    656 tokens vs 61.22% with 2141 tokens.
  - ReasoningBudgetController: Inference-time reasoning length control
    (Budget Guidance, ACL 2026 findings.1866). Predicts optimal thinking
    length via Gamma distribution predictor with soft token-level guidance.
"""
from .shortcut import (
    ShortcutDecoder, ShortcutConfig,
    SyncThinkTerminator, SyncThinkConfig,
    compute_step_entropy, detect_reasoning_saturation,
)
from .budget import ReasoningBudgetController, BudgetConfig

__all__ = [
    'ShortcutDecoder',
    'ShortcutConfig',
    'SyncThinkTerminator',
    'SyncThinkConfig',
    'compute_step_entropy',
    'detect_reasoning_saturation',
    'ReasoningBudgetController',
    'BudgetConfig',
]
