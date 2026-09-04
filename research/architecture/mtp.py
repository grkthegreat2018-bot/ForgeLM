"""Re-export MTPModule from forge.decoding.mtp for model_loader compat.

model_loader.py imports ``from research.architecture.mtp import MTPModule``.
The actual implementation lives in forge/decoding/mtp.py; this thin
shim avoids duplicating code while keeping the import path stable.
"""
from forge.decoding.mtp import MTPHead, MTPModule, MTPTrainer

__all__ = ["MTPHead", "MTPModule", "MTPTrainer"]
