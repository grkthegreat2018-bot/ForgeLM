"""Safety subsystem — hallucination detection via internal model representations.

R34-1: TriLens — logit-lens entropy trajectory (arXiv 2606.01033).
R34-2: PoP — depth-fused hidden-norm probing (arXiv 2608.27165).
"""
from .trilens import PoPDetector, TriLensDetector, TriLensPoPEnsemble

__all__ = ["TriLensDetector", "PoPDetector", "TriLensPoPEnsemble"]
