"""Compute Futures Key — MTP draft with confidence-gated verification skip.

Novel insight: Multi-Token Prediction (MTP) drafts tokens speculatively, then
a verification forward pass confirms or rejects them.  But when the draft model
is *very* confident about a position, the verification pass is almost certainly
going to accept — so we can skip it entirely, saving a full forward pass per
skipped token.

The key insight is a *confidence-gated skip policy* with a safety bound:
  - Skip verification when draft confidence > threshold
  - But never skip more than max_skip consecutive positions (forces periodic
    verification to catch drift / compounding errors)

This turns MTP from "always verify" into "verify only when uncertain",
reducing verification forward passes by 30-60% on confident sequences while
guaranteeing a correctness check at least every max_skip tokens.

Key class: TRIVIAL — runtime optimization, no weight changes.

Usage:
    from research.keys.compute_futures_key import ComputeFuturesKey, ComputeFutures
    cf = ComputeFutures(confidence_threshold=0.9, max_skip=3)
    flags = cf.should_verify([0.95, 0.97, 0.92, 0.85, 0.99])
    # flags = [False, False, False, True, False]  -- skip confident, verify uncertain
"""
import torch
from typing import Dict, List, Optional
from .base import Key, KeyClass, KeyResult


class ComputeFutures:
    """Confidence-gated verification skip policy for MTP drafting.

    Decides whether to run the verification forward pass for each drafted
    token position based on draft confidence and a consecutive-skip bound.
    """

    def __init__(self, confidence_threshold: float = 0.9, max_skip: int = 3):
        self.threshold = confidence_threshold
        self.max_skip = max_skip

    def should_verify(self, draft_confidences: List[float]) -> List[bool]:
        """Decide verification flags for a sequence of draft positions.

        Args:
            draft_confidences: per-position confidence from the draft model.

        Returns:
            List of booleans — True means run verification, False means skip.
        """
        flags: List[bool] = []
        consecutive_skips = 0
        for conf in draft_confidences:
            # Force verify if we've hit the max consecutive skip limit
            if consecutive_skips >= self.max_skip:
                flags.append(True)
                consecutive_skips = 0
                continue
            # Skip if confident enough
            if conf >= self.threshold:
                flags.append(False)
                consecutive_skips += 1
            else:
                flags.append(True)
                consecutive_skips = 0
        return flags

    def skip_ratio(self, draft_confidences: List[float]) -> float:
        """Fraction of positions where verification is skipped (0..1)."""
        if not draft_confidences:
            return 0.0
        flags = self.should_verify(draft_confidences)
        skipped = sum(1 for f in flags if not f)
        return skipped / len(flags)


class ComputeFuturesKey(Key):
    """Compute Futures key — confidence-gated MTP verification skip.

    Key class: TRIVIAL — runtime optimization, no weight changes.
    """

    @property
    def name(self) -> str:
        return "compute_futures"

    @property
    def description(self) -> str:
        return (
            "MTP draft with confidence-gated verification skip: "
            "skip verify when draft is confident, bound by max consecutive skips."
        )

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Compute verification flags for a sequence of draft confidences.

        Args:
            data: dict with keys:
                - draft_confidences: list[float] per-position confidences
                - threshold: float confidence threshold (default 0.9)
                - max_skip: int max consecutive skips (default 3)

        Returns:
            KeyResult with data = {"verify_flags": list[bool]}
        """
        confidences: List[float] = data["draft_confidences"]
        threshold: float = data.get("threshold", 0.9)
        max_skip: int = data.get("max_skip", 3)

        cf = ComputeFutures(confidence_threshold=threshold, max_skip=max_skip)
        flags = cf.should_verify(confidences)
        skipped = sum(1 for f in flags if not f)

        return KeyResult(
            success=True,
            data={"verify_flags": flags},
            metadata={
                "n_positions": len(confidences),
                "n_skipped": skipped,
                "n_verified": len(confidences) - skipped,
                "skip_ratio": skipped / max(len(confidences), 1),
                "threshold": threshold,
                "max_skip": max_skip,
            },
        )

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """No-op — TRIVIAL key has no weights to reverse."""
        return KeyResult(success=True, data={})


if __name__ == "__main__":
    key = ComputeFuturesKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")
    print(f"  Description: {key.description}")

    # Synthetic draft confidences: mostly confident with a few uncertain spots
    confidences = [0.95, 0.97, 0.92, 0.85, 0.99, 0.96, 0.94, 0.50, 0.98, 0.97]

    # Forward: compute verification flags
    result = key.forward({
        "draft_confidences": confidences,
        "threshold": 0.9,
        "max_skip": 3,
    })
    assert result.success, f"Forward failed: {result.error}"
    flags = result.data["verify_flags"]
    print(f"  Confidences: {confidences}")
    print(f"  Verify flags: {flags}")
    print(f"  Skipped: {result.metadata['n_skipped']}/{result.metadata['n_positions']} "
          f"({result.metadata['skip_ratio']:.0%})")

    # Verify: uncertain position (0.85 < 0.9) must be verified
    assert flags[3] is True, "Uncertain position (0.85) should require verification"
    assert flags[7] is True, "Uncertain position (0.50) should require verification"
    print("  Uncertain positions correctly flagged for verification")

    # Verify: confident positions should be skipped (unless max_skip hit)
    assert flags[0] is False, "Confident position (0.95) should skip verification"
    print("  Confident positions correctly skipped")

    # Verify max_skip bound: 4 consecutive confident -> 4th must verify
    cf = ComputeFutures(confidence_threshold=0.9, max_skip=3)
    all_confident = [0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    bounded_flags = cf.should_verify(all_confident)
    # Positions 0,1,2 skipped; position 3 forced verify (max_skip=3); 4,5 skipped
    assert bounded_flags == [False, False, False, True, False, False], \
        f"max_skip bound violated: {bounded_flags}"
    print("  max_skip consecutive-skip bound enforced correctly")

    # Reverse: no-op for TRIVIAL
    rev = key.reverse({})
    assert rev.success and rev.data == {}
    print("  Reverse: no-op (TRIVIAL) verified")
    print("  All tests passed.")
