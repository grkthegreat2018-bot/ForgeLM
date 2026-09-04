"""RewardGuard — centralized reward hardening + declarative scoring.

Replaces per-domain scoring logic. A ScoringSpec (loaded from domain JSON)
declares:
  - components: list of {metric, weight, transform?} summed into raw score
  - penalties: list of {flag, ...} applied as subtractions
  - hardening: list of invariant flags applied to every score

All 19 scoring fixes documented in AGENTS.md are expressed as flag entries
here, not as custom Python in each domain. Adding a new hardening rule =
add a flag handler in this file + set the flag in domain JSON.

Flags available:
  no_nan              — non-finite score → -1e9
  beat_baseline       — score must beat spec.baseline or be clamped
  monotonic           — track per-domain running best; reject regressions
  no_trivial          — apply penalty when a "trivial_solution" condition holds
  diversity_penalty   — penalize top_k=1 / single-option configs
  stability_penalty   — penalize zero-warmup / extreme configs
  smoothing_penalty   — penalize label_smoothing above threshold
  gamma_penalty       — penalize focal gamma above threshold
  temp_penalty        — penalize temperature above threshold
  aux_penalty         — penalize MoE aux_loss outside [1e-10, 0.01]
  skip_penalty        — penalize too many MoD skip layers
  router_quality      — score MoD router type (mlp > linear)
  tie_factor_eval     — score factorized-embed tie_factor (tying saves params)
  inference_penalty   — penalize configs that disable KV cache (n_recomp/full recompute)
  latency_penalty     — penalize merge_window / depth_ratio above threshold
  checkpoint_compat   — penalize RoPE theta deviation from baseline
  frozen_dim_detect   — detect frozen dimensions at long range
  long_range_diversity— require rotation diversity at long positions
  quant_error_measure — simulate FP8 rounding with correct mantissa bits
  focus_ratio         — measure gradient concentration on wrong predictions
  acceptance_threshold— fix backwards acceptance threshold formula
  svd_optimal_combine — use SVD-based optimal combination for learned mode
  prefetch_log        — logarithmic prefetch_depth (diminishing returns)
  prefetch_mem_cost   — deeper prefetch uses more staging VRAM
  overlap_mem_cost    — overlap keeps parts of previous chunks (more memory)
  rep_penalty_benefit — mild repetition_penalty has upside
  length_penalty_sweet— length_penalty peaks at 1.0, penalize extremes
  beam_width1_penalty — beam_width=1 is greedy, not beam search
  update_freq_score   — score titan update_freq (was decoded but ignored)
  freshness_model     — titan freshness decay
  gate_interference   — high gate dominates main signal
  n_heads_score       — score n_heads (was decoded but ignored)
  head_overhead       — per-head param cost
  n_kv_heads_score    — score n_kv_heads (was decoded but ignored)
  tie_strength_score  — score tie_strength (was decoded but ignored)
  tying_savings       — tying saves params benefit
  temperature_score   — score temperature (was decoded but ignored)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# ScoringSpec — declarative scoring policy (loaded from JSON)
# ---------------------------------------------------------------------------

@dataclass
class ScoringComponent:
    """One additive component of the raw score: weight * transform(metric)."""
    metric: str                          # name of metric from simulator
    weight: float = 1.0
    transform: str = "identity"          # identity | log1p | neg | clamp_pos

    def apply(self, metrics: dict[str, float]) -> float:
        v = float(metrics.get(self.metric, 0.0))
        if self.transform == "log1p":
            v = math.log1p(max(v, 0.0))
        elif self.transform == "log2":
            v = math.log2(max(v, 1e-12))
        elif self.transform == "neg":
            v = -v
        elif self.transform == "clamp_pos":
            v = max(v, 0.0)
        return self.weight * v


@dataclass
class PenaltySpec:
    """A declarative penalty applied as a subtraction from the score."""
    flag: str                            # name of penalty handler
    # Common fields (used by various handlers)
    metric: str | None = None
    op: str = "<"                        # < | > | <= | >= | ==
    value: float = 0.0
    penalty: float = 0.0                 # flat penalty
    scale: float = 1.0                   # multiplier for linear penalties
    threshold: float = 0.0               # for threshold-based penalties
    # For diversity_penalty
    top_k_metric: str | None = None
    top_k_low: int = 1
    # For range penalties
    low: float = 0.0
    high: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "PenaltySpec":
        return cls(
            flag=d["flag"],
            metric=d.get("metric"),
            op=d.get("op", "<"),
            value=float(d.get("value", 0.0)),
            penalty=float(d.get("penalty", 0.0)),
            scale=float(d.get("scale", 1.0)),
            threshold=float(d.get("threshold", 0.0)),
            top_k_metric=d.get("top_k_metric"),
            top_k_low=int(d.get("top_k_low", 1)),
            low=float(d.get("low", 0.0)),
            high=float(d.get("high", 1.0)),
        )


@dataclass
class ScoringSpec:
    """Full declarative scoring policy."""
    components: list[ScoringComponent] = field(default_factory=list)
    penalties: list[PenaltySpec] = field(default_factory=list)
    hardening: list[str] = field(default_factory=list)
    baseline: float = -1e9               # for beat_baseline flag
    bias: float = 0.0                    # constant added to raw score
    trivial_metric: str | None = None    # for no_trivial
    trivial_op: str = "<"
    trivial_value: float = 0.0
    trivial_penalty: float = -50.0
    # For monotonic — per-domain running best is tracked in RewardGuard
    monotonic_epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, d: dict) -> "ScoringSpec":
        comps = [ScoringComponent(metric=c["metric"],
                                  weight=float(c.get("weight", 1.0)),
                                  transform=c.get("transform", "identity"))
                 for c in d.get("components", [])]
        pens = [PenaltySpec.from_dict(p) for p in d.get("penalties", [])]
        return cls(
            components=comps,
            penalties=pens,
            hardening=list(d.get("hardening", [])),
            baseline=float(d.get("baseline", -1e9)),
            bias=float(d.get("bias", 0.0)),
            trivial_metric=d.get("trivial_metric"),
            trivial_op=d.get("trivial_op", "<"),
            trivial_value=float(d.get("trivial_value", 0.0)),
            trivial_penalty=float(d.get("trivial_penalty", -50.0)),
            monotonic_epsilon=float(d.get("monotonic_epsilon", 1e-6)),
        )

    def to_dict(self) -> dict:
        return {
            "components": [{"metric": c.metric, "weight": c.weight,
                            "transform": c.transform} for c in self.components],
            "penalties": [{"flag": p.flag, "metric": p.metric, "op": p.op,
                           "value": p.value, "penalty": p.penalty,
                           "scale": p.scale, "threshold": p.threshold,
                           "top_k_metric": p.top_k_metric,
                           "top_k_low": p.top_k_low,
                           "low": p.low, "high": p.high} for p in self.penalties],
            "hardening": list(self.hardening),
            "baseline": self.baseline,
            "bias": self.bias,
            "trivial_metric": self.trivial_metric,
            "trivial_op": self.trivial_op,
            "trivial_value": self.trivial_value,
            "trivial_penalty": self.trivial_penalty,
            "monotonic_epsilon": self.monotonic_epsilon,
        }


# ---------------------------------------------------------------------------
# RewardGuard — applies scoring spec + hardening to raw metrics
# ---------------------------------------------------------------------------

class RewardGuard:
    """Centralized reward composer + hardening layer.

    Wraps every domain's evaluate(). Given raw metrics from the simulator
    and a ScoringSpec, produces the final {score, behavioral, metadata}.
    """

    # Per-domain running best for monotonic guard (keyed by id(spec))
    _running_best: dict[int, float] = {}

    def __init__(self, spec: ScoringSpec):
        self.spec = spec
        self._my_best = float("-inf")

    def score(self, config: dict[str, Any], metrics: dict[str, float]) -> dict:
        """Compose final score from raw metrics.

        Args:
            config: the decoded config dict
            metrics: raw metrics from the simulator (e.g. {"sqnr": 32.1, ...})

        Returns:
            {score, behavioral, metadata} — same shape as BaseDomain.evaluate
        """
        # 1. Compose raw score from components + bias
        raw = self.spec.bias
        for comp in self.spec.components:
            raw += comp.apply(metrics)

        # 2. Apply declarative penalties
        penalty_total = 0.0
        penalty_log = {}
        for pen in self.spec.penalties:
            p_val = self._apply_penalty(pen, config, metrics)
            if p_val != 0.0:
                penalty_total += p_val
                penalty_log[pen.flag] = p_val

        score = raw + penalty_total

        # 3. Apply hardening flags
        hardening_log = {}
        for flag in self.spec.hardening:
            score, flag_meta = self._apply_hardening(flag, score, config, metrics)
            if flag_meta:
                hardening_log[flag] = flag_meta

        # 4. Final NaN/inf guard (always on, even without flag)
        if not np.isfinite(score):
            score = -1e9
            hardening_log["non_finite"] = True

        # 5. Build behavioral + metadata
        # Behavioral: pull from metrics if declared, else use (score,)
        behavioral = self._extract_behavioral(metrics)
        metadata = dict(metrics)
        if penalty_log:
            metadata["penalties"] = penalty_log
        if hardening_log:
            metadata["hardening"] = hardening_log
        metadata["raw_score"] = raw

        return {
            "score": float(score),
            "behavioral": behavioral,
            "metadata": metadata,
        }

    # ----- penalty handlers -----

    def _apply_penalty(self, pen: PenaltySpec, config: dict, metrics: dict) -> float:
        """Dispatch a penalty by flag name. Returns penalty value (negative)."""
        handler = _PENALTY_HANDLERS.get(pen.flag)
        if handler is None:
            # Generic threshold penalty: if metric op value, apply penalty
            return _generic_threshold_penalty(pen, config, metrics)
        return handler(pen, config, metrics)

    # ----- hardening handlers -----

    def _apply_hardening(self, flag: str, score: float, config: dict,
                         metrics: dict) -> tuple[float, Any]:
        handler = _HARDENING_HANDLERS.get(flag)
        if handler is None:
            return score, None
        return handler(self, score, config, metrics)

    # ----- behavioral extraction -----

    def _extract_behavioral(self, metrics: dict) -> tuple:
        """Pull behavioral dims from metrics.

        By convention, the simulator puts behavioral metrics under keys
        "behavioral_0", "behavioral_1", ... OR under the metric names listed
        in the spec's behavioral_dims. We try the explicit names first.
        """
        behav = []
        for entry in self.spec_components_behavioral_names():
            if entry in metrics:
                behav.append(float(metrics[entry]))
        if not behav:
            # Fall back to behavioral_N keys
            i = 0
            while f"behavioral_{i}" in metrics:
                behav.append(float(metrics[f"behavioral_{i}"]))
                i += 1
        if not behav:
            behav = [float(metrics.get("score", 0.0))]
        return tuple(behav)

    def spec_components_behavioral_names(self) -> list[str]:
        """Names of metrics that should populate the behavioral vector.

        Derived from the ScoringSpec's behavioral_dims (set externally by
        DomainSpec — we look it up via a side channel).
        """
        # DomainSpec sets _behavioral_names on the guard at construction time
        return getattr(self, "_behavioral_names", [])


# ---------------------------------------------------------------------------
# Penalty handlers — each returns a negative number (subtracted from score)
# ---------------------------------------------------------------------------

def _cmp(op: str, a: float, b: float) -> bool:
    if op == "<": return a < b
    if op == ">": return a > b
    if op == "<=": return a <= b
    if op == ">=": return a >= b
    if op == "==": return abs(a - b) < 1e-9
    return False


def _generic_threshold_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Default: if metrics[pen.metric] op pen.value, apply pen.penalty."""
    if pen.metric is None:
        return 0.0
    # Try metrics first, then config
    v = metrics.get(pen.metric, config.get(pen.metric, 0.0))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if _cmp(pen.op, v, pen.value):
        return pen.penalty
    return 0.0


def _linear_range_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Linear penalty when metric is outside [low, high]."""
    if pen.metric is None:
        return 0.0
    v = float(metrics.get(pen.metric, config.get(pen.metric, 0.0)))
    if v < pen.low:
        return -(pen.low - v) * pen.scale
    if v > pen.high:
        return -(v - pen.high) * pen.scale
    return 0.0


def _diversity_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Penalize top_k=1 (trivial MoE routing — no ensemble)."""
    if pen.top_k_metric is None:
        return 0.0
    k = int(config.get(pen.top_k_metric, 0))
    if k == 1:
        return pen.penalty  # large negative
    if k == 2:
        return pen.penalty * 0.2  # mild
    return 0.0


def _stability_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Scheduler stability: penalize zero warmup, <1% warmup, >30% warmup."""
    ws = config.get("warmup_steps", 0)
    ds = config.get("decay_steps", 1)
    ratio = ws / max(ds, 1)
    if ws == 0:
        return -8.0
    if ratio < 0.01:
        return -4.0
    if ratio > 0.3:
        return -2.0
    return 0.0


def _smoothing_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Label smoothing >0.2 hurts SFT — linear penalty above threshold."""
    ls = float(config.get("label_smoothing", 0.0))
    if ls > 0.2:
        return -(ls - 0.2) * 30.0
    return 0.0


def _gamma_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Focal gamma >5 causes gradient vanishing — linear penalty."""
    fg = float(config.get("focal_gamma", 0.0))
    if fg > 5:
        return -(fg - 5) * 5.0
    return 0.0


def _temp_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Temperature >1.5 softens gradients too much."""
    t = float(config.get("temperature", 1.0))
    if t > 1.5:
        return -(t - 1.5) * 4.0
    return 0.0


def _aux_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """MoE aux_loss outside [1e-10, 0.01]."""
    aux = float(config.get("aux_loss_weight", config.get("aux_loss", 0.0)))
    if aux > 0.01:
        return -(aux - 0.01) * 50.0
    if aux < 1e-10:
        return -1.0
    return 0.0


def _skip_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """MoD: too many skip layers (>8) hurts capacity."""
    n_skip = int(config.get("n_skip_layers", config.get("n_skip", 0)))
    if n_skip > 8:
        return -0.5 * (n_skip - 8)
    return 0.0


def _router_quality(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """MoD router: mlp=1.0, linear=0.85 (relative quality). Returns adjustment."""
    rt = config.get("router_type", "mlp")
    return {"mlp": 0.0, "linear": -0.15}.get(rt, 0.0)


def _tie_factor_eval(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Factorized embed: tying saves params but adds up to 10% recon error."""
    if config.get("tie_factor", False):
        # Benefit: param savings (positive); cost: recon error (negative)
        return -0.1 * float(metrics.get("recon_err", 0.0))
    return 0.0


def _inference_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Penalize configs that disable KV cache (n_recomp=16 or ratio=1.0)."""
    n_recomp = int(config.get("n_recomp", 0))
    if n_recomp >= 16:
        return -40.0
    ratio = float(config.get("recompute_ratio", 0.0))
    if ratio >= 1.0:
        return -50.0
    return 0.0


def _latency_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Penalize merge_window >50ms or depth_ratio above threshold."""
    mw = float(config.get("merge_window", 0))
    if mw > 50:
        return -(mw - 50) * 0.1
    depth_ratio = float(metrics.get("depth_ratio", 0.0))
    if depth_ratio > pen.threshold:
        return -(depth_ratio - pen.threshold) * pen.scale
    return 0.0


def _checkpoint_compat(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """RoPE theta deviation from baseline (1M) — log-scaled."""
    theta = float(config.get("rope_theta", 1_000_000))
    baseline_theta = 1_000_000.0
    if theta <= 0:
        return -20.0
    ratio = max(theta, baseline_theta) / max(min(theta, baseline_theta), 1.0)
    if ratio > 10:
        return -math.log10(ratio) * 5.0
    return 0.0


def _frozen_dim_detect(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Detect frozen dimensions at long range (4096 positions)."""
    frozen = float(metrics.get("frozen_dim_count", 0))
    return -frozen * 2.0


def _long_range_diversity(pen: PenaltySpec, config: config, metrics: dict) -> float:
    """Require rotation diversity at long positions."""
    div = float(metrics.get("long_range_diversity", 1.0))
    if div < 0.3:
        return -(0.3 - div) * 10.0
    return 0.0


def _quant_error_measure(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Simulate FP8 rounding with correct mantissa bits (e4m3=3, e5m2=2)."""
    # Already computed by simulator if it sets 'fp8_quant_error'
    # Here we just ensure the metric is present; if not, penalize
    if "fp8_quant_error" not in metrics:
        return -5.0  # missing measurement
    return 0.0


def _focus_ratio(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Gradient concentration on wrong predictions."""
    fr = float(metrics.get("focus_ratio", 1.0))
    return min(fr, 3.0) * 3.0  # benefit, not penalty


def _acceptance_threshold_fix(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Marker flag — the simulator should compute acceptance correctly."""
    # No-op here; the fix is in the simulator. This flag exists so rescore
    # knows to re-evaluate when the simulator changes.
    return 0.0


def _svd_optimal_combine(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Marker flag — learned mode uses SVD-based optimal combination."""
    return 0.0


def _prefetch_log(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Prefetch depth: logarithmic benefit (diminishing returns)."""
    depth = int(config.get("prefetch_depth", 0))
    if depth <= 0:
        return 0.0
    # Replace any linear benefit with log benefit
    # (simulator may have already added linear benefit; we subtract the diff)
    linear_benefit = float(metrics.get("prefetch_linear_benefit", 0.0))
    log_benefit = math.log1p(depth)
    return log_benefit - linear_benefit


def _prefetch_mem_cost(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Deeper prefetch uses more staging VRAM."""
    depth = int(config.get("prefetch_depth", 0))
    return -depth * 0.05


def _overlap_mem_cost(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Overlap keeps parts of previous chunks — more memory."""
    overlap = int(config.get("overlap", 0))
    return -overlap * 0.1


def _rep_penalty_benefit(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Mild repetition_penalty has upside (reduces repetition)."""
    rp = float(config.get("repetition_penalty", 1.0))
    if 1.0 < rp <= 1.3:
        return 1.0  # mild benefit
    return 0.0


def _length_penalty_sweet(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Length penalty peaks at 1.0; penalize extremes."""
    lp = float(config.get("length_penalty", 1.0))
    return -abs(lp - 1.0) * 2.0


def _beam_width1_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """beam_width=1 is greedy, not beam search."""
    bw = int(config.get("beam_width", 1))
    if bw == 1:
        return -5.0
    return 0.0


def _update_freq_score(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Titan: score update_freq (was decoded but ignored)."""
    uf = int(config.get("update_freq", 1))
    return min(uf, 8) * 0.5  # benefit


def _freshness_model(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Titan freshness decay — penalize stale memory."""
    freshness = float(metrics.get("freshness", 1.0))
    return (freshness - 0.5) * 2.0  # benefit if fresh, penalty if stale


def _gate_interference(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """High gate dominates main signal."""
    gate = float(metrics.get("gate_norm", 0.0))
    main = float(metrics.get("main_norm", 1.0))
    if gate > main * 2:
        return -(gate / main - 2) * 2.0
    return 0.0


def _n_heads_score(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Score n_heads (was decoded but ignored)."""
    n = int(config.get("n_heads", 1))
    return min(n, 16) * 0.3


def _head_overhead(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Per-head param cost."""
    n = int(config.get("n_heads", 1))
    return -n * 0.1


def _n_kv_heads_score(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Score n_kv_heads (was decoded but ignored)."""
    n = int(config.get("n_kv_heads", 1))
    return min(n, 8) * 0.4


def _tie_strength_score(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Score tie_strength (was decoded but ignored)."""
    ts = float(config.get("tie_strength", 0.5))
    return ts * 2.0


def _tying_savings(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Tying saves params benefit."""
    if config.get("tie_kv", False):
        return 2.0
    return 0.0


def _temperature_score(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Score temperature (was decoded but ignored) + penalize extremes."""
    t = float(config.get("temperature", 1.0))
    if t < 0.5 or t > 2.0:
        return -4.0
    return 0.0


def _loss_weight_penalty(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """MTP loss_weight >0.5 hurts main task."""
    lw = float(config.get("loss_weight", 0.3))
    if lw > 0.5:
        return -(lw - 0.5) * 10.0
    return 0.0


def _early_lr_jump(pen: PenaltySpec, config: dict, metrics: dict) -> float:
    """Penalize sharp early lr jumps."""
    jump = float(metrics.get("early_lr_jump", 0.0))
    if jump > 0.1:
        return -jump * 2.0
    return 0.0


# Registry of penalty handlers
_PENALTY_HANDLERS: dict[str, Callable] = {
    "linear_range": _linear_range_penalty,
    "diversity_penalty": _diversity_penalty,
    "stability_penalty": _stability_penalty,
    "smoothing_penalty": _smoothing_penalty,
    "gamma_penalty": _gamma_penalty,
    "temp_penalty": _temp_penalty,
    "aux_penalty": _aux_penalty,
    "skip_penalty": _skip_penalty,
    "router_quality": _router_quality,
    "tie_factor_eval": _tie_factor_eval,
    "inference_penalty": _inference_penalty,
    "latency_penalty": _latency_penalty,
    "checkpoint_compat": _checkpoint_compat,
    "frozen_dim_detect": _frozen_dim_detect,
    "long_range_diversity": _long_range_diversity,
    "quant_error_measure": _quant_error_measure,
    "focus_ratio": _focus_ratio,
    "acceptance_threshold": _acceptance_threshold_fix,
    "svd_optimal_combine": _svd_optimal_combine,
    "prefetch_log": _prefetch_log,
    "prefetch_mem_cost": _prefetch_mem_cost,
    "overlap_mem_cost": _overlap_mem_cost,
    "rep_penalty_benefit": _rep_penalty_benefit,
    "length_penalty_sweet": _length_penalty_sweet,
    "beam_width1_penalty": _beam_width1_penalty,
    "update_freq_score": _update_freq_score,
    "freshness_model": _freshness_model,
    "gate_interference": _gate_interference,
    "n_heads_score": _n_heads_score,
    "head_overhead": _head_overhead,
    "n_kv_heads_score": _n_kv_heads_score,
    "tie_strength_score": _tie_strength_score,
    "tying_savings": _tying_savings,
    "temperature_score": _temperature_score,
    "loss_weight_penalty": _loss_weight_penalty,
    "early_lr_jump": _early_lr_jump,
}


# ---------------------------------------------------------------------------
# Hardening handlers — invariants applied to every score
# ---------------------------------------------------------------------------

def _h_no_nan(guard: RewardGuard, score: float, config: dict,
              metrics: dict) -> tuple[float, Any]:
    if not np.isfinite(score):
        return -1e9, {"non_finite": True}
    return score, None


def _h_beat_baseline(guard: RewardGuard, score: float, config: dict,
                     metrics: dict) -> tuple[float, Any]:
    if score < guard.spec.baseline:
        return guard.spec.baseline - 1.0, {"below_baseline": True}
    return score, None


def _h_monotonic(guard: RewardGuard, score: float, config: dict,
                 metrics: dict) -> tuple[float, Any]:
    """Track running best; reject scores that regress below it.

    Note: this does NOT clamp the score (that would hide regressions from
    the search). It just logs when a regression happens so the DB layer
    can refuse to save it as canonical.
    """
    if score > guard._my_best + guard.spec.monotonic_epsilon:
        guard._my_best = score
        return score, {"new_best": True}
    return score, {"regression": True, "best": guard._my_best}


def _h_no_trivial(guard: RewardGuard, score: float, config: dict,
                  metrics: dict) -> tuple[float, Any]:
    """Apply trivial-solution penalty."""
    if guard.spec.trivial_metric is None:
        return score, None
    v = metrics.get(guard.spec.trivial_metric, config.get(guard.spec.trivial_metric, 0.0))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return score, None
    if _cmp(guard.spec.trivial_op, v, guard.spec.trivial_value):
        return score + guard.spec.trivial_penalty, {"trivial": True}
    return score, None


_HARDENING_HANDLERS: dict[str, Callable] = {
    "no_nan": _h_no_nan,
    "beat_baseline": _h_beat_baseline,
    "monotonic": _h_monotonic,
    "no_trivial": _h_no_trivial,
}
