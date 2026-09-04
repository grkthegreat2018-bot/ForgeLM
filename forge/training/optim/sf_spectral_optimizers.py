"""Schedule-Free spectral optimizers: SF-NorMuon, AMUSE, MONA.

Based on three 2026 papers:
  1. SF-NorMuon (arXiv 2605.23061): schedule-free spectral optimizer with
     per-neuron normalization. Matches tuned AdamW across 1-8× Chinchilla horizons.
  2. AMUSE (arXiv 2605.22432): Anytime Muon + Stable gradient Evaluation.
     Integrates Muon's rapid bulk progress with Schedule-Free averaging.
     Time-varying interpolation coefficient: fast Muon → stable averaged.
  3. MONA (arXiv 2605.26842): Muon + Nesterov Acceleration.
     Adds acceleration term from EMA of gradient differences.
     Escapes sharp minima while preserving spectral-norm regularization.
     SOTA on MoE pretraining 1B-68B.

All three eliminate learning-rate schedules (anytime training):
  - No warmup-stable-decay needed
  - Checkpoint quality is high at ANY point during training
  - 31% faster to target loss at 1000 TPP (ScheduleFree+)

For our 1.2B model:
  - Current: AdamW with cosine schedule (path-dependent, re-tuning needed)
  - SF-NorMuon: no schedule, matches tuned AdamW, anytime checkpoints
  - AMUSE: Muon speed + Schedule-Free stability, no schedule
  - MONA: Muon + Nesterov, escapes sharp minima, SOTA on MoE
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


def _newton_schulz_5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration for matrix orthogonalization (Muon core).

    Computes G @ (G^T G)^(-1/2) ≈ orthogonalized G.
    5 steps is sufficient for good convergence.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Normalize so spectral norm ≤ 1
    X = X / (X.norm() + 1e-7)
    # Transpose if tall (more rows than cols) for efficiency
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.bfloat16()


def _per_neuron_normalize(G: torch.Tensor) -> torch.Tensor:
    """Per-neuron (per-row) normalization (NorMuon).

    Normalizes each row of the gradient to unit norm before orthogonalization.
    This improves conditioning and convergence.
    """
    if G.ndim != 2:
        return G
    norms = G.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return G / norms


class SFNorMuon(Optimizer):
    """SF-NorMuon: Schedule-Free NorMuon optimizer.

    Combines:
      - Muon: matrix orthogonalization for hidden params
      - NorMuon: per-neuron normalization
      - Schedule-Free: iterate averaging (no LR schedule)
      - Weight decay at fast iterate Z (not Y)

    Three iterates:
      - X: slow averaged iterate (output/eval)
      - Y: fast iterate (gradient evaluation point)
      - Z: fastest iterate (Muon update target)

    Update: Z = Muon_update(Y, grad)
            Y = (1-c)*Y + c*Z  (interpolation)
            X = (1-β)*X + β*Y  (averaging)

    Args:
        params: model parameters
        lr: learning rate (can be fixed — no schedule needed)
        momentum: Muon momentum
        beta: averaging decay for X
        weight_decay: applied to Z (fast iterate)
        ns_steps: Newton-Schulz iterations
    """

    def __init__(self, params: Iterable, lr=0.02, momentum=0.95,
                 beta=0.999, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, beta=beta,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta = group['beta']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['X'] = p.clone()  # slow averaged
                    state['Y'] = p.clone()  # fast eval point
                    state['Z'] = p.clone()  # fastest update target
                    state['step'] = 0

                state['step'] += 1
                mom = state['momentum']
                X, Y, Z = state['X'], state['Y'], state['Z']

                # Muon update (for 2D matrix params)
                if grad.ndim == 2 and grad.shape[0] > 1 and grad.shape[1] > 1:
                    # Per-neuron normalize (NorMuon)
                    grad_norm = _per_neuron_normalize(grad)
                    # Momentum
                    mom.mul_(momentum).add_(grad_norm, alpha=1 - momentum)
                    # Orthogonalize
                    update = _newton_schulz_5(mom, ns_steps)
                    update *= max(1, grad.shape[0] / grad.shape[1]) ** 0.5
                else:
                    # AdamW for non-matrix params (vectors, scalars)
                    mom.mul_(momentum).add_(grad, alpha=1 - momentum)
                    update = mom

                # Weight decay at Z (fast iterate)
                if wd > 0:
                    Z.mul_(1 - lr * wd)

                # Update Z
                Z.add_(update, alpha=-lr)

                # Interpolation coefficient (time-varying)
                c = 1.0 / max(state['step'], 1) ** 0.5  # decreases over time
                # Update Y: interpolate toward Z
                Y.mul_(1 - c).add_(Z, alpha=c)

                # Update X: averaging
                X.mul_(beta).add_(Y, alpha=1 - beta)

                # Set parameter to Y (eval point)
                p.copy_(Y)

        return loss

    def get_eval_params(self):
        """Return the averaged (X) parameters for evaluation."""
        eval_params = {}
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    eval_params[id(p)] = self.state[p]['X'].clone()
        return eval_params


class AMUSE(Optimizer):
    """AMUSE: Anytime Muon with Stable Gradient Evaluation.

    Integrates Muon's rapid bulk progress with Schedule-Free averaging.
    Uses a time-varying interpolation coefficient:
      - Early: evaluate gradients near fast Muon sequence (rapid adaptation)
      - Late: shift toward stable averaged sequence (suppress oscillations)

    This prevents the valley-wall oscillations that plague pure Muon.

    Args:
        params: model parameters
        lr: learning rate
        momentum: Muon momentum
        beta: averaging decay
        interpolation_schedule: 'sqrt' (default) or 'linear'
        ns_steps: Newton-Schulz iterations
    """

    def __init__(self, params: Iterable, lr=0.02, momentum=0.95,
                 beta=0.999, interpolation_schedule='sqrt', ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, beta=beta,
                        interpolation_schedule=interpolation_schedule,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            beta = group['beta']
            sched = group['interpolation_schedule']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['X'] = p.clone()  # averaged (stable)
                    state['Z'] = p.clone()  # Muon fast sequence
                    state['step'] = 0

                state['step'] += 1
                t = state['step']
                mom = state['momentum']
                X, Z = state['X'], state['Z']

                # Muon update
                if grad.ndim == 2 and grad.shape[0] > 1 and grad.shape[1] > 1:
                    mom.mul_(momentum).add_(grad, alpha=1 - momentum)
                    update = _newton_schulz_5(mom, ns_steps)
                    update *= max(1, grad.shape[0] / grad.shape[1]) ** 0.5
                else:
                    mom.mul_(momentum).add_(grad, alpha=1 - momentum)
                    update = mom

                # Update Z (fast Muon sequence)
                Z.add_(update, alpha=-lr)

                # Time-varying interpolation coefficient
                # Early: c ≈ 1 (evaluate at Z, rapid adaptation)
                # Late: c ≈ 0 (evaluate at X, stable)
                if sched == 'sqrt':
                    c = 1.0 / math.sqrt(t)
                else:  # linear
                    c = max(0, 1.0 - t / 1000)

                # Evaluation point: interpolate between X (stable) and Z (fast)
                eval_point = X * (1 - c) + Z * c

                # Update X (averaging)
                X.mul_(beta).add_(Z, alpha=1 - beta)

                # Set parameter to evaluation point
                p.copy_(eval_point)

        return loss


class MONA(Optimizer):
    """MONA: Muon Optimizer with Nesterov Acceleration.

    Adds Nesterov acceleration to Muon:
      - Computes EMA of gradient differences (acceleration term)
      - Adds acceleration to Muon update
      - Escapes sharp minima while preserving spectral-norm regularization

    SOTA on MoE pretraining (1B-68B params, 1T tokens).

    Args:
        params: model parameters
        lr: learning rate
        momentum: Muon momentum
        accel_decay: EMA decay for acceleration term
        accel_weight: weight for acceleration term
        ns_steps: Newton-Schulz iterations
    """

    def __init__(self, params: Iterable, lr=0.02, momentum=0.95,
                 accel_decay=0.9, accel_weight=0.1, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, accel_decay=accel_decay,
                        accel_weight=accel_weight, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            accel_decay = group['accel_decay']
            accel_weight = group['accel_weight']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)
                    state['accel'] = torch.zeros_like(p)  # EMA of grad differences
                    state['step'] = 0

                state['step'] += 1
                mom = state['momentum']
                prev_grad = state['prev_grad']
                accel = state['accel']

                # Compute gradient difference (acceleration signal)
                grad_diff = grad - prev_grad

                # Update acceleration EMA
                accel.mul_(accel_decay).add_(grad_diff, alpha=1 - accel_decay)

                # Save current grad for next step
                prev_grad.copy_(grad)

                # Muon update + acceleration
                combined = grad + accel_weight * accel

                if combined.ndim == 2 and combined.shape[0] > 1 and combined.shape[1] > 1:
                    mom.mul_(momentum).add_(combined, alpha=1 - momentum)
                    update = _newton_schulz_5(mom, ns_steps)
                    update *= max(1, combined.shape[0] / combined.shape[1]) ** 0.5
                else:
                    mom.mul_(momentum).add_(combined, alpha=1 - momentum)
                    update = mom

                p.add_(update, alpha=-lr)

        return loss
