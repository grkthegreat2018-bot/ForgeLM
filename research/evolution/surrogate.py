"""Surrogate model: cheap predictor that pre-filters candidates.

Learns to predict evaluation scores from config parameters.
Trained online on all (config, score) pairs seen so far.
Used to filter 1000 candidates → top 50 for real evaluation.

Anti-overfitting techniques (from literature + evolution-tuned):
- Ensemble of 5 MLPs (was 3) with different seeds → average predictions
- 4-layer MLPs (was 3) with hidden=256 (was 128) for better score landscape fit
- LayerNorm in each MLP for training stability
- Disagreement between ensemble members = uncertainty signal
- Ranking-based loss (Spearman correlation) alongside MSE
- UCB-style acquisition: pred_mean + bonus * uncertainty
- Adaptive exploration: bonus decays over generations
- 5 training epochs per gen (was 3) for better convergence

CUDA: surrogate networks run on GPU when available.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def _make_mlp(input_dim: int, hidden_dim: int, seed: int = 0) -> nn.Sequential:
    """Create a 4-layer MLP surrogate with LayerNorm for stability."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.LayerNorm(hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
    )


class SurrogateModel:
    """Online-learning ensemble surrogate that predicts candidate scores.

    Uses an ensemble of N MLPs to reduce overfitting. Predictions are
    averaged across ensemble members, and disagreement (std) is used
    as an uncertainty signal for UCB-style exploration.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 256,
                 mode: str = "mlp", lr: float = 1e-3,
                 device: torch.device = None,
                 exploration_bonus: float = 2.0,
                 n_ensemble: int = 5):
        self.input_dim = input_dim
        self.mode = mode
        self.lr = lr
        self.device = device or torch.device("cpu")
        self.exploration_bonus = exploration_bonus
        self.x_history = None
        self.generation = 0
        self.n_ensemble = n_ensemble

        if mode == "mlp":
            # Ensemble of MLPs with different seeds
            self.nets = [_make_mlp(input_dim, hidden_dim, seed=i).to(self.device)
                         for i in range(n_ensemble)]
            self.optimizers = [torch.optim.Adam(net.parameters(), lr=lr)
                              for net in self.nets]
            self.loss_fn = nn.MSELoss()
            # Keep reference to first net for backward compat
            self.net = self.nets[0]

            # Note: torch.compile on surrogate nets causes FX tracing conflicts
            # with domain evaluation code. Skip compilation — the generator
            # forward pass is the hot path, not the surrogate.
        elif mode == "gp":
            self.x_history = None
            self.y_history = None
            self._length_scale = 1.0
            self._noise = 1e-3
        else:
            raise ValueError(f"Unknown surrogate mode: {mode}")

        self.n_trained = 0

    def predict(self, candidates: torch.Tensor) -> torch.Tensor:
        """Predict scores using ensemble + UCB-style acquisition.

        score = ensemble_mean + exploration_bonus * (disagreement + distance_uncertainty)

        Two uncertainty signals:
        1. Ensemble disagreement (std across MLPs) — model uncertainty
        2. Distance to nearest training point — epistemic uncertainty

        Bonus decays over generations (explore early, exploit late).

        Args:
            candidates: (N, input_dim) tensor in [0, 1]

        Returns:
            (N,) predicted scores (higher = better)
        """
        with torch.no_grad():
            if self.mode == "mlp":
                if self.n_trained < 10:
                    return torch.rand(candidates.shape[0], device=self.device)
                cands = candidates.to(self.device)

                # Ensemble predictions: (N_ensemble, N)
                preds = torch.stack([net(cands).squeeze(-1) for net in self.nets])
                pred_mean = preds.mean(dim=0)  # (N,)
                pred_std = preds.std(dim=0)    # (N,) — model uncertainty

                # Distance-based uncertainty (epistemic)
                dist_uncertainty = torch.zeros_like(pred_mean)
                if self.x_history is not None and len(self.x_history) > 0:
                    x_hist = torch.cat(self.x_history)
                    if len(x_hist) > 500:
                        idx = torch.randperm(len(x_hist))[:500]
                        x_hist = x_hist[idx]
                    dist = torch.cdist(cands, x_hist)
                    min_dist = dist.min(dim=-1).values
                    max_dist = (self.input_dim ** 0.5) * 0.5
                    dist_uncertainty = (min_dist / max_dist).clamp(0, 1)

                # Combined uncertainty: ensemble disagreement + distance
                # Normalize std to [0, 1] range
                std_norm = (pred_std / (pred_std.max() + 1e-8)).clamp(0, 1)
                total_uncertainty = 0.5 * std_norm + 0.5 * dist_uncertainty

                # Adaptive bonus: decays from 1.0 to 0.3 over 100 gens
                decay = max(0.3, 1.0 - self.generation / 100.0)
                pred = pred_mean + self.exploration_bonus * decay * total_uncertainty

                return pred
            elif self.mode == "gp":
                if self.x_history is None or len(self.x_history) < 5:
                    return torch.rand(candidates.shape[0], device=self.device)
                return self._gp_predict(candidates)

    def train(self, candidates: torch.Tensor, scores: torch.Tensor,
              epochs: int = 5):
        """Update ensemble surrogate on new (config, score) pairs.

        Each MLP trained independently with MSE + ranking loss.
        Limited epochs to prevent overfitting.
        """
        if self.mode == "mlp":
            cands = candidates.to(self.device)
            scs = scores.to(self.device)

            # Skip training if all scores are identical — MSE loss has no
            # gradient (loss tensor won't require grad), and training is
            # pointless since there's nothing to discriminate.
            if scs.numel() > 1 and torch.allclose(scs, scs[0]):
                self.n_trained += len(scs)
                if self.x_history is None:
                    self.x_history = []
                self.x_history.append(cands.clone())
                return

            # Sort scores for ranking loss computation
            sorted_idx = scs.argsort()
            ranks = torch.zeros_like(scs)
            ranks[sorted_idx] = torch.arange(len(scs), dtype=scs.dtype,
                                             device=self.device)
            ranks = ranks / max(len(scs) - 1, 1)  # normalize to [0, 1]

            for net, opt in zip(self.nets, self.optimizers):
                net.train()
                for _ in range(epochs):
                    pred = net(cands).squeeze(-1)
                    # MSE loss (primary)
                    mse_loss = self.loss_fn(pred, scs)

                    # Ranking loss: encourage correct ordering (Spearman proxy)
                    pred_sorted = pred.argsort()
                    rank_corr = (pred_sorted.float() - sorted_idx.float()).abs().mean()
                    rank_loss = rank_corr / max(len(scs), 1)

                    # Combined: 80% MSE + 20% ranking
                    loss = 0.8 * mse_loss + 0.2 * rank_loss
                    # Guard: if loss doesn't require grad (e.g. degenerate
                    # input), skip backward to avoid "element 0 of tensors
                    # does not require grad" error.
                    if not loss.requires_grad:
                        continue
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            self.n_trained += len(scs)

            # Track training data for distance uncertainty (list of tensors,
            # concatenated only when needed in predict() to avoid O(n²) reallocation)
            if self.x_history is None:
                self.x_history = []
            self.x_history.append(cands.clone())
            # Cap: remove oldest batches if total data points exceed 2000
            _total = sum(t.shape[0] for t in self.x_history)
            while _total > 2000 and len(self.x_history) > 1:
                _removed = self.x_history.pop(0)
                _total -= _removed.shape[0]

        elif self.mode == "gp":
            cands = candidates.to(self.device)
            scs = scores.to(self.device)
            if self.x_history is None:
                self.x_history = cands.clone()
                self.y_history = scs.clone()
            else:
                self.x_history = torch.cat([self.x_history, cands])
                self.y_history = torch.cat([self.y_history, scs])
            if len(self.x_history) > 2000:
                self.x_history = self.x_history[-2000:]
                self.y_history = self.y_history[-2000:]
            self.n_trained = len(self.y_history)

    def _gp_predict(self, candidates: torch.Tensor) -> torch.Tensor:
        """Simple GP regression with RBF kernel."""
        x = self.x_history
        y = self.y_history
        ls = self._length_scale
        noise = self._noise
        cands = candidates.to(self.device)

        k_s = torch.exp(-torch.cdist(cands, x, p=2) ** 2 / (2 * ls ** 2))
        k_xx = torch.exp(-torch.cdist(x, x, p=2) ** 2 / (2 * ls ** 2))
        k_xx_reg = k_xx + noise * torch.eye(len(x), device=self.device)

        try:
            alpha = torch.linalg.solve(k_xx_reg, y)
            mean = k_s @ alpha
            return mean
        except torch._C._LinAlgError:
            return torch.zeros(cands.shape[0], device=self.device)

    def filter_top_k(self, candidates: torch.Tensor, k: int) -> torch.Tensor:
        """Return indices of top-k candidates by predicted score."""
        scores = self.predict(candidates)
        return torch.topk(scores, k=min(k, len(scores))).indices
