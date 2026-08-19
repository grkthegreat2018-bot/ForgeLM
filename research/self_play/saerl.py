"""SAERL: SAE-guided data engineering for RL training.

Based on "Guiding LLM Post-training Data Engineering with Model Internals
from Sparse Autoencoders" (arXiv 2605.27354).

Key insight: model internals (extracted via Sparse Autoencoders) encode
rich information about how the model processes training data. SAERL uses
SAE features to guide three data engineering operations:

  1. Diversity control: SAE-space clustering with moderate batch mixing
  2. Difficulty ordering: SAE features as difficulty proxy for curriculum
  3. Quality filtering: SAE-based quality probe for data filtering

Results: +3.00% average accuracy over vanilla GRPO, 20% fewer training steps
to reach target accuracy. SAE transfers across model families and scales.

For our self-play + GRPO training:
  - Current: random batching, no difficulty ordering, no quality filtering
  - SAERL: SAE-guided batch composition, curriculum ordering, quality filtering
  - Especially valuable for long self-play runs (better data efficiency)

This implementation provides:
  1. SAERLFeatureExtractor: extracts SAE features from model internals
  2. SAERLDiversityController: clusters in SAE space for batch diversity
  3. SAERLDifficultyEstimator: SAE-based difficulty proxy
  4. SAERLQualityFilter: SAE-based quality probe
  5. SAERLDataEngineer: combines all three for RL data engineering
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAERLFeatureExtractor:
    """Extracts SAE features from model hidden states.

    Uses a simple autoencoder to learn sparse representations of the
    model's internal activations. These features capture semantic properties
    useful for data engineering.
    """

    def __init__(self, d_model: int, n_features: int = 256,
                 device: str = "cuda"):
        self.d_model = d_model
        self.n_features = n_features
        self.device = device

        # Simple SAE: encoder + decoder
        self.encoder = nn.Linear(d_model, n_features, bias=False).to(device)
        self.decoder = nn.Linear(n_features, d_model, bias=False).to(device)
        self._trained = False

    def extract(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract SAE features from hidden states.

        Args:
            hidden_states: (B, T, d_model) or (N, d_model)

        Returns:
            features: (B, T, n_features) sparse features (ReLU activated)
        """
        if hidden_states.dim() > 2:
            B, T, D = hidden_states.shape
            flat = hidden_states.reshape(-1, D)
        else:
            flat = hidden_states

        features = F.relu(self.encoder(flat))  # ReLU = sparse

        if hidden_states.dim() > 2:
            return features.reshape(B, T, -1)
        return features

    def train(self, hidden_states: torch.Tensor, n_steps: int = 1000,
              lr: float = 1e-3, sparsity_weight: float = 0.01):
        """Train the SAE on collected hidden states."""
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr)

        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        batch_size = min(256, flat.shape[0])

        for step in range(n_steps):
            idx = torch.randint(0, flat.shape[0], (batch_size,))
            batch = flat[idx]

            features = F.relu(self.encoder(batch))
            reconstructed = self.decoder(features)

            # Loss: reconstruction + sparsity
            recon_loss = F.mse_loss(reconstructed, batch)
            sparsity_loss = features.abs().mean()
            loss = recon_loss + sparsity_weight * sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._trained = True
        print(f"  [SAERL] SAE trained: {self.n_features} features, "
              f"final loss={loss.item():.4f}")

    def save(self, path: str):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'd_model': self.d_model,
            'n_features': self.n_features,
        }, path)

    def load(self, path: str):
        state = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        self._trained = True


class SAERLDiversityController:
    """Controls batch diversity using SAE-space clustering.

    Clusters training samples in SAE feature space and ensures each batch
    contains samples from different clusters (moderate mixing).
    """

    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self._cluster_centers: Optional[torch.Tensor] = None
        self._assignments: dict[int, int] = {}  # sample_id → cluster

    def fit(self, features: torch.Tensor):
        """Cluster samples in SAE feature space (k-means)."""
        # features: (N, n_features)
        N = features.shape[0]
        k = min(self.n_clusters, N)

        # K-means initialization: random points
        idx = torch.randperm(N)[:k]
        self._cluster_centers = features[idx].clone()

        for _ in range(20):  # k-means iterations
            # Assign
            dists = torch.cdist(features, self._cluster_centers)
            assignments = dists.argmin(dim=1)

            # Update centers
            for c in range(k):
                mask = assignments == c
                if mask.any():
                    self._cluster_centers[c] = features[mask].mean(dim=0)

        # Store assignments
        for i in range(N):
            self._assignments[i] = assignments[i].item()

    def compose_batch(self, sample_ids: list[int],
                      batch_size: int) -> list[int]:
        """Compose a diverse batch from different clusters."""
        # Group samples by cluster
        cluster_samples: dict[int, list[int]] = {}
        for sid in sample_ids:
            c = self._assignments.get(sid, 0)
            cluster_samples.setdefault(c, []).append(sid)

        # Sample from each cluster (round-robin)
        batch = []
        clusters = list(cluster_samples.keys())
        random.shuffle(clusters)

        while len(batch) < batch_size:
            for c in clusters:
                if len(batch) >= batch_size:
                    break
                if cluster_samples[c]:
                    batch.append(cluster_samples[c].pop(random.randint(0, len(cluster_samples[c]) - 1)))

        return batch[:batch_size]


class SAERLDifficultyEstimator:
    """Estimates sample difficulty using SAE features.

    Difficulty proxy: samples with high SAE feature activation variance
    are harder (more complex internal representation).
    """

    def __init__(self):
        self._difficulties: dict[int, float] = {}

    def estimate(self, sample_ids: list[int], features: torch.Tensor) -> dict[int, float]:
        """Estimate difficulty for each sample.

        Args:
            sample_ids: list of sample IDs
            features: (N, n_features) SAE features

        Returns:
            difficulties: sample_id → difficulty (0=easy, 1=hard)
        """
        # Difficulty = feature variance (high variance = complex = hard)
        variances = features.float().var(dim=-1)  # (N,)
        variances = (variances - variances.min()) / (variances.max() - variances.min() + 1e-8)

        for i, sid in enumerate(sample_ids):
            self._difficulties[sid] = variances[i].item()

        return self._difficulties

    def order_curriculum(self, sample_ids: list[int]) -> list[int]:
        """Order samples from easy to hard (curriculum learning)."""
        return sorted(sample_ids, key=lambda sid: self._difficulties.get(sid, 0.5))

    def get_difficulty(self, sample_id: int) -> float:
        return self._difficulties.get(sample_id, 0.5)


class SAERLQualityFilter:
    """Filters low-quality training data using SAE features.

    Quality probe: samples whose SAE features are close to the "quality
    centroid" (computed from high-reward samples) are high-quality.
    """

    def __init__(self, quality_threshold: float = 0.3):
        self.quality_threshold = quality_threshold
        self._quality_centroid: Optional[torch.Tensor] = None
        self._quality_scores: dict[int, float] = {}

    def fit(self, features: torch.Tensor, rewards: torch.Tensor):
        """Fit quality probe from features and rewards.

        Args:
            features: (N, n_features) SAE features
            rewards: (N,) reward scores (higher = better quality)
        """
        # Quality centroid = mean of top-25% reward samples
        top_threshold = rewards.quantile(0.75)
        high_quality_mask = rewards >= top_threshold
        if high_quality_mask.any():
            self._quality_centroid = features[high_quality_mask].mean(dim=0)
        else:
            self._quality_centroid = features.mean(dim=0)

    def score(self, sample_ids: list[int], features: torch.Tensor) -> dict[int, float]:
        """Score quality for each sample.

        Returns:
            quality_scores: sample_id → quality (0=low, 1=high)
        """
        if self._quality_centroid is None:
            return {sid: 0.5 for sid in sample_ids}

        # Quality = negative distance to quality centroid
        distances = torch.norm(features - self._quality_centroid, dim=-1)
        distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        quality = 1.0 - distances  # closer = higher quality

        for i, sid in enumerate(sample_ids):
            self._quality_scores[sid] = quality[i].item()

        return self._quality_scores

    def filter(self, sample_ids: list[int]) -> list[int]:
        """Filter out low-quality samples."""
        return [sid for sid in sample_ids
                if self._quality_scores.get(sid, 0.5) >= self.quality_threshold]


class SAERLDataEngineer:
    """Combines SAE-guided diversity, difficulty, and quality for RL data engineering.

    Usage:
        # Extract features
        de = SAERLDataEngineer(d_model=2048)
        de.fit_sae(hidden_states)

        # For each RL training batch:
        batch = de.engineer_batch(sample_ids, features, rewards, batch_size=32)
    """

    def __init__(self, d_model: int, n_features: int = 256,
                 n_clusters: int = 8, device: str = "cuda"):
        self.feature_extractor = SAERLFeatureExtractor(d_model, n_features, device)
        self.diversity = SAERLDiversityController(n_clusters=n_clusters)
        self.difficulty = SAERLDifficultyEstimator()
        self.quality = SAERLQualityFilter()

    def fit_sae(self, hidden_states: torch.Tensor):
        """Train the SAE on collected hidden states."""
        self.feature_extractor.train(hidden_states)

    def fit_all(self, sample_ids: list[int], hidden_states: torch.Tensor,
                rewards: torch.Tensor):
        """Fit all components (SAE, diversity, difficulty, quality)."""
        features = self.feature_extractor.extract(hidden_states)
        if features.dim() > 2:
            features = features.mean(dim=1)  # pool over sequence

        features_np = features.detach()
        self.diversity.fit(features_np)
        self.difficulty.estimate(sample_ids, features_np)
        self.quality.fit(features_np, rewards)

    def engineer_batch(self, sample_ids: list[int], features: torch.Tensor,
                       rewards: torch.Tensor, batch_size: int = 32) -> list[int]:
        """Engineer a training batch with SAE-guided diversity + quality filtering.

        1. Filter low-quality samples
        2. Order by difficulty (curriculum)
        3. Compose diverse batch from different clusters
        """
        # Extract features if needed
        if features.dim() > 2:
            features = self.feature_extractor.extract(features)
            if features.dim() > 2:
                features = features.mean(dim=1)

        # Score quality
        self.quality.score(sample_ids, features.detach())

        # Filter
        filtered = self.quality.filter(sample_ids)
        if not filtered:
            filtered = sample_ids

        # Order by difficulty (curriculum)
        ordered = self.difficulty.order_curriculum(filtered)

        # Compose diverse batch
        batch = self.diversity.compose_batch(ordered, batch_size)

        return batch

    def stats(self) -> dict:
        return {
            "sae_trained": self.feature_extractor._trained,
            "n_clusters": self.diversity.n_clusters,
            "n_difficulty_estimates": len(self.difficulty._difficulties),
            "n_quality_scores": len(self.quality._quality_scores),
        }
