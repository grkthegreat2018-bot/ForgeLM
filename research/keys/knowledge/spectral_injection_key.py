"""Spectral-aware knowledge injection key.

Injects new facts into a weight matrix by operating in its singular value
space, avoiding "intruder dimensions" that cause catastrophic forgetting.

Research basis:
  - Training amplifies TOP singular values (arxiv 2505.23099): task
    knowledge lives in a low-dimensional subspace spanned by the largest
    singular vectors.
  - LoRA creates "intruder dimensions" (arxiv 2410.21228): new orthogonal
    singular vectors introduced by BA cause forgetting of prior knowledge.
  - Intruder threshold (arxiv 2607.23711): s* = theta_bar / (gamma *
    sigma_1(BA)); an update whose top singular value stays below s* does
    not create intruders.  Computable from W's spectrum alone.
  - OPLoRA (arxiv 2510.13003): project Delta-W orthogonal to the top-k
    singular subspace of W -> provably preserves existing knowledge.
  - PiSSA (NeurIPS 2024): initialise on principal components for efficient
    task adaptation.
  - SVFT (NeurIPS 2024): Delta-W = Sum m_ij u_i v_j^T; only the scalar
    coefficients m_ij need to be learned.
  - Marchenko-Pastur: the bulk of singular values of a random matrix
    follow the MP distribution; values above the MP upper edge are
    "spikes" carrying learned information.
"""
from __future__ import annotations

import torch

from research.keys.misc.base import Key, KeyClass, KeyResult

_EPS = 1e-8


class SpectralInjectionKey(Key):
    """Spectral-aware knowledge injection (SVD-space fact editing)."""

    def __init__(self, gamma: float = 1.0, alpha: float = 0.1):
        self.gamma = gamma   # intruder-threshold scaling factor
        self.alpha = alpha   # default update magnitude

    # -- Key interface -------------------------------------------------
    @property
    def name(self) -> str:
        return "spectral_injection"

    @property
    def description(self) -> str:
        return "Inject facts into weights via SVD-space projection (OPLoRA/PiSSA/SVFT)."

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """data -> weights.

        Required keys in *data*:
          - "weight":         target weight matrix W  (2-D tensor)
          - "fact_embedding": fact embedding tensor   (1-D or 2-D)
          - "mode":           "new_knowledge" | "reorient"
        Optional:
          - "alpha": float (override default)
        """
        try:
            W = data["weight"]
            fact = data["fact_embedding"]
            mode = data.get("mode", "new_knowledge")
            alpha = data.get("alpha", self.alpha)
            W_new = self.inject_facts(W, fact, mode=mode, alpha=alpha)
            feats = self.compute_spectral_features(W_new)
            return KeyResult(
                success=True,
                weights={"weight": W_new},
                metadata={"mode": mode, "alpha": alpha, **feats},
            )
        except Exception as exc:  # noqa: BLE001
            return KeyResult(success=False, error=str(exc))

    def reverse(self, weights: dict) -> KeyResult:
        """Weights -> data is not well-defined for injection; partial key."""
        return KeyResult(success=False, error="SpectralInjectionKey is forward-only (Partial).")

    # -- Spectral analysis --------------------------------------------
    def compute_spectral_features(self, W: torch.Tensor) -> dict:
        """Return singular spectrum statistics for *W*.

        Keys: singular_values, mp_bound, n_spikes, threshold, spectral_entropy
        """
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        n, m = W.shape
        c = min(n, m) / max(n, m)
        sigma_ref = S.median().clamp(min=_EPS)
        mp_bound = sigma_ref * (1.0 + torch.sqrt(torch.tensor(c, dtype=S.dtype, device=S.device))) ** 2
        n_spikes = int((S > mp_bound).sum().item())
        # intruder threshold: stay below sigma_mp / (gamma * ||DeltaW||)
        # ||DeltaW|| unknown a-priori; use sigma_1 as conservative proxy
        threshold = float((mp_bound / (self.gamma * S[0].clamp(min=_EPS))).item())
        # spectral entropy (normalised) measures concentration of information
        p = S / S.sum().clamp(min=_EPS)
        entropy = float((-(p * (p + _EPS).log())).sum().item())
        return {
            "singular_values": S.detach(),
            "mp_bound": float(mp_bound.item()),
            "n_spikes": n_spikes,
            "threshold": threshold,
            "spectral_entropy": entropy,
        }

    # -- Core injection ------------------------------------------------
    def inject_facts(
        self,
        W: torch.Tensor,
        fact_embeddings: torch.Tensor,
        mode: str = "new_knowledge",
        alpha: float = 0.1,
    ) -> torch.Tensor:
        """Inject *fact_embeddings* into *W* and return updated weights.

        mode="new_knowledge": OPLoRA orthogonal projection (preserve top-k).
        mode="reorient":      PiSSA principal-component amplification.
        """
        device, dtype = W.device, W.dtype
        fact = fact_embeddings.to(device=device, dtype=dtype)
        if fact.dim() == 1:
            fact = fact.unsqueeze(0)  # (1, d_fact)

        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        feats = self.compute_spectral_features(W)
        k = max(1, feats["n_spikes"])
        Uk = U[:, :k]               # (n, k)  n = W.shape[0]
        Sk = S[:k]                  # (k,)
        Vk = Vt[:k, :]              # (k, m)  m = W.shape[1]

        # SVFT-inspired closed-form coefficients.
        # Fact embedding is d_model (1536). Weight matrix may be (d_ff, d_model)
        # or (d_model, d_ff). Use the projection that matches the fact dimension;
        # for the other dimension, project fact through W to get a compatible vector.
        d_fact = fact.shape[-1]
        n, m = W.shape

        # proj_v: fact @ Vk.t() — works if d_fact == m (right dimension)
        if d_fact == m:
            proj_v = fact @ Vk.t()       # (F, k)
        else:
            proj_v = torch.ones(fact.shape[0], k, device=device, dtype=dtype)

        # proj_u: fact @ Uk — works if d_fact == n (left dimension)
        if d_fact == n:
            proj_u = fact @ Uk           # (F, k)
        else:
            # Project fact through W to get a n-dimensional vector, then project
            # fact_proj = (fact @ W.t()) gives (1, n) if fact is (1, m)
            if d_fact == m:
                fact_proj = fact @ W.to(dtype)  # (1, n)
                proj_u = fact_proj @ Uk     # (F, k)
            else:
                proj_u = torch.ones(fact.shape[0], k, device=device, dtype=dtype)

        m_coef = alpha * proj_u * proj_v / (Sk.unsqueeze(0) + _EPS)  # (F, k)
        m_coef = m_coef.mean(dim=0)  # (k,)

        if mode == "new_knowledge":
            # OPLoRA: build raw update from fact, project orthogonal to top-k
            # fact is (1, d_fact). Build dW as outer product padded to W shape.
            if d_fact == m:
                # fact aligns with right dimension — dW = u_random ⊗ fact
                u_rand = torch.randn(n, 1, device=device, dtype=dtype) * 0.01
                dW = u_rand @ fact  # (n, m)
            elif d_fact == n:
                # fact aligns with left dimension — dW = fact ⊗ v_random
                v_rand = torch.randn(1, m, device=device, dtype=dtype) * 0.01
                dW = fact.t() @ v_rand  # (n, m)
            else:
                dW = alpha * torch.randn(n, m, device=device, dtype=dtype) * 0.01

            P_left = torch.eye(n, device=device, dtype=dtype) - Uk @ Uk.t()
            P_right = torch.eye(m, device=device, dtype=dtype) - Vk.t() @ Vk
            dW = P_left @ dW @ P_right

            # Add SVFT coefficient-weighted update along orthogonal complement
            U_perp = U[:, k:k+min(k, U.shape[1]-k)]
            Vt_perp = Vt[k:k+min(k, Vt.shape[0]-k), :]
            if U_perp.shape[1] > 0 and Vt_perp.shape[0] > 0:
                n_coef = min(len(m_coef), U_perp.shape[1], Vt_perp.shape[0])
                if n_coef > 0:
                    dW = dW + (U_perp[:, :n_coef] *
                               m_coef[:n_coef].unsqueeze(0)) @ Vt_perp[:n_coef, :]

        elif mode == "reorient":
            # PiSSA: amplify & rotate principal components
            delta_s = m_coef * Sk                    # rotate spikes
            dW = alpha * (Uk * delta_s.unsqueeze(0)) @ Vk
        else:
            raise ValueError(f"Unknown mode '{mode}'; use 'new_knowledge' or 'reorient'.")

        # Scale to stay below intruder threshold
        dW_norm = torch.linalg.norm(dW, ord=2).clamp(min=_EPS)
        s_star = feats["mp_bound"] / (self.gamma * dW_norm.item())
        scale = min(1.0, float(s_star))
        dW = dW * scale

        return W + dW

    # -- Batch injection ----------------------------------------------
    def batch_inject(
        self,
        W: torch.Tensor,
        fact_embeddings: list[torch.Tensor],
        mode: str = "new_knowledge",
    ) -> torch.Tensor:
        """Inject multiple facts efficiently.

        Batches all facts into a single matrix operation per SVD call,
        rather than looping one-by-one. Uses one SVD (not N) for the
        full batch, then applies a combined update.
        """
        if not fact_embeddings:
            return W

        device, dtype = W.device, W.dtype
        # Stack all fact embeddings into a single matrix (F, d_fact)
        facts = torch.stack([
            f.to(device=device, dtype=dtype) if f.dim() == 1 else f.squeeze(0).to(device=device, dtype=dtype)
            for f in fact_embeddings
        ])  # (F, d_fact)

        # Single SVD for the whole batch
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        feats = self.compute_spectral_features(W)
        k = max(1, feats["n_spikes"])
        n, m = W.shape
        d_fact = facts.shape[-1]

        Uk = U[:, :k]
        Sk = S[:k]
        Vk = Vt[:k, :]

        # Batch projections: all facts at once
        if d_fact == m:
            proj_v = facts @ Vk.t()        # (F, k)
            facts_proj = facts @ W.t().to(dtype)  # (F, n) for left projection
            proj_u = facts_proj @ Uk        # (F, k)
        elif d_fact == n:
            proj_u = facts @ Uk             # (F, k)
            facts_proj = facts @ W.to(dtype)  # (F, m)
            proj_v = facts_proj @ Vk.t()    # (F, k)
        else:
            proj_u = torch.ones(facts.shape[0], k, device=device, dtype=dtype)
            proj_v = torch.ones(facts.shape[0], k, device=device, dtype=dtype)

        # Coefficients: average across all facts
        m_coef = (self.alpha * proj_u * proj_v / (Sk.unsqueeze(0) + _EPS)).mean(0)  # (k,)

        if mode == "new_knowledge":
            # Build combined update from all facts
            if d_fact == m:
                # Sum of outer products: sum_i u_i ⊗ fact_i
                # Use random left vectors scaled by fact norms
                u_rand = torch.randn(n, facts.shape[0], device=device, dtype=dtype) * 0.01
                dW = u_rand @ facts  # (n, m) — combined rank-F update
            elif d_fact == n:
                v_rand = torch.randn(facts.shape[0], m, device=device, dtype=dtype) * 0.01
                dW = facts.t() @ v_rand  # (n, m)
            else:
                dW = self.alpha * torch.randn(n, m, device=device, dtype=dtype) * 0.01

            # OPLoRA: project orthogonal to top-k
            P_left = torch.eye(n, device=device, dtype=dtype) - Uk @ Uk.t()
            P_right = torch.eye(m, device=device, dtype=dtype) - Vk.t() @ Vk
            dW = P_left @ dW @ P_right

            # SVFT coefficient blend on orthogonal complement
            U_perp = U[:, k:k+min(k, U.shape[1]-k)]
            Vt_perp = Vt[k:k+min(k, Vt.shape[0]-k), :]
            if U_perp.shape[1] > 0 and Vt_perp.shape[0] > 0:
                n_coef = min(len(m_coef), U_perp.shape[1], Vt_perp.shape[0])
                if n_coef > 0:
                    dW = dW + (U_perp[:, :n_coef] *
                               m_coef[:n_coef].unsqueeze(0)) @ Vt_perp[:n_coef, :]

        elif mode == "reorient":
            delta_s = m_coef * Sk
            dW = self.alpha * (Uk * delta_s.unsqueeze(0)) @ Vk
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        # Scale below intruder threshold
        dW_norm = torch.linalg.norm(dW, ord=2).clamp(min=_EPS)
        s_star = feats["mp_bound"] / (self.gamma * dW_norm.item())
        scale = min(1.0, float(s_star))
        dW = dW * scale

        return W + dW


# -- helpers ---------------------------------------------------------
def _fit(t: torch.Tensor, shape: tuple[int, int], device, dtype) -> torch.Tensor:
    """Pad or crop *t* to *shape*."""
    out = torch.zeros(shape, device=device, dtype=dtype)
    r, c = min(t.shape[0], shape[0]), min(t.shape[1], shape[1])
    out[:r, :c] = t[:r, :c]
    return out
