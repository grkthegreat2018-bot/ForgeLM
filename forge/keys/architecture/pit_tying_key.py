"""R37-3: PIT Key (Pseudo-Inverse Tying) — coupled projections of shared
orthonormal token memory.

Standard weight tying uses a SINGLE matrix for both embedding and unembedding,
which biases the shared representation toward the output (unembedding) space
and degrades input representation quality — especially harmful at small scale
where embeddings dominate the parameter budget.

PIT (Pseudo-Inverse Tying, arXiv:2602.04556) decouples the two by introducing a
shared orthonormal token memory M and expressing BOTH the embedding and the
unembedding as projections of M:

    W_emb    = P_emb  @ M          (V, d) = (V, d) @ (d, d)
    W_unemb  = M      @ P_unemb    (d, V) = (d, d) @ (d, V)

The shared memory M is recovered from a tied (or independently trained)
embedding/unembedding pair via a thin polar decomposition of the product
W_emb @ W_unemb:

    W_emb @ W_unemb = U @ S @ Vh   (SVD)
    M = U @ Vh                     (orthonormal, closest unitary factor)

Because M is orthonormal (M @ M^T = I on the d×d block), the two projections
are well-conditioned and the forward/reverse directions are stable triangular
solves rather than matrix inversions.

Key class: BI — both forward (data -> weights) and reverse (weights -> data)
round-trip to identity when the projections are full rank.

Usage:
    from forge.keys.architecture.pit_tying_key import PITKey

    key = PITKey()
    # data -> weights: build shared memory + projections from W_emb, W_unemb
    res = key.forward({"W_emb": W_emb, "W_unemb": W_unemb})
    M, P_emb, P_unemb = res.weights["M"], res.weights["P_emb"], res.weights["P_unemb"]

    # weights -> data: reconstruct W_emb, W_unemb from shared M + projections
    res = key.reverse(res.weights)
    W_emb_rec, W_unemb_rec = res.data["W_emb"], res.data["W_unemb"]
"""
from __future__ import annotations

import torch

from forge.keys.misc.base import Key, KeyClass, KeyResult


class PITKey(Key):
    """PIT (Pseudo-Inverse Tying) key — coupled projections of shared memory.

    Synchronizes embedding (W_emb: V x d) and unembedding (W_unemb: d x V) as
    coupled projections of a shared orthonormal latent token memory M (d x d).

    forward(data): Given W_emb and W_unemb, compute the shared orthonormal
        memory M via thin polar decomposition of W_emb @ W_unemb, then solve
        for the per-side projection matrices P_emb and P_unemb so that
        W_emb = P_emb @ M and W_unemb = M @ P_unemb.

    reverse(weights): Reconstruct W_emb and W_unemb from the shared memory M
        and the projection matrices P_emb, P_unemb via stable triangular
        solves (no explicit matrix inversion).
    """

    @property
    def name(self) -> str:
        return "pit_tying"

    @property
    def description(self) -> str:
        return (
            "PIT (Pseudo-Inverse Tying): shared orthonormal token memory via "
            "thin polar decomposition; embedding and unembedding are coupled "
            "projections of M."
        )

    def key_class(self) -> KeyClass:
        return KeyClass.BI

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _polar_orthonormal(W_emb: torch.Tensor,
                           W_unemb: torch.Tensor) -> torch.Tensor:
        """Shared orthonormal memory M via thin polar decomposition.

        Computes the SVD of W_unemb @ W_emb (d x d) and returns the closest
        orthonormal factor M = U @ Vh. This is the unitary factor of the polar
        decomposition of the product.

        Args:
            W_emb: (V, d) embedding matrix.
            W_unemb: (d, V) unembedding matrix.

        Returns:
            M: (d, d) orthonormal matrix (M @ M.T ≈ I).
        """
        # Product is (d, d) — small and cheap to decompose.
        # W_emb is (V, d), W_unemb is (d, V) → W_unemb @ W_emb is (d, d)
        prod = W_unemb @ W_emb  # (d, d)
        # Thin SVD: prod = U @ diag(S) @ Vh
        U, _, Vh = torch.linalg.svd(prod, full_matrices=False)
        # Orthonormal factor of the polar decomposition.
        M = U @ Vh  # (d, d)
        return M

    @staticmethod
    def _solve_projection_left(W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Solve W = P @ M for P (left projection) via stable triangular solve.

        W = P @ M  =>  W^T = M^T @ P^T  =>  solve M^T @ P^T = W^T for P^T.
        Use a Cholesky-style stable solve through QR of M^T.

        Args:
            W: (V, d) target matrix (rows live in the d-dimensional space).
            M: (d, d) orthonormal shared memory.

        Returns:
            P: (V, d) projection matrix with W ≈ P @ M.
        """
        # M is orthonormal => M^T = M^{-1}, so P = W @ M^T.
        # Use lstsq for numerical stability when M is near-orthonormal.
        # W = P @ M  ->  M^T @ P^T = W^T  ->  solve for P^T.
        # torch.linalg.lstsq handles the (d, d) system robustly.
        P_T = torch.linalg.lstsq(M.T, W.T).solution  # (d, V)
        return P_T.T  # (V, d)

    @staticmethod
    def _solve_projection_right(W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Solve W = M @ P for P (right projection) via stable triangular solve.

        W = M @ P  =>  solve M @ P = W for P. M is orthonormal so P = M^T @ W,
        but we use lstsq for numerical robustness.

        Args:
            W: (d, V) target matrix.
            M: (d, d) orthonormal shared memory.

        Returns:
            P: (d, V) projection matrix with W ≈ M @ P.
        """
        P = torch.linalg.lstsq(M, W).solution  # (d, V)
        return P

    # ── forward / reverse ──────────────────────────────────────────────────

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights.

        Given embedding W_emb (V, d) and unembedding W_unemb (d, V), compute
        the shared orthonormal memory M via thin polar decomposition of
        W_emb @ W_unemb, then express both as projections of M:
            W_emb   = P_emb   @ M
            W_unemb = M       @ P_unemb

        Args:
            data: {"W_emb": (V, d), "W_unemb": (d, V)}.

        Returns:
            KeyResult with weights {"M", "P_emb", "P_unemb"}.
        """
        try:
            W_emb = data.get("W_emb")
            W_unemb = data.get("W_unemb")
            if W_emb is None or W_unemb is None:
                return KeyResult(
                    success=False,
                    error="Missing 'W_emb' or 'W_unemb' in data",
                )
            if W_emb.shape[1] != W_unemb.shape[0]:
                return KeyResult(
                    success=False,
                    error=(
                        f"Shape mismatch: W_emb {tuple(W_emb.shape)} and "
                        f"W_unemb {tuple(W_unemb.shape)} must share the d "
                        f"dimension (W_emb.shape[1] == W_unemb.shape[0])."
                    ),
                )

            V, d = W_emb.shape

            # Shared orthonormal memory via thin polar decomposition.
            M = self._polar_orthonormal(W_emb, W_unemb)  # (d, d)

            # Solve for projections: W_emb = P_emb @ M, W_unemb = M @ P_unemb.
            P_emb = self._solve_projection_left(W_emb, M)    # (V, d)
            P_unemb = self._solve_projection_right(W_unemb, M)  # (d, V)

            return KeyResult(
                success=True,
                weights={"M": M, "P_emb": P_emb, "P_unemb": P_unemb},
                metadata={
                    "vocab_size": V,
                    "d_model": d,
                    "method": "thin_polar_svd",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """weights -> data.

        Reconstruct W_emb and W_unemb from the shared memory M and the
        projection matrices:
            W_emb   = P_emb   @ M
            W_unemb = M       @ P_unemb

        Because M is orthonormal these are simple, stable matrix products
        (no inversion required).

        Args:
            weights: {"M": (d, d), "P_emb": (V, d), "P_unemb": (d, V)}.

        Returns:
            KeyResult with data {"W_emb", "W_unemb"}.
        """
        try:
            M = weights.get("M")
            P_emb = weights.get("P_emb")
            P_unemb = weights.get("P_unemb")
            if M is None or P_emb is None or P_unemb is None:
                return KeyResult(
                    success=False,
                    error="Missing 'M', 'P_emb', or 'P_unemb' in weights",
                )

            # Reconstruct via the projection equations.
            W_emb = P_emb @ M        # (V, d)
            W_unemb = M @ P_unemb    # (d, V)

            return KeyResult(
                success=True,
                data={"W_emb": W_emb, "W_unemb": W_unemb},
                metadata={
                    "vocab_size": W_emb.shape[0],
                    "d_model": W_emb.shape[1],
                    "orthonormal": bool(
                        torch.allclose(M @ M.T, torch.eye(M.shape[0],
                                                          device=M.device,
                                                          dtype=M.dtype),
                                       atol=1e-5)),
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))
