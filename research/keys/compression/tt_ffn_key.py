"""Tensor-Train (TT) decomposed linear layers for FFN weight compression.

Implements Tensor-Train (TT) format decomposition of the SwiGLU FFN weight
matrices. A dense weight matrix W of shape (out, in) is decomposed into a
chain of 4 low-rank cores, drastically reducing parameter count while
preserving the ability to represent the original linear map (up to the
chosen TT rank).

TT format for W (out, in) with 4 cores:
    out = o1 * o2 * o3 * o4
    in  = i1 * i2 * i3 * i4
    Cores (each G_k has shape (r_{k-1}, i_k, o_k, r_k), r_0 = r_4 = 1):
        G1 (1,  i1, o1, r)
        G2 (r,  i2, o2, r)
        G3 (r,  i3, o3, r)
        G4 (r,  i4, o4, 1)
    Parameter count: r*i1*o1 + r*r*i2*o2 + r*r*i3*o3 + r*i4*o4
    vs out*in for the dense matrix.

The forward pass contracts the input tensor through the core chain via
sequential einsum contractions, which is mathematically equivalent to
y = x @ W^T (up to the rank-induced approximation error).

Reference: Novikov et al., "Tensorizing Neural Networks" (NeurIPS 2015);
Oseledets, "Tensor-Train Decomposition" (SIAM J. Sci. Comput. 2011).

Usage:
    from research.keys.compression.tt_ffn_key import TTLinear, TTSwiGLUFFN

    # From a dense weight matrix:
    tt = TTLinear.from_dense(weight, tt_rank=4)

    # From a dense SwiGLUFFN:
    tt_ffn = TTSwiGLUFFN.from_dense_ffn(swiglu_ffn, tt_rank=4)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Factorization helpers
# ---------------------------------------------------------------------------

def _count_prime_factors(n: int) -> int:
    """Count prime factors of ``n`` with multiplicity."""
    c = 0
    d = 2
    v = n
    while d * d <= v:
        while v % d == 0:
            c += 1
            v //= d
        d += 1
    if v > 1:
        c += 1
    return c


def _factorize(n: int, n_factors: int = 4) -> List[int]:
    """Factorize ``n`` into ``n_factors`` integers (>= 2 when possible).

    Distributes prime factors greedily into balanced buckets so the resulting
    factors are as close to ``n^(1/n_factors)`` as possible.
    """
    if n <= 1:
        return [1] * n_factors
    primes: List[int] = []
    d = 2
    val = n
    while d * d <= val:
        while val % d == 0:
            primes.append(d)
            val //= d
        d += 1
    if val > 1:
        primes.append(val)

    if len(primes) < n_factors:
        # Not enough prime factors — pad with 1s on the left.
        return [1] * (n_factors - len(primes)) + primes

    buckets = [1] * n_factors
    for p in sorted(primes, reverse=True):
        idx = min(range(n_factors), key=lambda i: buckets[i])
        buckets[idx] *= p
    return buckets


def _pad_dim(n: int, n_factors: int = 4) -> Tuple[int, List[int]]:
    """Return (padded_n, factors) so ``padded_n`` has >= n_factors prime factors.

    Pads ``n`` upward to the next integer with enough prime factors to split
    into ``n_factors`` parts all >= 2.
    """
    candidate = n
    while True:
        if candidate >= 2 and _count_prime_factors(candidate) >= n_factors:
            return candidate, _factorize(candidate, n_factors)
        candidate += 1


# ---------------------------------------------------------------------------
# TT-SVD core algorithm
# ---------------------------------------------------------------------------

def _tt_svd(tensor: torch.Tensor, max_rank: int) -> List[torch.Tensor]:
    """Decompose ``tensor`` of shape (d1, ..., dN) into TT cores via TT-SVD.

    Each returned core has shape (r_{k-1}, d_k, r_k) with r_0 = r_N = 1.
    Internal bonds are truncated to at most ``max_rank``.

    Reference: Oseledets 2011, Algorithm 1.
    """
    device, dtype = tensor.device, tensor.dtype
    shape = list(tensor.shape)
    N = len(shape)
    cores: List[torch.Tensor] = []

    r_prev = 1
    current = tensor.reshape(r_prev, -1)  # (1, prod(shape))

    for k in range(N - 1):
        d_k = shape[k]
        rest = current.numel() // (r_prev * d_k)
        current = current.reshape(r_prev * d_k, rest)
        U, S, Vh = torch.linalg.svd(current, full_matrices=False)
        r_new = min(max_rank, U.shape[1], S.shape[0])
        U = U[:, :r_new]
        S = S[:r_new]
        Vh = Vh[:r_new, :]
        cores.append(U.reshape(r_prev, d_k, r_new))
        current = (S[:r_new].unsqueeze(1) * Vh).to(dtype=dtype)
        r_prev = r_new

    # Last core: (r_prev, d_N, 1)
    cores.append(current.reshape(r_prev, shape[-1], 1))
    return cores


# ---------------------------------------------------------------------------
# TTLinear
# ---------------------------------------------------------------------------

class TTLinear(nn.Module):
    """Linear layer with weight stored in Tensor-Train (TT) format.

    The weight matrix W (out_features, in_features) is represented as a chain
    of 4 TT cores. Forward computes ``y = x @ W^T`` via sequential tensor
    contractions.

    Cores (``self.cores``, an ``nn.ParameterList``):
        cores[0]: (1,  in_factors[0], out_factors[0], r)
        cores[1]: (r,  in_factors[1], out_factors[1], r)
        cores[2]: (r,  in_factors[2], out_factors[2], r)
        cores[3]: (r,  in_factors[3], out_factors[3], 1)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_factors: List[int],
        out_factors: List[int],
        tt_rank: int = 4,
        in_pad: int = 0,
        out_pad: int = 0,
    ) -> None:
        super().__init__()
        assert len(in_factors) == 4
        assert len(out_factors) == 4
        self.in_features = in_features
        self.out_features = out_features
        self.in_pad = in_pad
        self.out_pad = out_pad
        self.tt_rank = tt_rank
        self.in_factors = list(in_factors)
        self.out_factors = list(out_factors)

        r = tt_rank
        shapes = [
            (1, in_factors[0], out_factors[0], r),
            (r, in_factors[1], out_factors[1], r),
            (r, in_factors[2], out_factors[2], r),
            (r, in_factors[3], out_factors[3], 1),
        ]
        self.cores = nn.ParameterList(
            [nn.Parameter(torch.empty(*s)) for s in shapes]
        )
        for c in self.cores:
            nn.init.normal_(c, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct the full dense weight matrix (out_features, in_features).

        Materializes the entire matrix — mainly for debugging/inspection.
        """
        G1, G2, G3, G4 = self.cores
        g1 = G1.squeeze(0)    # (i1, o1, r1)
        g4 = G4.squeeze(-1)   # (r3, i4, o4)
        # Contract g1 and G2 along r1 -> (i1, o1, i2, o2, r2)
        t = torch.einsum("abc,cdef->abdef", g1, G2)
        # Contract with G3 along r2 -> (i1, o1, i2, o2, i3, o3, r3)
        t = torch.einsum("abdef,fghl->abdeghl", t, G3)
        # Contract with g4 along r3 -> (i1, o1, i2, o2, i3, o3, i4, o4)
        t = torch.einsum("abdeghl,lmn->abdeghmn", t, g4)
        # Permute to (i1, i2, i3, i4, o1, o2, o3, o4)
        i1, i2, i3, i4 = self.in_factors
        o1, o2, o3, o4 = self.out_factors
        t = t.view(i1, o1, i2, o2, i3, o3, i4, o4)
        t = t.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        padded_in = i1 * i2 * i3 * i4
        padded_out = o1 * o2 * o3 * o4
        W = t.view(padded_in, padded_out).t().contiguous()  # (padded_out, padded_in)
        return W[: self.out_features, : self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = x @ W^T using TT core contractions.

        Args:
            x: (..., in_features)
        Returns:
            y: (..., out_features)
        """
        orig_shape = x.shape[:-1]
        device, dtype = x.device, x.dtype

        # Pad input if necessary.
        if self.in_pad > 0:
            pad = torch.zeros(*orig_shape, self.in_pad, device=device, dtype=dtype)
            x = torch.cat([x, pad], dim=-1)

        i1, i2, i3, i4 = self.in_factors
        o1, o2, o3, o4 = self.out_factors
        G1, G2, G3, G4 = self.cores

        # Reshape input: (..., i1, i2, i3, i4)
        x = x.view(*orig_shape, i1, i2, i3, i4)

        # Contract with G1[0]: (i1, o1, r) -> (..., o1, r, i2, i3, i4)
        res = torch.einsum("...abcd,aer->...erbcd", x, G1[0])
        # Contract with G2: (r, i2, o2, r) -> (..., o1, o2, r, i3, i4)
        res = torch.einsum("...erbcd,rbpq->...epqcd", res, G2)
        # Contract with G3: (r, i3, o3, r) -> (..., o1, o2, o3, r, i4)
        res = torch.einsum("...epqcd,qcst->...epstd", res, G3)
        # Contract with G4 squeezed: (r, i4, o4) -> (..., o1, o2, o3, o4)
        res = torch.einsum("...epstd,tdu->...epsu", res, G4.squeeze(-1))

        # Reshape to (..., out)
        res = res.reshape(*orig_shape, o1 * o2 * o3 * o4)
        if res.shape[-1] != self.out_features:
            res = res[..., : self.out_features]
        return res

    @classmethod
    def from_dense(cls, weight: torch.Tensor, tt_rank: int = 4) -> "TTLinear":
        """Decompose a dense weight matrix into TT format via TT-SVD.

        Args:
            weight: Dense weight of shape (out_features, in_features).
            tt_rank: Maximum TT rank (same for all internal bonds).

        Returns:
            A ``TTLinear`` whose cores approximate ``weight``.
        """
        out_features, in_features = weight.shape
        device, dtype = weight.device, weight.dtype

        padded_in, in_factors = _pad_dim(in_features, n_factors=4)
        padded_out, out_factors = _pad_dim(out_features, n_factors=4)
        in_pad = padded_in - in_features
        out_pad = padded_out - out_features

        # Embed weight into padded matrix (pad with zeros).
        W_padded = torch.zeros(padded_out, padded_in, device=device, dtype=dtype)
        W_padded[:out_features, :in_features] = weight

        # Reshape to (o1, o2, o3, o4, i1, i2, i3, i4) then permute to
        # (i1, o1, i2, o2, i3, o3, i4, o4) so consecutive (i_k, o_k) pairs
        # are adjacent — the natural ordering for TT-SVD.
        i1, i2, i3, i4 = in_factors
        o1, o2, o3, o4 = out_factors
        T = W_padded.view(o1, o2, o3, o4, i1, i2, i3, i4)
        T = T.permute(4, 0, 5, 1, 6, 2, 7, 3).contiguous()
        # Flatten into interleaved mode order: (i1*o1, i2*o2, i3*o3, i4*o4)
        mode_sizes = [i1 * o1, i2 * o2, i3 * o3, i4 * o4]
        T = T.view(mode_sizes[0], mode_sizes[1], mode_sizes[2], mode_sizes[3])

        # Run TT-SVD over the 4 modes.
        svd_cores = _tt_svd(T, max_rank=tt_rank)

        # svd_cores[k]: (r_{k}, mode_size_k, r_{k+1}) with r_0 = r_4 = 1.
        # Split each mode_size_k back into (i_k, o_k) -> (r_{k-1}, i_k, o_k, r_k).
        tt_cores: List[torch.Tensor] = []
        for k, core in enumerate(svd_cores):
            r_left, ms, r_right = core.shape
            ik = in_factors[k]
            ok = out_factors[k]
            assert ms == ik * ok, f"mode size mismatch at core {k}: {ms} != {ik*ok}"
            tt_cores.append(core.view(r_left, ik, ok, r_right))

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            in_factors=in_factors,
            out_factors=out_factors,
            tt_rank=tt_rank,
            in_pad=in_pad,
            out_pad=out_pad,
        )
        for param, new_core in zip(layer.cores, tt_cores):
            param.data = new_core.to(dtype=dtype, device=device)
        return layer

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.cores)
        dense_params = self.in_features * self.out_features
        ratio = n_params / max(dense_params, 1)
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tt_rank={self.tt_rank}, "
            f"in_factors={self.in_factors}, out_factors={self.out_factors}, "
            f"params={n_params} (dense={dense_params}, ratio={ratio:.3f})"
        )


# ---------------------------------------------------------------------------
# TTSwiGLUFFN
# ---------------------------------------------------------------------------

def _make_tt_linear(in_features: int, out_features: int, tt_rank: int) -> TTLinear:
    """Construct a ``TTLinear`` with random small cores (for direct init)."""
    padded_in, in_factors = _pad_dim(in_features, n_factors=4)
    padded_out, out_factors = _pad_dim(out_features, n_factors=4)
    return TTLinear(
        in_features=in_features,
        out_features=out_features,
        in_factors=in_factors,
        out_factors=out_factors,
        tt_rank=tt_rank,
        in_pad=padded_in - in_features,
        out_pad=padded_out - out_features,
    )


class TTSwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with TT-decompressed linear layers.

    Mirrors ``SwiGLUFFN`` in ``research.model_loader`` but replaces the three
    dense ``nn.Linear`` projections (w_gate, w_up, w_down) with ``TTLinear``
    layers, compressing the FFN weight footprint via Tensor-Train decomposition.

    Forward (same as SwiGLUFFN):
        ``self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))``

    Args:
        d_model:    Model hidden dimension.
        hidden_dim: FFN intermediate dimension. Defaults to ``8 * d_model / 3``.
        tt_rank:    TT rank used for all three TT-linear layers.
    """

    def __init__(
        self,
        d_model: int = 2048,
        hidden_dim: int | None = None,
        tt_rank: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.tt_rank = tt_rank
        self.w_gate = _make_tt_linear(d_model, hidden_dim, tt_rank)
        self.w_up = _make_tt_linear(d_model, hidden_dim, tt_rank)
        self.w_down = _make_tt_linear(hidden_dim, d_model, tt_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: ``w_down(silu(w_gate(x)) * w_up(x))``."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    @classmethod
    def from_dense_ffn(cls, ffn: nn.Module, tt_rank: int = 4) -> "TTSwiGLUFFN":
        """Build a ``TTSwiGLUFFN`` by decomposing a dense ``SwiGLUFFN``'s weights.

        Args:
            ffn: A module with ``w_gate``, ``w_up``, ``w_down`` attributes
                (each an ``nn.Linear`` with ``.weight`` of shape (out, in)).
            tt_rank: TT rank for the decomposition.

        Returns:
            A ``TTSwiGLUFFN`` whose TT cores approximate the original weights.
        """
        d_model = ffn.w_gate.in_features
        hidden_dim = ffn.w_gate.out_features
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.d_model = d_model
        obj.hidden_dim = hidden_dim
        obj.tt_rank = tt_rank
        obj.w_gate = TTLinear.from_dense(ffn.w_gate.weight.data, tt_rank=tt_rank)
        obj.w_up = TTLinear.from_dense(ffn.w_up.weight.data, tt_rank=tt_rank)
        obj.w_down = TTLinear.from_dense(ffn.w_down.weight.data, tt_rank=tt_rank)
        return obj

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        dense = 3 * self.d_model * self.hidden_dim
        return (
            f"d_model={self.d_model}, hidden_dim={self.hidden_dim}, "
            f"tt_rank={self.tt_rank}, params={n_params} "
            f"(dense={dense}, ratio={n_params / max(dense, 1):.3f})"
        )


__all__ = ["TTLinear", "TTSwiGLUFFN", "_tt_svd"]
