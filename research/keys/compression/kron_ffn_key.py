"""Kronecker-factored linear layers for FFN weight compression.

Approximates a dense weight matrix W (out, in) as the Kronecker product
A ⊗ B, where A is (a, c) and B is (b, d) with a*b = out and c*d = in.
This reduces parameter count from out*in = a*b*c*d to a*c + b*d.

The best rank-1 Kronecker approximation is computed via the Van Loan &
Pitsianis (1993) nearest-Kronecker-product algorithm: reshape W into a
block matrix, permute to the "unfolding" layout, then take the top-1 SVD.

Importable as::

    from research.keys.compression.kron_ffn_key import (
        KroneckerLinear,
        KroneckerSwiGLUFFN,
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KroneckerLinear(nn.Module):
    """Linear layer with weight W = A ⊗ B (Kronecker product).

    The effective weight matrix is (out, in) with out = a*b and in = c*d,
    but only the small factors A (a, c) and B (b, d) are stored as
    parameters, giving a compression ratio of (a*c + b*d) / (a*b*c*d).

    Forward uses the identity (A ⊗ B) vec(X) = vec(B X A^T) to avoid
    materializing the full Kronecker product:

        x (..., in) -> (..., d, c) -> X @ A^T -> B @ (...) -> (..., out)
    """

    def __init__(self, a: int, b: int, c: int, d: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.a, self.b, self.c, self.d = a, b, c, d
        self.in_features = c * d
        self.out_features = a * b
        factory = {"device": device, "dtype": dtype}
        self.A = nn.Parameter(torch.empty(a, c, **factory))
        self.B = nn.Parameter(torch.empty(b, d, **factory))
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, **factory))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-uniform init scaled so the effective weight has unit variance."""
        # Effective weight = A ⊗ B.  Var(A⊗B) = Var(A)*Var(B).  For unit
        # variance of the product, scale each factor by 1/sqrt(fan_in_eff)
        # distributed equally: each factor gets 1/fan_in_factor.
        fan_in_eff = self.in_features
        std = (1.0 / fan_in_eff) ** 0.5
        factor_std = std ** 0.5  # sqrt so that product of two has variance std^2
        nn.init.normal_(self.A, mean=0.0, std=factor_std)
        nn.init.normal_(self.B, mean=0.0, std=factor_std)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_dense(cls, weight: torch.Tensor, a: int, b: int, c: int, d,
                   bias: torch.Tensor | None = None) -> "KroneckerLinear":
        """Fit A, B from a dense weight W (out, in) via rank-1 Kronecker SVD.

        Uses the Van Loan & Pitsianis (1993) nearest-Kronecker-product
        approximation: reshape W to (a, b, c, d), permute to (a, c, b, d),
        flatten to (a*c, b*d), then take the top singular triplet.

        Args:
            weight: Dense weight tensor of shape (out, in) with out=a*b,
                in=c*d.  May be a 2-D tensor or a 1-D/ND tensor that
                reshapes to (out, in).
            a, b, c, d: Kronecker factor dimensions (a*b == out, c*d == in).
            bias: Optional bias vector of shape (out,).

        Returns:
            A ``KroneckerLinear`` whose A ⊗ B best approximates *weight*
            in Frobenius norm (rank-1 Kronecker approximation).
        """
        weight = torch.as_tensor(weight)
        if weight.dim() != 2:
            weight = weight.reshape(a * b, c * d)
        out, in_ = weight.shape
        if out != a * b or in_ != c * d:
            raise ValueError(
                f"weight shape {tuple(weight.shape)} incompatible with "
                f"Kronecker dims a={a}, b={b}, c={c}, d={d} "
                f"(expected out={a*b}, in={c*d})"
            )

        # Unfolding of the Kronecker product (Van Loan & Pitsianis).
        # W (a*b, c*d) -> (a, b, c, d) -> (a, c, b, d) -> (a*c, b*d)
        W = weight.reshape(a, b, c, d).permute(0, 2, 1, 3).reshape(a * c, b * d)

        # Rank-1 SVD: W ≈ sigma * u v^T  =>  A = sqrt(sigma) u, B = sqrt(sigma) v
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        s0 = S[0].clamp_min(0.0)
        sqrt_s = torch.sqrt(s0)
        A_flat = U[:, 0] * sqrt_s          # (a*c,)
        B_flat = sqrt_s * Vh[0, :]         # (b*d,)

        layer = cls(a, b, c, d, bias=bias is not None,
                    device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            layer.A.copy_(A_flat.reshape(a, c))
            layer.B.copy_(B_flat.reshape(b, d))
            if bias is not None:
                layer.bias.copy_(bias.reshape(-1))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = (A ⊗ B) x via the vec(B X A^T) identity.

        With the standard Kronecker convention (A⊗B)[i*b+r, j*d+s] =
        A[i,j]*B[r,s], the identity (A⊗B)vec(X) = vec(B X A^T) holds where
        X is (d, c) and vec is *column-major*.  PyTorch reshape is row-major,
        so the final flatten transposes the last two dims to emulate
        column-major vectorization.

        x: (..., in) -> (..., d, c) -> matmul A^T -> (..., d, a)
        -> B @ ... (einsum) -> (..., b, a) -> transpose -> (..., a, b)
        -> reshape (..., a*b) = (..., out)
        """
        lead_shape = x.shape[:-1]
        x = x.reshape(*lead_shape, self.d, self.c)          # (..., d, c)
        x = x.matmul(self.A.t())                            # (..., d, a)
        # B (b, d) @ x (..., d, a) -> (..., b, a)
        x = torch.einsum("bd,...da->...ba", self.B, x)
        # Column-major vec of (b, a): transpose to (a, b) then row-major flatten.
        x = x.transpose(-1, -2).reshape(*lead_shape, self.out_features)
        if self.bias is not None:
            x = x + self.bias
        return x

    def effective_weight(self) -> torch.Tensor:
        """Materialize the full (out, in) = A ⊗ B weight matrix."""
        return torch.kron(self.A, self.B)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.use_bias}, kron=({self.a}x{self.c})⊗({self.b}x{self.d}), "
            f"params={self.a * self.c + self.b * self.d} "
            f"(dense={self.out_features * self.in_features})"
        )


class KroneckerSwiGLUFFN(nn.Module):
    """SwiGLU FFN with Kronecker-factored w_gate, w_up, w_down.

    Drop-in replacement for ``SwiGLUFFN`` that compresses each projection
    weight via a rank-1 Kronecker product.  The forward computation is
    identical: ``w_down(F.silu(w_gate(x)) * w_up(x))``.

    Args:
        d_model: Model hidden dimension.
        hidden_dim: FFN intermediate dimension (default 8/3 * d_model).
        gate_kron: (a, b) factor split for w_gate/w_up (out=a*b=hidden_dim).
        down_kron: (a, b) factor split for w_down (out=a*b=d_model).
        use_clamp: GPT-OSS clamped SwiGLU (see ``SwiGLUFFN``).
        clamp_alpha, clamp_limit: Clamping hyperparameters.
    """

    def __init__(self, d_model: int = 768, hidden_dim: int | None = None,
                 gate_kron: tuple[int, int] | None = None,
                 down_kron: tuple[int, int] | None = None,
                 use_clamp: bool = False, clamp_alpha: float = 1.702,
                 clamp_limit: float = 7.0) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * d_model / 3)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.use_clamp = use_clamp
        self.clamp_alpha = clamp_alpha
        self.clamp_limit = clamp_limit

        # w_gate / w_up: weight (hidden_dim, d_model) => out=hidden_dim, in=d_model
        ga, gb = gate_kron or _default_split(hidden_dim, d_model)
        # in = d_model = c*d  -> choose c, d from d_model
        gc, gd = _default_split(d_model, hidden_dim)  # c*d = d_model
        self.w_gate = KroneckerLinear(ga, gb, gc, gd, bias=False)

        ua, ub = gate_kron or _default_split(hidden_dim, d_model)
        uc, ud = _default_split(d_model, hidden_dim)
        self.w_up = KroneckerLinear(ua, ub, uc, ud, bias=False)

        # w_down: weight (d_model, hidden_dim) => out=d_model, in=hidden_dim
        da, db = down_kron or _default_split(d_model, hidden_dim)
        dc, dd = _default_split(hidden_dim, d_model)
        self.w_down = KroneckerLinear(da, db, dc, dd, bias=False)

    @classmethod
    def from_dense_ffn(cls, ffn: nn.Module,
                       gate_kron: tuple[int, int] | None = None,
                       down_kron: tuple[int, int] | None = None) -> "KroneckerSwiGLUFFN":
        """Build a ``KroneckerSwiGLUFFN`` from a dense ``SwiGLUFFN``.

        Extracts w_gate/w_up/w_down weights and fits each with a rank-1
        Kronecker approximation via ``KroneckerLinear.from_dense``.

        Args:
            ffn: A ``SwiGLUFFN`` (or compatible module with ``w_gate``,
                ``w_up``, ``w_down`` ``nn.Linear`` attributes).
            gate_kron: (a, b) split for w_gate/w_up out dims.
            down_kron: (a, b) split for w_down out dim.

        Returns:
            A ``KroneckerSwiGLUFFN`` approximating *ffn*.
        """
        d_model = ffn.w_gate.in_features
        hidden_dim = ffn.w_gate.out_features
        use_clamp = getattr(ffn, "use_clamp", False)
        clamp_alpha = getattr(ffn, "clamp_alpha", 1.702)
        clamp_limit = getattr(ffn, "clamp_limit", 7.0)

        layer = cls(d_model=d_model, hidden_dim=hidden_dim,
                    gate_kron=gate_kron, down_kron=down_kron,
                    use_clamp=use_clamp, clamp_alpha=clamp_alpha,
                    clamp_limit=clamp_limit)

        # w_gate / w_up: weight (hidden_dim, d_model)
        ga, gb = gate_kron or _default_split(hidden_dim, d_model)
        gc, gd = _default_split(d_model, hidden_dim)
        layer.w_gate = KroneckerLinear.from_dense(
            ffn.w_gate.weight.data, ga, gb, gc, gd)

        ua, ub = gate_kron or _default_split(hidden_dim, d_model)
        uc, ud = _default_split(d_model, hidden_dim)
        layer.w_up = KroneckerLinear.from_dense(
            ffn.w_up.weight.data, ua, ub, uc, ud)

        # w_down: weight (d_model, hidden_dim)
        da, db = down_kron or _default_split(d_model, hidden_dim)
        dc, dd = _default_split(hidden_dim, d_model)
        layer.w_down = KroneckerLinear.from_dense(
            ffn.w_down.weight.data, da, db, dc, dd)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        if self.use_clamp:
            gate = gate.clamp(min=None, max=self.clamp_limit)
            up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            glu = gate * torch.sigmoid(self.clamp_alpha * gate)
            return self.w_down((up + 1) * glu)
        return self.w_down(F.silu(gate) * up)


def _default_split(out: int, in_: int) -> tuple[int, int]:
    """Pick a balanced (a, b) with a*b == out, preferring near-square factors.

    Falls back to (out, 1) when *out* is prime or no good split exists.
    """
    if out <= 1:
        return out, 1
    best = (1, out)
    best_diff = out
    for a in range(int(out ** 0.5), 0, -1):
        if out % a == 0:
            b = out // a
            diff = abs(a - b)
            if diff < best_diff:
                best = (a, b)
                best_diff = diff
            break
    return best
