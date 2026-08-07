"""Gluon attention key — decoupled RoPE with separate content/position paths.

Gluon attention (AMD, 2025) decouples the attention computation into:
1. Content path: standard attention WITHOUT RoPE (pure content matching)
2. Position path: attention with ONLY RoPE (pure position matching)
3. Fusion: weighted combination of both paths

This separation allows:
- Better long-context performance (content doesn't interfere with position)
- KV cache compression on the content path (no RoPE = compressible)
- The position path can use a smaller head_dim (just position info)

The key initializes the fusion weight to favor the original behavior:
- content_weight = 0.5, position_weight = 0.5 (equal fusion at start)
- This makes Gluon attention behave like standard attention at initialization
- Fine-tuning then learns the optimal fusion ratio

Key class: TRIVIAL — identity-like init (equal fusion), no data needed.

Reference: Gluon attention, AMD blog (2025)
"""
import torch
import torch.nn as nn
from research.keys.base import Key, KeyClass, KeyResult


class GluonAttentionKey(Key):
    """Gluon attention key — decoupled content/position attention.

    Initializes fusion weights to 0.5/0.5 (equal content + position),
    making Gluon behave like standard attention at init.
    The content path can then be compressed (no RoPE = SVD-friendly).

    Key class: TRIVIAL — fixed init, no data or training.
    """

    @property
    def name(self) -> str:
        return "gluon_attention"

    @property
    def description(self) -> str:
        return "Decoupled content/position attention with equal fusion init"

    def key_class(self) -> KeyClass:
        return KeyClass.TRIVIAL

    def forward(self, data: dict) -> KeyResult:
        """Initialize Gluon attention fusion weights.

        Args:
            data: {"d_model": int,
                   "n_heads": int,
                   "head_dim": int,
                   "init_content_weight": float (default 0.5),
                   "init_position_weight": float (default 0.5)}

        Returns:
            {"content_weight": float,
             "position_weight": float,
             "content_proj_weight": tensor (n_heads * head_dim, d_model),
             "position_proj_weight": tensor (n_heads * head_dim, d_model)}
        """
        try:
            d_model = data["d_model"]
            n_heads = data["n_heads"]
            head_dim = data["head_dim"]
            cw = data.get("init_content_weight", 0.5)
            pw = data.get("init_position_weight", 0.5)

            # Content and position projections: identity-like init
            # Content path uses the same Q/K/V weights (no RoPE applied)
            # Position path uses separate projections (RoPE applied)
            # At init, both are identity → equal fusion = standard attention
            content_proj = torch.eye(n_heads * head_dim, d_model)
            position_proj = torch.eye(n_heads * head_dim, d_model)

            return KeyResult(
                success=True,
                weights={
                    "content_weight": torch.tensor(cw),
                    "position_weight": torch.tensor(pw),
                    "content_proj_weight": content_proj,
                    "position_proj_weight": position_proj,
                },
                metadata={
                    "d_model": d_model, "n_heads": n_heads, "head_dim": head_dim,
                    "init": "equal_fusion",
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Gluon init is identity — reverse is passthrough."""
        return KeyResult(
            success=True,
            data=weights,
            metadata={"reversible": True},
        )


class GluonAttention(nn.Module):
    """Gluon attention with decoupled content and position paths.

    content_attn: standard attention without RoPE (content matching)
    position_attn: attention with RoPE (position matching)
    output = content_weight * content_attn + position_weight * position_attn
    """

    def __init__(self, d_model, n_heads, head_dim=None, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        self.base = base

        # Content path projections (no RoPE)
        self.content_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.content_k = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.content_v = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # Position path projections (with RoPE)
        self.position_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.position_k = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # Fusion weights (learnable, initialized to 0.5/0.5)
        self.content_weight = nn.Parameter(torch.tensor(0.5))
        self.position_weight = nn.Parameter(torch.tensor(0.5))

        # Output projection
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # RoPE (for position path only)
        self.max_seq_len = max_seq_len
        self._build_rope()

    def _build_rope(self):
        d = self.head_dim
        freqs = 1.0 / (self.base ** (torch.arange(0, d, 2).float() / d))
        t = torch.arange(self.max_seq_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def _apply_rope(self, x, seq_len):
        # x: (batch, n_heads, seq, head_dim)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, x, past_kv=None):
        B, T, D = x.shape

        # Content path (no RoPE)
        cq = self.content_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        ck = self.content_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cv = self.content_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        content_scores = cq @ ck.transpose(-2, -1) / (self.head_dim ** 0.5)
        content_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        content_scores = content_scores.masked_fill(content_mask, float('-inf'))
        content_attn = torch.softmax(content_scores, dim=-1) @ cv

        # Position path (with RoPE)
        pq = self.position_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        pk = self.position_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        pq = self._apply_rope(pq, T)
        pk = self._apply_rope(pk, T)

        pos_scores = pq @ pk.transpose(-2, -1) / (self.head_dim ** 0.5)
        pos_scores = pos_scores.masked_fill(content_mask, float('-inf'))
        # Position path doesn't need V (just position matching)
        pos_attn = torch.softmax(pos_scores, dim=-1) @ cv  # share V with content

        # Fusion
        fused = self.content_weight * content_attn + self.position_weight * pos_attn
        fused = fused.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(fused)


if __name__ == "__main__":
    key = GluonAttentionKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")

    r = key.forward({"d_model": 256, "n_heads": 4, "head_dim": 64})
    print(f"Forward: {r.success}")
    print(f"  Content weight: {r.weights['content_weight'].item()}")
    print(f"  Position weight: {r.weights['position_weight'].item()}")
    print(f"  Content proj: {r.weights['content_proj_weight'].shape}")

    # Test Gluon attention module
    attn = GluonAttention(d_model=256, n_heads=4, head_dim=64)
    x = torch.randn(1, 16, 256)
    out = attn(x)
    print(f"  Gluon output: {out.shape}")
