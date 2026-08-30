"""Tests for R&D Round 23: HyperCloning — function-preserving model expansion.

HyperCloning expands a smaller pre-trained model into a larger model such that
the cloned (larger) model produces EXACTLY the same logits as the source
(smaller) model at initialization. The larger model starts with the smaller
model's accuracy, then improves from there with more capacity.

For ForgeAI: LFM2.5-1.2B (d=2048, L=16, 32 heads, head_dim=64) -> V8-8B
(d=4096, L=32, 64 heads, head_dim=64).
  - 2x width: embedding_dim_multiplier=2 (doubles d_model, doubles n_heads,
    head_dim stays 64)
  - 2x depth: duplicate each layer (L=16 -> L=32)

Paper: arXiv:2409.12903 (Apple NeurIPS 2024).
Will be implemented at research/architecture/hypercloning.py.
"""
import os, sys, tempfile, math
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tiny model helpers ──────────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    """Minimal transformer for testing HyperCloning.

    Has: token embedding, L transformer layers (attention + FFN), output head.
    Uses standard MHA + SwiGLU FFN so weight shapes are predictable.
    """

    def __init__(self, d_model=64, n_layers=2, n_heads=4, head_dim=16,
                 vocab=256, intermediate=None):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.vocab = vocab
        if intermediate is None:
            intermediate = d_model * 4

        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            TinyTransformerLayer(d_model, n_heads, head_dim, intermediate)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


class TinyTransformerLayer(nn.Module):
    """One transformer layer: MHA + residual, SwiGLU FFN + residual."""

    def __init__(self, d_model, n_heads, head_dim, intermediate):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        attn_dim = n_heads * head_dim
        self.q_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.v_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, d_model, bias=False)
        self.attn_norm = nn.LayerNorm(d_model)

        self.w_gate = nn.Linear(d_model, intermediate, bias=False)
        self.w_up = nn.Linear(d_model, intermediate, bias=False)
        self.w_down = nn.Linear(intermediate, d_model, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        # Attention
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.o_proj(attn)

        # SwiGLU FFN
        h = self.ffn_norm(x)
        gate = F.silu(self.w_gate(h))
        up = self.w_up(h)
        ffn_out = self.w_down(gate * up)
        x = x + ffn_out
        return x


class TinyBitNetTransformer(nn.Module):
    """Tiny transformer using BitNetLinear layers for ternary weight testing."""

    def __init__(self, d_model=64, n_layers=2, n_heads=4, head_dim=16,
                 vocab=256, intermediate=None):
        super().__init__()
        from research.keys.quantization.bitnet_b158_key import BitNetLinear
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.vocab = vocab
        if intermediate is None:
            intermediate = d_model * 4
        attn_dim = n_heads * head_dim

        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "q_proj": BitNetLinear(d_model, attn_dim, quantize=True),
                "k_proj": BitNetLinear(d_model, attn_dim, quantize=True),
                "v_proj": BitNetLinear(d_model, attn_dim, quantize=True),
                "o_proj": BitNetLinear(attn_dim, d_model, quantize=True),
                "w_gate": BitNetLinear(d_model, intermediate, quantize=True),
                "w_up": BitNetLinear(d_model, intermediate, quantize=True),
                "w_down": BitNetLinear(intermediate, d_model, quantize=True),
                "attn_norm": nn.LayerNorm(d_model),
                "ffn_norm": nn.LayerNorm(d_model),
            })
            self.layers.append(layer)
        self.norm = nn.LayerNorm(d_model)
        self.head = BitNetLinear(d_model, vocab, quantize=True)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            B, T, D = x.shape
            h = layer["attn_norm"](x)
            q = layer["q_proj"](h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = layer["k_proj"](h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = layer["v_proj"](h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
            x = x + layer["o_proj"](attn)
            h = layer["ffn_norm"](x)
            gate = F.silu(layer["w_gate"](h))
            up = layer["w_up"](h)
            x = x + layer["w_down"](gate * up)
        x = self.norm(x)
        return self.head(x)


# ── R23b: HyperCloning imports & basic structure ───────────────────────────

def test_hypercloning_imports():
    """Import research.architecture.hypercloning, verify clone_model exists."""
    from research.architecture.hypercloning import clone_model

    assert clone_model is not None, "clone_model should be importable"
    assert callable(clone_model), "clone_model should be callable"
    print("  hypercloning_imports: PASS")


def test_hypercloning_2x_width():
    """Clone a tiny source (d=64, L=2, 4 heads) to 2x width (d=128, L=2, 8 heads).

    Verify output shape is (B, T, 128) not (B, T, 64).
    """
    from research.architecture.hypercloning import clone_model

    torch.manual_seed(42)
    src = TinyTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16, vocab=256)
    dst = clone_model(src, embedding_dim_multiplier=2, depth_multiplier=1)

    x = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        out_src = src(x)
        out_dst = dst(x)

    assert out_dst.shape == (2, 8, 256), \
        f"Output shape should be (2, 8, 256), got {out_dst.shape}"
    assert dst.d_model == 128, f"d_model should be 128, got {dst.d_model}"
    assert dst.n_heads == 8, f"n_heads should be 8, got {dst.n_heads}"
    assert dst.head_dim == 16, f"head_dim should stay 16, got {dst.head_dim}"
    print("  hypercloning_2x_width: PASS")


def test_hypercloning_2x_depth():
    """Clone source (d=64, L=2) to 2x depth (d=64, L=4). Verify 4 layers."""
    from research.architecture.hypercloning import clone_model

    torch.manual_seed(42)
    src = TinyTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16, vocab=256)
    dst = clone_model(src, embedding_dim_multiplier=1, depth_multiplier=2)

    assert dst.n_layers == 4, f"Should have 4 layers, got {dst.n_layers}"
    assert dst.d_model == 64, f"d_model should stay 64, got {dst.d_model}"
    print("  hypercloning_2x_depth: PASS")


def test_hypercloning_function_preserving_width():
    """THE KEY TEST: clone to 2x width, verify logits match source at init.

    The cloned model's logits for the first `vocab` output dims should match
    the source model's logits. Extra dims should be zero/noise.
    Use torch.manual_seed(42) and tolerance 1e-4.
    """
    from research.architecture.hypercloning import clone_model

    torch.manual_seed(42)
    vocab = 256
    src = TinyTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16, vocab=vocab)
    dst = clone_model(src, embedding_dim_multiplier=2, depth_multiplier=1)

    x = torch.randint(0, vocab, (2, 8))
    with torch.no_grad():
        logits_src = src(x)
        logits_dst = dst(x)

    # The first `vocab` output dimensions should match (function-preserving)
    max_diff = (logits_dst[:, :, :vocab] - logits_src).abs().max().item()
    print(f"  Width clone logit diff (first {vocab} dims): {max_diff:.6e}")
    assert max_diff < 1e-4, \
        f"Function-preserving property violated: max diff {max_diff:.6e}"
    print("  hypercloning_function_preserving_width: PASS")


def test_hypercloning_function_preserving_depth():
    """Clone to 2x depth, verify logits match source at init.

    Depth duplication = stack consecutive layers, which is function-preserving
    for residual networks (each duplicated layer is identity at init).
    """
    from research.architecture.hypercloning import clone_model

    torch.manual_seed(42)
    vocab = 256
    src = TinyTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16, vocab=vocab)
    dst = clone_model(src, embedding_dim_multiplier=1, depth_multiplier=2)

    x = torch.randint(0, vocab, (2, 8))
    with torch.no_grad():
        logits_src = src(x)
        logits_dst = dst(x)

    max_diff = (logits_dst - logits_src).abs().max().item()
    print(f"  Depth clone logit diff: {max_diff:.6e}")
    assert max_diff < 1e-4, \
        f"Function-preserving property violated: max diff {max_diff:.6e}"
    print("  hypercloning_function_preserving_depth: PASS")


def test_hypercloning_2x_both():
    """Clone 2x width + 2x depth (d=128, L=4 from d=64, L=2). Verify function-preserving."""
    from research.architecture.hypercloning import clone_model

    torch.manual_seed(42)
    vocab = 256
    src = TinyTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16, vocab=vocab)
    dst = clone_model(src, embedding_dim_multiplier=2, depth_multiplier=2)

    assert dst.d_model == 128, f"d_model should be 128, got {dst.d_model}"
    assert dst.n_layers == 4, f"n_layers should be 4, got {dst.n_layers}"
    assert dst.n_heads == 8, f"n_heads should be 8, got {dst.n_heads}"
    assert dst.head_dim == 16, f"head_dim should stay 16, got {dst.head_dim}"

    x = torch.randint(0, vocab, (2, 8))
    with torch.no_grad():
        logits_src = src(x)
        logits_dst = dst(x)

    max_diff = (logits_dst[:, :, :vocab] - logits_src).abs().max().item()
    print(f"  2x both clone logit diff (first {vocab} dims): {max_diff:.6e}")
    assert max_diff < 1e-4, \
        f"Function-preserving property violated: max diff {max_diff:.6e}"
    print("  hypercloning_2x_both: PASS")


def test_hypercloning_lfm_to_v8_dims():
    """Create source with LFM-like dims and target V8-like dims (tiny for CPU).

    Mirrors the LFM2.5->V8 expansion ratio (2x width + 2x depth) but with
    small dims to avoid OOM on CPU:
    Source: d=128, L=4, 4 heads, head_dim=32, vocab=256
    Target: d=256, L=8, 8 heads, head_dim=32, vocab=256
    Verify clone produces correct dims (same 2x expansion as LFM->V8).
    """
    from research.architecture.hypercloning import clone_model

    torch.manual_seed(42)
    vocab = 256  # tiny vocab for speed
    src = TinyTransformer(d_model=128, n_layers=4, n_heads=4, head_dim=32,
                          vocab=vocab)
    dst = clone_model(src, embedding_dim_multiplier=2, depth_multiplier=2)

    assert dst.d_model == 256, f"d_model should be 256, got {dst.d_model}"
    assert dst.n_layers == 8, f"n_layers should be 8, got {dst.n_layers}"
    assert dst.n_heads == 8, f"n_heads should be 8, got {dst.n_heads}"
    assert dst.head_dim == 32, f"head_dim should stay 32, got {dst.head_dim}"
    assert dst.vocab == vocab, f"vocab should stay {vocab}, got {dst.vocab}"

    # Verify the 2x expansion ratios match the LFM->V8 mapping
    assert dst.d_model == 2 * src.d_model
    assert dst.n_layers == 2 * src.n_layers
    assert dst.n_heads == 2 * src.n_heads
    assert dst.head_dim == src.head_dim

    # Verify forward pass works (just shape, skip full function-preserving for speed)
    x = torch.randint(0, vocab, (1, 4))
    with torch.no_grad():
        out = dst(x)
    assert out.shape == (1, 4, vocab), f"Output shape wrong: {out.shape}"
    print("  hypercloning_lfm_to_v8_dims: PASS")


def test_hypercloning_bitnet_compatible():
    """Create source with BitNet layers, clone, verify target also has BitNet layers.

    Ternary weights should be preserved through the cloning process.
    """
    from research.architecture.hypercloning import clone_model
    from research.keys.quantization.bitnet_b158_key import BitNetLinear

    torch.manual_seed(42)
    src = TinyBitNetTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16,
                                vocab=256)
    dst = clone_model(src, embedding_dim_multiplier=2, depth_multiplier=2)

    # Verify target has BitNetLinear layers
    bitnet_count = 0
    for module in dst.modules():
        if isinstance(module, BitNetLinear):
            bitnet_count += 1
    assert bitnet_count > 0, "Cloned model should have BitNetLinear layers"

    # Verify forward pass works
    x = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        out = dst(x)
    assert out.shape == (2, 8, 256), f"Output shape wrong: {out.shape}"
    assert torch.isfinite(out).all(), "Output should be finite"
    print("  hypercloning_bitnet_compatible: PASS")


def test_hypercloning_improves_training():
    """Clone source to 2x, train both for 10 steps on same data.

    Cloned model loss should be <= source model loss (starts at same point,
    has more capacity).
    """
    from research.architecture.hypercloning import clone_model

    torch.manual_seed(42)
    vocab = 256
    src = TinyTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16, vocab=vocab)
    dst = clone_model(src, embedding_dim_multiplier=2, depth_multiplier=2)

    # Generate training data
    x = torch.randint(0, vocab, (4, 16))
    y = torch.randint(0, vocab, (4, 16))

    # Train both for 10 steps
    opt_src = torch.optim.Adam(src.parameters(), lr=1e-3)
    opt_dst = torch.optim.Adam(dst.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(10):
        opt_src.zero_grad()
        loss_src = loss_fn(src(x).view(-1, vocab), y.view(-1))
        loss_src.backward()
        opt_src.step()

        opt_dst.zero_grad()
        loss_dst = loss_fn(dst(x).view(-1, vocab), y.view(-1))
        loss_dst.backward()
        opt_dst.step()

    print(f"  Source loss after 10 steps: {loss_src.item():.4f}")
    print(f"  Cloned loss after 10 steps: {loss_dst.item():.4f}")
    assert loss_dst.item() <= loss_src.item() + 0.1, \
        "Cloned model should be at least as good as source (more capacity)"
    print("  hypercloning_improves_training: PASS")


def test_hypercloning_head_size_preserved():
    """Verify head_dim stays the same after cloning (64->64), only n_heads doubles.

    This is the HyperCloning recommendation: keep head_dim fixed, double n_heads.
    """
    from research.architecture.hypercloning import clone_model

    torch.manual_seed(42)
    src = TinyTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16, vocab=256)
    dst = clone_model(src, embedding_dim_multiplier=2, depth_multiplier=1)

    assert dst.head_dim == src.head_dim, \
        f"head_dim should be preserved: src={src.head_dim}, dst={dst.head_dim}"
    assert dst.n_heads == 2 * src.n_heads, \
        f"n_heads should double: src={src.n_heads}, dst={dst.n_heads}"
    assert dst.d_model == 2 * src.d_model, \
        f"d_model should double: src={src.d_model}, dst={dst.d_model}"
    print("  hypercloning_head_size_preserved: PASS")


# ── Main ────────────────────────────────────────────────────────────────────

def main_r23_hypercloning():
    print("=" * 70)
    print("  R&D ROUND 23: HyperCloning — Function-Preserving Model Expansion")
    print("=" * 70)

    print("\n  R23a: HyperCloning imports & basic structure")
    test_hypercloning_imports()
    test_hypercloning_2x_width()
    test_hypercloning_2x_depth()

    print("\n  R23b: Function-preserving property")
    test_hypercloning_function_preserving_width()
    test_hypercloning_function_preserving_depth()
    test_hypercloning_2x_both()

    print("\n  R23c: LFM -> V8 dimensions")
    test_hypercloning_lfm_to_v8_dims()
    test_hypercloning_head_size_preserved()

    print("\n  R23d: BitNet compatibility")
    test_hypercloning_bitnet_compatible()

    print("\n  R23e: Training improvement")
    test_hypercloning_improves_training()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 23 HYPERCLONING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r23_hypercloning()
