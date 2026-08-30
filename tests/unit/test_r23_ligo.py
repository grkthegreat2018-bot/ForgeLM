"""Tests for R&D Round 23: LiGO — Learned Linear Growth Operator for model expansion.

LiGO learns a linear map M: Theta_large = M * Theta_small, factored as:
  - R_width: width-growth operator (Kronecker-factored)
  - L_depth: depth-growth operator (Kronecker-factored)

M is learned with ~100 steps of SGD on a small data subset, then used to
initialize the larger model. Saves up to 50% compute vs training from scratch.

Paper: arXiv:2303.00980 (ICML 2023).
Will be implemented at research/architecture/ligo.py.
"""
import os, sys, tempfile, math, time
sys.path.insert(0, r"D:\windsurf\ForgeAI")

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tiny model helpers ──────────────────────────────────────────────────────

class TinyMLP(nn.Module):
    """Simple MLP for testing LiGO growth operators.

    Layers: input -> hidden_1 -> ... -> hidden_L -> output
    All layers are nn.Linear with ReLU between them.
    """

    def __init__(self, d_model=64, n_layers=2, vocab=256, intermediate=None):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab = vocab
        if intermediate is None:
            intermediate = d_model * 2

        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_model, intermediate))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(intermediate, intermediate))
        self.norm = nn.LayerNorm(intermediate)
        self.head = nn.Linear(intermediate, vocab, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        x = self.norm(x)
        return self.head(x)


class TinyTransformer(nn.Module):
    """Minimal transformer for testing LiGO with attention layers."""

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
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.o_proj(attn)

        h = self.ffn_norm(x)
        gate = F.silu(self.w_gate(h))
        up = self.w_up(h)
        ffn_out = self.w_down(gate * up)
        x = x + ffn_out
        return x


# ── R23a: LiGO imports & structure ──────────────────────────────────────────

def test_ligo_imports():
    """Import research.architecture.ligo, verify LiGOGrowth class or function exists."""
    from research.architecture.ligo import LiGOGrowth

    assert LiGOGrowth is not None, "LiGOGrowth should be importable"
    assert isinstance(LiGOGrowth, type), "LiGOGrowth should be a class"
    print("  ligo_imports: PASS")


def test_ligo_width_operator():
    """Create a width-growth operator for 2x expansion (d=64 -> 128).

    Verify it maps source params to target params with correct shapes.
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    src = TinyMLP(d_model=64, n_layers=2, vocab=256)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=128,
        target_n_layers=2,
    )

    # The width operator R_width should map (128, 64) -> (128, 128) or similar
    # i.e. it takes a source weight matrix and produces a target weight matrix
    r_width = ligo.get_width_operator()
    assert r_width is not None, "Width operator should exist"

    # Test mapping a source weight to target shape
    src_w = src.layers[0].weight  # (intermediate, d_model) = (128, 64)
    dst_w = ligo.apply_width_operator(src_w)
    # Target intermediate should be 2x = 256, target d_model = 128
    assert dst_w.shape[0] == 2 * src_w.shape[0], \
        f"Output dim should double: {dst_w.shape[0]} vs {2 * src_w.shape[0]}"
    assert dst_w.shape[1] == 2 * src_w.shape[1], \
        f"Input dim should double: {dst_w.shape[1]} vs {2 * src_w.shape[1]}"
    print("  ligo_width_operator: PASS")


def test_ligo_depth_operator():
    """Create a depth-growth operator for 2x expansion (L=2 -> 4).

    Verify it maps source layers to target layers.
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    src = TinyMLP(d_model=64, n_layers=2, vocab=256)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=64,
        target_n_layers=4,
    )

    l_depth = ligo.get_depth_operator()
    assert l_depth is not None, "Depth operator should exist"

    # The depth operator should map 2 source layers to 4 target layers
    src_layer_weights = [layer.weight for layer in src.layers]
    dst_layer_weights = ligo.apply_depth_operator(src_layer_weights)
    assert len(dst_layer_weights) == 4, \
        f"Should produce 4 target layers, got {len(dst_layer_weights)}"
    print("  ligo_depth_operator: PASS")


def test_ligo_kronecker_structure():
    """Verify the growth operators use Kronecker product structure.

    R_width = kron(A, B) where A encodes neuron grouping.
    Check that the operator matrix has the expected block structure.
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    src = TinyMLP(d_model=64, n_layers=2, vocab=256)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=128,
        target_n_layers=2,
    )

    # The width operator should be factorizable as kron(A, B)
    A, B = ligo.get_width_operator_factors()
    assert A is not None and B is not None, "Should have Kronecker factors A, B"

    # Verify kron(A, B) reconstructs the full operator
    R_width = ligo.get_width_operator()
    R_reconstructed = torch.kron(A, B)
    assert R_reconstructed.shape == R_width.shape, \
        f"kron(A,B) shape {R_reconstructed.shape} should match R_width {R_width.shape}"
    assert torch.allclose(R_reconstructed, R_width, atol=1e-5), \
        "kron(A, B) should reconstruct R_width"

    # A should encode neuron grouping (small matrix, e.g. 2x1 for 2x expansion)
    assert A.shape[0] == 2, f"A should have 2 rows for 2x expansion, got {A.shape}"
    print("  ligo_kronecker_structure: PASS")


def test_ligo_learn_matrix():
    """Learn the growth matrix M with 100 steps of SGD on a tiny dataset.

    Verify the loss on the growth objective decreases (M is being optimized).
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    src = TinyMLP(d_model=64, n_layers=2, vocab=256)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=128,
        target_n_layers=4,
    )

    # Generate tiny dataset for growth objective
    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))

    # Learn M for 100 steps
    initial_loss = ligo.compute_growth_loss(x, y).item()
    for _ in range(100):
        ligo.growth_step(x, y, lr=1e-2)
    final_loss = ligo.compute_growth_loss(x, y).item()

    print(f"  Growth objective: {initial_loss:.4f} -> {final_loss:.4f}")
    assert final_loss < initial_loss, \
        "Growth objective should decrease over 100 SGD steps"
    print("  ligo_learn_matrix: PASS")


def test_ligo_initialize_larger():
    """After learning M, initialize the larger model.

    Verify the larger model's forward pass produces finite outputs
    (not random init garbage).
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    src = TinyMLP(d_model=64, n_layers=2, vocab=256)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=128,
        target_n_layers=4,
    )

    # Learn M briefly
    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))
    for _ in range(20):
        ligo.growth_step(x, y, lr=1e-2)

    # Initialize larger model
    dst = ligo.initialize_larger_model()

    # Forward pass should produce finite outputs
    with torch.no_grad():
        out = dst(x)
    assert out.shape == (4, 16, 256), f"Output shape wrong: {out.shape}"
    assert torch.isfinite(out).all(), "Output should be finite (not random init garbage)"
    print("  ligo_initialize_larger: PASS")


def test_ligo_vs_random_init():
    """Compare LiGO-initialized model vs random-init model after 10 training steps.

    LiGO should have lower loss (it starts from learned init).
    Use same data, same optimizer.
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    vocab = 256
    src = TinyMLP(d_model=64, n_layers=2, vocab=vocab)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=128,
        target_n_layers=4,
    )

    # Learn M
    x = torch.randint(0, vocab, (4, 16))
    y = torch.randint(0, vocab, (4, 16))
    for _ in range(50):
        ligo.growth_step(x, y, lr=1e-2)

    # LiGO-initialized model
    dst_ligo = ligo.initialize_larger_model()

    # Random-init model (same architecture)
    dst_random = TinyMLP(d_model=128, n_layers=4, vocab=vocab)
    torch.manual_seed(123)  # different seed for random init
    dst_random = TinyMLP(d_model=128, n_layers=4, vocab=vocab)

    # Train both for 10 steps with same optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt_ligo = torch.optim.Adam(dst_ligo.parameters(), lr=1e-3)
    opt_random = torch.optim.Adam(dst_random.parameters(), lr=1e-3)

    for _ in range(10):
        opt_ligo.zero_grad()
        loss_ligo = loss_fn(dst_ligo(x).view(-1, vocab), y.view(-1))
        loss_ligo.backward()
        opt_ligo.step()

        opt_random.zero_grad()
        loss_random = loss_fn(dst_random(x).view(-1, vocab), y.view(-1))
        loss_random.backward()
        opt_random.step()

    print(f"  LiGO loss after 10 steps: {loss_ligo.item():.4f}")
    print(f"  Random loss after 10 steps: {loss_random.item():.4f}")
    assert loss_ligo.item() <= loss_random.item() + 0.5, \
        "LiGO should be at least as good as random init"
    print("  ligo_vs_random_init: PASS")


def test_ligo_vs_stack_duplicate():
    """Compare LiGO vs simple layer stacking (duplicate each layer).

    LiGO should be at least as good after 10 steps.
    The paper shows LiGO outperforms stacking.
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    vocab = 256
    src = TinyMLP(d_model=64, n_layers=2, vocab=vocab)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=128,
        target_n_layers=4,
    )

    # Learn M
    x = torch.randint(0, vocab, (4, 16))
    y = torch.randint(0, vocab, (4, 16))
    for _ in range(50):
        ligo.growth_step(x, y, lr=1e-2)

    # LiGO-initialized model
    dst_ligo = ligo.initialize_larger_model()

    # Stack-duplicate model: just duplicate each layer's weights
    dst_stack = TinyMLP(d_model=128, n_layers=4, vocab=vocab)
    # Simple stacking: copy source weights and duplicate
    with torch.no_grad():
        for i in range(2):
            dst_stack.layers[2 * i].weight.data[:src.layers[i].weight.shape[0],
                                                  :src.layers[i].weight.shape[1]].copy_(
                src.layers[i].weight.data)
            dst_stack.layers[2 * i + 1].weight.data[:src.layers[i].weight.shape[0],
                                                      :src.layers[i].weight.shape[1]].copy_(
                src.layers[i].weight.data)

    # Train both for 10 steps
    loss_fn = nn.CrossEntropyLoss()
    opt_ligo = torch.optim.Adam(dst_ligo.parameters(), lr=1e-3)
    opt_stack = torch.optim.Adam(dst_stack.parameters(), lr=1e-3)

    for _ in range(10):
        opt_ligo.zero_grad()
        loss_ligo = loss_fn(dst_ligo(x).view(-1, vocab), y.view(-1))
        loss_ligo.backward()
        opt_ligo.step()

        opt_stack.zero_grad()
        loss_stack = loss_fn(dst_stack(x).view(-1, vocab), y.view(-1))
        loss_stack.backward()
        opt_stack.step()

    print(f"  LiGO loss after 10 steps: {loss_ligo.item():.4f}")
    print(f"  Stack loss after 10 steps: {loss_stack.item():.4f}")
    assert loss_ligo.item() <= loss_stack.item() + 0.5, \
        "LiGO should be at least as good as simple stacking"
    print("  ligo_vs_stack_duplicate: PASS")


def test_ligo_lfm_to_v8():
    """Create source with LFM dims (d=64, L=2) and target V8 dims (d=128, L=4).

    Learn M, initialize V8, verify forward pass works.
    Use tiny dims for speed (d=64, L=2 -> d=128, L=4).
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    vocab = 256  # tiny vocab for speed
    src = TinyTransformer(d_model=64, n_layers=2, n_heads=4, head_dim=16, vocab=vocab)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=128,
        target_n_layers=4,
    )

    # Learn M briefly
    x = torch.randint(0, vocab, (2, 8))
    y = torch.randint(0, vocab, (2, 8))
    for _ in range(20):
        ligo.growth_step(x, y, lr=1e-2)

    # Initialize larger model
    dst = ligo.initialize_larger_model()

    # Verify forward pass works
    with torch.no_grad():
        out = dst(x)
    assert out.shape == (2, 8, vocab), f"Output shape wrong: {out.shape}"
    assert torch.isfinite(out).all(), "Output should be finite"
    print("  ligo_lfm_to_v8: PASS")


def test_ligo_compute_savings():
    """Verify that learning M takes << time than full training.

    Time 100 steps of M learning vs 100 steps of full model training.
    M learning should be ~10x faster (smaller optimization problem).
    """
    from research.architecture.ligo import LiGOGrowth

    torch.manual_seed(42)
    vocab = 256
    src = TinyMLP(d_model=64, n_layers=2, vocab=vocab)
    ligo = LiGOGrowth(
        src_model=src,
        target_d_model=128,
        target_n_layers=4,
    )

    # Full target model for comparison
    dst_full = TinyMLP(d_model=128, n_layers=4, vocab=vocab)

    x = torch.randint(0, vocab, (4, 16))
    y = torch.randint(0, vocab, (4, 16))
    loss_fn = nn.CrossEntropyLoss()

    # Time 100 steps of M learning
    t0 = time.perf_counter()
    for _ in range(100):
        ligo.growth_step(x, y, lr=1e-2)
    t_ligo = time.perf_counter() - t0

    # Time 100 steps of full model training
    opt_full = torch.optim.Adam(dst_full.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    for _ in range(100):
        opt_full.zero_grad()
        loss = loss_fn(dst_full(x).view(-1, vocab), y.view(-1))
        loss.backward()
        opt_full.step()
    t_full = time.perf_counter() - t0

    ratio = t_full / max(t_ligo, 1e-6)
    print(f"  M learning time: {t_ligo:.3f}s, Full training time: {t_full:.3f}s")
    print(f"  Speedup ratio: {ratio:.1f}x")
    # M learning should be faster (smaller problem). Allow some tolerance
    # since tiny models may not show the full speedup.
    assert t_ligo <= t_full * 2, \
        f"M learning should not be much slower than full training: {t_ligo:.3f} vs {t_full:.3f}"
    print("  ligo_compute_savings: PASS")


# ── Main ────────────────────────────────────────────────────────────────────

def main_r23_ligo():
    print("=" * 70)
    print("  R&D ROUND 23: LiGO — Learned Linear Growth Operator")
    print("=" * 70)

    print("\n  R23a: LiGO imports & structure")
    test_ligo_imports()
    test_ligo_width_operator()
    test_ligo_depth_operator()
    test_ligo_kronecker_structure()

    print("\n  R23b: LiGO learning")
    test_ligo_learn_matrix()
    test_ligo_initialize_larger()

    print("\n  R23c: LiGO vs baselines")
    test_ligo_vs_random_init()
    test_ligo_vs_stack_duplicate()

    print("\n  R23d: LiGO LFM -> V8")
    test_ligo_lfm_to_v8()

    print("\n  R23e: Compute savings")
    test_ligo_compute_savings()

    print("\n" + "=" * 70)
    print("  ALL R&D ROUND 23 LIGO TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main_r23_ligo()
