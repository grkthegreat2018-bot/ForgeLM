"""Tests for research.moe — Router gating, MoELayer forward, dense bypass.

CPU-only with tiny dims (d_model=64) to stay fast and GPU-independent.
"""

import torch

from research.moe import MoELayer, Router

# ── Router ───────────────────────────────────────────────────────────────────


class TestRouter:
    def test_topk_gating_shapes(self):
        router = Router(d_model=64, n_experts=4, top_k=2, noisy_gating=False)
        router.eval()
        x = torch.randn(2, 8, 64)  # (B, T, d_model) → N = 16 tokens
        dispatch_mask, gating_weights, aux_loss = router(x)
        assert dispatch_mask.shape == (16, 4)
        assert gating_weights.shape == (16, 4)
        assert aux_loss.dim() == 0  # scalar

    def test_topk_exactly_k_experts_per_token(self):
        router = Router(d_model=64, n_experts=4, top_k=2, noisy_gating=False)
        router.eval()
        x = torch.randn(16, 64)
        dispatch_mask, gating_weights, _ = router(x)
        # Each token dispatched to exactly top_k experts
        assert torch.allclose(dispatch_mask.sum(dim=1), torch.full((16,), 2.0))
        # Gating weights are zero for non-routed experts
        assert torch.all((gating_weights > 0) == (dispatch_mask > 0))

    def test_topk_clamped_to_n_experts(self):
        router = Router(d_model=64, n_experts=2, top_k=4, noisy_gating=False)
        assert router.top_k == 2

    def test_load_balancing_loss_positive(self):
        router = Router(d_model=64, n_experts=4, top_k=2, noisy_gating=False)
        router.eval()
        x = torch.randn(32, 64)
        _, _, aux_loss = router(x)
        assert aux_loss.item() > 0

    def test_flat_input_matches_batched(self):
        router = Router(d_model=64, n_experts=4, top_k=2, noisy_gating=False)
        router.eval()
        x = torch.randn(2, 8, 64)
        d_b, w_b, _ = router(x)
        d_f, w_f, _ = router(x.view(-1, 64))
        assert torch.allclose(d_b, d_f)
        assert torch.allclose(w_b, w_f)


# ── MoELayer ─────────────────────────────────────────────────────────────────


class TestMoELayer:
    def test_forward_shape(self):
        moe = MoELayer(d_model=64, n_experts=4, top_k=2, noisy_gating=False)
        moe.eval()
        x = torch.randn(2, 8, 64)
        out, aux_loss = moe(x)
        assert out.shape == (2, 8, 64)
        assert aux_loss.dim() == 0

    def test_forward_records_aux_loss(self):
        moe = MoELayer(d_model=64, n_experts=4, top_k=2, noisy_gating=False)
        moe.eval()
        x = torch.randn(1, 4, 64)
        _, aux_loss = moe(x)
        assert hasattr(moe, "_last_aux_loss")
        assert torch.allclose(moe._last_aux_loss, aux_loss)

    def test_no_shared_expert(self):
        moe = MoELayer(d_model=64, n_experts=4, top_k=2,
                       shared_expert=False, noisy_gating=False)
        moe.eval()
        x = torch.randn(2, 4, 64)
        out, _ = moe(x)
        assert out.shape == (2, 4, 64)
        assert not hasattr(moe, "shared")

    def test_dense_bypass_shape_and_zero_aux(self):
        moe = MoELayer(d_model=64, n_experts=4, top_k=2,
                       dense_bypass=True, noisy_gating=False)
        moe.eval()
        x = torch.randn(2, 8, 64)
        out, aux_loss = moe(x)
        assert out.shape == (2, 8, 64)
        assert aux_loss.item() == 0.0

    def test_dense_bypass_equals_uniform_expert_mean(self):
        # dense_bypass runs ALL experts with equal weight 1/n (+ shared expert)
        moe = MoELayer(d_model=64, n_experts=4, top_k=2,
                       dense_bypass=True, noisy_gating=False)
        moe.eval()
        x = torch.randn(2, 8, 64)
        out, _ = moe(x)
        x_flat = x.view(-1, 64)
        with torch.no_grad():
            expected = torch.stack([e(x_flat) for e in moe.experts]).mean(dim=0)
            expected = expected + moe.shared(x_flat)
        assert torch.allclose(out.view(-1, 64), expected, atol=1e-5)
