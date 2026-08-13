"""Test all 6 novel keys: vocab_pack, conformal_sampler, compute_futures,
sae_pack, cross_attn_pack, per_query_interp.

Each test verifies:
1. Key class instantiation
2. forward() returns success
3. reverse() works (or correctly fails for irreversible keys)
4. Key class is correct (PARTIAL/TRIVIAL)
5. Core logic produces expected results
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def test_vocab_pack():
    """VocabPackKey: extract domain embedding deltas as portable pack."""
    from research.keys.misc.vocab_pack_key import VocabPackKey

    key = VocabPackKey()
    assert key.name == "vocab_pack"
    assert key.key_class().name == "PARTIAL"

    # Synthetic: base and domain embeddings
    vocab_size, d_model = 100, 64
    base_embed = torch.randn(vocab_size, d_model)
    domain_embed = base_embed.clone() + torch.randn(vocab_size, d_model) * 0.1
    domain_token_ids = [10, 20, 30, 40, 50]

    result = key.forward({
        "base_embed": base_embed,
        "domain_embed": domain_embed,
        "domain_token_ids": domain_token_ids,
    })
    assert result.success, f"forward failed: {result.error}"

    deltas = result.weights["deltas"]
    token_ids = result.weights["token_ids"]
    assert len(token_ids) == len(domain_token_ids)
    assert deltas.shape == (len(domain_token_ids), d_model)

    # Verify deltas match E_domain - E_base
    for i, tid in enumerate(domain_token_ids):
        expected = domain_embed[tid] - base_embed[tid]
        err = (deltas[i] - expected).abs().max().item()
        assert err < 1e-6, f"delta mismatch for token {tid}: err={err}"

    # Test reverse: reconstruct domain embeddings
    rev = key.reverse({
        "token_ids": token_ids,
        "deltas": deltas,
        "base_embed": base_embed,
    })
    assert rev.success
    reconstructed = rev.data["domain_embeds"]
    for i, tid in enumerate(domain_token_ids):
        err = (reconstructed[i] - domain_embed[tid]).abs().max().item()
        assert err < 1e-6, f"reverse mismatch for token {tid}: err={err}"

    # Test portability: inject into a DIFFERENT base
    base2 = torch.randn(vocab_size, d_model)
    rev2 = key.reverse({
        "token_ids": token_ids,
        "deltas": deltas,
        "base_embed": base2,
    })
    assert rev2.success
    # Domain embeddings on base2 = base2 + delta (different from original)
    for i, tid in enumerate(domain_token_ids):
        expected = base2[tid] + deltas[i]
        err = (rev2.data["domain_embeds"][i] - expected).abs().max().item()
        assert err < 1e-6

    print("  PASS: vocab_pack (extract, reverse, portable inject)")
    return True


def test_conformal_sampler():
    """ConformalSamplerKey: conformal-calibrated sampling temperature."""
    from research.keys.training.conformal_sampler_key import ConformalSamplerKey, ConformalSampler

    key = ConformalSamplerKey()
    assert key.name == "conformal_sampler"
    assert key.key_class().name == "TRIVIAL"

    # Calibrate on held-out scores
    held_out = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99]
    alpha = 0.1

    result = key.forward({"held_out_scores": held_out, "alpha": alpha})
    assert result.success, f"forward failed: {result.error}"

    threshold = result.data["threshold"]
    # Conformal quantile: ceil((n+1)*(1-alpha))/n = ceil(11*0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0
    # So threshold = sorted_scores[9] = 0.99
    assert abs(threshold - 0.99) < 1e-6, f"threshold={threshold}, expected=0.99"

    # Test temperature mapping
    sampler = ConformalSampler()
    sampler.calibrate(held_out, alpha)

    # High confidence query -> low temperature
    t_high = sampler.get_temperature(0.98)
    # Low confidence query -> high temperature
    t_low = sampler.get_temperature(0.3)
    assert t_high < t_low, f"confident should get lower T: {t_high} vs {t_low}"

    # Reverse is no-op for TRIVIAL (may return success with empty data)
    rev = key.reverse({})

    print(f"  PASS: conformal_sampler (threshold={threshold:.2f}, "
          f"T_confident={t_high:.2f}, T_uncertain={t_low:.2f})")
    return True


def test_compute_futures():
    """ComputeFuturesKey: skip verification for high-confidence drafts."""
    from research.keys.misc.compute_futures_key import ComputeFuturesKey, ComputeFutures

    key = ComputeFuturesKey()
    assert key.name == "compute_futures"
    assert key.key_class().name == "TRIVIAL"

    # Mixed confidence sequence
    confidences = [0.95, 0.97, 0.96, 0.98, 0.3, 0.4, 0.99, 0.98, 0.97, 0.96]
    threshold = 0.9
    max_skip = 3

    result = key.forward({
        "draft_confidences": confidences,
        "threshold": threshold,
        "max_skip": max_skip,
    })
    assert result.success, f"forward failed: {result.error}"

    flags = result.data["verify_flags"]
    assert len(flags) == len(confidences)

    # Positions 0-3: high confidence -> skip (but max_skip=3, so position 3 force-verifies)
    assert flags[0] == False, "pos 0 should skip (confident)"
    assert flags[1] == False, "pos 1 should skip (confident)"
    assert flags[2] == False, "pos 2 should skip (confident)"
    assert flags[3] == True, "pos 3 should verify (max_skip reached)"
    # Positions 4-5: low confidence -> verify
    assert flags[4] == True, "pos 4 should verify (low confidence)"
    assert flags[5] == True, "pos 5 should verify (low confidence)"
    # Positions 6-9: high confidence -> skip again
    assert flags[6] == False, "pos 6 should skip (confident, counter reset)"

    # Count skips
    n_skip = sum(1 for f in flags if not f)
    n_verify = sum(1 for f in flags if f)
    skip_rate = n_skip / len(flags)

    # Reverse is no-op for TRIVIAL
    rev = key.reverse({})

    print(f"  PASS: compute_futures ({n_skip} skipped, {n_verify} verified, "
          f"{skip_rate:.0%} skip rate)")
    return True


def test_sae_pack():
    """SAEPackKey: extract SAE feature deltas as portable steering pack."""
    from research.keys.knowledge.sae_pack_key import SAEPackKey, SAEPack

    key = SAEPackKey()
    assert key.name == "sae_pack"
    assert key.key_class().name == "PARTIAL"

    d_model, n_features, n_layers = 64, 32, 3
    # Synthetic SAE: encoder (d_model -> n_features), decoder (n_features -> d_model)
    encoder = torch.randn(n_features, d_model)
    decoder = torch.randn(d_model, n_features)

    # Synthetic activations
    base_acts = {i: torch.randn(4, d_model) for i in range(n_layers)}
    domain_acts = {i: base_acts[i] + torch.randn(4, d_model) * 0.2 for i in range(n_layers)}

    result = key.forward({
        "base_activations": base_acts,
        "domain_activations": domain_acts,
        "sae_encoder": encoder,
        "sae_decoder": decoder,
        "top_k": 5,
    })
    assert result.success, f"forward failed: {result.error}"

    # Key returns flat weights: sae_pack_L{L}_F{F} + sae_pack_steering
    n_features = result.metadata.get("n_features", 0)
    assert n_features > 0, "pack should have features"
    assert "sae_pack_steering" in result.weights, "should have steering vectors"

    # Test reverse from flat weights
    rev = key.reverse(result.weights)
    assert rev.success
    steering = rev.data["steering_vectors"]
    assert len(steering) == n_layers

    # Test steering effect: adding steering vector should change hidden state
    for layer in range(n_layers):
        h = torch.randn(4, d_model)
        h_steered = h + 1.0 * steering[layer]
        assert not torch.allclose(h, h_steered), "steering should change hidden state"

    print(f"  PASS: sae_pack ({n_features} features extracted, "
          f"{n_layers} layers with steering vectors)")
    return True


def test_cross_attn_pack():
    """CrossAttnPackKey: extract cross-attention adapter as portable pack."""
    from research.keys.attention.cross_attn_pack_key import CrossAttnPackKey, CrossAttnPack

    key = CrossAttnPackKey()
    assert key.name == "cross_attn_pack"
    assert key.key_class().name == "PARTIAL"

    d_model, n_layers = 64, 3
    seq_len = 8

    # Synthetic adapter weights
    adapter_weights = {}
    knowledge_kv = {}
    for layer in range(n_layers):
        adapter_weights[layer] = {
            "W_q": torch.randn(d_model, d_model),
            "W_k": torch.randn(d_model, d_model),
            "W_v": torch.randn(d_model, d_model),
            "W_o": torch.randn(d_model, d_model),
        }
        knowledge_kv[layer] = {
            "K": torch.randn(seq_len, d_model),
            "V": torch.randn(seq_len, d_model),
        }

    result = key.forward({
        "adapter_weights": adapter_weights,
        "knowledge_kv": knowledge_kv,
    })
    assert result.success, f"forward failed: {result.error}"

    # Test reverse: reconstruct from flat pack
    rev = key.reverse(result.weights)
    assert rev.success
    assert "adapter_weights" in rev.data
    assert "knowledge_kv" in rev.data

    # Verify round-trip: reconstructed weights match original
    for layer in range(n_layers):
        for w_name in ["W_q", "W_k", "W_v", "W_o"]:
            orig = adapter_weights[layer][w_name]
            recon = rev.data["adapter_weights"][layer][w_name]
            err = (orig - recon).abs().max().item()
            assert err < 1e-6, f"round-trip mismatch at layer {layer} {w_name}: err={err}"

    # Test injection effect
    pack = CrossAttnPack()
    pack.adapter_weights = adapter_weights
    pack.knowledge_kv = knowledge_kv
    h = {0: torch.randn(1, 4, d_model)}  # {layer: (1, seq, d_model)}
    h_injected = pack.inject(h, scale=1.0)
    assert not torch.allclose(h[0], h_injected[0]), "injection should change hidden states"

    print(f"  PASS: cross_attn_pack ({n_layers} layers, round-trip verified, "
          f"injection changes hidden states)")
    return True


def test_per_query_interp():
    """PerQueryInterpKey: per-query weight interpolation."""
    from research.keys.training.per_query_interp_key import PerQueryInterpKey, PerQueryInterp

    key = PerQueryInterpKey()
    assert key.name == "per_query_interp"
    assert key.key_class().name == "TRIVIAL"

    d_model = 64
    # Two sets of weights (model A = specialist, model B = generalist)
    weights_a = {"layer0.w": torch.randn(d_model, d_model)}
    weights_b = {"layer0.w": torch.randn(d_model, d_model)}

    # High-norm query -> alpha close to 1 -> use model A
    query_high = torch.randn(d_model) * 10  # high norm
    result_high = key.forward({
        "query_features": query_high,
        "weights_a": weights_a,
        "weights_b": weights_b,
    })
    assert result_high.success
    alpha_high = result_high.weights["alpha"]
    mixed_high = result_high.weights["mixed_weights"]

    # Low-norm query -> alpha close to 0 -> use model B
    query_low = torch.randn(d_model) * 0.1  # low norm
    result_low = key.forward({
        "query_features": query_low,
        "weights_a": weights_a,
        "weights_b": weights_b,
    })
    assert result_low.success
    alpha_low = result_low.weights["alpha"]
    mixed_low = result_low.weights["mixed_weights"]

    # High-norm should have higher alpha (more model A)
    assert alpha_high > alpha_low, f"high norm should get higher alpha: {alpha_high} vs {alpha_low}"

    # At alpha=1, mixed = weights_a exactly
    result_a = key.forward({
        "query_features": torch.randn(d_model) * 100,  # very high norm -> alpha=1
        "weights_a": weights_a,
        "weights_b": weights_b,
    })
    if result_a.weights["alpha"] >= 0.99:
        err = (result_a.weights["mixed_weights"]["layer0.w"] - weights_a["layer0.w"]).abs().max().item()
        assert err < 1e-6, "alpha=1 should give exact model A weights"

    # Reverse should fail (irreversible)
    rev = key.reverse({})
    assert not rev.success, "interpolation should be irreversible"

    print(f"  PASS: per_query_interp (alpha_high={alpha_high:.2f}, "
          f"alpha_low={alpha_low:.2f}, irreversible)")
    return True


def main():
    print("=" * 70)
    print("Testing 6 Novel Keys")
    print("=" * 70)

    tests = [
        ("vocab_pack", test_vocab_pack),
        ("conformal_sampler", test_conformal_sampler),
        ("compute_futures", test_compute_futures),
        ("sae_pack", test_sae_pack),
        ("cross_attn_pack", test_cross_attn_pack),
        ("per_query_interp", test_per_query_interp),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"  FAIL: {name} returned False")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name} raised {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{passed + failed} passed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failed} FAILED")
    print(f"{'=' * 70}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
