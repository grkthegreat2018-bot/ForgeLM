"""Test conformal_sampler + vocab_pack integrations.

1. ConformalSampler: verify calibration + per-query temperature in sandbox
2. VocabPack: verify extracted packs load and contain shifted tokens
"""
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

VOCAB_PACK_DIR = "research/checkpoints/vocab_packs"


def test_conformal_sampler_integration():
    """Verify ConformalSampler is wired into SelfPlaySandbox."""
    print("[conformal_sampler integration]")
    from research.keys.training.conformal_sampler_key import ConformalSampler

    # 1. Unit test the sampler
    sampler = ConformalSampler()
    scores = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    sampler.calibrate(scores, alpha=0.1)

    assert sampler.threshold > 0, "threshold should be positive"
    print(f"  Calibrated: threshold={sampler.threshold:.3f}")

    # Confident query → low temperature
    t_confident = sampler.get_temperature(0.95)
    # Uncertain query → high temperature
    t_uncertain = sampler.get_temperature(0.2)
    assert t_confident < t_uncertain, f"confident={t_confident} should be < uncertain={t_uncertain}"
    print(f"  T(confident=0.95)={t_confident:.2f}, T(uncertain=0.20)={t_uncertain:.2f}")

    # 2. Verify sandbox has conformal_sampler attribute
    import inspect
    from research.self_play.self_play_sandbox import SelfPlaySandbox

    source = inspect.getsource(SelfPlaySandbox)
    assert "conformal_sampler" in source, "sandbox should reference conformal_sampler"
    assert "calibrate_conformal" in source, "sandbox should have calibrate_conformal method"
    assert "_get_query_temperature" in source, "sandbox should have _get_query_temperature method"
    print("  Sandbox has conformal_sampler wired in")

    # 3. Verify training script calls calibration
    from research.training.self_play_expert_training import ExpertSelfPlayTrainer
    trainer_source = inspect.getsource(ExpertSelfPlayTrainer)
    assert "calibrate_conformal" in trainer_source, "trainer should call calibrate_conformal"
    print("  Trainer calls calibrate_conformal()")

    print("  PASS: conformal_sampler integration\n")
    return True


def test_vocab_pack_extraction():
    """Verify vocab packs were extracted and contain data."""
    print("[vocab_pack extraction]")
    pack_dir = Path(VOCAB_PACK_DIR)
    assert pack_dir.exists(), f"vocab pack dir should exist: {pack_dir}"

    packs = list(pack_dir.glob("vocab_pack_*.pt"))
    assert len(packs) >= 7, f"should have 7 packs, found {len(packs)}"
    print(f"  Found {len(packs)} vocab packs")

    total_tokens = 0
    for pack_path in sorted(packs):
        pack = torch.load(str(pack_path), weights_only=False)
        topic = pack["topic"]
        token_ids = pack["token_ids"]
        prob_deltas = pack["prob_deltas"]
        top_tokens = pack["top_tokens"]

        assert len(token_ids) == len(prob_deltas), f"{topic}: token_ids/deltas length mismatch"
        assert len(token_ids) > 0, f"{topic}: should have shifted tokens"
        total_tokens += len(token_ids)
        print(f"  {topic}: {len(token_ids)} tokens, top: {top_tokens[:5]}")

    assert total_tokens > 0, "should have extracted some tokens"
    print(f"  Total: {total_tokens} shifted tokens across {len(packs)} packs")

    # Verify round-trip with VocabPackKey
    from research.keys.vocab_pack_key import VocabPackKey
    key = VocabPackKey()

    # Simulate: create base + domain embeddings, extract, reverse
    vocab_size, d_model = 100, 64
    base_embed = torch.randn(vocab_size, d_model)
    domain_embed = base_embed.clone()
    domain_embed[10] += 0.5  # domain-specific shift for token 10
    domain_embed[20] += 0.3  # domain-specific shift for token 20

    result = key.forward({
        "base_embed": base_embed,
        "domain_embed": domain_embed,
        "domain_token_ids": [10, 20],
    })
    assert result.success
    deltas = result.weights["deltas"]
    assert abs(deltas[0, 0].item() - 0.5) < 1e-5, "delta should match shift"
    print("  VocabPackKey round-trip verified")

    print("  PASS: vocab_pack extraction\n")
    return True


def main():
    print("=" * 60)
    print("Testing Key Integrations")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    for name, fn in [("conformal_sampler", test_conformal_sampler_integration),
                     ("vocab_pack", test_vocab_pack_extraction)]:
        try:
            if fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name} raised {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
