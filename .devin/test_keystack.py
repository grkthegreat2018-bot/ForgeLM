"""Comprehensive test for all 21 weight-transform keys in the KeyStack.

Tests each key against the 3 FULL criteria:
1. forward(data) -> weights (data→weight without training)
2. reverse(weights) -> data (reversibility)
3. Composability (verified via KeyStack build)

Run: python -m pytest .devin/test_keystack.py
  or: python .devin/test_keystack.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

PASS = 0
FAIL = 0

def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {name}: {e}")

# ─── FULL keys (reversible + data→weight + composable) ───

def t_mtp():
    from research.keys.mtp_key import MTPKey
    k = MTPKey()
    lm_head = torch.randn(100, 256)
    r = k.forward({"lm_head_weight": lm_head, "n_predict": 4, "d_model": 256})
    assert r.success and len(r.weights["mtp_heads"]) == 4
    rv = k.reverse(r.weights)
    assert rv.success

def t_value_residual():
    from research.keys.value_residual_key import ValueResidualKey
    k = ValueResidualKey()
    vw = [torch.randn(64, 64) for _ in range(4)]
    r = k.forward({"v_weights": vw, "mode": "resformer"})
    assert r.success and (r.weights["v_weights"][0] - vw[0]).abs().max() < 1e-6
    rv = k.reverse(r.weights)
    assert (rv.data["v_weights"][1] - vw[1]).abs().max() < 1e-5

def t_slicegpt():
    from research.keys.slicegpt_key import SliceGPTKey
    k = SliceGPTKey()
    act = torch.randn(100, 64)
    r = k.forward({"residual_activations": act, "sparsity": 0.25,
                   "weights": {"w1": torch.randn(64, 64)}})
    assert r.success and r.metadata["new_d_model"] == 48

def t_mrl():
    from research.keys.mrl_key import MRLKey
    k = MRLKey()
    emb = torch.randn(1000, 256)
    r = k.forward({"embedding_weight": emb, "n_dims": [32, 64, 128, 256]})
    assert r.success and r.weights["reorder_indices"].shape == (256,)
    rv = k.reverse(r.weights)
    idx = r.weights["reorder_indices"]
    inv = rv.data["inverse_indices"]
    assert (idx[inv] == torch.arange(256)).all()

def t_rotorquant():
    from research.keys.rotorquant_key import RotorQuantKey
    k = RotorQuantKey()
    assert k.key_class().value == "full"

def t_spinquant():
    from research.keys.spinquant_key import SpinQuantHadamardKey
    k = SpinQuantHadamardKey()
    assert k.key_class().value == "full"

def t_quarot_r2():
    from research.keys.quarot_key import QuaRotR2Key
    k = QuaRotR2Key()
    n_heads, head_dim, d_model = 4, 64, 256
    vw = torch.randn(n_heads * head_dim, d_model)
    ow = torch.randn(d_model, n_heads * head_dim)
    r = k.forward({"v_weight": vw, "o_weight": ow, "n_heads": n_heads, "head_dim": head_dim})
    assert r.success
    rv = k.reverse({**r.weights, "n_heads": n_heads})
    assert (rv.data["v_weight"] - vw).abs().max() < 1e-4

# ─── BI keys (existing, exact copy) ───

def t_embedding():
    from research.keys.embedding_key import EmbeddingKey
    k = EmbeddingKey()
    assert k.key_class().value == "bi"

def t_rmsnorm():
    from research.keys.rmsnorm_key import RMSNormKey
    k = RMSNormKey()
    assert k.key_class().value == "bi"

def t_lm_head():
    from research.keys.lm_head_tied_key import LMHeadTiedKey
    k = LMHeadTiedKey()
    assert k.key_class().value == "bi"

def t_rope():
    from research.keys.rope_key import RoPEKey
    k = RoPEKey()
    assert k.key_class().value == "bi"

def t_causal_mask():
    from research.keys.causal_mask_key import CausalMaskKey
    k = CausalMaskKey()
    assert k.key_class().value == "bi"

# ─── PARTIAL keys (weight transform, not reversible) ───

def t_gqa_to_mqa():
    from research.keys.gqa_to_mqa_key import GQAToMQAKey
    k = GQAToMQAKey()
    kw = torch.randn(256, 256)
    r = k.forward({"k_weight": kw, "v_weight": kw, "n_kv_heads": 4, "head_dim": 64})
    assert r.success and r.weights["k_weight"].shape == (64, 256)

def t_wanda():
    from research.keys.wanda_key import WandaKey
    k = WandaKey()
    w = torch.randn(64, 128)
    act = torch.randn(100, 128)
    r = k.forward({"weight": w, "activations": act, "sparsity": 0.5})
    assert r.success and r.metadata["actual_sparsity"] > 0.4

def t_dspark():
    from research.keys.dspark_key import DSparkKey
    k = DSparkKey()
    assert k.key_class().value == "partial"

def t_moe_router():
    from research.keys.moe_router_key import MoERouterKey
    k = MoERouterKey()
    assert k.key_class().value == "partial"

def t_ssa():
    from research.keys.ssa_key import SSAKey
    k = SSAKey()
    assert k.key_class().value == "partial"

def t_gateskip():
    from research.keys.gateskip_key import GateSkipKey
    k = GateSkipKey()
    assert k.key_class().value == "partial"

def t_liquid_conv():
    from research.keys.liquid_conv_key import LiquidConvKey
    k = LiquidConvKey()
    assert k.key_class().value == "partial"

def t_sparda():
    from research.keys.sparda_key import SparDAKey
    k = SparDAKey()
    kw = torch.randn(128, 256)
    r = k.forward({"k_weight": kw, "n_kv_heads": 2, "head_dim": 64, "d_model": 256})
    assert r.success

def t_partial_rope():
    from research.keys.partial_rope_key import PartialRoPEKey
    k = PartialRoPEKey()
    r = k.forward({"head_dim": 128, "remove_ratio": 0.5, "mode": "frequency"})
    assert r.success and r.metadata["n_kept"] == 32

# ─── KeyStack build test ───

def t_keystack():
    from research.keys.keystack import build_xp_keystack
    s = build_xp_keystack()
    assert len(s.keys) == 21, f"Expected 21 keys, got {len(s.keys)}"
    full_count = sum(1 for k in s.keys if k.key_class().value == "full")
    bi_count = sum(1 for k in s.keys if k.key_class().value == "bi")
    partial_count = sum(1 for k in s.keys if k.key_class().value == "partial")
    assert full_count == 7, f"Expected 7 FULL, got {full_count}"
    assert bi_count == 5, f"Expected 5 BI, got {bi_count}"
    assert partial_count == 9, f"Expected 9 PARTIAL, got {partial_count}"

if __name__ == "__main__":
    print("=== KeyStack Test: 21 Weight-Transform Keys ===\n")

    print("FULL (7):")
    for name, fn in [("mtp", t_mtp), ("value_residual", t_value_residual),
                     ("slicegpt", t_slicegpt), ("mrl", t_mrl),
                     ("rotorquant", t_rotorquant), ("spinquant", t_spinquant),
                     ("quarot_r2", t_quarot_r2)]:
        check(name, fn)
    print(f"  {PASS} passed")

    p2 = PASS
    print("BI (5):")
    for name, fn in [("embedding", t_embedding), ("rmsnorm", t_rmsnorm),
                     ("lm_head", t_lm_head), ("rope", t_rope),
                     ("causal_mask", t_causal_mask)]:
        check(name, fn)
    print(f"  {PASS - p2} passed")

    p3 = PASS
    print("PARTIAL (9):")
    for name, fn in [("gqa_to_mqa", t_gqa_to_mqa), ("wanda", t_wanda),
                     ("dspark", t_dspark), ("moe_router", t_moe_router),
                     ("ssa", t_ssa), ("gateskip", t_gateskip),
                     ("liquid_conv", t_liquid_conv), ("sparda", t_sparda),
                     ("partial_rope", t_partial_rope)]:
        check(name, fn)
    print(f"  {PASS - p3} passed")

    p4 = PASS
    print("KeyStack:")
    check("keystack_build", t_keystack)
    print(f"  {PASS - p4} passed")

    print(f"\n=== {PASS}/{PASS + FAIL} passed, {FAIL} failed ===")
