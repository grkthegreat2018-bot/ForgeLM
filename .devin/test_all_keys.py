"""Test all 38 keys in the KeyStack."""
import sys
sys.path.insert(0, '.')
import torch

print("=" * 60)
print("Testing All 38 Keys")
print("=" * 60)

passed = 0
failed = 0
errors = []

# ─── FULL keys (10) ──────────────────────────────────────────────────

print("\n--- FULL Keys ---")

# 1. MTP
try:
    from research.keys.mtp_key import MTPKey
    k = MTPKey()
    r = k.forward({"lm_head_weight": torch.randn(1000, 128), "n_predict": 4, "d_model": 128})
    assert r.success and len(r.weights) > 0
    print("  [PASS] mtp")
    passed += 1
except Exception as e:
    print(f"  [FAIL] mtp: {e}"); failed += 1; errors.append(f"mtp: {e}")

# 2. ValueResidual
try:
    from research.keys.value_residual_key import ValueResidualKey
    k = ValueResidualKey()
    print("  [PASS] value_residual (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] value_residual: {e}"); failed += 1; errors.append(f"value_residual: {e}")

# 3. SliceGPT
try:
    from research.keys.slicegpt_key import SliceGPTKey
    k = SliceGPTKey()
    print("  [PASS] slicegpt (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] slicegpt: {e}"); failed += 1; errors.append(f"slicegpt: {e}")

# 4. MRL
try:
    from research.keys.mrl_key import MRLKey
    k = MRLKey()
    print("  [PASS] mrl (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] mrl: {e}"); failed += 1; errors.append(f"mrl: {e}")

# 5. RotorQuant
try:
    from research.keys.rotorquant_key import RotorQuantKey
    k = RotorQuantKey()
    print("  [PASS] rotorquant (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] rotorquant: {e}"); failed += 1; errors.append(f"rotorquant: {e}")

# 6. SpinQuant
try:
    from research.keys.spinquant_key import SpinQuantHadamardKey
    k = SpinQuantHadamardKey()
    print("  [PASS] spinquant_hadamard (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] spinquant_hadamard: {e}"); failed += 1; errors.append(f"spinquant: {e}")

# 7. QuaRot
try:
    from research.keys.quarot_key import QuaRotR2Key
    k = QuaRotR2Key()
    print("  [PASS] quarot_r2 (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] quarot_r2: {e}"); failed += 1; errors.append(f"quarot: {e}")

# 8. QK-Norm MLA
try:
    from research.keys.qk_norm_mla_key import QKNormMLAKey
    k = QKNormMLAKey()
    r = k.forward({"n_layers": 4, "head_dim": 128})
    assert r.success and len(r.weights) == 8  # 4 layers x 2 norms
    assert all((v == 1.0).all() for v in r.weights.values())
    print("  [PASS] qk_norm_mla (identity init verified)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] qk_norm_mla: {e}"); failed += 1; errors.append(f"qk_norm_mla: {e}")

# 9. WQ Elim
try:
    from research.keys.wq_elim_key import WQElimKey
    k = WQElimKey()
    r = k.forward({"n_layers": 4, "d_model": 128, "has_bias": True})
    assert r.success and len(r.weights) == 8  # 4 layers x (weight + bias)
    for key, v in r.weights.items():
        if "weight" in key:
            assert torch.allclose(v.float(), torch.eye(128)), f"{key} not identity!"
    print("  [PASS] wq_elim (identity init verified)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] wq_elim: {e}"); failed += 1; errors.append(f"wq_elim: {e}")

# 10. Norm Folding
try:
    from research.keys.norm_folding_key import NormFoldingKey
    k = NormFoldingKey()
    state = {
        "blocks.0.ln1.weight": torch.ones(64, dtype=torch.bfloat16) * 2.0,
        "blocks.0.ln2.weight": torch.ones(64, dtype=torch.bfloat16) * 3.0,
        "blocks.0.attn.q_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        "blocks.0.attn.kv_down_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        "blocks.0.ffn.w_gate.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        "blocks.0.ffn.w_up.weight": torch.randn(128, 64, dtype=torch.bfloat16),
        "ln_f.weight": torch.ones(64, dtype=torch.bfloat16) * 1.5,
        "head.weight": torch.randn(500, 64, dtype=torch.bfloat16),
    }
    r = k.forward(state)
    assert r.success
    assert "blocks.0.ln1.weight" not in r.weights, "ln1 not folded!"
    assert "ln_f.weight" not in r.weights, "ln_f not folded!"
    # Verify q_proj columns scaled by gamma=2
    orig = state["blocks.0.attn.q_proj.weight"].float()
    folded = r.weights["blocks.0.attn.q_proj.weight"].float()
    assert torch.allclose(folded, orig * 2.0, atol=1e-5), "q_proj not scaled!"
    print("  [PASS] norm_folding (verified column scaling)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] norm_folding: {e}"); failed += 1; errors.append(f"norm_folding: {e}")

# ─── BI keys (5) ─────────────────────────────────────────────────────

print("\n--- BI Keys ---")
bi_keys = ["embedding", "rmsnorm", "lm_head_tied", "rope", "causal_mask"]
for kn in bi_keys:
    try:
        mod = __import__(f"research.keys.{kn}_key", fromlist=[f"{kn.capitalize()}Key"])
        cls = getattr(mod, f"{kn.capitalize()}Key", None)
        if cls is None:
            # Fallback: find class ending in "Key" (exclude KeyClass, KeyResult)
            for attr in dir(mod):
                if attr in ("Key", "KeyClass", "KeyResult"):
                    continue
                obj = getattr(mod, attr)
                if isinstance(obj, type) and attr.endswith("Key"):
                    cls = obj
                    break
        k = cls()
        assert k.key_class().value == "bi"
        print(f"  [PASS] {kn}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {kn}: {e}"); failed += 1; errors.append(f"{kn}: {e}")

# ─── PARTIAL keys (16) ───────────────────────────────────────────────

print("\n--- PARTIAL Keys ---")

# GQA→MQA
try:
    from research.keys.gqa_to_mqa_key import GQAToMQAKey
    k = GQAToMQAKey()
    print("  [PASS] gqa_to_mqa (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] gqa_to_mqa: {e}"); failed += 1; errors.append(f"gqa_to_mqa: {e}")

# Wanda
try:
    from research.keys.wanda_key import WandaKey
    k = WandaKey()
    print("  [PASS] wanda (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] wanda: {e}"); failed += 1; errors.append(f"wanda: {e}")

# DSpark
try:
    from research.keys.dspark_key import DSparkKey
    k = DSparkKey()
    print("  [PASS] dspark (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] dspark: {e}"); failed += 1; errors.append(f"dspark: {e}")

# MoERouter
try:
    from research.keys.moe_router_key import MoERouterKey
    k = MoERouterKey()
    print("  [PASS] moe_router (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] moe_router: {e}"); failed += 1; errors.append(f"moe_router: {e}")

# SSA
try:
    from research.keys.ssa_key import SSAKey
    k = SSAKey()
    print("  [PASS] ssa (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] ssa: {e}"); failed += 1; errors.append(f"ssa: {e}")

# GateSkip
try:
    from research.keys.gateskip_key import GateSkipKey
    k = GateSkipKey()
    print("  [PASS] gateskip (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] gateskip: {e}"); failed += 1; errors.append(f"gateskip: {e}")

# LiquidConv
try:
    from research.keys.liquid_conv_key import LiquidConvKey
    k = LiquidConvKey()
    print("  [PASS] liquid_conv (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] liquid_conv: {e}"); failed += 1; errors.append(f"liquid_conv: {e}")

# SparDA
try:
    from research.keys.sparda_key import SparDAKey
    k = SparDAKey()
    print("  [PASS] sparda (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] sparda: {e}"); failed += 1; errors.append(f"sparda: {e}")

# PartialRoPE
try:
    from research.keys.partial_rope_key import PartialRoPEKey
    k = PartialRoPEKey()
    print("  [PASS] partial_rope (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] partial_rope: {e}"); failed += 1; errors.append(f"partial_rope: {e}")

# Expert Consolidation
try:
    from research.keys.expert_consolidation_key import ExpertConsolidationKey
    k = ExpertConsolidationKey(threshold=0.9, min_experts=1)
    state = {}
    base_w = torch.randn(128, 64, dtype=torch.bfloat16)
    for i in range(2):
        for ei in range(4):
            w = base_w.clone() if ei < 2 else torch.randn(128, 64, dtype=torch.bfloat16)
            state[f"blocks.{i}.ffn.experts.{ei}.w_gate.weight"] = w
            state[f"blocks.{i}.ffn.experts.{ei}.w_up.weight"] = w.clone()
            state[f"blocks.{i}.ffn.experts.{ei}.w_down.weight"] = w.t().contiguous()
        state[f"blocks.{i}.ln1.weight"] = torch.ones(64, dtype=torch.bfloat16)
    r = k.forward(state)
    assert r.success and r.metadata["total_merged"] > 0
    print("  [PASS] expert_consolidation (merged identical experts)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] expert_consolidation: {e}"); failed += 1; errors.append(f"expert_consolidation: {e}")

# GRAIL
try:
    from research.keys.grail_key import GRAILKey
    k = GRAILKey()
    d, N = 64, 256
    x_orig = torch.randn(N, d)
    Q = torch.linalg.qr(torch.randn(d, d))[0]
    x_trans = x_orig @ Q + 0.01 * torch.randn(N, d)
    R = k.compute_reconstruction_map(x_orig, x_trans)
    x_recon = x_trans @ R
    err_before = (x_orig - x_trans).pow(2).mean().item()
    err_after = (x_orig - x_recon).pow(2).mean().item()
    assert err_after < err_before
    print(f"  [PASS] grail_compensation (err: {err_before:.4f} → {err_after:.6f})")
    passed += 1
except Exception as e:
    print(f"  [FAIL] grail_compensation: {e}"); failed += 1; errors.append(f"grail: {e}")

# Activation Transmute
try:
    from research.keys.activation_transmute_key import ActivationTransmuteKey
    k = ActivationTransmuteKey(target="reglu")
    g = torch.randn(512, 128) * 3.0
    alpha, beta = k.solve_per_channel(g)
    source = k.source_activation(g)
    target = k.target_activation(alpha * g + beta)
    rel_err = (source - target).pow(2).mean().item() / source.pow(2).mean().item()
    assert rel_err < 0.3
    print(f"  [PASS] activation_transmute (SwiGLU→ReGLU err: {rel_err*100:.1f}%)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] activation_transmute: {e}"); failed += 1; errors.append(f"act_transmute: {e}")

# Lossless Quant
try:
    from research.keys.lossless_quant_key import LosslessQuantKey
    k = LosslessQuantKey(bits=4, group_size=128, rotate=True)
    state = {
        "blocks.0.attn.q_proj.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "blocks.0.ffn.w_gate.weight": torch.randn(256, 128, dtype=torch.bfloat16),
        "embed.weight": torch.randn(1000, 128, dtype=torch.bfloat16),
        "ln_f.weight": torch.ones(128, dtype=torch.bfloat16),
    }
    r = k.forward(state)
    assert r.success
    assert r.metadata["n_quantized"] == 2  # skip embed and ln_f
    assert r.metadata["compression_ratio"] > 1.0
    print(f"  [PASS] lossless_quant ({r.metadata['compression_ratio']:.1f}x compression)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] lossless_quant: {e}"); failed += 1; errors.append(f"lossless_quant: {e}")

# Fact Injection
try:
    from research.keys.fact_injection_key import FactInjectionKey
    k = FactInjectionKey()
    d_model, d_ff = 128, 256
    state = {
        "blocks.0.ffn.w_gate.weight": torch.randn(d_ff, d_model, dtype=torch.bfloat16),
        "blocks.0.ffn.w_up.weight": torch.randn(d_ff, d_model, dtype=torch.bfloat16),
        "blocks.0.ffn.w_down.weight": torch.randn(d_model, d_ff, dtype=torch.bfloat16),
    }
    facts = [(torch.randn(d_model), torch.randn(d_model)) for _ in range(3)]
    r = k.forward({"state": state, "facts": facts, "n_layers": 1,
                    "d_model": d_model, "d_ff": d_ff, "layer_idx": 0})
    assert r.success and r.metadata["n_facts"] == 3
    print("  [PASS] fact_injection (3 facts injected)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] fact_injection: {e}"); failed += 1; errors.append(f"fact_injection: {e}")

# Context Patch
try:
    from research.keys.context_patch_key import ContextPatchKey
    k = ContextPatchKey(alpha=1.0)
    d_model, d_ff = 128, 256
    state = {"blocks.0.ffn.w_gate.weight": torch.randn(d_ff, d_model, dtype=torch.bfloat16)}
    patches = [(0, "w_gate", torch.randn(d_model), torch.randn(d_ff))]
    r = k.forward({"state": state, "patches": patches})
    assert r.success and r.metadata["n_patches"] == 1
    print("  [PASS] context_patch (rank-1 patch applied)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] context_patch: {e}"); failed += 1; errors.append(f"context_patch: {e}")

# Self-Play
try:
    from research.keys.self_play_key import SelfPlayKey
    k = SelfPlayKey(n_rounds=5, domain="science")
    assert k.n_rounds == 5
    print("  [PASS] self_play (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] self_play: {e}"); failed += 1; errors.append(f"self_play: {e}")

# ─── TRIVIAL keys (7) ────────────────────────────────────────────────

print("\n--- TRIVIAL Keys ---")

# AirLLM
try:
    from research.keys.airllm_key import AirLLMKey
    k = AirLLMKey()
    assert k.key_class().value == "trivial"
    print("  [PASS] airllm")
    passed += 1
except Exception as e:
    print(f"  [FAIL] airllm: {e}"); failed += 1; errors.append(f"airllm: {e}")

# DenseFormer
try:
    from research.keys.denseformer_key import DenseFormerKey
    k = DenseFormerKey()
    r = k.forward({"n_layers": 4, "dilation": 1})
    assert r.success
    for w in r.weights["dwa_weights"]:
        assert w[len(w)-1] == 1.0  # identity init
    print("  [PASS] denseformer (identity init verified)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] denseformer: {e}"); failed += 1; errors.append(f"denseformer: {e}")

# Logit Cap
try:
    from research.keys.logit_cap_key import LogitCapKey
    k = LogitCapKey(cap=30.0)
    assert k.cap == 30.0
    print("  [PASS] logit_cap")
    passed += 1
except Exception as e:
    print(f"  [FAIL] logit_cap: {e}"); failed += 1; errors.append(f"logit_cap: {e}")

# SwiGLU Clamp
try:
    from research.keys.swiglu_clamp_key import SwiGLUClampKey
    k = SwiGLUClampKey()
    gate_up = torch.randn(4, 128)
    r = k.forward({"gate_up": gate_up, "alpha": 1.702, "limit": 7.0})
    assert r.success
    large = torch.full((4, 128), 100.0)
    r2 = k.forward({"gate_up": large, "limit": 7.0})
    assert r2.weights["output"][0, 0].item() < 60  # bounded
    print("  [PASS] swiglu_clamp (bounded output verified)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] swiglu_clamp: {e}"); failed += 1; errors.append(f"swiglu_clamp: {e}")

# SandwichNorm
try:
    from research.keys.sandwich_norm_key import SandwichNormKey
    k = SandwichNormKey()
    r = k.forward({"d_model": 128, "n_layers": 4})
    assert r.success
    assert all((w == 1.0).all() for w in r.weights["post_attn_norms"])
    print("  [PASS] sandwich_norm (identity init verified)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] sandwich_norm: {e}"); failed += 1; errors.append(f"sandwich_norm: {e}")

# AirMoE
try:
    from research.keys.airmoe_key import AirMoEKey
    k = AirMoEKey(max_resident_experts=2)
    assert k.max_resident_experts == 2
    print("  [PASS] airmoe (instantiated)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] airmoe: {e}"); failed += 1; errors.append(f"airmoe: {e}")

# Knowledge Pack
try:
    from research.keys.knowledge_pack_key import KnowledgePackKey
    k = KnowledgePackKey()
    assert k.key_class().value == "trivial"
    print("  [PASS] knowledge_pack")
    passed += 1
except Exception as e:
    print(f"  [FAIL] knowledge_pack: {e}"); failed += 1; errors.append(f"knowledge_pack: {e}")

# ─── KeyStack verification ───────────────────────────────────────────

print("\n--- KeyStack ---")
try:
    from research.keys.keystack import build_xp_keystack
    s = build_xp_keystack()
    assert len(s.keys) == 38, f"Expected 38 keys, got {len(s.keys)}"
    full = sum(1 for k in s.keys if k.key_class().value == "full")
    bi = sum(1 for k in s.keys if k.key_class().value == "bi")
    partial = sum(1 for k in s.keys if k.key_class().value == "partial")
    trivial = sum(1 for k in s.keys if k.key_class().value == "trivial")
    print(f"  [PASS] keystack: {len(s.keys)} keys ({full}F/{bi}B/{partial}P/{trivial}T)")
    passed += 1
except Exception as e:
    print(f"  [FAIL] keystack: {e}"); failed += 1; errors.append(f"keystack: {e}")

# ─── Summary ─────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
if errors:
    print(f"\nFailures:")
    for e in errors:
        print(f"  - {e}")
print(f"{'='*60}")
