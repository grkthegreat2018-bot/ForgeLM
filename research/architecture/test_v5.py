"""Smoke test for ForgeLM V5 BitNet-MoE architecture.

Uses a tiny config to verify wiring on CPU without OOM.
The real V5 config (7.5B params) is only testable on GPU.
"""
import sys
sys.path.insert(0, 'D:/windsurf/ForgeAI')

import torch
from research.config import get_config, MODEL_CONFIGS, ModelConfig
from research.model_loader import ConfigurableResearchLLM

print('Available configs:', list(MODEL_CONFIGS.keys()))
print()

# Build a TINY V5 config to verify wiring without OOM
tiny_v5 = ModelConfig(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    intermediate_size=256,
    attn_type="gta",
    attn_bias=False,
    ffn_type="swiglu",
    norm_type="rmsnorm",
    rope_base=1_000_000.0,
    max_seq_len=128,
    conv_kernel_size=3,
    use_qk_norm=True,
    use_bitnet=True,
    bitnet_learned_scale=True,
    use_fused_gemm=True,
    layer_types=["conv", "conv", "attention", "conv"],
    use_titan_memory=True,
    titan_memory_rank=32,
    use_mod=True,
    mod_keep_fraction=1.0,
    ffn_skip_threshold=0.0,
    use_mhc=True,
    mhc_rank=0,
    use_attn_residual=True,
    attn_res_k=4,
    # V5: MoE
    use_moe=True,
    moe_n_experts=4,
    moe_top_k=2,
    moe_shared_expert=True,
    moe_dense_bypass=True,
    moe_noisy_gating=True,
    moe_load_balance_weight=0.01,
    moe_d_ff=256,
    moe_expert_tying=True,
    moe_tie_group_size=2,
    # V5: Factorized embedding
    use_factorized_embeddings=True,
    embed_factorized_rank=32,
    # V5: Full BitNet (embedding too)
    use_bitnet_embedding=True,
    # V5.1: Additional architecture keys (all lossless at init)
    use_mtp=True,
    mtp_n_heads=2,
    mtp_loss_weight=0.3,
    use_value_residual=True,
    value_residual_mode="resformer",
    value_residual_gate_init=0.0,
    use_sandwich_norm=True,
    use_learned_sink=True,
    learned_sink_init=0.0,
    learned_sink_init_method="zero",
    use_swiglu_clamp=True,
    swiglu_clamp_alpha=1.702,
    swiglu_clamp_limit=7.0,
    rope_variant="lerope",
    batch_size=2,
    seq_len=64,
    max_steps=20,
    warmup_steps=5,
)

print('Tiny V5 config:')
print(f'  d_model={tiny_v5.d_model}, n_layers={tiny_v5.n_layers}')
print(f'  use_moe={tiny_v5.use_moe}, n_experts={tiny_v5.moe_n_experts}, top_k={tiny_v5.moe_top_k}')
print(f'  moe_expert_tying={tiny_v5.moe_expert_tying}, tie_group_size={tiny_v5.moe_tie_group_size}')
print(f'  use_factorized_embeddings={tiny_v5.use_factorized_embeddings}, rank={tiny_v5.embed_factorized_rank}')
print(f'  use_bitnet={tiny_v5.use_bitnet}, use_bitnet_embedding={tiny_v5.use_bitnet_embedding}')
print(f'  attn_type={tiny_v5.attn_type}, use_fused_gemm={tiny_v5.use_fused_gemm}')
print(f'  use_mhc={tiny_v5.use_mhc}, use_titan_memory={tiny_v5.use_titan_memory}')
print()

# Test model construction
model = ConfigurableResearchLLM(tiny_v5)
total = sum(p.numel() for p in model.parameters())
print(f'Model built! Total params: {total/1e3:.1f}K')

# Count by component
cats = {}
for name, p in model.named_parameters():
    n = p.numel()
    if 'embed' in name and 'project' not in name: cat = 'embedding_lookup'
    elif 'project' in name and 'embed' in name: cat = 'embedding_project'
    elif 'head' in name: cat = 'head'
    elif 'experts' in name: cat = 'experts'
    elif 'shared' in name: cat = 'shared_expert'
    elif 'router' in name or ('gate' in name.lower() and 'attn' not in name and 'ffn' not in name):
        cat = 'router'
    elif 'attn' in name: cat = 'attention'
    elif 'conv' in name: cat = 'conv'
    elif 'mhc' in name: cat = 'mhc'
    elif 'titan' in name or 'memory' in name: cat = 'titan'
    elif 'norm' in name or 'ln' in name: cat = 'norm'
    else: cat = 'other'
    cats[cat] = cats.get(cat, 0) + n

for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f'  {cat:25s}: {n/1e3:8.2f}K  ({100*n/total:.1f}%)')

# Check expert tying was applied
tying_applied = getattr(model, '_expert_tying_applied', False)
print(f'\n  Expert tying applied: {tying_applied}')

# Check MoE layers exist
moe_count = sum(1 for b in model.blocks if hasattr(b.ffn, 'experts'))
print(f'  MoE FFN layers: {moe_count} / {len(model.blocks)}')

# Check expert tying: odd layers should share expert weights with even layers
if tying_applied:
    tied = 0
    for i in range(0, len(model.blocks), 2):
        if i+1 < len(model.blocks):
            even_ffn = model.blocks[i].ffn
            odd_ffn = model.blocks[i+1].ffn
            if hasattr(even_ffn, 'experts') and hasattr(odd_ffn, 'experts'):
                # Check if the first expert's weight is the same tensor object
                if len(even_ffn.experts) > 0 and len(odd_ffn.experts) > 0:
                    w_even = even_ffn.experts[0].w1.weight
                    w_odd = odd_ffn.experts[0].w1.weight
                    if w_even.data_ptr() == w_odd.data_ptr():
                        tied += 1
    print(f'  Tied layer pairs (weight shared): {tied}')

# Check factorized embedding
embed_type = type(model.embed).__name__
head_type = type(model.head).__name__
print(f'  Embedding type: {embed_type}')
print(f'  Head type: {head_type}')

# Quick forward test (eval mode)
print('\n--- Forward test (eval) ---')
model.eval()
with torch.no_grad():
    idx = torch.randint(0, tiny_v5.vocab_size, (1, 8))
    logits, loss = model(idx)
    print(f'  Forward OK: logits shape {logits.shape}')

# Training mode test (BitNet QAT + MoE routing)
print('\n--- Training mode test ---')
model.train()
with torch.no_grad():
    idx = torch.randint(0, tiny_v5.vocab_size, (2, 16))
    targets = torch.randint(0, tiny_v5.vocab_size, (2, 16))
    logits, loss = model(idx, targets=targets)
    print(f'  Training forward OK: loss={loss.item():.4f}')
    # Check MoE aux loss is collected
    aux = 0.0
    for b in model.blocks:
        if hasattr(b, '_last_aux_loss') and b._last_aux_loss is not None:
            aux += b._last_aux_loss.item()
    print(f'  MoE aux loss: {aux:.6f}')

# Test backward pass
print('\n--- Backward test ---')
model.train()
idx = torch.randint(0, tiny_v5.vocab_size, (2, 16))
targets = torch.randint(0, tiny_v5.vocab_size, (2, 16))
logits, loss = model(idx, targets=targets)
loss.backward()
print(f'  Backward OK: loss={loss.item():.4f}')

# Check gradients flow to experts
expert_grad_count = 0
for name, p in model.named_parameters():
    if 'experts' in name and p.grad is not None:
        expert_grad_count += 1
print(f'  Expert params with gradients: {expert_grad_count}')

# Check BitNet qscale params exist
bitnet_count = sum(1 for n, p in model.named_parameters() if 'qscale' in n)
print(f'  BitNet qscale params: {bitnet_count}')

# ── V5.1 new keys checks ──
print('\n--- V5.1 new architecture keys ---')

# MTP module
has_mtp = model.mtp_module is not None
print(f'  MTP module: {has_mtp}')
if has_mtp:
    print(f'    MTP heads: {model.mtp_module.n_heads}')

# ValueResidual
has_vr = getattr(model, '_use_value_residual', False)
print(f'  ValueResidual: {has_vr}')
if has_vr:
    n_gates = len(model._v0_gates) if model._v0_gates else 0
    print(f'    V_0 gates: {n_gates} (all zero={all(g.item() == 0.0 for g in model._v0_gates)})')

# SandwichNorm
has_sandwich = hasattr(model.blocks[0], 'post_attn_norm')
print(f'  SandwichNorm: {has_sandwich}')
if has_sandwich:
    # Check identity init (weight=1)
    w = model.blocks[0].post_attn_norm.weight
    print(f'    Post-attn norm identity: {(w == 1.0).all().item()}')

# LearnedSink — check the first attention layer (not conv)
has_sink = False
for block in model.blocks:
    if hasattr(block.attn, 'sinks') and block.attn.sinks is not None:
        has_sink = True
        sinks = block.attn.sinks
        print(f'  LearnedSink: {has_sink}')
        print(f'    Sinks shape: {sinks.shape}, all zero: {(sinks == 0.0).all().item()}')
        break
if not has_sink:
    print(f'  LearnedSink: {has_sink}')

# SwiGLU Clamp — check if any FFN has use_clamp (MoE experts use SwiGLU internally)
has_clamp = False
for block in model.blocks:
    if hasattr(block.ffn, 'use_clamp') and block.ffn.use_clamp:
        has_clamp = True
        break
print(f'  SwiGLU Clamp: {has_clamp}')

# LeRoPE
from research.keys.position.lerope_key import LeRoPEEmbedding
has_lerope = False
for block in model.blocks:
    if hasattr(block.attn, 'rope') and isinstance(block.attn.rope, LeRoPEEmbedding):
        has_lerope = True
        break
print(f'  LeRoPE: {has_lerope}')
if has_lerope:
    # Check identity init (freq_scale=1.0)
    for block in model.blocks:
        if hasattr(block.attn, 'rope') and isinstance(block.attn.rope, LeRoPEEmbedding):
            print(f'    freq_scale identity: {(block.attn.rope.freq_scale == 1.0).all().item()}')
            break

print('\n=== V5 smoke test PASSED ===')

