"""KeyStack — compose multiple keys into a full model key.

A KeyStack stitches component keys together using the model's forward pass
wiring. Each key handles one component; the stack handles the data flow
between them.

Building a KeyStack:
  1. List the component keys in forward order
  2. Provide stitching: how each key's output feeds the next
  3. Use dummy data to verify the stitching works
  4. The stack can then process real data/weights end-to-end
"""
from typing import Any, Dict, List, Optional

import torch

from .base import Key, KeyClass, KeyResult


class KeyStack:
    """A stack of keys that processes a full model.

    Keys are applied in order. Each key receives data from the previous key's
    output (or the initial input) and produces weights/data for its component.
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.keys: list[Key] = []
        self.stitching: dict[str, str] = {}  # output_name -> input_name mapping

    def add(self, key: Key) -> 'KeyStack':
        """Add a key to the stack."""
        self.keys.append(key)
        return self

    def describe(self) -> str:
        """Human-readable description of the stack."""
        lines = [f"KeyStack: {self.name}", f"  Components ({len(self.keys)}):"]
        for i, key in enumerate(self.keys):
            lines.append(f"    {i+1}. {key.name} [{key.key_class().value}]")
        bi_count = sum(1 for k in self.keys if k.key_class() in (KeyClass.BI, KeyClass.FULL))
        partial_count = sum(1 for k in self.keys if k.key_class() == KeyClass.PARTIAL)
        trivial_count = sum(1 for k in self.keys if k.key_class() == KeyClass.TRIVIAL)
        lines.append(f"  Summary: {bi_count} Bi, {partial_count} Partial, {trivial_count} Trivial")
        all_bi = partial_count == 0
        lines.append(f"  Full Bi KeyStack: {'YES' if all_bi else 'NO (has Partial keys)'}")
        return "\n".join(lines)

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights for ALL components in the stack.

        Calls each key's forward() with the relevant subset of data.
        Merges all weight outputs into one dict.
        """
        all_weights = {}
        all_metadata = {}
        errors = []

        for key in self.keys:
            # Filter data to what this key needs (pass everything, let key pick)
            result = key.forward(data)
            if result.success:
                all_weights.update(result.weights)
                all_metadata[key.name] = result.metadata
            else:
                errors.append(f"{key.name}: {result.error}")

        if errors and not all_weights:
            return KeyResult(success=False, error="; ".join(errors))

        return KeyResult(
            success=True,
            weights=all_weights,
            metadata={'stack': self.name, 'components': all_metadata,
                     'errors': errors if errors else None}
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """weights -> data for ALL components."""
        all_data = {}
        all_metadata = {}
        errors = []

        for key in self.keys:
            # Filter weights to what this key owns
            result = key.reverse(weights)
            if result.success:
                all_data.update(result.data)
                all_metadata[key.name] = result.metadata
            else:
                errors.append(f"{key.name}: {result.error}")

        return KeyResult(
            success=True if all_data else False,
            data=all_data,
            metadata={'stack': self.name, 'components': all_metadata,
                     'errors': errors if errors else None}
        )

    def cross_arch(self, weights_a: dict[str, torch.Tensor],
                   stack_b: 'KeyStack') -> KeyResult:
        """weights(A) -> data -> weights(B) using two KeyStacks."""
        # Decode A
        decode = self.reverse(weights_a)
        if not decode.success:
            return KeyResult(success=False, error=f"Reverse stack A failed: {decode.error}")

        # Encode B
        encode = stack_b.forward(decode.data)
        if not encode.success:
            return KeyResult(success=False, error=f"Forward stack B failed: {encode.error}")

        return KeyResult(
            success=True,
            weights=encode.weights,
            metadata={'source_stack': self.name, 'target_stack': stack_b.name,
                     'intermediate_keys': list(decode.data.keys())}
        )

    def __repr__(self):
        return f"KeyStack({self.name}, {len(self.keys)} keys)"


def build_qwen2_keystack() -> KeyStack:
    """Build the KeyStack for Qwen2 architecture.

    Qwen2 = Embedding + RMSNorm + RoPE + GQA + SwiGLU + Causal + LM Head
    """
    from .causal_mask_key import CausalMaskKey
    from .embedding_key import EmbeddingKey
    from .gqa_key import GQAKey
    from .lm_head_tied_key import LMHeadTiedKey
    from .rmsnorm_key import RMSNormKey
    from .rope_key import RoPEKey
    from .swiglu_key import SwiGLUKey

    stack = KeyStack("qwen2")
    stack.add(EmbeddingKey())
    stack.add(RMSNormKey())
    stack.add(RoPEKey())
    stack.add(GQAKey())
    stack.add(SwiGLUKey())
    stack.add(CausalMaskKey())
    stack.add(LMHeadTiedKey())
    return stack


def build_xp_keystack() -> KeyStack:
    """Build the KeyStack for the XP (target) model — weight-transform keys only.

    Only keys that actually CONVERT weights are included. Architecture config
    keys (identity init, runtime masks, formulas) are kept as files but excluded
    from the stack since they don't transform weights.

    FULL criteria (all 3):
      1. Reversible: reverse(weights) -> data
      2. Data→weight: forward(data) -> weights without training
      3. Composable: chains with other keys for weight-to-weight

    Classification:
      - FULL (7): MTP, Value Residual, SliceGPT, MRL, RotorQuant, SpinQuant, QuaRot R2
      - BI (5): Embedding, RMSNorm, LM Head, RoPE, Causal Mask (existing in source)
      - PARTIAL (9): GQA→MQA, Wanda, DSpark, MoE Router, SSA, GateSkip, Liquid Conv,
                      SparDA, PartialRoPE
    """
    from .activation_transmute_key import ActivationTransmuteKey
    from .airllm_key import AirLLMKey
    from .airmoe_key import AirMoEKey

    # --- New keys (Bonsai-27B inspired + self-play pipeline) ---
    from .binary_g128_key import BinaryG128Key  # LOSSY: binary {±1} g128, 14.2x compression
    from .causal_mask_key import CausalMaskKey
    from .context_patch_key import ContextPatchKey
    from .cot_knowledge_pack_key import CoTKnowledgePackKey  # LOSSLESS: CoT KV cache injection
    from .denseformer_key import DenseFormerKey
    from .dspark_key import DSparkKey
    from .embedding_key import EmbeddingKey
    from .expert_consolidation_key import ExpertConsolidationKey
    from .fact_injection_key import FactInjectionKey
    from .gateskip_key import GateSkipKey
    from .gqa_to_mqa_key import GQAToMQAKey
    from .grail_key import GRAILKey
    from .hybrid_linear_key import HybridLinearKey  # LOSSY: 75% linear / 25% full attention
    from .knowledge_pack_key import KnowledgePackKey
    from .kv4bit_key import KV4BitKey  # LOSSY: 4-bit KV cache with scale absorption
    from .liquid_conv_key import LiquidConvKey
    from .lm_head_tied_key import LMHeadTiedKey
    from .logit_cap_key import LogitCapKey
    from .lossless_quant_key import LosslessQuantKey
    from .moe_router_key import MoERouterKey
    from .mrl_key import MRLKey
    from .mtp_key import MTPKey
    from .norm_folding_key import NormFoldingKey
    from .partial_rope_key import PartialRoPEKey
    from .qk_norm_mla_key import QKNormMLAKey
    from .quarot_key import QuaRotR2Key
    from .rmsnorm_key import RMSNormKey
    from .rope_key import RoPEKey
    from .rotorquant_key import RotorQuantKey
    from .sandwich_norm_key import SandwichNormKey
    from .self_play_key import SelfPlayKey
    from .selfplay_context_patch_key import SelfPlayContextPatchKey  # LOSSLESS: self-play rank-1 patches
    from .slicegpt_key import SliceGPTKey
    from .sparda_key import SparDAKey
    from .spinquant_key import SpinQuantHadamardKey
    from .ssa_key import SSAKey
    from .swiglu_clamp_key import SwiGLUClampKey
    from .test_gated_injection_key import TestGatedFactInjectionKey  # LOSSLESS: test-verified fact injection
    from .value_residual_key import ValueResidualKey
    from .wanda_key import WandaKey
    from .wq_elim_key import WQElimKey

    stack = KeyStack("xp_model")
    # FULL — reversible + data→weight + composable (all 3 criteria)
    stack.add(MTPKey())
    stack.add(ValueResidualKey())
    stack.add(SliceGPTKey())
    stack.add(MRLKey())
    stack.add(RotorQuantKey())
    stack.add(SpinQuantHadamardKey())
    stack.add(QuaRotR2Key())
    stack.add(QKNormMLAKey())   # QK-Norm for MLA (2026 absorption trick)
    stack.add(WQElimKey())      # WQ elimination (saves 25% attention params)
    stack.add(NormFoldingKey()) # Fold RMSNorm into adjacent weights (TaperNorm 2026)
    # BI — existing in source model, exact copy both directions
    stack.add(EmbeddingKey())
    stack.add(RMSNormKey())
    stack.add(LMHeadTiedKey())
    stack.add(RoPEKey())
    stack.add(CausalMaskKey())
    # PARTIAL — weight transform but not reversible (lossy or needs calib)
    stack.add(GQAToMQAKey())
    stack.add(WandaKey())
    stack.add(DSparkKey())
    stack.add(MoERouterKey())
    stack.add(SSAKey())
    stack.add(GateSkipKey())
    stack.add(LiquidConvKey())
    stack.add(SparDAKey())
    stack.add(PartialRoPEKey())
    stack.add(ExpertConsolidationKey())  # merge similar MoE experts (novel)
    stack.add(GRAILKey())                # heal lossy transforms via ridge regression
    stack.add(ActivationTransmuteKey())  # swap SwiGLU → ReGLU/GeGLU (novel)
    stack.add(LosslessQuantKey())        # SpinQuant→QuaRot→int4 chain (sub-8GB VRAM)
    stack.add(FactInjectionKey())        # closed-form fact injection (COLM 2026)
    stack.add(ContextPatchKey())         # ICL → rank-1 weight patches (2026)
    stack.add(SelfPlayKey())             # self-play → closed-form knowledge injection
    # --- New lossy keys (DO NOT apply to V2/expert packs) ---
    stack.add(BinaryG128Key())           # binary g128 quantization (Bonsai-27B, 14.2x)
    stack.add(KV4BitKey())               # 4-bit KV cache quantization
    stack.add(HybridLinearKey())         # 75% linear attention (Bonsai-27B arch)
    # --- New lossless keys (safe for V2/expert packs) ---
    stack.add(TestGatedFactInjectionKey())  # test-verified fact injection (self-play gated)
    stack.add(SelfPlayContextPatchKey())    # self-play rank-1 weight patches
    # TRIVIAL — runtime strategy / identity init, no weights to learn
    stack.add(AirLLMKey())          # layer-streaming inference (70B on 4GB VRAM)
    stack.add(DenseFormerKey())     # depth-weighted averaging (identity init)
    stack.add(LogitCapKey())        # clamp logits to ±30 (runtime)
    stack.add(SwiGLUClampKey())     # GPT-OSS clamped SwiGLU (runtime)
    stack.add(SandwichNormKey())    # post-sublayer RMSNorm (identity init)
    stack.add(AirMoEKey())          # AirLLM+MoE: expert hotswap from disk (novel arch)
    stack.add(KnowledgePackKey())   # zero-token knowledge via KV cache injection (2026)
    stack.add(CoTKnowledgePackKey())  # CoT reasoning traces as KV packs (self-play + 2026)
    # --- Boot-time + storage optimization keys (lossless, 2026-08-08) ---
    from .attn_scale_fold_key import AttnScaleFoldKey
    from .dead_weight_key import DeadWeightKey
    from .rope_share_key import RoPEShareKey
    from .tensor_dedup_key import TensorDedupKey
    stack.add(TensorDedupKey())      # dedup exact-same tensors (467 MB save on V2)
    stack.add(AttnScaleFoldKey())    # fold 1/sqrt(d_k) into q/k_proj (compute save)
    stack.add(DeadWeightKey())       # prune all-zero tensors (dead weight from init keys)
    stack.add(RoPEShareKey())        # share RoPE cos/sin buffers across layers (VRAM save)
    return stack
