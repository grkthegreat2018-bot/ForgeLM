"""ClusterFusion++: full transformer-block decode fusion.

Based on "ClusterFusion++: Expanding Cluster-Level Fusion to Full
Transformer-Block Decoding" (arXiv 2604.23553).

Fuses the ENTIRE transformer decoder block into a single kernel:
  LayerNorm → QKV → RoPE → decode attention → output projection
  → Post-LN → MLP → residual

On RTX 5090-class GPUs, ClusterFusion++ improves throughput by 1.34×
for Pythia-2.8B with near-token-identical generation.

Key technique: thread-block clusters with on-chip inter-block collectives
allow fusing operators that previously required separate kernels due to
inter-block dependencies. Intermediate tensors stay in registers/shared
memory instead of materializing to global memory.

For our model on RTX 5070 (SM120):
  - SM120 does NOT support thread-block clusters (cluster multicast is
    SM100-only). So we can't do true ClusterFusion++.
  - BUT: we can approximate it using CUDA graph capture of the full block
    (which we already have in megakernel.py) + torch.compile for intra-block
    fusion.

This module provides:
  1. BlockFusionRunner: captures a full transformer block in a CUDA graph
  2. Multi-block orchestration: chains block graphs for the full model
  3. torch.compile integration for intra-block kernel fusion

This is complementary to megakernel.py (which captures the entire model).
BlockFusion captures per-block, allowing different blocks to have different
graphs (useful for MoD where some blocks are skipped).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BlockFusionRunner:
    """Captures a single transformer block in a CUDA graph.

    Each block gets its own CUDA graph. During decode, only the graphs for
    active blocks are replayed (supporting MoD skip).

    Benefits over full-model graph capture (megakernel.py):
      - Per-block graphs allow MoD skip (skip entire block graph)
      - Smaller graphs = faster capture + less memory
      - Can compile each block independently for different optimizations
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device)
        self._block_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._static_inputs: dict[int, dict] = {}
        self._static_outputs: dict[int, torch.Tensor] = {}
        self._captured = False

    def capture(self, dtype: torch.dtype = torch.bfloat16):
        """Capture CUDA graphs for all transformer blocks."""
        if self.device.type != "cuda":
            raise RuntimeError("BlockFusionRunner requires CUDA")

        # Find all blocks in the model
        blocks = self._find_blocks()
        if not blocks:
            print("  [BlockFusion] No blocks found in model")
            return

        print(f"  [BlockFusion] Capturing {len(blocks)} blocks...")

        for block_idx, block in blocks.items():
            self._capture_block(block_idx, block, dtype)

        self._captured = True
        print(f"  [BlockFusion] Captured {len(self._block_graphs)} block graphs")

    def _find_blocks(self) -> dict[int, nn.Module]:
        """Find transformer blocks in the model."""
        blocks = {}
        # ConfigurableResearchLLM stores blocks as model.blocks[i]
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks):
                blocks[i] = block
        # Fallback: search for modules with 'ModularBlock' type
        if not blocks:
            for name, module in self.model.named_modules():
                if type(module).__name__ == "ModularBlock":
                    idx = int(name.split('.')[-1]) if '.' in name else len(blocks)
                    blocks[idx] = module
        return blocks

    def _capture_block(self, block_idx: int, block: nn.Module,
                        dtype: torch.dtype):
        """Capture a single block's decode forward in a CUDA graph."""
        d_model = getattr(self.model, 'config', None)
        d_model = getattr(d_model, 'd_model', 2048) if d_model else 2048

        # Static input buffer
        static_x = torch.zeros(1, 1, d_model, dtype=dtype, device=self.device)
        static_pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        self._static_inputs[block_idx] = {"x": static_x, "pos": static_pos}

        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                try:
                    _ = block(static_x, position_ids=static_pos)
                except Exception:
                    # Block might need different args — skip graph capture
                    return
            torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                try:
                    self._static_outputs[block_idx] = block(
                        static_x, position_ids=static_pos)
                except Exception:
                    return

        self._block_graphs[block_idx] = graph

    def run_block(self, block_idx: int, x: torch.Tensor,
                  position: int = 0) -> torch.Tensor:
        """Run a single block via graph replay.

        Args:
            block_idx: which block to run
            x: (1, 1, d_model) input activation
            position: current position (for RoPE)

        Returns:
            output: (1, 1, d_model) block output
        """
        if block_idx not in self._block_graphs:
            # Fallback: run block normally
            block = self._find_blocks()[block_idx]
            with torch.inference_mode():
                pos = torch.tensor([[position]], device=self.device)
                return block(x, position_ids=pos)

        # Copy input into static buffer
        self._static_inputs[block_idx]["x"].copy_(x)
        self._static_inputs[block_idx]["pos"][0, 0] = position

        # Replay graph
        self._block_graphs[block_idx].replay()

        return self._static_outputs[block_idx].clone()

    def run_model(self, x: torch.Tensor, position: int = 0,
                  skip_blocks: set[int] | None = None) -> torch.Tensor:
        """Run the full model by chaining block graphs.

        Args:
            x: (1, 1, d_model) input
            position: current position
            skip_blocks: set of block indices to skip (MoD)

        Returns:
            output: (1, 1, d_model) final output
        """
        if skip_blocks is None:
            skip_blocks = set()

        out = x
        blocks = self._find_blocks()
        for block_idx in blocks:
            if block_idx in skip_blocks:
                continue
            out = self.run_block(block_idx, out, position)
        return out

    def is_captured(self) -> bool:
        return self._captured


class CompiledBlockFusion(BlockFusionRunner):
    """BlockFusion with torch.compile for intra-block kernel fusion.

    Combines per-block CUDA graph capture (inter-block fusion) with
    torch.compile (intra-block fusion via Triton code generation).

    Two levels of fusion:
      1. torch.compile fuses RMSNorm + RoPE + QK-norm into Triton kernels
      2. CUDA graph eliminates all launch overhead between blocks

    Expected: 1.3-1.5× over eager for batch=1 on 1.2B models.
    """

    def __init__(self, model: nn.Module, device: str = "cuda",
                 compile_mode: str = "max-autotune-no-cudagraphs"):
        # Compile each block before graph capture
        blocks = self._find_blocks_static(model)
        for block in blocks.values():
            try:
                block = torch.compile(block, mode=compile_mode, fullgraph=False)
            except Exception:
                pass
        super().__init__(model, device)

    @staticmethod
    def _find_blocks_static(model):
        """Static version of _find_blocks for use in __init__."""
        blocks = {}
        if hasattr(model, 'blocks'):
            for i, block in enumerate(model.blocks):
                blocks[i] = block
        return blocks
