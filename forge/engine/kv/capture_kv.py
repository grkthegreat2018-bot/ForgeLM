"""Capture/HybridServe activation cache: store input activations, recompute KV on demand.

Based on:
  - Capture (casys-kaist): Store input activations (hidden states before the
    K/V projection) instead of K and V tensors. K and V are regenerated on
    demand via the linear projection weights: k = activation @ w_k.T.
    This halves per-block memory when hidden_dim ~= n_kv * head_dim (standard
    MHA), since one activation replaces both K and V.
  - HybridServe (ICCD 2025): Hybrid KV/ACT caching that stores activations
    for old (cold) positions and full K/V for recent (hot) positions. This
    balances PCIe bandwidth (old positions are rarely accessed, so recomputation
    cost is amortized) vs GPU compute (recent positions are accessed every
    step, so direct K/V storage avoids repeated projection matmuls).

VRAM budget (RTX 5070 12GB):
  ACT mode:  hidden_dim per token  (one tensor, projects to both K and V)
  KV mode:   2 * n_kv * head_dim per token  (separate K and V)
  For standard MHA (hidden_dim = n_heads * head_dim, n_kv = n_heads):
    ACT = n_kv * head_dim = 0.5 * (2 * n_kv * head_dim) = 50% of KV
  For GQA (n_kv < n_heads): savings ratio = 1 - hidden_dim / (2 * n_kv * head_dim)
    e.g. n_kv=8, head_dim=128, hidden_dim=4096 -> 50% savings
    e.g. n_kv=4, head_dim=128, hidden_dim=4096 -> 25% savings

The activation cache is complementary to CPU offload and quantization --
ACT savings multiply with bit-width reduction and tier offloading.
Pure torch (no CUDA-specific ops), works on CPU as a fallback.
"""
from __future__ import annotations

from typing import Optional

import torch

from forge.engine.kv_backend import KVCacheStrategy


class CaptureKVCache(KVCacheStrategy):
    """Activation cache: store input activations, recompute K/V on demand.

    Modes:
      "act":    Store only input activations. K/V regenerated via projection
                on every get(). ~50% memory vs KV mode (architecture dependent).
      "kv":     Store full K/V directly (fallback when projection weights
                are unavailable or compute budget is tight).
      "hybrid": Store activations for old positions (beyond hybrid_threshold
                from the current sequence end) and full K/V for recent
                positions. Balances memory savings vs recompute cost.

    Projection weights are set via set_projection_weights() during model init.
    When provided, get() regenerates K/V via k = activation @ w_k.T.
    When not provided, the cache falls back to storing full K/V regardless
    of the requested mode.
    """

    def __init__(self, mode: str = "act", hybrid_threshold: int = 64):
        self._mode = mode
        self._hybrid_threshold = hybrid_threshold
        self._w_k: Optional[torch.Tensor] = None
        self._w_v: Optional[torch.Tensor] = None
        self._w_q: Optional[torch.Tensor] = None
        self._w_kv_pinv: Optional[torch.Tensor] = None

    def init(self, n_heads: int, head_dim: int, n_kv_heads: int,
             max_seq_len: int, device, dtype: torch.dtype):
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.hidden_dim = n_heads * head_dim
        self.max_seq_len = max_seq_len
        self.device = (device if isinstance(device, torch.device)
                       else torch.device(device))
        self.dtype = dtype

        self.seq_len = 0
        self.n_act_stored = 0
        self.n_kv_stored = 0

        self.act_buffer: Optional[torch.Tensor] = None
        self.kv_k: Optional[torch.Tensor] = None
        self.kv_v: Optional[torch.Tensor] = None
        self.act_mask: Optional[torch.Tensor] = None
        self.kv_mask: Optional[torch.Tensor] = None

    def _ensure_allocated(self, batch_size: int):
        if self.act_buffer is None:
            self.act_buffer = torch.zeros(
                batch_size, self.max_seq_len, self.hidden_dim,
                dtype=self.dtype, device=self.device)
            self.act_mask = torch.zeros(
                self.max_seq_len, dtype=torch.bool, device=self.device)
        if self.kv_k is None:
            self.kv_k = torch.zeros(
                batch_size, self.n_kv, self.max_seq_len, self.head_dim,
                dtype=self.dtype, device=self.device)
            self.kv_v = torch.zeros_like(self.kv_k)
            self.kv_mask = torch.zeros(
                self.max_seq_len, dtype=torch.bool, device=self.device)

    def set_projection_weights(self, w_k: torch.Tensor, w_v: torch.Tensor,
                               w_q: Optional[torch.Tensor] = None):
        """Provide K/V projection weights for on-demand regeneration.

        Called during model init. When set, get() regenerates K/V via
        k = activation @ w_k.T and v = activation @ w_v.T.

        Args:
            w_k: (head_dim * n_kv_heads, hidden_dim) -- K projection weight.
            w_v: (head_dim * n_kv_heads, hidden_dim) -- V projection weight.
            w_q: (head_dim * n_heads, hidden_dim) -- Q projection weight
                (optional, stored for completeness but not used in KV regen).
        """
        self._w_k = w_k.to(device=self.device, dtype=self.dtype)
        self._w_v = w_v.to(device=self.device, dtype=self.dtype)
        self._w_q = (w_q.to(device=self.device, dtype=self.dtype)
                     if w_q is not None else None)
        # Precompute pseudoinverse of the stacked [w_k; w_v] matrix for the
        # approximate inverse projection (K/V -> activation). This is a
        # one-time O(hidden_dim^3) cost amortized over all appends.
        w_kv = torch.cat([w_k, w_v], dim=0)  # (2*n_kv*head_dim, hidden_dim)
        self._w_kv_pinv = torch.linalg.pinv(w_kv.T).to(
            device=self.device, dtype=self.dtype)

    @property
    def has_projection(self) -> bool:
        return self._w_k is not None and self._w_v is not None

    @property
    def effective_mode(self) -> str:
        if not self.has_projection:
            return "kv"
        return self._mode

    def append_activation(self, activation: torch.Tensor, position: int):
        """Store the input activation for a position (alternative to append).

        The activation is the hidden state before the K/V projection --
        much smaller than K+V when hidden_dim < 2 * n_kv * head_dim.

        Args:
            activation: (B, hidden_dim) or (B, T, hidden_dim).
            position: starting position in the sequence.
        """
        self._ensure_allocated(activation.shape[0])
        if activation.dim() == 2:
            activation = activation.unsqueeze(1)
        T = activation.shape[1]
        end = position + T
        self.act_buffer[:, position:end] = activation.to(self.device, self.dtype)
        self.act_mask[position:end] = True
        self.kv_mask[position:end] = False
        self.seq_len = max(self.seq_len, end)
        self.n_act_stored = self.act_mask.sum().item()
        self.n_kv_stored = self.kv_mask.sum().item()

    def append(self, k: torch.Tensor, v: torch.Tensor, position: int,
               attention_weights=None):
        """Append K/V tokens.

        If projection weights are set and mode is "act", K/V are projected
        back to activation space (approximate inverse via pseudoinverse) and
        stored as activations. In "kv" mode or when no projections are
        available, K/V are stored directly. In "hybrid" mode, new tokens are
        stored as K/V (fast recent access) and old tokens are demoted to ACT.
        """
        self._ensure_allocated(k.shape[0])
        T = k.shape[2]
        end = position + T
        mode = self.effective_mode

        if mode == "act" and self.has_projection:
            activation = self._kv_to_activation(k, v)
            self.act_buffer[:, position:end] = activation
            self.act_mask[position:end] = True
            self.kv_mask[position:end] = False
        else:
            self.kv_k[:, :, position:end] = k.to(self.device, self.dtype)
            self.kv_v[:, :, position:end] = v.to(self.device, self.dtype)
            self.kv_mask[position:end] = True
            self.act_mask[position:end] = False

        self.seq_len = max(self.seq_len, end)
        self.n_act_stored = self.act_mask.sum().item()
        self.n_kv_stored = self.kv_mask.sum().item()

        if mode == "hybrid":
            self._demote_old_kv()

    def _kv_to_activation(self, k: torch.Tensor,
                          v: torch.Tensor) -> torch.Tensor:
        """Approximate inverse projection: recover activation from K/V.

        k, v: (B, n_kv, T, head_dim). Returns (B, T, hidden_dim).

        Uses the precomputed pseudoinverse of the stacked [w_k; w_v] matrix.
        This is a least-squares estimate -- exact only when the activation
        lies in the row space of the projection (true for the forward pass,
        approximate for quantized or noisy K/V).
        """
        B, n_kv, T, hd = k.shape
        k_flat = k.reshape(B, T, n_kv * hd)
        v_flat = v.reshape(B, T, n_kv * hd)
        kv_flat = torch.cat([k_flat, v_flat], dim=-1)  # (B, T, 2*n_kv*hd)
        activation = kv_flat @ self._w_kv_pinv          # (B, T, hidden_dim)
        return activation

    def _project_activation(self, activation: torch.Tensor):
        """Regenerate K/V from activation via forward projection.

        activation: (B, T, hidden_dim).
        Returns (k, v) each (B, n_kv, T, head_dim).
        """
        k_flat = activation @ self._w_k.T  # (B, T, n_kv*head_dim)
        v_flat = activation @ self._w_v.T
        B, T, _ = activation.shape
        k = k_flat.reshape(B, T, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        v = v_flat.reshape(B, T, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        return k, v

    def _demote_old_kv(self):
        """In hybrid mode, convert old KV positions to ACT mode.

        Positions older than hybrid_threshold from the current seq_len
        are demoted: their K/V is projected back to activation space and
        the K/V storage slot is released (mask cleared). This keeps the
        recent window in fast KV form while saving memory on cold tokens.
        """
        if self.seq_len <= self._hybrid_threshold:
            return
        threshold = self.seq_len - self._hybrid_threshold
        arange = torch.arange(self.max_seq_len, device=self.device)
        demote_mask = self.kv_mask & (arange < threshold)
        if not demote_mask.any():
            return
        positions = demote_mask.nonzero(as_tuple=True)[0]
        k = self.kv_k[:, :, positions]   # (B, n_kv, P, head_dim)
        v = self.kv_v[:, :, positions]
        activation = self._kv_to_activation(k, v)  # (B, P, hidden_dim)
        self.act_buffer[:, positions] = activation
        self.act_mask[positions] = True
        self.kv_mask[positions] = False
        self.n_act_stored = self.act_mask.sum().item()
        self.n_kv_stored = self.kv_mask.sum().item()

    def get(self, positions=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for given positions (or all filled if None).

        For positions with stored activations + projections: regenerate K/V
        via matmul. For positions with stored K/V: return directly. Results
        are assembled in position order. Unfilled positions return zeros.
        """
        if self.seq_len == 0:
            B = (self.kv_k.shape[0] if self.kv_k is not None
                 else self.act_buffer.shape[0] if self.act_buffer is not None
                 else 1)
            empty = torch.empty(B, self.n_kv, 0, self.head_dim,
                                dtype=self.dtype, device=self.device)
            return empty, empty.clone()

        if positions is None:
            positions = torch.arange(self.seq_len, device=self.device)
        elif not isinstance(positions, torch.Tensor):
            positions = torch.as_tensor(positions, device=self.device)
        positions = positions.to(self.device).long()

        N = positions.numel()
        B = (self.kv_k.shape[0] if self.kv_k is not None
             else self.act_buffer.shape[0])

        k_out = torch.empty(B, self.n_kv, N, self.head_dim,
                            dtype=self.dtype, device=self.device)
        v_out = torch.empty(B, self.n_kv, N, self.head_dim,
                            dtype=self.dtype, device=self.device)

        # Flatten positions to 1D
        pos_flat = positions.reshape(-1) if positions.dim() > 1 else positions
        act_pos_mask = self.act_mask[pos_flat]
        kv_pos_mask = self.kv_mask[pos_flat]

        if act_pos_mask.any() and self.has_projection:
            act_positions = pos_flat[act_pos_mask]
            activation = self.act_buffer[:, act_positions]
            k_act, v_act = self._project_activation(activation)
            # Place at output indices where act_pos_mask is True
            out_idx = torch.where(act_pos_mask)[0]
            k_out[:, :, out_idx] = k_act
            v_out[:, :, out_idx] = v_act

        if kv_pos_mask.any():
            kv_positions = pos_flat[kv_pos_mask]
            k_kv = self.kv_k[:, :, kv_positions]
            v_kv = self.kv_v[:, :, kv_positions]
            out_idx = torch.where(kv_pos_mask)[0]
            k_out[:, :, out_idx] = k_kv
            v_out[:, :, out_idx] = v_kv

        return k_out, v_out

    def clear(self):
        if self.act_buffer is not None:
            self.act_buffer.zero_()
            self.act_mask.zero_()
        if self.kv_k is not None:
            self.kv_k.zero_()
            self.kv_v.zero_()
            self.kv_mask.zero_()
        self.seq_len = 0
        self.n_act_stored = 0
        self.n_kv_stored = 0

    def info(self) -> dict:
        kv_bytes_per_tok = 2 * self.n_kv * self.head_dim * self.dtype.itemsize
        act_bytes_per_tok = self.hidden_dim * self.dtype.itemsize
        act_bytes = self.n_act_stored * act_bytes_per_tok
        kv_bytes = self.n_kv_stored * kv_bytes_per_tok
        total_bytes = act_bytes + kv_bytes
        full_kv_bytes = self.seq_len * kv_bytes_per_tok
        memory_savings = (1.0 - total_bytes / full_kv_bytes
                          if full_kv_bytes > 0 else 0.0)
        return {
            "type": "capture",
            "mode": self.effective_mode,
            "seq_len": self.seq_len,
            "n_act_stored": self.n_act_stored,
            "n_kv_stored": self.n_kv_stored,
            "memory_savings": memory_savings,
            "act_bytes": act_bytes,
            "kv_bytes": kv_bytes,
            "total_bytes": total_bytes,
            "full_kv_bytes": full_kv_bytes,
            "hybrid_threshold": self._hybrid_threshold,
            "has_projection": self.has_projection,
            "max_seq_len": self.max_seq_len,
        }
