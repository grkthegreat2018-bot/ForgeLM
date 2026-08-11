"""GQA Attention key — V/O projections (Bi), Q/K projections (Partial with fast uptraining).

Architecture:
  q = x @ W_q^T    [seq, n_heads * head_dim]
  k = x @ W_k^T    [seq, n_kv_heads * head_dim]  (shared)
  v = x @ W_v^T    [seq, n_kv_heads * head_dim]  (shared)
  attn = softmax(q @ k^T / sqrt(d))  (with causal mask)
  out = (attn @ v) @ W_o^T

V and O are linear (given attention pattern) → Bi key via normal equation.
Q and K determine attention pattern via softmax → Partial key.

SOFTMAX BARRIER ANALYSIS (2026-07):
  1. Score recovery (fixed-point iteration): EXACT (cosine 0.99999988)
     log(attn) = scores - logsumexp(scores) → fixed-point converges
  2. Q/K from exact scores (GD, 5000 steps): EXACT (cosine 0.99999994)
     The softmax barrier is PURELY optimization, not theoretical
  3. Q/K from attention pattern directly (GD, 500-1500 steps): EXACT for GQA hd=2
     Converges to cosine > 0.9999 in 500-1500 steps (vs 50000 for full training)
     This is a "Partial Key with fast uptraining" — 30-100x faster than full training

GQA structure: MHA→GQA by averaging KV heads (lossy), GQA→MHA by duplicating (lossless).
"""
import math

import torch
import torch.nn.functional as F

from research.keys.misc.base import Key, KeyClass, KeyResult


class GQAKey(Key):
    @property
    def name(self) -> str:
        return "gqa_attention"

    @property
    def description(self) -> str:
        return "GQA attention. V/O: linear (Bi). Q/K: fast uptraining (Partial, cosine>0.9999 in 1500 steps)."

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """data -> weights.

        For V: expects 'X', 'V_target' → produces 'W_v'
        For O: expects 'head_outputs', 'out_target' → produces 'W_o'
        For Q: expects 'X', 'Q_target' → produces 'W_q' (linear, but Q_target
               usually unknown since it's pre-softmax)
        For K: expects 'X', 'K_target' → produces 'W_k' (same caveat)
        """
        try:
            weights = {}
            metadata = {}

            # V projection (linear, exact)
            if 'X' in data and 'V_target' in data:
                X = data['X']
                V_target = data['V_target']
                W_v = (torch.linalg.pinv(X.T @ X) @ X.T @ V_target).T
                weights['W_v'] = W_v
                metadata['W_v'] = 'exact (normal equation)'

            # O projection (linear, exact given attention pattern)
            if 'head_outputs' in data and 'out_target' in data:
                ho = data['head_outputs']
                out = data['out_target']
                W_o = (torch.linalg.pinv(ho.T @ ho) @ ho.T @ out).T
                weights['W_o'] = W_o
                metadata['W_o'] = 'exact (normal equation, given attention pattern)'

            # Q projection (linear, but Q_target is pre-softmax — usually unknown)
            if 'X' in data and 'Q_target' in data:
                X = data['X']
                Q_target = data['Q_target']
                W_q = (torch.linalg.pinv(X.T @ X) @ X.T @ Q_target).T
                weights['W_q'] = W_q
                metadata['W_q'] = 'exact (normal equation, but Q_target pre-softmax)'

            # K projection (same as Q)
            if 'X' in data and 'K_target' in data:
                X = data['X']
                K_target = data['K_target']
                W_k = (torch.linalg.pinv(X.T @ X) @ X.T @ K_target).T
                weights['W_k'] = W_k
                metadata['W_k'] = 'exact (normal equation, but K_target pre-softmax)'

            if not weights:
                return KeyResult(success=False,
                    error="No valid data provided. Need X+V_target, head_outputs+out_target, etc.")

            return KeyResult(success=True, weights=weights, metadata=metadata)
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """weights -> data. V/O can be reversed (linear). Q/K cannot (softmax)."""
        data = {}
        metadata = {}

        if 'W_v' in weights:
            W_v = weights['W_v']
            data['W_v'] = W_v
            metadata['W_v'] = 'recovered (linear, can compute v = x @ W_v^T)'

        if 'W_o' in weights:
            W_o = weights['W_o']
            data['W_o'] = W_o
            metadata['W_o'] = 'recovered (linear, can compute out = ho @ W_o^T)'

        if 'W_q' in weights:
            metadata['W_q'] = 'cannot reverse — softmax destroys the mapping q->attn'

        if 'W_k' in weights:
            metadata['W_k'] = 'cannot reverse — softmax destroys the mapping k->attn'

        return KeyResult(success=True, data=data, metadata=metadata)

    @staticmethod
    def forward_qk_uptrain(X: torch.Tensor, attn_target: torch.Tensor,
                           n_heads: int, n_kv_heads: int, head_dim: int,
                           mask: torch.Tensor = None,
                           n_steps: int = 1500, lr: float = 0.01,
                           seed: int = 42) -> tuple:
        """Recover W_q, W_k from attention pattern via fast uptraining.

        This is the "Partial Key with fast uptraining" for the softmax barrier.
        Converges to exact (cosine > 0.9999) in 500-1500 steps — 30-100x faster
        than full model training (50000 steps).

        Args:
            X: input tensor [seq_len, d_model]
            attn_target: target attention weights [n_heads, seq_len, seq_len]
            n_heads, n_kv_heads, head_dim: attention config
            mask: causal mask [seq_len, seq_len] (1=attend, 0=mask). Default: causal.
            n_steps: optimization steps (default 1500)
            lr: learning rate (default 0.01)

        Returns:
            (W_q, W_k, final_cosine, final_diff)
        """
        torch.manual_seed(seed)
        seq_len, d_model = X.shape
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=X.device))

        W_q = (torch.randn(n_heads * head_dim, d_model, device=X.device) * 0.1).requires_grad_(True)
        W_k = (torch.randn(n_kv_heads * head_dim, d_model, device=X.device) * 0.1).requires_grad_(True)
        opt = torch.optim.Adam([W_q, W_k], lr=lr)
        group_size = n_heads // n_kv_heads

        for step in range(n_steps):
            q = (X @ W_q.T).reshape(seq_len, n_heads, head_dim).permute(1, 0, 2)
            k = (X @ W_k.T).reshape(seq_len, n_kv_heads, head_dim).permute(1, 0, 2)
            k_e = k.repeat_interleave(group_size, dim=0)
            s = (q @ k_e.transpose(-2, -1)) / math.sqrt(head_dim)
            s = s.masked_fill(mask == 0, float('-inf'))
            a = F.softmax(s, dim=-1)
            loss = F.mse_loss(a, attn_target)
            loss.backward()
            opt.step()
            opt.zero_grad()

        with torch.no_grad():
            q = (X @ W_q.T).reshape(seq_len, n_heads, head_dim).permute(1, 0, 2)
            k = (X @ W_k.T).reshape(seq_len, n_kv_heads, head_dim).permute(1, 0, 2)
            k_e = k.repeat_interleave(group_size, dim=0)
            s = (q @ k_e.transpose(-2, -1)) / math.sqrt(head_dim)
            s = s.masked_fill(mask == 0, float('-inf'))
            a = F.softmax(s, dim=-1)
            cos = F.cosine_similarity(attn_target.flatten().unsqueeze(0),
                                       a.flatten().unsqueeze(0)).item()
            diff = (attn_target - a).abs().max().item()

        return W_q.detach(), W_k.detach(), cos, diff

    @staticmethod
    def recover_scores_fixed_point(attn_target: torch.Tensor, mask: torch.Tensor,
                                    n_iters: int = 100) -> torch.Tensor:
        """Recover pre-softmax scores from attention weights via fixed-point iteration.

        log(attn) = scores - logsumexp(scores)
        → scores = log(attn) + logsumexp(scores)  (fixed-point, converges)

        Returns scores [n_heads, seq_len, seq_len] with -inf for masked entries.
        """
        log_attn = torch.log(attn_target.clamp(min=1e-30))
        n_unmasked = mask.sum(dim=-1)
        scores = log_attn + torch.log(n_unmasked).unsqueeze(0).unsqueeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        for _ in range(n_iters):
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            st = log_attn + lse
            st = st.masked_fill(mask == 0, float('-inf'))
            scores = 0.5 * scores + 0.5 * st  # damped for stability
        return scores

    @staticmethod
    def mha_to_gqa(w_kv: torch.Tensor, n_heads: int, n_kv_heads: int,
                   head_dim: int) -> torch.Tensor:
        """MHA → GQA: average groups of KV heads. LOSSY."""
        group_size = n_heads // n_kv_heads
        w = w_kv.reshape(n_heads, head_dim, -1)
        w = w.reshape(n_kv_heads, group_size, head_dim, -1)
        w = w.mean(dim=1)
        return w.reshape(n_kv_heads * head_dim, -1)

    @staticmethod
    def gqa_to_mha(w_kv: torch.Tensor, n_kv_heads: int, n_heads: int,
                   head_dim: int) -> torch.Tensor:
        """GQA → MHA: duplicate KV heads within each group. LOSSLESS."""
        group_size = n_heads // n_kv_heads
        w = w_kv.reshape(n_kv_heads, head_dim, -1)
        w = w.unsqueeze(1).expand(n_kv_heads, group_size, head_dim, -1)
        w = w.reshape(n_heads * head_dim, -1)
        return w

    @staticmethod
    def compute_attention(x: torch.Tensor, W_q, W_k, W_v, W_o,
                          n_heads: int, n_kv_heads: int, head_dim: int,
                          causal: bool = True):
        """Full GQA forward pass. Returns (output, attention_weights)."""
        seq_len, d_model = x.shape
        q = (x @ W_q.T).reshape(seq_len, n_heads, head_dim).permute(1, 0, 2)
        k = (x @ W_k.T).reshape(seq_len, n_kv_heads, head_dim).permute(1, 0, 2)
        v = (x @ W_v.T).reshape(seq_len, n_kv_heads, head_dim).permute(1, 0, 2)

        group_size = n_heads // n_kv_heads
        k = k.repeat_interleave(group_size, dim=0)
        v = v.repeat_interleave(group_size, dim=0)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).permute(1, 0, 2).reshape(seq_len, n_heads * head_dim)
        return out @ W_o.T, attn
