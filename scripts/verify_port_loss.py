"""Verify the lossy nature of the LFM2.5 -> V7-8B-B -> V8 port.

Quantifies exactly WHERE information is destroyed in the current warm-start
chain, so the plan can target the right fixes. Runs on CPU with tiny tensors
so it executes in seconds. No GPU/checkpoints needed.

Lossy steps under test:
  1. Width upscale by REPEAT (upscale_weight) vs HyperCloning (function-preserving)
  2. Depth doubling by DUPLICATE (stack) vs identity-residual safe stacking
  3. SVD embedding factorization (rank-512 approx of rank-2048)
  4. Dense FFN -> NLRQ (rank-1024 + INT8 factor quant)
  5. BitNet ternary {-1,0,+1}

For each we report a relative error metric on a representative op.
"""
import torch
import torch.nn.functional as F

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEV}")


def rel_err(a, b):
    return (a.float() - b.float()).norm().item() / (a.float().norm().item() + 1e-12)


def test_width_upscale_repeat_vs_hypercloning():
    """Repeat-upscale changes the function; HyperCloning preserves it."""
    torch.manual_seed(0)
    d_old, d_new = 2048, 4096
    x_old = torch.randn(1, 4, d_old)
    W = torch.randn(d_old, d_old) * 0.02
    y_old = x_old @ W.T  # reference output [1,4,2048]

    # Current port: repeat rows+cols
    reps = d_new // d_old
    W_repeat = W.repeat(reps, reps)
    x_new = x_old.repeat(1, 1, reps)  # [1,4,4096] (also repeated input)
    y_repeat = x_new @ W_repeat.T
    # Compare the first d_old output dims to the reference (should match if repeat
    # of input is consistent). The issue: real inputs are NOT repeated, so the
    # function on arbitrary x is different.
    err_repeat_block = rel_err(y_repeat[..., :d_old], y_old)

    # HyperCloning-style: W_big = [[W, 0],[0, W]] block-diagonal -> function
    # preserving for the duplicated-head interpretation (each new head copies old).
    W_hc = torch.zeros(d_new, d_new)
    W_hc[:d_old, :d_old] = W
    W_hc[d_old:, d_old:] = W
    x_hc = torch.cat([x_old, x_old], dim=-1)
    y_hc = x_hc @ W_hc.T
    err_hc = rel_err(y_hc[..., :d_old], y_old)

    # The real problem: on a FRESH random 4096 input (not repeated), repeat-W
    # produces garbage relative to a "true" 4096 model. Measure output magnitude
    # distortion vs the repeated-input case.
    x_fresh = torch.randn(1, 4, d_new)
    y_fresh_repeat = x_fresh @ W_repeat.T
    # A function-preserving init would map x_fresh[..., :d_old] through W and
    # x_fresh[..., d_old:] through W independently.
    y_fresh_hc = torch.cat([x_fresh[..., :d_old] @ W.T,
                            x_fresh[..., d_old:] @ W.T], dim=-1)
    distortion = rel_err(y_fresh_repeat, y_fresh_hc)

    print(f"[1] Width upscale (repeat vs HyperCloning-block):")
    print(f"    repeat block-match err = {err_repeat_block:.4f} (matches only if input repeated)")
    print(f"    HyperCloning err       = {err_hc:.4f} (function-preserving)")
    print(f"    fresh-input distortion = {distortion:.4f} (repeat-W != true 2x model)")
    return distortion


def test_depth_duplicate():
    """Duplicating a residual layer doubles the residual contribution."""
    torch.manual_seed(0)
    d = 2048
    x = torch.randn(1, 4, d)
    W1 = torch.randn(d, d) * 0.02
    W2 = torch.randn(d, d) * 0.02
    # Single layer: y = x + f(x)
    f = lambda x, W: F.gelu(x @ W.T)
    y_single = x + f(x, W1)
    # Duplicated (stack): y = x + f(x,W1) + f(x + f(x,W1), W1_copy)
    # Using the SAME weights (duplicate) -> residual stream grows.
    y_dup = x + f(x, W1) + f(x + f(x, W1), W1)
    drift = rel_err(y_dup, y_single)
    print(f"[2] Depth doubling by duplicate (residual drift):")
    print(f"    ||y_dup - y_single|| / ||y_single|| = {drift:.4f}")
    print(f"    (non-zero => not function-preserving; needs gate=0 on 2nd copy)")
    return drift


def test_svd_embedding():
    """SVD rank-512 approx of a [vocab, 2048] embedding, then upscaled to 4096."""
    torch.manual_seed(0)
    vocab, d_old, d_new, rank = 4096, 2048, 4096, 512  # tiny vocab for speed
    E = torch.randn(vocab, d_old) * 0.1
    # Port: upscale to 4096 by repeat, then SVD rank-512
    E_up = E.repeat(1, d_new // d_old)
    U, S, Vh = torch.linalg.svd(E_up.float(), full_matrices=False)
    E_recon = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
    err = rel_err(E_recon, E_up)
    # Effective rank of the repeated matrix is only d_old=2048, so rank-512
    # captures 512/2048 of energy at best.
    energy_captured = (S[:rank].pow(2).sum() / S.pow(2).sum()).item()
    print(f"[3] SVD embedding factorization (rank {rank} of eff-rank {d_old}):")
    print(f"    recon rel err        = {err:.4f}")
    print(f"    energy captured      = {energy_captured:.4f}")
    print(f"    => {1-energy_captured:.4f} of embedding info LOST at init")
    return err


def test_nlrq_ffn():
    """Dense FFN -> NLRQ (rank-1024 + INT8 factor quant)."""
    torch.manual_seed(0)
    d, inter, rank = 2048, 8192, 1024  # tiny; real is 4096x16384
    W_gate = torch.randn(inter, d) * 0.02
    W_up = torch.randn(inter, d) * 0.02
    W_down = torch.randn(d, inter) * 0.02
    x = torch.randn(1, 4, d)
    # Dense SwiGLU
    y_dense = (F.silu(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T

    # NLRQ: SVD rank-r then INT8 quantize factors
    def nlrq(W, r):
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
        # INT8 quantize U and V per-row scale
        def q8(t):
            scale = t.abs().max(dim=-1, keepdim=True).values / 127.0
            scale = scale.clamp(min=1e-8)
            return (t / scale).round().clamp(-128, 127).to(torch.int8), scale
        Uq, Us = q8(Ur * Sr.sqrt().unsqueeze(0))
        Vq, Vs = q8((Sr.sqrt().unsqueeze(1) * Vhr))
        return Uq.float() * Us, Vq.float() * Vs  # approx factors

    Ug, Vg = nlrq(W_gate, rank)
    Uu, Vu = nlrq(W_up, rank)
    Ud, Vd = nlrq(W_down, rank)
    W_gate_n = Ug @ Vg
    W_up_n = Uu @ Vu
    W_down_n = Ud @ Vd
    y_nlrq = (F.silu(x @ W_gate_n.T) * (x @ W_up_n.T)) @ W_down_n.T
    err = rel_err(y_nlrq, y_dense)
    print(f"[4] Dense FFN -> NLRQ (rank {rank}, INT8 factors):")
    print(f"    output rel err = {err:.4f}")
    return err


def test_bitnet_ternary():
    """BitNet b1.58 ternary {-1,0,+1} quantization error."""
    torch.manual_seed(0)
    d = 2048
    W = torch.randn(d, d) * 0.02
    x = torch.randn(1, 4, d)
    y_dense = x @ W.T
    scale = W.abs().mean().item()
    W_t = torch.round(W / (scale + 1e-8)).clamp(-1, 1)
    y_t = x @ (W_t * scale).T
    err = rel_err(y_t, y_dense)
    print(f"[5] BitNet b1.58 ternary:")
    print(f"    output rel err = {err:.4f}")
    return err


if __name__ == "__main__":
    print("=" * 60)
    print("LFM2.5 -> V7-8B-B -> V8 port loss audit (tiny CPU tensors)")
    print("=" * 60)
    d1 = test_width_upscale_repeat_vs_hypercloning()
    d2 = test_depth_duplicate()
    d3 = test_svd_embedding()
    d4 = test_nlrq_ffn()
    d5 = test_bitnet_ternary()
    print("=" * 60)
    print("SUMMARY (relative output error per lossy step):")
    print(f"  width repeat   : {d1:.4f}")
    print(f"  depth duplicate: {d2:.4f}")
    print(f"  SVD embed      : {d3:.4f}")
    print(f"  NLRQ FFN       : {d4:.4f}")
    print(f"  BitNet ternary : {d5:.4f}")
    print("CONCLUSION: the port is lossy at EVERY step. A function-preserving")
    print("warm-start (HyperCloning + gated depth + lossless FFN) is required to")
    print("start V8 at LFM2.5 quality rather than below it.")
