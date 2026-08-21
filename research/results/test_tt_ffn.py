"""Quick correctness test for TTLinear / TTSwiGLUFFN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from research.keys.compression.tt_ffn_key import TTLinear, TTSwiGLUFFN

torch.manual_seed(0)

# 1. TTLinear round-trip (small, high rank → near-exact)
W = torch.randn(64, 48)
tt = TTLinear.from_dense(W, tt_rank=8)
x = torch.randn(3, 48)
err = (x @ W.t() - tt(x)).abs().max().item()
print(f"TTLinear 64x48 rank8: max_err={err:.6f}  params={sum(p.numel() for p in tt.cores)}/{W.numel()}")

# 2. weight property matches forward
err_w = (W - tt.weight).abs().max().item()
print(f"  weight reconstruction err={err_w:.6f}")

# 3. Realistic dims
Wbig = torch.randn(8192, 2048)
ttbig = TTLinear.from_dense(Wbig, tt_rank=4)
xbig = torch.randn(2, 2048)
err_big = (xbig @ Wbig.t() - ttbig(xbig)).abs().max().item()
n_big = sum(p.numel() for p in ttbig.cores)
print(f"TTLinear 8192x2048 rank4: max_err={err_big:.6f}  params={n_big}/{Wbig.numel()} ratio={n_big/Wbig.numel():.4f}")

# 4. Odd dims (padding path)
Wodd = torch.randn(100, 70)
ttodd = TTLinear.from_dense(Wodd, tt_rank=16)
xodd = torch.randn(5, 70)
err_odd = (xodd @ Wodd.t() - ttodd(xodd)).abs().max().item()
print(f"TTLinear 100x70 rank16: max_err={err_odd:.6f}  in_pad={ttodd.in_pad} out_pad={ttodd.out_pad}")

# 5. TTSwiGLUFFN
class SwiGLUFFN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

ffn = SwiGLUFFN(256, 512)
tt_ffn = TTSwiGLUFFN.from_dense_ffn(ffn, tt_rank=16)
xt = torch.randn(4, 256)
err_ffn = (ffn(xt) - tt_ffn(xt)).abs().max().item()
n_tt = sum(p.numel() for p in tt_ffn.parameters())
n_dense = sum(p.numel() for p in ffn.parameters())
print(f"TTSwiGLUFFN 256x512 rank16: max_err={err_ffn:.6f}  params={n_tt}/{n_dense} ratio={n_tt/n_dense:.3f}")
print("OK")
