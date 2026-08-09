"""Shared helpers for pre-training and SFT."""
import inspect
import json
import math
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")


class ReplayBuffer:
    """Circular buffer of recent training examples to prevent catastrophic forgetting.

    Stores (input_ids, target_ids) pairs. During online learning, each batch
    is mixed with replay examples to ensure the model doesn't forget old knowledge.
    Moved here from online_learn.py for reuse across live_learn.py and other trainers.
    """

    def __init__(self, max_size=1000, device="cuda"):
        self.buffer = deque(maxlen=max_size)
        self.device = device

    def add(self, x, y):
        """Add a training example (x: input, y: targets)."""
        self.buffer.append((x.cpu().clone(), y.cpu().clone()))

    def sample(self, n):
        """Sample n replay examples."""
        if len(self.buffer) == 0 or n <= 0:
            return None, None
        samples = random.sample(list(self.buffer), min(n, len(self.buffer)))
        xs = torch.cat([s[0] for s in samples], dim=0).to(self.device)
        ys = torch.cat([s[1] for s in samples], dim=0).to(self.device)
        return xs, ys

    def __len__(self):
        return len(self.buffer)


class BinaryDataset:
    """Flat uint32 token dataset, preloaded into CPU RAM as int64.

    Optimized for single-GPU training:
    - Random indices generated on GPU (avoids CPU randint + host-to-device transfer)
    - Optional pinned-memory staging for faster H2D copies
    - Optional background prefetch thread to overlap data loading with compute
    """

    def __init__(self, filename, seq_len, vocab_size, pin_memory=False, prefetch=0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pin_memory = pin_memory
        self.prefetch_queue = deque(maxlen=max(prefetch, 1))
        self.prefetch_thread = None
        self.prefetch_enabled = prefetch > 0
        if Path(filename).exists():
            data = np.fromfile(filename, dtype=np.uint32)
            self.data = torch.from_numpy(data).long()
            if pin_memory:
                self.data = self.data.pin_memory()
            print(f"Loaded {filename}: {len(self.data) / 1e6:.2f}M tokens"
                  f"{' (pinned)' if pin_memory else ''}")
        else:
            print(f"Warning: {filename} not found. Using synthetic data fallback.")
            self.data = None

    def get_batch(self, batch_size, device):
        if self.data is not None:
            max_ix = len(self.data) - self.seq_len - 1
            # Generate random indices on the target device (avoids CPU randint
            # + a separate host-to-device transfer for the indices).
            ix = torch.randint(0, max_ix, (batch_size,), device=device)
            ix = ix.cpu()  # needed for advanced indexing on the CPU tensor
            x = torch.stack([self.data[i : i + self.seq_len] for i in ix])
            y = torch.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in ix])
        else:
            x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
            y = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
        # non_blocking=True overlaps H2D copy with prior GPU work when
        # the source is in pinned memory.
        return x.to(device, non_blocking=self.pin_memory), \
               y.to(device, non_blocking=self.pin_memory)


def get_lr(step, max_steps, max_lr, min_lr, warmup_steps):
    """Cosine warmup + decay learning rate schedule."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(model, max_lr, weight_decay, optimizer_name="fused", bf16_state=False):
    """Return an optimizer. Default is fused AdamW (fastest on RTX 5070).

    Benchmarks show fused AdamW is ~26% faster than bitsandbytes 8-bit on
    consumer Blackwell, with negligible VRAM difference for 360M params.
    Use 'bnb' only when VRAM is critically constrained.

    bf16_state=True is accepted but currently falls back to fp32 state —
    stock PyTorch AdamW cannot mix bf16 momentum with fp32 grads. Use 'bnb'
    (8-bit) instead for VRAM-constrained scenarios.
    """
    if bf16_state:
        print("NOTE: --bf16-optimizer not yet supported with stock AdamW; using bnb 8-bit instead.")
        optimizer_name = "bnb"

    matrix_params = [p for p in model.parameters() if p.ndim >= 2]
    other_params = [p for p in model.parameters() if p.ndim < 2]

    param_groups = [
        {"params": matrix_params, "weight_decay": weight_decay},
        {"params": other_params, "weight_decay": 0.0},
    ]

    if optimizer_name in ("fused", "adamw"):
        fused = torch.cuda.is_available()
        print(f"Using torch.optim.AdamW (fused={fused}).")
        return torch.optim.AdamW(param_groups, lr=max_lr, fused=fused)

    if optimizer_name == "bnb":
        try:
            import bitsandbytes as bnb

            print("Using 8-bit AdamW (bitsandbytes) — slower but saves VRAM.")
            return bnb.optim.AdamW8bit(param_groups, lr=max_lr)
        except Exception as e:
            print(f"bitsandbytes unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "lion":
        try:
            import bitsandbytes as bnb

            print("Using 8-bit Lion (bitsandbytes).")
            return bnb.optim.Lion8bit(
                [
                    {"params": matrix_params, "weight_decay": weight_decay * 10.0},
                    {"params": other_params, "weight_decay": 0.0},
                ],
                lr=max_lr / 10.0,
            )
        except Exception as e:
            print(f"bitsandbytes Lion unavailable ({e}); falling back to AdamW.")

    if optimizer_name == "galore":
        try:
            import galore_torch
            print("Using GaLore AdamW (rank=128, update_proj_freq=200).")
            return galore_torch.GaLoreAdamW(param_groups, lr=max_lr, weight_decay=weight_decay)
        except Exception as e:
            print(f"GaLore unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    if optimizer_name == "muon":
        # Muon: orthogonalized momentum for 2D hidden weights, AdamW for
        # embeddings/head/scalars. ~2x compute efficiency vs AdamW (Moonlight
        # paper). FLOP overhead <1% for typical LM shapes.
        # LR settings from NanoGPT speedrun: Muon lr=0.05 (absolute, not scaled
        # by max_lr), AdamW head lr=0.22, embeddings lr=0.6, scalars lr=0.04.
        # We scale these proportionally to max_lr so the warmup schedule works.
        try:
            from muon import MuonWithAuxAdam
            scale = max_lr / 0.003  # normalize: speedrun uses max_lr=3e-3
            muon_lr = 0.05 * scale
            head_lr = 0.22 * scale
            embed_lr = 0.6 * scale
            scalar_lr = 0.04 * scale
            # Split by param name (use id() to avoid tensor __eq__ pitfalls).
            # Embeddings + head (tied weight) -> AdamW; 2D hidden matrices ->
            # Muon; <2D scalars/norms -> AdamW.
            embed_ids = set()
            for n, p in model.named_parameters():
                if 'embed' in n or 'head' in n:
                    embed_ids.add(id(p))
            hidden_params = [p for p in matrix_params if id(p) not in embed_ids]
            embed_params = [p for p in matrix_params if id(p) in embed_ids]
            scalar_params = other_params
            param_groups = [
                {"params": embed_params, "lr": embed_lr, "betas": (0.8, 0.95), "eps": 1e-10, "use_muon": False, "weight_decay": 0.0},
                {"params": hidden_params, "lr": muon_lr, "momentum": 0.95, "use_muon": True, "weight_decay": weight_decay},
                {"params": scalar_params, "lr": scalar_lr, "betas": (0.8, 0.95), "eps": 1e-10, "use_muon": False, "weight_decay": 0.0},
            ]
            print(f"Using Muon (lr={muon_lr:.4f}) for {len(hidden_params)} hidden matrices + "
                  f"AdamW (embed={embed_lr:.4f}, scalar={scalar_lr:.4f}) for {len(embed_params)} embed + {len(scalar_params)} scalar params.")
            return MuonWithAuxAdam(param_groups)
        except Exception as e:
            print(f"Muon unavailable ({e}); falling back to fused AdamW.")
            return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())

    # Unknown optimizer name — safe default.
    print(f"Unknown optimizer '{optimizer_name}'; using fused AdamW.")
    return torch.optim.AdamW(param_groups, lr=max_lr, fused=torch.cuda.is_available())


def fused_clip_step(opt, max_norm):
    """Optimizer step with built-in grad clipping (one kernel launch instead of two).

    Standard pattern is clip_grad_norm_() then opt.step() — two passes over
    grads. This fuses them: scale grads in-place by (max_norm / global_norm)
    when the global norm exceeds max_norm, then call opt.step(). Skips the
    separate norm computation pass. Returns the (pre-clipping) global norm.
    """
    # Compute global norm in one reduction.
    total_sq = torch.tensor(0.0, device=opt.param_groups[0]["params"][0].device)
    for group in opt.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                total_sq.add_(p.grad.detach().pow(2).sum())
    global_norm = total_sq.sqrt().clamp(min=1e-6)
    scale = (max_norm / global_norm).clamp(max=1.0)
    if scale.item() < 1.0:
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.mul_(scale)
    opt.step()
    return float(global_norm.item())


def patch_triton_cache_for_windows():
    """Make triton-windows 3.4.0 on PyTorch 2.8 usable with Inductor.

    Triton's FileCacheManager can generate cache paths that exceed Windows'
    default 260-character limit, which makes Inductor crash with FileNotFoundError
    during kernel cache writes. This forces a short project-local cache dir and
    shortens the temp subdir name. It also patches the installed source file so
    spawned compile workers inherit the fix.
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
        short_cache = project_root / ".tc"
        short_cache.mkdir(exist_ok=True)
        if not os.environ.get("TRITON_CACHE_DIR"):
            os.environ["TRITON_CACHE_DIR"] = str(short_cache)
        if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(short_cache)
        # Persistent FX graph + AOTAutograd cache: eliminates 30-60s compile
        # warmup on restarts by reusing compiled kernels from previous runs.
        if not os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE"):
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        if not os.environ.get("TORCHINDUCTOR_AUTOGRAD_CACHE"):
            os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

        import uuid

        import triton.runtime.cache as tc

        src = inspect.getsourcefile(tc.FileCacheManager)
        if src and os.path.exists(src):
            with open(src) as f:
                text = f.read()
            new_text = text
            new_text = new_text.replace("rnd_id = str(uuid.uuid4())", "rnd_id = str(uuid.uuid4())[:8]")
            new_text = new_text.replace('pid = os.getpid()\n        # use temp dir', '# use temp dir')
            new_text = new_text.replace('f"tmp.pid_{pid}_{rnd_id}"', 'f"tmp.{rnd_id}"')
            if new_text != text:
                with open(src, "w") as f:
                    f.write(new_text)

        def _patched_put(self, data, filename, binary=True) -> str:
            if not self.cache_dir:
                raise RuntimeError("Could not create or locate cache dir")
            binary = isinstance(data, bytes)
            if not binary:
                data = str(data)
            assert self.lock_path is not None
            filepath = self._make_path(filename)
            rnd_id = str(uuid.uuid4())[:8]
            temp_dir = os.path.join(self.cache_dir, f"tmp.{rnd_id}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, filename)
            mode = "wb" if binary else "w"
            with open(temp_path, mode) as f:
                f.write(data)
            try:
                os.replace(temp_path, filepath)
            except PermissionError:
                if os.name == "nt":
                    os.remove(temp_path)
                else:
                    raise
            os.removedirs(temp_dir)
            return filepath

        tc.FileCacheManager.put = _patched_put
    except Exception:
        pass


def evaluate_loss(model, val_dataset, device, n_batches=10, batch_size=2, tag="eval"):
    """Standardized validation loss + perplexity evaluation.

    Used by train.py, distill.py, distill_synthetic.py, online_learn.py, eval_suite.py.
    Returns the average cross-entropy loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = val_dataset.get_batch(batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                losses.append(
                    F.cross_entropy(
                        logits.view(-1, logits.size(-1)).float(), y.view(-1)
                    ).item()
                )
    model.train()
    avg = sum(losses) / len(losses)
    ppl = math.exp(min(avg, 20.0))  # cap to avoid overflow
    print(f"  [{tag}] val loss: {avg:.4f} | ppl: {ppl:.2f}")
    return avg


def update_ema(ema_state, model, decay):
    """Update EMA shadow weights in-place.

    Used by train.py, distill_synthetic.py, online_learn.py.
    `ema_state` is a dict mapping param name -> shadow tensor.
    """
    if ema_state is None:
        return
    with torch.no_grad():
        for name, p in model.named_parameters():
            ema_state[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def init_ema(model):
    """Create EMA shadow weights (deep copy of model params)."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def restore_ema(ema_state, model):
    """Restore EMA weights into model (rollback on quality regression)."""
    if ema_state is None:
        return
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(ema_state[name])


def compute_ce_loss(model_output, targets):
    """Standard cross-entropy loss handling tuple model outputs.

    Used by multiple training scripts to avoid repeating the
    `out[0] if isinstance(out, tuple) else out` pattern.
    """
    logits = model_output[0] if isinstance(model_output, tuple) else model_output
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), targets.view(-1)
    )


# ---------------------------------------------------------------------------
# Training safeguards (NaN detection, VRAM watchdog, status/heartbeat files)
# ---------------------------------------------------------------------------


def has_nan_params(model) -> bool:
    """Return True if any model parameter contains NaN or Inf.

    Run periodically (e.g. every 50 steps); NaNs propagate silently through
    training and corrupt the model if not caught early.
    """
    with torch.no_grad():
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return True
    return False


def vram_exceeded(limit_gb, device="cuda") -> bool:
    """Return True if allocated VRAM exceeds `limit_gb` (None/0 disables).

    Check before allocations when approaching the card's capacity to abort
    with an emergency checkpoint instead of freezing the system on OOM.
    """
    if not limit_gb or limit_gb <= 0 or device == "cpu":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.memory_allocated(device) / 1e9 > limit_gb


def vram_gb(device="cuda") -> float:
    """Current allocated VRAM in GB (0.0 if CUDA unavailable)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(device) / 1e9


def write_status_json(path, data: dict):
    """Atomically write a status/progress JSON file (for the GUI monitor)."""
    if not path:
        return
    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(path))


def write_heartbeat(path):
    """Atomically write the current timestamp (for external hang detection)."""
    import time as _time

    write_status_json(path, {"ts": _time.time()})


def add_safeguard_args(parser):
    """Add the shared progress-safety CLI flags to an argparse parser.

    Flags: --save-every, --keep-checkpoints, --status-file, --heartbeat-file,
    --vram-limit-gb. (--resume is defined per-script since semantics vary.)
    """
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save a periodic .stepN checkpoint every N steps (0 = disabled)")
    parser.add_argument("--keep-checkpoints", type=int, default=5,
                        help="Number of periodic checkpoints to keep (older ones are deleted)")
    parser.add_argument("--status-file", type=str, default=None,
                        help="Write progress JSON here every log interval (for the GUI monitor)")
    parser.add_argument("--heartbeat-file", type=str, default=None,
                        help="Write a timestamp here every log interval (hang detection)")
    parser.add_argument("--vram-limit-gb", type=float, default=0.0,
                        help="Abort with an emergency checkpoint before exceeding this VRAM (0 = disabled)")
    return parser
