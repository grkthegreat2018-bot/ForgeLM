"""Boot-time benchmark — novel variations vs baseline.

Implements 6 variations identified in scratchpad research:
  V1: skip_init arch build (torch.nn.utils.skip_init)
  V2: parallel tokenizer load (ThreadPoolExecutor, overlap with arch build)
  V3: OS page cache prefetch (background thread reads 16MB blocks)
  V4: Windows PrefetchVirtualMemory (ctypes, cross-domain from DB indexing)
  V5: meta device init → materialize from state_dict
  V6: combined (V1 + V2 + V3)

Each variation replicates the baseline measurement harness so numbers are
directly comparable.

Run:
    python .devin/boot_bench_variations.py --var V1
    python .devin/boot_bench_variations.py --var all
"""
import argparse
import gc
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Helpers shared across variations
# ---------------------------------------------------------------------------

def _prefetch_file_thread(path: str, block: int = 16 * 1024 * 1024):
    """Background thread: read file in `block`-sized chunks to warm OS page cache.

    Mirrors vLLM PR #36012 prefetch strategy. Reads sequentially so the OS
    readahead predictor kicks in. Non-blocking — caller can join later.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            read = 0
            while read < size:
                chunk = f.read(min(block, size - read))
                if not chunk:
                    break
                read += len(chunk)
    except Exception:
        pass


def _prefetch_virtual_memory_windows(path: str) -> bool:
    """Windows 8+ PrefetchVirtualMemory on an mmap'd file.

    Cross-domain: applies DB indexing page-prefetch theory to safetensors mmap.
    llama.cpp uses this on Windows for mmap'd model files.

    Returns True if the API was called successfully, False otherwise.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORY_RANGE_ENTRY(ctypes.Structure):
            _fields_ = [("VirtualAddress", ctypes.c_void_p),
                        ("NumberOfBytes", ctypes.c_size_t)]

        # mmap the file read-only
        import mmap
        f = open(path, "rb")
        size = os.path.getsize(path)
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)

        ranges = (MEMORY_RANGE_ENTRY * 1)()
        ranges[0].VirtualAddress = ctypes.c_void_p(mm.__buffer__())  # may fail
        ranges[0].NumberOfBytes = size

        kernel32 = ctypes.windll.kernel32
        kernel32.PrefetchVirtualMemory.argtypes = [
            wintypes.HANDLE, ctypes.c_size_t,
            ctypes.POINTER(MEMORY_RANGE_ENTRY), wintypes.ULONG]
        kernel32.PrefetchVirtualMemory.restype = wintypes.BOOL

        hproc = kernel32.GetCurrentProcess()
        ok = kernel32.PrefetchVirtualMemory(
            hproc, 1, ranges, 0)
        mm.close()
        f.close()
        return bool(ok)
    except Exception as e:
        return False


def _load_tokenizer_async():
    """Start tokenizer load in a thread, return (future, executor)."""
    from research.tokenizer_cache import get_tokenizer
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(get_tokenizer, "research/checkpoints/lfm25_tokenizer")
    return fut, ex


def _do_forward(model, tok, device, stages):
    """Common: KV cache alloc + first forward + first decode. Records into stages."""
    import torch
    from research.model_loader import create_kv_cache

    t_kv = time.perf_counter()
    cache = create_kv_cache(model, max_total=2048, batch=1,
                            device=torch.device(device))
    if device == "cuda":
        torch.cuda.synchronize()
    stages["K_kv_cache_alloc"] = time.perf_counter() - t_kv

    t_ff = time.perf_counter()
    ids = tok("The capital of France is", return_tensors="pt")
    if hasattr(ids, "to"):
        ids = ids.to(device)
    else:
        ids = {k: v.to(device) for k, v in ids.items()}
    input_ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
    with torch.no_grad():
        out = model(input_ids, preallocated_cache=cache, use_cache=True)
        logits = out[0]
    if device == "cuda":
        torch.cuda.synchronize()
    stages["L_first_forward"] = time.perf_counter() - t_ff

    t_decode = time.perf_counter()
    with torch.no_grad():
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache.advance()
        out2 = model(next_tok, preallocated_cache=cache, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()
    stages["M_first_decode"] = time.perf_counter() - t_decode
    return cache


def _post_load_scan(model, device, stages):
    """Common: QK-norm + diff-attn identity scan (Stage I)."""
    import torch
    t_scan = time.perf_counter()
    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
            q_id = (attn.q_norm.weight == 1.0).all()
            k_id = (attn.k_norm.weight == 1.0).all()
            attn._qk_norm_identity = bool(q_id and k_id)
        if hasattr(attn, 'lambda_param') and hasattr(attn, 'set_identity'):
            attn.set_identity((attn.lambda_param == 0.0).all().item())
    if device == "cuda":
        torch.cuda.synchronize()
    stages["I_post_load_scan"] = time.perf_counter() - t_scan


def _reset_non_persistent_buffers(model, target_device=None):
    """Re-initialize non-persistent buffers left on meta/empty after meta-init.

    After meta device init + to_empty/assign, buffers registered with
    persistent=False (e.g. RoPE cos/sin tables) are either meta or empty.
    This re-computes them from scratch on the model's current device.

    For ForgeAI, the affected modules are:
    - RotaryEmbedding: inv_freq, cos_cached, sin_cached, cos_cached_bf16, sin_cached_bf16
    """
    import torch
    from research.model_loader import RotaryEmbedding

    if target_device is None:
        # Infer from the first parameter we can find
        target_device = next(model.parameters()).device

    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            base = getattr(module, 'base', 10000.0)
            max_seq_len = getattr(module, 'max_seq_len', module.cos_cached.shape[0])
            rope_scaling = getattr(module, 'rope_scaling', None)
            inv_freq = 1.0 / (base ** (torch.arange(0, module.dim, 2, device=target_device, dtype=torch.float32) / module.dim))
            if rope_scaling and rope_scaling.get("type") == "yarn":
                inv_freq = RotaryEmbedding._yarn_inv_freq(inv_freq, rope_scaling, max_seq_len)
            t = torch.arange(max_seq_len, device=target_device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            module.inv_freq = inv_freq
            module.cos_cached = emb.cos()
            module.sin_cached = emb.sin()
            module.cos_cached_bf16 = emb.cos().to(torch.bfloat16)
            module.sin_cached_bf16 = emb.sin().to(torch.bfloat16)


def _load_weights_and_convert(cfg, checkpoint_path, model, device, stages,
                              assign=False):
    """Common: F (weights load) + G (diff convert) + H (load_state_dict).

    If assign=True, uses load_state_dict(assign=True) which directly replaces
    model params with state_dict tensors (needed for meta-device init).
    """
    import torch
    from research.model_loader import ModelLoader

    if checkpoint_path is None:
        stages["F_weights_load"] = 0.0
        stages["G_diff_convert"] = 0.0
        stages["H_load_state_dict"] = 0.0
        return {}

    t_weights = time.perf_counter()
    state = ModelLoader._load_safetensors_mmap(
        checkpoint_path, model, device=torch.device(device))
    stages["F_weights_load"] = time.perf_counter() - t_weights

    t_conv = time.perf_counter()
    if cfg.attn_type == "diff":
        qk = next((k for k in state if "attn.q_proj.weight" in k), None)
        if qk is not None:
            exp_rows = cfg.n_heads * (cfg.d_model // cfg.n_heads)
            if state[qk].shape[0] == exp_rows:
                from research.keys.attention.differential_attn_key import (
                    DifferentialAttentionKey)
                res = DifferentialAttentionKey(
                    n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                    identity=True).forward(state)
                if res.success:
                    state = res.weights
    stages["G_diff_convert"] = time.perf_counter() - t_conv

    t_lsd = time.perf_counter()
    if assign:
        # assign=True: directly replace params (needed for meta init).
        # This skips the copy-into-existing-storage path and instead
        # swaps the Parameter objects to point at the state_dict tensors.
        missing, unexpected = model.load_state_dict(state, strict=False,
                                                    assign=True)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        stages["H_missing_keys"] = missing[:10]
    if unexpected:
        stages["H_unexpected_keys"] = unexpected[:10]
    if device == "cuda":
        torch.cuda.synchronize()
    stages["H_load_state_dict"] = time.perf_counter() - t_lsd
    return state


# ---------------------------------------------------------------------------
# Variations
# ---------------------------------------------------------------------------

def var_baseline(cfg, checkpoint_path, device, dtype, stages):
    """Reference: identical to boot_bench.py baseline."""
    import torch
    from research.model_loader import ConfigurableResearchLLM

    t_arch = time.perf_counter()
    model = ConfigurableResearchLLM(cfg).to(device)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    t_dtype = time.perf_counter()
    model = model.to(dtype)
    stages["E_dtype_convert"] = time.perf_counter() - t_dtype

    _load_weights_and_convert(cfg, checkpoint_path, model, device, stages)
    _post_load_scan(model, device, stages)
    model.eval()

    t_tok = time.perf_counter()
    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok

    _do_forward(model, tok, device, stages)
    return model, tok


def var_v1_skip_init(cfg, checkpoint_path, device, dtype, stages):
    """V1: torch.nn.utils.skip_init — skip param init kernels.

    Uses a wrapper class that accepts a device arg (required by skip_init).
    """
    import torch
    from research.model_loader import ConfigurableResearchLLM

    class _Wrapper(torch.nn.Module):
        """Thin wrapper that accepts device kwarg for skip_init compatibility."""
        def __init__(self, config, device=None):
            super().__init__()
            if device is not None:
                config = type(config)(**{**config.__dict__, "device": str(device)})
            self.inner = ConfigurableResearchLLM(config)

    t_arch = time.perf_counter()
    wrapper = torch.nn.utils.skip_init(_Wrapper, cfg, device=device)
    model = wrapper.inner
    model = model.to(device)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    t_dtype = time.perf_counter()
    model = model.to(dtype)
    stages["E_dtype_convert"] = time.perf_counter() - t_dtype

    # Re-tie weights (skip_init may break the sharing)
    if getattr(cfg, 'tie_word_embeddings', True) and not getattr(cfg, 'use_pit', False):
        model.head.weight = model.embed.weight

    # Re-initialize non-persistent buffers (RoPE cos/sin) — skip_init uses
    # meta internally, so non-persistent buffers are empty after to_empty.
    _reset_non_persistent_buffers(model, torch.device(device))

    _load_weights_and_convert(cfg, checkpoint_path, model, device, stages)
    _post_load_scan(model, device, stages)
    model.eval()

    t_tok = time.perf_counter()
    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok

    _do_forward(model, tok, device, stages)
    return model, tok


def var_v2_parallel_tokenizer(cfg, checkpoint_path, device, dtype, stages):
    """V2: start tokenizer load in a thread BEFORE arch build, join after."""
    import torch
    from research.model_loader import ConfigurableResearchLLM

    # Start tokenizer load immediately (overlaps with arch build + weight load)
    t_tok_start = time.perf_counter()
    tok_fut, tok_ex = _load_tokenizer_async()
    stages["J_tokenizer_load"] = 0.0  # will set to wall-clock of join below

    t_arch = time.perf_counter()
    model = ConfigurableResearchLLM(cfg).to(device)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    t_dtype = time.perf_counter()
    model = model.to(dtype)
    stages["E_dtype_convert"] = time.perf_counter() - t_dtype

    _load_weights_and_convert(cfg, checkpoint_path, model, device, stages)
    _post_load_scan(model, device, stages)
    model.eval()

    # Join tokenizer thread — measure ONLY the wait time (hidden part)
    t_tok_join = time.perf_counter()
    tok = tok_fut.result()
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok_join
    tok_ex.shutdown(wait=False)
    # Also record total tokenizer wall-clock for visibility
    stages["J_tokenizer_total"] = time.perf_counter() - t_tok_start

    _do_forward(model, tok, device, stages)
    return model, tok


def var_v3_os_prefetch(cfg, checkpoint_path, device, dtype, stages):
    """V3: background thread reads checkpoint in 16MB blocks to warm page cache."""
    import torch
    from research.model_loader import ConfigurableResearchLLM

    # Start prefetch immediately (overlaps with arch build)
    pf_thread = None
    if checkpoint_path:
        pf_thread = threading.Thread(
            target=_prefetch_file_thread, args=(checkpoint_path,),
            daemon=True)
        pf_thread.start()

    t_arch = time.perf_counter()
    model = ConfigurableResearchLLM(cfg).to(device)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    t_dtype = time.perf_counter()
    model = model.to(dtype)
    stages["E_dtype_convert"] = time.perf_counter() - t_dtype

    # Wait for prefetch to finish before weight load (so weights hit warm cache)
    if pf_thread is not None:
        pf_thread.join()
    stages["J_tokenizer_load"] = 0.0  # not parallelized in V3

    _load_weights_and_convert(cfg, checkpoint_path, model, device, stages)
    _post_load_scan(model, device, stages)
    model.eval()

    t_tok = time.perf_counter()
    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok

    _do_forward(model, tok, device, stages)
    return model, tok


def var_v4_prefetch_vm(cfg, checkpoint_path, device, dtype, stages):
    """V4: Windows PrefetchVirtualMemory on mmap'd checkpoint (cross-domain)."""
    import torch
    from research.model_loader import ConfigurableResearchLLM

    # PrefetchVirtualMemory is synchronous but fast (issues async I/O internally)
    t_pf = time.perf_counter()
    if checkpoint_path:
        ok = _prefetch_virtual_memory_windows(checkpoint_path)
    else:
        ok = False
    stages["P_prefetch_vm"] = time.perf_counter() - t_pf
    stages["P_prefetch_vm_ok"] = ok

    t_arch = time.perf_counter()
    model = ConfigurableResearchLLM(cfg).to(device)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    t_dtype = time.perf_counter()
    model = model.to(dtype)
    stages["E_dtype_convert"] = time.perf_counter() - t_dtype

    _load_weights_and_convert(cfg, checkpoint_path, model, device, stages)
    _post_load_scan(model, device, stages)
    model.eval()

    t_tok = time.perf_counter()
    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok

    _do_forward(model, tok, device, stages)
    return model, tok


def var_v5_meta_init(cfg, checkpoint_path, device, dtype, stages):
    """V5: init on meta device (zero alloc), load_state_dict(assign=True).

    Most aggressive: build the model graph on meta (no storage at all),
    then use load_state_dict(assign=True) to directly replace meta params
    with the state_dict tensors on the target device. Skips both init kernels
    AND the .to(device) copy AND the to_empty materialization copy.

    Requires PyTorch 2.1+ for assign=True support.
    """
    import torch
    from research.model_loader import ConfigurableResearchLLM

    t_arch = time.perf_counter()
    # Build on meta — no parameter storage allocated at all
    cfg_meta = type(cfg)(**{**cfg.__dict__, "device": "meta"})
    with torch.device("meta"):
        model = ConfigurableResearchLLM(cfg_meta)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    # Load weights with assign=True — directly replaces meta params.
    # No to_empty, no .to(device), no dtype convert needed — the state_dict
    # tensors are already on the right device + dtype from fastsafetensors.
    stages["E_dtype_convert"] = 0.0
    _load_weights_and_convert(cfg, checkpoint_path, model, device, stages,
                              assign=True)

    # Re-tie weights after assign (assign breaks parameter sharing)
    if getattr(cfg, 'tie_word_embeddings', True) and not getattr(cfg, 'use_pit', False):
        model.head.weight = model.embed.weight

    # Re-initialize non-persistent buffers (RoPE cos/sin) left on meta
    t_buf = time.perf_counter()
    _reset_non_persistent_buffers(model, torch.device(device))
    if device == "cuda":
        torch.cuda.synchronize()
    stages["E_dtype_convert"] = time.perf_counter() - t_buf

    _post_load_scan(model, device, stages)
    model.eval()

    t_tok = time.perf_counter()
    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok

    _do_forward(model, tok, device, stages)
    return model, tok


def var_v6_combined(cfg, checkpoint_path, device, dtype, stages):
    """V6: V1 (skip_init) + V2 (parallel tokenizer) + V3 (OS prefetch)."""
    import torch
    from research.model_loader import ConfigurableResearchLLM

    # Start tokenizer + prefetch immediately
    t_tok_start = time.perf_counter()
    tok_fut, tok_ex = _load_tokenizer_async()
    pf_thread = None
    if checkpoint_path:
        pf_thread = threading.Thread(
            target=_prefetch_file_thread, args=(checkpoint_path,),
            daemon=True)
        pf_thread.start()
    stages["J_tokenizer_load"] = 0.0

    # skip_init arch build (using wrapper that accepts device arg)
    t_arch = time.perf_counter()

    class _Wrapper(torch.nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            if device is not None:
                config = type(config)(**{**config.__dict__, "device": str(device)})
            self.inner = ConfigurableResearchLLM(config)

    wrapper = torch.nn.utils.skip_init(_Wrapper, cfg, device=device)
    model = wrapper.inner
    model = model.to(device)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    t_dtype = time.perf_counter()
    model = model.to(dtype)
    stages["E_dtype_convert"] = time.perf_counter() - t_dtype

    # Re-tie weights (skip_init may break the sharing)
    if getattr(cfg, 'tie_word_embeddings', True) and not getattr(cfg, 'use_pit', False):
        model.head.weight = model.embed.weight

    # Re-initialize non-persistent buffers (RoPE cos/sin)
    _reset_non_persistent_buffers(model, torch.device(device))

    # Wait for prefetch before weight load
    if pf_thread is not None:
        pf_thread.join()

    _load_weights_and_convert(cfg, checkpoint_path, model, device, stages)
    _post_load_scan(model, device, stages)
    model.eval()

    # Join tokenizer
    t_tok_join = time.perf_counter()
    tok = tok_fut.result()
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok_join
    tok_ex.shutdown(wait=False)
    stages["J_tokenizer_total"] = time.perf_counter() - t_tok_start

    _do_forward(model, tok, device, stages)
    return model, tok


VARIATIONS = {
    "baseline": var_baseline,
    "V1": var_v1_skip_init,
    "V2": var_v2_parallel_tokenizer,
    "V3": var_v3_os_prefetch,
    "V4": var_v4_prefetch_vm,
    "V5": var_v5_meta_init,
    "V6": var_v6_combined,
    "V7": None,  # placeholder, set below
    "V8": None,  # placeholder, set below
}


def var_v7_optimal(cfg, checkpoint_path, device, dtype, stages):
    """V7: V5 (meta+assign) + V2 (parallel tokenizer) + V3 (OS prefetch).

    The optimal combination: meta-init with assign=True (fastest arch build),
    parallel tokenizer load (hidden behind arch build), and OS page cache
    prefetch (overlaps with arch build, warms cache for weight load).
    """
    import torch
    from research.model_loader import ConfigurableResearchLLM

    # Start tokenizer + prefetch immediately (overlap with arch build)
    t_tok_start = time.perf_counter()
    tok_fut, tok_ex = _load_tokenizer_async()
    pf_thread = None
    if checkpoint_path:
        pf_thread = threading.Thread(
            target=_prefetch_file_thread, args=(checkpoint_path,),
            daemon=True)
        pf_thread.start()
    stages["J_tokenizer_load"] = 0.0

    # Meta-init arch build (fastest: no param alloc, no init kernels)
    t_arch = time.perf_counter()
    cfg_meta = type(cfg)(**{**cfg.__dict__, "device": "meta"})
    with torch.device("meta"):
        model = ConfigurableResearchLLM(cfg_meta)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    # Wait for prefetch before weight load (warms OS page cache)
    if pf_thread is not None:
        pf_thread.join()

    # Load weights with assign=True (directly replaces meta params)
    stages["E_dtype_convert"] = 0.0
    _load_weights_and_convert(cfg, checkpoint_path, model, device, stages,
                              assign=True)

    # Re-tie weights (assign breaks sharing; head.weight not in checkpoint)
    if getattr(cfg, 'tie_word_embeddings', True) and not getattr(cfg, 'use_pit', False):
        model.head.weight = model.embed.weight

    # Re-initialize RoPE non-persistent buffers
    t_buf = time.perf_counter()
    _reset_non_persistent_buffers(model, torch.device(device))
    if device == "cuda":
        torch.cuda.synchronize()
    stages["E_dtype_convert"] = time.perf_counter() - t_buf

    _post_load_scan(model, device, stages)
    model.eval()

    # Join tokenizer (should be done by now — hidden behind arch+weights)
    t_tok_join = time.perf_counter()
    tok = tok_fut.result()
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok_join
    tok_ex.shutdown(wait=False)
    stages["J_tokenizer_total"] = time.perf_counter() - t_tok_start

    _do_forward(model, tok, device, stages)
    return model, tok


VARIATIONS["V7"] = var_v7_optimal


def var_v8_parallel_weights(cfg, checkpoint_path, device, dtype, stages):
    """V8: V7 + parallel state_dict load (overlap weight I/O with meta init).

    The key insight: state_dict loading (fastsafetensors, ~1.5s) is I/O bound
    and can run in a background thread while meta init (~1.1s) happens in the
    main thread. This hides the weight load behind the arch build.

    Flow:
    1. Start tokenizer + OS prefetch + state_dict load (all background)
    2. Meta init in main thread (overlaps with step 1)
    3. Wait for state_dict load
    4. GQA->diff convert (CPU, fast)
    5. assign=True (replaces meta params with loaded GPU tensors)
    6. Re-tie + buffer reset + post-load scan
    """
    import torch
    from research.model_loader import (ConfigurableResearchLLM, ModelLoader)

    # Start tokenizer immediately
    t_tok_start = time.perf_counter()
    tok_fut, tok_ex = _load_tokenizer_async()
    stages["J_tokenizer_load"] = 0.0

    # Start state_dict load in background thread (overlaps with meta init)
    state_result = {}
    def _load_state():
        try:
            state_result["state"] = ModelLoader._load_safetensors_mmap(
                checkpoint_path, None, device=torch.device(device))
        except Exception as e:
            state_result["error"] = str(e)

    state_thread = None
    if checkpoint_path:
        state_thread = threading.Thread(target=_load_state, daemon=True)
        state_thread.start()

    # Meta init in main thread (overlaps with state_dict load)
    t_arch = time.perf_counter()
    cfg_meta = type(cfg)(**{**cfg.__dict__, "device": "meta"})
    with torch.device("meta"):
        model = ConfigurableResearchLLM(cfg_meta)
    stages["D_arch_build"] = time.perf_counter() - t_arch

    # Wait for state_dict load to complete
    t_weights_wait = time.perf_counter()
    if state_thread is not None:
        state_thread.join()
    if "error" in state_result:
        raise RuntimeError(f"Background weight load failed: {state_result['error']}")
    state = state_result.get("state", {})
    stages["F_weights_load"] = time.perf_counter() - t_weights_wait

    # GQA -> diff warm start (CPU-side transform on the loaded state_dict)
    t_conv = time.perf_counter()
    if cfg.attn_type == "diff":
        qk = next((k for k in state if "attn.q_proj.weight" in k), None)
        if qk is not None:
            exp_rows = cfg.n_heads * (cfg.d_model // cfg.n_heads)
            if state[qk].shape[0] == exp_rows:
                from research.keys.attention.differential_attn_key import (
                    DifferentialAttentionKey)
                res = DifferentialAttentionKey(
                    n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                    identity=True).forward(state)
                if res.success:
                    state = res.weights
    stages["G_diff_convert"] = time.perf_counter() - t_conv

    # assign=True: directly replace meta params with loaded GPU tensors
    stages["E_dtype_convert"] = 0.0
    t_lsd = time.perf_counter()
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if device == "cuda":
        torch.cuda.synchronize()
    stages["H_load_state_dict"] = time.perf_counter() - t_lsd

    # Re-tie weights (assign breaks sharing; head.weight not in checkpoint)
    if getattr(cfg, 'tie_word_embeddings', True) and not getattr(cfg, 'use_pit', False):
        model.head.weight = model.embed.weight

    # Re-initialize RoPE non-persistent buffers
    t_buf = time.perf_counter()
    ModelLoader._reset_non_persistent_buffers(model, torch.device(device))
    if device == "cuda":
        torch.cuda.synchronize()
    stages["E_dtype_convert"] = time.perf_counter() - t_buf

    _post_load_scan(model, device, stages)
    model.eval()

    # Join tokenizer
    t_tok_join = time.perf_counter()
    tok = tok_fut.result()
    stages["J_tokenizer_load"] = time.perf_counter() - t_tok_join
    tok_ex.shutdown(wait=False)
    stages["J_tokenizer_total"] = time.perf_counter() - t_tok_start

    _do_forward(model, tok, device, stages)
    return model, tok


VARIATIONS["V8"] = var_v8_parallel_weights


def run_variation(name, config_name, checkpoint_path, device, dtype_str):
    import torch
    from research.config import get_config
    cfg = get_config(config_name, device=device)
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    stages = {"A_imports_setup": 0.0, "B_config_fetch": 0.0}
    t0 = time.perf_counter()

    fn = VARIATIONS[name]
    model, tok = fn(cfg, checkpoint_path, device, dtype, stages)

    t_end = time.perf_counter()
    stages["TOTAL"] = t_end - t0
    stage_sum = sum(v for k, v in stages.items()
                    if isinstance(v, float) and k != "TOTAL"
                    and k != "J_tokenizer_total" and k != "vram_gb"
                    and k != "n_params_m")
    if t_end - t0 - stage_sum > 0.1:
        print(f"  DEBUG: TOTAL={(t_end-t0)*1000:.1f}ms sum={stage_sum*1000:.1f}ms "
              f"gap={(t_end-t0-stage_sum)*1000:.1f}ms")
    stages["variation"] = name
    stages["config"] = config_name
    stages["device"] = device
    if device == "cuda":
        stages["vram_gb"] = torch.cuda.memory_allocated() / 1e9
    stages["n_params_m"] = sum(p.numel() for p in model.parameters()) / 1e6

    # Quick correctness probe: generate 5 tokens greedily, record them
    try:
        from research.model_loader import create_kv_cache
        cache = create_kv_cache(model, max_total=64, batch=1,
                                device=torch.device(device))
        ids = tok("The capital of France is", return_tensors="pt")
        if hasattr(ids, "to"):
            ids = ids.to(device)
        else:
            ids = {k: v.to(device) for k, v in ids.items()}
        input_ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
        gen_ids = []
        with torch.no_grad():
            for step in range(5):
                if step == 0:
                    out = model(input_ids, preallocated_cache=cache, use_cache=True)
                else:
                    out = model(next_tok, preallocated_cache=cache, use_cache=True)
                logits = out[0]
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache.advance()
                gen_ids.append(next_tok.item())
        stages["correctness_tokens"] = gen_ids
        stages["correctness_text"] = tok.decode(gen_ids)
    except Exception as e:
        stages["correctness_error"] = str(e)

    del model, tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="all",
                    help="Variation name or 'all'")
    ap.add_argument("--config", default="forgelm_v3")
    ap.add_argument("--checkpoint", default="research/checkpoints/ForgeLM_V3_Base.safetensors")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    if args.var == "all":
        names = list(VARIATIONS.keys())
    else:
        names = [args.var]

    print(f"\n{'='*70}")
    print(f"BOOT BENCH VARIATIONS")
    print(f"  config: {args.config}  device: {args.device}")
    print(f"  variations: {names}")
    print(f"{'='*70}\n")

    all_results = []
    for name in names:
        print(f"\n--- {name} ---")
        try:
            stages = run_variation(
                name, args.config, args.checkpoint, args.device, args.dtype)
        except Exception as e:
            import traceback
            traceback.print_exc()
            stages = {"variation": name, "ERROR": str(e)}
        all_results.append(stages)
        if "ERROR" not in stages:
            for k, v in stages.items():
                if k in ("vram_gb",):
                    print(f"  {k:25s} {v:8.3f} GB")
                elif k in ("n_params_m",):
                    print(f"  {k:25s} {v:8.1f} M")
                elif k in ("P_prefetch_vm_ok",):
                    print(f"  {k:25s} {v}")
                elif isinstance(v, float):
                    print(f"  {k:25s} {v*1000:8.1f} ms")
                else:
                    print(f"  {k:25s} {v}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY (TOTAL ms)")
    print(f"{'='*70}")
    for r in all_results:
        name = r.get("variation", "?")
        if "ERROR" in r:
            print(f"  {name:12s}  ERROR: {r['ERROR'][:60]}")
        else:
            total = r.get("TOTAL", 0) * 1000
            vram = r.get("vram_gb", 0)
            print(f"  {name:12s}  {total:8.1f} ms   VRAM {vram:.2f} GB")

    out_path = PROJECT / ".devin" / "boot_bench_variations_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
