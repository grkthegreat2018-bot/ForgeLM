"""Instrumented V8: measure actual background thread time + find unaccounted ms."""
import time, os, sys, threading, gc
sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import torch
from concurrent.futures import ThreadPoolExecutor
from research.config import get_config
from research.model_loader import ConfigurableResearchLLM, ModelLoader
from research.tokenizer_cache import get_tokenizer

device = "cuda"
dtype = torch.bfloat16
cfg = get_config("forgelm_v3", device=device)
checkpoint_path = "research/checkpoints/ForgeLM_V3_Base.safetensors"

t0 = time.perf_counter()

# Start tokenizer
t_tok_start = time.perf_counter()
ex = ThreadPoolExecutor(max_workers=1)
tok_fut = ex.submit(get_tokenizer, "research/checkpoints/lfm25_tokenizer")

# Start state_dict load in background
state_result = {}
bg_timing = {}
def _load_state():
    t_bg_start = time.perf_counter()
    bg_timing["start"] = t_bg_start - t0
    state = ModelLoader._load_safetensors_mmap(
        checkpoint_path, None, device=torch.device(device))
    bg_timing["end"] = time.perf_counter() - t0
    bg_timing["duration"] = bg_timing["end"] - bg_timing["start"]
    state_result["state"] = state

state_thread = threading.Thread(target=_load_state, daemon=True)
state_thread.start()

# Meta init
t_arch = time.perf_counter()
cfg_meta = type(cfg)(**{**cfg.__dict__, "device": "meta"})
with torch.device("meta"):
    model = ConfigurableResearchLLM(cfg_meta)
t_arch = time.perf_counter() - t_arch
print(f"D_arch_build:        {t_arch*1000:7.1f} ms")

# Wait for state_dict
t_wait = time.perf_counter()
state_thread.join()
t_wait = time.perf_counter() - t_wait
print(f"F_weights_wait:      {t_wait*1000:7.1f} ms")
print(f"BG thread start:     {bg_timing.get('start', 0)*1000:7.1f} ms (after t0)")
print(f"BG thread duration:  {bg_timing.get('duration', 0)*1000:7.1f} ms")
print(f"BG thread end:       {bg_timing.get('end', 0)*1000:7.1f} ms (after t0)")

state = state_result.get("state", {})
print(f"State keys:          {len(state)}")

# GQA->diff
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
t_conv = time.perf_counter() - t_conv
print(f"G_diff_convert:      {t_conv*1000:7.1f} ms")

# assign=True
t_lsd = time.perf_counter()
model.load_state_dict(state, strict=False, assign=True)
if device == "cuda":
    torch.cuda.synchronize()
t_lsd = time.perf_counter() - t_lsd
print(f"H_load_state_dict:   {t_lsd*1000:7.1f} ms")

# Re-tie + buffer reset
t_buf = time.perf_counter()
model.head.weight = model.embed.weight
ModelLoader._reset_non_persistent_buffers(model, torch.device(device))
if device == "cuda":
    torch.cuda.synchronize()
t_buf = time.perf_counter() - t_buf
print(f"E_dtype_convert:     {t_buf*1000:7.1f} ms")

# Post-load scan
t_scan = time.perf_counter()
for block in model.blocks:
    attn = block.attn
    if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
        q_id = (attn.q_norm.weight == 1.0).all()
        k_id = (attn.k_norm.weight == 1.0).all()
        attn._qk_norm_identity = bool(q_id and k_id)
    if hasattr(attn, 'lambda_param') and hasattr(attn, 'set_identity'):
        attn.set_identity((attn.lambda_param == 0.0).all().item())
t_scan = time.perf_counter() - t_scan
print(f"I_post_load_scan:    {t_scan*1000:7.1f} ms")

model.eval()

# Join tokenizer
t_tok_join = time.perf_counter()
tok = tok_fut.result()
t_tok_join = time.perf_counter() - t_tok_join
t_tok_total = time.perf_counter() - t_tok_start
ex.shutdown(wait=False)
print(f"J_tokenizer_load:    {t_tok_join*1000:7.1f} ms")
print(f"J_tokenizer_total:   {t_tok_total*1000:7.1f} ms")

# Forward
from research.model_loader import create_kv_cache
t_kv = time.perf_counter()
cache = create_kv_cache(model, max_total=2048, batch=1, device=torch.device(device))
if device == "cuda":
    torch.cuda.synchronize()
t_kv = time.perf_counter() - t_kv
print(f"K_kv_cache_alloc:    {t_kv*1000:7.1f} ms")

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
t_ff = time.perf_counter() - t_ff
print(f"L_first_forward:     {t_ff*1000:7.1f} ms")

t_dec = time.perf_counter()
with torch.no_grad():
    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache.advance()
    out2 = model(next_tok, preallocated_cache=cache, use_cache=True)
if device == "cuda":
    torch.cuda.synchronize()
t_dec = time.perf_counter() - t_dec
print(f"M_first_decode:      {t_dec*1000:7.1f} ms")

total = time.perf_counter() - t0
print(f"TOTAL:               {total*1000:7.1f} ms")

# Sum of stages
stage_sum = t_arch + t_wait + t_conv + t_lsd + t_buf + t_scan + t_tok_join + t_kv + t_ff + t_dec
print(f"Sum of stages:       {stage_sum*1000:7.1f} ms")
print(f"Unaccounted:         {(total - stage_sum)*1000:7.1f} ms")

# Correctness
gen_ids = []
with torch.no_grad():
    for step in range(5):
        if step == 0:
            o = model(input_ids, preallocated_cache=cache, use_cache=True)
        else:
            o = model(next_tok, preallocated_cache=cache, use_cache=True)
        next_tok = o[0][:, -1, :].argmax(dim=-1, keepdim=True)
        cache.advance()
        gen_ids.append(next_tok.item())
print(f"Correctness:         {tok.decode(gen_ids)} (tokens: {gen_ids})")
