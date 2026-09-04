"""Debug Mamba state management during generation."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["FORGE_NO_COMPILE"] = "1"

from forge.config import get_config
from forge.model_loader import ModelLoader, unpack_output_with_kv
from research.tokenizer_cache import get_tokenizer

CHECKPOINT = "research/checkpoints/Jamba_Reasoning_3B.safetensors"
TOKENIZER = "research/checkpoints/forgelm_v2_tokenizer"

cfg = get_config("forgelm_v2", device="cpu")
tokenizer = get_tokenizer(TOKENIZER)

model = ModelLoader.build_model_fast(
    cfg, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16, fast_load=True)
model = model.to("cuda")
model.eval()

ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. This is a test of",
                       return_tensors="pt").to("cuda")
print(f"Input: {ids.shape}")

# Prefill
with torch.inference_mode():
    out = model(ids, use_cache=True)
    logits, past_kv = unpack_output_with_kv(out)

print(f"Logits: {logits.shape}")
print(f"past_kv type: {type(past_kv)}, len: {len(past_kv) if past_kv else 0}")

# Check what each layer's KV state looks like
if past_kv:
    for i, kv in enumerate(past_kv):
        if i < 3 or i in (7, 21) or i > 25:
            if isinstance(kv, dict):
                print(f"  Layer {i:2d} (mamba): dict keys={list(kv.keys())}")
                if 'ssm_state' in kv:
                    print(f"    ssm_state: {kv['ssm_state'].shape}")
                if 'conv_state' in kv:
                    print(f"    conv_state: {kv['conv_state'].shape}")
            elif isinstance(kv, tuple):
                print(f"  Layer {i:2d} (attn): tuple len={len(kv)}, k={kv[0].shape}, v={kv[1].shape}")
            else:
                print(f"  Layer {i:2d}: type={type(kv)}")

# Generate one token and check
next_token = logits[:, -1:].argmax(-1)
print(f"\nNext token: {next_token.item()} -> {tokenizer.decode([next_token.item()])!r}")

with torch.inference_mode():
    out2 = model(next_token, past_key_values=past_kv, use_cache=True)
    logits2, past_kv2 = unpack_output_with_kv(out2)

print(f"Logits2: {logits2.shape}")
next_token2 = logits2[:, -1:].argmax(-1)
print(f"Next token 2: {next_token2.item()} -> {tokenizer.decode([next_token2.item()])!r}")

# Generate a few more tokens manually
print("\nManual generation:")
tokens = [next_token.item(), next_token2.item()]
cur_kv = past_kv2
cur_token = next_token2  # (1, 1)
for step in range(18):
    with torch.inference_mode():
        out_n = model(cur_token, past_key_values=cur_kv, use_cache=True)
        cur_logits, cur_kv = unpack_output_with_kv(out_n)
        cur_token = cur_logits[:, -1, :].argmax(-1, keepdim=True)  # (B, 1)
        tok = cur_token.item()
        tokens.append(tok)
        print(f"  Step {step+2}: token={tok} -> {tokenizer.decode([tok])!r}")

print(f"\nFull output: {tokenizer.decode(tokens)!r}")
