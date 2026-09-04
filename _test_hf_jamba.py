"""Compare HF Jamba output vs ForgeEngine output on the same checkpoint."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["FORGE_NO_COMPILE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM

HF_DIR = "research/checkpoints/hf_models/models--ai21labs--AI21-Jamba-Reasoning-3B/snapshots/7524370414253301341370beba35e3d549cbf5f3"
PROMPT = "The quick brown fox jumps over the lazy dog. This is a test of"

print("=" * 60)
print("Loading via HuggingFace transformers")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(HF_DIR)
print(f"Tokenizer: {tokenizer.__class__.__name__}, vocab={tokenizer.vocab_size}")
print(f"BOS={tokenizer.bos_token}({tokenizer.bos_token_id}) EOS={tokenizer.eos_token}({tokenizer.eos_token_id})")

# Encode
ids = tokenizer.encode(PROMPT, return_tensors="pt").to("cuda")
print(f"Input ids: {ids[0][:20].tolist()}")

print("\nLoading model (bf16 on GPU)...")
model = AutoModelForCausalLM.from_pretrained(
    HF_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
    trust_remote_code=True, attn_implementation="eager")
model.eval()

print(f"Model: {model.__class__.__name__}")
print(f"Config: d_model={model.config.hidden_size}, n_layers={model.config.num_hidden_layers}")

with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, temperature=1.0, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
text = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"\nHF Output: {text!r}")

# Also get raw logits for first token to compare
with torch.no_grad():
    logits = model(ids).logits
print(f"Logits shape: {logits.shape}")
print(f"First token logits (top 5): {torch.topk(logits[0, -1], 5)}")
print(f"Argmax token: {logits[0, -1].argmax().item()} -> {tokenizer.decode([logits[0, -1].argmax().item()])!r}")
