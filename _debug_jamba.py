"""Debug the Jamba forward pass step by step."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["FORGE_NO_COMPILE"] = "1"

from research.config import get_config
from research.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer

CHECKPOINT = "research/checkpoints/Jamba_Reasoning_3B.safetensors"
TOKENIZER = "research/checkpoints/forgelm_v2_tokenizer"

cfg = get_config("forgelm_v2", device="cpu")
tokenizer = get_tokenizer(TOKENIZER)

print("Building model on CPU...")
model = ModelLoader.build_model_fast(
    cfg, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16, fast_load=True)
model.eval()

# Encode prompt
ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. This is a test of",
                       return_tensors="pt")
print(f"Input ids: {ids[0][:20].tolist()}")
print(f"Input shape: {ids.shape}")

# Run forward pass step by step
with torch.no_grad():
    # 1. Embedding
    x = model.embed(ids)
    print(f"\n1. Embed output: shape={x.shape} norm={x.norm().item():.4f} mean={x.mean().item():.6f}")
    print(f"   First token embed: norm={x[0,0].norm().item():.4f}")

    # 2. Run through each block, printing stats
    for i, block in enumerate(model.blocks):
        x_before = x.norm().item()
        x = block(x, use_cache=False)
        if isinstance(x, tuple):
            x = x[0]
        x_after = x.norm().item()
        if i < 5 or i in (7, 21) or i > 25:
            print(f"   Block {i:2d} ({block.layer_type:8s}): norm {x_before:.4f} -> {x_after:.4f}")

    # 3. Final norm
    if model.ln_f is not None:
        x = model.ln_f(x)
    print(f"\n3. After ln_f: norm={x.norm().item():.4f}")

    # 4. Head
    logits = model.head(x)
    print(f"4. Logits: shape={logits.shape} norm={logits.norm().item():.4f}")

    # 5. Argmax
    next_token = logits[0, -1].argmax()
    print(f"5. Next token: {next_token.item()} -> {tokenizer.decode([next_token.item()])!r}")

    # 6. Top 5
    top5 = torch.topk(logits[0, -1], 5)
    print(f"   Top 5: {[(t.item(), tokenizer.decode([t.item()]), v.item()) for t, v in zip(top5.indices, top5.values)]}")

# Compare with HF reference (using the same model loaded via transformers)
print("\n" + "="*60)
print("Loading via HF transformers for comparison...")
print("="*60)

from transformers import AutoTokenizer, AutoModelForCausalLM

HF_DIR = "research/checkpoints/hf_models/models--ai21labs--AI21-Jamba-Reasoning-3B/snapshots/7524370414253301341370beba35e3d549cbf5f3"
hf_tok = AutoTokenizer.from_pretrained(HF_DIR)
hf_model = AutoModelForCausalLM.from_pretrained(
    HF_DIR, dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="eager")
hf_model.eval()
# Disable mamba kernels (not available on Windows)
hf_model.config.use_mamba_kernels = False

ids_hf = hf_tok.encode("The quick brown fox jumps over the lazy dog. This is a test of",
                        return_tensors="pt").to("cuda")
print(f"HF Input ids: {ids_hf[0][:20].tolist()}")
print(f"Same ids: {ids_hf[0][:20].tolist() == ids[0][:20].tolist()}")

with torch.no_grad():
    hf_logits = hf_model(ids_hf).logits
    print(f"HF Logits: shape={hf_logits.shape} norm={hf_logits.norm().item():.4f}")
    hf_next = hf_logits[0, -1].argmax()
    print(f"HF Next token: {hf_next.item()} -> {hf_tok.decode([hf_next.item()])!r}")
    top5_hf = torch.topk(hf_logits[0, -1], 5)
    print(f"HF Top 5: {[(t.item(), hf_tok.decode([t.item()]), v.item()) for t, v in zip(top5_hf.indices, top5_hf.values)]}")
