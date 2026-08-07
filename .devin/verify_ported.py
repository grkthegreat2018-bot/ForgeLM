"""Compare our ported model's output against HuggingFace's Qwen2.5-Coder-1.5B-Instruct.

This is the definitive test: if our architecture is truly identical, both models
should produce identical logits on the same input tokens.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.config import get_config
from research.model_loader import ModelLoader

# Load HuggingFace's Qwen2.5-Coder-1.5B-Instruct
print("Loading HF Qwen2.5-Coder-1.5B-Instruct...")
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
cache_dir = ".devin/hf_cache"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, cache_dir=cache_dir, torch_dtype=torch.bfloat16, device_map="cuda"
)
hf_model.eval()

# Load our ported model
print("Loading our ported model...")
cfg = get_config("qwen25_coder_1.5b")
our_model = ModelLoader.build_model(cfg)
from safetensors.torch import load_file
state = load_file("research/checkpoints/qwen25_coder_1.5b_ported.safetensors")
our_model.load_state_dict(state, strict=True)
our_model.eval()
our_model = our_model.to("cuda", dtype=torch.bfloat16)

# Test with a few different inputs
test_texts = [
    "def hello_world():",
    "The quick brown fox",
    "import numpy as",
]

print("\n=== Logit Comparison (max abs diff + cosine similarity) ===")
print(f"{'Input':<30} {'Max abs diff':<15} {'Mean abs diff':<15} {'Cosine sim':<15}")
print("-" * 75)

with torch.inference_mode():
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]

        # HF model
        hf_out = hf_model(input_ids).logits

        # Our model
        our_out = our_model(input_ids)
        our_logits = our_out[0] if isinstance(our_out, tuple) else our_out

        # Compare
        max_diff = (hf_out - our_logits).abs().max().item()
        mean_diff = (hf_out - our_logits).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            hf_out.flatten().unsqueeze(0),
            our_logits.flatten().unsqueeze(0)
        ).item()

        print(f"{text:<30} {max_diff:<15.6f} {mean_diff:<15.6f} {cos_sim:<15.8f}")

# Also compare top-5 predictions for the last token of first input
print("\n=== Top-5 next-token predictions for 'def hello_world():' ===")
with torch.inference_mode():
    inputs = tokenizer("def hello_world():", return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    hf_logits = hf_model(input_ids).logits[0, -1, :]
    our_out = our_model(input_ids)
    our_logits = (our_out[0] if isinstance(our_out, tuple) else our_out)[0, -1, :]

    hf_top5 = torch.topk(hf_logits, 5)
    our_top5 = torch.topk(our_logits, 5)

    print(f"{'Rank':<6} {'HF token':<20} {'HF prob':<12} {'Our token':<20} {'Our prob':<12} {'Match'}")
    print("-" * 80)
    for i in range(5):
        hf_tok = tokenizer.decode(hf_top5.indices[i].item())
        our_tok = tokenizer.decode(our_top5.indices[i].item())
        hf_prob = torch.softmax(hf_logits, dim=-1)[hf_top5.indices[i]].item()
        our_prob = torch.softmax(our_logits, dim=-1)[our_top5.indices[i]].item()
        match = "OK" if hf_top5.indices[i].item() == our_top5.indices[i].item() else "DIFF"
        print(f"{i+1:<6} {repr(hf_tok):<20} {hf_prob:<12.6f} {repr(our_tok):<20} {our_prob:<12.6f} {match}")
