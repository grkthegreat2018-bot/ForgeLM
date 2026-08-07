"""Quick test: does the model generate coherent code?"""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.config import get_config
from research.model_loader import ModelLoader
from transformers import AutoTokenizer

cfg = get_config("forgelm_v2", device="cuda")
model = ModelLoader.build_model_fast(
    cfg, checkpoint_path="research/checkpoints/forgelm_v2.safetensors", moe_top_k=2)
model.to("cuda").eval()
tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

prompt = (
    "# Goal: Compute the sum of all elements in a list\n"
    "# Define a function sum_list(lst: list) -> int.\n"
    "def sum_list(lst):\n"
    '    """Compute the sum of all elements in a list"""\n'
    "    "
)

ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
print(f"Prompt tokens: {ids.shape[1]}")
print(f"Prompt: {repr(prompt[:100])}...")

with torch.inference_mode():
    for step in range(80):
        logits, _ = model(ids)
        next_token = logits[0, -1].argmax()
        ids = torch.cat([ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        if next_token.item() == tok.eos_token_id:
            break

gen = tok.decode(ids[0], skip_special_tokens=True)
print("\n=== FULL OUTPUT ===")
print(gen)
print("=== GENERATED PART ===")
print(gen[len(prompt):])
