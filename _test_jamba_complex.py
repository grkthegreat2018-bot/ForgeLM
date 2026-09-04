"""Test Jamba-3B with longer, complex prompts to verify coherent output."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["FORGE_NO_COMPILE"] = "1"

from research.config import get_config
from research.model_loader import ModelLoader
from research.tokenizer_cache import get_tokenizer
from research.inference.forge_engine import ForgeEngine

CHECKPOINT = "research/checkpoints/Jamba_Reasoning_3B.safetensors"
TOKENIZER = "research/checkpoints/forgelm_v2_tokenizer"

PROMPTS = [
    "Explain the difference between supervised and unsupervised machine learning. Give a concrete example of each and describe when you would use one over the other.",
    "Write a Python function that takes a list of integers and returns the two numbers that sum to a target value. Include error handling for edge cases.",
    "What are the main causes of climate change? List three primary factors and explain how each contributes to global warming.",
    "Translate the following into French: 'The weather is beautiful today and I would like to go for a walk in the park.'",
    "Solve step by step: If a train travels 60 mph for 2 hours, then 80 mph for 3 hours, what is the total distance traveled?",
]

cfg = get_config("forgelm_v2", device="cpu")
tokenizer = get_tokenizer(TOKENIZER)

print("Building model on CPU...")
model = ModelLoader.build_model_fast(
    cfg, checkpoint_path=CHECKPOINT, dtype=torch.bfloat16, fast_load=True)

print("Quantizing to INT4 on CPU...")
from research.quantization.inference_quant import quantize_model_int4
quantize_model_int4(model, group_size=128)

print("Moving to GPU...")
model = model.to("cuda")
torch.cuda.synchronize()
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

engine = ForgeEngine(model, tokenizer, device="cuda", checkpoint_path=CHECKPOINT)
engine.activate_optimal(quantize=None, kv_cache="cpu_offload", use_compile=False)

for i, prompt in enumerate(PROMPTS):
    print(f"\n{'='*70}")
    print(f"PROMPT {i+1}: {prompt}")
    print(f"{'='*70}")
    output = engine.generate(prompt, max_new_tokens=150, temperature=0.0)
    print(f"\nOUTPUT:\n{output}")
    print(f"\n[Output length: {len(output)} chars]")
