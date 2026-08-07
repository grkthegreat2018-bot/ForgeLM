"""Debug self-play execution."""
import sys, os
sys.path.insert(0, '.')
from research.config import get_config
from research.model_loader import ModelLoader
from research.recursive_self_play import RecursiveSelfPlay
from transformers import AutoTokenizer

cfg = get_config("forgelm_v2", device="cuda")
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
model.to("cuda").eval()
tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

engine = RecursiveSelfPlay(model, tok, max_gen_tokens=120, max_rounds=2)
prompt = 'def greet(name):\n    """Greet someone"""\n    '
result = engine.run_recursive_task(prompt)

print()
print("=== RESULT ===")
print("final_success:", result["final_success"])
print("rounds_used:", result["rounds_used"])
for i, a in enumerate(result["attempts"]):
    print(f"--- Attempt {i} ---")
    print("code:", repr(a["code"][:300]))
    print("error:", a["error"][:500])
    print("stdout:", a.get("stdout", "")[:200])
    print("success:", a["success"])
