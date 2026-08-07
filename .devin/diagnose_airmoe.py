"""Diagnose self-play training: AirMoE loading, expert application, prompts, model output."""
import sys, os, torch, time
sys.path.insert(0, '.')

from research.config import get_config
from research.model_loader import ModelLoader
from research.airmoe_infinite import InfiniteAirMoE
from transformers import AutoTokenizer
from safetensors.torch import load_file

OUT = "D:/windsurf/ForgeAI/.devin/tmp/diagnose_airmoe"

print("=" * 70)
print("AirMoE Self-Play Diagnosis")
print("=" * 70)

# ── Q1: Is AirMoE loading correctly from disk? ─────────────────────
print("\n" + "=" * 70)
print("Q1: Is AirMoE loading correctly from disk?")
print("=" * 70)

v4_dir = "D:/windsurf/ForgeAI/research/checkpoints/forgelm_v4"
expert_path = os.path.join(v4_dir, "experts", "expert_l0_python_general.safetensors")
print(f"\n  File: {expert_path}")
print(f"  Exists: {os.path.exists(expert_path)}")

if os.path.exists(expert_path):
    state = load_file(expert_path)
    print(f"  Tensors: {len(state)}")
    for k in sorted(state.keys()):
        v = state[k]
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}, "
              f"min={v.float().min():.3f}, max={v.float().max():.3f}")

    # Check if U_shape and Vh_shape are present
    has_shapes = any("U_shape" in k for k in state)
    print(f"\n  Has shape metadata: {has_shapes}")

# ── Q2: Are experts being applied to the model correctly? ──────────
print("\n" + "=" * 70)
print("Q2: Are experts being applied to the model correctly?")
print("=" * 70)

print("\n  Loading v2 base model...")
cfg = get_config("forgelm_v2", device="cuda")
model = ModelLoader.build_model_fast(cfg, checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
model.to("cuda").eval()
tok = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

# Check original expert 0 weights at layer 0
block0 = model.blocks[0]
ffn0 = block0.ffn
print(f"\n  Layer 0 FFN type: {type(ffn0).__name__}")
print(f"  Layer 0 has experts: {hasattr(ffn0, 'experts')}")
if hasattr(ffn0, 'experts'):
    print(f"  Layer 0 n_experts: {len(ffn0.experts)}")
    exp0 = ffn0.experts[0]
    w1_orig = exp0.w1.weight.data.clone()
    print(f"  Expert 0 w1: shape={w1_orig.shape}, dtype={w1_orig.dtype}")
    print(f"    mean={w1_orig.float().mean():.4f}, std={w1_orig.float().std():.4f}")
    print(f"    first 5 values: {w1_orig[0, :5].tolist()}")

# Now load AirMoE expert and check if weights change
print("\n  Loading AirMoE python_general expert...")
airmoe = InfiniteAirMoE(model, tok, v4_dir, device="cuda", vram_budget_gb=2.0, max_cached_topics=5)
airmoe.load_topic("python_general")

# Check if expert 0 weights changed
w1_after = model.blocks[0].ffn.experts[0].w1.weight.data.clone()
print(f"\n  After AirMoE load:")
print(f"    Expert 0 w1: shape={w1_after.shape}, dtype={w1_after.dtype}")
print(f"    mean={w1_after.float().mean():.4f}, std={w1_after.float().std():.4f}")
print(f"    first 5 values: {w1_after[0, :5].tolist()}")

# Check if weights actually changed
changed = not torch.equal(w1_orig, w1_after)
diff = (w1_orig.float() - w1_after.float()).abs().max().item()
print(f"\n  Weights changed: {changed}")
print(f"  Max diff: {diff:.6f}")
if not changed:
    print("  *** WARNING: AirMoE load did NOT change expert weights! ***")
    print("  *** The expert is being loaded but not injected into the model ***")

# Check all 28 layers
print(f"\n  Checking all 28 layers for weight changes...")
n_changed = 0
for i in range(28):
    exp = model.blocks[i].ffn.experts[0]
    # We can't compare to original anymore (already overwritten), 
    # but we can check if the weight looks reasonable
    w = exp.w1.weight.data
    has_nan = torch.isnan(w).any().item()
    has_inf = torch.isinf(w).any().item()
    if has_nan or has_inf:
        print(f"    Layer {i}: NaN={has_nan}, Inf={has_inf} *** BAD ***")
    else:
        n_changed += 1
print(f"  {n_changed}/28 layers have valid weights")

# ── Q3: Are prompts properly formed? ───────────────────────────────
print("\n" + "=" * 70)
print("Q3: Are prompts properly formed?")
print("=" * 70)

test_prompts = [
    'def greet(name):\n    """Greet someone"""\n    ',
    'def is_prime(n):\n    """Check if prime"""\n    ',
    'def reverse(s):\n    """Reverse string"""\n    ',
]

for p in test_prompts:
    print(f"\n  Prompt: {repr(p)}")
    ids = tok(p, return_tensors="pt").input_ids
    print(f"  Token IDs: {ids[0].tolist()[:20]}...")
    print(f"  Decoded: {repr(tok.decode(ids[0], skip_special_tokens=True))}")
    # Check if it's valid Python
    try:
        compile(p, "<test>", "exec")
        print(f"  Valid Python: YES (compiles)")
    except SyntaxError as e:
        print(f"  Valid Python: PARTIAL (expected — needs completion): {e.msg}")

# ── Q4: What is the model producing? ───────────────────────────────
print("\n" + "=" * 70)
print("Q4: What is the model producing?")
print("=" * 70)

def generate_raw(prompt, max_tokens=80):
    """Generate and return both the full text and just the completion."""
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    prompt_len = ids.shape[1]
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits, _ = model(ids)
            nxt = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() == tok.eos_token_id:
                break
    full = tok.decode(ids[0], skip_special_tokens=True)
    completion = tok.decode(ids[0, prompt_len:], skip_special_tokens=True)
    return full, completion

for prompt in test_prompts:
    print(f"\n  Prompt: {repr(prompt[:50])}")
    full, completion = generate_raw(prompt, max_tokens=60)
    print(f"  Completion: {repr(completion[:120])}")
    # Check if prompt + completion is valid Python
    try:
        compile(full, "<test>", "exec")
        print(f"  Full code valid: YES")
    except SyntaxError as e:
        print(f"  Full code valid: NO — {e.msg} (line {e.lineno})")

# ── Q5: Is the model thinking correctly? (quality check) ───────────
print("\n" + "=" * 70)
print("Q5: Is the model thinking correctly? (quality check)")
print("=" * 70)

quality_prompts = [
    ("Code", 'def factorial(n):\n    """Return factorial"""\n    '),
    ("Code", 'def is_palindrome(s):\n    """Check palindrome"""\n    '),
    ("Knowledge", "The capital of Japan is "),
    ("Knowledge", "The largest planet in our solar system is "),
    ("Math", "What is 15 + 27? "),
    ("Reasoning", "Why does ice float on water? "),
]

for category, prompt in quality_prompts:
    _, completion = generate_raw(prompt, max_tokens=50)
    print(f"\n  [{category}] {prompt.strip()[:40]}")
    print(f"  => {completion.strip()[:80]}")

# ── Q6: Does the sandbox execute code correctly? ───────────────────
print("\n" + "=" * 70)
print("Q6: Does the sandbox execute code correctly?")
print("=" * 70)

from research.self_play_sandbox import SelfPlaySandbox
sandbox = SelfPlaySandbox(model, tok, device="cuda", max_gen_tokens=80)

# Test with a known-good code snippet
test_code = 'def greet(name):\n    """Greet someone"""\n    return f"Hello, {name}!"\n\nprint(greet("World"))'
print(f"\n  Test code: {repr(test_code[:80])}")
code, telem = sandbox.generate_code('def greet(name):\n    """Greet someone"""\n    ')
print(f"  Generated: {repr(code[:120])}")
prompt_str = 'def greet(name):\n    """Greet someone""\"\n    '
full_code = prompt_str + code
print(f"  Full (prompt+gen): {repr(full_code[:120])}")

# Execute the test code directly
from research.recursive_self_play import DDriveSandboxExecutor
executor = DDriveSandboxExecutor(temp_dir="D:/windsurf/ForgeAI/.devin/tmp")
result = executor.execute(test_code, expected_output=None)
print(f"\n  Direct execution of known-good code:")
print(f"    returncode: {result['returncode']}")
print(f"    stdout: {repr(result['stdout'][:100])}")
print(f"    stderr: {repr(result['stderr'][:200])}")

print("\n" + "=" * 70)
print("Diagnosis Complete")
print("=" * 70)
