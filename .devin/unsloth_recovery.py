"""Recovery fine-tune using Unsloth for 2x faster training.

Converts XP model to HuggingFace Qwen2 format, uses Unsloth for LoRA
fine-tuning on K/V projections (the only lossy transform), then converts
back to our format.

Unsloth provides:
  - 2x faster training via custom Triton kernels
  - 70% less VRAM via 4-bit quantization
  - Automatic gradient checkpointing
"""
import sys, os, json, torch, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from pathlib import Path

SRC = "research/checkpoints/xp_model_keystack.safetensors"
OUT = "research/checkpoints/xp_model_recovered.safetensors"
HF_DIR = "research/checkpoints/xp_hf"
HF_OUT = "research/checkpoints/xp_hf_finetuned"
# Teacher (original Qwen) for self-distillation
TEACHER_SRC = "research/checkpoints/qwen25_coder_1.5b_ported.safetensors"
TEACHER_HF = "research/checkpoints/qwen_hf"

# Map our weight names → Qwen2 HF names
def our_to_hf(name):
    """Convert our tensor names to HuggingFace Qwen2 format."""
    mapping = [
        ("embed.weight", "model.embed_tokens.weight"),
        ("head.weight", "lm_head.weight"),
        ("ln_f.weight", "model.norm.weight"),
    ]
    for old, new in mapping:
        if name == old:
            return new

    # Block-level: blocks.N.X → model.layers.N.X
    if name.startswith("blocks."):
        parts = name.split(".")
        layer = parts[1]
        rest = ".".join(parts[2:])

        block_map = [
            ("attn.q_proj", "self_attn.q_proj"),
            ("attn.k_proj", "self_attn.k_proj"),
            ("attn.v_proj", "self_attn.v_proj"),
            ("attn.out_proj", "self_attn.o_proj"),
            ("ln1", "input_layernorm"),
            ("ln2", "post_attention_layernorm"),
            ("ffn.w_gate", "mlp.gate_proj"),
            ("ffn.w_up", "mlp.up_proj"),
            ("ffn.w_down", "mlp.down_proj"),
        ]
        for old, new in block_map:
            if rest.startswith(old):
                suffix = rest[len(old):]
                return f"model.layers.{layer}.{new}{suffix}"
    return name

def hf_to_our(name):
    """Convert HuggingFace Qwen2 names back to our format."""
    reverse_map = [
        ("model.embed_tokens.weight", "embed.weight"),
        ("lm_head.weight", "head.weight"),
        ("model.norm.weight", "ln_f.weight"),
    ]
    for old, new in reverse_map:
        if name == old:
            return new

    if name.startswith("model.layers."):
        parts = name.split(".")
        layer = parts[2]
        rest = ".".join(parts[3:])

        block_map = [
            ("self_attn.q_proj", "attn.q_proj"),
            ("self_attn.k_proj", "attn.k_proj"),
            ("self_attn.v_proj", "attn.v_proj"),
            ("self_attn.o_proj", "attn.out_proj"),
            ("input_layernorm", "ln1"),
            ("post_attention_layernorm", "ln2"),
            ("mlp.gate_proj", "ffn.w_gate"),
            ("mlp.up_proj", "ffn.w_up"),
            ("mlp.down_proj", "ffn.w_down"),
        ]
        for old, new in block_map:
            if rest.startswith(old):
                suffix = rest[len(old):]
                return f"blocks.{layer}.{new}{suffix}"
    return name

def convert_to_hf():
    """Convert XP model safetensors to HuggingFace Qwen2 format."""
    print("Converting XP model to HuggingFace Qwen2 format...")
    os.makedirs(HF_DIR, exist_ok=True)

    # Write config.json for Qwen2 with MQA (n_kv_heads=1)
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 151936,
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 1,  # MQA (was 2 in original)
        "intermediate_size": 8960,
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "attention_bias": True,
        "bias": True,
    }
    with open(f"{HF_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Convert weights
    hf_state = {}
    with safe_open(SRC, framework="pt") as f:
        for kn in sorted(f.keys()):
            tensor = f.get_tensor(kn)
            hf_name = our_to_hf(kn)
            if hf_name != kn:
                hf_state[hf_name] = tensor.contiguous().to(torch.bfloat16)
            else:
                # Skip non-model tensors (mtp, rotorquant, value_residual)
                print(f"  Skipping non-model tensor: {kn}")

    save_file(hf_state, f"{HF_DIR}/model.safetensors")
    print(f"  Saved {len(hf_state)} tensors to {HF_DIR}/model.safetensors")

    # Copy tokenizer from Qwen cache
    tokenizer_src = ".devin/hf_cache/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots"
    if os.path.exists(tokenizer_src):
        snap = os.listdir(tokenizer_src)[0]
        tok_dir = os.path.join(tokenizer_src, snap)
        for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
            src_f = os.path.join(tok_dir, f)
            if os.path.exists(src_f):
                shutil.copy(src_f, f"{HF_DIR}/{f}")
                print(f"  Copied {f}")
    print("  Conversion complete.")

def convert_teacher_to_hf():
    """Convert original Qwen (teacher) to HF format for self-distillation."""
    if os.path.exists(f"{TEACHER_HF}/model.safetensors"):
        print(f"  Teacher HF model already exists at {TEACHER_HF}")
        return

    print("Converting original Qwen (teacher) to HuggingFace Qwen2 format...")
    os.makedirs(TEACHER_HF, exist_ok=True)

    # Teacher uses original GQA config (n_kv_heads=2)
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "vocab_size": 151936,
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,  # Original GQA (not MQA)
        "intermediate_size": 8960,
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "attention_bias": True,
        "bias": True,
    }
    with open(f"{TEACHER_HF}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    hf_state = {}
    with safe_open(TEACHER_SRC, framework="pt") as f:
        for kn in sorted(f.keys()):
            tensor = f.get_tensor(kn)
            hf_name = our_to_hf(kn)
            if hf_name != kn:
                hf_state[hf_name] = tensor.contiguous().to(torch.bfloat16)

    save_file(hf_state, f"{TEACHER_HF}/model.safetensors")

    # Copy tokenizer
    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src_f = f"{HF_DIR}/{fn}"
        if os.path.exists(src_f):
            shutil.copy(src_f, f"{TEACHER_HF}/{fn}")
    print(f"  Teacher converted: {len(hf_state)} tensors")

def finetune_with_unsloth():
    """Use Unsloth for 8-bit LoRA recovery with self-distillation.

    Optimizations from research:
    - 8-bit LoRA (1.8-2x faster than 16-bit, identical quality)
    - r=32 + rsLoRA scaling (alpha=256, use_rslora=True)
    - target ALL attention modules (q/k/v/o_proj)
    - batch_size=4, grad_accum=4 (effective batch=16)
    - Self-distillation from original Qwen (4-10x faster convergence)
    - lr=5e-4, cosine decay, warmup_ratio=0.05
    - WiSE-FT post-processing (free 2-5% improvement)
    """
    print("\n" + "=" * 70)
    print("UNSLOTH 8-BIT LoRA + SELF-DISTILLATION RECOVERY")
    print("=" * 70)

    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer
    from trl import SFTTrainer, SFTConfig
    import torch.nn.functional as F
    import numpy as np

    max_seq_length = 512

    # --- Load student (XP model) in 4-bit QLoRA ---
    # 8-bit hits bnb+inductor+checkpointing bug on Windows; 4-bit is most tested
    print("  Loading student (XP model) in 4-bit QLoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=HF_DIR,
        max_seq_length=max_seq_length,
        load_in_4bit=True,    # 4-bit QLoRA — most reliable with Unsloth
        load_in_8bit=False,
        load_in_16bit=False,
    )

    # --- LoRA: r=16, standard alpha=32 (CoreML-LLM MQA recovery recipe) ---
    # Research found: alpha=256 + lr=5e-4 is 10x too aggressive.
    # CoreML-LLM uses r=16, alpha=32, lr=2e-4, KD_ALPHA=0.5 for same GQA→MQA task.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                                           # CoreML-LLM recipe
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # All attention
        lora_alpha=32,                                  # Standard 2*r (not 256!)
        lora_dropout=0,                                 # 0 for fast patching
        bias="none",
        use_rslora=False,                               # No rsLoRA at r=16
        use_gradient_checkpointing="unsloth",           # 4x longer context, 30% less VRAM
        random_state=42,
    )
    print(f"  LoRA: r=16, alpha=32, target=q/k/v/o_proj (CoreML-LLM recipe)")

    # --- Load teacher (original Qwen) in 4-bit, frozen ---
    print("  Loading teacher (original Qwen) in 4-bit...")
    convert_teacher_to_hf()
    teacher, _ = FastLanguageModel.from_pretrained(
        model_name=TEACHER_HF,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  Teacher loaded and frozen")

    # --- Custom trainer with self-distillation loss ---
    class DistillationSFTTrainer(SFTTrainer):
        """SFT trainer with KL distillation from teacher logits.

        Loss = alpha * KL(student || teacher) + (1-alpha) * CE(student, targets)
        Temperature scaling on KL for softer distributions.
        """
        def __init__(self, teacher=None, distill_alpha=0.5, temperature=2.0, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher = teacher
            self.distill_alpha = distill_alpha
            self.temperature = temperature

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")

            # Student forward
            outputs = model(**inputs)
            student_logits = outputs.logits

            # CE loss (standard SFT)
            # Shift for causal LM
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if self.teacher is not None:
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_out = self.teacher(**inputs)
                    teacher_logits = teacher_out.logits[..., :-1, :].contiguous()

                # Top-K KL divergence (much more stable than full 152K vocab)
                T = self.temperature
                k = 100  # Top-100 tokens for KL
                s_logits = shift_logits.float() / T
                t_logits = teacher_logits.float() / T

                # Get top-K teacher tokens
                topk_t = t_logits.topk(k, dim=-1)
                # Gather student logits at same positions
                s_topk = s_logits.gather(-1, topk_t.indices)

                # KL on top-K subset
                s_log_probs = F.log_softmax(s_topk, dim=-1)
                t_log_probs = F.log_softmax(topk_t.values, dim=-1)
                t_probs = t_log_probs.exp()

                kl_loss = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1).mean()
                kl_loss *= (T * T)  # Temperature scaling

                total_loss = self.distill_alpha * kl_loss + (1 - self.distill_alpha) * ce_loss
            else:
                total_loss = ce_loss

            return (total_loss, outputs) if return_outputs else total_loss

    # --- Training data ---
    train_np = np.memmap("research/data/train.bin", dtype=np.uint16, mode="r")
    print(f"  Train tokens: {len(train_np)/1e6:.1f}M")

    from datasets import Dataset
    def make_dataset(n_samples=2000, seq_len=256):
        texts = []
        for _ in range(n_samples):
            idx = np.random.randint(0, len(train_np) - seq_len)
            tokens = train_np[idx:idx + seq_len]
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=False)
            texts.append({"text": text})
        return Dataset.from_list(texts)

    dataset = make_dataset(5000)              # CoreML-LLM uses 20k; 5k for speed
    print(f"  Dataset: {len(dataset)} samples")

    # --- Training config (CoreML-LLM MQA recovery recipe) ---
    training_args = SFTConfig(
        output_dir=HF_OUT,
        num_train_epochs=1,                   # 1 epoch (CoreML-LLM recipe)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,        # Effective batch = 16
        learning_rate=2e-4,                   # CoreML-LLM recipe (not 5e-4)
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=max_seq_length,
        max_length=max_seq_length,
        packing=True,                        # 3-5x faster with Unsloth
        logging_steps=20,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",                  # 75% optimizer VRAM savings
        seed=42,
        report_to="none",
    )

    trainer = DistillationSFTTrainer(
        teacher=teacher,
        distill_alpha=0.5,                   # 50% distillation, 50% CE (CoreML-LLM)
        temperature=2.0,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    print("  Starting training (4-bit QLoRA + self-distillation)...")
    trainer.train()
    print("  Training complete.")

    # Save LoRA adapter only (avoid bnb 4-bit merge corruption)
    print("  Saving LoRA adapter (no merge — avoids 4-bit dequant corruption)...")
    model.save_pretrained(f"{HF_OUT}_adapter")
    print(f"  Saved LoRA adapter to {HF_OUT}_adapter")

    # --- WiSE-FT post-processing ---
    print("  WiSE-FT will be applied during convert_back")

def convert_back():
    """Apply LoRA adapter to original XP weights in bf16 (no 4-bit dequant).

    Loads the LoRA A/B matrices from the adapter checkpoint, then applies
    them to the original XP model weights in full precision:
      W_new = W_orig + (alpha / sqrt(r)) * B @ A

    This avoids the bnb 4-bit merge corruption that produces negative cosine.
    Also applies WiSE-FT interpolation: W_final = (1-w) * W_orig + w * W_recovered
    """
    print("\n" + "=" * 70)
    print("Applying LoRA adapter to XP weights (bf16, no 4-bit dequant)...")
    print("=" * 70)

    from safetensors.torch import load_file as lf
    import json

    # Load LoRA adapter
    adapter_path = f"{HF_OUT}_adapter/adapter_model.safetensors"
    adapter_config_path = f"{HF_OUT}_adapter/adapter_config.json"

    with open(adapter_config_path) as f:
        adapter_cfg = json.load(f)
    r = adapter_cfg["r"]
    alpha = adapter_cfg["lora_alpha"]
    use_rslora = adapter_cfg.get("use_rslora", False)
    scale = alpha / (r ** 0.5) if use_rslora else alpha / r
    print(f"  LoRA: r={r}, alpha={alpha}, rslora={use_rslora}, scale={scale:.4f}")

    adapter_state = lf(adapter_path)
    print(f"  Adapter tensors: {len(adapter_state)}")

    # Group A and B matrices by target module
    # PEFT naming: base_model.model.model.layers.N.self_attn.X_proj.lora_A.weight
    #              base_model.model.model.layers.N.self_attn.X_proj.lora_B.weight
    lora_pairs = {}  # (layer, module) -> {"A": tensor, "B": tensor}
    for kn, tensor in adapter_state.items():
        # Strip base_model.model. prefix from PEFT naming
        kn = kn.replace("base_model.model.", "")
        if "lora_A" in kn:
            key = kn.replace(".lora_A.weight", "")
            lora_pairs.setdefault(key, {})["A"] = tensor.float()
        elif "lora_B" in kn:
            key = kn.replace(".lora_B.weight", "")
            lora_pairs.setdefault(key, {})["B"] = tensor.float()

    print(f"  LoRA pairs: {len(lora_pairs)}")

    # Map HF module names to our tensor names
    def hf_module_to_our(hf_mod_name):
        """model.layers.N.self_attn.X_proj -> blocks.N.attn.Y_proj"""
        if "model.layers." in hf_mod_name:
            parts = hf_mod_name.split(".")
            layer = parts[2]
            rest = ".".join(parts[3:])
            mod_map = [
                ("self_attn.q_proj", "attn.q_proj"),
                ("self_attn.k_proj", "attn.k_proj"),
                ("self_attn.v_proj", "attn.v_proj"),
                ("self_attn.o_proj", "attn.out_proj"),
            ]
            for old, new in mod_map:
                if rest == old:
                    return f"blocks.{layer}.{new}.weight"
        return None

    # Load original XP weights
    our_state = {}
    with safe_open(SRC, framework="pt") as f:
        for kn in f.keys():
            our_state[kn] = f.get_tensor(kn).to(torch.float32)

    # Apply LoRA: W_new = W_orig + scale * B @ A
    applied = 0
    for hf_mod, pair in lora_pairs.items():
        our_name = hf_module_to_our(hf_mod)
        if our_name is None or our_name not in our_state:
            print(f"  SKIP: {hf_mod} -> {our_name}")
            continue

        A = pair["A"]  # (r, in_features)
        B = pair["B"]  # (out_features, r)
        delta = (B @ A) * scale  # (out_features, in_features)

        W = our_state[our_name]
        if W.shape != delta.shape:
            print(f"  SHAPE MISMATCH: {our_name} W={W.shape} delta={delta.shape}")
            continue

        our_state[our_name] = W + delta
        applied += 1

    print(f"  Applied LoRA to {applied} weight matrices")

    # WiSE-FT: interpolate between original and recovered
    # Load original XP weights again for interpolation
    wise_alpha = 0.5  # 50/50 blend — conservative
    print(f"  WiSE-FT: alpha={wise_alpha} (blend recovered with original)")
    with safe_open(SRC, framework="pt") as f:
        for kn in f.keys():
            if kn in our_state:
                orig = f.get_tensor(kn).to(torch.float32)
                if orig.shape == our_state[kn].shape:
                    our_state[kn] = (1 - wise_alpha) * orig + wise_alpha * our_state[kn]

    # Convert to bf16 and save
    for kn in our_state:
        our_state[kn] = our_state[kn].to(torch.bfloat16)

    save_file(our_state, OUT)
    print(f"  Saved {len(our_state)} tensors to {OUT}")

def main():
    # Skip XP conversion if already done
    if not os.path.exists(f"{HF_DIR}/model.safetensors"):
        convert_to_hf()
    else:
        print(f"XP HF model already exists at {HF_DIR}, skipping conversion")
    finetune_with_unsloth()
    convert_back()
    print("\nDone! Recovered model at:", OUT)

if __name__ == "__main__":
    main()
