"""Entropy-ABF — Attention Bias Free context extension with 100 samples.

Simpler alternative to LongRoPE2: extends context 4x with minimal fine-tuning
(~100 samples). Uses entropy stabilization to identify which RoPE dimensions
need scaling, then applies attention bias to stabilize.

Three steps:
1. Measure attention entropy at base context → find "high-entropy" dimensions
2. Apply ABF (Attention Bias Free) scaling to those dimensions only
3. Fine-tune on 100 long-context samples (vs LongRoPE2's evolutionary search)

Usage:
    from research.entropy_abf import EntropyABF, apply_abf_scaling

    # Measure entropy
    abf = EntropyABF(model, tokenizer)
    abf.measure_entropy(calib_prompts)
    abf.compute_scaling(target_factor=4.0)

    # Apply scaling to model's RoPE
    apply_abf_scaling(model, abf.scaling_factors)

    # Fine-tune on 100 samples
    abf.finetune(long_context_samples, steps=100)

Reference: "Extending LLMs' Context Window with 100 Samples" (GAIR-NLP)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class EntropyABF:
    """Entropy-ABF context extension.

    Args:
        model: the LLM with RoPE positional encoding
        tokenizer: tokenizer
        device: cuda or cpu
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.entropies: List[float] = []
        self.scaling_factors: Optional[torch.Tensor] = None
        self.head_dim = None

        # Find RoPE configuration in the model.
        self._find_rope_config()

    def _find_rope_config(self):
        """Find RoPE-related parameters in the model."""
        # Look for rotary embedding modules or configs.
        for name, module in self.model.named_modules():
            if hasattr(module, "head_dim"):
                self.head_dim = module.head_dim
                return
            if hasattr(module, "dim") and "rotary" in name.lower():
                self.head_dim = module.dim
                return

        # Fallback: infer from config.
        if hasattr(self.model, "config"):
            cfg = self.model.config
            if hasattr(cfg, "head_dim"):
                self.head_dim = cfg.head_dim
            elif hasattr(cfg, "d_model") and hasattr(cfg, "n_heads"):
                self.head_dim = cfg.d_model // cfg.n_heads
        else:
            self.head_dim = 64  # default

    def measure_entropy(self, prompts: List[str], max_length: int = 512):
        """Measure attention entropy across RoPE frequency dimensions.

        Registers forward hooks on attention layers to capture actual attention
        weights, then computes Shannon entropy per frequency band. High-entropy
        dimensions are stable and need less scaling; low-entropy need more.

        Args:
            prompts: calibration prompts (short, base-context)
            max_length: max token length for calibration
        """
        self.model.eval()
        all_entropies = []

        # Register forward hooks to capture attention weights.
        captured_attn = []

        def make_hook(storage):
            def hook(module, input, output):
                # Try to extract attention weights from output.
                # Different models return different things; handle common cases.
                if isinstance(output, tuple):
                    # Could be (output, attn_weights) or (output, kv_cache)
                    if len(output) >= 2 and isinstance(output[1], torch.Tensor):
                        if output[1].dim() == 4:  # (B, n_heads, T, T)
                            storage.append(output[1].detach())
                elif isinstance(output, torch.Tensor) and output.dim() == 4:
                    storage.append(output.detach())
            return hook

        # Find attention modules and register hooks.
        hooks = []
        attn_modules = []
        for name, module in self.model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                if hasattr(module, "forward"):
                    h = module.register_forward_hook(make_hook(captured_attn))
                    hooks.append(h)
                    attn_modules.append(name)

        for prompt in prompts[:20]:  # limit to 20 for speed
            ids = self.tokenizer(prompt, return_tensors="pt",
                                max_length=max_length, truncation=True).input_ids.to(self.device)

            captured_attn.clear()
            with torch.no_grad():
                self.model(ids)

            # Compute entropy from captured attention weights.
            if captured_attn:
                # Use the first attention layer's weights (representative).
                attn = captured_attn[0]  # (B, n_heads, T, T)
                # Convert to probabilities (if not already).
                if attn.max() > 1.0 or attn.min() < 0.0:
                    attn = torch.softmax(attn, dim=-1)
                # Entropy per head: H = -sum(p * log(p))
                attn_safe = attn.clamp(min=1e-10)
                entropy = -(attn_safe * attn_safe.log()).sum(dim=-1)  # (B, n_heads, T)
                # Average across batch and positions → per-head entropy.
                avg_entropy = entropy.mean(dim=(0, 2))  # (n_heads,)
                all_entropies.append(avg_entropy.cpu())
            else:
                # Fallback: use hidden state variance if no attention hooks fired.
                with torch.no_grad():
                    out = self.model(ids)
                    hidden = out[0] if isinstance(out, tuple) else out
                dim_var = hidden[0].var(dim=0)
                n_bands = self.head_dim // 2
                if dim_var.shape[0] >= n_bands:
                    band_vars = dim_var[:n_bands * 2].view(2, n_bands).mean(dim=0)
                else:
                    band_vars = dim_var[:n_bands]
                band_entropies = (band_vars / band_vars.sum()).clamp(min=1e-8)
                all_entropies.append(band_entropies.cpu())

        # Remove hooks.
        for h in hooks:
            h.remove()

        if all_entropies:
            self.entropies = torch.stack(all_entropies).mean(dim=0).tolist()
            print(f"  [Entropy-ABF] measured entropy across {len(all_entropies)} prompts, "
                  f"{len(self.entropies)} bands (hooked {len(attn_modules)} attn modules)")
        else:
            print("  [Entropy-ABF] WARNING: no entropies measured")

            # Entropy proxy: higher variance = higher entropy.
            band_entropies = (band_vars / band_vars.sum()).clamp(min=1e-8)
            all_entropies.append(band_entropies.cpu())

        if all_entropies:
            self.entropies = torch.stack(all_entropies).mean(dim=0).tolist()
            print(f"  [Entropy-ABF] measured entropy across {len(all_entropies)} prompts, "
                  f"{len(self.entropies)} frequency bands")
        else:
            print("  [Entropy-ABF] WARNING: no entropies measured")

    def compute_scaling(self, target_factor: float = 4.0):
        """Compute per-dimension scaling factors.

        High-entropy dimensions get less scaling (they're already stable).
        Low-entropy dimensions get more scaling (they need help).

        Args:
            target_factor: target context extension factor (e.g. 4.0 = 4x)
        """
        if not self.entropies:
            # Uniform scaling fallback.
            n_bands = self.head_dim // 2
            self.scaling_factors = torch.full((n_bands,), target_factor)
            return

        entropies = torch.tensor(self.entropies)
        # Normalize entropies to [0, 1].
        ent_norm = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)

        # Low-entropy dimensions get more scaling.
        # Scale = target_factor * (1 - ent_norm * 0.5) → range [0.5*target, target]
        self.scaling_factors = target_factor * (1.0 - ent_norm * 0.5)

        print(f"  [Entropy-ABF] scaling factors: min={self.scaling_factors.min():.2f}, "
              f"max={self.scaling_factors.max():.2f}, mean={self.scaling_factors.mean():.2f}")

    def apply_to_model(self):
        """Apply the computed scaling factors to the model's RoPE.

        This modifies the model's positional encoding to use per-dimension
        scaling instead of uniform scaling.
        """
        if self.scaling_factors is None:
            print("  [Entropy-ABF] WARNING: no scaling factors computed, call compute_scaling() first")
            return

        # Find and modify RoPE modules.
        scaling = self.scaling_factors.to(self.device)
        n_modified = 0

        for name, module in self.model.named_modules():
            # Look for rotary embedding or attention modules with RoPE.
            if hasattr(module, "rotary_emb") or "rotary" in name.lower():
                if hasattr(module, "scaling_factor"):
                    # Replace uniform scaling with per-dim scaling.
                    module.scaling_factor = scaling
                    n_modified += 1
                elif hasattr(module, "inv_freq"):
                    # Modify inv_freq directly.
                    with torch.no_grad():
                        # inv_freq shape: (head_dim/2,)
                        if module.inv_freq.shape[0] == scaling.shape[0]:
                            module.inv_freq *= scaling
                            n_modified += 1

        if n_modified == 0:
            # Fallback: store scaling on model for manual application.
            self.model.abf_scaling = scaling
            print(f"  [Entropy-ABF] stored scaling on model.abf_scaling "
                  f"(no RoPE module found to modify directly)")
        else:
            print(f"  [Entropy-ABF] applied scaling to {n_modified} RoPE modules")

    def finetune(self, long_context_samples: List[str], steps: int = 100,
                 lr: float = 1e-5, max_length: int = 4096):
        """Fine-tune on a small set of long-context samples.

        Args:
            long_context_samples: list of long text strings
            steps: fine-tuning steps (~100 is sufficient per the paper)
            lr: low learning rate for fine-tuning
            max_length: max sequence length
        """
        if not long_context_samples:
            print("  [Entropy-ABF] no samples provided, skipping fine-tune")
            return

        self.model.train()
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        if not lora_params:
            # Enable grads for all params (small model, short training).
            for p in self.model.parameters():
                p.requires_grad_(True)
            lora_params = list(self.model.parameters())

        optimizer = torch.optim.AdamW(lora_params, lr=lr)

        print(f"  [Entropy-ABF] fine-tuning {steps} steps on {len(long_context_samples)} samples")
        for step in range(steps):
            # Sample a random chunk from random sample.
            import random
            sample = random.choice(long_context_samples)
            ids = self.tokenizer(sample, return_tensors="pt",
                                max_length=max_length, truncation=True).input_ids.to(self.device)

            if ids.shape[1] < 2:
                continue

            # Standard next-token prediction.
            x = ids[:, :-1]
            y = ids[:, 1:]

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                out = self.model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()

            if (step + 1) % 20 == 0:
                print(f"    step {step+1}/{steps} | loss: {loss.item():.4f}")

        self.model.eval()
        print(f"  [Entropy-ABF] fine-tuning complete")


def apply_abf_scaling(model, scaling_factors: torch.Tensor):
    """Standalone function to apply ABF scaling to a model.

    Args:
        model: the LLM
        scaling_factors: per-dimension scaling factors (head_dim/2,)
    """
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and module.inv_freq is not None:
            with torch.no_grad():
                if module.inv_freq.shape[0] == scaling_factors.shape[0]:
                    module.inv_freq *= scaling_factors.to(module.inv_freq.device)
                    return True
    # Fallback: store on model.
    model.abf_scaling = scaling_factors
    return False
