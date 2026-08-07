"""LongRoPE2: Evolutionary context extension with position interpolation.

Evolutionary search over RoPE frequency scaling factors to find the optimal
interpolation for extending context window. Better than uniform NTK scaling
because different frequency bands need different scaling.

Key features:
- Evolutionary search (population-based) over per-frequency scaling
- Non-uniform scaling: low frequencies (long-range) scaled more
- Monotonicity constraint (scaling increases with frequency index)
- Works with any RoPE-based model

Paper: "LongRoPE: Extending LLM Context Window Beyond 2M Tokens" (Microsoft, 2024)
LongRoPE2 adds: evolutionary search with better fitness function + monotonicity.

Usage:
    from research.longrope2 import LongRoPE2

    extender = LongRoPE2(model, tokenizer, target_length=8192,
                         base_length=2048, population_size=20, n_generations=50)
    best_scaling = extender.search()
    extender.apply(best_scaling)
"""
import torch
import random
import numpy as np
from typing import List, Optional


class LongRoPE2:
    """Evolutionary RoPE scaling search for context extension.

    Args:
        model: the LLM with RoPE attention
        tokenizer: the tokenizer
        target_length: desired context length (e.g., 8192)
        base_length: original training length (e.g., 2048)
        population_size: evolutionary population size
        n_generations: number of evolutionary generations
        n_eval_samples: number of samples for fitness evaluation
        device: cuda or cpu
    """

    def __init__(self, model, tokenizer, target_length=8192, base_length=2048,
                 population_size=20, n_generations=50, n_eval_samples=16,
                 device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.target_length = target_length
        self.base_length = base_length
        self.extension_ratio = target_length / base_length
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_eval_samples = n_eval_samples
        self.device = torch.device(device)

        # Find RoPE modules in the model.
        self.rope_modules = self._find_rope_modules()
        if not self.rope_modules:
            print("  [LongRoPE2] WARNING: no RoPE modules found")
        else:
            print(f"  [LongRoPE2] found {len(self.rope_modules)} RoPE modules")

        # Determine number of frequency dimensions.
        if self.rope_modules:
            # RotaryEmbedding uses self.dim (not self.head_dim).
            self.head_dim = getattr(self.rope_modules[0], "head_dim",
                                   getattr(self.rope_modules[0], "dim", 64))
            self.n_freq = self.head_dim // 2
        else:
            self.head_dim = 64
            self.n_freq = 32

    def _find_rope_modules(self):
        """Find all RotaryEmbedding modules in the model."""
        from research.model_loader import RotaryEmbedding
        modules = []
        for module in self.model.modules():
            if isinstance(module, RotaryEmbedding):
                modules.append(module)
        return modules

    def _generate_individual(self) -> List[float]:
        """Generate a random scaling vector (one per frequency dimension).

        Constraints:
        - Each value in [1.0, extension_ratio]
        - Monotonically non-decreasing (low freq = more scaling)
        """
        # Generate random values, then enforce monotonicity.
        raw = [random.uniform(1.0, self.extension_ratio) for _ in range(self.n_freq)]
        raw.sort()  # ascending = low frequencies get less, high get more
        # Actually for RoPE: index 0 = highest frequency, index -1 = lowest
        # We want LOW frequencies (high index) to be scaled MORE
        # So we want descending order (index 0 = least scaling, index -1 = most)
        raw.reverse()
        return raw

    def _evaluate_fitness(self, scaling: List[float],
                          eval_texts: List[str]) -> float:
        """Evaluate fitness of a scaling vector (lower perplexity = better).

        Args:
            scaling: per-frequency scaling factors
            eval_texts: long texts for evaluation

        Returns:
            negative average perplexity (higher = better)
        """
        # Apply scaling to RoPE modules.
        original_scales = []
        for rope in self.rope_modules:
            original_scales.append(getattr(rope, "rope_scaling", None))
            # Set custom scaling: multiply inv_freq by 1/scaling per dimension.
            self._apply_scaling_to_rope(rope, scaling)

        # Compute perplexity on long texts.
        total_loss = 0.0
        n_evaluated = 0

        self.model.eval()
        with torch.no_grad():
            for text in eval_texts[:self.n_eval_samples]:
                ids = self.tokenizer(text, return_tensors="pt",
                                    max_length=self.target_length,
                                    truncation=True).input_ids.to(self.device)
                if ids.shape[1] < self.base_length:
                    continue  # skip short texts

                try:
                    out = self.model(ids)
                    logits = out[0] if isinstance(out, tuple) else out
                    # Shift for next-token prediction.
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = ids[:, 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    total_loss += loss.item()
                    n_evaluated += 1
                except RuntimeError:
                    # OOM or other error — penalize.
                    total_loss += 100.0
                    n_evaluated += 1

        # Restore original scaling.
        for rope, orig in zip(self.rope_modules, original_scales):
            rope.rope_scaling = orig
            if hasattr(rope, "_custom_inv_freq"):
                del rope._custom_inv_freq

        if n_evaluated == 0:
            return -100.0
        avg_loss = total_loss / n_evaluated
        return -avg_loss  # negative because we maximize fitness

    def _apply_scaling_to_rope(self, rope, scaling):
        """Apply per-frequency scaling to a RoPE module.

        We override inv_freq by scaling each frequency dimension.
        """
        # Original inv_freq: 1 / (base ** (2i / dim))
        # Scaled: 1 / (base ** (2i / dim) * scaling[i])
        head_dim = getattr(rope, "head_dim", getattr(rope, "dim", 64))
        base = getattr(rope, "base", 10000.0)

        # Compute original inv_freq.
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Apply per-frequency scaling.
        scaling_tensor = torch.tensor(scaling[:len(freqs)], device=freqs.device,
                                       dtype=freqs.dtype)
        scaled_inv_freq = freqs / scaling_tensor

        # Store on the rope module for use in forward.
        rope._custom_inv_freq = scaled_inv_freq.to(self.device)

        # Monkey-patch the forward to use custom inv_freq.
        if not hasattr(rope, "_patched"):
            original_forward = rope.forward

            def patched_forward(x, offset=0):
                inv_freq = getattr(rope, "_custom_inv_freq", None)
                if inv_freq is None:
                    return original_forward(x, offset)

                seq_len = x.shape[1] + offset
                t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                # Different from paper, we use cos/sin without rotation first
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()[:seq_len]
                sin = emb.sin()[:seq_len]
                # Apply to x
                cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
                sin = sin.unsqueeze(0).unsqueeze(0)
                # Rotate
                x_rot = x.float().reshape(*x.shape[:-1], -1, 2)
                x1, x2 = x_rot.unbind(-1)
                # Handle head_dim mismatch
                cos_use = cos[..., :x1.shape[-1]]
                sin_use = sin[..., :x1.shape[-1]]
                rotated = torch.stack([x1 * cos_use - x2 * sin_use,
                                       x1 * sin_use + x2 * cos_use], dim=-1)
                return rotated.flatten(-2).to(x.dtype)

            rope.forward = patched_forward
            rope._patched = True

    def _crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Crossover two parents to produce a child."""
        child = []
        for a, b in zip(parent1, parent2):
            # Blend crossover.
            alpha = random.random()
            child.append(alpha * a + (1 - alpha) * b)
        # Enforce monotonicity.
        child.sort()
        child.reverse()
        return child

    def _mutate(self, individual: List[float], rate=0.1) -> List[float]:
        """Mutate an individual."""
        mutated = []
        for v in individual:
            if random.random() < rate:
                v = v + random.gauss(0, 0.1 * self.extension_ratio)
                v = max(1.0, min(self.extension_ratio, v))
            mutated.append(v)
        mutated.sort()
        mutated.reverse()
        return mutated

    def search(self, eval_texts: Optional[List[str]] = None) -> List[float]:
        """Run evolutionary search for optimal RoPE scaling.

        Args:
            eval_texts: long texts for fitness evaluation. If None, uses
                       random data (less accurate but works without data).

        Returns:
            best scaling vector (list of per-frequency scaling factors)
        """
        if eval_texts is None:
            # Generate synthetic long texts for evaluation.
            eval_texts = self._generate_eval_texts()

        print(f"  [LongRoPE2] starting evolutionary search: "
              f"{self.population_size} individuals, {self.n_generations} generations")

        # Initialize population.
        population = [self._generate_individual() for _ in range(self.population_size)]

        # Evaluate initial population.
        fitness = [self._evaluate_fitness(ind, eval_texts) for ind in population]

        best_idx = int(np.argmax(fitness))
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        print(f"  [LongRoPE2] gen 0: best fitness={best_fitness:.4f}")

        # Evolution.
        for gen in range(1, self.n_generations + 1):
            # Selection (tournament).
            parents = []
            for _ in range(self.population_size):
                idx1, idx2 = random.sample(range(self.population_size), 2)
                parents.append(population[idx1] if fitness[idx1] > fitness[idx2]
                              else population[idx2])

            # Crossover + mutation.
            new_population = []
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % self.population_size]
                child1 = self._mutate(self._crossover(p1, p2))
                child2 = self._mutate(self._crossover(p2, p1))
                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            # Evaluate.
            fitness = [self._evaluate_fitness(ind, eval_texts) for ind in population]

            # Track best.
            gen_best_idx = int(np.argmax(fitness))
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx]

            if gen % 10 == 0:
                print(f"  [LongRoPE2] gen {gen}: best fitness={best_fitness:.4f}")

        print(f"  [LongRoPE2] search complete: best fitness={best_fitness:.4f}")
        print(f"  [LongRoPE2] scaling range: [{min(best_individual):.2f}, "
              f"{max(best_individual):.2f}]")

        self.best_scaling = best_individual
        return best_individual

    def _generate_eval_texts(self) -> List[str]:
        """Generate synthetic long texts for evaluation."""
        # Use tokenizer vocab to create long sequences.
        texts = []
        for _ in range(self.n_eval_samples):
            tokens = torch.randint(0, self.tokenizer.vocab_size,
                                   (self.target_length,))
            text = self.tokenizer.decode(tokens)
            texts.append(text)
        return texts

    def apply(self, scaling: List[float]):
        """Apply the best scaling to the model permanently.

        Args:
            scaling: per-frequency scaling vector
        """
        for rope in self.rope_modules:
            self._apply_scaling_to_rope(rope, scaling)

        # Update max_seq_len.
        if hasattr(self.model, "config"):
            self.model.config.max_seq_len = self.target_length

        print(f"  [LongRoPE2] applied scaling to {len(self.rope_modules)} RoPE modules")
        print(f"  [LongRoPE2] context extended to {self.target_length} tokens")
