"""Thinking Model Wrapper for ForgeLM.

Converts ForgeLM into a thinking model (DeepSeek-R1 / QwQ style):
  1. Model generates a  reasoning block first
  2. Then generates the final answer after </think>
  3. Reasoning trace is extracted and logged separately
  4. Answer is extracted for execution/evaluation

Prompt format:
  User: {task}
  Assistant: <think>
  Let me reason through this step by step...
  First, I need to...
  </think>
  {final answer / code}

The thinking block:
  - Encourages chain-of-thought reasoning before committing to an answer
  - Reasoning traces become training data for the self-play loop
  - Failed reasoning paths teach what NOT to do
  - Successful reasoning paths become knowledge packets

Usage:
    from research.architecture.thinking_model import ThinkingModel
    thinker = ThinkingModel(model, tokenizer, device="cuda")
    result = thinker.generate_with_thinking("Check if 17 is prime")
    # result = {"reasoning": "...", "answer": "...", "code": "..."}
"""
import re
import time
from typing import Dict, List, Optional, Tuple

import torch


class ThinkingModel:
    """Wraps ForgeLM with  reasoning blocks.

    The model is prompted to:
      1. Think step-by-step inside  tags
      2. Produce the final answer after </think>
      3. The answer is parsed to extract code (for self-play)
    """

    # Token strings for thinking blocks
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(self, model, tokenizer, device: str = "cuda",
                 max_think_tokens: int = 200,
                 max_answer_tokens: int = 200,
                 temperature: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_think_tokens = max_think_tokens
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature

        # Pre-compute token IDs for think tags
        self.think_open_ids = tokenizer.encode(
            self.THINK_OPEN, add_special_tokens=False)
        self.think_close_ids = tokenizer.encode(
            self.THINK_CLOSE, add_special_tokens=False)
        self.think_close_id = self.think_close_ids[-1] if self.think_close_ids else None

        # Newline token for stopping
        nl = tokenizer.encode("\n", add_special_tokens=False)
        self.newline_id = nl[0] if nl else 198

        # Track reasoning history
        self.reasoning_history: list[dict] = []

    def _build_thinking_prompt(self, task: str,
                                context: str = "",
                                code_prefix: str = "") -> str:
        """Build a prompt that encourages thinking before answering."""
        prompt = ""

        if context:
            prompt += f"# Context:\n{context}\n\n"

        if code_prefix:
            prompt += f"{code_prefix}\n"

        prompt += f'"""{task}"""\n'
        prompt += f"{self.THINK_OPEN}\n"
        prompt += "Let me reason through this step by step.\n"

        return prompt

    def _build_fix_thinking_prompt(self, task: str,
                                    broken_code: str,
                                    error: str,
                                    round_num: int) -> str:
        """Build a thinking prompt for fixing broken code."""
        error_short = error.strip()[:400]
        code_short = broken_code.strip()[:600]

        prompt = (
            f"# Task: {task}\n"
            f"# Previous attempt (BROKEN, round {round_num}):\n"
            f"{code_short}\n"
            f"# Error:\n"
            f"{error_short}\n"
            f"# Let me think about what went wrong and fix it.\n"
            f"{self.THINK_OPEN}\n"
            f"The error was: {error_short[:100]}\n"
            f"This means I need to fix the {self._classify_error(error)}.\n"
        )
        return prompt

    @staticmethod
    def _classify_error(error: str) -> str:
        """Classify error type from error message."""
        err = error.lower()
        if "syntaxerror" in err:
            return "syntax"
        if "indentationerror" in err:
            return "indentation"
        if "nameerror" in err:
            return "undefined variable"
        if "typeerror" in err:
            return "type mismatch"
        if "valueerror" in err:
            return "invalid value"
        if "indexerror" in err:
            return "out of bounds"
        if "attributeerror" in err:
            return "missing attribute"
        if "importerror" in err or "modulenotfounderror" in err:
            return "missing import"
        return "logic error"

    def _generate_tokens(self, input_ids: torch.Tensor,
                          max_tokens: int,
                          stop_token_ids: list[int] | None = None,
                          stop_on_double_newline: bool = False) -> tuple[torch.Tensor, list[float], int]:
        """Generate tokens with optional stop conditions.

        Returns:
            (generated_ids, log_probs, n_tokens_generated)
        """
        log_probs = []
        cur_ids = input_ids
        newline_count = 0

        with torch.inference_mode():
            for step in range(max_tokens):
                logits, _ = self.model(cur_ids)
                next_logits = logits[0, -1]

                if self.temperature > 0:
                    probs = torch.softmax(next_logits / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax()

                # Batch sync: single .tolist() for log prob + token id.
                lp = torch.log_softmax(next_logits, dim=-1)[next_token]
                lp_val, next_id = torch.stack([lp, next_token]).tolist()
                log_probs.append(lp_val)

                cur_ids = torch.cat(
                    [cur_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                # Check stop tokens
                if stop_token_ids and next_id in stop_token_ids:
                    break

                # EOS
                if next_id == self.tokenizer.eos_token_id:
                    break

                # Double newline stop (for code blocks)
                if stop_on_double_newline:
                    if next_id == self.newline_id:
                        newline_count += 1
                        if newline_count >= 3 and step > 5:
                            break
                    else:
                        newline_count = 0

        n_generated = cur_ids.shape[1] - input_ids.shape[1]
        return cur_ids, log_probs, n_generated

    def generate_with_thinking(self, task: str,
                                context: str = "",
                                code_prefix: str = "") -> dict:
        """Generate a response with thinking + answer.

        Two-phase generation:
          Phase 1: Generate reasoning inside  block
          Phase 2: Generate answer/code after </think>

        Returns:
            {
                "reasoning": str,      # text inside  block
                "answer": str,         # text after </think>
                "code": str,           # extracted code from answer
                "think_tokens": int,   # tokens in reasoning
                "answer_tokens": int,  # tokens in answer
                "think_time_ms": float,
                "answer_time_ms": float,
                "confidence": float,   # mean prob of generated tokens
                "full_text": str,      # complete generation
            }
        """
        # Phase 1: Generate reasoning (thinking block)
        think_prompt = self._build_thinking_prompt(task, context, code_prefix)
        think_input = self.tokenizer(
            think_prompt, return_tensors="pt").input_ids.to(self.device)

        t0 = time.time()
        # Generate thinking tokens — stop when we see </think> or hit limit
        think_ids, think_logprobs, n_think = self._generate_tokens(
            think_input,
            max_tokens=self.max_think_tokens,
            stop_token_ids=[self.think_close_id] if self.think_close_id else None,
            stop_on_double_newline=False)
        think_time_ms = (time.time() - t0) * 1000

        # Decode reasoning
        think_text = self.tokenizer.decode(
            think_ids[0, think_input.shape[1]:],
            skip_special_tokens=True)

        # Remove the </think> tag from reasoning if present
        think_text = think_text.replace(self.THINK_CLOSE, "").strip()
        think_text = think_text.replace(self.THINK_OPEN, "").strip()

        # Phase 2: Generate answer (after </think>)
        # Build the full prompt: original + thinking + </think> + newline
        answer_prompt_ids = think_ids
        # Append </think> + newline if not already there
        if self.think_close_ids:
            close_ids = torch.tensor(
                [self.think_close_ids], device=self.device).unsqueeze(0)
            answer_prompt_ids = torch.cat([answer_prompt_ids, close_ids], dim=1)
        # Append newline
        nl_ids = torch.tensor([[self.newline_id]], device=self.device)
        answer_prompt_ids = torch.cat([answer_prompt_ids, nl_ids], dim=1)

        t1 = time.time()
        answer_ids, answer_logprobs, n_answer = self._generate_tokens(
            answer_prompt_ids,
            max_tokens=self.max_answer_tokens,
            stop_on_double_newline=True)
        answer_time_ms = (time.time() - t1) * 1000

        # Decode answer
        answer_text = self.tokenizer.decode(
            answer_ids[0, answer_prompt_ids.shape[1]:],
            skip_special_tokens=True).strip()

        # Extract code from answer
        code = self._extract_code(answer_text)

        # Compute confidence
        all_logprobs = think_logprobs + answer_logprobs
        confidence = pow(2.71828, sum(all_logprobs) / max(len(all_logprobs), 1))

        # Full text
        full_text = f"{self.THINK_OPEN}\n{think_text}\n{self.THINK_CLOSE}\n{answer_text}"

        result = {
            "reasoning": think_text,
            "answer": answer_text,
            "code": code,
            "think_tokens": n_think,
            "answer_tokens": n_answer,
            "think_time_ms": think_time_ms,
            "answer_time_ms": answer_time_ms,
            "total_time_ms": think_time_ms + answer_time_ms,
            "confidence": confidence,
            "full_text": full_text,
        }

        # Track reasoning
        self.reasoning_history.append({
            "task": task[:80],
            "reasoning_length": len(think_text),
            "answer_length": len(answer_text),
            "confidence": confidence,
            "think_tokens": n_think,
        })

        return result

    def generate_fix_with_thinking(self, task: str,
                                    broken_code: str,
                                    error: str,
                                    round_num: int) -> dict:
        """Generate a fix with thinking — used in recursive self-play retry loop.

        The model:
          1. Thinks about what went wrong (error analysis)
          2. Reasons about how to fix it
          3. Produces the fixed code after </think>
        """
        fix_prompt = self._build_fix_thinking_prompt(
            task, broken_code, error, round_num)
        fix_input = self.tokenizer(
            fix_prompt, return_tensors="pt").input_ids.to(self.device)

        # Phase 1: Thinking (analyze error + plan fix)
        t0 = time.time()
        think_ids, think_logprobs, n_think = self._generate_tokens(
            fix_input,
            max_tokens=self.max_think_tokens,
            stop_token_ids=[self.think_close_id] if self.think_close_id else None)
        think_time_ms = (time.time() - t0) * 1000

        think_text = self.tokenizer.decode(
            think_ids[0, fix_input.shape[1]:],
            skip_special_tokens=True)
        think_text = think_text.replace(self.THINK_CLOSE, "").strip()
        think_text = think_text.replace(self.THINK_OPEN, "").strip()

        # Phase 2: Fixed code
        answer_prompt_ids = think_ids
        if self.think_close_ids:
            close_ids = torch.tensor(
                [self.think_close_ids], device=self.device).unsqueeze(0)
            answer_prompt_ids = torch.cat([answer_prompt_ids, close_ids], dim=1)
        nl_ids = torch.tensor([[self.newline_id]], device=self.device)
        answer_prompt_ids = torch.cat([answer_prompt_ids, nl_ids], dim=1)

        t1 = time.time()
        answer_ids, answer_logprobs, n_answer = self._generate_tokens(
            answer_prompt_ids,
            max_tokens=self.max_answer_tokens,
            stop_on_double_newline=True)
        answer_time_ms = (time.time() - t1) * 1000

        answer_text = self.tokenizer.decode(
            answer_ids[0, answer_prompt_ids.shape[1]:],
            skip_special_tokens=True).strip()

        code = self._extract_code(answer_text)

        all_logprobs = think_logprobs + answer_logprobs
        confidence = pow(2.71828, sum(all_logprobs) / max(len(all_logprobs), 1))

        return {
            "reasoning": think_text,
            "answer": answer_text,
            "code": code,
            "think_tokens": n_think,
            "answer_tokens": n_answer,
            "think_time_ms": think_time_ms,
            "answer_time_ms": answer_time_ms,
            "total_time_ms": think_time_ms + answer_time_ms,
            "confidence": confidence,
            "error_type": self._classify_error(error),
            "full_text": f"{self.THINK_OPEN}\n{think_text}\n{self.THINK_CLOSE}\n{answer_text}",
        }

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from generated text.

        Handles:
          - ```python ... ``` blocks
          - ``` ... ``` blocks
          - Raw code (no markdown)
        """
        # Try to extract from markdown code blocks
        code_block = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Strip markdown fences if incomplete
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def print_reasoning_stats(self):
        """Print statistics about reasoning traces."""
        if not self.reasoning_history:
            print("  No reasoning traces yet.")
            return

        n = len(self.reasoning_history)
        avg_think_tokens = sum(r["think_tokens"] for r in self.reasoning_history) / n
        avg_reasoning_len = sum(r["reasoning_length"] for r in self.reasoning_history) / n
        avg_confidence = sum(r["confidence"] for r in self.reasoning_history) / n

        print(f"\n{'='*70}")
        print("Thinking Model Statistics")
        print(f"{'='*70}")
        print(f"  Total reasoning traces: {n}")
        print(f"  Avg thinking tokens:    {avg_think_tokens:.1f}")
        print(f"  Avg reasoning length:   {avg_reasoning_len:.0f} chars")
        print(f"  Avg confidence:         {avg_confidence:.3f}")
        print(f"{'='*70}")


def main():
    """Quick test of the thinking model."""
    from research.config import get_config
    from research.model_loader import ModelLoader
    from research.tokenizer_cache import get_tokenizer

    print("=" * 70)
    print("Thinking Model Test — ForgeLM V2")
    print("=" * 70)

    # Load model
    print("\n[1] Loading ForgeLM V2...")
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(cfg,
        checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
    model.to("cuda").eval()
    tokenizer = get_tokenizer("research/checkpoints/qwen_hf")

    # Create thinking model
    print("\n[2] Creating thinking model wrapper...")
    thinker = ThinkingModel(model, tokenizer, device="cuda",
                            max_think_tokens=150,
                            max_answer_tokens=150)

    # Test tasks
    test_tasks = [
        "Check if 17 is prime",
        "Compute fibonacci(10)",
        "Reverse the string 'hello'",
        "Sort the list [5, 2, 8, 1, 9]",
    ]

    print("\n[3] Running thinking model on test tasks...")
    for task in test_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        result = thinker.generate_with_thinking(task)

        print(f"\n--- Reasoning ({result['think_tokens']} tokens, "
              f"{result['think_time_ms']:.0f}ms) ---")
        print(result["reasoning"][:300])

        print(f"\n--- Answer ({result['answer_tokens']} tokens, "
              f"{result['answer_time_ms']:.0f}ms) ---")
        print(result["answer"][:300])

        print("\n--- Extracted Code ---")
        print(result["code"][:200])

        print(f"\nConfidence: {result['confidence']:.3f}")

    thinker.print_reasoning_stats()


if __name__ == "__main__":
    main()
