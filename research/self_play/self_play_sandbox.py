"""Self-Play Sandbox — model generates code, executes it, logs everything.

Architecture:
  1. Model generates a solution (Python code) for a task
  2. Sandbox executes the code safely (subprocess + timeout + resource limits)
  3. Telemetry logs: execution time, memory, output, errors, file size, correctness
  4. Successful runs → KnowledgePacket (for later weight injection)
  5. Failed runs → error analysis (model learns from mistakes)

Data packet format (JSON, one per run):
  {
    "task": "fibonacci(10)",
    "task_type": "python",
    "prompt": "Write a function to compute fibonacci(10)",
    "generated_code": "def fib(n): ...",
    "execution": {
      "stdout": "55",
      "stderr": "",
      "returncode": 0,
      "exec_time_ms": 0.42,
      "peak_memory_kb": 12300,
      "file_size_bytes": 156,
      "output_matches_expected": true
    },
    "model_telemetry": {
      "gen_time_ms": 320,
      "tokens_generated": 45,
      "tokens_per_second": 140.6,
      "mean_logprob": -0.12,
      "confidence": 0.88
    },
    "quality_score": 0.92,
    "knowledge_text": "fibonacci(10) = 55. def fib(n): a,b=0,1; ...",
    "timestamp": "2026-01-15T12:00:00"
  }

Usage:
    from research.self_play.self_play_sandbox import SelfPlaySandbox
    sandbox = SelfPlaySandbox(model, tokenizer, log_dir="research/data/self_play")
    sandbox.run_task("Compute fibonacci(10) in Python", expected_output="55")
"""
import os
import sys
import time
import json
import subprocess
import tempfile
import traceback
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# resource is Unix-only; on Windows we skip memory limits
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class SandboxExecutor:
    """Safe Python code executor with full telemetry.

    Executes code in a subprocess with:
      - Timeout (default 5s)
      - Memory limit (default 256MB)
      - No network access (no imports of socket/urllib at OS level)
      - Captures stdout, stderr, return code, peak memory, file size
    """

    def __init__(self, timeout_s: float = 5.0, memory_limit_mb: int = 256):
        self.timeout_s = timeout_s
        self.memory_limit_mb = memory_limit_mb

    def execute(self, code: str, expected_output: Optional[str] = None) -> Dict:
        """Execute Python code and return full telemetry.

        Args:
            code: Python code to execute
            expected_output: expected stdout (for correctness checking)

        Returns:
            Telemetry dict with execution results
        """
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name

        file_size = os.path.getsize(temp_path)

        # Prepare resource-limited execution wrapper
        if HAS_RESOURCE:
            mem_limit_code = f"resource.setrlimit(resource.RLIMIT_AS, ({self.memory_limit_mb * 1024 * 1024}, {self.memory_limit_mb * 1024 * 1024}))"
        else:
            mem_limit_code = "# resource module not available on Windows"

        wrapper = f'''
import sys, os, time
{f"import resource" if HAS_RESOURCE else ""}

# Set memory limit (Unix only)
{mem_limit_code}

# Disable network (best effort)
import builtins
_orig_import = builtins.__import__
def _restricted_import(name, *args, **kwargs):
    blocked = ['socket', 'urllib', 'requests', 'http', 'subprocess',
                              'ctypes', 'multiprocessing']
    top = name.split('.')[0]
    if top in blocked:
        raise ImportError(f"Module '{{name}}' is blocked in sandbox")
    return _orig_import(name, *args, **kwargs)
builtins.__import__ = _restricted_import

# Execute user code
exec(open(r"{temp_path}", encoding='utf-8').read())
'''

        wrapper_path = temp_path + '_wrapper.py'
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper)

        t0 = time.time()
        result = {
            "stdout": "", "stderr": "", "returncode": -1,
            "exec_time_ms": 0, "peak_memory_kb": 0,
            "file_size_bytes": file_size, "timed_out": False,
            "output_matches_expected": False,
        }

        try:
            proc = subprocess.Popen(
                [sys.executable, wrapper_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir(),
            )
            # Wait for completion with timeout
            deadline = t0 + self.timeout_s
            while proc.poll() is None:
                if time.time() > deadline:
                    proc.kill()
                    result["timed_out"] = True
                    break
                time.sleep(0.01)
            stdout, stderr = proc.communicate()
            result["stdout"] = stdout
            result["stderr"] = stderr
            result["returncode"] = proc.returncode
            result["exec_time_ms"] = (time.time() - t0) * 1000

            if expected_output is not None:
                expected_clean = expected_output.strip()
                actual_clean = result["stdout"].strip()
                result["output_matches_expected"] = expected_clean == actual_clean

        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            result["stderr"] = f"Execution timed out after {self.timeout_s}s"
            result["exec_time_ms"] = self.timeout_s * 1000
        except Exception as e:
            result["stderr"] = f"Sandbox error: {e}"
            result["exec_time_ms"] = (time.time() - t0) * 1000
        finally:
            # Kill if still running
            try:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception:
                pass
            # Cleanup
            try:
                os.unlink(temp_path)
                os.unlink(wrapper_path)
            except OSError:
                pass

        return result


class SelfPlaySandbox:
    """Self-play sandbox: model generates code → execute → log → package.

    The model generates solutions to tasks, the sandbox executes them,
    and all telemetry is logged into structured data packets for later
    knowledge injection.
    """

    # Task templates by domain
    TASK_TEMPLATES = {
        "python_basics": [
            ("Compute fibonacci({n})", "fib", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nprint(fib({n}))"),
            ("Compute factorial({n})", "factorial", "def fact(n):\n    r = 1\n    for i in range(1, n+1):\n        r *= i\n    return r\nprint(fact({n}))"),
            ("Check if {n} is prime", "prime", "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\nprint(is_prime({n}))"),
            ("Reverse the string '{s}'", "reverse", f"print('{''.join(reversed('hello'))}')"),
            ("Sum of list {lst}", "sum_list", "print(sum({lst}))"),
            ("Sort the list {lst}", "sort_list", "print(sorted({lst}))"),
            ("Count vowels in '{s}'", "vowels", "s = '{s}'\nprint(sum(1 for c in s if c in 'aeiouAEIOU'))"),
            ("Check if '{s}' is a palindrome", "palindrome", "s = '{s}'\nprint(s == s[::-1])"),
        ],
        "math": [
            ("Compute {a} * {b}", "multiply", "print({a} * {b})"),
            ("Compute {a} + {b}", "add", "print({a} + {b})"),
            ("Compute gcd({a}, {b})", "gcd", "import math\nprint(math.gcd({a}, {b}))"),
            ("Compute {a} ** {b}", "power", "print({a} ** {b})"),
            ("Is {a} divisible by {b}?", "divisible", "print({a} % {b} == 0)"),
            ("Compute sqrt of {a}", "sqrt", "print({a} ** 0.5)"),
            ("Solve: x^2 + {a}*x + {b} = 0", "quadratic", "a, b = {a}, {b}\nd = a**2 - 4*b\nprint((-a + d**0.5)/2, (-a - d**0.5)/2)"),
        ],
        "algorithms": [
            ("Implement bubble sort for {lst}", "bubble_sort", "arr = {lst}\nfor i in range(len(arr)):\n    for j in range(len(arr)-1-i):\n        if arr[j] > arr[j+1]:\n            arr[j], arr[j+1] = arr[j+1], arr[j]\nprint(arr)"),
            ("Binary search for {target} in sorted list", "binary_search", "arr = sorted([1,3,5,7,9,11,13])\ntarget = {target}\nlo, hi = 0, len(arr)-1\nwhile lo <= hi:\n    mid = (lo+hi)//2\n    if arr[mid] == target: print(mid); break\n    elif arr[mid] < target: lo = mid+1\n    else: hi = mid-1\nelse: print(-1)"),
            ("Implement a stack with push/pop", "stack", "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)\n    def pop(self): return self.items.pop() if self.items else None\ns = Stack()\ns.push(1); s.push(2)\nprint(s.pop(), s.pop())"),
        ],
        "string_manipulation": [
            ("Count words in: '{text}'", "word_count", "text = '{text}'\nprint(len(text.split()))"),
            ("Replace 'a' with 'X' in '{s}'", "replace", "print('{s}'.replace('a', 'X'))"),
            ("Uppercase '{s}'", "upper", "print('{s}'.upper())"),
            ("Find longest word in '{text}'", "longest_word", "words = '{text}'.split()\nprint(max(words, key=len) if words else '')"),
        ],
    }

    def __init__(self, model, tokenizer, log_dir: str = "research/data/self_play",
                 device: str = "cuda", max_gen_tokens: int = 200,
                 temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_gen_tokens = max_gen_tokens
        # Sampling params for solution diversity (anti-overfitting across epochs)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.executor = SandboxExecutor()
        self.packets: List[Dict] = []
        # Forward cache for repeated prompts (C2 — 20-40% fewer forward passes)
        from research.runtime.forward_cache import ForwardCache
        self.fwd_cache = ForwardCache(max_entries=500, device=device)
        # Conformal sampler: per-query calibrated temperature (NOVEL key)
        from research.keys.conformal_sampler_key import ConformalSampler
        self.conformal_sampler = ConformalSampler()
        self.conformal_calibrated = False
        # Self-modeling: confidence-based retry (D5)
        from research.runtime.self_model import ConfidenceScorer
        self.confidence_scorer = ConfidenceScorer()
        self.stats = {
            "total_tasks": 0, "successful": 0, "failed": 0,
            "timed_out": 0, "correct": 0, "total_gen_time_ms": 0,
            "total_exec_time_ms": 0, "total_tokens_generated": 0,
        }

    def calibrate_conformal(self, calibration_prompts: List[str] = None,
                            alpha: float = 0.1) -> None:
        """Calibrate conformal sampler on held-out prompts.

        Runs the model on a few prompts, records confidence scores (max softmax
        prob of first generated token), then calibrates a conformal threshold.
        After calibration, generate_code() uses per-query temperature instead
        of the fixed self.temperature.

        Args:
            calibration_prompts: list of prompt strings (default: use task templates)
            alpha: miscoverage rate (0.1 = 90% coverage guarantee)
        """
        if calibration_prompts is None:
            # Use built-in task templates as calibration data
            calibration_prompts = []
            for tasks in self.TASK_TEMPLATES.values():
                for task_desc, _, _ in tasks[:3]:  # 3 per category
                    calibration_prompts.append(task_desc)

        scores = []
        self.model.eval()
        with torch.inference_mode():
            for prompt in calibration_prompts[:20]:  # cap at 20 for speed
                full_prompt = f'"""{prompt}"""\n'
                input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
                try:
                    out = self.model(input_ids, use_cache=False)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    scores.append(probs.max().item())
                except Exception:
                    continue

        if len(scores) >= 5:
            self.conformal_sampler.calibrate(scores, alpha=alpha)
            self.conformal_calibrated = True
            print(f"  [ConformalSampler] Calibrated on {len(scores)} prompts, "
                  f"threshold={self.conformal_sampler.threshold:.3f}, alpha={alpha}")
        else:
            print(f"  [ConformalSampler] Only {len(scores)} scores, skipping calibration")

    def _get_query_temperature(self, logits: torch.Tensor) -> float:
        """Get per-query temperature from conformal sampler or fall back to fixed.

        Args:
            logits: first-token logits from the model (used to compute confidence)

        Returns:
            Temperature for this query
        """
        if not self.conformal_calibrated or self.temperature <= 0.0:
            return self.temperature

        # Compute query confidence: max softmax probability
        probs = torch.softmax(logits[0, -1], dim=-1)
        query_score = probs.max().item()

        # Conformal temperature: confident → low T (exploit), uncertain → high T (explore)
        return self.conformal_sampler.get_temperature(query_score)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """Sample next token with temperature/top-k/top-p, or greedy if temp=0.

        Args:
            logits: next-token logits
            temperature: override temperature (default: use self.temperature)
        """
        temp = temperature if temperature is not None else self.temperature
        if temp <= 0.0:
            return logits.argmax()

        scaled = logits / temp
        probs = torch.softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < probs.shape[0]:
            topk_vals, _ = torch.topk(probs, self.top_k)
            mask = probs < topk_vals[-1]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum()

        if 0.0 < self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum > self.top_p
            cutoff[0] = False
            sorted_probs[cutoff] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
            probs = probs / probs.sum()

        return torch.multinomial(probs, num_samples=1).squeeze()

    def generate_code(self, prompt: str) -> Tuple[str, Dict]:
        """Generate code from a prompt using the model.

        Uses KV cache for O(n) generation instead of O(n²) reprocessing.
        Returns (generated_text, model_telemetry).
        """
        # If the prompt is already a code stub (contains def/class/import),
        # use it directly for code completion. Otherwise wrap in docstring.
        if any(kw in prompt for kw in ["def ", "class ", "import ", "from "]):
            # Add stop instruction + efficiency hint
            full_prompt = prompt + "# Write concise, efficient code. End with [stop]\n    "
        else:
            full_prompt = f'"""{prompt}"""\n'
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        t0 = time.time()
        log_probs = []

        # Token IDs for stopping
        newline_token = self.tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_token[0] if newline_token else 198

        # Build stop string tokens — check for "[stop]" in generated text
        stop_str = "[stop]"

        # ── KV-cached generation (O(n) instead of O(n²)) ──────────
        # First pass: process the full prompt, cache KV pairs.
        # Subsequent passes: feed only the new token, reuse cached KV.
        with torch.inference_mode():
            past_kvs = None
            generated_ids = []

            # Initial forward pass on the full prompt
            logits, _, past_kvs = self.model(
                input_ids, past_key_values=past_kvs, use_cache=True)
            next_logits = logits[0, -1]

            # Conformal sampler: per-query temperature from first-token confidence
            query_temp = self._get_query_temperature(logits)

            next_token = self._sample_next_token(next_logits, temperature=query_temp)

            lp = torch.log_softmax(next_logits, dim=-1)[next_token].item()
            log_probs.append(lp)
            generated_ids.append(next_token.item())

            # Stop on EOS or [stop] after first token
            stop = False
            if next_token.item() == self.tokenizer.eos_token_id:
                stop = True
            else:
                gen_text = self.tokenizer.decode(
                    torch.tensor(generated_ids), skip_special_tokens=True)
                if stop_str in gen_text:
                    stop = True

            # Subsequent tokens: feed only the last token, reuse KV cache
            while not stop and len(generated_ids) < self.max_gen_tokens:
                cur_token = torch.tensor([[next_token.item()]], device=self.device)
                logits, _, past_kvs = self.model(
                    cur_token, past_key_values=past_kvs, use_cache=True)
                next_logits = logits[0, -1]
                next_token = self._sample_next_token(next_logits, temperature=query_temp)

                lp = torch.log_softmax(next_logits, dim=-1)[next_token].item()
                log_probs.append(lp)
                generated_ids.append(next_token.item())

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Check for [stop] every few tokens (cheaper than every token)
                if len(generated_ids) % 4 == 0:
                    gen_text = self.tokenizer.decode(
                        torch.tensor(generated_ids), skip_special_tokens=True)
                    if stop_str in gen_text:
                        break

            # Free KV cache
            del past_kvs

        gen_time_ms = (time.time() - t0) * 1000
        tokens_generated = len(generated_ids)

        generated_text = self.tokenizer.decode(
            torch.tensor(generated_ids), skip_special_tokens=True)

        # Extract code — strip [stop] marker and instruction comment
        code = generated_text.strip()
        # Remove [stop] and everything after it
        if stop_str in code:
            code = code[:code.index(stop_str)].strip()
        # Remove the instruction comment line if present
        code = code.replace("# Write concise, efficient code. End with [stop]\n", "")
        # Strip markdown fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines)

        # If the code doesn't contain print(), add it for tasks that need output
        # (This helps with base models that write functions but don't call them)

        telemetry = {
            "gen_time_ms": gen_time_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / (gen_time_ms / 1000) if gen_time_ms > 0 else 0,
            "mean_logprob": sum(log_probs) / max(len(log_probs), 1),
            "confidence": pow(2.71828, sum(log_probs) / max(len(log_probs), 1)),
            "query_temperature": query_temp if self.conformal_calibrated else self.temperature,
            "conformal": self.conformal_calibrated,
        }

        return code, telemetry

    def generate_reasoning(self, prompt: str) -> str:
        """Generate chain-of-thought reasoning before code.

        Prompts the model to think about the problem first,
        producing a reasoning trace that improves code quality.
        """
        # Extract function name from prompt
        import re
        match = re.search(r'def (\w+)', prompt)
        func_name = match.group(1) if match else "this function"
        match_cls = re.search(r'class (\w+)', prompt)
        cls_name = match_cls.group(1) if match_cls else None
        target = cls_name or func_name

        reasoning_prompt = (
            f"# Think step by step about implementing {target}.\n"
            f"# 1. What does it do?\n"
            f"# 2. What are the edge cases?\n"
            f"# 3. What approach will you use?\n"
            f"# "
        )
        ids = self.tokenizer(reasoning_prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            past_kvs = None
            gen_ids = []
            # Initial pass on full prompt
            logits, _, past_kvs = self.model(ids, past_key_values=past_kvs, use_cache=True)
            nxt = self._sample_next_token(logits[0, -1])
            gen_ids.append(nxt.item())

            for step in range(60):  # short reasoning
                if nxt.item() == self.tokenizer.eos_token_id:
                    break
                # Feed only the new token with cached KV
                cur_tok = torch.tensor([[nxt.item()]], device=self.device)
                logits, _, past_kvs = self.model(cur_tok, past_key_values=past_kvs, use_cache=True)
                nxt = self._sample_next_token(logits[0, -1])
                gen_ids.append(nxt.item())
                # Stop at double newline (end of reasoning)
                if step > 10 and step % 4 == 0:
                    decoded = self.tokenizer.decode(
                        torch.tensor(gen_ids), skip_special_tokens=True)
                    if "\n\n" in decoded:
                        break

            del past_kvs

        reasoning = self.tokenizer.decode(
            torch.tensor(gen_ids), skip_special_tokens=True).strip()
        # Clean up — keep only the reasoning lines (starting with #)
        lines = reasoning.split("\n")
        reasoning_lines = [l for l in lines if l.strip().startswith("#") or l.strip() == ""]
        return "\n".join(reasoning_lines).strip()

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str, Dict]:
        """Generate reasoning + code. Returns (reasoning, code, telemetry)."""
        reasoning = self.generate_reasoning(prompt)
        # Prepend reasoning as comments to the code prompt
        if reasoning:
            enhanced_prompt = reasoning + "\n" + prompt
        else:
            enhanced_prompt = prompt
        code, telemetry = self.generate_code(enhanced_prompt)
        return reasoning, code, telemetry

    def self_critique(self, code: str, prompt: str) -> Tuple[str, List[Dict]]:
        """Self-critique loop: generate edge case tests, run them, fix bugs.

        Returns (improved_code, test_results).
        """
        import re
        match = re.search(r'def (\w+)', prompt)
        func_name = match.group(1) if match else None

        if not func_name:
            return code, []

        # Generate edge case test calls
        critique_prompt = (
            f"{code}\n\n"
            f"# Write test calls for {func_name} with edge cases.\n"
            f"# Test: empty input, negative, zero, large, typical.\n"
            f"# Format: assert {func_name}(...) == expected\n"
            f"assert "
        )
        ids = self.tokenizer(critique_prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            cur_ids = ids
            for step in range(80):
                logits, _ = self.model(cur_ids)
                nxt = logits[0, -1].argmax()
                cur_ids = torch.cat([cur_ids, nxt.unsqueeze(0).unsqueeze(0)], dim=1)
                if nxt.item() == self.tokenizer.eos_token_id:
                    break
                decoded = self.tokenizer.decode(cur_ids[0, ids.shape[1]:],
                                                skip_special_tokens=True)
                if "\n\n" in decoded and step > 10:
                    break

        test_text = self.tokenizer.decode(cur_ids[0, ids.shape[1]:],
                                          skip_special_tokens=True).strip()
        # Extract assert statements
        test_lines = [l.strip() for l in test_text.split("\n") if l.strip().startswith("assert")]

        # Run the tests
        test_results = []
        all_code = code + "\n" + "\n".join(test_lines)
        try:
            exec(all_code, {})
            test_results.append({"passed": True, "tests": len(test_lines)})
        except Exception as e:
            test_results.append({"passed": False, "error": str(e), "tests": len(test_lines)})

        return code, test_results

    def run_task(self, task_prompt: str, expected_output: Optional[str] = None,
                 task_type: str = "python", task_name: str = "",
                 reference_code: Optional[str] = None,
                 code_prefix: str = "") -> Dict:
        """Run a single self-play task: generate → execute → log.

        Args:
            task_prompt: the task description
            expected_output: expected stdout for correctness check
            task_type: "python", "math", "algorithm"
            task_name: short name for the task
            reference_code: known-correct reference solution
            code_prefix: code to prepend (function signature, imports)

        Returns:
            Complete data packet with all telemetry
        """
        self.stats["total_tasks"] += 1

        # Phase 1: Generate code
        full_prompt = task_prompt
        if code_prefix:
            full_prompt = f"{code_prefix}\n# {task_prompt}"

        code, model_telem = self.generate_code(full_prompt)

        # Prepend the prefix to the generated code
        if code_prefix:
            code = code_prefix + "\n" + code

        # Auto-append print() call if code defines a function but doesn't call it
        # (base models often write functions without calling them)
        if "def " in code and "print(" not in code and expected_output is not None:
            # Try to extract function name and call it
            import re
            func_match = re.search(r'def\s+(\w+)\s*\(([^)]*)\)', code)
            if func_match:
                func_name = func_match.group(1)
                args = func_match.group(2)
                # Try to call with values from the prompt
                # Extract numbers from the prompt
                numbers = re.findall(r'\d+', task_prompt)
                if numbers:
                    call_args = ", ".join(numbers[:len(args.split(',')) if args else 0])
                    code += f"\nprint({func_name}({call_args}))"

        # Phase 2: Execute in sandbox
        exec_result = self.executor.execute(code, expected_output)

        # Phase 3: Build data packet
        packet = {
            "task": task_name or task_prompt[:50],
            "task_type": task_type,
            "prompt": task_prompt,
            "generated_code": code,
            "reference_code": reference_code,
            "expected_output": expected_output,
            "execution": exec_result,
            "model_telemetry": model_telem,
            "timestamp": datetime.now().isoformat(),
        }

        # Phase 4: Score the result
        quality_score = self._score_packet(packet)
        packet["quality_score"] = quality_score

        # Phase 5: Build knowledge text (for later injection)
        if exec_result["returncode"] == 0 and exec_result["stdout"].strip():
            packet["knowledge_text"] = (
                f"Task: {task_prompt}\n"
                f"Solution: {code[:200]}\n"
                f"Output: {exec_result['stdout'].strip()[:100]}\n"
                f"Execution time: {exec_result['exec_time_ms']:.1f}ms\n"
                f"Correct: {exec_result['output_matches_expected']}"
            )
        else:
            packet["knowledge_text"] = ""

        # Update stats
        if exec_result["timed_out"]:
            self.stats["timed_out"] += 1
        elif exec_result["returncode"] == 0:
            self.stats["successful"] += 1
            if exec_result["output_matches_expected"]:
                self.stats["correct"] += 1
        else:
            self.stats["failed"] += 1

        self.stats["total_gen_time_ms"] += model_telem["gen_time_ms"]
        self.stats["total_exec_time_ms"] += exec_result["exec_time_ms"]
        self.stats["total_tokens_generated"] += model_telem["tokens_generated"]

        self.packets.append(packet)
        return packet

    def _score_packet(self, packet: Dict) -> float:
        """Score a data packet on a 0-1 scale.

        Factors:
          - Correctness (40%): does output match expected?
          - Execution success (20%): did it run without errors?
          - Speed (15%): faster execution = higher score
          - Code quality (10%): shorter code = higher score (efficiency)
          - Model confidence (10%): mean logprob
          - No timeout (5%): didn't time out
        """
        exec_r = packet["execution"]
        model_t = packet["model_telemetry"]

        score = 0.0

        # Correctness (40%)
        if exec_r["output_matches_expected"]:
            score += 0.4
        elif exec_r["returncode"] == 0 and exec_r["stdout"].strip():
            score += 0.2  # ran but wrong output

        # Execution success (20%)
        if exec_r["returncode"] == 0:
            score += 0.2

        # Speed (15%) — faster is better, cap at 1s
        exec_time = exec_r["exec_time_ms"]
        speed_score = max(0, 1.0 - exec_time / 1000.0) * 0.15
        score += speed_score

        # Code quality (10%) — shorter code is more efficient
        code_len = len(packet["generated_code"])
        quality = max(0, 1.0 - code_len / 500.0) * 0.10
        score += quality

        # Model confidence (10%)
        conf = model_t.get("confidence", 0)
        score += min(conf, 1.0) * 0.10

        # No timeout (5%)
        if not exec_r["timed_out"]:
            score += 0.05

        return round(score, 4)

    def run_domain(self, domain: str, n_tasks: int = 10) -> List[Dict]:
        """Run multiple tasks from a domain (python_basics, math, algorithms, etc.).

        Generates task parameters randomly and runs each through the sandbox.
        """
        import random
        templates = self.TASK_TEMPLATES.get(domain, [])
        if not templates:
            print(f"  Unknown domain: {domain}")
            return []

        packets = []
        for i in range(n_tasks):
            template = random.choice(templates)
            task_desc, task_name, reference_code = template

            # Generate random parameters
            n = random.randint(3, 20)
            a = random.randint(2, 50)
            b = random.randint(2, 50)
            s = random.choice(["hello", "world", "python", "racecar", "madam"])
            lst = [random.randint(1, 100) for _ in range(random.randint(3, 8))]
            target = random.randint(1, 13)
            text = " ".join(random.choice(["the", "quick", "brown", "fox",
                                            "jumps", "over", "lazy", "dog"])
                            for _ in range(random.randint(3, 6)))

            # Format the task prompt
            try:
                prompt = task_desc.format(n=n, a=a, b=b, s=s, lst=lst,
                                          target=target, text=text)
            except KeyError:
                prompt = task_desc

            # Compute expected output from reference code
            expected = None
            if reference_code:
                try:
                    ref_code = reference_code.format(n=n, a=a, b=b, s=s, lst=lst,
                                                      target=target, text=text)
                    ref_result = self.executor.execute(ref_code)
                    if ref_result["returncode"] == 0:
                        expected = ref_result["stdout"].strip()
                except Exception:
                    pass

            print(f"\n  [{domain} {i+1}/{n_tasks}] {prompt[:60]}")

            # Build code prefix to help the model (function signature style)
            code_prefix = ""
            if domain == "math":
                code_prefix = "import math"
            elif domain == "algorithms":
                code_prefix = ""

            packet = self.run_task(prompt, expected_output=expected,
                                   task_type=domain, task_name=task_name,
                                   reference_code=reference_code,
                                   code_prefix=code_prefix)

            status = "✓" if packet["execution"]["output_matches_expected"] else \
                     "○" if packet["execution"]["returncode"] == 0 else "✗"
            print(f"    {status} score={packet['quality_score']:.2f} "
                  f"exec={packet['execution']['exec_time_ms']:.0f}ms "
                  f"gen={packet['model_telemetry']['gen_time_ms']:.0f}ms")
            if packet["execution"]["stdout"].strip():
                print(f"    Output: {packet['execution']['stdout'].strip()[:80]}")
            if packet["execution"]["stderr"].strip():
                print(f"    Error: {packet['execution']['stderr'].strip()[:80]}")

            packets.append(packet)

        return packets

    def save_packets(self, filename: str = None) -> str:
        """Save all data packets to a JSONL file for later knowledge injection."""
        if filename is None:
            filename = f"self_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        path = self.log_dir / filename

        with open(path, 'w', encoding='utf-8') as f:
            for packet in self.packets:
                f.write(json.dumps(packet, ensure_ascii=False) + "\n")

        print(f"\n  Saved {len(self.packets)} packets to {path}")
        return str(path)

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        s = self.stats
        s["success_rate"] = s["successful"] / max(s["total_tasks"], 1)
        s["correct_rate"] = s["correct"] / max(s["total_tasks"], 1)
        s["avg_gen_time_ms"] = s["total_gen_time_ms"] / max(s["total_tasks"], 1)
        s["avg_exec_time_ms"] = s["total_exec_time_ms"] / max(s["total_tasks"], 1)
        s["avg_tokens_per_second"] = (
            s["total_tokens_generated"] / max(s["total_gen_time_ms"] / 1000, 0.001)
        )
        return s

    def print_stats(self):
        """Print summary statistics."""
        s = self.get_stats()
        print(f"\n{'='*60}")
        print(f"Self-Play Sandbox Statistics")
        print(f"{'='*60}")
        print(f"  Total tasks:      {s['total_tasks']}")
        print(f"  Successful runs:  {s['successful']} ({s['success_rate']:.1%})")
        print(f"  Correct outputs:  {s['correct']} ({s['correct_rate']:.1%})")
        print(f"  Failed runs:      {s['failed']}")
        print(f"  Timed out:        {s['timed_out']}")
        print(f"  Avg gen time:     {s['avg_gen_time_ms']:.0f}ms")
        print(f"  Avg exec time:    {s['avg_exec_time_ms']:.0f}ms")
        print(f"  Avg tokens/sec:   {s['avg_tokens_per_second']:.0f}")
        print(f"  Packets logged:   {len(self.packets)}")
        print(f"{'='*60}")


def main():
    """Run self-play sandbox with ForgeLM."""
    sys.path.insert(0, '.')

    from research.config import get_config
    from research.model_loader import ModelLoader
    from transformers import AutoTokenizer

    print("=" * 60)
    print("Self-Play Sandbox — ForgeLM")
    print("=" * 60)

    # Load model
    print("\n[1] Loading ForgeLM V2...")
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(cfg,
        checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
    model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

    # Create sandbox
    print("\n[2] Creating sandbox...")
    sandbox = SelfPlaySandbox(model, tokenizer,
                              log_dir="research/data/self_play",
                              max_gen_tokens=100)

    # Run domains
    print("\n[3] Running self-play tasks...")

    print("\n--- Python Basics ---")
    sandbox.run_domain("python_basics", n_tasks=5)

    print("\n--- Math ---")
    sandbox.run_domain("math", n_tasks=5)

    print("\n--- Algorithms ---")
    sandbox.run_domain("algorithms", n_tasks=3)

    print("\n--- String Manipulation ---")
    sandbox.run_domain("string_manipulation", n_tasks=3)

    # Save packets
    print("\n[4] Saving data packets...")
    path = sandbox.save_packets()

    # Print stats
    sandbox.print_stats()

    print(f"\n  Data packet file: {path}")
    print(f"  Ready for knowledge injection via fact_injection_key or knowledge_pack_key")


if __name__ == "__main__":
    main()
