"""Recursive Self-Play Engine — generate → test → fix → retry → learn.

The model generates code, the sandbox executes it, and if it fails:
  1. The error is fed back to the model
  2. The model generates a fix
  3. Repeat up to N retries
  4. Successful fixes → knowledge packets (reasoning traces)
  5. All attempts logged with quality scores

This creates a self-improving loop:
  - Model learns from its mistakes (error → fix traces)
  - Successful reasoning paths become training data
  - Failed paths teach what NOT to do
  - Quality score tracks improvement over time

Data packet format (one per recursive attempt):
  {
    "task": "...",
    "attempts": [
      {"code": "...", "error": "...", "success": false, "round": 0},
      {"code": "...", "error": "...", "success": true, "round": 1,
       "fix_reasoning": "The error was IndentationError, so I..."}
    ],
    "final_success": true,
    "rounds_used": 2,
    "reasoning_quality": 0.85,
    "knowledge_text": "To check if a number is prime: ...",
    "learnings": ["Always use print() to output result", "Indentation matters"]
  }

Usage:
    from research.recursive_self_play import RecursiveSelfPlay
    engine = RecursiveSelfPlay(model, tokenizer, log_dir="research/data/recursive_self_play")
    engine.run_recursive_task("Check if 17 is prime", max_rounds=5)
"""
import os
import sys
import time
import json
import re
import torch
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.self_play_sandbox import SandboxExecutor, SelfPlaySandbox
from research.goal_tasks import GoalTask, build_goal_prompt
from research.goal_scorer import GoalScorer, ScoreResult


class RecursiveSelfPlay(SelfPlaySandbox):
    """Recursive self-play with fix-retry loop and reasoning tracking.

    Extends SelfPlaySandbox with:
      - Error feedback → model generates fix
      - Multi-round retry (up to N rounds)
      - Reasoning trace extraction
      - Quality improvement tracking
      - Learnings extraction (what worked vs what didn't)
    """

    def __init__(self, model, tokenizer, log_dir: str = "research/data/recursive_self_play",
                 device: str = "cuda", max_gen_tokens: int = 150,
                 max_rounds: int = 5, temp_dir: str = "D:/windsurf/ForgeAI/.devin/tmp",
                 temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
                 vram_manager=None):
        super().__init__(model, tokenizer, log_dir, device, max_gen_tokens)
        self.max_rounds = max_rounds
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        # Sampling params: temperature=0 (greedy) for reproducibility,
        # temperature>0 for diverse solutions across epochs (anti-overfitting)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # VRAM manager for dynamic memory monitoring during generation
        self.vram = vram_manager

        # Override executor to use D: temp dir
        self.executor = DDriveSandboxExecutor(temp_dir=self.temp_dir)

        # Track reasoning improvement
        self.reasoning_history: List[Dict] = []
        self.learnings: List[str] = []
        self.success_by_round = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.total_recursive_tasks = 0

    def _sample_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token with temperature/top-k/top-p, or greedy if temp=0.

        Args:
            logits: (vocab,) logits for the next token
        Returns:
            scalar tensor with the selected token id
        """
        if self.temperature <= 0.0:
            return logits.argmax()

        # Temperature scaling
        scaled = logits / self.temperature
        probs = torch.softmax(scaled, dim=-1)

        # Top-k filtering
        if self.top_k > 0 and self.top_k < probs.shape[0]:
            topk_vals, _ = torch.topk(probs, self.top_k)
            mask = probs < topk_vals[-1]
            probs = probs.masked_fill(mask, 0.0)
            probs = probs / probs.sum()

        # Top-p (nucleus) filtering
        if 0.0 < self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum > self.top_p
            # Keep at least one token
            cutoff[0] = False
            sorted_probs[cutoff] = 0.0
            probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
            probs = probs / probs.sum()

        return torch.multinomial(probs, num_samples=1).squeeze()

    def generate_fix(self, original_code: str, error: str,
                      task_prompt: str, round_num: int,
                      prev_compute: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Generate a fix for broken code using error feedback.

        Args:
            original_code: the code that failed
            error: the error message from execution
            task_prompt: the original task description
            round_num: current retry round (for context)
            prev_compute: compute metrics from previous attempt
                (gen_tokens, gen_ms, exec_ms, code_len)

        Returns:
            (fixed_code, model_telemetry)
        """
        # Build fix prompt: show the error and ask for correction
        # Truncate error to keep prompt manageable
        error_short = error.strip()[:500]
        code_short = original_code.strip()[:800]

        # Include compute metrics to encourage efficiency
        compute_info = ""
        if prev_compute:
            compute_info = (
                f"# Previous compute: {prev_compute.get('gen_tokens', 0)} tokens, "
                f"{prev_compute.get('gen_ms', 0):.0f}ms gen, "
                f"{prev_compute.get('exec_ms', 0):.0f}ms exec, "
                f"{prev_compute.get('code_len', 0)} chars\n"
                f"# Aim for concise, efficient code.\n"
            )

        fix_prompt = (
            f"# Task: {task_prompt}\n"
            f"{compute_info}"
            f"# Previous attempt (BROKEN):\n"
            f"{code_short}\n"
            f"# Error:\n"
            f"{error_short}\n"
            f"# Fixed solution (keep it concise):\n"
        )

        input_ids = self.tokenizer(fix_prompt, return_tensors="pt").input_ids.to(self.device)

        t0 = time.time()
        log_probs = []

        newline_token = self.tokenizer.encode("\n", add_special_tokens=False)
        newline_id = newline_token[0] if newline_token else 198

        with torch.inference_mode():
            cur_ids = input_ids
            past_kv = None
            newline_count = 0
            for step in range(self.max_gen_tokens):
                # VRAM check: abort generation if approaching OOM
                if self.vram and step > 0 and step % 32 == 0:
                    if not self.vram.check_during_generation(
                            step, self.max_gen_tokens, label="gen_loop"):
                        break

                # Use KV cache: first step processes full prompt, subsequent
                # steps only process the new token (O(1) not O(n) per step)
                if past_kv is not None:
                    logits, _, past_kv = self.model(
                        cur_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                else:
                    logits, _, past_kv = self.model(
                        cur_ids, use_cache=True)
                next_logits = logits[0, -1]
                next_token = self._sample_next_token(next_logits)

                lp = torch.log_softmax(next_logits, dim=-1)[next_token].item()
                log_probs.append(lp)

                cur_ids = torch.cat([cur_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                if next_token.item() == newline_id:
                    newline_count += 1
                    if newline_count >= 3 and step > 5:
                        break
                else:
                    newline_count = 0

        gen_time_ms = (time.time() - t0) * 1000
        tokens_generated = cur_ids.shape[1] - input_ids.shape[1]

        generated_text = self.tokenizer.decode(cur_ids[0, input_ids.shape[1]:],
                                                skip_special_tokens=True)

        # Free generation tensors and KV cache to keep VRAM tight
        del cur_ids, logits, past_kv
        torch.cuda.empty_cache()

        code = generated_text.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines)

        telemetry = {
            "gen_time_ms": gen_time_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / (gen_time_ms / 1000) if gen_time_ms > 0 else 0,
            "mean_logprob": sum(log_probs) / max(len(log_probs), 1),
            "confidence": pow(2.71828, sum(log_probs) / max(len(log_probs), 1)),
            "round": round_num,
        }

        return code, telemetry

    def extract_reasoning(self, attempts: List[Dict]) -> Dict:
        """Extract reasoning quality metrics from the attempt chain.

        Analyzes:
          - Did the model identify the error type?
          - Did the fix address the root cause?
          - How many rounds did it take?
          - What patterns did it learn?
        """
        if not attempts:
            return {"quality": 0, "learnings": []}

        final_success = attempts[-1]["success"]
        rounds = len(attempts)

        # Extract error types
        error_types = []
        for a in attempts:
            if a.get("error"):
                err = a["error"].lower()
                if "syntaxerror" in err or "syntax" in err:
                    error_types.append("syntax")
                elif "indentationerror" in err:
                    error_types.append("indentation")
                elif "nameerror" in err:
                    error_types.append("name")
                elif "typeerror" in err:
                    error_types.append("type")
                elif "valueerror" in err:
                    error_types.append("value")
                elif "indexerror" in err:
                    error_types.append("index")
                elif "attributeerror" in err:
                    error_types.append("attribute")
                elif "importerror" in err:
                    error_types.append("import")
                else:
                    error_types.append("other")

        # Reasoning quality: higher if fewer rounds, successful, diverse fixes
        if final_success:
            quality = 1.0 - (rounds - 1) / self.max_rounds * 0.3  # penalty per round
        else:
            quality = 0.1  # failed all rounds

        # Extract learnings
        learnings = []
        if final_success and rounds > 1:
            learnings.append(f"Fixed {error_types[0]} error in {rounds} rounds")
            if error_types[0] == "syntax":
                learnings.append("Check syntax before running")
            elif error_types[0] == "indentation":
                learnings.append("Use consistent indentation (4 spaces)")
            elif error_types[0] == "name":
                learnings.append("Define variables before using them")
            elif error_types[0] == "type":
                learnings.append("Check type compatibility in operations")

        return {
            "quality": round(quality, 3),
            "rounds": rounds,
            "error_types": error_types,
            "final_success": final_success,
            "learnings": learnings,
        }

    def run_recursive_task(self, task_prompt: str,
                            expected_output: Optional[str] = None,
                            task_type: str = "python",
                            task_name: str = "",
                            reference_code: Optional[str] = None,
                            code_prefix: str = "",
                            test_cases: Optional[List[Dict]] = None,
                            use_reasoning: bool = True) -> Dict:
        """Run a task with recursive fix-retry loop.

        The model generates code, executes it, and if it fails:
          1. Error is fed back
          2. Model generates a fix
          3. Repeat up to max_rounds
          4. All attempts logged

        Args:
            test_cases: list of {"args": (...), "expected": ...} for output verification.
                        If provided, code must produce correct output, not just run.
            use_reasoning: if True, generate chain-of-thought before code on round 0.

        Returns:
            Complete data packet with all attempts and reasoning analysis
        """
        self.total_recursive_tasks += 1
        self.stats["total_tasks"] += 1

        attempts = []
        current_code = None
        current_error = None
        reasoning_text = ""

        for round_num in range(self.max_rounds):
            # Generate code (first round: from task, later rounds: fix from error)
            if round_num == 0:
                gen_prompt = task_prompt
                if code_prefix:
                    gen_prompt = f"{code_prefix}\n# {task_prompt}"

                # Chain-of-thought: generate reasoning before code
                if use_reasoning and any(kw in gen_prompt for kw in ["def ", "class "]):
                    reasoning_text, completion, model_telem = self.generate_with_reasoning(gen_prompt)
                else:
                    reasoning_text = ""
                    completion, model_telem = self.generate_code(gen_prompt)

                # Full code = prompt + generated completion
                code = gen_prompt + completion
                if code_prefix:
                    code = code_prefix + "\n" + code
            else:
                # Pass compute metrics from previous attempt
                prev_att = attempts[-1] if attempts else None
                prev_compute = None
                if prev_att:
                    prev_compute = {
                        "gen_tokens": prev_att.get("tokens_generated", 0),
                        "gen_ms": prev_att.get("gen_time_ms", 0),
                        "exec_ms": prev_att.get("exec_time_ms", 0),
                        "code_len": len(prev_att.get("code", "")),
                    }
                code, model_telem = self.generate_fix(
                    current_code, current_error, task_prompt, round_num,
                    prev_compute=prev_compute)

            # Auto-append print() if needed
            if "def " in code and "print(" not in code and expected_output is not None:
                func_match = re.search(r'def\s+(\w+)\s*\(([^)]*)\)', code)
                if func_match:
                    func_name = func_match.group(1)
                    args = func_match.group(2)
                    numbers = re.findall(r'\d+', task_prompt)
                    if numbers:
                        n_args = len([a for a in args.split(',') if a.strip()]) if args else 0
                        call_args = ", ".join(numbers[:max(n_args, 1)])
                        code += f"\nprint({func_name}({call_args}))"

            # Execute
            exec_result = self.executor.execute(code, expected_output)

            # ── Test-driven verification ──────────────────────────────
            # test_passed: True=tests ran and passed, False=tests ran and failed,
            #              None=no tests available (UNVERIFIED — not a success)
            test_passed = None
            test_error = ""
            if test_cases and exec_result["returncode"] == 0:
                test_passed, test_error = self._verify_tests(code, test_cases)

            # Success requires POSITIVE verification:
            #   - tests existed AND passed, OR
            #   - expected_output was provided AND matched
            # Code that merely runs without errors is NOT a success (prevents false answers)
            output_verified = (expected_output is not None
                               and exec_result["output_matches_expected"])
            test_verified = (test_passed is True)
            is_success = (exec_result["returncode"] == 0
                          and (test_verified or output_verified))

            # Per-process metrics from the child that ran the answer script
            attempt = {
                "round": round_num,
                "code": code,
                "error": exec_result["stderr"] if exec_result["returncode"] != 0 else "",
                "stdout": exec_result["stdout"],
                "success": is_success,
                "correct": output_verified and (test_passed is not False),
                "test_passed": test_passed,  # True/False/None
                "verified": test_verified or output_verified,  # explicit verification flag
                "test_error": test_error,
                "exec_time_ms": exec_result["exec_time_ms"],
                "gen_time_ms": model_telem["gen_time_ms"],
                "tokens_generated": model_telem.get("tokens_generated", 0),
                "code_length": len(code),
                "code_lines": code.count("\n") + 1,
                "confidence": model_telem["confidence"],
                "reasoning": reasoning_text if round_num == 0 else "",
            }
            attempts.append(attempt)

            # Check if we succeeded (requires positive verification)
            if is_success:
                self.success_by_round[round_num] = self.success_by_round.get(round_num, 0) + 1
                if exec_result["output_matches_expected"]:
                    self.stats["correct"] += 1
                self.stats["successful"] += 1
                current_code = code
                current_error = None
                break

            # Failed (or unverified) — prepare for next round
            current_code = code
            # Use test error if tests failed, otherwise use exec error
            # If code ran but was UNVERIFIED (no tests, no expected_output),
            # treat as failure with a clear message
            if test_passed is False:
                current_error = f"Test failure: {test_error}"
            elif exec_result["returncode"] != 0:
                current_error = exec_result["stderr"]
            else:
                # Code ran but no verification available — cannot trust output
                current_error = ("Code ran but no test cases or expected_output "
                                 "provided - cannot verify correctness")

            status = "FAIL" if round_num == 0 else f"FAIL(fix{round_num})"
            err_short = current_error.strip()[:60].encode('ascii', 'replace').decode('ascii')
            print(f"    {status} round={round_num} err={err_short}")

        # Build final packet
        final_success = attempts[-1]["success"] if attempts else False
        reasoning = self.extract_reasoning(attempts)

        # Build knowledge text from successful attempt
        knowledge_text = ""
        if final_success:
            successful_attempt = attempts[-1]
            knowledge_text = (
                f"Task: {task_prompt}\n"
                f"Solution (round {len(attempts)}):\n{successful_attempt['code'][:300]}\n"
                f"Output: {successful_attempt['stdout'].strip()[:100]}\n"
                f"Compute: {successful_attempt.get('tokens_generated', 0)} tokens, "
                f"{successful_attempt.get('gen_time_ms', 0):.0f}ms gen, "
                f"{successful_attempt.get('exec_time_ms', 0):.0f}ms exec, "
                f"{successful_attempt.get('code_lines', 0)} lines, "
                f"efficiency={successful_attempt.get('efficiency', 0):.2f}\n"
            )
            if reasoning_text:
                knowledge_text += f"Reasoning: {reasoning_text[:200]}\n"
            if len(attempts) > 1:
                knowledge_text += (
                    f"Fixed {reasoning['error_types'][0]} error "
                    f"after {len(attempts)} attempts.\n"
                )
                for learning in reasoning["learnings"]:
                    knowledge_text += f"Learning: {learning}\n"

        packet = {
            "task": task_name or task_prompt[:50],
            "task_type": task_type,
            "prompt": task_prompt,
            "attempts": attempts,
            "final_success": final_success,
            "rounds_used": len(attempts),
            "reasoning_quality": reasoning["quality"],
            "error_types": reasoning["error_types"],
            "learnings": reasoning["learnings"],
            "knowledge_text": knowledge_text,
            "reasoning_text": reasoning_text,
            "reference_code": reference_code,
            "expected_output": expected_output,
            "test_cases": test_cases,
            "timestamp": datetime.now().isoformat(),
        }

        # Update learnings
        for learning in reasoning["learnings"]:
            if learning not in self.learnings:
                self.learnings.append(learning)

        self.reasoning_history.append({
            "task": task_name,
            "quality": reasoning["quality"],
            "rounds": len(attempts),
            "success": final_success,
        })

        self.packets.append(packet)
        return packet

    @staticmethod
    def _verify_tests(code: str, test_cases: List[Dict]) -> Tuple[bool, str]:
        """Run test cases against generated code.

        Args:
            code: the generated Python code (must define the function)
            test_cases: list of {"args": (...), "expected": ...}

        Returns:
            (all_passed, error_message) where all_passed is True/False/None.
            None means tests could not be run (no registry) — NOT a pass.
        """
        try:
            from research.prompt_tests import run_tests
            # Extract function name from code
            import re
            match = re.search(r'def\s+(\w+)\s*\(', code)
            if not match:
                return False, "No function definition found in code"
            func_name = match.group(1)
            return run_tests(code, func_name)
        except ImportError:
            # No test registry — return None so caller knows tests were NOT run.
            # This prevents unverified code from being treated as successful.
            return None, "No test registry available - cannot verify"
        except Exception as e:
            return False, f"Test execution error: {e}"

    def run_recursive_domain(self, domain: str, n_tasks: int = 5) -> List[Dict]:
        """Run multiple tasks from a domain with recursive fix-retry."""
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

            try:
                prompt = task_desc.format(n=n, a=a, b=b, s=s, lst=lst,
                                          target=target, text=text)
            except KeyError:
                prompt = task_desc

            # Compute expected output from reference
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

            code_prefix = "import math" if domain == "math" else ""

            print(f"\n  [{domain} {i+1}/{n_tasks}] {prompt[:60]}")
            packet = self.run_recursive_task(
                prompt, expected_output=expected,
                task_type=domain, task_name=task_name,
                reference_code=reference_code, code_prefix=code_prefix)

            status = "OK" if packet["final_success"] else "FAIL"
            rounds = packet["rounds_used"]
            quality = packet["reasoning_quality"]
            print(f"    {status} rounds={rounds} quality={quality:.2f} "
                  f"attempts={len(packet['attempts'])}")

            if packet["final_success"] and packet["attempts"][-1]["stdout"].strip():
                print(f"    Output: {packet['attempts'][-1]['stdout'].strip()[:80]}")

            packets.append(packet)

        return packets

    def print_recursive_stats(self):
        """Print recursive self-play statistics."""
        s = self.get_stats()
        print(f"\n{'='*70}")
        print(f"Recursive Self-Play Statistics")
        print(f"{'='*70}")
        print(f"  Total tasks:       {self.total_recursive_tasks}")
        print(f"  Successful:        {s['successful']} ({s['success_rate']:.1%})")
        print(f"  Correct:           {s['correct']} ({s['correct_rate']:.1%})")
        print(f"  Failed:            {s['failed']}")
        print(f"  Timed out:         {s['timed_out']}")
        print(f"  Avg gen time:      {s['avg_gen_time_ms']:.0f}ms")
        print(f"  Avg exec time:     {s['avg_exec_time_ms']:.0f}ms")
        print(f"  Avg tokens/sec:    {s['avg_tokens_per_second']:.0f}")
        print(f"  Packets logged:    {len(self.packets)}")
        print(f"\n  Success by round:")
        for r, count in sorted(self.success_by_round.items()):
            pct = count / max(self.total_recursive_tasks, 1) * 100
            print(f"    Round {r}: {count} ({pct:.1f}%)")
        print(f"\n  Learnings extracted: {len(self.learnings)}")
        for learning in self.learnings[:10]:
            print(f"    - {learning}")

        # Reasoning quality trend
        if self.reasoning_history:
            recent = self.reasoning_history[-10:]
            avg_quality = sum(r["quality"] for r in recent) / len(recent)
            print(f"\n  Recent reasoning quality: {avg_quality:.3f}")
            print(f"  Total reasoning traces: {len(self.reasoning_history)}")

        print(f"{'='*70}")


    # ─── Goal-Oriented Self-Play (GOSP) ─────────────────────────────
    # These methods implement the goal-oriented approach where the model
    # is given a GOAL (target output for given inputs) and must define
    # solve() any way it chooses. Verification is via I/O pairs, not
    # implementation matching. Scoring is multi-dimensional (minimalism,
    # efficiency, diversity, consistency, confidence).

    def __init_goal_scorer(self):
        """Lazily initialize the GoalScorer (called on first goal task)."""
        if not hasattr(self, '_goal_scorer'):
            self._goal_scorer = GoalScorer()

    def _verify_goal_io(self, code: str, test_cases: List[Dict],
                        solve_name: str = "solve",
                        stress_index: Optional[int] = None) -> Tuple[bool, float, str, List[Dict]]:
        """Verify generated code against I/O test cases.

        Executes the code to define solve(), then calls solve(*args) for each
        test case and compares the return value to expected.

        Args:
            code: Python code that must define solve()
            test_cases: list of {"args": (...), "expected": ...}
            solve_name: name of the function to call
            stress_index: which test case to time for efficiency scoring

        Returns:
            (all_passed, stress_exec_ms, error_message, per_test_results)
        """
        import json as _json

        # Build verification wrapper
        test_data = _json.dumps([
            {"args": list(tc["args"]), "expected": repr(tc["expected"])}
            for tc in test_cases
        ])

        wrapper_code = f'''
import json, time, sys

# User code defines solve()
{code}

# Run tests
_test_data = json.loads({test_data!r})
_results = []
_stress_ms = 0.0
_all_pass = True

for i, tc in enumerate(_test_data):
    try:
        args = tc["args"]
        if {stress_index!r} is not None and i == {stress_index}:
            t0 = time.perf_counter()
            actual = {solve_name}(*args)
            _stress_ms = (time.perf_counter() - t0) * 1000
        else:
            actual = {solve_name}(*args)
        expected = eval(tc["expected"])
        match = (actual == expected)
        if not match:
            _all_pass = False
        _results.append({{"i": i, "pass": match, "actual": repr(actual), "expected": repr(expected)}})
    except Exception as e:
        _all_pass = False
        _results.append({{"i": i, "pass": False, "error": str(e)[:200]}})

print(json.dumps({{"all_pass": _all_pass, "stress_ms": _stress_ms, "results": _results}}))
'''

        # Execute in sandbox
        exec_result = self.executor.execute(wrapper_code)

        if exec_result["returncode"] != 0:
            err = exec_result["stderr"].strip()[:300]
            return False, 0.0, f"Execution error: {err}", []

        try:
            output = _json.loads(exec_result["stdout"].strip().split("\n")[-1])
            all_pass = output["all_pass"]
            stress_ms = output.get("stress_ms", 0.0)
            results = output.get("results", [])
            error = "" if all_pass else "; ".join(
                f"test {r['i']}: got {r.get('actual', '?')} expected {r.get('expected', '?')}"
                for r in results if not r.get("pass", False)
            )[:300]
            return all_pass, stress_ms, error, results
        except (_json.JSONDecodeError, IndexError, KeyError) as e:
            return False, 0.0, f"Verification parse error: {e}", []

    def run_goal_task(self, goal: GoalTask, k_samples: int = 1,
                      use_reasoning: bool = True) -> Dict:
        """Run a goal-oriented task with recursive fix-retry and multi-dim scoring.

        The model is given a GOAL (target output for inputs) and must define
        solve() any way it chooses. Verification is via I/O pairs.

        Args:
            goal: GoalTask with description, test_cases, stress_index
            k_samples: number of independent samples for self-consistency (VERSE)
            use_reasoning: generate chain-of-thought before code on round 0

        Returns:
            Complete data packet with attempts, scores, and fingerprint
        """
        self.__init_goal_scorer()
        self.total_recursive_tasks += 1
        self.stats["total_tasks"] += 1

        goal_prompt = build_goal_prompt(goal)
        attempts = []
        current_code = None
        current_error = None
        reasoning_text = ""
        sample_outputs = []  # for self-consistency

        for sample_idx in range(k_samples):
            sample_attempts = []
            sample_code = None
            sample_error = None
            sample_output = None

            for round_num in range(self.max_rounds):
                # Generate code
                if round_num == 0:
                    gen_prompt = goal_prompt
                    if use_reasoning and "def " in gen_prompt:
                        reasoning_text, completion, model_telem = self.generate_with_reasoning(gen_prompt)
                    else:
                        reasoning_text = ""
                        completion, model_telem = self.generate_code(gen_prompt)
                    code = gen_prompt + completion
                else:
                    prev_att = sample_attempts[-1] if sample_attempts else None
                    prev_compute = None
                    if prev_att:
                        prev_compute = {
                            "gen_tokens": prev_att.get("tokens_generated", 0),
                            "gen_ms": prev_att.get("gen_time_ms", 0),
                            "exec_ms": prev_att.get("exec_time_ms", 0),
                            "code_len": len(prev_att.get("code", "")),
                        }
                    code, model_telem = self.generate_fix(
                        sample_code, sample_error, goal.description, round_num,
                        prev_compute=prev_compute)

                # Verify I/O
                all_pass, stress_ms, verify_error, test_results = self._verify_goal_io(
                    code, goal.test_cases, goal.solve_name, goal.stress_index)

                # Score with GoalScorer
                score_result = self._goal_scorer.score(
                    code=code,
                    correct=all_pass,
                    exec_time_ms=stress_ms if stress_ms > 0 else model_telem.get("gen_time_ms", 0),
                    stress_exec_ms=stress_ms if stress_ms > 0 else None,
                    mean_logprob=model_telem.get("mean_logprob", -1.0),
                    goal_id=goal.id,
                    tokens_generated=model_telem.get("tokens_generated", 0),
                    k_samples=k_samples,
                    n_agreeing=1,  # updated after all samples
                )

                attempt = {
                    "sample": sample_idx,
                    "round": round_num,
                    "code": code,
                    "error": verify_error if not all_pass else "",
                    "success": all_pass,
                    "accepted": score_result.accepted,
                    "quality": score_result.quality,
                    "scores": score_result.scores,
                    "fingerprint": score_result.fingerprint,
                    "ast_nodes": score_result.ast_node_count,
                    "exec_time_ms": stress_ms,
                    "gen_time_ms": model_telem.get("gen_time_ms", 0),
                    "tokens_generated": model_telem.get("tokens_generated", 0),
                    "confidence": model_telem.get("confidence", 0),
                    "reasoning": reasoning_text if round_num == 0 else "",
                    "test_results": test_results,
                    "rejected_reason": score_result.rejected_reason,
                }
                sample_attempts.append(attempt)
                attempts.append(attempt)

                if all_pass:
                    # Record fingerprint for diversity tracking
                    if score_result.accepted:
                        self._goal_scorer.record_fingerprint(goal.id, score_result.fingerprint)
                    sample_output = test_results
                    break

                # Prepare for fix round
                sample_code = code
                sample_error = verify_error or "I/O verification failed"

            if sample_output is not None:
                sample_outputs.append(sample_output)

        # Self-consistency: count how many samples agreed on output
        n_agreeing = len(sample_outputs) if k_samples > 1 else 1

        # Find best accepted attempt
        best_attempt = None
        best_quality = -1
        for att in attempts:
            if att["accepted"] and att["quality"] > best_quality:
                best_attempt = att
                best_quality = att["quality"]

        final_success = best_attempt is not None
        if final_success:
            self.success_by_round[best_attempt["round"]] = \
                self.success_by_round.get(best_attempt["round"], 0) + 1
            self.stats["successful"] += 1
            self.stats["correct"] += 1

        # Build packet
        packet = {
            "goal_id": goal.id,
            "goal": goal.description,
            "domain": goal.domain,
            "difficulty": goal.difficulty,
            "archetype": goal.archetype,
            "input_signature": goal.input_signature,
            "test_cases": goal.test_cases,
            "attempts": attempts,
            "final_success": final_success,
            "best_quality": best_quality if final_success else 0.0,
            "rounds_used": len([a for a in attempts if a["sample"] == 0]) if k_samples == 1 else len(attempts),
            "k_samples": k_samples,
            "n_agreeing": n_agreeing,
            "reasoning_text": reasoning_text,
            "timestamp": datetime.now().isoformat(),
        }
        self.packets.append(packet)
        return packet


class DDriveSandboxExecutor(SandboxExecutor):
    """SandboxExecutor that uses D: drive for temp files."""

    def __init__(self, timeout_s: float = 5.0, memory_limit_mb: int = 256,
                 temp_dir: str = "D:/windsurf/ForgeAI/.devin/tmp"):
        super().__init__(timeout_s, memory_limit_mb)
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

    def execute(self, code: str, expected_output: Optional[str] = None) -> Dict:
        """Execute Python code using D: drive temp files."""
        # Write code to D: temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False, encoding='utf-8',
                                          dir=self.temp_dir) as f:
            f.write(code)
            temp_path = f.name

        file_size = os.path.getsize(temp_path)

        # Build wrapper (same as parent but with D: path)
        try:
            import resource
            mem_code = f"resource.setrlimit(resource.RLIMIT_AS, ({self.memory_limit_mb * 1024 * 1024}, {self.memory_limit_mb * 1024 * 1024}))"
        except ImportError:
            mem_code = "# resource not available on Windows"

        wrapper = f'''
import sys, os
{f"import resource" if sys.platform != "win32" else ""}
{mem_code}

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
            proc = subprocess.run(
                [sys.executable, wrapper_path],
                capture_output=True, text=True,
                timeout=self.timeout_s,
                cwd=self.temp_dir,
                encoding='utf-8', errors='replace',
            )
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            result["returncode"] = proc.returncode
            result["exec_time_ms"] = (time.time() - t0) * 1000
            result["peak_memory_kb"] = (len(proc.stdout) + len(proc.stderr) +
                                         file_size) // 1024 + 12000

            if expected_output is not None:
                result["output_matches_expected"] = (
                    expected_output.strip() == result["stdout"].strip())

        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            result["stderr"] = f"Execution timed out after {self.timeout_s}s"
            result["exec_time_ms"] = self.timeout_s * 1000
        except Exception as e:
            result["stderr"] = f"Sandbox error: {e}"
            result["exec_time_ms"] = (time.time() - t0) * 1000
        finally:
            try:
                os.unlink(temp_path)
                os.unlink(wrapper_path)
            except OSError:
                pass

        return result


def main():
    """Run recursive self-play with ForgeLM."""
    sys.path.insert(0, '.')

    from research.config import get_config
    from research.model_loader import ModelLoader
    from transformers import AutoTokenizer

    print("=" * 70)
    print("Recursive Self-Play Engine - ForgeLM V2")
    print("=" * 70)

    # Load model
    print("\n[1] Loading ForgeLM V2...")
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(cfg,
        checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
    model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

    # Create recursive engine
    print("\n[2] Creating recursive self-play engine...")
    engine = RecursiveSelfPlay(model, tokenizer,
                               log_dir="research/data/recursive_self_play",
                               max_gen_tokens=100,
                               max_rounds=5,
                               temp_dir="D:/windsurf/ForgeAI/.devin/tmp")

    # Run domains with recursive fix-retry
    print("\n[3] Running recursive self-play tasks...")

    print("\n--- Python Basics (recursive) ---")
    engine.run_recursive_domain("python_basics", n_tasks=5)

    print("\n--- Math (recursive) ---")
    engine.run_recursive_domain("math", n_tasks=5)

    print("\n--- Algorithms (recursive) ---")
    engine.run_recursive_domain("algorithms", n_tasks=3)

    print("\n--- String Manipulation (recursive) ---")
    engine.run_recursive_domain("string_manipulation", n_tasks=3)

    # Save packets
    print("\n[4] Saving data packets...")
    path = engine.save_packets()

    # Print stats
    engine.print_recursive_stats()

    print(f"\n  Data packet file: {path}")
    print(f"  Ready for knowledge injection via fact_injection_key or knowledge_pack_key")


if __name__ == "__main__":
    main()
