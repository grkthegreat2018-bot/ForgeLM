"""Thinking Recursive Self-Play — generate→think→test→fix→retry→learn.

Combines:
  1. ThinkingModel:  reasoning before answer
  2. AirMoEHotswapLoader: topic-based knowledge hotswap from disk
  3. RecursiveSelfPlay: fix-retry loop with reasoning tracking

Pipeline per task:
  1. Router classifies task → topic
  2. AirMoE loader hotswaps the relevant knowledge module
  3. ThinkingModel generates reasoning + code
  4. Sandbox executes the code
  5. If failed: error fed back → ThinkingModel generates fix reasoning + fixed code
  6. Repeat up to max_rounds
  7. Successful reasoning traces → knowledge packets
  8. All attempts logged with reasoning quality scores

Usage:
    from research.thinking_self_play import ThinkingSelfPlay
    engine = ThinkingSelfPlay(model, tokenizer, router, loader)
    engine.run_task("Check if 17 is prime")
"""
import os
import sys
import time
import json
import re
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from research.thinking_model import ThinkingModel
from research.recursive_self_play import DDriveSandboxExecutor
from research.self_play_sandbox import SelfPlaySandbox


class ThinkingSelfPlay(SelfPlaySandbox):
    """Recursive self-play with thinking model + AirMoE hotswap.

    The model thinks before answering, and the reasoning traces are
    captured as training data for continuous improvement.
    """

    def __init__(self, model, tokenizer, router=None, loader=None,
                 log_dir: str = "research/data/thinking_self_play",
                 device: str = "cuda",
                 max_think_tokens: int = 150,
                 max_answer_tokens: int = 150,
                 max_rounds: int = 5,
                 temp_dir: str = "D:/windsurf/ForgeAI/.devin/tmp"):
        super().__init__(model, tokenizer, log_dir, device, max_answer_tokens)

        self.thinker = ThinkingModel(
            model, tokenizer, device=device,
            max_think_tokens=max_think_tokens,
            max_answer_tokens=max_answer_tokens)

        self.router = router
        self.loader = loader
        self.max_rounds = max_rounds
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        self.executor = DDriveSandboxExecutor(temp_dir=temp_dir)

        # Stats
        self.thinking_history: List[Dict] = []
        self.topic_usage: Dict[str, int] = {}
        self.total_tasks = 0

    def run_task(self, task_prompt: str,
                 expected_output: Optional[str] = None,
                 task_type: str = "python",
                 task_name: str = "",
                 reference_code: Optional[str] = None,
                 code_prefix: str = "") -> Dict:
        """Run a single task with thinking + recursive fix-retry.

        Steps:
          1. Route task to topic → hotswap knowledge module
          2. Generate thinking + code
          3. Execute code
          4. If failed: feed error back, generate fix thinking + fixed code
          5. Repeat up to max_rounds
          6. Log reasoning traces
        """
        self.total_tasks += 1
        self.stats["total_tasks"] += 1

        # Route to topic and hotswap knowledge
        context = ""
        topic = "general"
        if self.router and self.loader:
            topic = self.router.classify(task_prompt)
            self.loader.load_topic(topic)
            context = self.loader.get_context_prefix()
            self.topic_usage[topic] = self.topic_usage.get(topic, 0) + 1

        attempts = []
        current_code = None
        current_error = None

        for round_num in range(self.max_rounds):
            # Generate with thinking
            if round_num == 0:
                # First attempt: generate from task
                result = self.thinker.generate_with_thinking(
                    task_prompt, context=context, code_prefix=code_prefix)
                code = result["code"]
                if code_prefix:
                    code = code_prefix + "\n" + code
            else:
                # Fix attempt: generate fix with thinking
                result = self.thinker.generate_fix_with_thinking(
                    task_prompt, current_code, current_error, round_num)
                code = result["code"]

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

            attempt = {
                "round": round_num,
                "topic": topic,
                "reasoning": result["reasoning"][:500],
                "code": code,
                "error": exec_result["stderr"] if exec_result["returncode"] != 0 else "",
                "stdout": exec_result["stdout"],
                "success": exec_result["returncode"] == 0,
                "correct": exec_result["output_matches_expected"],
                "exec_time_ms": exec_result["exec_time_ms"],
                "think_tokens": result["think_tokens"],
                "answer_tokens": result["answer_tokens"],
                "think_time_ms": result["think_time_ms"],
                "answer_time_ms": result["answer_time_ms"],
                "confidence": result["confidence"],
                "error_type": result.get("error_type", ""),
            }
            attempts.append(attempt)

            # Check success
            if exec_result["returncode"] == 0:
                self.stats["successful"] += 1
                if exec_result["output_matches_expected"]:
                    self.stats["correct"] += 1
                break

            # Failed — prepare for next round
            current_code = code
            current_error = exec_result["stderr"]

            status = "x" if round_num == 0 else f"x(fix{round_num})"
            err_short = current_error.strip()[:60]
            print(f"    {status} round={round_num} err={err_short}")
            print(f"      reasoning: {result['reasoning'][:80]}...")

        # Build packet
        final_success = attempts[-1]["success"] if attempts else False
        rounds = len(attempts)

        # Reasoning quality
        if final_success:
            quality = 1.0 - (rounds - 1) / self.max_rounds * 0.3
        else:
            quality = 0.1

        # Build knowledge text from successful attempt
        knowledge_text = ""
        if final_success:
            successful = attempts[-1]
            knowledge_text = (
                f"Task: {task_prompt}\n"
                f"Reasoning: {successful['reasoning'][:300]}\n"
                f"Solution (round {rounds}):\n{successful['code'][:300]}\n"
                f"Output: {successful['stdout'].strip()[:100]}\n"
            )

        packet = {
            "task": task_name or task_prompt[:50],
            "task_type": task_type,
            "topic": topic,
            "prompt": task_prompt,
            "attempts": attempts,
            "final_success": final_success,
            "rounds_used": rounds,
            "reasoning_quality": round(quality, 3),
            "knowledge_text": knowledge_text,
            "reference_code": reference_code,
            "expected_output": expected_output,
            "timestamp": datetime.now().isoformat(),
        }

        self.thinking_history.append({
            "task": task_name,
            "topic": topic,
            "quality": quality,
            "rounds": rounds,
            "success": final_success,
            "total_think_tokens": sum(a["think_tokens"] for a in attempts),
        })

        self.packets.append(packet)
        return packet

    def run_domain(self, domain: str, n_tasks: int = 5) -> List[Dict]:
        """Run multiple tasks from a domain."""
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

            # Compute expected output
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
            packet = self.run_task(
                prompt, expected_output=expected,
                task_type=domain, task_name=task_name,
                reference_code=reference_code, code_prefix=code_prefix)

            status = "OK" if packet["final_success"] else "FAIL"
            rounds = packet["rounds_used"]
            quality = packet["reasoning_quality"]
            think_tokens = sum(a["think_tokens"] for a in packet["attempts"])
            print(f"    {status} rounds={rounds} quality={quality:.2f} "
                  f"think_tokens={think_tokens}")

            if packet["final_success"] and packet["attempts"][-1]["stdout"].strip():
                print(f"    Output: {packet['attempts'][-1]['stdout'].strip()[:80]}")
            if packet["attempts"][-1]["reasoning"]:
                print(f"    Reasoning: {packet['attempts'][-1]['reasoning'][:80]}...")

            packets.append(packet)

        return packets

    def print_stats(self):
        """Print thinking self-play statistics."""
        s = self.get_stats()
        print(f"\n{'='*70}")
        print(f"Thinking Self-Play Statistics")
        print(f"{'='*70}")
        print(f"  Total tasks:       {self.total_tasks}")
        print(f"  Successful:        {s['successful']} ({s['success_rate']:.1%})")
        print(f"  Correct:           {s['correct']} ({s['correct_rate']:.1%})")
        print(f"  Failed:            {s['failed']}")
        print(f"  Packets logged:    {len(self.packets)}")

        # Thinking stats
        if self.thinking_history:
            total_think = sum(t["total_think_tokens"] for t in self.thinking_history)
            avg_think = total_think / len(self.thinking_history)
            avg_quality = sum(t["quality"] for t in self.thinking_history) / len(self.thinking_history)
            print(f"\n  Thinking stats:")
            print(f"    Total think tokens: {total_think}")
            print(f"    Avg think tokens/task: {avg_think:.1f}")
            print(f"    Avg reasoning quality: {avg_quality:.3f}")

        # Topic usage
        if self.topic_usage:
            print(f"\n  Topic usage (AirMoE hotswap):")
            for topic, count in sorted(self.topic_usage.items(),
                                        key=lambda x: -x[1]):
                print(f"    {topic}: {count} tasks")

        # AirMoE stats
        if self.loader:
            self.loader.print_stats()

        # Thinking model stats
        self.thinker.print_reasoning_stats()

        print(f"{'='*70}")


def main():
    """Run thinking self-play with ForgeLM + AirMoE hotswap."""
    sys.path.insert(0, '.')

    from research.config import get_config
    from research.model_loader import ModelLoader
    from transformers import AutoTokenizer
    from research.airmoe_hotswap import TopicRouter, AirMoEHotswapLoader

    print("=" * 70)
    print("Thinking Self-Play Engine — ForgeLM V2 + AirMoE")
    print("=" * 70)

    # Load model
    print("\n[1] Loading ForgeLM V2...")
    cfg = get_config("forgelm_v2", device="cuda")
    model = ModelLoader.build_model_fast(cfg,
        checkpoint_path="research/checkpoints/forgelm_v2.safetensors")
    model.to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained("research/checkpoints/qwen_hf")

    # Create router and loader
    print("\n[2] Setting up AirMoE hotswap...")
    index_path = "D:/windsurf/ForgeAI/research/checkpoints/airmoe_modules/index.json"

    router = None
    loader = None
    if os.path.exists(index_path):
        router = TopicRouter(index_path)
        loader = AirMoEHotswapLoader(model, tokenizer, device="cuda",
                                      cache_size=3, injection_method="context")
        print(f"  Available topics: {router.list_topics()}")
    else:
        print(f"  WARNING: No AirMoE modules found at {index_path}")
        print(f"  Run: python -m research.training_packs_airmoe")

    # Create thinking self-play engine
    print("\n[3] Creating thinking self-play engine...")
    engine = ThinkingSelfPlay(
        model, tokenizer, router=router, loader=loader,
        log_dir="research/data/thinking_self_play",
        max_think_tokens=120,
        max_answer_tokens=120,
        max_rounds=5,
        temp_dir="D:/windsurf/ForgeAI/.devin/tmp")

    # Run tasks
    print("\n[4] Running thinking self-play tasks...")

    print("\n--- Python Basics (thinking) ---")
    engine.run_domain("python_basics", n_tasks=5)

    print("\n--- Math (thinking) ---")
    engine.run_domain("math", n_tasks=5)

    print("\n--- Algorithms (thinking) ---")
    engine.run_domain("algorithms", n_tasks=3)

    print("\n--- String Manipulation (thinking) ---")
    engine.run_domain("string_manipulation", n_tasks=3)

    # Save packets
    print("\n[5] Saving data packets...")
    path = engine.save_packets()

    # Print stats
    engine.print_stats()

    print(f"\n  Data packet file: {path}")
    print(f"  Ready for knowledge injection via fact_injection_key")


if __name__ == "__main__":
    main()
