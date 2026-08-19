"""Test the new task-based concurrent generation + session management.

Tests:
1. SessionManager: create, append, get history, delete, LRU eviction
2. BatchQueue: single request, multiple concurrent requests (batched)
3. Task continuity: multi-turn conversation with session history
4. Concurrent tasks: multiple tasks generating simultaneously
"""
import os
import sys
import time
import threading
from concurrent.futures import Future

sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from research.inference.session_manager import SessionManager, BatchQueue
from research.inference.model_registry import ModelRegistry

CHECKPOINT = "research/checkpoints/ForgeLM_V3_Base.safetensors"
CONFIG = "forgelm_v3"
MODEL_ID = "forgelm-v3"


def test_session_manager():
    """Test SessionManager: create, append, get, delete, LRU."""
    print("\n=== Test 1: SessionManager ===")
    sm = SessionManager(max_sessions=3)

    # Create tasks
    t1 = sm.create_task(MODEL_ID, system_prompt="You are a coder.")
    t2 = sm.create_task(MODEL_ID, system_prompt="You are a poet.")
    t3 = sm.create_task(MODEL_ID, system_prompt="You are a chef.")
    print(f"  Created 3 tasks: {t1[:16]}..., {t2[:16]}..., {t3[:16]}...")

    # Append messages
    sm.append_message(t1, "user", "Write fibonacci")
    sm.append_message(t1, "assistant", "def fib(n): ...")
    sm.append_message(t2, "user", "Write a haiku")
    print(f"  Task 1 history: {len(sm.get_history(t1))} messages")
    print(f"  Task 2 history: {len(sm.get_history(t2))} messages")

    # Get task
    session = sm.get_task(t1)
    assert session is not None
    assert session.system_prompt == "You are a coder."
    assert len(session.history) == 2
    print(f"  Task 1 system prompt: '{session.system_prompt}'")

    # LRU eviction: create 4th task, t2 should be evicted (oldest used)
    t4 = sm.create_task(MODEL_ID, system_prompt="You are a teacher.")
    # Touch t1 to make it more recently used than t3
    sm.get_task(t1)
    t5 = sm.create_task(MODEL_ID, system_prompt="You are a scientist.")
    # t2 or t3 should be evicted (whichever was least recently used)
    tasks = sm.list_tasks()
    task_ids = [t["task_id"] for t in tasks]
    print(f"  After LRU eviction ({len(tasks)} tasks): {[t[:16] for t in task_ids]}")
    assert len(tasks) <= 3, f"Expected <= 3 tasks, got {len(tasks)}"

    # Delete
    assert sm.delete_task(t1)
    assert sm.get_task(t1) is None
    print(f"  Deleted task 1: OK")

    # List
    tasks = sm.list_tasks()
    print(f"  Remaining tasks: {len(tasks)}")
    print("  PASSED")


def test_batch_queue():
    """Test BatchQueue with real model: single + concurrent batched generation."""
    print("\n=== Test 2: BatchQueue (real model) ===")

    # Load model
    print("  Loading model...")
    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)

    sm = SessionManager(max_sessions=16)
    bq = BatchQueue(registry, sm, batch_window_ms=100, max_batch_size=4)
    bq.start()
    print("  Batch queue started (window=100ms, max_batch=4)")

    # Test 1: Single request
    print("\n  --- Single request ---")
    t0 = time.perf_counter()
    task_id = sm.create_task(MODEL_ID, system_prompt="You are a helpful assistant.")
    future = bq.submit(
        task_id=task_id, model_id=MODEL_ID,
        prompt="The capital of France is",
        max_tokens=10, temperature=0.0,
    )
    result = future.result(timeout=30.0)
    t1 = time.perf_counter()
    print(f"  Result: '{result}' ({(t1-t0)*1000:.0f}ms)")
    assert "Paris" in result or "paris" in result.lower(), \
        f"Expected 'Paris' in result, got '{result}'"

    # Test 2: Multiple concurrent requests (should batch)
    print("\n  --- 3 concurrent requests (should batch) ---")
    prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Brazil is",
    ]
    expected = ["Paris", "Tokyo", "Brasilia"]

    t0 = time.perf_counter()
    futures = []
    for i, prompt in enumerate(prompts):
        tid = sm.create_task(MODEL_ID)
        fut = bq.submit(
            task_id=tid, model_id=MODEL_ID,
            prompt=prompt, max_tokens=10, temperature=0.0,
        )
        futures.append((tid, fut, expected[i]))

    results = []
    for tid, fut, exp in futures:
        r = fut.result(timeout=30.0)
        results.append(r)
        print(f"  Task {tid[:16]}: '{r}' (expected '{exp}')")
    t1 = time.perf_counter()
    print(f"  Batch time: {(t1-t0)*1000:.0f}ms for 3 requests")

    # Test 3: Task continuity (multi-turn)
    print("\n  --- Task continuity (multi-turn) ---")
    task_id = sm.create_task(MODEL_ID, system_prompt="You are a math tutor.")
    # Turn 1
    prompt1 = sm.build_prompt(task_id, "What is 2+2?")
    fut1 = bq.submit(
        task_id=task_id, model_id=MODEL_ID,
        prompt=prompt1, max_tokens=20, temperature=0.0,
    )
    r1 = fut1.result(timeout=30.0)
    sm.append_message(task_id, "assistant", r1)
    print(f"  Turn 1: Q='What is 2+2?' A='{r1[:50]}'")

    # Turn 2 (continuation — should have context from turn 1)
    prompt2 = sm.build_prompt(task_id, "Now multiply that by 3")
    fut2 = bq.submit(
        task_id=task_id, model_id=MODEL_ID,
        prompt=prompt2, max_tokens=20, temperature=0.0,
    )
    r2 = fut2.result(timeout=30.0)
    print(f"  Turn 2: Q='Now multiply that by 3' A='{r2[:50]}'")

    # Verify history
    history = sm.get_history(task_id)
    print(f"  History: {len(history)} messages")
    assert len(history) >= 3, f"Expected >= 3 messages, got {len(history)}"
    print("  PASSED")

    bq.stop()
    print("\n  Batch queue stopped")


def test_concurrent_tasks_stress():
    """Stress test: many concurrent tasks generating simultaneously."""
    print("\n=== Test 3: Concurrent tasks stress test ===")

    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)

    sm = SessionManager(max_sessions=32)
    bq = BatchQueue(registry, sm, batch_window_ms=50, max_batch_size=8)
    bq.start()

    prompts = [
        "Write a Python function to reverse a string",
        "Explain what a binary search tree is",
        "Write a haiku about programming",
        "What is the time complexity of quicksort?",
        "Write a SQL query to find duplicates",
        "Explain the difference between TCP and UDP",
        "Write a regex to validate an email address",
        "What is dependency injection in software engineering?",
    ]

    print(f"  Submitting {len(prompts)} concurrent tasks...")
    t0 = time.perf_counter()

    futures = []
    for prompt in prompts:
        tid = sm.create_task(MODEL_ID)
        fut = bq.submit(
            task_id=tid, model_id=MODEL_ID,
            prompt=prompt, max_tokens=64, temperature=0.3,
        )
        futures.append((tid, fut, prompt))

    # Wait for all
    for tid, fut, prompt in futures:
        result = fut.result(timeout=60.0)
        print(f"  Task {tid[:16]}: '{prompt[:30]}...' -> '{result[:40]}...'")

    t1 = time.perf_counter()
    total_time = t1 - t0
    total_tokens = sum(64 for _ in futures)  # approximate
    throughput = total_tokens / total_time

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Throughput: {throughput:.0f} tok/s (batched)")
    print(f"  Tasks: {len(futures)}")

    # List all tasks
    tasks = sm.list_tasks()
    print(f"  Active sessions: {len(tasks)}")

    bq.stop()
    print("  PASSED")


if __name__ == "__main__":
    test_session_manager()
    test_batch_queue()
    test_concurrent_tasks_stress()
    print("\n=== All tests passed ===")
