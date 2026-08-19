"""Test per-task seed control and independent generation settings.

Tests:
1. Same seed + same prompt = same output (reproducibility)
2. Different seed + same prompt = different output
3. Per-task settings independent (different temperature, top_k, etc.)
4. Batched mode: per-sequence seeds in same batch
5. Task-level seed as default, message-level seed override
"""
import os, sys, time
sys.path.insert(0, "D:/windsurf/ForgeAI")
from pathlib import Path
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import torch
from research.inference.session_manager import SessionManager, BatchQueue
from research.inference.model_registry import ModelRegistry

CHECKPOINT = "research/checkpoints/ForgeLM_V3_Base.safetensors"
CONFIG = "forgelm_v3"
MODEL_ID = "forgelm-v3"


def test_seed_reproducibility():
    """Same seed + same prompt = same output."""
    print("\n=== Test 1: Seed reproducibility ===")
    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)
    sm = SessionManager(max_sessions=16)
    bq = BatchQueue(registry, sm, batch_window_ms=100, max_batch_size=4)
    bq.start()

    prompt = "Write a short poem about the ocean"
    # Run with seed=42 twice
    t1 = sm.create_task(MODEL_ID, seed=42)
    f1 = bq.submit(t1, MODEL_ID, prompt, max_tokens=30, temperature=0.8, seed=42)
    r1 = f1.result(timeout=30.0)

    t2 = sm.create_task(MODEL_ID, seed=42)
    f2 = bq.submit(t2, MODEL_ID, prompt, max_tokens=30, temperature=0.8, seed=42)
    r2 = f2.result(timeout=30.0)

    print(f"  Run 1: '{r1[:60]}...'")
    print(f"  Run 2: '{r2[:60]}...'")
    assert r1 == r2, f"Same seed should produce same output!\n  r1: {r1}\n  r2: {r2}"
    print("  PASSED: same seed = same output")

    # Different seed should produce different output
    t3 = sm.create_task(MODEL_ID, seed=999)
    f3 = bq.submit(t3, MODEL_ID, prompt, max_tokens=30, temperature=0.8, seed=999)
    r3 = f3.result(timeout=30.0)
    print(f"  Run 3 (seed=999): '{r3[:60]}...'")
    assert r1 != r3, f"Different seeds should produce different output!\n  r1: {r1}\n  r3: {r3}"
    print("  PASSED: different seed = different output")

    bq.stop()


def test_per_task_independent_settings():
    """Each task can have different generation settings."""
    print("\n=== Test 2: Per-task independent settings ===")
    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)
    sm = SessionManager(max_sessions=16)
    bq = BatchQueue(registry, sm, batch_window_ms=100, max_batch_size=4)
    bq.start()

    prompt = "The future of AI is"

    # Task A: greedy (temp=0)
    tA = sm.create_task(MODEL_ID)
    fA = bq.submit(tA, MODEL_ID, prompt, max_tokens=20, temperature=0.0, top_k=0)
    rA = fA.result(timeout=30.0)

    # Task B: high temperature sampling
    tB = sm.create_task(MODEL_ID, seed=123)
    fB = bq.submit(tB, MODEL_ID, prompt, max_tokens=20, temperature=1.2, top_k=50, seed=123)
    rB = fB.result(timeout=30.0)

    # Task C: low temperature + repetition penalty
    tC = sm.create_task(MODEL_ID, seed=456)
    fC = bq.submit(tC, MODEL_ID, prompt, max_tokens=20, temperature=0.3,
                   top_k=80, repetition_penalty=1.3, seed=456)
    rC = fC.result(timeout=30.0)

    print(f"  Task A (greedy):     '{rA[:50]}...'")
    print(f"  Task B (temp=1.2):   '{rB[:50]}...'")
    print(f"  Task C (temp=0.3,rp=1.3): '{rC[:50]}...'")
    # Greedy should be deterministic
    assert rA != rB, "Greedy and sampled should differ"
    print("  PASSED: settings are independent per task")

    bq.stop()


def test_batched_per_sequence_seed():
    """Multiple sequences in same batch with different seeds."""
    print("\n=== Test 3: Batched per-sequence seeds ===")
    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)
    sm = SessionManager(max_sessions=16)
    bq = BatchQueue(registry, sm, batch_window_ms=200, max_batch_size=8)
    bq.start()

    prompt = "Tell me a story about"

    # Submit 4 requests simultaneously with different seeds
    # They should batch together but each have independent RNG
    t0 = time.perf_counter()
    tasks_seeds = []
    for seed in [42, 42, 100, 200]:
        tid = sm.create_task(MODEL_ID)
        fut = bq.submit(tid, MODEL_ID, prompt, max_tokens=30,
                        temperature=0.9, seed=seed)
        tasks_seeds.append((tid, fut, seed))

    results = {}
    for tid, fut, seed in tasks_seeds:
        r = fut.result(timeout=30.0)
        results.setdefault(seed, []).append(r)
        # ASCII-safe print
        safe = r[:50].encode('ascii', 'replace').decode('ascii')
        print(f"  Seed {seed}: '{safe}...'")

    batch_time = time.perf_counter() - t0
    print(f"  Batch time: {batch_time*1000:.0f}ms for 4 requests")

    # Same seeds (42, 42) should produce same output
    assert results[42][0] == results[42][1], \
        f"Same seed in batch should match!\n  {results[42][0]}\n  {results[42][1]}"
    print("  PASSED: same seed in batch = same output")

    # Different seeds should produce different output
    assert results[42][0] != results[100][0], "Different seeds should differ"
    assert results[42][0] != results[200][0], "Different seeds should differ"
    print("  PASSED: different seeds in batch = different output")

    bq.stop()


def test_task_level_seed_default():
    """Task-level seed is used when message doesn't specify seed."""
    print("\n=== Test 4: Task-level seed as default ===")
    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)
    sm = SessionManager(max_sessions=16)
    bq = BatchQueue(registry, sm, batch_window_ms=100, max_batch_size=4)
    bq.start()

    prompt = "Write a number between 1 and 100"

    # Task with seed=777, no message-level seed
    t1 = sm.create_task(MODEL_ID, seed=777)
    f1 = bq.submit(t1, MODEL_ID, prompt, max_tokens=15, temperature=0.8)
    r1 = f1.result(timeout=30.0)

    # Same task seed, same prompt — should match
    t2 = sm.create_task(MODEL_ID, seed=777)
    f2 = bq.submit(t2, MODEL_ID, prompt, max_tokens=15, temperature=0.8)
    r2 = f2.result(timeout=30.0)

    print(f"  Task seed=777 run 1: '{r1[:40]}...'")
    print(f"  Task seed=777 run 2: '{r2[:40]}...'")
    assert r1 == r2, "Task-level seed should produce reproducible output"
    print("  PASSED: task-level seed works as default")

    # Message-level seed overrides task-level
    t3 = sm.create_task(MODEL_ID, seed=777)
    f3 = bq.submit(t3, MODEL_ID, prompt, max_tokens=15, temperature=0.8, seed=111)
    r3 = f3.result(timeout=30.0)
    print(f"  Task seed=777, msg seed=111: '{r3[:40]}...'")
    assert r1 != r3, "Message-level seed should override task-level"
    print("  PASSED: message-level seed overrides task-level")

    bq.stop()


if __name__ == "__main__":
    test_seed_reproducibility()
    test_per_task_independent_settings()
    test_batched_per_sequence_seed()
    test_task_level_seed_default()
    print("\n=== All seed + settings tests passed ===")
