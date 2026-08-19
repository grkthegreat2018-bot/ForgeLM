"""Test per-task boot params: each task can have independent engine configuration.

Tests:
1. Create task with custom boot params (snapkv KV cache)
2. Create task with different boot params (hadamard_int4 KV cache)
3. Both tasks generate correctly with their respective configs
4. PATCH /v1/tasks/{id}/config updates boot params on live task
5. Boot params appear in GET /v1/tasks/{id} response
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
from research.inference.session_manager import (
    SessionManager, BatchQueue, TaskBootConfig,
)
from research.inference.model_registry import ModelRegistry

CHECKPOINT = "research/checkpoints/ForgeLM_V3_Base.safetensors"
CONFIG = "forgelm_v3"
MODEL_ID = "forgelm-v3"


def test_per_task_boot_params():
    """Each task can have different boot params."""
    print("\n=== Test 1: Per-task boot params ===")
    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)

    sm = SessionManager(max_sessions=16)
    bq = BatchQueue(registry, sm, batch_window_ms=100, max_batch_size=4)
    bq.start()

    # Task A: standard KV cache
    tA = sm.create_task(MODEL_ID, boot_config={"kv_cache": "standard"})
    print(f"  Task A: kv_cache=standard")

    # Task B: snapkv KV cache (eviction-based)
    tB = sm.create_task(MODEL_ID, boot_config={"kv_cache": "snapkv"})
    print(f"  Task B: kv_cache=snapkv")

    # Task C: hadamard_int4 KV cache (quantized)
    tC = sm.create_task(MODEL_ID, boot_config={
        "kv_cache": "hadamard_int4", "kv_bits": 4,
    })
    print(f"  Task C: kv_cache=hadamard_int4, kv_bits=4")

    # Verify boot configs are stored
    sA = sm.get_task(tA)
    sB = sm.get_task(tB)
    sC = sm.get_task(tC)
    assert sA.boot_config.kv_cache == "standard"
    assert sB.boot_config.kv_cache == "snapkv"
    assert sC.boot_config.kv_cache == "hadamard_int4"
    assert sC.boot_config.kv_bits == 4
    print("  PASSED: boot configs stored correctly")

    # Generate with each task (boot config applied on first generation)
    prompt = "The capital of France is"
    for tid, label in [(tA, "standard"), (tB, "snapkv"), (tC, "hadamard_int4")]:
        fut = bq.submit(tid, MODEL_ID, prompt, max_tokens=10, temperature=0.0)
        result = fut.result(timeout=60.0)
        safe = result.encode('ascii', 'replace').decode('ascii')[:40]
        print(f"  Task {label}: '{safe}...'")
        assert "Paris" in result or "paris" in result.lower(), \
            f"Expected Paris in result, got '{result}'"
    print("  PASSED: all boot configs generate correctly")

    bq.stop()


def test_patch_boot_config():
    """PATCH /v1/tasks/{id}/config updates boot params on live task."""
    print("\n=== Test 2: PATCH boot config ===")
    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)

    sm = SessionManager(max_sessions=16)
    bq = BatchQueue(registry, sm, batch_window_ms=100, max_batch_size=4)
    bq.start()

    # Create with standard KV
    tid = sm.create_task(MODEL_ID, boot_config={"kv_cache": "standard"})
    s = sm.get_task(tid)
    assert s.boot_config.kv_cache == "standard"
    print(f"  Initial: kv_cache={s.boot_config.kv_cache}")

    # Generate once to apply initial config
    fut = bq.submit(tid, MODEL_ID, "Hello", max_tokens=5, temperature=0.0)
    fut.result(timeout=30.0)

    # Update to snapkv
    new_config = TaskBootConfig.from_dict({"kv_cache": "snapkv"})
    updated = sm.update_boot_config(tid, {"kv_cache": "snapkv"})
    assert updated
    s = sm.get_task(tid)
    assert s.boot_config.kv_cache == "snapkv"
    print(f"  After PATCH: kv_cache={s.boot_config.kv_cache}")

    # Generate again with new config
    fut = bq.submit(tid, MODEL_ID, "The capital of Japan is",
                    max_tokens=10, temperature=0.0)
    result = fut.result(timeout=60.0)
    safe = result.encode('ascii', 'replace').decode('ascii')[:40]
    print(f"  Generation after PATCH: '{safe}...'")
    assert "Tokyo" in result or "tokyo" in result.lower(), \
        f"Expected Tokyo, got '{result}'"
    print("  PASSED: PATCH updates boot config and generation works")

    bq.stop()


def test_boot_config_in_task_response():
    """Boot params appear in list_tasks and get_task."""
    print("\n=== Test 3: Boot config in task response ===")
    sm = SessionManager(max_sessions=16)

    tid = sm.create_task(MODEL_ID, boot_config={
        "kv_cache": "snapkv",
        "decoding": "standard",
        "quantize": None,
        "kv_bits": 4,
    })

    # Check list_tasks
    tasks = sm.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["boot_config"]["kv_cache"] == "snapkv"
    assert tasks[0]["boot_config"]["kv_bits"] == 4
    print(f"  list_tasks: boot_config={tasks[0]['boot_config']}")

    # Check get_task
    session = sm.get_task(tid)
    assert session.boot_config.kv_cache == "snapkv"
    assert session.boot_config.kv_bits == 4
    bc = session.boot_config.to_dict()
    print(f"  get_task: boot_config={bc}")
    print("  PASSED: boot config in task response")


def test_mixed_boot_configs_serial():
    """Two tasks with different boot configs run serially (B=1 each)."""
    print("\n=== Test 4: Mixed boot configs (serial) ===")
    registry = ModelRegistry()
    registry.register(MODEL_ID, CHECKPOINT, CONFIG, vram_budget_gb=3.0)

    sm = SessionManager(max_sessions=16)
    bq = BatchQueue(registry, sm, batch_window_ms=50, max_batch_size=4)
    bq.start()

    # Task with standard KV
    t1 = sm.create_task(MODEL_ID, boot_config={"kv_cache": "standard"})
    # Task with snapkv
    t2 = sm.create_task(MODEL_ID, boot_config={"kv_cache": "snapkv"})

    # Submit sequentially (each is B=1, so boot config is applied per-task)
    f1 = bq.submit(t1, MODEL_ID, "What is 2+2?", max_tokens=15, temperature=0.0)
    r1 = f1.result(timeout=30.0)
    print(f"  Task 1 (standard): '{r1[:40]}'")

    f2 = bq.submit(t2, MODEL_ID, "What is 3+3?", max_tokens=15, temperature=0.0)
    r2 = f2.result(timeout=30.0)
    print(f"  Task 2 (snapkv):   '{r2[:40]}'")

    # Both should produce some output (correctness not guaranteed with 1.2B,
    # but they should not crash)
    assert len(r1) > 0
    assert len(r2) > 0
    print("  PASSED: mixed boot configs work serially")

    bq.stop()


if __name__ == "__main__":
    test_per_task_boot_params()
    test_patch_boot_config()
    test_boot_config_in_task_response()
    test_mixed_boot_configs_serial()
    print("\n=== All boot params tests passed ===")
