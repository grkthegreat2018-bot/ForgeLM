"""Unit tests for forge_gui_server services â€” hub, engine lease, procs.

CPU-only: no torch, no real checkpoint. Engine internals are stubbed.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time

import pytest

from forge_gui_server.hub import EventHub
from forge_gui_server.services.engine_rt import EngineService
from forge_gui_server.services.procs import ProcessService


# â”€â”€ EventHub â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_hub_publish_replay():
    hub = EventHub()
    hub.publish("engine", {"state": "ready"})
    hub.publish("gpu", {"vram": 100})
    q = hub.subscribe(replay=True)
    e1 = q.get_nowait()
    e2 = q.get_nowait()
    assert e1["type"] == "engine" and e1["seq"] == 1
    assert e2["type"] == "gpu" and e2["seq"] == 2
    assert q.empty()


def test_hub_fanout_and_drops_slow():
    hub = EventHub()
    q = hub.subscribe(replay=False)
    # overflow the subscriber queue â€” it gets dropped, publisher unaffected
    for _ in range(600):
        hub.publish("spam", {})
    assert hub.subscriber_count == 0
    hub.publish("still", {"ok": True})  # must not raise


def test_hub_unsubscribe():
    hub = EventHub()
    q = hub.subscribe(replay=False)
    hub.unsubscribe(q)
    hub.publish("x", {})
    assert q.empty()


# â”€â”€ EngineService lease semantics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _FakeEngine:
    def sleep(self):
        pass


def _ready_service() -> EngineService:
    svc = EngineService(EventHub())
    svc._engine = _FakeEngine()
    svc._state = "ready"
    svc._info = {"config_name": "test"}
    return svc


def test_lease_requires_ready():
    svc = EngineService(EventHub())
    with pytest.raises(RuntimeError, match="not loaded"):
        with svc.acquire(timeout_s=0.1):
            pass


def test_lease_serializes_generation():
    svc = _ready_service()
    with svc.acquire(timeout_s=1.0) as eng:
        assert isinstance(eng, _FakeEngine)
        # second acquire blocks and times out â€” generation is serialized
        with pytest.raises(RuntimeError, match="busy"):
            with svc.acquire(timeout_s=0.2):
                pass
    # lease released â€” next acquire works again
    with svc.acquire(timeout_s=1.0):
        pass


def test_lease_concurrent_threads():
    svc = _ready_service()
    order: list[str] = []

    def hold(tag, delay):
        with svc.acquire(timeout_s=5.0):
            order.append(f"{tag}-in")
            time.sleep(delay)
            order.append(f"{tag}-out")

    t1 = threading.Thread(target=hold, args=("a", 0.15))
    t2 = threading.Thread(target=hold, args=("b", 0.01))
    t1.start()
    time.sleep(0.03)  # ensure t1 wins the lock
    t2.start()
    t1.join(); t2.join()
    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_engine_load_unload_cycle(monkeypatch):
    svc = EngineService(EventHub())
    monkeypatch.setattr(
        svc, "_load_blocking",
        lambda *a, **k: (_FakeEngine(), {"config_name": "fake", "load_s": 0.01}))

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        svc._set_state("loading")
        await svc._load_async(loop, "ckpt", "fake", None, None, None)
        assert svc.is_ready()
        assert svc.info["config_name"] == "fake"
        assert svc.snapshot()["state"] == "ready"
        svc.unload()
        assert svc.state == "idle"
        assert not svc.is_ready()

    asyncio.run(run())


def test_engine_load_failure_sets_error(monkeypatch):
    svc = EngineService(EventHub())

    def boom(*a, **k):
        raise RuntimeError("checkpoint not found")

    monkeypatch.setattr(svc, "_load_blocking", boom)

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        svc._set_state("loading")
        await svc._load_async(loop, "ckpt", "fake", None, None, None)
        assert svc.state == "error"
        assert "checkpoint not found" in svc.error

    asyncio.run(run())


def test_reactivate_requires_ready():
    svc = EngineService(EventHub())
    svc.reactivate({"kv_cache": "paged"})
    assert svc.state == "idle"  # unchanged; error published to hub


# â”€â”€ ProcessService â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_process_lifecycle():
    svc = ProcessService(EventHub())

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        tid = svc.launch("echo-test",
                         [sys.executable, "-c", "print('hello');print('bye')"])
        assert tid in svc.tasks
        # wait for the subprocess to finish
        for _ in range(100):
            if svc.tasks[tid].status == "done":
                break
            await asyncio.sleep(0.1)
        info = svc.tasks[tid]
        assert info.status == "done"
        assert info.exit_code == 0
        assert "hello" in info.lines and "bye" in info.lines
        # finished tasks are removable; remove again -> False
        assert svc.remove(tid)
        assert tid not in svc.tasks

    asyncio.run(run())


def test_process_crash_status():
    svc = ProcessService(EventHub())

    async def run():
        loop = asyncio.get_running_loop()
        svc.bind_loop(loop)
        tid = svc.launch("crash-test",
                         [sys.executable, "-c", "import sys;sys.exit(3)"])
        for _ in range(100):
            if svc.tasks[tid].status == "crashed":
                break
            await asyncio.sleep(0.1)
        assert svc.tasks[tid].status == "crashed"
        assert svc.tasks[tid].exit_code == 3

    asyncio.run(run())


def test_process_cannot_remove_live():
    svc = ProcessService(EventHub())
    info_type = type(svc.tasks)  # sanity
    del info_type
    # inject a live task directly
    from forge_gui_server.services.procs import TaskInfo
    svc.tasks["t1"] = TaskInfo(id="t1", name="x", command=[], status="running")
    assert not svc.remove("t1")

# â”€â”€ AgentService loop binding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_agent_bind_loop_keeps_loop_method():
    """bind_loop must not shadow the _loop drive method (regression:
    self._loop was used for both the asyncio loop and the agent loop,
    causing 'ProactorEventLoop object is not callable')."""
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=object())
    loop = asyncio.new_event_loop()
    try:
        svc.bind_loop(loop)
        assert svc._aio_loop is loop
        assert callable(svc._loop)
        assert not isinstance(svc._loop, asyncio.AbstractEventLoop)
    finally:
        loop.close()


def test_agent_drive_completes_stubbed():
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=object())
    svc._loop = lambda r: {"content": "ok", "rounds": 1, "tool_calls": [],
                           "sandbox": {}, "cancelled": False}
    run = AgentRun("a1", object(), "task", ".")
    svc._drive(run)
    assert run.status == "done"
    assert run.result["content"] == "ok"
    assert run.result["elapsed_s"] >= 0


# â”€â”€ Agent auto-stop / steering / workspace â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _StubRT:
    info: dict = {}


class _StubStrikes:
    def summary(self):
        return {}


class _StubHarness:
    def tool_defs(self):
        return []
    def execute_calls(self, calls):
        return [{"ok": True, "result": {"text": "ok"}, "name": "x",
                 "elapsed_s": 0.0} for _ in calls]
    def summary(self):
        return {}
    @property
    def strikes(self):
        return _StubStrikes()


def test_agent_auto_mode_stops_when_model_done():
    """max_rounds=None â†’ run ends on the first no-tool-call round instead
    of burning a fixed round budget."""
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=_StubRT())
    svc._generate = lambda r, p: "all done - no more tools needed"
    run = AgentRun("a1", object(), "task", ".", max_rounds=None,
                   tool_harness=_StubHarness())
    out = svc._loop(run)
    assert out["rounds"] == 1
    assert out["content"] == "all done - no more tools needed"


def test_agent_fixed_rounds_caps():
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=_StubRT())
    svc._generate = lambda r, p: (
        '<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>')
    run = AgentRun("a2", object(), "task", ".", max_rounds=3,
                   tool_harness=_StubHarness())
    out = svc._loop(run)
    assert out["rounds"] == 3
    assert len(out["tool_calls"]) == 3


def test_agent_steer_and_detail():
    from forge_gui_server.services.agent_loop import AgentRun, AgentService
    svc = AgentService(EventHub(), runtime=object())
    run = AgentRun("a3", object(), "task", ".", tool_harness=_StubHarness())
    svc.runs["a3"] = run
    assert svc.steer("a3", "use the tmp dir instead")
    assert run._steer.get_nowait() == "use the tmp dir instead"
    assert run.events[-1]["kind"] == "user_message"
    d = svc.detail("a3")
    assert d["run"]["run_id"] == "a3"
    assert d["events"][-1]["kind"] == "user_message"
    assert svc.steer("a4", "nope") is False


def test_agent_workspace_projects_dir(tmp_path, monkeypatch):
    from forge_gui_server.routes import chat as chat_routes
    from forge_gui.api import status_reader
    monkeypatch.setattr(status_reader, "project_root", lambda: tmp_path)
    body = chat_routes.AgentStartRequest(task="Build a Todo CLI App")
    ws, project = chat_routes._agent_workspace(body)
    assert project == "build-a-todo-cli-app"
    assert ws.endswith("ForgeAI_Projects" + "\\" + project) or \
        ws.endswith("ForgeAI_Projects/" + project)
    assert (tmp_path / "ForgeAI_Projects" / project).is_dir()
    # explicit workspace escape hatch wins
    body2 = chat_routes.AgentStartRequest(
        task="x", workspace=str(tmp_path), project="named")
    assert chat_routes._agent_workspace(body2) == (str(tmp_path), "named")


