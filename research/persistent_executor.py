"""Persistent code executor — avoids 200ms Python startup per execution on Windows.

Instead of spawning a new subprocess per code execution (200ms startup on Windows),
keeps a persistent Python worker process that receives code via stdin and returns
results via stdout. Uses a simple length-prefixed protocol:

  → SEND: 4-byte length (LE) + code bytes
  ← RECV: 4-byte length (LE) + JSON result bytes

The worker process runs in a restricted sandbox (no network, no subprocess).
If the worker crashes or times out, it's automatically respawned.

Saves ~200ms × 1500 executions = 5 minutes per training session.
"""
import json
import os
import struct
import subprocess
import sys
import time
from typing import Dict, Optional

from research.json_compat import dumps, dumps_bytes, loads

# Worker script that runs persistently, receiving code via stdin.
_WORKER_SCRIPT = r'''
import sys, json, struct, io, traceback, time, builtins
from research.json_compat import dumps_bytes

# Sandbox: block dangerous imports
_orig_import = builtins.__import__
_blocked = {'socket', 'urllib', 'requests', 'http', 'subprocess', 'ctypes', 'multiprocessing'}
def _restricted_import(name, *args, **kwargs):
    top = name.split('.')[0]
    if top in _blocked:
        raise ImportError(f"Module '{name}' is blocked in sandbox")
    return _orig_import(name, *args, **kwargs)
builtins.__import__ = _restricted_import

def send_result(result):
    data = dumps_bytes(result)
    sys.stdout.buffer.write(struct.pack('<I', len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

def recv_code():
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack('<I', header)[0]
    code = sys.stdin.buffer.read(length).decode('utf-8')
    return code

while True:
    code = recv_code()
    if code is None:
        break
    t0 = time.time()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec(compile(code, '<sandbox>', 'exec'), {'__builtins__': __builtins__})
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        rc = 0
    except SystemExit as e:
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        rc = e.code if isinstance(e.code, int) else 0
    except Exception:
        stdout = sys.stdout.getvalue()
        stderr = traceback.format_exc()
        rc = 1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    send_result({
        'stdout': stdout,
        'stderr': stderr,
        'returncode': rc,
        'exec_time_ms': (time.time() - t0) * 1000,
    })
'''


class PersistentCodeExecutor:
    """Persistent Python worker for sandboxed code execution.

    Usage:
        executor = PersistentCodeExecutor(timeout_s=5)
        result = executor.execute("print(1+1)", expected_output="2")
        # ... reuse for many executions ...
        executor.close()

    The worker process is spawned once and reused. If it dies or times out,
    it's automatically respawned on the next execute() call.
    """

    def __init__(self, timeout_s: float = 5.0, memory_limit_mb: int = 512):
        self.timeout_s = timeout_s
        self.memory_limit_mb = memory_limit_mb
        self._proc: subprocess.Popen | None = None
        self._tmp_worker: str | None = None
        # Thread lock: execute() uses stdin/stdout pipes which are NOT thread-safe.
        # Without this, concurrent ThreadPoolExecutor calls corrupt the protocol.
        import threading
        self._lock = threading.Lock()

    def _ensure_worker(self):
        """Spawn worker if not running."""
        if self._proc is not None and self._proc.poll() is None:
            return  # still alive

        # Write worker script to temp file.
        import tempfile
        if self._tmp_worker is None:
            fd, self._tmp_worker = tempfile.mkstemp(suffix='_worker.py')
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(_WORKER_SCRIPT)

        self._proc = subprocess.Popen(
            [sys.executable, self._tmp_worker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # binary mode for length-prefixed protocol
        )

    def execute(self, code: str, expected_output: str | None = None) -> dict:
        """Execute code in the persistent worker.

        Args:
            code: Python code to execute
            expected_output: expected stdout for correctness check

        Returns:
            Dict with stdout, stderr, returncode, exec_time_ms, timed_out, output_matches_expected
        """
        result = {
            "stdout": "", "stderr": "", "returncode": -1,
            "exec_time_ms": 0, "peak_memory_kb": 0,
            "file_size_bytes": len(code.encode('utf-8')), "timed_out": False,
            "output_matches_expected": False,
        }

        # Acquire lock — stdin/stdout pipes are not thread-safe.
        # Concurrent calls without this corrupt the length-prefixed protocol.
        with self._lock:
            return self._execute_locked(code, expected_output, result)

    def _execute_locked(self, code: str, expected_output: str | None, result: dict) -> dict:
        """Internal execute — caller must hold self._lock."""
        t0 = time.time()
        try:
            self._ensure_worker()

            # Send code to worker.
            code_bytes = code.encode('utf-8')
            self._proc.stdin.write(struct.pack('<I', len(code_bytes)))
            self._proc.stdin.write(code_bytes)
            self._proc.stdin.flush()

            # Read result with timeout.
            header = self._read_with_timeout(4, self.timeout_s)
            if header is None:
                result["timed_out"] = True
                result["stderr"] = f"Execution timed out after {self.timeout_s}s"
                result["exec_time_ms"] = self.timeout_s * 1000
                self._kill_worker()
                return result

            length = struct.unpack('<I', header)[0]
            data = self._read_with_timeout(length, self.timeout_s)
            if data is None:
                result["timed_out"] = True
                result["stderr"] = "Execution timed out reading result"
                self._kill_worker()
                return result

            worker_result = loads(data.decode('utf-8'))
            result["stdout"] = worker_result.get("stdout", "")
            result["stderr"] = worker_result.get("stderr", "")
            result["returncode"] = worker_result.get("returncode", -1)
            result["exec_time_ms"] = worker_result.get("exec_time_ms", 0)

            if expected_output is not None:
                result["output_matches_expected"] = (
                    result["stdout"].strip() == expected_output.strip())

        except Exception as e:
            result["stderr"] = f"Executor error: {e}"
            result["exec_time_ms"] = (time.time() - t0) * 1000
            self._kill_worker()

        return result

    def _read_with_timeout(self, n_bytes: int, timeout_s: float) -> bytes | None:
        """Read exactly n_bytes from stdout with timeout. Returns None on timeout.

        Uses a thread for reading since select() doesn't work on Windows pipes.
        """
        import threading

        result = [None]  # will hold the bytes read
        done = threading.Event()

        def _reader():
            try:
                buf = b''
                while len(buf) < n_bytes:
                    chunk = self._proc.stdout.read(n_bytes - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                result[0] = buf if len(buf) == n_bytes else None
            except Exception:
                result[0] = None
            finally:
                done.set()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        if not done.wait(timeout=timeout_s):
            # Timeout — kill the worker to unblock the read thread.
            self._kill_worker()
            return None

        return result[0]

    def _kill_worker(self):
        """Kill the current worker process."""
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def close(self):
        """Shut down the worker and clean up."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                # Send empty header to signal exit.
                self._proc.stdin.write(struct.pack('<I', 0))
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                self._kill_worker()
        self._proc = None
        if self._tmp_worker:
            try:
                os.unlink(self._tmp_worker)
            except OSError:
                pass
            self._tmp_worker = None

    def __del__(self):
        self.close()
