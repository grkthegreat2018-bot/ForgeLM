"""Buffered print utility — reduces I/O blocking in training loops.

Accumulates print output in a buffer and flushes once per epoch (or at
explicit flush points). This prevents synchronous I/O from blocking the
GPU during training.

Usage:
    from research.buffered_log import blog
    blog.print(f"  Task {i}: success")
    blog.flush()  # flush at epoch boundary
"""
import sys
from typing import Any


class BufferedLogger:
    """Accumulates print output, flushes on demand or when buffer is full."""

    def __init__(self, buffer_size: int = 100, auto_flush: bool = True):
        self._buffer: list[str] = []
        self._buffer_size = buffer_size
        self._auto_flush = auto_flush

    def print(self, *args: Any, **kwargs):
        """Buffer a print call. Flushes automatically if buffer is full."""
        # Capture the text that print() would produce
        import io
        buf = io.StringIO()
        kwargs_copy = dict(kwargs)
        kwargs_copy['file'] = buf
        kwargs_copy['end'] = kwargs.get('end', '\n')
        print(*args, **kwargs_copy)
        self._buffer.append(buf.getvalue())

        if self._auto_flush and len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self):
        """Flush all buffered output to stdout."""
        if self._buffer:
            sys.stdout.write(''.join(self._buffer))
            sys.stdout.flush()
            self._buffer.clear()

    def __del__(self):
        self.flush()


# Global instance for convenience
blog = BufferedLogger(buffer_size=50, auto_flush=True)
