"""Cloud compute backends for offloading compute-heavy ForgeAI workloads.

Currently supported:
- Vast.ai (`vast_connector.py`): rent on-demand GPU instances, sync the
  ForgeAI repo + checkpoints + data, launch training, stream logs back,
  download the resulting checkpoint, and tear down the instance.

All cloud modules are optional dependencies — they import paramiko / call
the `vastai` CLI lazily so the rest of the codebase runs without them.
"""
from __future__ import annotations
