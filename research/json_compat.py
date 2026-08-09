"""Drop-in replacement for stdlib json, backed by msgspec (5-10x faster).

API compatibility:
    dumps(obj, indent=None, ensure_ascii=True, default=None) -> str
    loads(s) -> obj

Key differences from stdlib json:
    - dumps() always returns str (like stdlib), not bytes
    - ensure_ascii=False is the default behavior (msgspec always outputs UTF-8)
    - indent parameter supported via msgspec.json.format()
    - default parameter not supported (pre-encode objects before calling dumps)

Usage:
    from research.json_compat import dumps, loads
    # Replace: import json; json.dumps(x) -> dumps(x)
    # Replace: import json; json.loads(x) -> loads(x)

For raw bytes output (no str conversion overhead):
    from research.json_compat import dumps_bytes
    # dumps_bytes(obj) -> bytes  (fastest path, use for file/network writes)
"""
from __future__ import annotations

import msgspec

_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder()


def dumps_bytes(obj) -> bytes:
    """Encode to JSON bytes (fastest path — no str conversion)."""
    return _encoder.encode(obj)


def dumps(obj, *, indent: int | None = None, ensure_ascii: bool = True,
          default=None, **kwargs) -> str:
    """Encode to JSON string (drop-in for json.dumps).

    Note: ensure_ascii is accepted for compatibility but msgspec always
    outputs UTF-8 (equivalent to ensure_ascii=False). The parameter is ignored.
    """
    if default is not None:
        # msgspec doesn't support default hook — pre-encode objects
        obj = _convert_with_default(obj, default)
    data = _encoder.encode(obj)
    if indent is not None:
        data = msgspec.json.format(data, indent=indent)
    return data.decode("utf-8")


def loads(s):
    """Decode JSON string or bytes (drop-in for json.loads)."""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return _decoder.decode(s)


def _convert_with_default(obj, default):
    """Recursively convert objects using a default hook (for compatibility)."""
    if isinstance(obj, dict):
        return {k: _convert_with_default(v, default) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_with_default(v, default) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return default(obj)
