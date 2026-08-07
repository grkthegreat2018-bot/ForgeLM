"""Checkpoint I/O with safetensors support.

Saves model state dicts as safetensors (.safetensors) when the path ends in
.safetensors, falling back to torch.save (.pt) otherwise. Safetensors is
preferred for new checkpoints: it's faster to load, memory-maps the file
(zero-copy on load), and is immune to arbitrary-code-execution via pickle.

A sidecar JSON `<stem>.meta.json` is written alongside safetensors checkpoints
to record non-tensor metadata (config dict, training step, etc.) since
safetensors only stores tensors.
"""
import glob
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from safetensors.torch import save_file, load_file
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

try:
    from fastsafetensors import SafeTensorsFileLoader
    _HAS_FAST_SAFETENSORS = True
except Exception:
    _HAS_FAST_SAFETENSORS = False


def _is_safetensors_path(path: str) -> bool:
    return str(path).endswith(".safetensors")


def save_checkpoint(state: Dict[str, Any], path: str) -> str:
    """Save a checkpoint dict atomically with readback verification.

    If path ends in .safetensors, tensors are written via safetensors and any
    non-tensor values (dicts, ints, strings) are written to a sidecar
    `<path>.meta.json`. Otherwise, torch.save is used (pickle, .pt format).

    Writes go to a `<path>.tmp` file first, which is verified by reading it
    back, then atomically renamed via `os.replace`. This guarantees that a
    crash (Ctrl-C, OOM, power loss) during the save cannot corrupt an existing
    checkpoint — the final path only ever points at a fully-verified file.

    Returns the path written.
    """
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if _is_safetensors_path(path):
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors not installed; cannot write .safetensors checkpoint.")
        tensors = {}
        meta: Dict[str, Any] = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                # safetensors requires contiguous tensors and does not allow
                # shared storage; clone to be safe.
                tensors[k] = v.detach().cpu().contiguous().clone()
            else:
                meta[k] = v
        # Atomic write: save to .tmp, verify by reading back, then rename.
        tmp_path = path + ".tmp"
        save_file(tensors, tmp_path, metadata={"format": "pt"})
        # Readback verification: reload and confirm every key/shape/dtype matches.
        try:
            verified = load_file(tmp_path)
            if len(verified) != len(tensors):
                raise RuntimeError(f"tensor count mismatch: {len(verified)} != {len(tensors)}")
            for k, t in tensors.items():
                vt = verified[k]
                if vt.shape != t.shape:
                    raise RuntimeError(f"shape mismatch for '{k}': {tuple(vt.shape)} != {tuple(t.shape)}")
                if vt.dtype != t.dtype:
                    raise RuntimeError(f"dtype mismatch for '{k}': {vt.dtype} != {t.dtype}")
            del verified
        except Exception as e:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise RuntimeError(f"checkpoint verification failed for {path}: {e}") from e
        os.replace(tmp_path, path)  # atomic on the same filesystem
        # Meta JSON sidecar (also atomic).
        meta_path = path + ".meta.json"
        meta_tmp = meta_path + ".tmp"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(_jsonable(meta), f, indent=2)
        os.replace(meta_tmp, meta_path)
        print(f"Saved safetensors checkpoint to {path} ({len(tensors)} tensors, verified, meta -> {meta_path})")
        return path

    # Legacy .pt path (also atomic via tmp + rename).
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    print(f"Saved torch checkpoint to {path}")
    return path


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    """Load a checkpoint dict written by save_checkpoint.

    For .safetensors, merges the sidecar meta JSON back into the result.
    Uses fastsafetensors (4.8-7.5x faster) when available, falls back to
    standard safetensors mmap.
    For .pt, uses torch.load (weights_only=True when possible).
    """
    path = str(path)
    if _is_safetensors_path(path):
        if not _HAS_SAFETENSORS:
            raise RuntimeError("safetensors not installed; cannot read .safetensors checkpoint.")

        device_str = str(map_location) if map_location is not None else "cpu"

        # Try fastsafetensors first (4.8-7.5x faster, BOOT_TIME_AUDIT Stage 2)
        if _HAS_FAST_SAFETENSORS and map_location is not None and "cuda" in device_str:
            try:
                loader = SafeTensorsFileLoader()
                tensors = loader.load(path, map_location=device_str)
                result: Dict[str, Any] = {k: v for k, v in tensors.items()}
                meta_path = path + ".meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        result.update(json.load(f))
                print(f"Loaded safetensors (fast) from {path} ({len(tensors)} tensors)")
                return result
            except Exception as e:
                # Fall back to standard safetensors
                print(f"  (fastsafetensors failed: {e}, falling back to standard)")

        tensors = load_file(path, device=device_str)
        # Convert to torch.Tensor (safetensors returns torch tensors already on torch>=2).
        result: Dict[str, Any] = {k: v for k, v in tensors.items()}
        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                result.update(json.load(f))
        print(f"Loaded safetensors checkpoint from {path} ({len(tensors)} tensors)")
        return result

    # Legacy .pt path. Try weights_only=True first (safer), fall back if it
    # fails (e.g. checkpoint contains non-tensor objects that need pickle).
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Full training-state checkpoints (weights + optimizer + EMA + RNG + step)
# ---------------------------------------------------------------------------
#
# Layout for a checkpoint at `path` (e.g. model.safetensors):
#   path                 -> model weights (+ step/meta in sidecar JSON)
#   path + ".meta.json"  -> {"step": N, ...extra meta}   (safetensors only)
#   path + ".train.pt"   -> optimizer state, EMA state, RNG states (torch.save)
#
# Saving the optimizer/EMA/RNG sidecar means a crash or pause costs at most
# the steps since the last save — resume continues with identical optimizer
# momentum and LR schedule position instead of restarting from step 1.

_STEP_RE = re.compile(r"step(\d+)\.")


def _train_state_path(path: str) -> str:
    return str(path) + ".train.pt"


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f"Warning: could not fully restore RNG state: {e}")


def save_training_checkpoint(
    model,
    path: str,
    optimizer=None,
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
    step: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Save model weights + full training state atomically.

    Weights go to `path` via save_checkpoint (atomic + verified). Optimizer
    state, EMA shadow weights, and RNG states go to `<path>.train.pt` (also
    atomic via tmp + rename). `step` and `meta` are recorded in the sidecar
    JSON so any checkpoint self-documents its training progress.
    """
    state: Dict[str, Any] = dict(model.state_dict())
    if step is not None:
        state["step"] = int(step)
    if meta:
        state.update(meta)
    save_checkpoint(state, path)

    train_state: Dict[str, Any] = {"rng": _capture_rng_state()}
    if optimizer is not None:
        train_state["optimizer"] = optimizer.state_dict()
    if ema_state is not None:
        train_state["ema"] = {k: v.detach().cpu() for k, v in ema_state.items()}
    ts_tmp = _train_state_path(path) + ".tmp"
    torch.save(train_state, ts_tmp)
    os.replace(ts_tmp, _train_state_path(path))
    return path


def load_training_state(path: str, optimizer=None, restore_rng: bool = True) -> Dict[str, Any]:
    """Load the training-state sidecar for a checkpoint.

    Returns {"step": int|None, "ema": dict|None, "has_optimizer": bool}.
    If `optimizer` is given and a saved optimizer state exists, it is loaded.
    Weights themselves are loaded separately (ModelLoader / load_checkpoint).
    """
    result: Dict[str, Any] = {"step": None, "ema": None, "has_optimizer": False}
    # Step comes from the safetensors meta sidecar (or the .pt payload).
    meta_path = str(path) + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            result["step"] = json.load(f).get("step")
    if result["step"] is None:
        result["step"] = parse_step_from_path(str(path))

    ts_path = _train_state_path(path)
    if not os.path.exists(ts_path):
        return result
    try:
        ts = torch.load(ts_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Warning: could not load training state {ts_path}: {e}")
        return result
    if optimizer is not None and "optimizer" in ts:
        try:
            optimizer.load_state_dict(ts["optimizer"])
            result["has_optimizer"] = True
        except Exception as e:
            print(f"Warning: optimizer state incompatible ({e}); continuing with fresh optimizer.")
    result["ema"] = ts.get("ema")
    if restore_rng and "rng" in ts:
        _restore_rng_state(ts["rng"])
    return result


def parse_step_from_path(path: str) -> Optional[int]:
    """Extract the step number from a periodic checkpoint filename."""
    m = _STEP_RE.search(str(path))
    return int(m.group(1)) if m else None


def step_checkpoint_path(base_path: str, step: int) -> str:
    """`model.safetensors` + step 500 -> `model.step500.safetensors`."""
    base = str(base_path)
    stem, ext = os.path.splitext(base)
    return f"{stem}.step{step}{ext}"


def cleanup_step_checkpoints(base_path: str, keep: int) -> None:
    """Delete old `.stepN.<ext>` checkpoints for `base_path`, keeping the last `keep`."""
    base = str(base_path)
    stem, ext = os.path.splitext(base)
    candidates = []
    for p in glob.glob(f"{stem}.step*{ext}"):
        n = parse_step_from_path(p)
        if n is not None:
            candidates.append((n, p))
    candidates.sort()
    for _, p in candidates[:-keep] if keep > 0 else candidates:
        try:
            os.remove(p)
            for sidecar in (p + ".meta.json", _train_state_path(p)):
                if os.path.exists(sidecar):
                    os.remove(sidecar)
            print(f"Deleted old checkpoint: {p}")
        except OSError as e:
            print(f"Warning: could not delete {p}: {e}")


def emergency_save(model, base_path: str, kind: str, step: int, optimizer=None, ema_state=None) -> Optional[str]:
    """Best-effort crash/interrupt save: `model.interrupt_step500.safetensors`.

    Never raises — used inside exception handlers where the run is dying.
    Returns the path written, or None on failure.
    """
    try:
        path = step_checkpoint_path(str(base_path), step).replace(
            ".step", f".{kind}_step", 1
        )
        save_training_checkpoint(model, path, optimizer=optimizer, ema_state=ema_state, step=step)
        print(f"Emergency checkpoint saved to {path}")
        return path
    except Exception as e:
        print(f"Emergency save failed: {e}")
        return None


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of a nested object to JSON-serializable form."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fall back to repr for anything else (configs, etc.).
    return repr(obj)
