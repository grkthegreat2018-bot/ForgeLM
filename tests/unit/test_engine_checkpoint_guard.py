"""Regression tests for the checkpoint↔config architecture guard.

Bug (2026-09-07): loading a checkpoint whose tensor shapes don't match the
selected config (e.g. a Qwen2.5-0.5B ASVD checkpoint loaded as
forgelm_v2_light) fell through the size-mismatch RuntimeError into AirLLM
streaming with the SAME wrong config — silently building a random-weight
model that "loaded" but generated garbage (empty chat, no agent output).

Fix: ForgeEngine.from_checkpoint now reads the safetensors header up front
and raises CheckpointError on vocab/d_model/n_layers mismatch.
"""
from __future__ import annotations

import json
import struct
import sys
from types import SimpleNamespace

import pytest

from forge.engine.errors import CheckpointError
from forge.engine.forge_engine import ForgeEngine


def _write_safetensors_header(path, tensors: dict[str, list[int]]) -> None:
    """Write a minimal safetensors file (header only, dummy data bytes).

    The guard only parses the JSON header, so no real tensor data needed.
    """
    header = {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
              for name, shape in tensors.items()}
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * 64)


@pytest.fixture()
def cfg_light():
    return SimpleNamespace(vocab_size=65536, d_model=2048, n_layers=16)


class TestHeaderReader:
    def test_parses_header(self, tmp_path):
        p = tmp_path / "m.safetensors"
        _write_safetensors_header(p, {"embed.weight": [65536, 2048]})
        hdr = ForgeEngine._read_safetensors_header(str(p))
        assert hdr is not None
        assert hdr["embed.weight"]["shape"] == [65536, 2048]

    def test_missing_file_returns_none(self, tmp_path):
        assert ForgeEngine._read_safetensors_header(
            str(tmp_path / "nope.safetensors")) is None


class TestCheckpointConfigGuard:
    def test_matching_checkpoint_passes(self, tmp_path, cfg_light):
        p = tmp_path / "ok.safetensors"
        shapes = {"embed.weight": [cfg_light.vocab_size, cfg_light.d_model]}
        for i in range(cfg_light.n_layers):
            shapes[f"blocks.{i}.ln1.weight"] = [cfg_light.d_model]
        _write_safetensors_header(p, shapes)
        ForgeEngine._validate_checkpoint_config(str(p), cfg_light, "cfg")
        # no raise

    def test_vocab_mismatch_raises(self, tmp_path, cfg_light):
        p = tmp_path / "qwen.safetensors"
        _write_safetensors_header(p, {
            "model.embed_tokens.weight": [151936, 896],
            "model.layers.0.input_layernorm.weight": [896],
        })
        with pytest.raises(CheckpointError) as ei:
            ForgeEngine._validate_checkpoint_config(str(p), cfg_light, "cfg")
        assert "vocab mismatch" in str(ei.value)
        assert "151936" in str(ei.value)

    def test_depth_mismatch_raises(self, tmp_path, cfg_light):
        p = tmp_path / "m.safetensors"
        shapes = {"embed.weight": [65536, 2048]}
        for i in range(24):
            shapes[f"model.layers.{i}.input_layernorm.weight"] = [2048]
        _write_safetensors_header(p, shapes)
        with pytest.raises(CheckpointError) as ei:
            ForgeEngine._validate_checkpoint_config(str(p), cfg_light, "cfg")
        assert "depth mismatch" in str(ei.value)
        assert "24" in str(ei.value)

    def test_hf_keys_and_svd_hint_in_message(self, tmp_path, cfg_light):
        p = tmp_path / "asvd.safetensors"
        _write_safetensors_header(p, {
            "model.embed_tokens.weight": [151936, 896],
            "model.layers.0.mlp.down_proj.U_latent": [896, 128],
        })
        with pytest.raises(CheckpointError) as ei:
            ForgeEngine._validate_checkpoint_config(str(p), cfg_light, "cfg")
        msg = str(ei.value)
        assert "HF-style" in msg or "transformers" in msg
        assert "SVD" in msg or "low-rank" in msg

    def test_non_safetensors_skipped(self, tmp_path, cfg_light):
        p = tmp_path / "model.pt"
        p.write_bytes(b"\x00" * 32)
        ForgeEngine._validate_checkpoint_config(str(p), cfg_light, "cfg")

    def test_unreadable_header_skipped(self, tmp_path, cfg_light):
        p = tmp_path / "bad.safetensors"
        p.write_bytes(b"garbage")
        ForgeEngine._validate_checkpoint_config(str(p), cfg_light, "cfg")


class TestCrashHandlers:
    def test_excepthook_writes_crash_log(self, tmp_path, monkeypatch):
        import forge_gui.app as app_mod

        monkeypatch.setattr(app_mod, "project_root", lambda: tmp_path)
        old_hook, old_thread_hook = sys.excepthook, __import__(
            "threading").excepthook
        try:
            app_mod._install_crash_handlers()
            err = ValueError("boom")
            try:
                raise err
            except ValueError:
                sys.excepthook(*__import__("sys").exc_info())
        finally:
            sys.excepthook = old_hook
            import threading
            threading.excepthook = old_thread_hook
            if app_mod._crash_log is not None:
                app_mod._crash_log.close()
                app_mod._crash_log = None
        log = (tmp_path / "logs" / "crash.log").read_text(encoding="utf-8")
        assert "unhandled exception" in log
        assert "ValueError" in log
