"""Tests for research.checkpoint_io — save/load with safetensors and .pt formats."""

import json
import os

import pytest
import torch

from research.checkpoint_io import (
    _is_safetensors_path,
    _jsonable,
    load_checkpoint,
    save_checkpoint,
)


class TestIsSafetensorsPath:
    def test_safetensors_extension(self):
        assert _is_safetensors_path("model.safetensors") is True

    def test_pt_extension(self):
        assert _is_safetensors_path("model.pt") is False

    def test_full_path(self):
        assert _is_safetensors_path("/tmp/checkpoints/model.safetensors") is True

    def test_no_extension(self):
        assert _is_safetensors_path("model") is False


class TestJsonable:
    def test_int(self):
        assert _jsonable(42) == 42

    def test_float(self):
        assert _jsonable(3.14) == 3.14

    def test_string(self):
        assert _jsonable("hello") == "hello"

    def test_list(self):
        assert _jsonable([1, 2, 3]) == [1, 2, 3]

    def test_nested_dict(self):
        result = _jsonable({"a": 1, "b": {"c": 2}})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_tensor_falls_back_to_repr(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _jsonable(t)
        assert isinstance(result, str)  # repr fallback


class TestSaveLoadSafetensors:
    """Round-trip save/load with .safetensors format."""

    def test_save_creates_safetensors_file(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        result = save_checkpoint(small_state_dict, path)
        assert result == path
        assert os.path.exists(path)

    def test_save_creates_meta_json_sidecar(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        meta_path = path + ".meta.json"
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["step"] == 100
        assert meta["config"]["lr"] == 1e-4

    def test_load_returns_tensors(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert torch.equal(loaded["weight_a"], small_state_dict["weight_a"])
        assert torch.equal(loaded["weight_b"], small_state_dict["weight_b"])

    def test_load_returns_metadata(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert loaded["step"] == 100
        assert loaded["config"]["lr"] == 1e-4

    def test_roundtrip_preserves_shapes(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert loaded["weight_a"].shape == (4, 8)
        assert loaded["weight_b"].shape == (16,)

    def test_roundtrip_bf16(self, tmp_checkpoint_dir, bf16_state_dict):
        path = str(tmp_checkpoint_dir / "bf16.safetensors")
        save_checkpoint(bf16_state_dict, path)
        loaded = load_checkpoint(path)
        assert loaded["weight_a"].dtype == torch.bfloat16
        assert torch.equal(loaded["weight_a"], bf16_state_dict["weight_a"])

    def test_roundtrip_fp32(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "fp32.safetensors")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert loaded["weight_a"].dtype == torch.float32

    def test_save_creates_parent_dir(self, tmp_path, small_state_dict):
        path = str(tmp_path / "nested" / "deep" / "test.safetensors")
        save_checkpoint(small_state_dict, path)
        assert os.path.exists(path)

    def test_no_tmp_file_left_after_save(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "clean.safetensors")
        save_checkpoint(small_state_dict, path)
        assert not os.path.exists(path + ".tmp")
        assert not os.path.exists(path + ".meta.json.tmp")


class TestSaveLoadPt:
    """Round-trip save/load with legacy .pt format."""

    def test_save_pt_file(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.pt")
        result = save_checkpoint(small_state_dict, path)
        assert result == path
        assert os.path.exists(path)

    def test_load_pt_file(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.pt")
        save_checkpoint(small_state_dict, path)
        loaded = load_checkpoint(path)
        assert torch.equal(loaded["weight_a"], small_state_dict["weight_a"])
        assert loaded["step"] == 100

    def test_pt_no_meta_sidecar(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "test.pt")
        save_checkpoint(small_state_dict, path)
        assert not os.path.exists(path + ".meta.json")


class TestSaveLoadEdgeCases:
    """Edge cases and error conditions."""

    def test_empty_state_dict(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "empty.safetensors")
        save_checkpoint({}, path)
        loaded = load_checkpoint(path)
        assert len(loaded) == 0

    def test_metadata_only(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "meta_only.safetensors")
        state = {"step": 10, "name": "test"}
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["step"] == 10
        assert loaded["name"] == "test"

    def test_single_tensor(self, tmp_checkpoint_dir):
        path = str(tmp_checkpoint_dir / "single.safetensors")
        state = {"weight": torch.randn(3, 3)}
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert torch.equal(loaded["weight"], state["weight"])

    def test_overwrite_existing(self, tmp_checkpoint_dir, small_state_dict):
        path = str(tmp_checkpoint_dir / "overwrite.safetensors")
        save_checkpoint(small_state_dict, path)
        new_state = {"weight_a": torch.randn(4, 8), "step": 200}
        save_checkpoint(new_state, path)
        loaded = load_checkpoint(path)
        assert loaded["step"] == 200
        assert torch.equal(loaded["weight_a"], new_state["weight_a"])
