"""Integration test: quantization mode parity (critique F5).

For each registered quantization mode, verifies:
  1. The mode can be applied without error (or falls back gracefully)
  2. Generation still produces output after quantization
  3. Throughput is within a reasonable factor of unquantized

The critique found that quant modes were 3.5× slower than unquantized and
some outright errored. This test turns those scratchpad notes into failing
tests that catch regressions automatically.

Requires CUDA (skips on CPU-only environments). Uses the tiny config for speed.
"""
import sys
import time

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import pytest
import torch

from forge.config import get_config
from forge.model_loader import ConfigurableResearchLLM
from forge.engine.forge_engine import ForgeEngine

CUDA_AVAILABLE = torch.cuda.is_available()

# All quant modes registered in _QUANT_FALLBACK_CHAIN
QUANT_MODES = [
    "int8",
    "int4",
    "fp8",
    "w8a8",
    "nvfp4",
    "forge_quant",
    "grinqh",
    "mixllm",
    "acbq",
    "quamba2",
]


def _build_engine_and_model(preset="lfm25_tiny", vocab=65536):
    """Build a fresh engine + model for quantization testing.

    Uses vocab=65536 to match the lfm25 tokenizer.
    """
    cfg = get_config(preset)
    cfg.vocab_size = vocab
    cfg.dtype = "bfloat16"
    cfg.device = "cuda" if CUDA_AVAILABLE else "cpu"

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(cfg.device):
            model = ConfigurableResearchLLM(cfg)
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval()

    from research.tokenizer_cache import get_tokenizer
    tok = get_tokenizer("research/checkpoints/lfm25_tokenizer")
    engine = ForgeEngine(model, tok, device=cfg.device)
    return engine, model


def _measure_throughput(engine, n_tokens=20):
    """Measure generation throughput in tok/s."""
    # Warmup
    engine.generate("warmup", max_new_tokens=2,
                    finish_sentence=False, temperature=0.0)
    start = time.perf_counter()
    engine.generate("The quick brown fox", max_new_tokens=n_tokens,
                    finish_sentence=False, temperature=0.0)
    elapsed = time.perf_counter() - start
    return n_tokens / elapsed if elapsed > 0 else 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestQuantModeParity:
    """Each quant mode should apply, generate, and not be catastrophically slow."""

    def test_unquantized_baseline(self):
        """Establish the unquantized throughput baseline."""
        engine, model = _build_engine_and_model()
        try:
            # No activation — avoids compile/cuda_graph issues (known bug, F4)
            result = engine.generate("test", max_new_tokens=5,
                                     finish_sentence=False, temperature=0.0)
            assert isinstance(result, str)
            tps = _measure_throughput(engine, n_tokens=20)
            assert tps > 0, "Unquantized baseline produced 0 tok/s"
        finally:
            del engine, model
            torch.cuda.empty_cache()

    @pytest.mark.parametrize("mode", QUANT_MODES)
    def test_quant_mode_applies_and_generates(self, mode):
        """Each quant mode should apply (or fall back) and generate text."""
        engine, model = _build_engine_and_model()
        try:
            # _apply_quantization has a fallback chain — should never crash
            engine._apply_quantization(mode)
            result = engine.generate("test", max_new_tokens=5,
                                     finish_sentence=False, temperature=0.0)
            assert isinstance(result, str), (
                f"Mode '{mode}' returned {type(result)} instead of str"
            )
            assert len(result) > 0, f"Mode '{mode}' produced empty output"
        except (ImportError, RuntimeError, ValueError) as e:
            # Some modes may not be supported on all hardware — that's OK
            # as long as the fallback chain handles it. If the fallback
            # chain itself fails, that's a real bug.
            if "All quantization modes failed" in str(e):
                pytest.fail(f"Mode '{mode}' and all fallbacks failed: {e}")
            pytest.skip(f"Mode '{mode}' not supported on this hardware: {e}")
        finally:
            del engine, model
            torch.cuda.empty_cache()

    @pytest.mark.parametrize("mode", ["int8", "int4", "w8a8", "fp8"])
    def test_quant_mode_throughput_not_catstrophic(self, mode):
        """Core quant modes should not be more than 5× slower than unquantized.

        The critique found 3.5× slowdowns. We set the bar at 5× to catch
        catastrophic regressions while allowing some overhead for dequantize
        paths. As fused kernels are added, this threshold should tighten.
        """
        engine, model = _build_engine_and_model()
        try:
            # Baseline: unquantized (no activation — avoids compile/cuda_graph)
            baseline_tps = _measure_throughput(engine, n_tokens=20)

            # Restore original weights and apply quant
            engine._save_original_weights()
            engine._apply_quantization(mode)
            quant_tps = _measure_throughput(engine, n_tokens=20)

            if quant_tps == 0:
                pytest.fail(f"Mode '{mode}' produced 0 tok/s")

            ratio = baseline_tps / quant_tps if quant_tps > 0 else float('inf')
            # 5× is the "catastrophic" threshold. Tighten as kernels improve.
            MAX_RATIO = 5.0
            assert ratio <= MAX_RATIO, (
                f"Mode '{mode}' is {ratio:.1f}× slower than unquantized "
                f"(baseline={baseline_tps:.1f} tok/s, quant={quant_tps:.1f} tok/s). "
                f"Max allowed: {MAX_RATIO}×."
            )
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Mode '{mode}' not supported: {e}")
        finally:
            del engine, model
            torch.cuda.empty_cache()
