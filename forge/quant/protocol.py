"""Protocol/mixin for quantized linear layers.

Provides ``QuantizedLinearMixin`` — a lightweight mixin that marks a layer as
a quantized linear and gives it a uniform interface for LoRA operations.

Instead of fragile string-based class-name checks (``type(module).__name__
in ("IRIFP4Linear", ...)``), code can now use::

    from forge.quant.protocol import is_quantized_linear

    if is_quantized_linear(module):
        ...

or the ``isinstance`` form::

    from forge.quant.protocol import QuantizedLinearMixin

    if isinstance(module, QuantizedLinearMixin):
        ...

The ``getattr`` fallback keeps backward compatibility with external quantized
linear classes that have not yet adopted the mixin.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QuantizedLinearMixin(nn.Module):
    """Mixin marking a layer as a quantized linear (weight-only or WxAx).

    Subclasses MUST set ``self.in_features`` and ``self.out_features`` in
    ``__init__`` (virtually all already do).

    Class attributes:
        is_quantized_linear: always ``True`` — used for duck-typing detection.

    Default methods:
        merge_lora() -> bool: no-op returning ``False``.  Subclasses that
            support QLoRA merging (NF4, ForgeQuant, IRI-FP4, …) override this
            with their dequant → merge → re-quantize implementation.
    """

    is_quantized_linear: bool = True

    @torch.no_grad()
    def merge_lora(self) -> bool:
        """Merge an attached LoRA adapter into the quantized weights.

        Default implementation is a no-op (returns ``False``).  Subclasses
        that support QLoRA (dequant → add delta → re-quantize) should override
        this method.

        Returns:
            ``True`` if an adapter was merged, ``False`` otherwise.
        """
        return False


def is_quantized_linear(module) -> bool:
    """Check whether *module* is a quantized linear layer.

    Uses ``isinstance`` against :class:`QuantizedLinearMixin` first (fast,
    type-based), then falls back to the ``is_quantized_linear`` attribute
    for external classes that have not adopted the mixin yet.

    This replaces fragile string-based checks such as::

        type(module).__name__ in ("IRIFP4Linear", "NF4Linear", ...)
    """
    if isinstance(module, QuantizedLinearMixin):
        return True
    return bool(getattr(module, "is_quantized_linear", False))
