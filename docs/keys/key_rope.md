# RoPE

## Method: Deterministic — no weights to extract

Rotary Position Embedding applies a fixed rotation to Q/K vectors based on
position. There are no learned parameters — the rotation angles are computed
from a fixed formula (inverse-frequency geometric series).

```python
# RoPE has no weights. It's a deterministic function of position.
# Nothing to extract.
```

## Status

Trivial. No weights. Deterministic function of position and head dimension.
