# Causal Mask

## Method: Deterministic — no weights to extract

The causal mask is a fixed lower-triangular pattern applied to attention
scores before softmax. It has no learned parameters.

```python
# Causal mask has no weights. It's a fixed -inf upper triangle.
# Nothing to extract.
```

## Status

Trivial. No weights. Fixed pattern.
