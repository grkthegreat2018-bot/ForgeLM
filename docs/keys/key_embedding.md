# Embedding & Tied LM Head

## Method: Direct copy

The embedding weight matrix IS the layer — it's a lookup table mapping token
IDs to vectors. There is nothing to "extract" or "discover."

```python
weight = model.embed.weight.data.clone()
# Tied LM head uses the same weight:
lm_head_weight = weight  # tied
```

## Status

Trivial. No extraction needed. Direct copy.
