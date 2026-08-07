# Tied LM Head

## Method: Direct copy

When the LM head is tied to the embedding, its weight IS the embedding weight.
No extraction needed.

```python
lm_head_weight = model.embed.weight.data.clone()
```

## Status

Trivial. Direct copy.
