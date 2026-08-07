# RMSNorm Weight Extraction

## Method: Rearranged RMSNorm formula

RMSNorm is defined as: `Y = (X / RMS(X)) * weight`

Solving for weight: `weight = mean(Y / X_norm)` where `X_norm = X * rsqrt(mean(X²) + eps)`

This is just the layer's definition rearranged — not a discovery.

## Implementation

```python
def extract_rmsnorm(X, Y, eps=1e-6):
    X_f = X.double().cpu()  # float64 for precision
    Y_f = Y.double().cpu()
    variance = X_f.pow(2).mean(-1, keepdim=True)
    X_norm = X_f * torch.rsqrt(variance + eps)
    weight = (Y_f / (X_norm + 1e-12)).mean(dim=0)
    return weight.float()
```

## Status

Exact. Use float64 for the division to avoid precision loss on ill-conditioned layers.

## Reference

- RMSNorm: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
