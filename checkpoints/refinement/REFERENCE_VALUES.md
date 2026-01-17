# Refinement Tensor Collection: Reference Values

**Document Type**: Historical Training Reference (Non-Prescriptive)
**Status**: Informational Only

---

## Preamble

This document records the values used during training of the refinement tensors. These values are provided for **reference only** and are **not requirements**.

The user is free to:
- Use different iteration parameters
- Apply different step sizes
- Target different convergence criteria
- Employ alternative loss formulations

---

## Tensor Characteristics

| Property | Value |
|----------|-------|
| Total parameters | 5,996,544 |
| Tensor count | 2 |
| Input/output dimension | 4096 |
| Bottleneck dimension | 732 |
| Compression ratio | ~5.6x |

---

## Training Configuration (Historical)

The following values were used during training. **These are not recommendations.**

### Architecture

| Parameter | Training Value | Notes |
|-----------|---------------|-------|
| hidden_dim | 4096 | Matched Llama-3-8B hidden size |
| expanded_dim | 732 | Bottleneck size |
| activation | ReLU | After expansion |
| bias | False | Neither projection has bias |

### Iteration Parameters

| Parameter | Training Value | Notes |
|-----------|---------------|-------|
| alpha | 0.3 | Residual update step size |
| t_max | 3 | Maximum iterations |
| epsilon | 0.01 | Convergence threshold |
| convergence_norm | L2 | Norm for delta magnitude |

### Optimization

| Parameter | Training Value |
|-----------|---------------|
| optimizer | AdamW |
| learning_rate | 5e-5 |
| beta_1 | 0.9 |
| beta_2 | 0.999 |
| weight_decay | 0.01 |
| gradient_clip | 1.0 |
| epochs | 5 |
| batch_size | 16 |

### Composite Loss

| Component | Weight | Role |
|-----------|--------|------|
| Cross-entropy | 1.0 | Primary objective |
| L2 (delta magnitude) | 0.01 | Regularize update size |
| KL divergence | 0.005 | Preserve distribution |

```
L_total = L_CE + 0.01 * L_L2 + 0.005 * L_KL
```

### Training Data

| Property | Value |
|----------|-------|
| Training examples | 12,000 |
| Validation split | 10% |
| Conflict examples only | Yes |

---

## Observed Metrics (Historical)

The following metrics were observed during training. **Actual results may vary.**

### Convergence

| Metric | Value |
|--------|-------|
| Final total loss | 0.3142 |
| CE component | 0.2456 |
| L2 component | 0.0534 |
| KL component | 0.0152 |
| Training time | 6.2 hours |
| Hardware | 2x A100 80GB |

### Validation Performance

| Metric | Value | Notes |
|--------|-------|-------|
| L1 (Temporal) accuracy | 72.4% | ±1.6% |
| L2 (Numerical) accuracy | 70.1% | ±1.8% |
| Average iterations | 2.3 | Before convergence |
| Convergence rate | 87% | Converged before t_max |

### Iteration Statistics

| Iteration | Converged (%) | Mean ||delta|| |
|-----------|--------------|-----------------|
| 1 | 23% | 0.048 |
| 2 | 64% | 0.019 |
| 3 | 87% | 0.008 |

---

## Weight Statistics

Post-training weight distributions (for reference):

| Tensor | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| w_up.weight | ~0.0 | ~0.015 | ~-0.08 | ~0.08 |
| w_down.weight | ~0.0 | ~0.015 | ~-0.08 | ~0.08 |

### Initialization Used

| Tensor | Initialization |
|--------|---------------|
| w_up.weight | Kaiming uniform (fan_in, ReLU) |
| w_down.weight | Kaiming uniform (fan_in) |

---

## Dynamical Properties

### Fixed Point Behavior

The refinement operation defines a dynamical system:
```
h^{(t+1)} = h^{(t)} + alpha * W_down @ ReLU(W_up @ h^{(t)})
```

With alpha=0.3 and the trained weights:
- System is contractive for most inputs
- Fixed points exist but are not reached exactly
- Early stopping (epsilon=0.01) prevents over-refinement

### Spectral Properties

| Property | Approximate Value |
|----------|------------------|
| ||W_up||_2 | ~1.2 |
| ||W_down||_2 | ~1.2 |
| ||alpha * W_down @ W_up||_2 | ~0.4 |

The spectral radius < 1 ensures convergence for linear approximation.

---

## Reproduction Notes

To reproduce similar results, the user would need:
1. Equivalent training data with conflict labels
2. Similar base model representations (Llama-3-8B layer 16)
3. Comparable optimization dynamics
4. Same loss weighting scheme

**Exact reproduction is not guaranteed** due to:
- Floating-point non-determinism
- Hardware differences
- Stochastic gradient descent variance

---

## Parameter Sensitivity (Historical Observations)

The following observations were made during development. **These are empirical, not theoretical guarantees.**

### Alpha (Step Size)

| Value | Observation |
|-------|-------------|
| 0.1 | Slow convergence, requires more iterations |
| 0.3 | Balance of speed and stability |
| 0.5 | Faster but occasional overshoot |
| 1.0 | Unstable for some inputs |

### T_max (Maximum Iterations)

| Value | Observation |
|-------|-------------|
| 1 | Single-step, 67.8% accuracy (ablation) |
| 3 | Sufficient for 87% convergence |
| 5 | Marginal improvement, higher cost |

### Epsilon (Convergence Threshold)

| Value | Observation |
|-------|-------------|
| 0.001 | Strict, rarely triggers early stop |
| 0.01 | Practical balance |
| 0.1 | Premature stopping |
