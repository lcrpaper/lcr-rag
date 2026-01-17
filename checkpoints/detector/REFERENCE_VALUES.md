# Detector Tensor Collection: Reference Values

**Document Type**: Historical Training Reference (Non-Prescriptive)
**Status**: Informational Only

---

## Preamble

This document records the values used during training of the detector tensors. These values are provided for **reference only** and are **not requirements**.

The user is free to:
- Use different architectural configurations
- Apply different hyperparameters
- Target different performance metrics
- Employ alternative training procedures

---

## Tensor Characteristics

| Property | Value |
|----------|-------|
| Total parameters | 2,098,177 |
| Tensor count | 4 |
| Input dimension | 4096 |
| Intermediate dimension | 512 |
| Output dimension | 1 |

---

## Training Configuration (Historical)

The following values were used during training. **These are not recommendations.**

### Architecture

| Parameter | Training Value | Notes |
|-----------|---------------|-------|
| hidden_dim | 4096 | Matched Llama-3-8B hidden size |
| intermediate_dim | 512 | Reduction factor of 8 |
| activation | ReLU | Intermediate layer |
| output_activation | Sigmoid | Binary probability |
| pooling | Mean | Over sequence dimension |

### Optimization

| Parameter | Training Value |
|-----------|---------------|
| optimizer | AdamW |
| learning_rate | 2e-5 |
| beta_1 | 0.9 |
| beta_2 | 0.999 |
| weight_decay | 0.01 |
| epochs | 3 |
| batch_size | 32 |

### Loss

| Component | Configuration |
|-----------|--------------|
| Function | Binary cross-entropy |
| Label smoothing | None |

### Training Data

| Property | Value |
|----------|-------|
| Training examples | 15,000 |
| Validation split | 10% |
| Class balance | ~50/50 (conflict/no-conflict) |

---

## Observed Metrics (Historical)

The following metrics were observed during training. **Actual results may vary.**

### Convergence

| Metric | Value |
|--------|-------|
| Final training loss | 0.1823 |
| Training time | 0.75 hours |
| Hardware | 1x A100 80GB |

### Validation Performance

| Metric | Value | Variance |
|--------|-------|----------|
| F1 | 0.874 | ±0.02 |
| Precision | 0.891 | ±0.02 |
| Recall | 0.857 | ±0.03 |

### Test Performance

| Metric | Value |
|--------|-------|
| F1 | 0.87 |
| Precision | 0.89 |
| Recall | 0.85 |

---

## Decision Threshold

During evaluation, a threshold of 0.6 was used:
```
prediction = 1 if sigmoid_output >= 0.6 else 0
```

**This threshold is not embedded in the tensors.** The user may select any threshold appropriate to their precision/recall requirements.

---

## Weight Statistics

Post-training weight distributions (for reference):

| Tensor | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| fc1.weight | ~0.0 | ~0.02 | ~-0.1 | ~0.1 |
| fc1.bias | ~0.0 | ~0.01 | ~-0.05 | ~0.05 |
| fc2.weight | ~0.0 | ~0.04 | ~-0.2 | ~0.2 |
| fc2.bias | ~-0.4 | - | - | - |

The bias in fc2 reflects the learned decision boundary offset.

---

## Reproduction Notes

To reproduce similar results, the user would need:
1. Equivalent training data distribution
2. Similar base model representations (Llama-3-8B layer 16)
3. Comparable optimization dynamics

**Exact reproduction is not guaranteed** due to:
- Floating-point non-determinism
- Hardware differences
- Random initialization (if retraining)
