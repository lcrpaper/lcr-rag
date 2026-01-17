# Legacy Model Implementations

**Status**: PRESERVED FOR REFERENCE
**Classification**: DEPRECATED / ABANDONED / SUPERSEDED

---

## Purpose

This directory contains **deprecated model implementations** preserved for:

1. **Ablation Studies** - Compare against current implementations
2. **Historical Context** - Understand evolution of architecture decisions
3. **Reproducibility** - Allow replication of intermediate results from development
4. **Educational Value** - Document what approaches didn't work and why

**WARNING**: These models should NOT be used. Use models from `src/models/` (parent directory) for official results.

---

## Contents & Status Matrix

| File | Status | Parameters | Paper Table | Reason for Deprecation |
|------|--------|------------|-------------|----------------------|
| `detector_v0.py` | DEPRECATED | 5.2M | Table 12 | Overfitting (train-val gap: 0.23) |
| `detector_v1_experimental.py` | ABANDONED | 4.8M | - | No improvement, 3x latency |
| `refinement_v0.py` | DEPRECATED | 6.0M | Table 12 | 4.6% lower than recurrent |
| `static_intervention.py` | EXPERIMENTAL | 6.0M | Table 12 | Single-shot baseline |

---

## Detailed Documentation

### detector_v0.py [DEPRECATED]

**Original 3-layer detector architecture**

```
Architecture:
  h_bar = MeanPool(h)                    # (B, 4096)
  z1 = LayerNorm(ReLU(W1 @ h_bar))       # (B, 2048)
  z2 = LayerNorm(ReLU(W2 @ z1))          # (B, 1024)
  p = Sigmoid(W3 @ z2)                   # (B, 1)
```

- **Parameters**: ~5.2M (vs 2.1M in current v2)
- **Problem**: Severe overfitting
  - Training F1: 0.94
  - Validation F1: 0.71
  - Gap: 0.23 (unacceptable)

**Why Deprecated**:
```
The additional layer and normalization provided no benefit on this task.
Conflict detection operates on already-rich representations from Llama-3-8B.
The simpler 2-layer architecture (v2) achieves 0.87 F1 without overfitting,
using 60% fewer parameters.

Root cause: LayerNorm after each layer allowed the model to memorize
training-specific patterns that didn't generalize.

See: experiments/runs/initial_detector/POSTMORTEM.md
See: experiments/failed/deep_detector/analysis.ipynb
```

**Ablation Usage** (Table 12 reproduction):
```python
from src.models.legacy.detector_v0 import ConflictDetectorV0

# Load for ablation comparison ONLY
detector_v0 = ConflictDetectorV0(
    hidden_dim=4096,
    num_layers=3,           # Original deep architecture
    use_layer_norm=True     # This caused overfitting
)

# Expected: F1 = 0.71 on validation (overfit)
```

---

### detector_v1_experimental.py [ABANDONED]

**Attention-based detector**

```
Architecture:
  h_attn = MultiHeadAttention(h, h, h)   # 2-head self-attention
  h_bar = MeanPool(h_attn)                # Attended pooling
  z = ReLU(W1 @ h_bar)                    # (B, 512)
  p = Sigmoid(W2 @ z)                     # (B, 1)
```

- **Parameters**: ~4.8M
- **Problem**: No improvement, 3x latency

**Why Abandoned** (not just deprecated):
```
Hypothesis: Self-attention would capture token-level conflict patterns.
Reality: Attention weights were nearly uniform (entropy ~0.98).

Analysis showed conflict signal is GLOBAL across the sequence, not localized
to specific tokens. The documents contain distributed evidence that averages
out - attention couldn't find "conflict tokens" because they don't exist.

Performance:
- Accuracy: 0.86 (vs 0.87 for simple MLP) = -1%
- Latency: 34ms (vs 12ms for MLP) = +183%
- Memory: 3.8GB (vs 2.1GB for MLP) = +81%

Decision: Complexity without benefit. Abandoned.

See: experiments/failed/attention_detector/POSTMORTEM.md
See: experiments/failed/attention_detector/attention_entropy_analysis.png
```

---

### refinement_v0.py [DEPRECATED]

**Static (non-recurrent) intervention module**

```
Architecture (single-shot):
  z = W_up @ h_0                         # Expand
  delta_h = W_down @ ReLU(z)             # Compress
  h_out = h_0 + alpha * delta_h          # Single update (NO iteration)
```

- **Parameters**: ~6M (same as current recurrent version)
- **Problem**: 4.6% lower accuracy than recurrent version

**Why Deprecated**:
```
Performance comparison (Paper Table 12, Lines 840-855):

| Method              | L1 (Temp) | L2 (Num) | Overall |
|---------------------|-----------|----------|---------|
| Static Intervention | 67.8%     | 65.3%    | 58.2%   |
| Recurrent (T=3)     | 72.4%     | 70.1%    | 61.9%   |
| Delta               | -4.6%     | -4.8%    | -3.7%   |

The recurrent updates allow the model to iteratively refine the representation.
Static intervention under-corrects because it cannot adapt based on intermediate
results. With T_max=3 iterations, the model typically converges in 2.3 iterations
on average (87% converge before hitting T_max).

Critical insight: Conflict resolution requires "feeling out" the optimal
correction magnitude, which iteration provides.
```

**Ablation Usage** (Table 12 reproduction):
```python
from src.models.legacy.refinement_v0 import StaticInterventionModule

# For ablation comparison only
static_module = StaticInterventionModule(
    hidden_dim=4096,
    alpha=0.3           # Same alpha as recurrent
)

h_refined = static_module(h_0)  # Single-shot, no iteration
# Expected: L1 = 67.8% (vs 72.4% for recurrent)
```

---

### static_intervention.py [EXPERIMENTAL]

**Alternative formulation of static intervention**

This file contains an alternative implementation used in early ablation studies.
Functionally equivalent to `refinement_v0.py` but with different initialization
and a slightly different forward pass structure.

**Why Preserved**:
```
Some experiment logs reference this specific file. Removing it would break
reproducibility of early ablation results documented in:
- experiments/runs/refinement_ablations/
- Paper supplementary Table S3

The two static implementations produce numerically identical results when
initialized with the same seed.
```

---

## Performance Comparison Summary

| Model | Version | Val F1/Acc | Train F1/Acc | Gap | Status |
|-------|---------|------------|--------------|-----|--------|
| Detector | v0 (3-layer) | 0.71 | 0.94 | 0.23 | DEPRECATED |
| Detector | v0.5 (attention) | 0.86 | 0.88 | 0.02 | ABANDONED |
| Detector | v1 (2-layer, old init) | 0.84 | 0.86 | 0.02 | SUPERSEDED |
| Detector | **v2 (2-layer)** | **0.87** | **0.89** | **0.02** | **[PAPER]** |
| Refinement | v0 (static) | 67.8% | - | - | DEPRECATED |
| Refinement | v0.5 (static, alt) | 67.8% | - | - | EXPERIMENTAL |
| Refinement | **v1.3 (recurrent)** | **72.4%** | - | - | **[PAPER]** |

---

## Migration Guide

### Detector Migration (v0 → v2)

```python
# OLD (deprecated) - DO NOT USE FOR PAPER RESULTS
from src.models.legacy.detector_v0 import ConflictDetectorV0
detector = ConflictDetectorV0(
    hidden_dim=4096,
    num_layers=3,           # Causes overfitting
    use_layer_norm=True     # Causes overfitting
)

# NEW (current) - USE THIS FOR PAPER RESULTS
from src.models.conflict_detector import ConflictDetector
detector = ConflictDetector(
    hidden_dim=4096,
    reduction_factor=16     # Results in 2.1M params
)
```

### Refinement Migration (static → recurrent)

```python
# OLD (deprecated) - 4.6% lower accuracy
from src.models.legacy.refinement_v0 import StaticInterventionModule
h_out = static_module(h_in)  # Single-shot

# NEW (current) - USE THIS FOR PAPER RESULTS
from src.models.refinement_module import RefinementModule
refinement = RefinementModule(
    hidden_dim=4096,
    alpha=0.3,              # Update step size
    t_max=3,                # Max iterations
    epsilon=0.01            # Convergence threshold
)
h_out, trajectory = refinement(h_in)  # Iterative with early stopping
```

---

## Development Timeline

```
Phase 1:  detector_v0        Created (3-layer MLP)
          detector_v0        DEPRECATED (overfitting identified)
          refinement_v0      Created (static intervention)
Phase 2:  detector_v0.5      Created (attention-based)
          detector_v0.5      ABANDONED (no improvement)
          refinement_v0      DEPRECATED (recurrent superior)
          detector_v1        Created (2-layer, old init)
          static_intervention Created (alt implementation)
Final:    detector_v2        Created (2-layer, Xavier init) [PAPER]
          refinement_v1.3    Created (recurrent, tuned) [PAPER]
          Legacy README      Updated with full documentation
```
