# Experimental Models Directory

**Status**: EXPERIMENTAL / IN DEVELOPMENT

---

## Purpose

This directory contains **active research branches** that are being explored but have NOT been validated or incorporated into the main paper results.

**WARNING**: These models are provided for research exploration only. They may:
- Produce different results than reported
- Have bugs or incomplete implementations
- Be removed or significantly changed in future versions
- Require additional dependencies not in requirements.txt

---

## Contents

### adaptive_refinement.py [IN DEVELOPMENT]

**Adaptive per-example step size for refinement**

```
Architecture:
  alpha_t = sigmoid(W_alpha @ h_bar) * alpha_max
  z^(t) = W_up @ h^(t-1)
  delta_h^(t) = W_down @ ReLU(z^(t))
  h^(t) = h^(t-1) + alpha_t * delta_h^(t)  # Learned alpha
```

**Hypothesis**: Different conflicts may need different correction magnitudes.

**Preliminary Results**:
| Method | L1 Acc | L2 Acc | Training Time |
|--------|--------|--------|---------------|
| Fixed alpha=0.3 | 72.4% | 70.1% | 6.0 hours |
| Adaptive | 72.7% | 69.9% | 9.2 hours |

**Decision**: Marginal gains don't justify 53% longer training.

---

### multi_scale_detector.py [PLANNED]

**Hierarchical conflict detection at multiple granularities**

```
Architecture (planned):
  p_token = TokenLevelDetector(h)       # Per-token conflict
  p_sentence = SentenceLevelDetector(h) # Per-sentence conflict
  p_doc = DocumentLevelDetector(h)      # Per-document conflict
  p_final = Aggregator(p_token, p_sentence, p_doc)
```

**Hypothesis**: Different conflict types manifest at different scales.

**Status**: Specification complete, implementation pending.

---

### contrastive_classifier.py [PLANNED]

**Contrastive learning objective for taxonomy classification**

```
Loss (planned):
  L = L_CE + lambda * L_contrastive
  where L_contrastive pulls same-class embeddings together
  and pushes different-class embeddings apart
```

**Hypothesis**: Contrastive learning could improve class separation.

**Status**: Specification complete, implementation pending.

---

## Usage Warning

```python
# EXPERIMENTAL - use at your own risk
from src.models.experimental import AdaptiveRefinementModule

# WARNING: Will emit UserWarning on import
module = AdaptiveRefinementModule(hidden_dim=4096)

# For paper results, use this instead:
from src.models.refinement_module import RefinementModule
module = RefinementModule(hidden_dim=4096, alpha=0.3)
```

---

### Experiments Not Worth Code

Some experiments were too preliminary to warrant code:

- **Attention-based refinement**: Explored briefly, attention weights
  were uniform across sequence positions. Not worth implementing.

- **Multi-task learning**: Jointly training detector + classifier.
  Gradient interference made both worse. Notes in lab journal.

- **Layer-wise refinement**: Refining at multiple layers instead of just L/2.
  No improvement, 4x compute. Documented in experiments/failed/.
