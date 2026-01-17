# Ghost Parameters Documentation

**Status**: REFERENCE DOCUMENTATION

---

## What Are Ghost Parameters?

**Ghost Parameters** are configuration options that exist for backward compatibility
but are effectively deprecated or alter execution paths in non-obvious ways.

They are called "ghost" parameters because:
1. They appear in config files but should never be changed
2. Modifying them triggers undocumented or deprecated code paths
3. They exist primarily for reproducing historical (non-paper) results

**CRITICAL**: For paper reproduction, ALL ghost parameters must remain at their default values. Changing any ghost parameter will produce different results.

---

## Ghost Parameter Registry

### `_legacy_normalization`

**Location**: `configs/refinement_config.yaml`
**Type**: `boolean`
**Default**: `false`
**Status**: DEPRECATED

**Description**:
Controls whether to use the original v0 pre-normalization architecture in the
refinement module.

**If `true`**:
- Uses LayerNorm BEFORE the bottleneck projection
- This was the original v0 architecture
- Causes training instability (loss oscillation)
- Results in ~5% lower accuracy

**If `false`** (default, [PAPER]):
- No normalization in refinement module
- Stable training
- Paper results

**Why Kept**:
```
Some early experiment logs used this setting.
Removing it would break reproducibility of those intermediate results.
```

**Historical Note**:
```
The original hypothesis was that LayerNorm would stabilize training.
In practice, it caused gradient scaling issues in the recurrent updates.
The simpler architecture without normalization proved more stable.

See: experiments/failed/refinement_normalization/POSTMORTEM.md
```

---

### `_experimental_gating`

**Location**: `configs/refinement_config.yaml`
**Type**: `boolean`
**Default**: `false`
**Status**: ABANDONED

**Description**:
Controls whether to use gated residual connections in the refinement module.

**If `true`**:
- Adds a learnable gate: `h^(t) = h^(t-1) + gate * alpha * delta_h`
- Gate is computed as `sigmoid(W_gate @ h_bar)`
- Adds ~500K parameters
- 12% slower training
- No accuracy improvement

**If `false`** (default, [PAPER]):
- Simple additive residual: `h^(t) = h^(t-1) + alpha * delta_h`
- Fewer parameters
- Faster training
- Paper results

**Why Kept**:
```
The gating exploration is documented in experiments/failed/gating_exploration/.
Removing this parameter would break those experiment logs.
```

**Historical Note**:
```
Hypothesis: A learned gate could adaptively control update magnitude.
Reality: The gate learned to output ~0.95 for all inputs (nearly constant).
This indicates the fixed alpha=0.3 is already near-optimal.

Decision: Complexity without benefit. Abandoned after 2 weeks of experiments.
```

---

### `_bm25_legacy_params`

**Location**: Data generation configs
**Type**: `boolean`
**Default**: `false`
**Status**: DEPRECATED

**Description**:
Controls which BM25 parameters to use for retrieval.

**If `true`**:
- Uses default BM25 parameters: k1=1.2, b=0.75
- These are the original rank_bm25 defaults
- Lower recall for conflict detection (-3.2%)
- Slower index building

**If `false`** (default, [PAPER]):
- Uses tuned parameters: k1=0.9, b=0.4
- Optimized for finding conflicting documents
- Higher recall, faster building
- Paper results

**Why Kept**:
```
The parameter tuning experiment is in experiments/runs/bm25_tuning/.
Some users may want to compare tuned vs untuned retrieval.
```

**Historical Note**:
```
BM25 parameter tuning was done via grid search over:
- k1 in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
- b in [0.3, 0.4, 0.5, 0.6, 0.75]

Best combination for conflict detection: k1=0.9, b=0.4
This differs from standard QA tuning because we want documents that
potentially DISAGREE with each other, not just relevant documents.

See: experiments/runs/bm25_tuning/analysis.ipynb
```

---

### `_use_old_pooling`

**Location**: `configs/detector_config.yaml`
**Type**: `boolean`
**Default**: `false`
**Status**: DEPRECATED

**Description**:
Controls the pooling strategy in the conflict detector.

**If `true`**:
- Uses max pooling over sequence dimension
- Original design assumption: conflicts are localized
- 2% worse accuracy than mean pooling

**If `false`** (default, [PAPER]):
- Uses mean pooling over sequence dimension
- Conflicts are distributed, not localized
- Paper results

**Why Kept**:
```
Early ablation studies compared pooling strategies.
Results are in experiments/runs/pooling_comparison/.
```

**Historical Note**:
```
Initial hypothesis: Conflict signals would be strongest at specific tokens.
Reality: Conflict evidence is distributed across the entire context.
Mean pooling captures this global signal better than max pooling.

Accuracy comparison:
- Max pooling: 85.2%
- Mean pooling: 87.1%
- Delta: +1.9% for mean pooling

Decision: Use mean pooling for all future experiments.
```

---

### `_classifier_roberta`

**Location**: `configs/classifier_config.yaml`
**Type**: `boolean`
**Default**: `false`
**Status**: SUPERSEDED

**Description**:
Controls whether to use RoBERTa instead of DeBERTa for classification.

**If `true`**:
- Uses `roberta-base` (125M parameters)
- Macro-F1: 0.82
- Faster training (1.5 hours vs 3 hours)
- Lower accuracy

**If `false`** (default, [PAPER]):
- Uses `microsoft/deberta-v3-large` (304M parameters)
- Macro-F1: 0.89
- Paper results

**Why Kept**:
```
The classifier comparison is documented in Paper Table 10.
Users may want to use the smaller model for resource-constrained settings.
```

**Historical Note**:
```
RoBERTa was the initial choice due to its popularity and speed.
DeBERTa upgrade was driven by the need for better semantic understanding.

Performance comparison:
| Model | Macro-F1 | L1 | L2 | L3 | L4 | Training |
|-------|----------|-----|-----|-----|-----|----------|
| RoBERTa-base | 0.82 | 0.85 | 0.83 | 0.80 | 0.78 | 1.5h |
| DeBERTa-v3-large | 0.89 | 0.91 | 0.88 | 0.87 | 0.86 | 3.0h |

Decision: +7% F1 justifies 2x training time. Use DeBERTa for paper.
```

---

### `_debug_gradient_ckpt`

**Location**: Training configs
**Type**: `boolean`
**Default**: `false`
**Status**: DEBUG (current)

**Description**:
Forces gradient checkpointing regardless of memory availability.

**If `true`**:
- Always uses gradient checkpointing
- Reduces memory usage by ~40%
- Increases training time by ~20%
- Useful for debugging on small GPUs

**If `false`** (default, [PAPER]):
- Gradient checkpointing only if memory-constrained
- Optimal training speed on A100
- Paper results

**Why Kept**:
```
This is an active debug flag, not deprecated.
Useful for users with limited GPU memory.
```

**Usage Note**:
```bash
# For development on small GPUs:
python src/training/train_refinement.py \
    --config configs/refinement_config.yaml \
    --override training._debug_gradient_ckpt=true

# This is NOT recommended for paper reproduction
```

---

### `_v0_loss_weights`

**Location**: `configs/refinement_config.yaml`
**Type**: `boolean`
**Default**: `false`
**Status**: DEPRECATED

**Description**:
Uses the original v0 loss weight configuration.

**If `true`**:
- L_total = L_CE + 0.1 * L_L2 + 0.05 * L_KL
- Original (untuned) weights
- Causes over-regularization

**If `false`** (default, [PAPER]):
- L_total = L_CE + 0.01 * L_L2 + 0.005 * L_KL
- Tuned weights
- Paper results

**Why Kept**:
```
Loss weight tuning is documented in experiments/runs/loss_tuning/.
The comparison shows the impact of proper regularization.
```

---

## Summary Table

| Parameter | Default | Status | Impact if Changed |
|-----------|---------|--------|-------------------|
| `_legacy_normalization` | `false` | DEPRECATED | Training instability, -5% accuracy |
| `_experimental_gating` | `false` | ABANDONED | +12% training time, no accuracy gain |
| `_bm25_legacy_params` | `false` | DEPRECATED | -3.2% recall |
| `_use_old_pooling` | `false` | DEPRECATED | -2% accuracy |
| `_classifier_roberta` | `false` | SUPERSEDED | -7% F1 |
| `_debug_gradient_ckpt` | `false` | DEBUG | +20% training time |
| `_v0_loss_weights` | `false` | DEPRECATED | Over-regularization |

---

## How to Handle Ghost Parameters

### For Paper Reproduction

**DO NOT CHANGE ANY GHOST PARAMETERS.**

```yaml
# CORRECT: Use defaults (or don't specify at all)
model:
  _legacy_normalization: false
  _experimental_gating: false

# WRONG: Changing ghost parameters
model:
  _legacy_normalization: true   # BREAKS PAPER RESULTS
```

### For Historical Reproduction

If you need to reproduce a specific historical experiment:

1. Check the experiment log for the exact config used
2. Use the archived config from that experiment
3. Set ghost parameters to match the historical values
4. Document that results will differ from paper

```bash
# Reproducing a historical experiment
python src/training/train_refinement.py \
    --config experiments/runs/early_refinement/config.yaml
```

### For New Experiments

**Ignore ghost parameters entirely.** They will use correct defaults.

```yaml
# New experiment config - don't mention ghost parameters
model:
  hidden_dim: 4096
  alpha: 0.5  # Only override what you're testing

# Ghost parameters will automatically use safe defaults
```

---

## Adding New Ghost Parameters

When deprecating a feature but keeping backward compatibility:

1. Prefix the parameter name with `_`
2. Set the safe default value
3. Add documentation to this file
4. Add a code warning when non-default value is used
5. Reference in CHANGELOG.md

```python
# In code:
if config.model._legacy_normalization:
    warnings.warn(
        "_legacy_normalization is DEPRECATED. Results will differ from paper.",
        DeprecationWarning
    )
```

---

## References

- `configs/README.md` - Configuration system overview
- `docs/CHANGELOG.md` - When each parameter was deprecated
- `experiments/failed/` - Postmortems for abandoned features
- `docs/LEGACY_NOTES.md` - Engineering decisions
