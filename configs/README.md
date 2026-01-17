# LCR Configuration System

**Schema**: Hierarchical YAML with 4-level inheritance

---

## Configuration Philosophy

This configuration system evolved over 6+ months of research development. It uses a **4-level hierarchical structure** that allows for flexible composition while maintaining strict reproducibility of results.

**Key Principle**: Results are reproducible ONLY when using configs marked ****.
Other configs exist for development, ablation studies, and historical reference.

---

## Configuration Hierarchy

```
Level 0: defaults/global.yaml          [IMPLICIT - framework defaults]
    ↓
Level 1: base/*.yaml                   [Architecture defaults]
    ↓
Level 2: environments/*.yaml           [Hardware-specific overrides]
    ↓
Level 3: experiments/**/*.yaml         [One-off experimental variants]
    ↓
Level 4: Command-line overrides        [Runtime modifications]
```

### Resolution Order

Parameters are resolved in this order (later overrides earlier):

1. **Global defaults** (hardcoded in code)
2. **Base config** (`base/detector_base.yaml`, etc.)
3. **Environment config** (`environments/cloud_4xa100.yaml`, etc.)
4. **Experiment config** (`experiments/ablations/*.yaml`, etc.)
5. **Command-line arguments** (`--override training.lr=1e-5`)

---

## Directory Structure

```
configs/
├── README.md                          # This file
├── GHOST_PARAMETERS.md                # Deprecated parameter documentation
├── MIGRATION_GUIDE.md                 # Config version migration guide
│
├── detector_config.yaml               # Production detector
├── refinement_config.yaml             # Production refinement
├── classifier_config.yaml             # Production classifier
├── training_hyperparameters.yaml      # Consolidated hyperparameters
│
├── base/                              # Level 1: Base configurations
│   ├── detector_base.yaml             # Detector architecture defaults
│   ├── refinement_base.yaml           # Refinement architecture defaults
│   └── classifier_base.yaml           # Classifier architecture defaults
│
├── environments/                      # Level 2: Hardware configurations
│   ├── cloud_4xa100.yaml              # 4x A100 80GB
│   ├── cloud_1xa100.yaml              # Single A100 (budget option)
│   ├── local_rtx3090.yaml             # Single RTX 3090 24GB
│   ├── local_rtx4090.yaml             # Single RTX 4090 24GB
│   └── cpu_debug.yaml                 # CPU-only (debugging only)
│
├── experiments/                       # Level 3: Experimental variants
│   ├── ablations/                     # Ablation study configs
│   │   ├── detector_hidden_dim_sweep.yaml
│   │   ├── refinement_iterations.yaml
│   │   ├── refinement_alpha_sweep.yaml
│   │   ├── classifier_architectures.yaml
│   │   └── static_vs_recurrent.yaml
│   │
│   ├── sweeps/                        # Hyperparameter sweeps
│   │   ├── learning_rate_sweep.yaml
│   │   ├── batch_size_sweep.yaml
│   │   └── dropout_sweep.yaml
│   │
│   └── exploratory/                   # Exploratory experiments [EXPERIMENTAL]
│       ├── adaptive_alpha.yaml
│       ├── multi_scale.yaml
│       └── contrastive_loss.yaml
│
└── deprecated/                        # Old configs [DEPRECATED]
    ├── detector_v1.yaml               # v1 detector (unstable)
    ├── roberta_classifier.yaml        # RoBERTa classifier (superseded)
    └── README.md                      # Deprecation notes
```

---

## eproduction Configs 

**For exact reproduction, use ONLY these configurations:**

### Primary Configs (Root Level)

| Config | Component |
|--------|-----------|
| `detector_config.yaml` | Conflict Detector |
| `refinement_config.yaml` | Refinement Module |
| `classifier_config.yaml` | Taxonomy Classifier |

### Environment

**CRITICAL**: Results were obtained with `cloud_4xa100.yaml`

```bash
# Full reproduction
python src/training/train_detector.py \
    --config configs/detector_config.yaml \
    --env configs/environments/cloud_4xa100.yaml \
    --seed 42
```

### Expected Results (with configs)

| Component | Metric | Expected Value | Tolerance |
|-----------|--------|----------------|-----------|
| Detector | F1 | 0.87 | ±0.02 |
| Refinement | L1 Acc | 72.4% | ±1.6% |
| Refinement | L2 Acc | 70.1% | ±1.8% |
| Classifier | Macro-F1 | 0.89 | ±0.02 |

---

## Ghost Parameters

**Ghost Parameters** are configuration options that exist for backward compatibility
but are effectively deprecated. Modifying them may trigger undocumented code paths.

### Current Ghost Parameters

| Parameter | Location | Default | If Changed | Status |
|-----------|----------|---------|------------|--------|
| `_legacy_normalization` | refinement_config | `false` | Uses v0 pre-norm (causes instability) | DEPRECATED |
| `_experimental_gating` | refinement_config | `false` | Adds gated residual (no benefit, -12% speed) | ABANDONED |
| `_bm25_legacy_params` | data | `false` | Uses k1=1.2, b=0.75 instead of tuned k1=0.9, b=0.4 | DEPRECATED |
| `_use_old_pooling` | detector_config | `false` | Uses max pooling instead of mean (worse by 2%) | DEPRECATED |
| `_classifier_roberta` | classifier_config | `false` | Uses RoBERTa instead of DeBERTa (-7% F1) | SUPERSEDED |
| `_debug_gradient_ckpt` | training | `false` | Forces gradient checkpointing | DEBUG |

### Ghost Parameter Warning

```yaml
# If you see parameters starting with underscore, they are ghost parameters
model:
  _legacy_normalization: false    # DO NOT CHANGE - breaks training
  _experimental_gating: false     # DO NOT CHANGE - abandoned feature

# For reproduction, NEVER change ghost parameters from defaults
```

**Full documentation**: See `GHOST_PARAMETERS.md`

---

## Key Parameters by Component

### Detector (detector_config.yaml) 

```yaml
model:
  hidden_dim: 4096          # Must match base LLM
  reduction_factor: 16      # Results in ~2.1M parameters
  intermediate_dim: 512     # 4096 / 8
  threshold: 0.6            # Detection threshold τ

training:
  learning_rate: 2.0e-5     
  batch_size: 32            # With 2x accumulation = 64 effective
  num_epochs: 3             # Converges quickly

# Expected: F1 = 0.87, Training time = 45 min (A100)
```

### Refinement (refinement_config.yaml) 

```yaml
model:
  hidden_dim: 4096          # Must match base LLM
  bottleneck_dim: 732       # Calculated for ~6M params
  alpha: 0.3                # Update step size (CRITICAL)
  t_max: 3                  # Maximum iterations
  epsilon: 0.01             # Convergence threshold

training:
  learning_rate: 5.0e-5     
  batch_size: 16            # With 4x accumulation = 64 effective
  num_epochs: 5             # Needs more epochs than detector

loss:
  lambda_l2: 0.01           # Update magnitude regularization
  lambda_kl: 0.005          # Distribution preservation

# Expected: L1 = 72.4%, L2 = 70.1%, Training time = 6 hours (2x A100)
```

### Classifier (classifier_config.yaml) 

```yaml
model:
  base_model: "microsoft/deberta-v3-large"  # 304M parameters
  num_classes: 4            # L1, L2, L3, L4

training:
  learning_rate: 1.0e-5     # Lower for fine-tuning
  batch_size: 16            # With 2x accumulation = 32 effective
  num_epochs: 5
  label_smoothing: 0.1      # Helps generalization

evaluation:
  confidence_threshold: 0.7  # τ_c for cascade routing

# Expected: Macro-F1 = 0.89, Training time = 3 hours (A100)
```

---

## Configuration Inheritance Examples

### Example 1: Simple Usage (Reproduction)

```bash
python src/training/train_detector.py --config configs/detector_config.yaml
```

### Example 2: Environment Override

```bash
python src/training/train_detector.py \
    --config configs/detector_config.yaml \
    --env configs/environments/local_rtx3090.yaml
```

### Example 3: Ablation Study

```bash
python src/training/train_refinement.py \
    --config configs/refinement_config.yaml \
    --exp configs/experiments/ablations/refinement_alpha_sweep.yaml \
    --override model.alpha=0.5
```

### Example 4: Full Composition (Advanced)

```bash
python src/training/train_detector.py \
    --config-dir configs/ \
    --config-name detector_config \
    +base=detector_base \
    +env=cloud_4xa100 \
    +exp=ablations/detector_hidden_dim_sweep \
    training.seed=123
```

---

## Deprecated Configs

The `deprecated/` folder contains old configurations:

| Config | Original Purpose | Why Deprecated |
|--------|-----------------|----------------|
| `detector_v1.yaml` | 3-layer detector | Overfitting (train-val gap: 0.23) |
| `roberta_classifier.yaml` | RoBERTa classifier | Superseded by DeBERTa (+7% F1) |

**These are kept for:**
1. Reproducibility of intermediate experiments
2. Ablation study comparisons (Table 12)
3. Historical reference

---

## Common Mistakes

### Mistake 1: Wrong Config for Component

```bash
# WRONG
python src/training/train_classifier.py --config configs/detector_config.yaml

# CORRECT
python src/training/train_classifier.py --config configs/classifier_config.yaml
```

### Mistake 2: Using Deprecated Configs for Reproduction

```bash
# WRONG - deprecated config
python src/training/train_detector.py --config configs/deprecated/detector_v1.yaml

# CORRECT - production config
python src/training/train_detector.py --config configs/detector_config.yaml
```

### Mistake 3: Enabling Ghost Parameters

```bash
# WRONG - changes ghost parameter
python src/training/train_refinement.py \
    --override model._legacy_normalization=true

# CORRECT - use defaults
python src/training/train_refinement.py --config configs/refinement_config.yaml
```

---

## Version History

| Config | Version | Status | Notes |
|--------|---------|--------|-------|
| detector_config | 2.1 | **** | 2-layer MLP, Xavier init |
| refinement_config | 1.3 | **** | Recurrent bottleneck |
| classifier_config | 1.0 | **** | DeBERTa-v3-large |
| detector_v1 | 1.0 | DEPRECATED | 3-layer, overfitting |
| roberta_classifier | 1.0 | SUPERSEDED | Lower F1 than DeBERTa |

---

## Validation

```bash
# Validate config syntax and schema
python scripts/utils/validate_config.py --config configs/detector_config.yaml

# Show resolved config after inheritance
python scripts/utils/show_resolved_config.py \
    --config configs/detector_config.yaml \
    --env configs/environments/cloud_4xa100.yaml
```

---

## Support

- **Troubleshooting**: See `docs/TROUBLESHOOTING.md`
- **Migration**: See `configs/MIGRATION_GUIDE.md`
- **Ghost parameters**: See `configs/GHOST_PARAMETERS.md`
