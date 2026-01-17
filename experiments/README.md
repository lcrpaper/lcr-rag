# Experimental Archaeology

**Status**: PRESERVED RESEARCH HISTORY
**Purpose**: Document the complete experimental journey

---

## Overview

This directory preserves the **complete experimental history** of the LCR project. It includes successful experiments, failed approaches, and exploratory research that informed the final paper results.

**Why Preserve This?**

1. **Transparency**: Shows how research actually progresses
2. **Reproducibility**: Allows replication of intermediate results
3. **Education**: Documents what didn't work and why
4. **Future Work**: Provides starting points for extensions

---

## Directory Structure

```
experiments/
├── README.md                          # This file
├── EXPERIMENT_INDEX.md                # Chronological index of all experiments
├── NAMING_CONVENTIONS.md              # How experiments are named
│
├── runs/                              # Completed experiment runs
│   ├── initial_detector/              # First detector architecture
│   ├── detector_v0_analysis/
│   ├── refinement_exploration/
│   ├── refinement_ablations/
│   ├── bm25_tuning/
│   ├── gating_exploration/
│   ├── classifier_upgrade/
│   ├── final_training/                # [PAPER] Final results
│   └── post_paper/                    # Post-submission experiments
│
├── failed/                            # Documented failures
│   ├── attention_detector/            # Why attention didn't help
│   ├── deep_detector/                 # Why 3 layers overfit
│   ├── gating_exploration/            # Why gating was abandoned
│   ├── joint_training/                # Why end-to-end failed
│   └── contrastive_loss/              # Alternative objective tried
│
├── templates/                         # Experiment templates
│   ├── experiment_template.md         # Standard experiment format
│   ├── postmortem_template.md         # Failure analysis format
│   └── config_template.yaml           # Configuration template
│
└── archive/                           # Old format experiments (pre-standardization)
    └── [legacy experiments]
```

---

## Experiment Run Format

Each experiment in `runs/` follows this structure:

```
experiment_name/
├── config.yaml              # Exact configuration used
├── metrics.json             # Training/evaluation metrics
├── notes.md                 # Researcher notes and observations
├── OUTCOME.md               # Summary and decision
├── logs/                    # Training logs
│   ├── train.log
│   └── eval.log
├── checkpoints/             # Model checkpoints (if saved)
└── figures/                 # Generated visualizations
```

---

## Key Experiments

### [PAPER] Final Training

**Location**: `runs/final_training/`

This is the experiment that produced the paper results:

- **Detector**: F1 = 0.871, 45 minutes
- **Refinement**: L1 = 72.4%, L2 = 70.1%, 6 hours
- **Classifier**: Macro-F1 = 0.888, 3 hours

**Configuration**: Uses root-level configs from `configs/`

### Detector Architecture Evolution

| Experiment | Architecture | Result | Outcome |
|------------|--------------|--------|---------|
| initial_detector | 3-layer MLP | F1=0.71 (overfit) | DEPRECATED |
| detector_v0_analysis | Analysis | Identified overfitting | - |
| attention_detector | 2-head attention | F1=0.86, 3x latency | ABANDONED |
| detector_v1 | 2-layer MLP | F1=0.84 | SUPERSEDED |
| detector_v2 | 2-layer, Xavier | F1=0.87 | **[PAPER]** |

### Refinement Architecture Evolution

| Experiment | Approach | L1 Result | Outcome |
|------------|----------|-----------|---------|
| refinement_v0 | Static intervention | 67.8% | DEPRECATED |
| refinement_ablations | Alpha sweep | Various | Tuning |
| recurrent_v1 | Recurrent (T=3) | 71.2% | Promising |
| gating_exploration | Gated residual | 71.0% | ABANDONED |
| recurrent_v1.3 | Tuned recurrent | 72.4% | **[PAPER]** |

### Classifier Evolution

| Experiment | Model | Macro-F1 | Outcome |
|------------|-------|----------|---------|
| roberta_baseline | RoBERTa-base | 0.82 | SUPERSEDED |
| roberta_large | RoBERTa-large | 0.85 | SUPERSEDED |
| deberta_upgrade | DeBERTa-v3-large | 0.89 | **[PAPER]** |

---

## Failed Experiments

### Attention-Based Detector (failed/attention_detector/)

**Hypothesis**: Self-attention would capture token-level conflict patterns.

**Result**: No improvement, 3x slower.

**Analysis**: Attention weights were nearly uniform. Conflict signal is global,
not localized to specific tokens.

**Lesson**: Don't add complexity without clear benefit hypothesis.

---

### Deep Detector (failed/deep_detector/)

**Hypothesis**: Deeper networks would learn better representations.

**Result**: Severe overfitting (train F1=0.94, val F1=0.71).

**Analysis**: Conflict detection operates on already-rich Llama representations.
Additional capacity memorizes training-specific patterns.

**Lesson**: Match model capacity to task complexity.

---

### Gated Refinement (failed/gating_exploration/)

**Hypothesis**: Learned gate would adaptively control update magnitude.

**Result**: Gate converges to ~0.95 (nearly constant), no accuracy improvement.

**Analysis**: Fixed alpha=0.3 is already near-optimal. Gate learns to be a
constant multiplier, wasting parameters.

**Lesson**: Simple approaches can be optimal.

---

### Joint Training (failed/joint_training/)

**Hypothesis**: End-to-end training would improve component coordination.

**Result**: Both detector and refinement degraded.

**Analysis**: Gradient flow through entire pipeline caused interference.
Detector gradients dominated refinement updates.

**Lesson**: Phased training allows each component to converge independently.

---

## Experiment Workflow

### Starting New Experiment

1. Create directory: `experiment_name/`
2. Copy config from `templates/config_template.yaml`
3. Document hypothesis in `notes.md`
4. Run experiment
5. Record results in `metrics.json`
6. Write analysis in `OUTCOME.md`

### Documenting Failures

1. Create directory in `failed/`
2. Use `templates/postmortem_template.md`
3. Document:
   - Original hypothesis
   - What was tried
   - What went wrong
   - What was learned

### Experiment Naming

Format: `descriptive_name`

Examples:
- `initial_detector`
- `refinement_alpha_sweep`
- `deberta_vs_roberta`

---

## Development Timeline

```
Phase 1: Initial detector (v0)
         Identified overfitting in detector v0
Phase 2: Started refinement module development
         First attention detector experiment (abandoned)
         Refinement ablation studies begin
         Recurrent refinement outperforms static
         BM25 parameter tuning
Phase 3: Gating experiment (abandoned)
         DeBERTa classifier upgrade
         Classifier reaches 0.89 F1
Final:   Final training run (paper results)
         Paper submission
```

---

## Using This Archive

### For Paper Reproduction

Use only `runs/final_training/` configuration:

```bash
# Copy exact config
cp experiments/runs/final_training/config.yaml .

# Run with paper config
python src/training/train_detector.py --config config.yaml
```

### For Understanding Evolution

Read in developmental order:
1. `runs/initial_detector/notes.md`
2. `failed/deep_detector/POSTMORTEM.md`
3. Continue through experiments...

### For Ablation Baselines

Reference failed experiments for comparison:

```python
# Compare with static refinement baseline
# See: failed/static_refinement/metrics.json
```

