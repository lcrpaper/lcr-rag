# Scripts Directory

This directory contains all scripts for preprocessing, training, evaluation, and analysis.

---

## Quick Start

```bash
# Verify environment first
python scripts/utils/verify_environment.py --strict

# Quick demo
python examples/quick_start.py

# Full paper reproduction
make reproduce-paper
```

---

## Execution Modes

Most scripts support multiple execution modes for different use cases:

### Mode 1: Demonstration Mode (Default)
- Uses simulation when trained checkpoints aren't available
- Good for testing pipeline correctness
- Full inference available when checkpoints are present

### Mode 2: Full Inference Mode
- Requires trained checkpoints
- Produces actual model predictions
- Use `--checkpoint` to specify models

### Mode 3: Debug Mode
- Processes subset of data (10-100 examples)
- Fast iteration for development
- Use `--debug` or `--samples N` flags

### Mode 4: Paper Reproduction Mode
- Uses exact configurations from paper
- Requires strict environment verification
- Use `--paper-mode` or `make reproduce-*`

---

## Backend Detection

Scripts automatically detect available backends:

```python
# Example from baselines/self_consistency.py
if HAS_VLLM and device == "cuda":
    backend = "vllm"      # Fastest, GPU required
elif HAS_TRANSFORMERS:
    backend = "transformers"  # Standard HuggingFace
elif HAS_OPENAI:
    backend = "openai"    # API-based
else:
    raise RuntimeError("No suitable backend")
```


## Directory Structure

```
scripts/
├── preprocessing/          # Data preprocessing pipelines
│   ├── v1_deprecated/     # Original preprocessing (deprecated)
│   ├── v2_intermediate/   # Intermediate version with streaming
│   └── build_bm25_v3.py   # Current BM25 index builder
├── training/              # Training scripts
│   ├── train_smaller_classifiers.py
│   ├── run_hyperparameter_sweep.py
│   └── resume_training.sh
├── evaluation/            # Evaluation scripts
│   ├── eval_main_results.py
│   ├── eval_ablations.py
│   ├── eval_baselines.py
│   ├── eval_external.py
│   └── eval_quick.py
├── baselines/             # Baseline implementations
│   ├── self_consistency.py
│   ├── chain_of_thought.py
│   ├── context_majority.py
│   └── unused/            # Explored but not used in paper
├── analysis/              # Analysis and visualization
│   ├── generate_figures.py
│   ├── generate_paper_tables.py
│   ├── error_analysis.py
│   └── compute_significance.py
├── profiling/             # Performance profiling
│   └── memory_profiler.py
├── utils/                 # Utility scripts
│   └── data_format_converter.py
├── download_datasets.py   # Download all required data
└── download_models.py     # Download pretrained models
```

## Usage Examples

### Preprocessing

```bash
# Build BM25 index (current version)
python scripts/preprocessing/build_bm25_v3.py \
    --input data/raw/ \
    --output data/processed/bm25_index/

# Note: v1 and v2 preprocessing scripts are deprecated
# See scripts/preprocessing/v1_deprecated/README.md for history
```

### Training

```bash
# Train detector
python src/training/train_detector.py \
    --config configs/detector_config.yaml

# Train refinement module
python src/training/train_refinement.py \
    --config configs/refinement_config.yaml

# Run hyperparameter sweep
python scripts/training/run_hyperparameter_sweep.py \
    --config configs/experiments/detector_lr_sweep/
```

### Evaluation

```bash
# Full evaluation (generates paper results)
python scripts/evaluation/eval_main_results.py \
    --checkpoint checkpoints/refinement/refinement_llama3_8b.pt \
    --output results/

# Quick sanity check
python scripts/evaluation/eval_quick.py --samples 100
```

### Baselines

```bash
# Self-consistency (k=8)
python scripts/baselines/self_consistency.py \
    --k 8 \
    --data data/test/

# Chain-of-thought
python scripts/baselines/chain_of_thought.py \
    --data data/test/
```
