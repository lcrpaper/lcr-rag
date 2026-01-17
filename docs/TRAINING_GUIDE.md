# Training Guide

This guide covers training all LCR system components from scratch.

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 24GB VRAM | 80GB VRAM (A100) |
| RAM | 32GB | 64GB |
| Storage | 50GB | 200GB |

### Software Requirements

```bash
# Create environment
conda create -n lcr python=3.10
conda activate lcr

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Training Pipeline

### Step 1: Data Preparation

```bash
# Download and prepare data
python scripts/download_datasets.py

# Build BM25 index (v3 configuration)
python scripts/preprocessing/build_bm25_v3.py \
    --input data/raw/ \
    --output data/intermediate/ \
    --k1 0.9 \
    --b 0.4

# Generate train/dev/test splits
python scripts/preprocessing/generate_splits.py \
    --input data/intermediate/ \
    --output data/
```

### Step 2: Train Conflict Detector

```bash
python src/training/train_detector.py \
    --config configs/base/detector_base.yaml \
    --train_data data/train/conflicts.jsonl \
    --dev_data data/dev/conflicts.jsonl \
    --output_dir checkpoints/detector/ \
    --seed 42
```

**Expected Output**:
```
Epoch 23/30: train_loss=0.121, val_f1=0.875
Early stopping triggered
Best checkpoint: epoch_23_f1_0.875.pt
Final test F1: 0.871
```

**Training Time**: ~45 minutes on A100

### Step 3: Train Taxonomy Classifier

```bash
python src/training/train_classifier.py \
    --config configs/classifier_config.yaml \
    --train_data data/train/conflicts.jsonl \
    --dev_data data/dev/conflicts.jsonl \
    --output_dir checkpoints/classifier/ \
    --model microsoft/deberta-v3-large \
    --seed 42
```

**Expected Output**:
```
Epoch 7/10: train_loss=0.234, val_macro_f1=0.891
Best checkpoint: epoch_7_f1_0.891.pt
Final test macro F1: 0.888
```

**Training Time**: ~3 hours on A100

### Step 4: Train Refinement Module

```bash
python src/training/train_refinement.py \
    --config configs/base/refinement_base.yaml \
    --train_data data/train/conflicts.jsonl \
    --dev_data data/dev/conflicts.jsonl \
    --output_dir checkpoints/refinement/ \
    --base_model meta-llama/Meta-Llama-3-8B \
    --alpha 0.3 \
    --num_iterations 3 \
    --seed 42
```

**Expected Output**:
```
Epoch 15/20: train_loss=0.744, val_accuracy=0.652
Best checkpoint: epoch_15_acc_0.652.pt
Final test accuracy: 0.649
```

**Training Time**: ~6 hours on A100

## Hyperparameter Tuning

### Detector Hyperparameters

| Parameter | Default | Search Range | Notes |
|-----------|---------|--------------|-------|
| learning_rate | 2e-5 | [1e-5, 5e-5] | Lower for stability |
| batch_size | 64 | [32, 128] | Larger improves convergence |
| dropout | 0.1 | [0.05, 0.2] | Regularization |
| hidden_dim | 512 | [256, 1024] | Intermediate layer size |

### Refinement Hyperparameters

| Parameter | Default | Search Range | Notes |
|-----------|---------|--------------|-------|
| alpha | 0.3 | [0.1, 0.5] | **Most sensitive** |
| num_iterations | 3 | [1, 5] | Diminishing returns after 3 |
| bottleneck_dim | 732 | [256, 2048] | ~hidden_dim/5.6 |
| kl_weight | 0.1 | [0.01, 0.5] | Regularization strength |

### Running Hyperparameter Sweep

```bash
# Alpha sweep
python scripts/training/run_hyperparameter_sweep.py \
    --config configs/experiments/ablations/alpha_sweep.yaml \
    --output experiments/runs/alpha_sweep/

# Learning rate sweep
python scripts/training/run_hyperparameter_sweep.py \
    --config configs/experiments/detector_lr_sweep/ \
    --output experiments/runs/lr_sweep/
```

## Multi-GPU Training

For larger models or faster training:

```bash
# Using PyTorch DDP
torchrun --nproc_per_node=4 src/training/train_refinement.py \
    --config configs/environments/cloud_4xa100.yaml \
    ...

# Using DeepSpeed (for very large models)
deepspeed src/training/train_refinement.py \
    --deepspeed configs/deepspeed_config.json \
    ...
```

## Checkpointing and Recovery

### Automatic Checkpointing

Checkpoints are saved every epoch:
```
checkpoints/
├── detector/
│   ├── epoch_1.pt
│   ├── epoch_2.pt
│   └── best_model.pt
└── refinement/
    ├── epoch_1.pt
    └── best_model.pt
```

### Resuming Training

```bash
python src/training/train_refinement.py \
    --resume checkpoints/refinement/epoch_10.pt \
    ...
```

## Validation and Monitoring

### TensorBoard Logging

```bash
# Start TensorBoard
tensorboard --logdir experiments/runs/

# Metrics logged:
# - train/loss, val/loss
# - val/accuracy, val/f1
# - learning_rate
# - gradient_norm
```

### Sanity Checks

```bash
# Quick validation on small subset
python scripts/utils/quick_sanity_test.py \
    --checkpoint checkpoints/refinement/best_model.pt \
    --data data/debug/sanity_10.jsonl
```

## Common Issues

### Out of Memory

```
RuntimeError: CUDA out of memory
```

Solutions:
1. Reduce batch_size
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Use mixed precision: `--fp16`
4. Reduce max_length for classifier

### Overfitting

Symptoms: Train loss decreasing but val loss increasing

Solutions:
1. Increase dropout
2. Add weight decay
3. Reduce model capacity (smaller hidden_dim)
4. Use early stopping (default: patience=5)

### Training Instability

Symptoms: Loss spikes or NaN values

Solutions:
1. Reduce learning rate
2. Add gradient clipping: `--gradient_clip 1.0`
3. Increase warmup steps
4. Check data for anomalies

## Reproducibility

For exact reproduction of paper results:

```bash
# Set all seeds
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python src/training/train_refinement.py \
    --seed 42 \
    --deterministic \
    ...
```

Seeds are set for:
- Python random
- NumPy random
- PyTorch random
- CUDA random (if deterministic=True)
