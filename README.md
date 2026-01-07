# Latent Refinement for RAG Conflict Resolution

## Overview

This repository constitutes the official artifact release for the LCR (Latent Conflict Refinement) framework, a test-time intervention methodology for resolving evidential conflicts in retrieval-augmented generation systems. The framework operates through lightweight latent-space manipulation at intermediate transformer layers, achieving competitive performance with computationally expensive verification methods on shallow conflict types (temporal, numerical) while incurring 48x lower token overhead.

**Key Result**: Achieves parity with self-consistency verification on L1/L2 conflicts (p > 0.35) while using only 6% additional tokens vs 287% for explicit reasoning.

| Conflict Type | LCR Accuracy | Self-Consistency | 
|---------------|--------------|------------------|
| L1 (Temporal) | 72.4% | 73.1% |
| L2 (Numerical) | 70.1% | 71.2% | 
| L3 (Entity) | 59.7% | 66.5% | 
| L4 (Semantic) | 57.8% | 66.9% | 


## Quick Start

### Requirements

- Python 3.10
- CUDA 11.8
- GPU with 16GB+ VRAM (24GB recommended)
- PyTorch 2.1.0

### Installation

```bash
# Clone and setup
git clone https://github.com/.../lcr-rag.git
cd lcr-rag

# Create environment
python3.10 -m venv lcr_env
source lcr_env/bin/activate  # Linux/Mac
# or: lcr_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### Verify Setup

```bash
python scripts/utils/verify_environment.py
```

## Usage

### Inference with Pre-trained Checkpoints

```bash
python scripts/evaluation/eval_main_results.py \
    --detector-checkpoint checkpoints/detector/detector_llama3_8b.pt \
    --refinement-checkpoint checkpoints/refinement/refinement_llama3_8b.pt \
    --data-dir data/test/
```

### Training (Optional)

```bash
# Phase 1: Detector
python src/training/train_detector.py --config configs/detector_config.yaml

# Phase 2: Refinement
python src/training/train_refinement.py --config configs/refinement_config.yaml

# Phase 3: Classifier
python src/training/train_classifier.py --config configs/classifier_config.yaml
```

## Data

| Split | Examples | Format |
|-------|----------|--------|
| Train | 10,200 | JSONL |
| Dev | 1,460 | JSONL |
| Test | 2,940 | JSONL |

## Repository Structure

```
├── src/models/           # Core implementations
│   ├── conflict_detector.py
│   ├── refinement_module.py
│   └── taxonomy_classifier.py
├── src/training/         # Training scripts
├── configs/              # Configuration files
├── checkpoints/          # Pre-trained weights
├── data/                 # Datasets
└── scripts/              # Utilities
```

## Troubleshooting

**CUDA out of memory**: Reduce batch size or enable `mixed_precision: fp16`

**Results differ from paper**: Run `python scripts/utils/verify_environment.py --strict`

**Dependency issues**: Use `requirements-exact.txt` for reproducibility

## License

MIT License. See [LICENSE](LICENSE) for details.

