# LCR Source Code

This directory contains the core implementation of the Latent Conflict Resolution (LCR) system.

## Directory Structure

```
src/
├── models/           # Model implementations
│   ├── conflict_detector.py      # Binary conflict detection (2.1M params)
│   ├── refinement_module.py      # Iterative refinement (6.3M params)
│   ├── taxonomy_classifier.py    # 4-way conflict classification
│   ├── lcr_system.py            # End-to-end pipeline
│   └── legacy/                  # Deprecated model versions
├── data/             # Data loading and processing
│   └── dataset_loader.py        # Unified data loading
└── training/         # Training scripts
    ├── train_detector.py
    ├── train_refinement.py
    ├── train_classifier.py
    └── losses.py               # Custom loss functions

```

## Model Overview

### Conflict Detector
- Architecture: 2-layer MLP (4096 → 512 → 1)
- Parameters: 2,098,177
- Purpose: Binary classification of conflict presence

### Refinement Module
- Architecture: Bottleneck MLP (4096 → 768 → 4096)
- Parameters: 6,296,320
- Hyperparameters: α=0.3, T_max=3, ε=0.01
- Purpose: Iterative latent representation refinement

### Taxonomy Classifier
- Base: DeBERTa-v3-large (recommended)
- Alternatives: DistilBERT, TinyBERT
- Purpose: Classify conflict into L1-L4 categories

## Quick Start

```python
from src.models.lcr_system import LCRSystem

# Initialize system
lcr = LCRSystem(
    detector_path="checkpoints/detector/detector_llama3_8b.pt",
    refinement_path="checkpoints/refinement/refinement_llama3_8b.pt",
    classifier_path="checkpoints/classifier/classifier_deberta_v3_large.pt"
)

# Process query with contexts
result = lcr.resolve(
    query="What year was the company founded?",
    contexts=["Founded in 2018...", "Established in 2019..."]
)
```

## Dependencies

Core requirements:
- PyTorch >= 2.1.0
- Transformers >= 4.35.0
- NumPy >= 1.24.0

See `requirements.txt` for full list.

## Legacy Code

The `legacy/` subdirectory contains deprecated implementations:
- `detector_v0.py`: Original 3-layer detector (replaced due to overfitting)
- `refinement_v0.py`: Additive refinement (replaced by interpolation approach)
