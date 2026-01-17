# When Does Latent Refinement Suffice? Identifying Verification Boundaries in RAG Conflict Resolution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

**Anonymous ACL 2026 Submission**

> When retrieved documents disagree, RAG systems hallucinate at nearly double their baseline rate. We introduce LCR (Latent Conflict Refinement), an 8.4M-parameter module that identifies when lightweight latent-space intervention suffices versus when explicit verification is required.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Data](#data)
7. [Pre-trained Checkpoints](#pre-trained-checkpoints)
8. [Training](#training)
9. [Evaluation](#evaluation)
10. [Configuration System](#configuration-system)
11. [Hardware Requirements](#hardware-requirements)
12. [Reproducibility](#reproducibility)
13. [Limitations](#limitations)
14. [Ethical Considerations](#ethical-considerations)
15. [AI Assistant Disclosure](#ai-assistant-disclosure)
16. [Citation](#citation)
17. [License](#license)

---

## Overview

Retrieval-augmented generation (RAG) systems hallucinate at nearly double their baseline rate when retrieved documents disagree. Current verification methods (self-consistency, chain-of-thought) improve accuracy but incur 3-5x token overhead, creating deployment barriers for latency-sensitive applications.

**Latent Conflict Refinement (LCR)** is an 8.4M-parameter module (2.1M detector + 6.3M refinement) that refines hidden states when conflicts are detected. Through controlled experiments across four conflict types, we establish *boundaries* for when latent intervention suffices versus when explicit verification is required:

- **Shallow conflicts** (temporal and numerical disagreements, 46% of cases): LCR matches self-consistency accuracy (72.4% vs. 73.1%, p=0.42) while adding only 6% tokens versus 287% for SC-8
- **Deep conflicts** (entity and semantic, 54% of cases): LCR underperforms by 7-9 points, confirming these require explicit verification

For practical deployment, we provide a **hybrid system** (312M total: 8.4M LCR + 304M DeBERTa classifier) that routes queries based on predicted conflict type, achieving 98.7% of best-case accuracy at 28% of the token cost (81% overhead vs. 287% for SC-8).

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LCR SYSTEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Query + Documents                                              │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────────────────────┐                       │
│   │     Frozen LLM Encoder              │                       │
│   │     (Llama-3-8B, layers 0-15)       │                       │
│   └────────────────┬────────────────────┘                       │
│                    │ h^{(L/2)}                                   │
│                    ▼                                             │
│   ┌─────────────────────────────────────┐                       │
│   │     CONFLICT DETECTOR (2.1M params) │                       │
│   │     MeanPool → FC(4096→512) →       │                       │
│   │     ReLU → FC(512→1) → Sigmoid      │                       │
│   └────────────────┬────────────────────┘                       │
│                    │ p_conflict                                  │
│                    ▼                                             │
│            ┌───────┴───────┐                                    │
│            │   p < τ=0.6   │                                    │
│            └───────┬───────┘                                    │
│         NO ◄───────┴───────► YES                                │
│          │                   │                                   │
│          │    ┌──────────────┴──────────────┐                   │
│          │    │   REFINEMENT MODULE (6.3M)  │                   │
│          │    │   for t = 1 to T_max:       │                   │
│          │    │     z = W_up @ h            │                   │
│          │    │     Δh = W_down @ ReLU(z)   │                   │
│          │    │     h = h + α·Δh            │                   │
│          │    │     if ‖Δh‖ < ε: break      │                   │
│          │    └──────────────┬──────────────┘                   │
│          │                   │ h_refined                         │
│          └─────────► ┌───────▼───────┐                          │
│                      │ Frozen Decoder │                          │
│                      │ (layers 16-31) │                          │
│                      └───────┬───────┘                          │
│                              ▼                                   │
│                           Answer                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results

### Main Performance Comparison

| Conflict Type | LCR Accuracy | Self-Consistency (k=8) | p-value | Token Overhead |
|---------------|--------------|------------------------|---------|----------------|
| **L1 (Temporal)** | **72.4%** ±1.6 | 73.1% ±1.6 | p=0.42 | +6% vs +287% |
| **L2 (Numerical)** | **70.1%** ±1.8 | 71.2% ±1.8 | p=0.38 | +6% vs +287% |
| L3 (Entity) | 59.7% ±1.8 | 66.5% ±1.6 | p<0.01 | Requires SC |
| L4 (Semantic) | 57.8% ±2.1 | 66.9% ±1.8 | p<0.001 | Requires SC |

**Key Finding**: LCR matches self-consistency on shallow conflicts (L1-L2) with statistical equivalence (p > 0.35), using 98% fewer additional tokens. On deep conflicts (L3-L4), LCR underperforms by 7-9 points, confirming these require explicit verification.

### System Comparison

| Configuration | L1 Acc. | L2 Acc. | L3 Acc. | L4 Acc. | Overall | Token Overhead |
|---------------|---------|---------|---------|---------|---------|----------------|
| Standard RAG | 61.4% | 59.2% | 55.8% | 58.1% | 58.4% | 100% (baseline) |
| Self-Consistency-8 | 73.1% | 71.2% | 66.5% | 66.9% | 69.2% | 387% |
| **LCR (Ours)** | **72.4%** | **70.1%** | 59.7% | 57.8% | 64.8% | **106%** |
| Hybrid Routing | 72.4% | 70.1% | 66.2% | 66.1% | **68.4%** | 181% |

---

## Repository Structure

```
lcr-rag-verification/
├── src/                          # Core implementation
│   ├── models/                   # ConflictDetector, RefinementModule, LCRSystem
│   ├── training/                 # Training scripts for each component
│   ├── data/                     # Dataset loader
│   └── utils/                    # Calibration, hardware compatibility
├── configs/                      # Hierarchical YAML configuration
│   ├── base/                     # Architecture defaults
│   ├── environments/             # Hardware-specific (A100, RTX3090, CPU)
│   └── experiments/              # Ablations and sweeps
├── checkpoints/                  # Pre-trained models
│   ├── detector/                 # 2.1M params, ~8.4MB
│   ├── refinement/               # 6.3M params, ~25MB
│   └── classifier/               # DeBERTa (304M), DistilBERT (66M), TinyBERT (14M)
├── data/                         # Benchmark datasets (14.6K examples)
│   ├── train/                    # 10,200 training examples
│   ├── dev/                      # 1,460 validation examples
│   ├── test/                     # 2,940 test examples
│   └── natural/                  # Human-curated natural conflicts
├── scripts/                      # Training, evaluation, and analysis utilities
├── examples/                     # Quick start and batch inference demos
├── notebooks/                    # Jupyter notebooks for analysis
├── docs/                         # Extended documentation
├── tests/                        # Unit tests
├── annotation/                   # Annotation guidelines and IAA metrics
└── results/                      # Experimental results (JSON)
```

---

## Installation

### Prerequisites

- Python 3.9+ (tested with 3.10.12)
- PyTorch 2.1+ with CUDA 11.8+
- 24GB+ GPU VRAM (for full pipeline)
- 64GB+ system RAM

### Environment Setup

```bash
# Create conda environment (recommended)
conda create -n lcr python=3.10 -y
conda activate lcr

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.8+)
pip install -r requirements-gpu.txt

# For exact reproduction of paper results
pip install -r requirements-exact.txt

# Verify installation
python scripts/utils/verify_environment.py
```

### Expected Verification Output

```
══════════════════════════════════════════════════════════════════
LCR ENVIRONMENT VERIFICATION REPORT
══════════════════════════════════════════════════════════════════
  ✓ Python Version: 3.10.12
  ✓ Package torch: 2.1.0+cu118
  ✓ Package transformers: 4.35.0
  ✓ CUDA Toolkit: 11.8
  ✓ GPU 0: NVIDIA A100-SXM4-80GB
══════════════════════════════════════════════════════════════════
Summary: 24/24 checks passed
Status: READY FOR REPRODUCTION
══════════════════════════════════════════════════════════════════
```

### Download Pre-trained Models

```bash
# Download trained LCR checkpoints
python scripts/download_models.py --all

# Download base LLM (requires Hugging Face authentication)
huggingface-cli login
python scripts/download_models.py --model llama3-8b --precision fp16
```

---

## Quick Start

### Basic Usage

```python
from src.models import LCRSystem

# Initialize system with pre-trained checkpoints
lcr = LCRSystem(
    base_model_name="meta-llama/Llama-3-8B-Instruct",
    detector_path="checkpoints/detector/",
    refinement_path="checkpoints/refinement/",
    use_hybrid=False
)

# Example: conflicting documents about Eiffel Tower completion
query = "When was the Eiffel Tower completed?"
documents = [
    "The Eiffel Tower was completed in 1889.",
    "Construction finished in March 1889.",
    "The tower was built in 1887.",  # Minority conflict
    "Opened for the 1889 World's Fair."
]

# Resolve conflict
result = lcr.generate_with_refinement(query, documents)
print(f"Answer: {result['answer']}")           # Output: 1889
print(f"Conflict detected: {result['conflict_detected']}")
print(f"Conflict probability: {result['conflict_prob']:.3f}")
```

### Running the Demo

```bash
# Run quick start example
python examples/quick_start.py

# Run with custom query
python examples/quick_start.py --query "Who is the CEO of Twitter?"

# Batch inference
python examples/batch_inference.py --input data/test/l1_temporal.jsonl --output results/
```

See [examples/README.md](examples/README.md) for more usage examples.

---

## Data

### Benchmark Overview

Our benchmark contains **14,600 labeled examples** across four conflict levels:

| Level | Type | Total | Train | Dev | Test | Natural % |
|-------|------|-------|-------|-----|------|-----------|
| L1 | Temporal | 4,070 | 2,849 | 407 | 814 | 62% |
| L2 | Numerical | 3,600 | 2,520 | 360 | 720 | 48% |
| L3 | Entity | 3,400 | 2,380 | 340 | 680 | 71% |
| L4 | Semantic | 3,530 | 2,451 | 353 | 726 | 83% |
| **Total** | | **14,600** | **10,200** | **1,460** | **2,940** | **66%** |

Split ratio: 70% train / 10% dev / 20% test. Composition: 66% natural (from NaturalQuestions/HotpotQA with web retrieval), 34% synthetic augmentation. Inter-annotator agreement: κ = 0.83.

### Conflict Taxonomy

- **L1 (Temporal)**: Documents provide different dates for the same event. Resolution: majority voting over temporal expressions. Complexity: C₁ (constant); ~92% are pure aggregation, ~8% require timezone/relative date reasoning.
- **L2 (Numerical)**: Documents provide different quantities. Resolution: majority voting for discrete values. Complexity: C₁ (constant); 94% involve exact matches, ~6% require unit conversion.
- **L3 (Entity)**: Documents attribute facts to different entities where recency matters. Resolution requires multi-step reasoning (extract, compare, select). Complexity: C₂.
- **L4 (Semantic)**: Documents make incompatible claims requiring logical inference. Complexity: Cₙ, n ≥ 3.

### Data Format

Each example is stored in JSONL format:

```json
{
  "id": "l1_temporal_001",
  "query": "When was the Eiffel Tower completed?",
  "documents": [
    {"doc_id": "d1", "content": "The Eiffel Tower was completed in 1889.", "source": "wikipedia"},
    {"doc_id": "d2", "content": "The tower was built in 1887.", "source": "trivia_site"}
  ],
  "conflict_label": true,
  "conflict_type": "L1",
  "gold_answer": "1889",
  "metadata": {"source_dataset": "NQ", "is_augmented": false, "annotator_agreement": 0.95}
}
```

### Data Acquisition

```bash
# Download and verify datasets
python scripts/download_datasets.py --split all

# Dataset statistics
python -c "from src.data import LCRDataModule; dm = LCRDataModule('data/'); dm.print_statistics()"
```

---

## Pre-trained Checkpoints

### Available Models

| Component | Size | Parameters | Performance |
|-----------|------|------------|-------------|
| Detector | ~8.4 MB | 2.1M | F1=0.874 |
| Refinement | ~25 MB | 6.3M | See paper |
| Classifier (DeBERTa) | ~1.2 GB | 304M | Macro F1=0.888 |
| Classifier (DistilBERT) | ~300 MB | 66M | Routing F1=0.847 |

### Loading Checkpoints

```python
from src.models import ConflictDetector, RefinementModule, TaxonomyClassifier

# Load individual components
detector = ConflictDetector.from_pretrained("checkpoints/detector/")
refinement = RefinementModule.from_pretrained("checkpoints/refinement/")
classifier = TaxonomyClassifier.from_pretrained("checkpoints/classifier/")

# Or load complete system
from src.models import LCRSystem
lcr = LCRSystem(
    detector_path="checkpoints/detector/",
    refinement_path="checkpoints/refinement/",
    classifier_path="checkpoints/classifier/",
    use_hybrid=True
)
```

### Checkpoint Verification

```bash
# Verify checkpoint integrity
python scripts/utils/verify_checksums.py --models

# Verify parameter counts
python -c "
import torch
state = torch.load('checkpoints/detector/demo_detector.pt', map_location='cpu')
print(f'Detector parameters: {sum(p.numel() for p in state.values()):,}')
"
```

---

## Training

Training proceeds in three sequential phases. Core LCR training (detector + refinement) takes ~6 hours on 4×A100 GPUs.

### Phase 1: Training the Conflict Detector

```bash
python -m src.training.train_detector \
    --config configs/detector_config.yaml \
    --data_dir data/ \
    --output_dir checkpoints/detector/ \
    --seed 42

# Key hyperparameters:
# - Learning rate: 2e-5
# - Batch size: 32 (effective: 64 with gradient accumulation)
# - Epochs: 3
# - Optimizer: AdamW with cosine scheduler
```

### Phase 2: Training the Refinement Module

```bash
python -m src.training.train_refinement \
    --config configs/refinement_config.yaml \
    --detector_checkpoint checkpoints/detector/best_model.pt \
    --data_dir data/ \
    --output_dir checkpoints/refinement/ \
    --seed 42

# Key hyperparameters:
# - Learning rate: 5e-5
# - Batch size: 16 (effective: 64 with gradient accumulation)
# - Epochs: 5
# - Training examples: 10,200
# - Alpha (step size): 0.3
# - T_max (iterations): 3
# - Epsilon (convergence): 0.01
# - Loss: L_CE + λ₁*L_detect + λ₂*L_KL (λ₁=1.0, λ₂=0.005)
```

### Phase 3: Training the Taxonomy Classifier (for hybrid system)

```bash
python -m src.training.train_classifier \
    --config configs/classifier_config.yaml \
    --data_dir data/ \
    --output_dir checkpoints/classifier/ \
    --seed 42

# Key hyperparameters:
# - Base model: microsoft/deberta-v3-large
# - Learning rate: 1e-5
# - Batch size: 16
# - Epochs: 5
```

### Hybrid Routing Evaluation

```bash
python scripts/evaluation/eval_hybrid.py \
    --classifier_checkpoint checkpoints/classifier/ \
    --confidence_threshold 0.7 \
    --output_dir results/hybrid/
```

### Statistical Significance Testing

```bash
python scripts/analysis/compute_significance.py \
    --results_dir results/ \
    --baseline self-consistency \
    --method bootstrap \
    --n_resamples 10000 \
    --output results/significance_tests.json
```

---

## Configuration System

### Hierarchical Configuration

The configuration system uses 4 levels of inheritance:

1. **Global defaults** (hardcoded in source)
2. **Base configs** (`configs/base/*.yaml`) - Architecture defaults
3. **Environment configs** (`configs/environments/*.yaml`) - Hardware-specific settings
4. **Experiment configs** (`configs/experiments/**/*.yaml`) - Ablation variants

### Key Configuration Files

| File | Purpose |
|------|---------|
| `configs/detector_config.yaml` | Detector training settings |
| `configs/refinement_config.yaml` | Refinement module settings |
| `configs/classifier_config.yaml` | Taxonomy classifier settings |
| `configs/training_hyperparameters.yaml` | Complete hyperparameter reference |

### Command-Line Overrides

```bash
# Override config values from command line
python -m src.training.train_refinement \
    --config configs/refinement_config.yaml \
    --config.model.alpha 0.4 \
    --config.training.learning_rate 1e-4
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection_threshold` (τ) | 0.6 | Binary conflict detection threshold |
| `alpha` (α) | 0.3 | Refinement step size |
| `T_max` | 3 | Maximum refinement iterations |
| `epsilon` (ε) | 0.01 | Convergence threshold |
| `confidence_threshold` (τ_c) | 0.7 | Cascade routing threshold |

---

## Hardware Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 16GB VRAM | 24GB+ VRAM |
| System RAM | 32GB | 64GB |
| Storage | 100GB free | 200GB free |
| CPU | 8 cores | 16+ cores |

### Configuration-Specific Requirements

| Configuration | GPU Memory | Training Time | Inference Speed |
|---------------|------------|---------------|-----------------|
| Full (4×A100 80GB) | 320 GB total | ~6 hours | 184 q/s (batch=32) |
| Single A100 | 80 GB | ~24 hours | 46 q/s |
| RTX 3090/4090 | 24 GB | ~36 hours | 28 q/s (SC: 28 q/s) |
| CPU (debug only) | - | N/A | <1 q/s |

Note: At batch size 32, LCR achieves 184 q/s vs. 28 q/s for self-consistency.

### GPU Memory by Operation

| Operation | FP32 VRAM | FP16 VRAM | Notes |
|-----------|-----------|-----------|-------|
| Detector inference | 4 GB | 2 GB | Excludes base LLM |
| Refinement inference | 12 GB | 8 GB | Per-token refinement |
| Full pipeline | 32 GB | 24 GB | With Llama-3-8B |
| Refinement training | 48 GB | 32 GB | BPTT through iterations |

---

## Reproducibility

### Complete Reproduction Protocol

```bash
# Step 1: Environment Setup
conda activate lcr
python scripts/utils/verify_environment.py --strict

# Step 2: Data Preparation
python scripts/download_datasets.py --split all

# Step 3: Training
python -m src.training.train_detector --config configs/detector_config.yaml
python -m src.training.train_refinement --config configs/refinement_config.yaml
python -m src.training.train_classifier --config configs/classifier_config.yaml
```

### Statistical Testing

- Method: Paired bootstrap test with 10,000 resamples
- Multiple comparison correction: Holm-Bonferroni
- Confidence intervals: 95% bootstrap CI

### Expected Variance

Due to GPU non-determinism:

| Metric | Paper Value | Acceptable Range |
|--------|-------------|------------------|
| L1 Accuracy | 72.4% | ±1.6% |
| L2 Accuracy | 70.1% | ±1.8% |
| Detector F1 | 0.874 | ±0.02 |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for detailed reproduction guide.

---

## Limitations

This work has several important limitations that practitioners and researchers should be aware of:

### Methodological Limitations

1. **Conflict Type Coverage**: Our taxonomy covers four conflict types (temporal, numerical, entity, semantic), but real-world RAG systems may encounter additional conflict patterns not represented in our benchmark (e.g., modal conflicts, conditional statements, implicit contradictions).

2. **Shallow vs. Deep Conflict Boundary**: While we demonstrate that LCR is effective for L1-L2 conflicts and less effective for L3-L4 conflicts, the boundary between "shallow" and "deep" conflicts is not always clear-cut. Some conflicts may exhibit characteristics of multiple levels.

3. **Base Model Dependency**: Results are obtained using Llama-3-8B. Performance characteristics may differ with other base models, particularly those with different hidden dimensions or architectural choices.

4. **English-Only Evaluation**: All experiments are conducted on English-language data. Cross-lingual transfer and performance on other languages remains untested.

### Practical Limitations

5. **Latency Trade-offs**: While LCR reduces token overhead compared to self-consistency, it adds computational overhead for conflict detection and refinement, which may be significant for latency-critical applications.

6. **Calibration Sensitivity**: The detector's probability outputs require careful calibration. Performance may degrade if deployed in domains significantly different from the training distribution.

7. **Hybrid Routing Complexity**: The full hybrid system (LCR + classifier) adds architectural complexity and requires maintaining multiple model components.

### Evaluation Limitations

8. **Benchmark Composition**: While 66% of our benchmark consists of natural conflicts, 34% are synthetically augmented. Synthetic conflicts may not fully capture the distribution of real-world retrieval conflicts.

9. **Static Document Sets**: Our evaluation uses fixed document sets. Dynamic retrieval scenarios with varying document quality and relevance are not evaluated.

10. **Single-Answer Focus**: We focus on factoid QA with single correct answers. Open-ended questions or tasks requiring synthesis across conflicting sources are out of scope.

---

## Ethical Considerations

### Intended Use

LCR is designed to improve the factual reliability of RAG systems by detecting and resolving conflicts in retrieved documents. The intended use cases include:

- Factual question answering systems
- Knowledge-intensive NLP applications
- Information retrieval systems requiring high accuracy

### Potential Risks and Mitigations

1. **Overconfidence in Resolved Answers**: Users may over-trust LCR's conflict resolution, particularly for L3-L4 conflicts where performance is limited. We recommend:
   - Always routing L3-L4 conflicts to explicit verification methods
   - Providing confidence scores to end users
   - Implementing human-in-the-loop verification for high-stakes applications

2. **Bias Propagation**: LCR inherits biases present in the base LLM and training data. Majority-vote resolution may systematically favor common misconceptions over minority correct answers.

3. **Adversarial Manipulation**: An adversary who can influence retrieved documents could potentially exploit the conflict resolution mechanism. Robust retrieval pipelines and source verification are essential complements to LCR.

### Data and Privacy

- Our benchmark is derived from publicly available datasets (NaturalQuestions, HotpotQA) with no personally identifiable information
- All human annotations were conducted following institutional guidelines
- Inter-annotator agreement (κ = 0.83) is reported for transparency

### Environmental Impact

- Full training requires approximately 6 GPU hours on 4×A100 (estimated 12 kWh, ~5 kg CO2e)
- Inference is designed to be efficient (6% token overhead vs. 287% for self-consistency)
- We provide pre-trained checkpoints to reduce unnecessary retraining

---

## AI Assistant Disclosure

In accordance with ACL 2026 guidelines on AI-assisted research:

- Code development was assisted by AI coding tools for boilerplate generation and documentation
- All scientific contributions, experimental design, and analysis were conducted by the authors
- AI assistants were not used for writing the paper text or generating experimental results
- All code has been reviewed and validated by the authors

---

## Citation

```bibtex
@inproceedings{anonymous2026lcr,
  title={When Does Latent Refinement Suffice? Identifying Verification Boundaries in RAG Conflict Resolution},
  author={Anonymous},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year={2026},
  note={Anonymous submission}
}
```

*Citation will be updated upon acceptance.*

---

## License

This code is released under the [MIT License](LICENSE).

---

<p align="center">
<em>This is an anonymous submission to ACL 2026. For reproducible results, use provided checkpoints and follow the reproduction protocol exactly.</em>
</p>
