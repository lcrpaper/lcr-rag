# Reproducibility Guide

This document provides comprehensive instructions for reproducing the results in:

**"When Does Latent Refinement Suffice? Identifying Verification Boundaries in RAG Conflict Resolution"**

---

## ⚠️ Critical: Read This First

Reproducing deep learning research results requires **exact environment matching**. Our experiments involve:
- Specific CUDA/cuDNN versions affecting numerical precision
- Version-locked dependencies with known compatibility
- Hardware-specific optimizations and behaviors
- Calibrated probability outputs sensitive to environment

**Attempting reproduction without following these steps carefully may yield different results.**

---

## Prerequisite Checklist

Before attempting reproduction, complete ALL of the following:

- [ ] Read `PREREQUISITES.md` completely
- [ ] Verify hardware meets minimum requirements
- [ ] Install CUDA 11.8 and cuDNN 8.7+
- [ ] Create isolated Python 3.10 environment
- [ ] Install exact dependency versions
- [ ] Run environment verification script
- [ ] Validate hardware compatibility
- [ ] Download and verify checkpoint integrity
- [ ] Run calibration validation on loaded models

---

## Step-by-Step Reproduction Guide

### Step 1: Environment Setup (Critical)

```bash
# Create isolated environment
python3.10 -m venv lcr_reproduce
source lcr_reproduce/bin/activate

# Install EXACT versions (required for reproducibility)
pip install -r requirements-exact.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Verify environment
python scripts/utils/verify_environment.py --strict
```

**Expected output:**
```
======================================================================
LCR ENVIRONMENT VERIFICATION REPORT
======================================================================
  ✓ Python Version: 3.10.x
  ✓ Package: torch (2.1.0+cu118)
  ✓ Package: transformers (4.35.0)
  ✓ Package: numpy (1.24.3)
  ✓ CUDA (11.8)
  ✓ cuDNN (8700+)
  ✓ GPU Hardware
  ...
Summary: X/X OK, 0 warnings, 0 errors
✓ Environment is fully compatible
======================================================================
```

**Do not proceed if any errors are reported.**

### Step 2: Hardware Validation

```bash
# Check hardware compatibility
python -c "from src.utils.hardware_compatibility import HardwareManager; m = HardwareManager(); m.validate_hardware(); m.print_hardware_report()"
```

**Minimum requirements:**
- GPU: Compute capability >= 7.0 (V100, T4, RTX 20xx+)
- VRAM: >= 8GB (16GB+ recommended)
- RAM: >= 32GB

**Paper results obtained on:**
- 2x NVIDIA A100-SXM4-80GB
- CUDA 11.8, cuDNN 8.7.0
- 512GB system RAM

### Step 3: Data Preparation

```bash
# Download all datasets
python scripts/download_datasets.py --all

### Step 4: Model Loading (Use Proper Infrastructure)

**DO NOT use direct `torch.load()`.** Use the validated loader:

```python
from src.models.model_loader import LCRModelLoader

# Initialize with all validations enabled
loader = LCRModelLoader(
    checkpoint_dir="checkpoints/",
    strict_validation=True,
    calibration_check=True,
    verify_integrity=True,
    precision="fp16"
)

# Validate environment (required before loading)
env_report = loader.validate_environment()
assert env_report.fully_compatible, "Environment check failed"

# Load models with validation
detector = loader.load_detector()
refinement = loader.load_refinement()
classifier = loader.load_classifier()
```

### Step 5: Calibration Validation

After loading, verify models are calibrated correctly:

```python
from src.utils.calibration_validator import CalibrationValidator

validator = CalibrationValidator(strict=True)

# Validate each component
det_valid, det_report = validator.validate_detector(detector)
ref_valid, ref_report = validator.validate_refinement(refinement)
cls_valid, cls_report = validator.validate_classifier(classifier)

if not (det_valid and ref_valid and cls_valid):
    print("WARNING: Calibration drift detected!")
    print("Results may differ from paper values.")
```

### Step 6: Set Reproducibility Flags

```python
import torch
import random
import numpy as np

def set_reproducibility(seed=42):
    """Set all seeds and determinism flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 2.0
    torch.use_deterministic_algorithms(True)

    # Environment variable for CUDA determinism
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

set_reproducibility(42)
```

### Step 7: Run Evaluation

```bash
# Main results (Table 1)
python scripts/evaluation/eval_main_results.py \
    --checkpoint-dir checkpoints/ \
    --test-set data/test/ \
    --seed 42 \
    --strict-reproducibility

# With full validation
python scripts/evaluation/eval_main_results.py \
    --validate-environment \
    --validate-calibration \
    --save-report results/reproduction_report.json
```

---

## Training Reproduction

### Conflict Detector

```bash
python src/training/train_detector.py \
    --config configs/detector_config.yaml \
    --seed 42 \
    --deterministic \
    --output checkpoints/detector_reproduced/

# Expected: ~45 minutes on A100
```

### Refinement Module

```bash
python src/training/train_refinement.py \
    --config configs/refinement_config.yaml \
    --seed 42 \
    --deterministic \
    --output checkpoints/refinement_reproduced/

# Expected: ~6 hours on 2x A100
```

### Taxonomy Classifier

```bash
python src/training/train_classifier.py \
    --config configs/classifier_deberta_config.yaml \
    --seed 42 \
    --deterministic \
    --output checkpoints/classifier_reproduced/

# Expected: ~4 hours on A100
```

---

## Troubleshooting

### Environment Verification Fails

**Issue:** `EnvironmentValidationError`

**Solutions:**
1. Check CUDA version: `nvcc --version` (needs 11.8+)
2. Check cuDNN: `ldconfig -p | grep cudnn`
3. Reinstall PyTorch with correct CUDA:
   ```bash
   pip uninstall torch
   pip install torch==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/evaluation/eval_main_results.py --batch-size 8

# Or use gradient checkpointing for training
python src/training/train_refinement.py --gradient-checkpointing
```

---

## Known Limitations

1. **Hardware Variance:** Results may vary ±0.5% across different GPU architectures
2. **CUDA Non-determinism:** Some CUDA operations have inherent randomness
3. **Library Updates:** Future library versions may break reproducibility
4. **Floating Point:** Different compilers may produce slight differences

---

## Support

For reproducibility issues:
1. First, verify all steps in this guide were followed exactly
2. Check the troubleshooting section
3. Review `PREREQUISITES.md` for environment setup
4. Open a GitHub issue with:
   - Output of `python scripts/utils/verify_environment.py --verbose`
   - Your hardware configuration
   - Exact error messages or result differences
