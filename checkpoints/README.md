# Model Checkpoints

This directory contains trained model checkpoints for the LCR system.

## ⚠️ Important: Before Loading Checkpoints

**DO NOT** attempt to load checkpoints without first completing the following steps:

1. **Environment Verification** (Required)
   ```bash
   python scripts/utils/verify_environment.py --strict
   ```
   All checks must pass before proceeding.

2. **Prerequisites Review**
   - Read `PREREQUISITES.md` in full
   - Ensure CUDA 11.8+ and cuDNN 8.7+ are installed
   - Verify GPU compute capability >= 7.0

3. **Dependency Installation**
   ```bash
   pip install -r requirements-exact.txt --extra-index-url https://download.pytorch.org/whl/cu118
   ```

4. **Hardware Validation**
   ```bash
   python -c "from src.utils.hardware_compatibility import HardwareManager; HardwareManager().print_hardware_report()"
   ```

## Available Checkpoints

### Detector Module
```
checkpoints/detector/
├── detector_llama3_8b.pt              # Main checkpoint (2.1M params)
├── config.json                        # Architecture & hyperparameters
└── final_v2.1_a100_fp16_metadata.json # Training metadata
```

### Refinement Module
```
checkpoints/refinement/
├── refinement_llama3_8b.pt            # Main checkpoint (6.3M params)
├── config.json                        # Architecture & hyperparameters
└── final_v1.3_a100_fp16_metadata.json # Training metadata
```

### Taxonomy Classifier
```
checkpoints/classifier/
├── classifier_deberta_v3_large.pt     # Recommended (304M params)
├── classifier_distilbert_base.pt      # Efficient (66M params)
├── classifier_tinybert.pt             # Lightweight (14M params)
└── config.json                        # All variant configs
```

### Demo Checkpoints
```
checkpoints/demo/
├── detector_demo.pt                   # Smaller demo model
└── refinement_demo.pt                 # Smaller demo model
```

## Proper Checkpoint Loading

**DO NOT use direct `torch.load()` calls.** Use the validated loading infrastructure:

```python
from src.models.model_loader import LCRModelLoader

# Initialize loader with validation
loader = LCRModelLoader(
    checkpoint_dir="checkpoints/",
    strict_validation=True,      # Enforce environment checks
    calibration_check=True,      # Validate probability calibration
    verify_integrity=True,       # Check checkpoint hashes
    device="cuda:0",
    precision="fp16"             # Match training precision
)

# Validate environment FIRST
env_report = loader.validate_environment()
if not env_report.fully_compatible:
    print("Environment incompatible. See PREREQUISITES.md")
    sys.exit(1)

# Load models (validation happens automatically)
detector = loader.load_detector()
refinement = loader.load_refinement()
classifier = loader.load_classifier(variant="deberta_v3_large")

# Or load all at once
models = loader.load_all()
```

### Why Not Direct Loading?

Direct `torch.load()` will:
- Skip environment validation (may cause silent numerical errors)
- Skip calibration validation (may produce incorrect probabilities)
- Skip integrity checks (may load corrupted weights)
- Miss precision/device configuration
- Not apply architecture-specific optimizations

### Calibration Validation

After loading, validate model calibrations:

```python
from src.utils.calibration_validator import CalibrationValidator

validator = CalibrationValidator(strict=True)

# Validate individual models
is_valid, report = validator.validate_detector(detector)
if not is_valid:
    print("Detector calibration drift detected!")
    validator.print_report(report)

# Or validate full system
all_valid, reports = validator.validate_full_system(
    detector=detector,
    refinement=refinement,
    classifier=classifier
)
```

## Checkpoint Metadata

Each checkpoint includes a `*_metadata.json` file with:
- Training configuration (exact hyperparameters)
- Hardware details (GPU type, CUDA version)
- Framework versions (PyTorch, Transformers, NumPy)
- Training curves and final metrics
- Reproducibility seeds and settings
- Calibration reference values

**Example metadata inspection:**
```python
import json
with open("checkpoints/detector/final_v2.1_a100_fp16_metadata.json") as f:
    metadata = json.load(f)

print(f"Trained with PyTorch: {metadata['framework']['pytorch_version']}")
print(f"Training GPU: {metadata['hardware']['device_name']}")
print(f"Expected accuracy: {metadata['validation']['expected_l1_accuracy']}")
```

## Model Performance

| Checkpoint | Size | L1 Acc | L2 Acc | L3 Acc | L4 Acc |
|------------|------|--------|--------|--------|--------|
| detector_llama3_8b | 8.4 MB | 0.978 | 0.967 | 0.943 | 0.912 |
| refinement_llama3_8b | 25 MB | 0.724 | 0.701 | 0.438 | 0.312 |
| classifier_deberta | 1.2 GB | F1: 0.921 | F1: 0.904 | F1: 0.872 | F1: 0.859 |

**Note:** These numbers were obtained with:
- Hardware: 2x NVIDIA A100-SXM4-80GB
- CUDA: 11.8, cuDNN: 8.7.0
- PyTorch: 2.1.0+cu118
- Transformers: 4.35.0

Running on different hardware/software may yield different results.

## Reproducibility Requirements

To reproduce paper results, you MUST:

1. **Use exact environment**
   ```bash
   pip install -r requirements-exact.txt
   ```

2. **Set reproducibility flags**
   ```python
   import torch
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

3. **Use provided data splits**
   ```
   data/train/  - Training set (10,200 examples)
   data/dev/    - Validation set (1,460 examples)
   data/test/   - Test set (2,940 examples)
   ```

4. **Verify calibration**
   ```bash
   python scripts/evaluation/eval_calibration.py --checkpoint checkpoints/
   ```

See `REPRODUCIBILITY.md` in root directory for complete instructions.

## Troubleshooting

### "Checkpoint hash mismatch"
The checkpoint file may be corrupted. Re-download from the official source.

### "Calibration drift detected"
Your environment differs from training environment. Check:
- PyTorch version matches exactly
- CUDA/cuDNN versions match
- Using same precision (fp16)

### "CUDA out of memory"
- Reduce batch size in config
- Use smaller classifier variant (distilbert or tinybert)
- Enable gradient checkpointing

### "Numerical differences from paper"
Some variance is expected across hardware. Acceptable ranges:
- Accuracy: ±0.5%
- F1 Score: ±0.005

Larger differences indicate environment issues.

## External Hosting

For large classifier checkpoints, see:
- `CHECKPOINT_HOSTING_GUIDE.md` for upload instructions
- Checkpoints also available via Hugging Face Hub: `lcr-research/lcr-checkpoints`
