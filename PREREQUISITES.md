# Prerequisites for LCR Model Execution

---

This document details the complete prerequisites for running LCR models and reproducing paper results.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Operating System](#operating-system)
3. [CUDA Toolkit Installation](#cuda-toolkit-installation)
4. [cuDNN Installation](#cudnn-installation)
5. [Python Environment Setup](#python-environment-setup)
6. [Dependency Installation](#dependency-installation)
7. [Pre-trained Model Dependencies](#pre-trained-model-dependencies)
8. [Environment Verification](#environment-verification)
9. [Common Issues](#common-issues)

---

## Hardware Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 8GB VRAM | NVIDIA A100 40GB / RTX 3090 24GB |
| GPU Compute Capability | 7.0 (V100) | 8.0+ (A100, RTX 30xx/40xx) |
| RAM | 32GB | 64GB |
| Storage | 50GB free | 200GB free |
| CPU | 8 cores | 16+ cores |

### Tested Hardware Configurations

The following configurations have been tested and verified:

1. **Paper Results (Primary)**
   - 2x NVIDIA A100-SXM4-80GB
   - 512GB RAM
   - AMD EPYC 7742 64-Core
   - CUDA 11.8, cuDNN 8.7.0

2. **Consumer Hardware (Verified)**
   - NVIDIA RTX 3090 24GB
   - 64GB RAM
   - Intel i9-12900K
   - CUDA 11.8, cuDNN 8.6.0

3. **Cloud Instances (Verified)**
   - AWS p4d.24xlarge (8x A100)
   - GCP a2-highgpu-8g (8x A100)
   - Azure NC24ads_A100_v4

### GPU Memory Requirements

| Operation | Minimum VRAM | Recommended |
|-----------|--------------|-------------|
| Detector inference | 2GB | 4GB |
| Refinement inference | 8GB | 16GB |
| Classifier (DeBERTa) | 12GB | 16GB |
| Full pipeline | 16GB | 24GB |
| Training (detector) | 8GB | 16GB |
| Training (refinement) | 24GB | 40GB |

---

## Operating System

### Supported

- Ubuntu 20.04 LTS / 22.04 LTS (Recommended)
- CentOS 7.9 / Rocky Linux 8
- Windows 10/11 with WSL2 (Ubuntu 22.04)
- macOS (CPU only, not recommended)

### Not Supported

- Windows native (CUDA compatibility issues)
- ARM-based systems
- Systems without NVIDIA GPU

---

## CUDA Toolkit Installation

### Required Version

**CUDA Toolkit 11.8** (Primary tested version)

Other supported versions: 12.0, 12.1, 12.2 (with potential numerical variations)

### Installation Steps

#### Ubuntu/Debian

```bash
# Remove existing CUDA installations
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
    "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"

# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 11.8
sudo apt-get install cuda-toolkit-11-8

# Set environment variables (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### Verification

```bash
nvcc --version
# Expected: Cuda compilation tools, release 11.8

nvidia-smi
# Should show driver version and CUDA version
```

---

## cuDNN Installation

### Required Version

**cuDNN 8.7.0** or higher for CUDA 11.8

### Installation Steps

1. Download cuDNN from [NVIDIA Developer](https://developer.nvidia.com/cudnn)
   - Requires NVIDIA Developer account
   - Select: cuDNN v8.7.0 for CUDA 11.x

2. Install cuDNN:

```bash
# Extract archive
tar -xzvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz

# Copy files
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*

# Update library cache
sudo ldconfig
```

3. Verify installation:

```bash
cat /usr/local/cuda-11.8/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
# Should show version 8.7.0 or higher
```

---

## Python Environment Setup

### Required Python Version

**Python 3.10.x** (3.10.12 tested)

Python 3.11 may work but has not been extensively tested.

### Virtual Environment Setup

```bash
# Create virtual environment
python3.10 -m venv lcr_env

# Activate environment
source lcr_env/bin/activate  # Linux/macOS
# or
.\lcr_env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Conda Alternative

```bash
# Create conda environment
conda create -n lcr python=3.10.12
conda activate lcr

# Install CUDA toolkit via conda (alternative to system install)
conda install -c conda-forge cudatoolkit=11.8
```

---

## Dependency Installation

### Option 1: Exact Reproducibility (Recommended)

```bash
# Install exact versions used in paper
pip install -r requirements-exact.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### Option 2: Compatible Versions

```bash
# Install compatible version ranges
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

### Post-Installation Verification

```bash
# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Verify transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Full environment check
python scripts/utils/verify_environment.py --strict
```

---

## Pre-trained Model Dependencies

### Base Language Model

The refinement module requires access to LLaMA-3-8B hidden states:

```bash
# Option 1: Download via Hugging Face (requires access approval)
huggingface-cli login
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Meta-Llama-3-8B')"

# Option 2: Use provided extraction script
python scripts/download_models.py --model llama3-8b
```

### Classifier Base Model

DeBERTa-v3-large is downloaded automatically:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/deberta-v3-large")
```

### SpaCy Models (for preprocessing)

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

---

## Environment Verification

Run the complete verification suite before using LCR models:

```bash
# Basic verification
python scripts/utils/verify_environment.py

# Strict verification (recommended for reproducibility)
python scripts/utils/verify_environment.py --strict

# Verbose output with details
python scripts/utils/verify_environment.py --verbose

# Save report to file
python scripts/utils/verify_environment.py --output env_report.json
```

### Expected Output

```
======================================================================
LCR ENVIRONMENT VERIFICATION REPORT
======================================================================
  ✓ Python Version
  ✓ Package: torch
  ✓ Package: transformers
  ✓ Package: numpy
  ✓ CUDA
  ✓ cuDNN
  ✓ GPU 0: NVIDIA A100-SXM4-80GB
  ✓ BLAS/LAPACK Configuration
  ✓ PyTorch Backends
  ✓ Checkpoint: checkpoints/detector/detector_llama3_8b.pt
  ...

Summary: 15/15 OK, 0 warnings, 0 errors

✓ Environment is fully compatible
======================================================================
```

---

## Common Issues

### Issue 1: CUDA Version Mismatch

**Symptom**: `CUDA error: no kernel image is available for execution on the device`

**Solution**:
```bash
# Check actual CUDA version
nvidia-smi  # Shows driver CUDA version
nvcc --version  # Shows toolkit version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in config files
2. Enable gradient checkpointing
3. Use mixed precision (FP16)
4. Use smaller classifier variant (DistilBERT or TinyBERT)

```python
# Enable memory optimizations
import torch
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Issue 3: Transformers Tokenization Differences

**Symptom**: Slightly different model outputs than expected

**Cause**: Different tokenizer versions produce different token sequences

**Solution**: Use exact transformers version:
```bash
pip install transformers==4.35.0 tokenizers==0.15.0
```

### Issue 4: NumPy ABI Incompatibility

**Symptom**: `ImportError: numpy.core.multiarray failed to import`

**Solution**:
```bash
pip uninstall numpy
pip install numpy==1.24.3
```

### Issue 5: cuDNN Not Found

**Symptom**: `Could not load library libcudnn_cnn_infer.so.8`

**Solution**:
```bash
# Check cuDNN installation
ldconfig -p | grep cudnn

# If missing, reinstall cuDNN and update library path
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
sudo ldconfig
```

---

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)

For issues not covered here, please open an issue on the repository.
