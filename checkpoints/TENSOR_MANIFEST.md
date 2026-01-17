# Tensor Manifest

**Document Type**: Pure Tensor Dictionary Specification
**Classification**: Implementation-Neutral Checkpoint Format

---

## Preamble

This document specifies the tensor contents of each checkpoint file. Checkpoints are provided as **pure tensor dictionaries** without embedded architectural metadata.

**Design Principle**: By decoupling weights from architecture specifications, we enable:
- Weight transfer across architectural variants
- Ablation studies with modified configurations
- Layer-wise analysis without metadata assumptions
- Framework-agnostic loading

**User Responsibility**: The user must construct their architecture independently and bind these tensors to the appropriate computational nodes.

---

## Checkpoint Index

| Checkpoint | File | Format | Tensor Count | Total Parameters |
|------------|------|--------|--------------|------------------|
| Detector | `detector/detector_llama3_8b.pt` | PyTorch | 4 | 2,098,177 |
| Refinement | `refinement/refinement_llama3_8b.pt` | PyTorch | 2 | 5,996,544 |
| Classifier | `classifier/classifier_deberta_v3_large.pt` | PyTorch | Variable | ~304M |

---

## 1. Detector Tensors

### File Location
```
checkpoints/detector/detector_llama3_8b.pt
```

### Tensor Enumeration

| Key | Shape | Dtype | Parameters | Description |
|-----|-------|-------|------------|-------------|
| `fc1.weight` | (512, 4096) | float32/float16 | 2,097,152 | First projection matrix |
| `fc1.bias` | (512,) | float32/float16 | 512 | First projection bias |
| `fc2.weight` | (1, 512) | float32/float16 | 512 | Second projection matrix |
| `fc2.bias` | (1,) | float32/float16 | 1 | Second projection bias |

**Total**: 2,098,177 parameters

### Shape Invariants

```
ASSERT fc1.weight.shape[0] == fc1.bias.shape[0]
ASSERT fc1.weight.shape[0] == fc2.weight.shape[1]
ASSERT fc2.weight.shape[0] == fc2.bias.shape[0] == 1
```

### Dimensional Semantics

| Dimension | Name | Semantic |
|-----------|------|----------|
| `fc1.weight` dim 0 | intermediate_dim | Hidden layer width |
| `fc1.weight` dim 1 | hidden_dim | Input dimension (from base model) |
| `fc2.weight` dim 0 | output_dim | Output dimension (binary) |

### Loading Example

```python
# Illustrative - user must adapt to their environment
import torch

state_dict = torch.load(
    "checkpoints/detector/detector_llama3_8b.pt",
    map_location="cpu",
    weights_only=True
)

# Enumerate contents
for key, tensor in state_dict.items():
    print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")

# Expected output:
# fc1.weight: shape=torch.Size([512, 4096]), dtype=torch.float32
# fc1.bias: shape=torch.Size([512]), dtype=torch.float32
# fc2.weight: shape=torch.Size([1, 512]), dtype=torch.float32
# fc2.bias: shape=torch.Size([1]), dtype=torch.float32
```

---

## 2. Refinement Tensors

### File Location
```
checkpoints/refinement/refinement_llama3_8b.pt
```

### Tensor Enumeration

| Key | Shape | Dtype | Parameters | Description |
|-----|-------|-------|------------|-------------|
| `w_up.weight` | (732, 4096) | float32/float16 | 2,998,272 | Expansion matrix |
| `w_down.weight` | (4096, 732) | float32/float16 | 2,998,272 | Compression matrix |

**Total**: 5,996,544 parameters

### Shape Invariants

```
ASSERT w_up.weight.shape[1] == w_down.weight.shape[0]  # hidden_dim
ASSERT w_up.weight.shape[0] == w_down.weight.shape[1]  # expanded_dim
```

### Dimensional Semantics

| Dimension | Name | Semantic |
|-----------|------|----------|
| `w_up.weight` dim 0 | expanded_dim | Bottleneck width |
| `w_up.weight` dim 1 | hidden_dim | Base model hidden dimension |
| `w_down.weight` dim 0 | hidden_dim | Output dimension (matches input) |
| `w_down.weight` dim 1 | expanded_dim | Bottleneck width |

### Absence of Bias

The refinement projection layers do not include bias terms. This is intentional:
- Reduces parameter count
- Preserves zero-input zero-output property
- Simplifies residual dynamics

### Loading Example

```python
# Illustrative - user must adapt to their environment
import torch

state_dict = torch.load(
    "checkpoints/refinement/refinement_llama3_8b.pt",
    map_location="cpu",
    weights_only=True
)

# Enumerate contents
for key, tensor in state_dict.items():
    print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")

# Expected output:
# w_up.weight: shape=torch.Size([732, 4096]), dtype=torch.float32
# w_down.weight: shape=torch.Size([4096, 732]), dtype=torch.float32
```

---

## 3. Classifier Tensors

### File Location
```
checkpoints/classifier/classifier_deberta_v3_large.pt
```

### Architecture Dependency

The classifier checkpoint contains the full state dictionary of a fine-tuned DeBERTa-v3-large model. The tensor structure depends on the base model architecture.

**The user must consult the DeBERTa architecture specification for tensor semantics.**

### Partial Enumeration (Major Components)

| Key Pattern | Shape Pattern | Description |
|-------------|---------------|-------------|
| `deberta.embeddings.word_embeddings.weight` | (vocab_size, 1024) | Token embeddings |
| `deberta.embeddings.LayerNorm.*` | (1024,) | Embedding normalization |
| `deberta.encoder.layer.{i}.attention.*` | Various | Self-attention parameters |
| `deberta.encoder.layer.{i}.intermediate.*` | Various | FFN parameters |
| `deberta.encoder.layer.{i}.output.*` | Various | Layer output parameters |
| `pooler.dense.*` | (1024, 1024), (1024,) | Pooler projection |
| `classifier.*` | (4, 1024), (4,) | Classification head |

### Classification Head

| Key | Shape | Description |
|-----|-------|-------------|
| `classifier.weight` | (4, 1024) | 4-class output projection |
| `classifier.bias` | (4,) | Classification bias |

The 4 classes correspond to conflict taxonomy levels:
- Index 0: L1 (Temporal)
- Index 1: L2 (Numerical)
- Index 2: L3 (Entity)
- Index 3: L4 (Semantic)

### Loading Note

Due to the large size (~304M parameters), the user should consider:
- Memory-mapped loading
- Streaming tensor loading
- Precision reduction post-load

---

## Precision Formats

### Provided Formats

Checkpoints may be provided in multiple precision formats:

| Format | Extension | Typical Size |
|--------|-----------|--------------|
| FP32 | `.pt` | 1x (baseline) |
| FP16 | `_fp16.pt` | 0.5x |
| BF16 | `_bf16.pt` | 0.5x |

### User Conversion

The user may convert between formats at load time:

```python
# Illustrative - user must adapt
state_dict = torch.load("checkpoint.pt", map_location="cpu")

# Convert to desired precision
for key in state_dict:
    state_dict[key] = state_dict[key].to(torch.float16)  # or bfloat16
```

**Precision selection affects numerical accuracy. The user is responsible for validating that their precision choice is appropriate.**

---

## Integrity Verification

### Parameter Count Verification

```python
def verify_parameter_count(state_dict, expected_count):
    actual_count = sum(t.numel() for t in state_dict.values())
    assert actual_count == expected_count, f"Expected {expected_count}, got {actual_count}"

# Detector
verify_parameter_count(detector_state_dict, 2098177)

# Refinement
verify_parameter_count(refinement_state_dict, 5996544)
```

### Shape Verification

```python
def verify_detector_shapes(state_dict):
    assert state_dict["fc1.weight"].shape == (512, 4096)
    assert state_dict["fc1.bias"].shape == (512,)
    assert state_dict["fc2.weight"].shape == (1, 512)
    assert state_dict["fc2.bias"].shape == (1,)

def verify_refinement_shapes(state_dict):
    assert state_dict["w_up.weight"].shape == (732, 4096)
    assert state_dict["w_down.weight"].shape == (4096, 732)
```

### Checksum Verification (Optional)

If SHA-256 checksums are provided:

```python
import hashlib

def compute_checksum(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()

# Compare against published checksum
```

---

## Architecture Binding

The user must map tensor keys to their computational graph. Example pattern:

```python
# Illustrative pseudocode

class UserDetector:
    def __init__(self, state_dict):
        self.fc1_weight = state_dict["fc1.weight"]
        self.fc1_bias = state_dict["fc1.bias"]
        self.fc2_weight = state_dict["fc2.weight"]
        self.fc2_bias = state_dict["fc2.bias"]

    def forward(self, h):
        # User implements computation using bound tensors
        # ...
```

**No binding API is provided. The user constructs their own architecture.**

---

## What Is NOT Included

The checkpoint files contain **only tensor data**. The following are NOT embedded:

| Excluded Item | Reason |
|---------------|--------|
| Architecture class definitions | User must provide |
| Hyperparameters (alpha, t_max, etc.) | User must specify |
| Optimizer state | Inference-only release |
| Training configuration | User must reconstruct if retraining |
| Framework version | Environment-agnostic design |
| Device placement | User determines |
| Precision specification | User converts as needed |

---

## File Format Specification

### PyTorch Format (.pt)

Standard PyTorch serialization via `torch.save()`:

```
Structure:
    Magic number (2 bytes)
    Protocol version (1 byte)
    Pickled object (state_dict)
    Storage data (tensor bytes)
```

### SafeTensors Format (.safetensors) - If Provided

```
Structure:
    Header length (8 bytes, little-endian)
    JSON header (tensor metadata)
    Tensor data (contiguous bytes)
```

SafeTensors provides:
- No arbitrary code execution
- Memory-mapped loading
- Partial tensor loading

**The user should select the format compatible with their framework.**
