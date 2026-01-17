# Troubleshooting Guide

This guide covers common issues and their solutions when working with the LCR system.

## Installation Issues

### CUDA Version Mismatch

**Symptom**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**:
1. Check your CUDA version: `nvcc --version`
2. Install matching PyTorch: `pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html`
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Transformers Version Conflict

**Symptom**:
```
ImportError: cannot import name 'DebertaV2ForSequenceClassification'
```

**Solution**:
```bash
pip install transformers>=4.35.0
```

### Missing Dependencies

**Symptom**:
```
ModuleNotFoundError: No module named 'rank_bm25'
```

**Solution**:
```bash
pip install rank-bm25 nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Training Issues

### Out of Memory (OOM)

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Solutions**:

1. **Reduce batch size**:
```yaml
# configs/base/refinement_base.yaml
training:
  batch_size: 16  # Reduce from 32
```

2. **Enable gradient checkpointing**:
```python
model.gradient_checkpointing_enable()
```

3. **Use mixed precision**:
```bash
python src/training/train_refinement.py --fp16
```

4. **Clear cache between batches**:
```python
torch.cuda.empty_cache()
```

### NaN Loss

**Symptom**:
```
Loss: nan at step 1234
```

**Causes and Solutions**:

1. **Learning rate too high**:
```yaml
training:
  learning_rate: 1.0e-5  # Reduce from 2e-5
```

2. **Missing gradient clipping**:
```yaml
training:
  gradient_clip: 1.0
```

3. **Data issue** - check for invalid values:
```python
# Add to data loading
assert not torch.isnan(inputs).any(), f"NaN in inputs at idx {idx}"
```

### Training Not Converging

**Symptom**: Loss not decreasing after many epochs

**Checklist**:
1. ✅ Verify data is shuffled: `DataLoader(shuffle=True)`
2. ✅ Check learning rate warmup is configured
3. ✅ Verify labels are correct format
4. ✅ Try different optimizer: AdamW instead of Adam
5. ✅ Check for data leakage between splits

### Overfitting

**Symptom**: Train loss decreasing, validation loss increasing

**Solutions**:

1. **Increase regularization**:
```yaml
model:
  dropout: 0.2  # Increase from 0.1
training:
  weight_decay: 0.05  # Increase from 0.01
```

2. **Reduce model capacity**:
```yaml
model:
  intermediate_dim: 256  # Reduce from 512
```

3. **Add early stopping**:
```yaml
training:
  early_stopping_patience: 3  # Stop sooner
```

4. **Data augmentation**:
```python
# Enable augmentation
dataset = ConflictDataset(path, augment=True)
```

## Inference Issues

### Slow Inference

**Symptom**: Processing takes >100ms per example

**Solutions**:

1. **Use batch inference**:
```python
# Instead of
for example in examples:
    result = model(example)

# Use
results = model.batch_process(examples, batch_size=32)
```

2. **Enable TorchScript**:
```python
model = torch.jit.script(model)
```

3. **Use FP16 inference**:
```python
model = model.half()
```

4. **Cache embeddings**:
```python
# Pre-compute and cache LLM embeddings
embeddings = compute_embeddings(documents)
torch.save(embeddings, "cache/embeddings.pt")
```

### Inconsistent Results

**Symptom**: Different outputs for same input

**Solutions**:

1. **Set eval mode**:
```python
model.eval()  # Disables dropout
```

2. **Set seeds**:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

3. **Disable non-deterministic ops**:
```python
torch.use_deterministic_algorithms(True)
```

### Low Accuracy on New Data

**Symptom**: Good dev accuracy but poor on deployment data

**Possible Causes**:

1. **Distribution shift** - training data differs from deployment
   - Solution: Collect and fine-tune on domain-specific data

2. **Different document formats**
   - Solution: Ensure same preprocessing pipeline

3. **Different query types**
   - Solution: Analyze error cases, expand training data

## Data Issues

### Data Loading Errors

**Symptom**:
```
json.decoder.JSONDecodeError: Expecting value
```

**Solution**:
```python
# Use robust loading
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error at line {i}: {e}")
    return data
```

### Tokenization Issues

**Symptom**: Sequences truncated or padded incorrectly

**Solution**:
```python
# Verify tokenization
tokens = tokenizer(text, max_length=512, truncation=True, padding='max_length')
print(f"Input length: {len(text.split())}, Token length: {len(tokens['input_ids'])}")
```

### Missing Labels

**Symptom**:
```
KeyError: 'conflict_type'
```

**Solution**:
```python
# Add validation to data loading
required_keys = ['query', 'documents', 'conflict_type', 'gold_answer']
for example in data:
    missing = [k for k in required_keys if k not in example]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
```

## Model Loading Issues

### Checkpoint Not Found

**Symptom**:
```
FileNotFoundError: checkpoints/detector/best_model.pt
```

**Solution**:
1. Verify checkpoint exists: `ls checkpoints/detector/`
2. Check file permissions
3. Use absolute path:
```python
import os
path = os.path.abspath("checkpoints/detector/best_model.pt")
```

### Architecture Mismatch

**Symptom**:
```
RuntimeError: Error(s) in loading state_dict: size mismatch for fc1.weight
```

**Solution**:
```python
# Load config from checkpoint
config = torch.load(path)['config']
model = ConflictDetector(**config)
model.load_state_dict(torch.load(path)['model_state_dict'])
```

### Version Incompatibility

**Symptom**:
```
ModuleNotFoundError: No module named 'transformers.models.deberta_v2'
```

**Solution**:
Save models with version info:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'pytorch_version': torch.__version__,
    'transformers_version': transformers.__version__
}, path)
```

## Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: See the repository's Issues page
2. **Open new issue** with:
   - Error message and full traceback
   - Environment info (`pip freeze`)
   - Steps to reproduce
   - Relevant config files

3. **For urgent issues**, include:
   - Minimal reproducible example
   - Input data sample (if not sensitive)
