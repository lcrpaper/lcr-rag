# Conflict Detector Model

**Status**: Pre-trained on conflict detection task

## Specifications
- Parameters: 2,098,177
- Architecture: 2-layer MLP with mean pooling
- Input: Hidden states at layer L/2
- Output: Binary conflict probability

## Performance
- F1 Score: 0.87
- Precision: 0.89
- Recall: 0.85

## Usage
```python
from src.models import ConflictDetector

detector = ConflictDetector.from_pretrained("checkpoints/detector")
result = detector.detect(hidden_states)
```
