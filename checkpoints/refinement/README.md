# Refinement Module

**Status**: Pre-trained refinement module

## Specifications
- Parameters: 5,996,544
- Architecture: Bottleneck (d → 732 → d)
- Iterations: T_max = 3
- Update step: α = 0.3
- Convergence: ε = 0.01

## Algorithm
```
For t = 1 to T_max:
    z^(t) = W_up · h^(t-1)
    Δh^(t) = W_down · ReLU(z^(t))
    h^(t) = h^(t-1) + α·Δh^(t)
    if ||Δh^(t)|| < ε: break
```

## Training Loss
L = L_CE + 0.01·L_L2 + 0.005·L_KL

where:
- L_CE: Cross-entropy on final answer
- L_L2: Regularize update magnitudes
- L_KL: Keep refined distribution close to original

## Usage
```python
from src.models import RefinementModule

refinement = RefinementModule.from_pretrained("checkpoints/refinement")
refined_states = refinement(hidden_states, num_iterations=3)
```
