"""
Adaptive Refinement Module - EXPERIMENTAL

**Status**: IN DEVELOPMENT, Not included in paper (exploratory research)

This module explores learning the refinement step size (alpha) dynamically
per-example.

Hypothesis:
-----------
Different conflict types may require different correction magnitudes.
A learned alpha could adapt to the specific characteristics of each example.

Preliminary Results:
--------------------
- Training stability: Worse than fixed alpha (requires careful initialization)
- Final performance: +0.3% on L1, -0.2% on L2 (within noise)
- Complexity: 2x parameters, 1.5x training time
- Decision: Not worth the added complexity for marginal/inconsistent gains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import warnings


class AdaptiveRefinementModule(nn.Module):
    """
    Refinement module with learned per-example step size.

    EXPERIMENTAL - Not used in paper results.

    Instead of fixed alpha=0.3, this module learns to predict the optimal
    alpha for each example based on the hidden state characteristics.

    Architecture:
        alpha_t = sigmoid(W_alpha @ h_bar) * alpha_max
        z^(t) = W_up @ h^(t-1)
        delta_h^(t) = W_down @ ReLU(z^(t))
        h^(t) = h^(t-1) + alpha_t * delta_h^(t)

    Args:
        hidden_dim: Base model hidden dimension (default: 4096)
        expansion_factor: Expansion factor for bottleneck
        alpha_max: Maximum allowed alpha value (default: 0.5)
        alpha_min: Minimum allowed alpha value (default: 0.1)
        t_max: Maximum refinement iterations (default: 3)
        epsilon: Convergence threshold (default: 0.01)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        expansion_factor: Optional[float] = None,
        alpha_max: float = 0.5,
        alpha_min: float = 0.1,
        t_max: int = 3,
        epsilon: float = 0.01
    ):
        super().__init__()

        warnings.warn(
            "AdaptiveRefinementModule is EXPERIMENTAL and not validated. "
            "For paper results, use RefinementModule with fixed alpha=0.3.",
            UserWarning
        )

        self.hidden_dim = hidden_dim
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.alpha_range = alpha_max - alpha_min
        self.t_max = t_max
        self.epsilon = epsilon

        if expansion_factor is None:
            target_params = 6_000_000
            self.expanded_dim = int(target_params / (2 * hidden_dim))
        else:
            self.expanded_dim = int(hidden_dim * expansion_factor)

        self.w_up = nn.Linear(hidden_dim, self.expanded_dim, bias=False)
        self.relu = nn.ReLU()
        self.w_down = nn.Linear(self.expanded_dim, hidden_dim, bias=False)

        self.alpha_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        nn.init.kaiming_uniform_(self.w_up.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.w_down.weight, nonlinearity='relu')

        nn.init.zeros_(self.alpha_predictor[-2].weight)
        nn.init.zeros_(self.alpha_predictor[-2].bias)

    def predict_alpha(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-example alpha from hidden state.

        Args:
            h: Hidden state (batch_size, seq_len, hidden_dim)

        Returns:
            alpha: Per-example alpha (batch_size, 1, 1) for broadcasting
        """
        h_bar = h.mean(dim=1)

        alpha_normalized = self.alpha_predictor(h_bar)

        alpha = self.alpha_min + self.alpha_range * alpha_normalized

        return alpha.unsqueeze(-1)

    def forward(
        self,
        h_0: torch.Tensor,
        return_trajectory: bool = False,
        return_alphas: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Iterative refinement with adaptive step size.

        Args:
            h_0: Initial hidden state (batch_size, seq_len, hidden_dim)
            return_trajectory: If True, return all intermediate states
            return_alphas: If True, return predicted alphas for each iteration

        Returns:
            h_T: Refined hidden state
            trajectory: List of h^(t) if return_trajectory=True
            alphas: List of alpha_t if return_alphas=True
        """
        h = h_0
        trajectory = [h.clone()] if return_trajectory else None
        alphas = [] if return_alphas else None

        for t in range(self.t_max):
            alpha_t = self.predict_alpha(h)

            if return_alphas:
                alphas.append(alpha_t.squeeze())

            z = self.w_up(h)
            delta_h = self.w_down(self.relu(z))

            h = h + alpha_t * delta_h

            if return_trajectory:
                trajectory.append(h.clone())

            delta_norm = torch.norm(delta_h, dim=-1).mean()
            if delta_norm < self.epsilon:
                break

        return h, trajectory, alphas

    def refine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convenience method for refinement."""
        refined, _, _ = self.forward(hidden_states)
        return refined

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_alpha_statistics(self, h_0: torch.Tensor) -> dict:
        """
        Get statistics about predicted alphas.

        Useful for analyzing what the model learned about alpha.
        """
        _, _, alphas = self.forward(h_0, return_alphas=True)

        if not alphas:
            return {"error": "No alphas collected"}

        alpha_tensor = torch.stack(alphas)

        return {
            "mean_alpha": alpha_tensor.mean().item(),
            "std_alpha": alpha_tensor.std().item(),
            "min_alpha": alpha_tensor.min().item(),
            "max_alpha": alpha_tensor.max().item(),
            "per_iteration_mean": [a.mean().item() for a in alphas],
        }

if __name__ == "__main__":
    print("AdaptiveRefinementModule - EXPERIMENTAL")
    print("=" * 50)

    module = AdaptiveRefinementModule(hidden_dim=4096)
    print(f"Parameters: {module.count_parameters():,}")

    x = torch.randn(2, 128, 4096)
    out, traj, alphas = module(x, return_trajectory=True, return_alphas=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Iterations: {len(traj) - 1}")

    stats = module.get_alpha_statistics(x)
    print(f"Alpha statistics: {stats}")

    print("\n" + "=" * 50)
