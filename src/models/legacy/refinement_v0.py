#!/usr/bin/env python3
"""
[DEPRECATED] Original Refinement Module v0

Initial refinement approach using additive perturbations.
This was replaced because:
- Additive updates were too aggressive
- No convergence guarantee
- Required manual tuning of step size per layer

Replaced by: src/models/refinement_module.py

The v1 module uses a learned bottleneck projection with
controlled update magnitude.
"""

import warnings
warnings.warn(
    "refinement_v0 is deprecated. Use refinement_module.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class AdditiveRefinementV0(nn.Module):
    """
    Original additive refinement approach.

    This directly added a learned perturbation to hidden states,
    which caused:
    - Representation drift
    - Inconsistent behavior across layers
    - Poor generalization to unseen conflict types
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_refinement_steps: int = 5,
        step_size: float = 0.1,
        use_layerwise_adaptation: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_steps = num_refinement_steps
        self.step_size = step_size

        self.perturbation_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_layerwise_adaptation:
            self.layer_weights = nn.Parameter(torch.ones(32))
        else:
            self.layer_weights = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply iterative refinement.

        Returns:
            refined_states: Final refined hidden states
            trajectory: List of intermediate states
        """
        num_steps = num_steps or self.num_steps
        trajectory = [hidden_states]

        current = hidden_states
        for step in range(num_steps):
            delta = self.perturbation_net(current)

            delta = delta * self.step_size

            current = current + delta

            trajectory.append(current)

        return current, trajectory

    def refine_with_early_stopping(
        self,
        hidden_states: torch.Tensor,
        threshold: float = 0.01,
        max_steps: int = 10,
    ) -> Tuple[torch.Tensor, int]:
        """
        Refine with early stopping based on update magnitude.

        Note: This early stopping criterion was flawed.
        It often stopped too early or not at all.
        """
        current = hidden_states

        for step in range(max_steps):
            delta = self.perturbation_net(current) * self.step_size

            update_norm = delta.norm(dim=-1).mean()

            if update_norm < threshold:
                break

            current = current + delta

        return current, step + 1


class GatedRefinementV0(nn.Module):
    """
    Alternative: Gated refinement (experimental, not in paper).

    This used a gating mechanism to control updates,
    but was abandoned due to training instability.
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.update_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply gated update."""
        update = self.update_value(hidden_states)

        if context is not None:
            gate_input = torch.cat([hidden_states, context], dim=-1)
        else:
            gate_input = torch.cat([hidden_states, hidden_states], dim=-1)

        gate = self.update_gate(gate_input)

        return hidden_states + gate * update


class StaticInterventionV0(nn.Module):
    """
    Static intervention baseline (for comparison).

    This applies a fixed learned intervention regardless of input.
    Included to show that dynamic refinement is necessary.
    """

    def __init__(self, hidden_dim: int = 4096, num_interventions: int = 4):
        super().__init__()

        self.interventions = nn.Parameter(
            torch.randn(num_interventions, hidden_dim) * 0.01
        )

        self.classifier = nn.Linear(hidden_dim, num_interventions)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conflict_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply static intervention."""
        if conflict_type is None:
            logits = self.classifier(hidden_states.mean(dim=1) if hidden_states.dim() == 3 else hidden_states)
            conflict_type = logits.argmax(dim=-1)

        intervention = self.interventions[conflict_type]

        if hidden_states.dim() == 3:
            intervention = intervention.unsqueeze(1)

        return hidden_states + intervention


if __name__ == '__main__':
    print("RefinementV0 Modules - DEPRECATED")
    print("These modules are retained for historical reference only.")

    model = AdditiveRefinementV0()
    x = torch.randn(2, 4096)
    refined, trajectory = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {refined.shape}")
    print(f"Trajectory length: {len(trajectory)}")
