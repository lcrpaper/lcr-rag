"""
Gated Refinement Module - Experimental Architecture

This module implements a gated variant of the refinement mechanism
where the update magnitude is controlled by a learned gate.

Architecture:
    z = W_up @ h
    delta_h = W_down @ ReLU(z)
    gate = sigmoid(W_gate @ h + b_gate)  # Per-dimension gating
    h_new = h + alpha * gate * delta_h

Hypothesis:
    Learned gating could allow the model to selectively refine
    different aspects of the representation based on conflict type.

Status: EXPERIMENTAL - Not used in paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class GatedRefinementModule(nn.Module):
    """
    Gated refinement with per-dimension learned gates.

    The gate allows the model to control which dimensions
    receive refinement updates based on input features.

    Args:
        hidden_dim: Input/output dimension (default: 4096)
        bottleneck_dim: Bottleneck dimension (default: 732)
        alpha: Base interpolation coefficient (default: 0.3)
        num_iterations: Maximum refinement iterations (default: 3)
        epsilon: Convergence threshold (default: 0.01)
        gate_bias_init: Initial bias for gate (default: -2.0)
            Negative bias encourages sparse gating initially
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 732,
        alpha: float = 0.3,
        num_iterations: int = 3,
        epsilon: float = 0.01,
        gate_bias_init: float = -2.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        self.W_up = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.W_down = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        self.W_gate = nn.Linear(hidden_dim, hidden_dim)

        nn.init.constant_(self.W_gate.bias, gate_bias_init)

        nn.init.xavier_uniform_(self.W_up.weight)
        nn.init.xavier_uniform_(self.W_down.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_iterations: Optional[int] = None,
        return_intermediates: bool = False,
        return_gates: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with gated refinement.

        Args:
            hidden_states: Input hidden states [batch_size, hidden_dim]
            num_iterations: Override default iterations
            return_intermediates: Return all intermediate states
            return_gates: Return gate values for analysis

        Returns:
            Dictionary containing:
                - 'output': Refined hidden states
                - 'iterations': Number of iterations used
                - 'converged': Whether convergence was reached
                - 'intermediates': (optional) List of intermediate states
                - 'gates': (optional) Gate values at each iteration
        """
        T = num_iterations if num_iterations is not None else self.num_iterations

        h = hidden_states
        intermediates = [h] if return_intermediates else None
        gates = [] if return_gates else None

        converged = False
        actual_iterations = T

        for t in range(T):
            z = self.W_up(h)
            delta_h = self.W_down(F.relu(z))

            gate = torch.sigmoid(self.W_gate(h))

            if return_gates:
                gates.append(gate.detach())

            h_new = h + self.alpha * gate * delta_h

            delta_norm = torch.norm(self.alpha * delta_h, dim=-1).mean()
            if delta_norm < self.epsilon:
                converged = True
                actual_iterations = t + 1
                h = h_new
                if return_intermediates:
                    intermediates.append(h)
                break

            h = h_new
            if return_intermediates:
                intermediates.append(h)

        result = {
            'output': h,
            'iterations': actual_iterations,
            'converged': converged,
        }

        if return_intermediates:
            result['intermediates'] = intermediates
        if return_gates:
            result['gates'] = gates

        return result

    def analyze_gating(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze gating behavior for interpretability.

        Returns statistics about which dimensions are gated.
        """
        with torch.no_grad():
            gate = torch.sigmoid(self.W_gate(hidden_states))

            stats = {
                'mean_gate': gate.mean(dim=0),
                'gate_sparsity': (gate < 0.1).float().mean(),
                'high_gate_dims': (gate.mean(dim=0) > 0.5).sum(),
            }

        return stats

    @property
    def num_parameters(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


class TypeConditionedGatedRefinement(nn.Module):
    """
    Variant with conflict-type-conditioned gating.

    Uses the predicted conflict type to modulate the gate,
    allowing type-specific refinement patterns.

    Status: EXPERIMENTAL - Did not improve over standard gating

    Args:
        hidden_dim: Input dimension
        bottleneck_dim: Bottleneck dimension
        num_types: Number of conflict types (default: 4)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 732,
        num_types: int = 4,
        alpha: float = 0.3,
        num_iterations: int = 3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_types = num_types
        self.alpha = alpha
        self.num_iterations = num_iterations

        self.W_up = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.W_down = nn.Linear(bottleneck_dim, hidden_dim, bias=False)

        self.type_embeddings = nn.Embedding(num_types, hidden_dim)

        self.W_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conflict_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with type-conditioned gating.

        Args:
            hidden_states: [batch_size, hidden_dim]
            conflict_type: [batch_size] tensor of type indices (0-3)
                If None, uses average of all type embeddings.
        """
        batch_size = hidden_states.size(0)

        if conflict_type is not None:
            type_emb = self.type_embeddings(conflict_type)
        else:
            type_emb = self.type_embeddings.weight.mean(dim=0, keepdim=True)
            type_emb = type_emb.expand(batch_size, -1)

        h = hidden_states

        for _ in range(self.num_iterations):
            z = self.W_up(h)
            delta_h = self.W_down(F.relu(z))

            gate_input = torch.cat([h, type_emb], dim=-1)
            gate = torch.sigmoid(self.W_gate(gate_input))

            h = h + self.alpha * gate * delta_h

        return h


if __name__ == '__main__':
    model = GatedRefinementModule()
    print(f"GatedRefinementModule parameters: {model.num_parameters:,}")

    x = torch.randn(4, 4096)
    result = model(x, return_gates=True)

    print(f"Output shape: {result['output'].shape}")
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")

    stats = model.analyze_gating(x)
    print(f"Gate sparsity: {stats['gate_sparsity']:.3f}")
