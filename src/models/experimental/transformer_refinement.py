"""
Transformer-based Refinement Module - Experimental Architecture

This module implements a single-layer Transformer for refinement,
treating the refinement task as a sequence-to-sequence transformation.

Architecture:
    Input: h ∈ R^{S×d} (sequence of hidden states)
    Output: h_refined ∈ R^{S×d}

    h_refined = h + alpha * TransformerLayer(h)

Where TransformerLayer includes:
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization

Hypothesis:
    Self-attention could allow tokens to exchange information
    about conflicting evidence, leading to better refinement.

Results:
    - Accuracy: 65.1% (vs 65.0% for MLP)
    - Parameters: 12.5M (vs 6.0M for MLP)
    - Inference time: 2.3x slower
    - Conclusion: Marginal improvement not worth 2x parameters

Status: EXPERIMENTAL - Not used in paper due to efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class TransformerRefinementLayer(nn.Module):
    """
    Single Transformer layer for refinement.

    Implements standard Transformer architecture:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Residual connections and layer normalization

    Args:
        hidden_dim: Model dimension (default: 4096)
        num_heads: Number of attention heads (default: 8)
        ff_dim: Feed-forward hidden dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ff_up = nn.Linear(hidden_dim, ff_dim)
        self.ff_down = nn.Linear(ff_dim, hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.ff_up.weight)
        nn.init.xavier_uniform_(self.ff_down.weight)

        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.ff_down.weight, std=0.02 / math.sqrt(2))

    def attention(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head self-attention.

        Args:
            x: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] or None

        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        output = self.o_proj(attn_output)

        return output, attn_weights

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Transformer layer.

        Args:
            x: [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with 'output' and optionally 'attention'
        """
        normed = self.ln1(x)
        attn_out, attn_weights = self.attention(normed, attention_mask)
        x = x + self.dropout(attn_out)

        normed = self.ln2(x)
        ff_out = self.ff_down(F.gelu(self.ff_up(normed)))
        x = x + self.dropout(ff_out)

        result = {'output': x}
        if return_attention:
            result['attention'] = attn_weights

        return result


class TransformerRefinementModule(nn.Module):
    """
    Full Transformer-based refinement module.

    Wraps TransformerRefinementLayer with iterative application
    and interpolation, matching the MLP refinement interface.

    Args:
        hidden_dim: Model dimension
        num_heads: Attention heads
        ff_dim: Feed-forward dimension
        alpha: Interpolation coefficient
        num_iterations: Maximum iterations
        epsilon: Convergence threshold
        num_layers: Number of Transformer layers (default: 1)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_heads: int = 8,
        ff_dim: int = 2048,
        alpha: float = 0.3,
        num_iterations: int = 3,
        epsilon: float = 0.01,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        self.layers = nn.ModuleList([
            TransformerRefinementLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_iterations: Optional[int] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with iterative Transformer refinement.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            attention_mask: Optional mask
            num_iterations: Override default iterations
            return_attention: Return attention weights

        Returns:
            Dictionary with 'output', 'iterations', 'converged', etc.
        """
        squeeze_output = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
            squeeze_output = True

        T = num_iterations if num_iterations is not None else self.num_iterations
        h = hidden_states

        all_attentions = [] if return_attention else None
        converged = False
        actual_iterations = T

        for t in range(T):
            delta_h = h
            for layer in self.layers:
                result = layer(delta_h, attention_mask, return_attention)
                delta_h = result['output']
                if return_attention:
                    all_attentions.append(result['attention'])

            delta = delta_h - h

            delta_norm = torch.norm(delta, dim=-1).mean()
            if delta_norm < self.epsilon:
                converged = True
                actual_iterations = t + 1

            h = h + self.alpha * delta

            if converged:
                break

        if squeeze_output:
            h = h.squeeze(1)

        result = {
            'output': h,
            'iterations': actual_iterations,
            'converged': converged,
        }

        if return_attention:
            result['attention'] = all_attentions

        return result

    @property
    def num_parameters(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    model = TransformerRefinementModule()
    print(f"TransformerRefinementModule parameters: {model.num_parameters:,}")

    x = torch.randn(4, 32, 4096)
    result = model(x, return_attention=True)

    print(f"Output shape: {result['output'].shape}")
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")

    x_2d = torch.randn(4, 4096)
    result_2d = model(x_2d)
    print(f"2D input output shape: {result_2d['output'].shape}")
