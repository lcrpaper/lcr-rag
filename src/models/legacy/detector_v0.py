#!/usr/bin/env python3
"""
[DEPRECATED] Original Conflict Detector v0

This was the initial detector architecture from the pilot study (Apr 2025).
It used a 3-layer MLP with skip connections, which proved to be:
- Overparameterized (5M params vs 2.1M in final)
- Prone to overfitting on small datasets
- Slower inference due to additional layers

Replaced by: src/models/conflict_detector.py (v1/v2)
"""

import warnings
warnings.warn(
    "detector_v0 is deprecated. Use conflict_detector.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class ResidualBlock(nn.Module):
    """Residual block with skip connection (unused in final version)."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + residual)


class ConflictDetectorV0(nn.Module):
    """
    Original 3-layer conflict detector.

    Architecture:
        Input (4096) -> FC1 (8192) -> ResBlock (8192) -> FC2 (4096) -> FC3 (1)

    Problems identified:
    1. ResBlock added unnecessary parameters
    2. GELU activation caused gradient issues
    3. LayerNorm placement was suboptimal

    DO NOT USE for new experiments.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 8192,
        intermediate_dim: int = 4096,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        if use_residual:
            self.res_block = ResidualBlock(hidden_dim, dropout)
        else:
            self.fc_mid = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, intermediate_dim)
        self.norm2 = nn.LayerNorm(intermediate_dim)

        self.fc_out = nn.Linear(intermediate_dim, 1)

        self.dropout = nn.Dropout(dropout)

        self._init_weights_v0()

    def _init_weights_v0(self):
        """Original weight initialization (suboptimal)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            attention_mask: Optional mask
            return_intermediate: Return intermediate activations

        Returns:
            Dict with 'logits' and optionally 'intermediate'
        """
        if hidden_states.dim() == 3:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden_states = (hidden_states * mask).sum(1) / mask.sum(1)
            else:
                hidden_states = hidden_states.mean(dim=1)

        intermediates = {}

        x = self.fc1(hidden_states)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        if return_intermediate:
            intermediates['layer1'] = x.detach()

        if self.use_residual:
            x = self.res_block(x)
        else:
            x = self.fc_mid(x)
            x = F.gelu(x)

        if return_intermediate:
            intermediates['layer2'] = x.detach()

        x = self.fc2(x)
        x = F.gelu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        logits = self.fc_out(x).squeeze(-1)

        output = {'logits': logits}
        if return_intermediate:
            output['intermediate'] = intermediates

        return output

    def predict(
        self,
        hidden_states: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions."""
        with torch.no_grad():
            output = self.forward(hidden_states)
            probs = torch.sigmoid(output['logits'])
            preds = (probs > threshold).long()
        return preds, probs

    @classmethod
    def from_pretrained_v0(cls, checkpoint_path: str) -> 'ConflictDetectorV0':
        """Load v0 checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            config = {}

        model = cls(**config)
        model.load_state_dict(state_dict, strict=False)

        return model


def create_detector_v0(**kwargs) -> ConflictDetectorV0:
    """Factory function for v0 detector."""
    warnings.warn(
        "create_detector_v0 is deprecated. Use ConflictDetector from conflict_detector.py",
        DeprecationWarning
    )
    return ConflictDetectorV0(**kwargs)


if __name__ == '__main__':
    print("ConflictDetectorV0 - DEPRECATED")
    print("This module is retained for loading old checkpoints only.")

    model = ConflictDetectorV0()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(2, 4096)
    out = model(x)
    print(f"Output shape: {out['logits'].shape}")
