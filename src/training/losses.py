"""
Loss Functions for LCR Training

Implements the three loss components:
1. L_CE: Cross-entropy on final answer
2. L_L2: Regularize update magnitudes
3. L_KL: Keep refined distribution close to original

Combined loss: L = L_CE + λ₁·L_L2 + λ₂·L_KL
where λ₁ = 0.01, λ₂ = 0.005
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


def detector_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss for conflict detector (Eq. 4, Line 456).

    L_det = -Σ[y_i log(p_i) + (1-y_i)log(1-p_i)]

    Args:
        predictions: Conflict probabilities, shape (batch_size,)
        labels: Binary labels (0=no conflict, 1=conflict), shape (batch_size,)

    Returns:
        Scalar loss value
    """
    return F.binary_cross_entropy(predictions, labels)


def refinement_loss(
    h_0: torch.Tensor,
    h_T: torch.Tensor,
    trajectory: List[torch.Tensor],
    labels: torch.Tensor,
    lm_head: nn.Module,
    lambda_l2: float = 0.01,
    lambda_kl: float = 0.005
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for refinement module (Eq. 8-11, Lines 463-471).

    L = L_CE + λ₁·L_L2 + λ₂·L_KL

    Args:
        h_0: Initial hidden states, shape (batch_size, seq_len, hidden_dim)
        h_T: Final refined hidden states
        trajectory: List of intermediate states [h^(0), h^(1), ..., h^(T)]
        labels: Gold answer token IDs, shape (batch_size,)
        lm_head: Language model head for computing logits
        lambda_l2: L2 regularization coefficient (default: 0.01)
        lambda_kl: KL regularization coefficient (default: 0.005)

    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary with individual loss components
    """
    logits_T = lm_head(h_T[:, -1, :])
    L_CE = F.cross_entropy(logits_T, labels)

    L_L2 = torch.tensor(0.0, device=h_0.device)

    for t in range(1, len(trajectory)):
        delta_h = trajectory[t] - trajectory[t-1]
        L_L2 = L_L2 + torch.norm(delta_h, p=2, dim=-1).mean()

    logits_0 = lm_head(h_0[:, -1, :])

    p_0 = F.softmax(logits_0, dim=-1)
    p_T = F.softmax(logits_T, dim=-1)

    L_KL = F.kl_div(
        p_T.log(),
        p_0,
        reduction='batchmean',
        log_target=False
    )

    total_loss = L_CE + lambda_l2 * L_L2 + lambda_kl * L_KL

    loss_dict = {
        'loss_total': total_loss.item(),
        'loss_ce': L_CE.item(),
        'loss_l2': L_L2.item(),
        'loss_kl': L_KL.item(),
        'lambda_l2': lambda_l2,
        'lambda_kl': lambda_kl
    }

    return total_loss, loss_dict


def classifier_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for taxonomy classifier.

    Args:
        logits: Class logits, shape (batch_size, 4) for L1-L4
        labels: Class labels (0=L1, 1=L2, 2=L3, 3=L4), shape (batch_size,)

    Returns:
        Scalar loss value
    """
    return F.cross_entropy(logits, labels)


class RefinementLossWithRegularization(nn.Module):
    """
    Packaged refinement loss as a module for easy reuse.

    Implements L = L_CE + λ₁·L_L2 + λ₂·L_KL
    """

    def __init__(self, lm_head: nn.Module, lambda_l2: float = 0.01, lambda_kl: float = 0.005):
        super().__init__()
        self.lm_head = lm_head
        self.lambda_l2 = lambda_l2
        self.lambda_kl = lambda_kl

    def forward(
        self,
        h_0: torch.Tensor,
        h_T: torch.Tensor,
        trajectory: List[torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute refinement loss.

        Args:
            h_0: Initial hidden states
            h_T: Final refined states
            trajectory: List of all intermediate states
            labels: Gold answer tokens

        Returns:
            loss: Total loss
            metrics: Dict with loss breakdown
        """
        return refinement_loss(
            h_0, h_T, trajectory, labels, self.lm_head,
            self.lambda_l2, self.lambda_kl
        )


EXPECTED_LOSS_VALUES = {
    'detector': {
        'final_loss': 0.18,
        'initial_loss': 0.69,
    },
    'refinement': {
        'final_total': 0.31,
        'final_ce': 0.25,
        'final_l2': 0.05,
        'final_kl': 0.02,
    }
}


if __name__ == "__main__":
    print("Testing loss functions...")

    batch_size = 4
    seq_len = 128
    hidden_dim = 4096
    vocab_size = 32000

    pred = torch.rand(batch_size)
    labels = torch.randint(0, 2, (batch_size,)).float()
    det_loss = detector_loss(pred, labels)
    print(f"Detector loss: {det_loss.item():.4f}")

    h_0 = torch.randn(batch_size, seq_len, hidden_dim)
    h_1 = h_0 + 0.1 * torch.randn_like(h_0)
    h_2 = h_1 + 0.1 * torch.randn_like(h_1)
    trajectory = [h_0, h_1, h_2]

    lm_head = nn.Linear(hidden_dim, vocab_size)
    answer_labels = torch.randint(0, vocab_size, (batch_size,))

    ref_loss, metrics = refinement_loss(h_0, h_2, trajectory, answer_labels, lm_head)
    print(f"Refinement loss: {ref_loss.item():.4f}")
    print(f"  Components: CE={metrics['loss_ce']:.4f}, L2={metrics['loss_l2']:.4f}, KL={metrics['loss_kl']:.4f}")

    logits = torch.randn(batch_size, 4)
    class_labels = torch.randint(0, 4, (batch_size,))
    cls_loss = classifier_loss(logits, class_labels)
    print(f"Classifier loss: {cls_loss.item():.4f}")
