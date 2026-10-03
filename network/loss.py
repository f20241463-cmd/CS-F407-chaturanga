"""
ChaturangZero Loss Function (Student 2 Deliverable).

Loss = Value Loss (MSE) + Policy Loss (Cross-Entropy).
Conforms to Technical Interfaces Section 5 & 8.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroLoss(nn.Module):
    """
    Combined Loss for AlphaZero:
      - Value loss: Mean Squared Error (MSE) between predicted value v and game outcome z in [-1, +1]
      - Policy loss: Cross-Entropy between predicted policy logits p and MCTS target distribution pi
    """

    def __init__(self, value_weight: float = 1.0, policy_weight: float = 1.0):
        super().__init__()
        self.value_weight = value_weight
        self.policy_weight = policy_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_values: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_logits: [B, 4096] raw network policy logits
            pred_values: [B, 1] or [B] predicted values in [-1, 1]
            target_policies: [B, 4096] target probability distribution (e.g. from MCTS visit counts)
            target_values: [B, 1] or [B] target outcomes z in {-1, 0, 1}
        Returns:
            total_loss: scalar tensor for backward()
            metrics: dict with detached float values for logging ('loss', 'value_loss', 'policy_loss')
        """
        # Ensure values match shape [B, 1]
        if pred_values.dim() == 1:
            pred_values = pred_values.unsqueeze(1)
        if target_values.dim() == 1:
            target_values = target_values.unsqueeze(1)

        # Value loss: Mean Squared Error (v - z)^2
        val_loss = F.mse_loss(pred_values, target_values)

        # Policy loss: Cross-Entropy = - sum(target_pi * log_softmax(pred_logits))
        log_probs = F.log_softmax(pred_logits, dim=-1)
        pol_loss = -(target_policies * log_probs).sum(dim=-1).mean()

        total = (self.value_weight * val_loss) + (self.policy_weight * pol_loss)

        metrics = {
            "loss": float(total.item()),
            "value_loss": float(val_loss.item()),
            "policy_loss": float(pol_loss.item()),
        }

        return total, metrics
