"""
ChaturangZero Neural Network Architecture (Student 2 Deliverable).

Implements a deep dual-headed Convolutional ResNet for Chaturanga:
  - Input: Board tensor [B, 13, 8, 8] (float32)
  - Backbone: Input ConvBlock + 4-6 Residual Blocks (default 4 blocks, 64 filters)
  - Policy Head: Conv(1x1) -> FC -> [B, 4096] raw policy logits
  - Value Head: Conv(1x1) -> FC -> FC -> Tanh -> [B, 1] scalar in [-1, +1]

Conforms to Technical Interfaces Section 5 and Resource Constraints Section 14.
"""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard 3x3 Conv Residual Block with Batch Normalization and skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)


class ChaturangaNet(nn.Module):
    """
    AlphaZero-style dual-headed neural network for Chaturanga.
    
    Parameters:
        in_channels: Number of input planes (13 according to contract).
        num_res_blocks: Number of residual blocks (4-6 recommended, default 4).
        num_filters: Hidden feature filters (64 or 128, default 64).
        num_actions: Size of action space (64x64 = 4096).
    """

    def __init__(
        self,
        in_channels: int = 13,
        num_res_blocks: int = 4,
        num_filters: int = 64,
        num_actions: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.num_actions = num_actions

        # Initial feature extractor
        self.input_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual backbone
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy Head: maps board features to 4096 move logits
        # 1x1 conv reduces channels to 2, then dense layer maps 2*8*8 (128) to 4096
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_actions)

        # Value Head: evaluates position [-1, 1] from perspective of active player
        # 1x1 conv reduces channels to 1, then 64 -> 64 -> 1 with Tanh
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Tensor of shape [B, 13, 8, 8] or [13, 8, 8]
        Returns:
            policy_logits: [B, 4096]
            value: [B, 1] with values in [-1.0, 1.0]
        """
        single_input = (x.dim() == 3)
        if single_input:
            x = x.unsqueeze(0)

        policy_logits, value = self.heads(self.features(x))

        if single_input:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)

        return policy_logits, value

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The residual backbone's output for a batch [B, 13, 8, 8]."""
        out = F.relu(self.input_bn(self.input_conv(x)), inplace=True)
        for block in self.res_blocks:
            out = block(out)
        return out

    def heads(self, out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Policy logits [B, 4096] and value [B, 1] from backbone features."""
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)), inplace=True)
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)), inplace=True)
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v))
        return policy_logits, value

    @torch.no_grad()
    def predict_numpy(
        self, state_tensor: np.ndarray, device: Union[str, torch.device] = "cpu"
    ) -> Tuple[np.ndarray, float]:
        """
        Inference helper matching the stub_network contract:
            input:  state_tensor [13, 8, 8] (float32 numpy array)
            output: (policy_logits: [4096] float32 numpy array, value: float in [-1, 1])
        """
        if self.training:
            self.eval()
        if state_tensor.shape != (13, 8, 8):
            raise ValueError(f"Expected state tensor shape (13, 8, 8), got {state_tensor.shape}")

        t = torch.from_numpy(state_tensor).to(device=device, dtype=torch.float32)
        policy_logits, value = self.forward(t)
        return policy_logits.cpu().numpy().astype(np.float32), float(value.item())


class MovesLeftHead(nn.Module):
    """
    Auxiliary head used only in training (from KataGo): predicts how many
    plies the game still lasts, in units of 100, from the backbone features.
    Its loss shapes the shared backbone; it is not saved with the network
    and plays no part in search.
    """

    def __init__(self, num_filters: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        m = F.relu(self.bn(self.conv(features)), inplace=True).flatten(start_dim=1)
        return self.fc2(F.relu(self.fc1(m), inplace=True)).squeeze(1)
