"""
Checkpoint saving and loading utilities (Student 2 Deliverable).

Conforms to Technical Interfaces Section 12:
  - Model parameters (state_dict)
  - Optimizer state
  - Training iteration / step
  - Model configuration metadata
"""

import os
from typing import Any, Dict, Optional, Tuple
import torch

from .model import ChaturangaNet


def save_checkpoint(
    path: str,
    model: ChaturangaNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model parameters, optimizer state, and training metadata to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": model.in_channels,
            "num_res_blocks": model.num_res_blocks,
            "num_filters": model.num_filters,
            "num_actions": model.num_actions,
        },
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[ChaturangaNet] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Tuple[ChaturangaNet, Optional[torch.optim.Optimizer], int, Dict[str, Any]]:
    """
    Load model parameters and optimizer state from a checkpoint file.
    If `model` is None, reconstructs a new ChaturangaNet instance using saved config.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", {})

    if model is None:
        model = ChaturangaNet(
            in_channels=config.get("in_channels", 13),
            num_res_blocks=config.get("num_res_blocks", 4),
            num_filters=config.get("num_filters", 64),
            num_actions=config.get("num_actions", 4096),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint.get("step", 0)
    metadata = checkpoint.get("metadata", {})
    return model, optimizer, step, metadata
