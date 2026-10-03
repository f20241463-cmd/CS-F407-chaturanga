"""
Inference agent and bridge between ChaturangaNet and the Environment / MCTS.

Provides:
  - make_network_fn: wraps a ChaturangaNet into a Callable(state_tensor) -> (logits, value)
    matching chaturanga.stub_network exactly.
  - choose_network_move: selects a legal move using masked policy logits.
"""

from typing import Callable, Optional, Tuple
import numpy as np
import torch

from chaturanga.encoding import action_to_move, legal_action_mask, state_to_tensor
from chaturanga.game import GameState
from .model import ChaturangaNet


def make_network_fn(
    model: ChaturangaNet, device: str = "cpu"
) -> Callable[[np.ndarray], Tuple[np.ndarray, float]]:
    """
    Returns a function conforming to chaturanga/stub_network.py contract:
        fn(state_tensor) -> (policy_logits: [4096] float32, value: float in [-1, 1])
    This is what Student 3 (MCTS) and Student 4 (Self-Play) plug directly into their pipelines.
    """
    model.eval()

    def network_fn(state_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        return model.predict_numpy(state_tensor, device=device)

    return network_fn


def choose_network_move(
    state: GameState,
    model: ChaturangaNet,
    temperature: float = 0.0,
    device: str = "cpu",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[Tuple[int, int]], float]:
    """
    Evaluate a GameState, apply legal-move masking, and choose a move.
    
    Args:
        state: Current GameState
        model: Trained or initialized ChaturangaNet
        temperature: 0.0 for deterministic argmax (greedy best move),
                     > 0.0 for sampling proportional to softmax(logits / temp).
        device: 'cpu' or 'cuda'
        rng: Random generator for sampling
    Returns:
        (move, value_estimate) where move is (from_sq, to_sq) or None if no legal moves.
    """
    tensor = state_to_tensor(state)
    policy_logits, value = model.predict_numpy(tensor, device=device)
    mask = legal_action_mask(state)

    if not mask.any():
        return None, value

    # Mask illegal moves with -infinity before softmax (Technical Interfaces Section 6)
    masked_logits = np.where(mask, policy_logits, -1e9)

    if temperature <= 1e-4:
        # Greedy best move
        action_id = int(np.argmax(masked_logits))
    else:
        # Temperature sampling
        shifted = (masked_logits - np.max(masked_logits)) / max(temperature, 1e-4)
        exp = np.exp(shifted)
        # Re-zero masked out positions
        exp = np.where(mask, exp, 0.0)
        sum_exp = exp.sum()
        if sum_exp > 0:
            probs = exp / sum_exp
        else:
            probs = mask.astype(np.float32) / mask.sum()

        rng = rng or np.random.default_rng()
        action_id = int(rng.choice(len(probs), p=probs))

    move = action_to_move(action_id)
    return move, value
