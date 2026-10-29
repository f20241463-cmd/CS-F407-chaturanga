"""
Inference agent and bridge between ChaturangaNet and the Environment / MCTS.

Provides:
  - make_network_fn: wraps a ChaturangaNet into a Callable(state_tensor) -> (logits, value)
    matching chaturanga.stub_network exactly.
  - make_batch_network_fn: the same for a batch of states, for batched self-play.
  - choose_network_move: selects a legal move using masked policy logits.
"""

import copy
from typing import Callable, Optional, Tuple
import numpy as np
import torch

from chaturanga.encoding import action_to_move, legal_action_mask, state_to_tensor
from chaturanga.game import GameState
from .model import ChaturangaNet

# (convolution, batch norm) attribute pairs in ChaturangaNet and ResidualBlock.
_CONV_BN_PAIRS = [
    ("input_conv", "input_bn"),
    ("conv1", "bn1"),
    ("conv2", "bn2"),
    ("policy_conv", "policy_bn"),
    ("value_conv", "value_bn"),
]


def _fuse_conv_bn(net: torch.nn.Module) -> torch.nn.Module:
    """
    Fold every batch norm into the convolution before it (eval mode only).
    In eval mode a batch norm is a fixed per-channel scale and shift, so
    conv followed by bn equals one conv with adjusted weights and a bias.
    The batch norm is replaced by Identity, so forward() is unchanged.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    for module in net.modules():
        for conv_name, bn_name in _CONV_BN_PAIRS:
            conv = getattr(module, conv_name, None)
            bn = getattr(module, bn_name, None)
            if isinstance(conv, torch.nn.Conv2d) and isinstance(
                bn, torch.nn.BatchNorm2d
            ):
                setattr(module, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, torch.nn.Identity())
    return net


def make_network_fn(
    model: ChaturangaNet, device: str = "cpu"
) -> Callable[[np.ndarray], Tuple[np.ndarray, float]]:
    """
    Returns a function conforming to chaturanga/stub_network.py contract:
        fn(state_tensor) -> (policy_logits: [4096] float32, value: float in [-1, 1])
    This is what Student 3 (MCTS) and Student 4 (Self-Play) plug directly into their pipelines.

    The function evaluates a copy of `model` taken now, so training `model`
    afterwards does not change it (make a new one after each update). For
    speed, the copy has its batch norms folded into the convolutions and,
    on CPU, uses the channels-last memory layout (together about 2x faster
    for this network); outputs match the original to float rounding (~1e-5).
    """
    net = _fuse_conv_bn(copy.deepcopy(model).to(device).eval())
    if torch.device(device).type == "cpu":
        net = net.to(memory_format=torch.channels_last)

    def network_fn(state_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        if state_tensor.shape != (13, 8, 8):
            raise ValueError(
                f"Expected state tensor shape (13, 8, 8), got {state_tensor.shape}"
            )
        x = torch.from_numpy(state_tensor).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            policy_logits, value = net(x.unsqueeze(0))
        return (
            policy_logits[0].cpu().numpy().astype(np.float32),
            float(value.reshape(-1)[0]),
        )

    return network_fn


class BatchNetwork:
    """
    Evaluates a batch of states: net(float32[B, 13, 8, 8]) returns
    (policy_logits float32[B, 4096], values float32[B]). Used to evaluate
    the positions of many self-play games in one call, which is what makes
    a GPU fast.

    submit(states) starts the evaluation and returns at once (GPU work runs
    in the background); fetch(handle) waits for it and returns the outputs.
    Between the two the caller can do other work, such as advancing other
    games. Batches smaller than cpu_below go to a CPU copy instead, since
    one GPU call has a fixed cost that a few positions do not repay.
    Like make_network_fn, it evaluates a copy of the model taken now.
    """

    def __init__(self, model: ChaturangaNet, device: str = "cpu", cpu_below: int = 0):
        self.device = device
        self.net = _fuse_conv_bn(copy.deepcopy(model).to(device).eval())
        self.cpu_below = cpu_below
        self.cpu_fn = make_network_fn(model) if cpu_below > 0 else None

    def submit(self, states: np.ndarray):
        if states.ndim != 4 or states.shape[1:] != (13, 8, 8):
            raise ValueError(f"Expected shape (B, 13, 8, 8), got {states.shape}")
        if len(states) < self.cpu_below:
            outputs = [self.cpu_fn(s) for s in states]
            return (
                np.stack([p for p, _ in outputs]),
                np.array([v for _, v in outputs], dtype=np.float32),
            )
        x = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32))
        with torch.inference_mode():
            policy_logits, values = self.net(x.to(self.device, non_blocking=True))
            if self.device == "cpu":
                return policy_logits.numpy(), values.reshape(-1).numpy()
            policy_logits = policy_logits.to("cpu", non_blocking=True)
            values = values.reshape(-1).to("cpu", non_blocking=True)
        event = None
        if torch.device(self.device).type == "mps":
            event = torch.mps.event.Event()
            event.record()
        return policy_logits, values, event

    def fetch(self, handle) -> Tuple[np.ndarray, np.ndarray]:
        if len(handle) == 2:  # evaluated on the CPU already
            return handle
        policy_logits, values, event = handle
        if event is not None:
            event.synchronize()
        elif policy_logits.device.type == "cuda":
            torch.cuda.synchronize()
        return policy_logits.numpy(), values.numpy()

    def __call__(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.fetch(self.submit(states))


def make_batch_network_fn(
    model: ChaturangaNet, device: str = "cpu", cpu_below: int = 0
) -> BatchNetwork:
    """A BatchNetwork for model on device (see BatchNetwork)."""
    return BatchNetwork(model, device=device, cpu_below=cpu_below)


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
