"""
A fake neural network that honors the real network's contract
(section 2.5) exactly, so Student 3 (MCTS) can build and test search
before Student 2's real ResNet exists. Swapping this for the real
network later should require changing zero lines of MCTS code - only
which function gets passed in.

Contract:
    input:  state tensor, float32 [13, 8, 8]  (see encoding.state_to_tensor)
    output: (policy_logits, value)
        policy_logits : float32 [4096], raw logits over ALL actions,
                        not yet masked or softmaxed (masking is a
                        separate step - see encoding.legal_action_mask)
        value         : python float in [-1, 1], from the perspective
                        of the player to move in the given state

Two variants are provided:
    zero_stub_network   - deterministic, uniform-ish logits, value 0.
                           Good for a first smoke test: is the pipeline
                           wired correctly at all, with no randomness
                           to obscure a bug.
    random_stub_network - random logits and value each call. Good for
                           stress-testing MCTS/masking against noisy,
                           contradictory-looking output, closer to what
                           an untrained real network actually produces.
"""

from typing import Tuple

import numpy as np

from .encoding import NUM_ACTIONS

StubOutput = Tuple[np.ndarray, float]


def zero_stub_network(state_tensor: np.ndarray) -> StubOutput:
    _validate_input(state_tensor)
    policy_logits = np.zeros(NUM_ACTIONS, dtype=np.float32)
    value = 0.0
    return policy_logits, value


def random_stub_network(state_tensor: np.ndarray, rng: np.random.Generator = None) -> StubOutput:
    _validate_input(state_tensor)
    rng = rng if rng is not None else np.random.default_rng()
    policy_logits = rng.standard_normal(NUM_ACTIONS).astype(np.float32)
    value = float(rng.uniform(-1.0, 1.0))
    return policy_logits, value


def _validate_input(state_tensor: np.ndarray) -> None:
    if state_tensor.shape != (13, 8, 8):
        raise ValueError(
            f"Expected state tensor shape (13, 8, 8), got {state_tensor.shape}. "
            "This will also be a bug in the real network's input handling if it "
            "doesn't check for this."
        )
