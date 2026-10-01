#!/usr/bin/env python3
"""
A small preview of the Week 3 integration goal:

    Environment -> [13,8,8] -> "Neural Network" -> policy + value
                -> mask illegal actions -> softmax -> chosen move
                -> Environment -> new state

Uses the stub network (not Student 2's real one) so this can run and
be understood today, with zero dependency on anyone else's code. When
the real network exists, only the `network_fn` passed to
`choose_move_via_policy` needs to change - everything else in this
pipeline stays identical, which is the whole point of freezing the
contracts in Week 0.

Run:
    python integration_preview.py
"""

import numpy as np

from chaturanga import (
    GameState,
    action_to_move,
    legal_action_mask,
    move_to_algebraic,
    random_stub_network,
    state_to_tensor,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def choose_move_via_policy(state: GameState, network_fn, rng: np.random.Generator):
    """The Environment -> Network -> mask -> move pipeline, step by step."""
    tensor = state_to_tensor(state)                    # Environment -> [13,8,8]
    policy_logits, value = network_fn(tensor)           # "Network" -> policy + value
    mask = legal_action_mask(state)                     # Environment provides legality

    if not mask.any():
        return None, value  # no legal moves - state should already be terminal

    masked_logits = np.where(mask, policy_logits, -np.inf)
    probs = softmax(masked_logits)                      # mask -> softmax
    action_id = rng.choice(len(probs), p=probs)
    move = action_to_move(action_id)                    # policy -> move
    return move, value


def main():
    rng = np.random.default_rng(seed=0)
    state = GameState()
    half_moves = 0
    max_moves = 200

    print("Environment -> tensor -> stub network -> masked policy -> move\n")

    while not state.is_game_over() and half_moves < max_moves:
        move, value = choose_move_via_policy(state, random_stub_network, rng)
        print(
            f"{half_moves + 1:>3}. {state.side_to_move.name:<5} "
            f"plays {move_to_algebraic(move):<5} "
            f"(stub value estimate: {value:+.2f})"
        )
        state = state.apply_move(move)
        half_moves += 1

    print()
    if state.is_game_over():
        r = state.result()
        print(f"Game over after {half_moves} half-moves. result()={r} "
              f"(from {state.side_to_move.name}'s perspective)")
    else:
        print(f"Hit the {max_moves}-move demo cap without a result "
              f"(expected sometimes - the stub network is just noise, not a real "
              f"player, so games can run long; see play.py --mode random for the "
              f"same behavior).")


if __name__ == "__main__":
    main()
