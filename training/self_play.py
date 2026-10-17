"""
training/self_play.py -- generate training data by having MCTS play itself.

One call to play_self_play_game plays a full game where both sides use
MCTS guided by the same network, with Dirichlet noise at the root so
games differ from each other. Every position becomes one training
example:

    (state_tensor, policy_target, value)

    policy_target = MCTS visit counts at that position, normalized to sum to 1
    value         = the final game result from the point of view of the
                    player who was to move at that position (+1 won,
                    -1 lost, draw_value for a draw or an unfinished game)

No torch import: network_fn is any callable honoring the
tensor -> (policy_logits, value) contract, so this also runs with the
stub networks.
"""

import numpy as np

from chaturanga import encoding
from chaturanga.game import GameState
from chaturanga.types import Color
from mcts.search import run_mcts, select_action_from_visits


def outcome_values(movers, result, final_side_to_move):
    """
    Turn one final result into a value for every recorded position.

    result is from final_side_to_move's perspective (GameState.result()
    convention). A position where that same side was to move gets
    result; a position where the other side was to move gets -result.
    """
    return [
        float(result) if mover == final_side_to_move else -float(result)
        for mover in movers
    ]


def play_self_play_game(
    network_fn,
    num_simulations=30,
    c_puct=1.5,
    temperature_moves=30,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    max_plies=300,
    draw_value=0.0,
    rng=None,
):
    """
    Returns (examples, info).

    examples: list of (state_tensor float32[13,8,8], policy float32[4096], value float)
    info:     {"plies": int, "outcome": "white" | "black" | "draw" | "unfinished"}

    For the first temperature_moves plies the move is sampled in
    proportion to visit counts (temperature 1); after that the most
    visited move is played. The policy target is always the normalized
    visit counts, whichever way the move was chosen.
    """
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")
    rng = rng if rng is not None else np.random.default_rng()

    state = GameState()
    records = []  # (tensor, policy, side_to_move) per ply
    plies = 0

    while not state.is_game_over() and plies < max_plies:
        visit_counts, _ = run_mcts(
            state,
            network_fn,
            num_simulations,
            c_puct,
            encoding.state_to_tensor,
            encoding.legal_action_mask,
            encoding.action_to_move,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            rng=rng,
        )
        policy = (visit_counts / visit_counts.sum()).astype(np.float32)
        records.append((encoding.state_to_tensor(state), policy, state.side_to_move))

        temperature = 1.0 if plies < temperature_moves else 0.0
        action = select_action_from_visits(visit_counts, temperature, rng)
        state = state.apply_move(encoding.action_to_move(action))
        plies += 1

    if not state.is_game_over():
        outcome = "unfinished"
        values = [float(draw_value)] * len(records)
    else:
        result = state.result()
        if result == 0:
            outcome = "draw"
            values = [float(draw_value)] * len(records)
        else:
            movers = [mover for _, _, mover in records]
            values = outcome_values(movers, result, state.side_to_move)
            winner = state.side_to_move if result > 0 else state.side_to_move.opponent
            outcome = "white" if winner == Color.WHITE else "black"

    examples = [
        (tensor, policy, value) for (tensor, policy, _), value in zip(records, values)
    ]
    return examples, {"plies": plies, "outcome": outcome}
