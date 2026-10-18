"""
evaluation/game_endings.py -- how do games end?

Plays games under the benchmark protocol (50 simulations, temperature 0.3,
repetition avoidance on, 800-ply cap) and counts the rule that ended each one:
checkmate, stalemate, the 100-ply no-progress rule, threefold repetition, a
captured king, or the ply cap. Useful for explaining why most games are draws.

Run from the repo root:
    python -m evaluation.game_endings --checkpoint checkpoints/final_B/latest.pt --opponent random
    python -m evaluation.game_endings --checkpoint untrained --opponent self --games 10
"""

import argparse
import collections

import numpy as np

from chaturanga.game import NO_PROGRESS_LIMIT, GameState
from chaturanga.types import Color
from evaluation.players import greedy_capture_player, make_mcts_player, random_player

ENDINGS = [
    "checkmate",
    "stalemate",
    "no-progress rule",
    "threefold repetition",
    "king captured",
    "unfinished (ply cap)",
]


def classify_ending(state):
    """Name the rule that ended the game at `state` (checked in GameState.result order)."""
    if not state.is_game_over():
        return "unfinished (ply cap)"
    if state.board.find_king_square(state.side_to_move) is None:
        return "king captured"
    if not state.legal_moves():
        return "checkmate" if state.is_in_check() else "stalemate"
    if state.halfmove_clock >= NO_PROGRESS_LIMIT:
        return "no-progress rule"
    return "threefold repetition"


def count_endings(agent, opponent, games, rng, max_plies=800, on_game=None):
    """Play `games` games, agent alternating colors. Returns a Counter of endings."""
    counts = collections.Counter()
    for g in range(games):
        agent_white = g % 2 == 0
        white, black = (agent, opponent) if agent_white else (opponent, agent)
        state, plies = GameState(), 0
        while not state.is_game_over() and plies < max_plies:
            player = white if state.side_to_move == Color.WHITE else black
            state = state.apply_move(player(state, rng))
            plies += 1
        ending = classify_ending(state)
        counts[ending] += 1
        if on_game:
            on_game(g, ending, plies)
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description="Count how games end")
    p.add_argument(
        "--checkpoint", required=True, help="checkpoint path, or 'untrained'"
    )
    p.add_argument(
        "--opponent",
        default="random",
        choices=["random", "greedy", "untrained", "self"],
        help="'self' plays the agent against itself",
    )
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--max_plies", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    # Torch is only needed here, so classify_ending stays importable without it.
    from evaluation.benchmark import untrained_network_fn
    from network.agent import make_network_fn
    from network.checkpoint import load_checkpoint

    def network(spec):
        if spec == "untrained":
            return untrained_network_fn(args.seed)
        model, _, _, _ = load_checkpoint(spec)
        return make_network_fn(model)

    agent = make_mcts_player(network(args.checkpoint))
    opponent = {
        "random": lambda: random_player,
        "greedy": lambda: greedy_capture_player,
        "untrained": lambda: make_mcts_player(network("untrained")),
        "self": lambda: agent,
    }[args.opponent]()

    def show(g, ending, plies):
        print(f"  game {g + 1}: {ending} ({plies} plies)", flush=True)

    counts = count_endings(
        agent,
        opponent,
        args.games,
        np.random.default_rng(args.seed),
        max_plies=args.max_plies,
        on_game=show,
    )
    print(f"\n{args.checkpoint} vs {args.opponent}, {args.games} games:")
    for ending in ENDINGS:
        if counts[ending]:
            n = counts[ending]
            print(f"  {ending:22s} {n:3d}  ({100 * n / args.games:.0f}%)")


if __name__ == "__main__":
    main()
