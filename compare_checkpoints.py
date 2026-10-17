"""
compare_checkpoints.py

Checkpoint vs checkpoint: the test for "is the newer network actually
stronger than the older one?". Both sides play with MCTS (no noise), each
using its own network. Colors alternate every game so neither network
gets a first-move edge.

Without any randomness, two deterministic players would replay the same
game over and over (only two distinct games, one per color assignment).
So for the first --temp_plies plies of each game, both sides SAMPLE their
move from the MCTS visit counts (at --temperature), which gives varied
games; after that, both play the most-visited move.

With greedy play, games between these networks tend to end early in a
threefold-repetition draw, so in practice --temp_plies needs to cover
the whole game (e.g. 1000). --temperature 1.0 then makes play so random
that small strength differences get lost; a lower value such as 0.3
still varies the games but mostly plays the best moves.

--avoid_repetition stops both sides from choosing a move that would
repeat a position for the third time (a threefold-repetition draw),
unless every move MCTS considered would. Without it, near-greedy play
between these networks ends almost every game in a quick repetition
draw, which says nothing about which network is stronger.

Usage:
    python compare_checkpoints.py --a checkpoints/iter_0010.pt --b checkpoints/iter_0001.pt --games 20
    python compare_checkpoints.py --a checkpoints/latest.pt --b untrained --games 20

--b untrained uses a freshly initialized, never-trained network (the
"iteration 0" reference).
"""

import argparse
import math
import time

import numpy as np

from chaturanga.game import GameState
from chaturanga.types import Color
from chaturanga import encoding

from mcts.search import run_mcts, select_action_from_visits


def load_network_fn(spec, seed):
    import torch
    from network.agent import make_network_fn
    from network.model import ChaturangaNet

    if spec == "untrained":
        torch.manual_seed(seed)
        return make_network_fn(ChaturangaNet()), "untrained"

    from network.checkpoint import load_checkpoint

    model, _, step, _ = load_checkpoint(spec)
    return make_network_fn(model), f"{spec} (step {step})"


def score_summary(a_wins, b_wins, draws):
    """
    A's match score = (wins + 0.5 * draws) / finished games (unfinished
    games are left out of all three counts), plus a rough 95% margin on
    it (normal approximation), so a 55% score over 20 games isn't
    mistaken for a real improvement.
    """
    games = a_wins + b_wins + draws
    if games == 0:
        return 0.0, 0.0
    score = (a_wins + 0.5 * draws) / games
    per_game = [1.0] * a_wins + [0.0] * b_wins + [0.5] * draws
    var = sum((x - score) ** 2 for x in per_game) / games
    return score, 1.96 * math.sqrt(var / games)


def filter_repeating_moves(state, visit_counts):
    """
    Return (visit_counts, avoided): a copy of visit_counts with every
    visited move that would cause a threefold-repetition draw set to 0.
    If every visited move would repeat, the counts come back unchanged so
    a move can still be chosen. avoided is True if anything was removed.
    """
    filtered = np.array(visit_counts, dtype=np.float32)
    for action in np.flatnonzero(filtered):
        child = state.apply_move(encoding.action_to_move(int(action)))
        # GameState has no public repetition check; result() == 0 alone
        # would also match stalemate and the no-progress rule.
        if child._is_threefold_repetition():
            filtered[action] = 0.0
    if filtered.sum() == 0:
        return np.asarray(visit_counts, dtype=np.float32), False
    return filtered, bool((filtered != visit_counts).any())


def play_game(
    white_fn,
    black_fn,
    simulations,
    c_puct,
    temp_plies,
    max_plies,
    rng,
    temperature=1.0,
    avoid_repetition=False,
):
    """Returns ("white" | "black" | "draw" | "unfinished", plies, repetitions_avoided)."""
    state = GameState()
    plies = 0
    avoided = 0
    while not state.is_game_over() and plies < max_plies:
        network_fn = white_fn if state.side_to_move == Color.WHITE else black_fn
        visit_counts, _ = run_mcts(
            root_state=state,
            network_fn=network_fn,
            num_simulations=simulations,
            c_puct=c_puct,
            state_to_tensor_fn=encoding.state_to_tensor,
            legal_action_mask_fn=encoding.legal_action_mask,
            action_to_move_fn=encoding.action_to_move,
        )
        if avoid_repetition:
            visit_counts, did_avoid = filter_repeating_moves(state, visit_counts)
            avoided += did_avoid
        temp = temperature if plies < temp_plies else 0.0
        action = select_action_from_visits(visit_counts, temp, rng)
        state = state.apply_move(encoding.action_to_move(action))
        plies += 1

    if not state.is_game_over():
        return "unfinished", plies, avoided

    result = state.result()
    if result == 0:
        return "draw", plies, avoided
    winner = state.side_to_move if result > 0 else state.side_to_move.opponent
    return winner.name.lower(), plies, avoided


def main():
    parser = argparse.ArgumentParser(description="Checkpoint A vs checkpoint B")
    parser.add_argument(
        "--a", required=True, help="checkpoint path (the one being tested)"
    )
    parser.add_argument("--b", required=True, help="checkpoint path, or 'untrained'")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=30)
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument(
        "--temp_plies",
        type=int,
        default=8,
        help="sample moves for this many opening plies to vary the games",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="sampling temperature during the first --temp_plies plies: "
        "visits ** (1 / temperature). Lower values play closer to the best move.",
    )
    parser.add_argument(
        "--avoid_repetition",
        action="store_true",
        help="never pick a move that causes a threefold-repetition draw "
        "unless every considered move does",
    )
    parser.add_argument("--max_plies", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fn_a, name_a = load_network_fn(args.a, args.seed)
    fn_b, name_b = load_network_fn(args.b, args.seed)
    print(f"A = {name_a}\nB = {name_b}")
    print(
        f"{args.games} games, {args.simulations} sims/move, temperature "
        f"{args.temperature} for the first {args.temp_plies} plies, "
        f"max {args.max_plies} plies"
        + (", avoiding repetition draws" if args.avoid_repetition else "")
    )

    a_wins = b_wins = draws = unfinished = 0
    plies_list = []
    total_avoided = 0
    start = time.time()

    for g in range(args.games):
        a_is_white = g % 2 == 0
        white_fn, black_fn = (fn_a, fn_b) if a_is_white else (fn_b, fn_a)
        outcome, plies, avoided = play_game(
            white_fn,
            black_fn,
            args.simulations,
            args.c_puct,
            args.temp_plies,
            args.max_plies,
            rng,
            temperature=args.temperature,
            avoid_repetition=args.avoid_repetition,
        )
        total_avoided += avoided
        plies_list.append(plies)

        if outcome == "draw":
            draws += 1
        elif outcome == "unfinished":
            unfinished += 1
        elif (outcome == "white") == a_is_white:
            a_wins += 1
        else:
            b_wins += 1

        print(
            f"  game {g + 1:3d}: A played {'White' if a_is_white else 'Black'} "
            f"-> {outcome:10s} ({plies} plies)",
            flush=True,
        )

    finished = args.games - unfinished
    print()
    print(f"Games:       {args.games}")
    print(f"A wins:      {a_wins}")
    print(f"B wins:      {b_wins}")
    print(f"Draws:       {draws}")
    print(f"Unfinished:  {unfinished}")
    if finished:
        score, margin = score_summary(a_wins, b_wins, draws)
        print(
            f"A score (win=1, draw=0.5, unfinished excluded): "
            f"{score:.3f} +/- {margin:.3f}  (0.5 = equal strength)"
        )
    print(f"Avg length:  {sum(plies_list) / len(plies_list):.1f} plies")
    if args.avoid_repetition:
        print(f"Repetition draws avoided: {total_avoided} moves")
    print(f"Elapsed:     {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
