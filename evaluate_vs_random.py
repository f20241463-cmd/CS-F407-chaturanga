"""
evaluate_vs_random.py

Benchmark: MCTS (using the untrained random stub network) vs pure
random move selection. This is the "is search adding value at all,
even before any real network exists" check -- exactly the kind of
result that belongs in a Milestone 4 writeup.

Each game alternates who plays White, so a 100-game run is really 50
games with MCTS as White + 50 with MCTS as Black. That cancels out any
first-move advantage instead of letting it bias the win rate.

Usage:
    python evaluate_vs_random.py --games 100 --simulations 30
    python evaluate_vs_random.py --games 20 --simulations 50 --seed 7 -v
"""

import argparse
import random
import time
from functools import partial

import numpy as np

from chaturanga.game import GameState
from chaturanga.types import Color
from chaturanga import encoding
from chaturanga.stub_network import random_stub_network

from mcts.search import run_mcts


def random_move(state, rng):
    return rng.choice(state.legal_moves())


def make_seeded_network_fn(seed):
    """
    random_stub_network(tensor, rng=None) draws from its own internal
    randomness unless given an explicit numpy Generator. Without this,
    --seed only controls the random-move baseline, NOT what the stub
    network predicts, so two runs of the identical command silently
    produce different games. This wraps it with a Generator seeded from
    our own --seed, via functools.partial, so run_mcts can keep calling
    it as a plain one-argument network_fn(tensor) (the real trained
    network's contract) while still being fully reproducible underneath.
    """
    stub_rng = np.random.default_rng(seed)
    return partial(random_stub_network, rng=stub_rng)


def mcts_move(state, rng, num_simulations, c_puct, network_fn):
    _, move = run_mcts(
        root_state=state,
        network_fn=network_fn,
        num_simulations=num_simulations,
        c_puct=c_puct,
        state_to_tensor_fn=encoding.state_to_tensor,
        legal_action_mask_fn=encoding.legal_action_mask,
        action_to_move_fn=encoding.action_to_move,
    )
    if move is None:
        # Defensive fallback -- shouldn't happen mid-game, only if the
        # root state handed in was already terminal.
        return rng.choice(state.legal_moves())
    return move


def play_one_game(white_fn, black_fn, rng, max_plies=600):
    """
    white_fn/black_fn: functions (state, rng) -> move.
    Returns one of "white", "black", "draw", "unfinished".

    "draw" means the environment itself declared a draw (stalemate,
    repetition, no-progress limit -- state.is_game_over() was True
    with result() of 0). "unfinished" means we hit max_plies before
    the game ended naturally -- NOT the same thing, and reported
    separately so it can't quietly inflate the draw rate. Her own
    play.py makes this same distinction for the same reason.
    """
    state = GameState()
    plies = 0

    while not state.is_game_over() and plies < max_plies:
        mover = white_fn if state.side_to_move == Color.WHITE else black_fn
        move = mover(state, rng)
        state = state.apply_move(move)
        plies += 1

    if not state.is_game_over():
        return "unfinished", plies

    result = state.result()  # from the perspective of state.side_to_move
    if result is None or result == 0:
        return "draw", plies

    loser = state.side_to_move  # side to move with no good options lost
    winner_color = Color.BLACK if loser == Color.WHITE else Color.WHITE
    return ("white" if winner_color == Color.WHITE else "black"), plies


def main():
    parser = argparse.ArgumentParser(
        description="MCTS (random stub) vs pure random baseline"
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--simulations", type=int, default=30, help="MCTS simulations per move"
    )
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument(
        "--max_plies",
        type=int,
        default=600,
        help="her random-vs-random baseline averages ~487 plies/game, "
        "with some over 650 -- keep this comfortably above that",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print every game's result"
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=5,
        help="print elapsed/ETA after every N games (0 to disable)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    network_fn = make_seeded_network_fn(
        args.seed
    )  # same --seed now also controls the stub network

    mcts_wins = 0
    random_wins = 0
    draws = 0
    unfinished = 0
    ply_counts = []

    start = time.time()

    for game_num in range(args.games):
        mcts_plays_white = game_num % 2 == 0

        def mcts_fn(state, rng):
            return mcts_move(state, rng, args.simulations, args.c_puct, network_fn)

        if mcts_plays_white:
            white_fn, black_fn = mcts_fn, random_move
        else:
            white_fn, black_fn = random_move, mcts_fn

        outcome, plies = play_one_game(white_fn, black_fn, rng, args.max_plies)
        ply_counts.append(plies)

        if outcome == "draw":
            draws += 1
        elif outcome == "unfinished":
            unfinished += 1
        elif (outcome == "white" and mcts_plays_white) or (
            outcome == "black" and not mcts_plays_white
        ):
            mcts_wins += 1
        else:
            random_wins += 1

        if args.verbose:
            side = "White" if mcts_plays_white else "Black"
            print(
                f"  game {game_num + 1:3d}: MCTS played {side:5s} -> {outcome:10s} ({plies} plies)"
            )
        elif args.progress_every and (game_num + 1) % args.progress_every == 0:
            elapsed_so_far = time.time() - start
            per_game = elapsed_so_far / (game_num + 1)
            remaining = (args.games - game_num - 1) * per_game
            print(
                f"  ...{game_num + 1}/{args.games} games, "
                f"{elapsed_so_far:.0f}s elapsed, ~{remaining:.0f}s remaining"
            )

    elapsed = time.time() - start
    total = args.games
    avg_plies = sum(ply_counts) / len(ply_counts) if ply_counts else 0.0
    decided = mcts_wins + random_wins

    print()
    print(f"Games played:      {total}")
    print(f"MCTS wins:         {mcts_wins}  ({100 * mcts_wins / total:.1f}%)")
    print(f"Random wins:       {random_wins}  ({100 * random_wins / total:.1f}%)")
    print(f"Draws:             {draws}  ({100 * draws / total:.1f}%)")
    print(
        f"Unfinished:        {unfinished}  ({100 * unfinished / total:.1f}%)"
        + ("  <- raise --max_plies, these aren't real draws" if unfinished else "")
    )
    if decided:
        print(f"MCTS win rate among decided games: {100 * mcts_wins / decided:.1f}%")
    print(f"Avg game length:   {avg_plies:.1f} plies")
    print(f"Elapsed:           {elapsed:.1f}s ({elapsed / total:.2f}s/game)")


if __name__ == "__main__":
    main()
