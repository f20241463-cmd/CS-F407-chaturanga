"""
compare_checkpoints.py

Benchmark: a newer checkpoint vs an older one, both playing through
MCTS with the same simulation budget. This is the "is generation N
actually stronger than generation N-1?" check, which win rate against
random play alone can't answer once both beat random comfortably.

Colors alternate every game, same as evaluate_vs_random.py, so a
first-move advantage can't favor either side.

--old can also be the word "untrained", meaning a fresh ChaturangaNet
seeded from --seed -- handy for comparing the first trained checkpoint
against the starting point.

Usage:
    python compare_checkpoints.py --new checkpoints/iter_002.pt --old checkpoints/iter_001.pt
    python compare_checkpoints.py --new checkpoints/iter_001.pt --old untrained --games 40
"""

import argparse
import math
import random
import time

from evaluate_vs_random import make_real_network_fn, mcts_move, play_one_game


def load_side(path, seed):
    if path == "untrained":
        return make_real_network_fn(seed=seed)
    return make_real_network_fn(checkpoint=path)


def score_summary(new_wins, old_wins, draws):
    """
    Score = (wins + 0.5 * draws) / games, the usual chess-style match
    score. Also returns a rough 95% interval on it (normal approximation)
    so a 55% score over 20 games isn't mistaken for a real improvement.
    """
    games = new_wins + old_wins + draws
    if games == 0:
        return 0.0, 0.0
    score = (new_wins + 0.5 * draws) / games
    per_game = [1.0] * new_wins + [0.0] * old_wins + [0.5] * draws
    var = sum((x - score) ** 2 for x in per_game) / games
    return score, 1.96 * math.sqrt(var / games)


def main():
    parser = argparse.ArgumentParser(description="Newer checkpoint vs older checkpoint")
    parser.add_argument("--new", required=True, help="path to the newer checkpoint")
    parser.add_argument(
        "--old", required=True, help='path to the older checkpoint, or "untrained"'
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--simulations", type=int, default=30, help="MCTS simulations per move"
    )
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--max_plies", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print every game's result"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    new_fn = load_side(args.new, args.seed)
    old_fn = load_side(args.old, args.seed)

    def player(network_fn):
        return lambda state, rng: mcts_move(
            state, rng, args.simulations, args.c_puct, network_fn
        )

    new_player, old_player = player(new_fn), player(old_fn)

    new_wins = old_wins = draws = unfinished = 0
    ply_counts = []
    start = time.time()

    for game_num in range(args.games):
        new_plays_white = game_num % 2 == 0
        if new_plays_white:
            white_fn, black_fn = new_player, old_player
        else:
            white_fn, black_fn = old_player, new_player

        outcome, plies = play_one_game(white_fn, black_fn, rng, args.max_plies)
        ply_counts.append(plies)

        if outcome == "draw":
            draws += 1
        elif outcome == "unfinished":
            unfinished += 1
        elif (outcome == "white") == new_plays_white:
            new_wins += 1
        else:
            old_wins += 1

        if args.verbose:
            side = "White" if new_plays_white else "Black"
            print(
                f"  game {game_num + 1:3d}: new played {side:5s} -> {outcome:10s} ({plies} plies)"
            )

    elapsed = time.time() - start
    total = args.games
    decided = new_wins + old_wins
    # Unfinished games count as draws for the match score.
    score, margin = score_summary(new_wins, old_wins, draws + unfinished)

    print()
    print(f"New:               {args.new}")
    print(f"Old:               {args.old}")
    print(f"Games played:      {total}")
    print(f"New wins:          {new_wins}")
    print(f"Old wins:          {old_wins}")
    print(f"Draws:             {draws}")
    print(
        f"Unfinished:        {unfinished}"
        + ("  <- raise --max_plies, these aren't real draws" if unfinished else "")
    )
    if decided:
        print(f"New win rate among decided games: {100 * new_wins / decided:.1f}%")
    print(f"New match score:   {100 * score:.1f}% (+/- {100 * margin:.1f}%, ~95%)")
    if total and score - margin > 0.5:
        print("  -> new checkpoint is stronger (interval clears 50%)")
    else:
        print("  -> no clear difference yet; play more games to tell")
    print(f"Avg game length:   {sum(ply_counts) / max(len(ply_counts), 1):.1f} plies")
    print(f"Elapsed:           {elapsed:.1f}s")


if __name__ == "__main__":
    main()
