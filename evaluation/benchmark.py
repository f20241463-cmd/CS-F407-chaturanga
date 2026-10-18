"""
evaluation/benchmark.py -- strength curve across training iterations.

Plays every saved iteration of a training run (checkpoints/iter_NNNN.pt),
plus a freshly initialized network as iteration 0, against one fixed
opponent, using one fixed evaluation protocol. Writes one CSV row per
iteration, so plot_benchmark.py can draw "score vs iteration".

Opponents:
  random     uniformly random legal moves
  greedy     greedy-capture heuristic (see evaluation/players.py)
  untrained  MCTS with a freshly initialized network (seeded with --seed)

The protocol (defaults below) is the same for every iteration, so the
rows are comparable:
  - MCTS with --simulations 50 (the top of the project's 30-50 budget),
    no Dirichlet noise
  - moves sampled from visit counts at --temperature 0.3 for the whole
    game, so games differ but mostly follow the best moves
  - repetition avoidance on (see players.make_mcts_player)
  - colors alternate every game; games stop at --max_plies (unfinished
    games are reported but left out of the score)
  - each iteration plays the same seeded sequence of games

With --workers N the games run in N processes (evaluation/parallel_match.py)
and each game gets its own seed from (--seed, game number) instead of all
games sharing one random stream. Results are then the same for any N, but
differ game by game from a run without --workers, so compare rows made
the same way.

Usage:
    python -m evaluation.benchmark --checkpoint_dir checkpoints --opponent random
    python -m evaluation.benchmark --checkpoint_dir checkpoints --opponent untrained --every 2
    python -m evaluation.benchmark --checkpoint_dir checkpoints/long --opponent untrained \
        --every 20 --games 100 --workers 6
"""

import argparse
import csv
import glob
import os
import re
import time

import numpy as np
import torch

from compare_checkpoints import score_summary
from evaluation.parallel_match import make_pool, mcts_spec, play_match_parallel
from evaluation.players import (
    greedy_capture_player,
    make_mcts_player,
    play_match,
    random_player,
)
from network.agent import make_network_fn
from network.checkpoint import load_checkpoint
from network.model import ChaturangaNet

FIELDS = [
    "iteration",
    "checkpoint",
    "opponent",
    "games",
    "wins",
    "losses",
    "draws",
    "unfinished",
    "score",
    "margin",
    "avg_plies",
    "seconds",
]


def find_iterations(checkpoint_dir):
    """Return [(iteration, path)] for iter_NNNN.pt files, sorted by iteration."""
    found = []
    for path in glob.glob(os.path.join(checkpoint_dir, "iter_*.pt")):
        match = re.search(r"iter_(\d+)\.pt$", path)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def untrained_network_fn(seed):
    torch.manual_seed(seed)
    return make_network_fn(ChaturangaNet())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark every training iteration")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument(
        "--opponent", choices=["random", "greedy", "untrained"], required=True
    )
    p.add_argument("--games", type=int, default=20, help="games per iteration")
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument(
        "--avoid_repetition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip moves that cause a threefold-repetition draw (default on)",
    )
    p.add_argument("--max_plies", type=int, default=800)
    p.add_argument("--every", type=int, default=1, help="only every Nth iteration")
    p.add_argument(
        "--skip_untrained",
        action="store_true",
        help="leave out iteration 0 (the fresh network)",
    )
    p.add_argument(
        "--out", default=None, help="default: results/benchmark_<opponent>.csv"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="play games in this many processes, with per-game seeds (see above)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=None,
        help="only these iterations (0 = the untrained network)",
    )
    args = p.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.out or f"results/benchmark_{args.opponent}.csv"

    iterations = [
        (it, path)
        for it, path in find_iterations(args.checkpoint_dir)
        if it % args.every == 0
    ]
    if not args.skip_untrained and args.opponent != "untrained":
        iterations.insert(0, (0, "untrained"))
    if args.iterations is not None:
        wanted = set(args.iterations)
        everything = [(0, "untrained")] + find_iterations(args.checkpoint_dir)
        iterations = [(it, path) for it, path in everything if it in wanted]
        if args.opponent == "untrained":
            iterations = [(it, path) for it, path in iterations if it != 0]
        if not iterations:
            raise SystemExit(f"none of iterations {args.iterations} found")
    if not iterations:
        raise SystemExit(f"no iter_*.pt checkpoints found in {args.checkpoint_dir}")

    def mcts(network_fn):
        return make_mcts_player(
            network_fn,
            simulations=args.simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
            avoid_repetition=args.avoid_repetition,
        )

    if args.opponent == "random":
        opponent = random_player
    elif args.opponent == "greedy":
        opponent = greedy_capture_player
    else:
        opponent = mcts(untrained_network_fn(args.seed))

    opponent_spec = (
        {"kind": args.opponent}
        if args.opponent != "untrained"
        else mcts_spec("untrained", seed=args.seed, simulations=args.simulations,
                       c_puct=args.c_puct, temperature=args.temperature,
                       avoid_repetition=args.avoid_repetition)
    )  # fmt: skip
    pool = make_pool(args.workers) if args.workers and args.workers > 1 else None

    print(
        f"Benchmark vs {args.opponent}: {len(iterations)} iterations x {args.games} games, "
        f"{args.simulations} sims, temperature {args.temperature}, "
        f"repetition avoidance {'on' if args.avoid_repetition else 'off'}, "
        f"max {args.max_plies} plies -> {out}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    start = time.time()
    try:
        _run(args, out, iterations, mcts, opponent, opponent_spec, pool, start)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _run(args, out, iterations, mcts, opponent, opponent_spec, pool, start):
    write_header = not os.path.exists(out) or os.path.getsize(out) == 0
    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for done, (iteration, path) in enumerate(iterations, start=1):
            if args.workers is None:
                if path == "untrained":
                    network_fn = untrained_network_fn(args.seed)
                else:
                    model, _, _, _ = load_checkpoint(path)
                    network_fn = make_network_fn(model)

            def on_game(g, a_is_white, outcome, plies):
                print(
                    f"    game {g + 1:2d}: agent {'White' if a_is_white else 'Black'} "
                    f"-> {outcome:10s} ({plies} plies)",
                    flush=True,
                )

            print(f"  iteration {iteration} ({path})")
            t0 = time.time()
            # Same seed for every iteration, so each one faces the same game sequence.
            if args.workers is None:
                rng = np.random.default_rng(args.seed)
                counts = play_match(
                    mcts(network_fn), opponent, args.games, rng, args.max_plies, on_game
                )
            else:
                agent_spec = mcts_spec(
                    path, seed=args.seed, simulations=args.simulations,
                    c_puct=args.c_puct, temperature=args.temperature,
                    avoid_repetition=args.avoid_repetition,
                )  # fmt: skip
                counts = play_match_parallel(
                    agent_spec, opponent_spec, args.games, args.seed,
                    args.max_plies, pool, on_game,
                )  # fmt: skip
            finished = counts["wins"] + counts["losses"] + counts["draws"]
            if finished:
                score, margin = score_summary(
                    counts["wins"], counts["losses"], counts["draws"]
                )
                score_text = f"score {score:.3f} +/- {margin:.3f}"
                score, margin = round(score, 4), round(margin, 4)
            else:
                # No finished games: a score of 0 would read as "lost them all".
                score = margin = ""
                score_text = "no finished games, no score"
            row = {
                "iteration": iteration,
                "checkpoint": path,
                "opponent": args.opponent,
                "games": args.games,
                "wins": counts["wins"],
                "losses": counts["losses"],
                "draws": counts["draws"],
                "unfinished": counts["unfinished"],
                "score": score,
                "margin": margin,
                "avg_plies": round(sum(counts["plies"]) / len(counts["plies"]), 1),
                "seconds": round(time.time() - t0, 1),
            }
            writer.writerow(row)
            f.flush()

            elapsed = time.time() - start
            remaining = (len(iterations) - done) * elapsed / done
            print(
                f"  iteration {iteration}: W {row['wins']} / L {row['losses']} / "
                f"D {row['draws']} / unfinished {row['unfinished']}, "
                f"{score_text}. "
                f"Estimated time left: {remaining / 60:.0f} min",
                flush=True,
            )


if __name__ == "__main__":
    main()
