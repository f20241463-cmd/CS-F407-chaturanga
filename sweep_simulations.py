"""
sweep_simulations.py

Runs the MCTS-vs-random benchmark at several different --simulations
values and plots decided-game win rate against search depth. This is
the "does search depth actually help" experiment -- a much stronger
claim than any single win-rate number, since it isolates what you're
actually trying to prove.

Reuses play_one_game/mcts_move/random_move from evaluate_vs_random.py
rather than duplicating the game-playing logic, so a fix there doesn't
need to be made twice.

Usage:
    python sweep_simulations.py --sim_values 5 15 30 50 --games_per_point 20
    python sweep_simulations.py --sim_values 5 15 30 50 100 --games_per_point 30 --max_plies 900

Output:
    sweep_results.csv              raw numbers, one row per sim value
    sweep_win_rate_vs_sims.png     the plot
"""

import argparse
import csv
import random
import time

import matplotlib

matplotlib.use("Agg")  # no display available when run headless / in background
import matplotlib.pyplot as plt

from evaluate_vs_random import (
    play_one_game,
    random_move,
    mcts_move,
    make_seeded_network_fn,
)


def run_one_sim_value(num_simulations, games_per_point, max_plies, c_puct, seed):
    rng = random.Random(seed)
    network_fn = make_seeded_network_fn(
        seed
    )  # one seeded stub per sweep point -- reproducible,
    # and each point's randomness is independent of the others
    mcts_wins = random_wins = draws = unfinished = 0

    for game_num in range(games_per_point):
        mcts_plays_white = game_num % 2 == 0

        def mcts_fn(state, rng, _n=num_simulations):
            return mcts_move(state, rng, _n, c_puct, network_fn)

        if mcts_plays_white:
            white_fn, black_fn = mcts_fn, random_move
        else:
            white_fn, black_fn = random_move, mcts_fn

        outcome, _plies = play_one_game(white_fn, black_fn, rng, max_plies)

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

    decided = mcts_wins + random_wins
    win_rate = (mcts_wins / decided) if decided > 0 else None
    return {
        "simulations": num_simulations,
        "games": games_per_point,
        "mcts_wins": mcts_wins,
        "random_wins": random_wins,
        "draws": draws,
        "unfinished": unfinished,
        "decided": decided,
        "win_rate_among_decided": win_rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep --simulations and plot win rate vs search depth"
    )
    parser.add_argument("--sim_values", type=int, nargs="+", default=[5, 15, 30, 50])
    parser.add_argument("--games_per_point", type=int, default=20)
    parser.add_argument("--max_plies", type=int, default=900)
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv_out", type=str, default="sweep_results.csv")
    parser.add_argument("--plot_out", type=str, default="sweep_win_rate_vs_sims.png")
    args = parser.parse_args()

    results = []
    overall_start = time.time()

    for i, sims in enumerate(args.sim_values):
        print(
            f"[{i + 1}/{len(args.sim_values)}] Running {args.games_per_point} games at "
            f"simulations={sims} ..."
        )
        t0 = time.time()
        row = run_one_sim_value(
            sims,
            args.games_per_point,
            args.max_plies,
            args.c_puct,
            seed=args.seed + sims,
        )  # different seed per point, still reproducible
        elapsed = time.time() - t0
        results.append(row)

        wr = row["win_rate_among_decided"]
        wr_str = f"{100 * wr:.1f}%" if wr is not None else "n/a (0 decided games)"
        print(
            f"    wins={row['mcts_wins']} losses={row['random_wins']} draws={row['draws']} "
            f"unfinished={row['unfinished']}  win_rate={wr_str}  ({elapsed:.0f}s)"
        )

    total_elapsed = time.time() - overall_start
    print(f"\nTotal sweep time: {total_elapsed:.0f}s")

    # Write CSV
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Wrote {args.csv_out}")

    # Plot: win rate vs simulations, only for points with at least one decided game
    plot_sims = [
        r["simulations"] for r in results if r["win_rate_among_decided"] is not None
    ]
    plot_rates = [
        100 * r["win_rate_among_decided"]
        for r in results
        if r["win_rate_among_decided"] is not None
    ]
    plot_decided = [
        r["decided"] for r in results if r["win_rate_among_decided"] is not None
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(plot_sims, plot_rates, marker="o", linewidth=2)
    ax.axhline(
        50,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="50% (no advantage over random)",
    )

    # Color/size each point by how many decided games it's actually built
    # on, and flag the thin ones -- a point resting on 1-2 decided games
    # is noise, not signal, and the plot should say so rather than let a
    # single lucky/unlucky game look as credible as a 20-game point.
    for x, y, n in zip(plot_sims, plot_rates, plot_decided):
        label = f"n={n}" + (" (!)" if n < 5 else "")
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            fontsize=8,
            ha="center",
        )

    ax.set_xlabel("MCTS simulations per move")
    ax.set_ylabel("MCTS win rate among decided games (%)")
    ax.set_title(
        "Does search depth improve play?\n(random stub network, vs. random baseline)",
        pad=14,
    )
    ax.set_ylim(
        0, 112
    )  # headroom above 100 so labels on points at 100% don't collide with the title
    ax.text(
        0.01,
        0.99,
        "(!) = fewer than 5 decided games at this point -- noisy, not reliable",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        color="gray",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=150)
    print(f"Wrote {args.plot_out}")


if __name__ == "__main__":
    main()
