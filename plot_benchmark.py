"""
plot_benchmark.py

Strength curve from one or more CSVs written by python -m evaluation.benchmark:
the agent's match score (win = 1, draw = 0.5, unfinished games left out)
at each training iteration, with the rough 95% margin as an error bar.
Iteration 0 is the untrained network. The dashed line at 0.5 means
"even with the opponent".

Usage:
    python plot_benchmark.py --csvs results/benchmark_random.csv results/benchmark_greedy.csv
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_training_curves import (  # noqa: E402
    GRID,
    SERIES_COLORS,
    SERIES_MARKERS,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    style_axis,
)

OPPONENT_NAMES = {
    "random": "vs random",
    "greedy": "vs greedy-capture",
    "untrained": "vs untrained network",
}


def read_rows(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["score"] not in ("", None)]
    if not rows:
        raise ValueError(f"{path} has no scored rows")
    # If an iteration was benchmarked twice, keep the last row.
    by_iteration = {int(r["iteration"]): r for r in rows}
    return [by_iteration[it] for it in sorted(by_iteration)], rows[0]["opponent"]


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark score vs iteration")
    parser.add_argument("--csvs", nargs="+", required=True)
    parser.add_argument("--out", default="results/benchmark_curve.png")
    parser.add_argument("--title", default="Agent strength across training iterations")
    args = parser.parse_args()
    if len(args.csvs) > len(SERIES_COLORS):
        parser.error(f"at most {len(SERIES_COLORS)} CSVs per figure")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    style_axis(ax, "Match score (win = 1, draw = 0.5), with rough 95% range")
    ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        "even",
        (0, 0.5),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )

    for i, path in enumerate(args.csvs):
        rows, opponent = read_rows(path)
        xs = [int(r["iteration"]) for r in rows]
        ys = [float(r["score"]) for r in rows]
        errs = [float(r["margin"]) for r in rows]
        label = OPPONENT_NAMES.get(opponent, opponent)
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            color=SERIES_COLORS[i],
            linewidth=2,
            marker=SERIES_MARKERS[i],
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            elinewidth=1,
            capsize=3,
            label=label,
        )
        ax.annotate(
            label,
            (xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=TEXT_SECONDARY,
        )

    ax.set_ylim(0, 1)
    ax.set_xlabel(
        "Training iteration (0 = untrained)", color=TEXT_SECONDARY, fontsize=9
    )
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.spines["left"].set_color(GRID)
    if len(args.csvs) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, loc="upper left")
    fig.suptitle(args.title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
