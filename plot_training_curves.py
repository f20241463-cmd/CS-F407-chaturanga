"""
plot_training_curves.py

Learning curves from one or more training logs written by
python -m training.train (one CSV row per iteration).

Four small panels, one line per run, so runs can be compared directly:
  policy loss | value loss | decisive games (%) | average game length

Policy and value loss get separate panels because their scales differ
by about 100x; putting both on one axis would flatten the value loss.

Usage:
    python plot_training_curves.py --logs results/training_log.csv
    python plot_training_curves.py \\
        --logs results/training_log.csv checkpoints/material_long/log.csv \\
        --labels "draws = 0" "material targets" \\
        --out results/training_curves.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# First three slots of the reference categorical palette. These three pass
# the colorblind checks for any pair, so up to three runs can share a panel.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_MARKERS = ["o", "s", "^"]  # second encoding, so identity is not color alone
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

PANELS = [
    ("policy_loss", "Policy loss (cross-entropy)", 1.0),
    ("value_loss", "Value loss (MSE)", 1.0),
    ("decisive_fraction", "Decisive self-play games (%)", 100.0),
    ("avg_plies", "Average game length (plies)", 1.0),
]


def read_log(path):
    """Return {field: [floats]} plus 'iteration'; blank cells (no training yet) become None."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    data = {}
    for field in rows[0]:
        values = []
        for row in rows:
            cell = row[field]
            values.append(float(cell) if cell not in ("", None) else None)
        data[field] = values
    return data


def style_axis(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=TEXT_PRIMARY, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_xlabel("Iteration", color=TEXT_SECONDARY, fontsize=9)


def add_end_labels(ax, items, min_gap_pt=11):
    """Direct labels at each line's end, in text ink, pushed apart so they never overlap."""
    to_pt = 72.0 / ax.figure.dpi
    placed = []
    for label, x, y in items:
        y_pt = ax.transData.transform((x, y))[1] * to_pt
        placed.append([y_pt, y_pt, label, x, y])
    placed.sort()
    for prev, cur in zip(placed, placed[1:]):
        if cur[1] - prev[1] < min_gap_pt:
            cur[1] = prev[1] + min_gap_pt
    return [
        ax.annotate(
            label,
            (x, y),
            xytext=(6, label_pt - y_pt),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=TEXT_SECONDARY,
            # Surface-colored backing so a gridline or reference line never runs through the text.
            bbox=dict(boxstyle="square,pad=0.1", facecolor=SURFACE, edgecolor="none"),
        )
        for y_pt, label_pt, label, x, y in placed
    ]


def plot(logs, labels, out_path, title):
    if len(logs) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} runs per figure")
    runs = [(label, read_log(path)) for label, path in zip(labels, logs)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    end_labels = {}
    for ax, (field, panel_title, scale) in zip(axes.flat, PANELS):
        style_axis(ax, panel_title)
        for i, (label, data) in enumerate(runs):
            points = [
                (it, v * scale)
                for it, v in zip(data["iteration"], data[field])
                if v is not None
            ]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                color=SERIES_COLORS[i],
                linewidth=2,
                marker=SERIES_MARKERS[i],
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label,
            )
            if len(runs) > 1:
                end_labels.setdefault(ax, []).append((label, xs[-1], ys[-1]))
        ax.set_xlim(left=0.5)
        if field == "decisive_fraction":
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(bottom=0)
        ax.xaxis.get_major_locator().set_params(integer=True)

    if len(runs) > 1:
        handles, names = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            names,
            loc="upper right",
            ncol=len(runs),
            frameon=False,
            fontsize=9,
            labelcolor=TEXT_PRIMARY,
        )
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # Place the labels, let tight_layout make room for them, then place them
    # again so the overlap check uses the final axes size.
    placed = [a for ax, items in end_labels.items() for a in add_end_labels(ax, items)]
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for a in placed:
        a.remove()
    for ax, items in end_labels.items():
        add_end_labels(ax, items)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training learning curves")
    parser.add_argument("--logs", nargs="+", required=True, help="training CSV logs")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="one name per log (default: file names)",
    )
    parser.add_argument("--out", default="results/training_curves.png")
    parser.add_argument("--title", default="ChaturangZero self-play training")
    args = parser.parse_args()

    labels = args.labels or args.logs
    if len(labels) != len(args.logs):
        parser.error("--labels needs one name per --logs entry")
    plot(args.logs, labels, args.out, args.title)


if __name__ == "__main__":
    main()
