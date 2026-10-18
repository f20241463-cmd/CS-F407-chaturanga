"""
evaluation/human_log.py -- record and summarize Agent vs. Human games.

play_vs_ai.py appends one row per game to a CSV (default
results/human_games.csv). summarize() turns that file into the agent's score
against humans, using the same scoring as the benchmark: win = 1,
draw = 0.5, unfinished (quit) games left out. With only a handful of games,
the range is a Wilson 95% interval, which stays meaningful even when one side
wins every game (the benchmark's +/- margin collapses to 0 there).
"""

import csv
import datetime
import math
import os

FIELDS = [
    "date",
    "player",
    "checkpoint",
    "simulations",
    "human_color",
    "result",  # agent_win, human_win, draw, unfinished
    "ending",  # checkmate, stalemate, no-progress rule, ..., resigned, quit
    "plies",
    "moves",
]

RESULTS = ("agent_win", "human_win", "draw", "unfinished")


def append_game(path, **fields):
    """Append one game to the CSV at `path`, writing the header if the file is new."""
    if fields.get("result") not in RESULTS:
        raise ValueError(
            f"result must be one of {RESULTS}, got {fields.get('result')!r}"
        )
    row = {name: fields.get(name, "") for name in FIELDS}
    row["date"] = row["date"] or datetime.datetime.now().isoformat(timespec="seconds")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


def wilson_interval(score, n, z=1.96):
    """95% Wilson interval for a score over n games (draws count as half a win)."""
    if n == 0:
        return None, None
    denom = 1 + z * z / n
    center = (score + z * z / (2 * n)) / denom
    half = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(path, checkpoint=None):
    """Totals and the agent's score from the CSV, optionally for one checkpoint only."""
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = [
                r
                for r in csv.DictReader(f)
                if checkpoint is None or r["checkpoint"] == checkpoint
            ]
    counts = {name: sum(r["result"] == name for r in rows) for name in RESULTS}
    finished = counts["agent_win"] + counts["human_win"] + counts["draw"]
    score = low = high = None
    if finished:
        score = (counts["agent_win"] + 0.5 * counts["draw"]) / finished
        low, high = wilson_interval(score, finished)
    return {"games": len(rows), **counts, "score": score, "low": low, "high": high}


def format_summary(summary):
    s = summary
    text = (
        f"{s['games']} games: agent {s['agent_win']} wins, human {s['human_win']} wins, "
        f"{s['draw']} draws, {s['unfinished']} unfinished"
    )
    if s["score"] is not None:
        text += (
            f". Agent score {s['score']:.3f}, 95% range {s['low']:.3f} to {s['high']:.3f}"
            " (unfinished left out)"
        )
    return text
