"""Fit one Elo-style rating per player from every match already played.

Uses the head-to-head match files under results/ that follow the standard
protocol (50 simulations, temperature 0.3, repetition avoidance, 800-ply cap).
A draw counts as half a win. Iteration 560, the first final model, is fixed at
0, so every rating says how far above or below 560 a model is.

Matches against random, untrained and greedy capture are left out by default:
most of those games are draws because the weak side cannot be mated in time,
which squashes every rating towards the weak player. --bench adds them.

    python -m evaluation.elo_table            # prints the table
    python -m evaluation.elo_table --all      # every player, not only the named ones
"""
import argparse
import csv
import glob
import math
import random
import re
from collections import defaultdict

STANDARD = "50 sims, temperature 0.3, repetition avoidance on, max 800 plies"
ANCHOR = "checkpoints/small_more2/iter_0560.pt"
BENCH_FILES = ["results/benchmark_long_*.csv", "results/final_model/benchmark_*.csv"]

# Short names for the players the report talks about.
NAMES = {
    "random": "Random moves",
    "untrained": "Untrained network + MCTS",
    "greedy": "Greedy capture",
    "checkpoints/long/iter_0420.pt": "First long run, iteration 420",
    "checkpoints/small_more2/iter_0560.pt": "Iteration 560 (first final model)",
    "checkpoints/ideas/gumbel/iter_0620.pt": "Gumbel self-play, iteration 620",
    "checkpoints/ideas/gumbel_more3/iter_0800.pt": "Gumbel, iteration 800",
    "checkpoints/ideas/gumbel_chain_1100/iter_1100.pt": "Gumbel, iteration 1100",
    "checkpoints/ideas/lowlr_chain_1220/iter_1220.pt": "Iteration 1220 (second final model)",
    "checkpoints/ideas/lowlr_chain_1280/iter_1280.pt": "Iteration 1280, 16 root moves",
    "checkpoints/ideas/g_lowlr/iter_1160.pt": "Gumbel, learning rate 3e-5, iteration 1160",
    "checkpoints/ideas/g_norep/iter_1280.pt": "Tweak: no repetition avoidance (1280)",
    "checkpoints/ideas/g_wd1e5/iter_1280.pt": "Tweak: weight decay 1e-5 (1280)",
    "checkpoints/ideas/g_wd1e3/iter_1280.pt": "Tweak: weight decay 1e-3 (1280)",
    "checkpoints/ideas/g_sims32/iter_1280.pt": "Tweak: 150 games x 32 simulations (1280)",
    "checkpoints/ideas/g_buffer100k/iter_1280.pt": "Tweak: replay buffer 100k (1280)",
    "checkpoints/ideas/g_steps1000/iter_1280.pt": "Tweak: 1000 training steps (1280)",
    "checkpoints/ideas/g_lr3e4/iter_1280.pt": "Tweak: learning rate 3e-4 (1280)",
    "checkpoints/ideas/g_scale03/iter_1280.pt": "Tweak: Gumbel value scale 0.3 (1280)",
    "checkpoints/ideas/play_ideas/soup.pt": "Averaged weights of the 1220 chain (soup)",
    "checkpoints/ideas/g_actions32b/iter_1280.pt": "Iteration 1280, 32 root moves (final model)",
}


def load_results(bench=False):
    """Return a list of (a, b, a_points, games), unfinished games left out."""
    out = []
    for path in glob.glob("results/**/*.txt", recursive=True):
        text = open(path).read()
        a = re.search(r"^A = (.*)$", text, re.M)
        b = re.search(r"^B = (.*)$", text, re.M)
        s = re.search(r"A: (\d+) wins, (\d+) losses, (\d+) draws", text)
        if not (a and b and s):
            continue
        header = text.split("\n")[2]
        if re.sub(r"^\d+ games, ", "", re.sub(r", \d+ workers$", "", header)) != STANDARD:
            continue
        w, l, d = map(int, s.groups())
        out.append((a.group(1).strip(), b.group(1).strip(), w + d / 2, w + l + d))
    for pattern in BENCH_FILES if bench else []:
        for path in glob.glob(pattern):
            for row in csv.DictReader(open(path)):
                w, l, d = int(row["wins"]), int(row["losses"]), int(row["draws"])
                if w + l + d:
                    out.append((row["checkpoint"], row["opponent"], w + d / 2, w + l + d))
    return out


def fit(results, iters=6000, lr=20.0, start=None):
    """Maximum-likelihood Elo (logistic, 400-point scale), iteration 560 fixed at 0."""
    players = sorted({p for a, b, _, _ in results for p in (a, b)})
    r = {p: (start or {}).get(p, 0.0) for p in players}
    for _ in range(iters):
        grad = defaultdict(float)
        for a, b, pts, n in results:
            e = 1 / (1 + 10 ** ((r[b] - r[a]) / 400))
            grad[a] += pts - n * e
            grad[b] -= pts - n * e
        games = defaultdict(int)
        for a, b, _, n in results:
            games[a] += n
            games[b] += n
        for p in players:
            r[p] += lr * grad[p] / max(games[p], 1) * 4
        shift = r.get(ANCHOR, 0.0)
        for p in players:
            r[p] -= shift
    return r


def bootstrap(results, start, runs=100, seed=0):
    """Resample every match's games (draws as halves) and refit; returns 90% ranges."""
    rng = random.Random(seed)
    fits = []
    for _ in range(runs):
        sample = []
        for a, b, pts, n in results:
            p = pts / n
            new = sum(1.0 if rng.random() < p else 0.0 for _ in range(n))
            sample.append((a, b, new, n))
        fits.append(fit(sample, iters=1500, start=start))
    ranges = {}
    for p in fits[0]:
        vals = sorted(f[p] for f in fits)
        ranges[p] = (vals[int(0.05 * runs)], vals[int(0.95 * runs) - 1])
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="list every player")
    ap.add_argument("--bench", action="store_true", help="also use the random/untrained/greedy benchmarks")
    ap.add_argument("--bootstrap", type=int, default=100, help="resamples for the ranges (0 = none)")
    args = ap.parse_args()
    results = load_results(args.bench)
    r = fit(results)
    ranges = bootstrap(results, r, args.bootstrap) if args.bootstrap else {}
    games = defaultdict(int)
    for a, b, _, n in results:
        games[a] += n
        games[b] += n
    print(f"{len(results)} matches, {sum(n for *_, n in results)} games")
    print("| Player | Rating | 90% range | Games |")
    print("|---|---|---|---|")
    for p in sorted(r, key=lambda p: -r[p]):
        if not args.all and p not in NAMES:
            continue
        lo, hi = ranges.get(p, (r[p], r[p]))
        print(f"| {NAMES.get(p, p)} | {r[p]:.0f} | {lo:.0f} to {hi:.0f} | {games[p]} |")


if __name__ == "__main__":
    main()
