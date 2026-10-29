"""
evaluation/resign_check.py -- would resigning hopeless self-play games work?

Plays self-play games with a checkpoint (the training settings, no
resignation) and records the search's evaluation of every position. Then,
for each resign threshold, counts how many games a side would have resigned
(evaluation below -threshold on N of its moves in a row) and in how many of
those it would have been wrong (the side did not go on to lose).

    python -m evaluation.resign_check --checkpoint models/chaturangzero_final.pt \
        --games 160 --workers 4 --device mps
"""

import argparse
import multiprocessing

import numpy as np


def _play(task):
    import torch

    from network.agent import make_batch_network_fn
    from network.checkpoint import load_checkpoint
    from training.batched_self_play import play_games_batched
    from training.self_play import self_play_game_steps

    torch.set_num_threads(1)
    checkpoint, indices, seed, device, simulations, max_plies = task
    model, _, _, _ = load_checkpoint(checkpoint, device="cpu")
    network = make_batch_network_fn(model, device=device, cpu_below=4)
    traces = {i: [] for i in indices}
    games = [
        self_play_game_steps(num_simulations=simulations, max_plies=max_plies,
                             temperature_moves=1000, avoid_repetition=True,
                             rng=np.random.default_rng([seed, i]),
                             value_trace=traces[i])  # fmt: skip
        for i in indices
    ]
    results = play_games_batched(games, network)
    return [
        (info["outcome"], [(c.name.lower(), v) for c, v in traces[i]])
        for i, (_, info) in zip(indices, results)
    ]


def would_resign(trace, threshold, after):
    """(side, ply) of the first side whose evaluation stays below -threshold for `after` moves."""
    streak = {}
    for ply, (side, value) in enumerate(trace):
        streak[side] = streak.get(side, 0) + 1 if value < -threshold else 0
        if streak[side] >= after:
            return side, ply
    return None, None


def summarize(games, thresholds, after_values):
    rows = []
    for threshold in thresholds:
        for after in after_values:
            resigned = wrong = 0
            for outcome, trace in games:
                side, _ = would_resign(trace, threshold, after)
                if side is None:
                    continue
                resigned += 1
                winner = "black" if side == "white" else "white"
                wrong += outcome != winner
            rows.append((threshold, after, resigned, wrong))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", default="models/chaturangzero_final.pt")
    p.add_argument("--games", type=int, default=160)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--max_plies", type=int, default=800)
    p.add_argument("--seed", type=int, default=77)
    args = p.parse_args(argv)

    chunks = [list(range(w, args.games, args.workers)) for w in range(args.workers)]
    tasks = [(args.checkpoint, c, args.seed, args.device, args.simulations,
              args.max_plies) for c in chunks if c]  # fmt: skip
    with multiprocessing.get_context("spawn").Pool(len(tasks)) as pool:
        games = [g for part in pool.map(_play, tasks) for g in part]

    outcomes = [o for o, _ in games]
    values = np.array([v for _, t in games for _, v in t])
    print(f"{len(games)} games: {outcomes.count('white')} White wins, "
          f"{outcomes.count('black')} Black wins, {outcomes.count('draw')} draws, "
          f"{outcomes.count('unfinished')} unfinished")  # fmt: skip
    pct = np.percentile(values, [1, 5, 50, 95, 99]).round(3)
    print(f"evaluation percentiles 1/5/50/95/99: {pct.tolist()}")
    print("\nthreshold  moves  would resign  wrong")
    for threshold, after, resigned, wrong in summarize(
        games, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [2, 3]
    ):
        print(f"{threshold:9.1f}  {after:5d}  {resigned:12d}  {wrong:5d}")
    print("\nLosing side's lowest evaluation in each decisive game:")
    for outcome, trace in games:
        if outcome in ("white", "black"):
            loser = "black" if outcome == "white" else "white"
            low = min(v for side, v in trace if side == loser)
            print(f"  {outcome} won in {len(trace)} plies; loser's lowest {low:.2f}")


if __name__ == "__main__":
    main()
