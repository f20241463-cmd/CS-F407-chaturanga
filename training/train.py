"""
training/train.py -- the AlphaZero self-play / training loop.

Each iteration:
  1. Self-play: the current network plays --games_per_iter games against
     itself through MCTS (with root Dirichlet noise). Every position
     becomes a training example and goes into the replay buffer.
  2. Training: once the buffer holds at least --min_buffer examples,
     run --train_steps gradient steps on random minibatches from it.
  3. Checkpoint: save checkpoints/iter_NNNN.pt and checkpoints/latest.pt,
     and append one row to the CSV log.

Run from the repo root:
    python -m training.train
    python -m training.train --iterations 20 --resume checkpoints/latest.pt

--iterations is the total to reach, so resuming from iteration 10 with
--iterations 20 runs 10 more.
"""

import argparse
import csv
import multiprocessing
import os
import random
import time

import numpy as np
import torch

from network.agent import make_network_fn
from network.checkpoint import load_checkpoint, save_checkpoint
from network.loss import AlphaZeroLoss
from network.model import ChaturangaNet
from training.replay_buffer import ReplayBuffer, compact_example
from training.self_play import play_self_play_game

LOG_FIELDS = [
    "iteration",
    "games",
    "white_wins",
    "black_wins",
    "draws",
    "unfinished",
    "avg_plies",
    "decisive_fraction",
    "buffer_size",
    "train_steps",
    "loss",
    "policy_loss",
    "value_loss",
    "selfplay_seconds",
    "train_seconds",
]


def train_step(model, optimizer, loss_fn, batch, device="cpu"):
    """One gradient step on a numpy batch (states, policies, values). Returns the metrics dict."""
    states, policies, values = batch
    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    target_p = torch.as_tensor(policies, dtype=torch.float32, device=device)
    target_v = torch.as_tensor(values, dtype=torch.float32, device=device)

    model.train()
    optimizer.zero_grad()
    pred_logits, pred_values = model(x)
    loss, metrics = loss_fn(pred_logits, pred_values, target_p, target_v)
    loss.backward()
    optimizer.step()
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ChaturangZero self-play training loop")
    p.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="total iterations to reach (counting any already done in --resume)",
    )
    p.add_argument("--games_per_iter", type=int, default=10)
    p.add_argument(
        "--simulations", type=int, default=30, help="MCTS simulations per move"
    )
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument(
        "--temperature_moves",
        type=int,
        default=1000,
        help="sample moves in proportion to visit counts (temperature 1) for this "
        "many plies, then play the most visited move. The default is 1000, "
        "effectively the whole game, on purpose: with the chess-style 30, "
        "0 of 36 test games were decisive, because greedy play repeats "
        "positions and every game ended in a threefold-repetition draw, so "
        "the value head only learned that everything is a draw. Sampling "
        "the whole game gave about 25 percent decisive games.",
    )
    p.add_argument("--dirichlet_alpha", type=float, default=0.3)
    p.add_argument("--dirichlet_eps", type=float, default=0.25)
    p.add_argument(
        "--max_plies",
        type=int,
        default=300,
        help="stop a self-play game after this many plies and mark it unfinished",
    )
    p.add_argument(
        "--draw_value",
        type=float,
        default=0.0,
        help="value target for every position of a drawn or unfinished game",
    )
    p.add_argument(
        "--material_scale",
        type=float,
        default=None,
        help="off by default. If set (5.0 is a reasonable start), drawn and "
        "unfinished games get value targets tanh(material balance / scale) "
        "from the final position instead of --draw_value. Most self-play games "
        "are draws, and with all-zero draw targets the value head learned to "
        "predict about 0 everywhere, even a rook up or down.",
    )
    p.add_argument(
        "--avoid_repetition",
        action="store_true",
        help="off by default. In self-play, never choose a move that causes a "
        "threefold-repetition draw unless every considered move does (the policy "
        "target leaves those moves out too). Without it most self-play games end "
        "in repetition draws, so the value head sees almost only 0 targets. "
        "Uses only the rules, so it fits the no-domain-knowledge requirement.",
    )
    p.add_argument(
        "--train_steps", type=int, default=100, help="gradient steps per iteration"
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization, applied through Adam's weight_decay",
    )
    p.add_argument("--buffer_capacity", type=int, default=20000)
    p.add_argument(
        "--min_buffer",
        type=int,
        default=256,
        help="skip training until the replay buffer holds at least this many positions",
    )
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log", default="results/training_log.csv")
    p.add_argument(
        "--resume",
        default=None,
        help="checkpoint to continue from (model and optimizer; the replay buffer restarts empty)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="self-play processes. 1 (default) plays the games one after another "
        "in this process, as before. With more, each game runs in a worker with "
        "its own seed derived from (--seed, iteration, game number), so the games "
        "do not depend on how many workers there are. Use about one per CPU core.",
    )
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def append_log_row(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


_SELF_PLAY_KEYS = (
    "simulations",
    "c_puct",
    "temperature_moves",
    "dirichlet_alpha",
    "dirichlet_eps",
    "max_plies",
    "draw_value",
    "material_scale",
    "avoid_repetition",
)

# Per-process state of a self-play worker: the network it is currently
# playing with, and which iteration's weights those are.
_worker = {"iteration": None, "network_fn": None, "model": None}


def _init_worker():
    # One thread per worker: the network is small, so threads in different
    # workers would only compete for the same cores.
    torch.set_num_threads(1)


def _play_one_game(task):
    """Worker side: play one self-play game with this iteration's weights."""
    iteration, game_index, seed, state_dict, settings = task
    if _worker["iteration"] != iteration:
        if _worker["model"] is None:
            _worker["model"] = ChaturangaNet()
        _worker["model"].load_state_dict(state_dict)
        _worker["network_fn"] = make_network_fn(_worker["model"])
        _worker["iteration"] = iteration
    t0 = time.time()
    examples, info = play_self_play_game(
        _worker["network_fn"],
        num_simulations=settings["simulations"],
        c_puct=settings["c_puct"],
        temperature_moves=settings["temperature_moves"],
        dirichlet_alpha=settings["dirichlet_alpha"],
        dirichlet_eps=settings["dirichlet_eps"],
        max_plies=settings["max_plies"],
        draw_value=settings["draw_value"],
        rng=np.random.default_rng(seed),
        material_scale=settings["material_scale"],
        avoid_repetition=settings["avoid_repetition"],
    )
    compact = [compact_example(t, p, v) for t, p, v in examples]
    return game_index, compact, info, time.time() - t0


def game_seed(run_seed, iteration, game_index):
    """Seed for one self-play game, independent of the number of workers."""
    return [int(run_seed), int(iteration), int(game_index)]


def run_self_play_parallel(model, args, pool, iteration, run_seed):
    """
    Play this iteration's games in the worker pool. Results come back in
    game order, each game as a list of compact_example entries.
    """
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    settings = {k: getattr(args, k) for k in _SELF_PLAY_KEYS}
    tasks = [
        (iteration, g, game_seed(run_seed, iteration, g), state_dict, settings)
        for g in range(args.games_per_iter)
    ]
    results = [None] * args.games_per_iter
    for done, (g, examples, info, seconds) in enumerate(
        pool.imap_unordered(_play_one_game, tasks), start=1
    ):
        results[g] = (examples, info)
        print(
            f"  game {g + 1} ({done}/{args.games_per_iter} done): "
            f"{info['outcome']:10s} {info['plies']:4d} plies  ({seconds:.1f}s)",
            flush=True,
        )
    return [r[0] for r in results], [r[1] for r in results]


def run_self_play(model, args, rng):
    model.eval()
    network_fn = make_network_fn(model, device=args.device)
    examples_per_game, infos = [], []
    for g in range(args.games_per_iter):
        t0 = time.time()
        examples, info = play_self_play_game(
            network_fn,
            num_simulations=args.simulations,
            c_puct=args.c_puct,
            temperature_moves=args.temperature_moves,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
            max_plies=args.max_plies,
            draw_value=args.draw_value,
            rng=rng,
            material_scale=args.material_scale,
            avoid_repetition=args.avoid_repetition,
        )
        examples_per_game.append(examples)
        infos.append(info)
        print(
            f"  game {g + 1}/{args.games_per_iter}: {info['outcome']:10s} "
            f"{info['plies']:4d} plies  ({time.time() - t0:.1f}s)",
            flush=True,
        )
    return examples_per_game, infos


def run_training(model, optimizer, loss_fn, buffer, args):
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    for _ in range(args.train_steps):
        batch = buffer.sample(args.batch_size)
        metrics = train_step(model, optimizer, loss_fn, batch, device=args.device)
        for k in totals:
            totals[k] += metrics[k]
    return {k: v / args.train_steps for k, v in totals.items()}


def main(argv=None):
    args = parse_args(argv)

    # Seed before building the network, so a fresh run's initial weights
    # come from --seed too (they used to depend on whatever torch's random
    # state happened to be).
    torch.manual_seed(args.seed)
    model = ChaturangaNet()
    start_iter = 0
    if args.resume:
        model, _, start_iter, _ = load_checkpoint(args.resume, device=args.device)

    # Seed after deciding the start iteration, so a resumed run does not
    # replay exactly the same self-play games as the original run did.
    run_seed = args.seed + start_iter
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)

    model.to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.resume:
        load_checkpoint(
            args.resume, model=model, optimizer=optimizer, device=args.device
        )
        # The saved optimizer state carries its own lr/weight_decay; the CLI values win.
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["weight_decay"] = args.weight_decay
        print(
            f"Resumed from {args.resume} at iteration {start_iter} "
            "(model and optimizer restored; the replay buffer starts empty)."
        )

    loss_fn = AlphaZeroLoss()
    buffer = ReplayBuffer(capacity=args.buffer_capacity, seed=run_seed)

    if start_iter >= args.iterations:
        print(
            f"Already at iteration {start_iter}, nothing to do (--iterations {args.iterations})."
        )
        return

    print(
        f"Training iterations {start_iter + 1}..{args.iterations}: "
        f"{args.games_per_iter} games/iter, {args.simulations} sims/move, "
        f"{args.train_steps} steps/iter of batch {args.batch_size}, device {args.device}"
    )

    pool = None
    if args.workers > 1:
        # "spawn" starts clean processes (no copied torch state), the macOS default.
        pool = multiprocessing.get_context("spawn").Pool(
            args.workers, initializer=_init_worker
        )
        print(f"Self-play in {args.workers} worker processes.")
    try:
        _train_loop(model, optimizer, loss_fn, buffer, args, rng, run_seed, start_iter, pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _train_loop(model, optimizer, loss_fn, buffer, args, rng, run_seed, start_iter, pool):
    run_start = time.time()
    for done, iteration in enumerate(
        range(start_iter + 1, args.iterations + 1), start=1
    ):
        print(f"\nIteration {iteration}/{args.iterations}: self-play")
        t0 = time.time()
        if pool is None:
            examples_per_game, infos = run_self_play(model, args, rng)
            for examples in examples_per_game:
                buffer.add_game(examples)
        else:
            compact_games, infos = run_self_play_parallel(
                model, args, pool, iteration, run_seed
            )
            for entries in compact_games:
                buffer.add_compact_game(entries)
            del compact_games
        selfplay_seconds = time.time() - t0

        t0 = time.time()
        steps = 0
        losses = {"loss": "", "policy_loss": "", "value_loss": ""}
        if len(buffer) >= args.min_buffer and args.train_steps > 0:
            losses = {
                k: round(v, 6)
                for k, v in run_training(
                    model, optimizer, loss_fn, buffer, args
                ).items()
            }
            steps = args.train_steps
        train_seconds = time.time() - t0

        outcomes = [info["outcome"] for info in infos]
        games = len(infos)
        white_wins, black_wins = outcomes.count("white"), outcomes.count("black")
        row = {
            "iteration": iteration,
            "games": games,
            "white_wins": white_wins,
            "black_wins": black_wins,
            "draws": outcomes.count("draw"),
            "unfinished": outcomes.count("unfinished"),
            "avg_plies": (
                round(sum(info["plies"] for info in infos) / games, 2) if games else 0.0
            ),
            "decisive_fraction": (
                round((white_wins + black_wins) / games, 4) if games else 0.0
            ),
            "buffer_size": len(buffer),
            "train_steps": steps,
            **losses,
            "selfplay_seconds": round(selfplay_seconds, 2),
            "train_seconds": round(train_seconds, 2),
        }
        append_log_row(args.log, row)

        # Plain Python types only: torch.load defaults to weights_only=True.
        metadata = {
            "iteration": iteration,
            "args": dict(vars(args)),
            "log_row": dict(row),
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        iter_path = os.path.join(args.checkpoint_dir, f"iter_{iteration:04d}.pt")
        save_checkpoint(iter_path, model, optimizer, step=iteration, metadata=metadata)
        save_checkpoint(
            os.path.join(args.checkpoint_dir, "latest.pt"),
            model,
            optimizer,
            step=iteration,
            metadata=metadata,
        )

        elapsed = time.time() - run_start
        remaining = (args.iterations - iteration) * elapsed / done
        if steps:
            loss_text = (
                f"loss {losses['loss']:.4f} (policy {losses['policy_loss']:.4f}, "
                f"value {losses['value_loss']:.4f})"
            )
        elif args.train_steps == 0:
            loss_text = "training disabled (--train_steps 0)"
        else:
            loss_text = (
                f"no training yet (buffer {len(buffer)} < min_buffer {args.min_buffer})"
            )
        print(
            f"Iteration {iteration} done: W {white_wins} / B {black_wins} / "
            f"D {row['draws']} / unfinished {row['unfinished']}, "
            f"avg {row['avg_plies']} plies, decisive {row['decisive_fraction']:.0%}, "
            f"buffer {len(buffer)}, {loss_text}. "
            f"Self-play {selfplay_seconds:.0f}s, train {train_seconds:.0f}s. "
            f"Saved {iter_path}. Estimated time left: {remaining / 60:.1f} min"
        )


if __name__ == "__main__":
    main()
