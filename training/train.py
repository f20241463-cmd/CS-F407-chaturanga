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

from network.agent import make_batch_network_fn, make_network_fn
from network.checkpoint import load_checkpoint, save_checkpoint
from network.loss import AlphaZeroLoss
from network.model import ChaturangaNet
from training.replay_buffer import ReplayBuffer, compact_example
from training.batched_self_play import play_games_batched, play_keyed_games_batched
from training.self_play import play_self_play_game, reanalyse_game_steps, self_play_game_steps

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
    "resigned",
    "resign_checks",
    "resign_wrong",
]


def train_step(model, optimizer, loss_fn, batch, device="cpu", aux=None):
    """
    One gradient step on a numpy batch (states, policies, values). Returns
    the metrics dict. aux: optional (MovesLeftHead, its optimizer, weight);
    the batch then also holds each position's plies left, and the head's
    loss (weighted) is added, training the shared backbone too.
    """
    states, policies, values = batch[:3]
    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    target_p = torch.as_tensor(policies, dtype=torch.float32, device=device)
    target_v = torch.as_tensor(values, dtype=torch.float32, device=device)

    model.train()
    optimizer.zero_grad()
    if aux is None:
        pred_logits, pred_values = model(x)
        loss, metrics = loss_fn(pred_logits, pred_values, target_p, target_v)
    else:
        head, aux_optimizer, weight = aux
        head.train()
        aux_optimizer.zero_grad()
        features = model.features(x)
        pred_logits, pred_values = model.heads(features)
        loss, metrics = loss_fn(pred_logits, pred_values, target_p, target_v)
        target_m = torch.as_tensor(batch[3], dtype=torch.float32, device=device).clamp(max=400) / 100
        m_loss = torch.nn.functional.mse_loss(head(features), target_m)
        loss = loss + weight * m_loss
        metrics["moves_left_loss"] = float(m_loss.item())
    loss.backward()
    optimizer.step()
    if aux is not None:
        aux_optimizer.step()
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
        "--q_mix",
        type=float,
        default=0.0,
        help="0 (default): value targets are the game result only. With w > 0 "
        "each position's target is (1 - w) x result + w x the search's own value "
        "of that position, so drawn games still carry a signal. Uses only the "
        "agent's own search, no game knowledge.",
    )
    p.add_argument(
        "--late_start_fraction",
        type=float,
        default=0.0,
        help="0 (default): every self-play game starts from the opening. With f > 0, "
        "that share of games starts instead from a late position of one of the "
        "previous iteration's games (between --late_start_from and --late_start_to "
        "of its length), so the agent practises more endgames. Uses only its own "
        "games. Needs --selfplay_device.",
    )
    p.add_argument("--late_start_from", type=float, default=0.5)
    p.add_argument(
        "--reanalyse_games",
        type=int,
        default=0,
        help="0 (default): off. With R > 0, each iteration also replays R games "
        "from the last --reanalyse_window iterations and searches every position "
        "again with the current network (MuZero's Reanalyse); those positions are "
        "added with fresh policy targets and the original results. Needs "
        "--selfplay_device.",
    )
    p.add_argument("--reanalyse_window", type=int, default=5)
    p.add_argument("--late_start_to", type=float, default=0.9)
    p.add_argument(
        "--fast_simulations",
        type=int,
        default=None,
        help="off by default. If set (for example 30), each move gets the full "
        "--simulations search and is recorded with probability --full_fraction, "
        "and otherwise a fast search with this many simulations that is not "
        "recorded (KataGo's playout cap randomization).",
    )
    p.add_argument("--full_fraction", type=float, default=1.0)
    p.add_argument(
        "--gumbel",
        action="store_true",
        help="off by default. Self-play uses the Gumbel root search "
        "(mcts/gumbel.py) instead of Dirichlet noise and visit counts.",
    )
    p.add_argument("--gumbel_actions", type=int, default=16,
                   help="moves the Gumbel search considers at the root")  # fmt: skip
    p.add_argument("--gumbel_scale", type=float, default=0.1,
                   help="weight of searched values against the prior in the Gumbel "
                   "search (mctx value_scale)")  # fmt: skip
    p.add_argument(
        "--opponent_checkpoints",
        default=None,
        help="off by default. Comma-separated earlier checkpoints. With "
        "--opponent_fraction f, that share of games is played against one of "
        "them (a different one each iteration, alternating colors), and only "
        "the current network's moves are recorded. Needs --selfplay_device.",
    )
    p.add_argument("--opponent_fraction", type=float, default=0.0)
    p.add_argument(
        "--resign_threshold",
        type=float,
        default=None,
        help="off by default. If set (for example 0.9), a side whose search "
        "rates its position below -threshold on --resign_after of its moves in a "
        "row resigns, and the game counts as its loss, so hopeless games end with "
        "a real result instead of a 100-ply-rule draw. Uses only the agent's own "
        "evaluation, as AlphaZero does.",
    )
    p.add_argument("--resign_after", type=int, default=2,
                   help="moves in a row below the threshold before resigning")  # fmt: skip
    p.add_argument(
        "--resign_check_fraction",
        type=float,
        default=0.1,
        help="with --resign_threshold, this share of games is played out without "
        "resigning, to count how often a resignation would have been wrong",
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
        "--moves_left_weight",
        type=float,
        default=0.0,
        help="0 (default): off. With w > 0, a small extra head (used only in "
        "training) learns to predict how many plies each game still lasts, and "
        "w x its loss is added, so the shared backbone learns it too (KataGo's "
        "moves-left head). Saved networks are unchanged.",
    )
    p.add_argument(
        "--decisive_weight",
        type=float,
        default=1.0,
        help="1 (default): sample every stored position equally. With w > 1, "
        "positions from won or lost games are sampled w times as often.",
    )
    p.add_argument(
        "--mirror",
        action="store_true",
        help="off by default. Flip half of the sampled training positions left "
        "to right (files a <-> h), a symmetry of the rules.",
    )
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
        "--res_blocks",
        type=int,
        default=4,
        help="residual blocks of a new network (4-6). Ignored with --resume, "
        "which keeps the checkpoint's size.",
    )
    p.add_argument(
        "--filters",
        type=int,
        default=64,
        help="filters per layer of a new network (64-128). Ignored with --resume.",
    )
    p.add_argument(
        "--bootstrap_model",
        default=None,
        help="checkpoint that plays the self-play games of the first "
        "--bootstrap_iters iterations instead of the network being trained. "
        "Lets a new (for example bigger) network start from games of an "
        "earlier self-play agent instead of random play.",
    )
    p.add_argument(
        "--bootstrap_iters",
        type=int,
        default=0,
        help="iterations 1..N use --bootstrap_model for self-play (default 0)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="self-play processes. 1 (default) plays the games one after another "
        "in this process, as before. With more, each game runs in a worker with "
        "its own seed derived from (--seed, iteration, game number), so the games "
        "do not depend on how many workers there are. Use about one per CPU core.",
    )
    p.add_argument(
        "--selfplay_device",
        default=None,
        help="off by default (self-play runs one game at a time on the CPU). If "
        "set (mps or cuda), each worker plays its share of the iteration's games "
        "side by side and evaluates their positions in one batch on this device. "
        "Games use the same per-game seeds as with --workers; the moves only "
        "differ where GPU rounding changes a close choice.",
    )
    p.add_argument(
        "--cpu_below",
        type=int,
        default=4,
        help="with --selfplay_device, batches of fewer positions than this run on "
        "the CPU (near the end of an iteration, when few games are left)",
    )
    p.add_argument(
        "--batch_groups",
        type=int,
        default=1,
        help="with --selfplay_device, split each worker's games into this many "
        "groups that take turns, so the CPU searches one group while the GPU "
        "evaluates another. 1 (default) was as fast or faster in our tests, "
        "since smaller batches cost the GPU more",
    )
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if not 4 <= args.res_blocks <= 6:
        p.error("--res_blocks must be between 4 and 6")
    if not 64 <= args.filters <= 128:
        p.error("--filters must be between 64 and 128")
    if args.bootstrap_iters < 0:
        p.error("--bootstrap_iters must be at least 0")
    if not 0.0 <= args.q_mix <= 1.0:
        p.error("--q_mix must be between 0 and 1")
    if args.bootstrap_iters and not args.bootstrap_model:
        p.error("--bootstrap_iters needs --bootstrap_model")
    if args.opponent_fraction and not (args.opponent_checkpoints and args.selfplay_device):
        p.error("--opponent_fraction needs --opponent_checkpoints and --selfplay_device")
    if args.reanalyse_games and not args.selfplay_device:
        p.error("--reanalyse_games needs --selfplay_device")
    if args.late_start_fraction and not args.selfplay_device:
        p.error("--late_start_fraction needs --selfplay_device")
    if not 0.0 < args.full_fraction <= 1.0:
        p.error("--full_fraction must be in (0, 1]")
    return args


def resign_counts(infos):
    """
    How many games ended by resignation, how many check games (played out
    although a side would have resigned) there were, and in how many of
    those the side that would have resigned did not lose. Empty without
    resignation.
    """
    if not infos or "resigned" not in infos[0]:
        return {"resigned": "", "resign_checks": "", "resign_wrong": ""}
    checks = [i for i in infos if not i["resigned"] and i["would_resign"]]
    wrong = sum(1 for i in checks if i["outcome"] != _opponent_name(i["would_resign"]))
    return {
        "resigned": sum(1 for i in infos if i["resigned"]),
        "resign_checks": len(checks),
        "resign_wrong": wrong,
    }


def _opponent_name(color_name):
    return "black" if color_name == "white" else "white"


def append_log_row(path, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    fields = LOG_FIELDS
    if not write_header:
        # Keep an existing log's columns, so a resumed run started by an
        # older version of this script still lines up.
        with open(path, newline="") as f:
            fields = next(csv.reader(f))
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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
    "resign_threshold",
    "resign_after",
    "resign_check_fraction",
    "q_mix",
    "fast_simulations",
    "full_fraction",
    "gumbel",
    "gumbel_actions",
    "gumbel_scale",
)

# Per-process state of a self-play worker: the network it is currently
# playing with, and which iteration's weights those are.
_worker = {"iteration": None, "network_fn": None, "model": None, "config": None}


def model_config(model):
    """The constructor arguments that rebuild a network of the same size."""
    return {
        "in_channels": model.in_channels,
        "num_res_blocks": model.num_res_blocks,
        "num_filters": model.num_filters,
    }


def _init_worker():
    # One thread per worker: the network is small, so threads in different
    # workers would only compete for the same cores.
    torch.set_num_threads(1)


def _play_one_game(task):
    """Worker side: play one self-play game with this iteration's weights."""
    iteration, game_index, seed, state_dict, config, settings = task
    if _worker["iteration"] != iteration:
        if _worker["config"] != config:
            _worker["model"] = ChaturangaNet(**config)
            _worker["config"] = config
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
        q_mix=settings["q_mix"],
        fast_simulations=settings["fast_simulations"],
        full_fraction=settings["full_fraction"],
        gumbel=settings["gumbel"],
        gumbel_actions=settings["gumbel_actions"],
        gumbel_scale=settings["gumbel_scale"],
        **_resign_settings(settings, _check_rng(seed)),
    )
    compact = [compact_example(t, p, v) for t, p, v in examples]
    return game_index, compact, info, time.time() - t0


def _resign_settings(settings, check_rng):
    """Resignation arguments for one game; check_rng decides if it is a check game."""
    if settings["resign_threshold"] is None:
        return {}
    return {
        "resign_threshold": settings["resign_threshold"],
        "resign_after": settings["resign_after"],
        "resign_disabled": bool(check_rng.random() < settings["resign_check_fraction"]),
    }


def _check_rng(seed):
    """A generator for the check-game draw, separate from the game's own."""
    return np.random.default_rng([*seed, 1])


def _game_steps(settings, seed, start_moves=None, record_moves=False, opponent_color=None):
    return self_play_game_steps(
        opponent_color=opponent_color,
        start_moves=start_moves,
        record_moves=record_moves,
        fast_simulations=settings["fast_simulations"],
        full_fraction=settings["full_fraction"],
        gumbel=settings["gumbel"],
        gumbel_actions=settings["gumbel_actions"],
        gumbel_scale=settings["gumbel_scale"],
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
        q_mix=settings["q_mix"],
        **_resign_settings(settings, _check_rng(seed)),
    )


def _play_game_batch(task):
    """
    Worker side of batched self-play: play several games side by side,
    evaluating their positions together on the given device.
    """
    (game_indices, seeds, state_dict, config, settings, device, cpu_below, groups, total,
     starts, record_moves, opponent_path, opponent_colors, reanalyse) = task
    model = ChaturangaNet(**config)
    model.load_state_dict(state_dict)
    batch_fn = make_batch_network_fn(model, device=device, cpu_below=cpu_below)
    t0 = time.time()

    def done(i, result):
        # Keep a compact copy of each game as soon as it ends: full examples
        # are about 20 KB each.
        examples, info = result
        print(
            f"  game {game_indices[i] + 1}/{total}: {info['outcome']:10s} "
            f"{info['plies']:4d} plies  ({time.time() - t0:.1f}s)",
            flush=True,
        )
        return [compact_example(t, p, v) for t, p, v in examples], info

    games = [_game_steps(settings, seed, start, record_moves, opp)
             for seed, start, opp in zip(seeds, starts, opponent_colors)]  # fmt: skip
    for _, moves in reanalyse:
        games.append(reanalyse_game_steps(moves, settings["simulations"], settings["c_puct"],
                                          settings["draw_value"], settings["avoid_repetition"]))  # fmt: skip
    game_indices = list(game_indices) + [g for g, _ in reanalyse]
    if opponent_path is None:
        results = play_games_batched(games, batch_fn, groups=groups, on_done=done)
    else:
        from training.self_play import _tagged

        old_model, _, _, _ = load_checkpoint(opponent_path)
        networks = {"cur": batch_fn,
                    "old": make_batch_network_fn(old_model, device=device, cpu_below=cpu_below)}  # fmt: skip
        games = [g if opp is not None else _tagged("cur", g) for g, opp in zip(games, opponent_colors)]
        results = play_keyed_games_batched(games, networks, on_done=done)
    return [(g, examples, info) for g, (examples, info) in zip(game_indices, results)]


def choose_start_moves(recent_games, n, run_seed, iteration, fraction, lo, hi):
    """
    For each of n games, None (start from the opening) or the first moves of
    one of recent_games, cut at a random point between lo and hi of its
    length. Seeded by (run_seed, iteration), so runs are repeatable.
    """
    starts = [None] * n
    if not fraction or not recent_games:
        return starts
    rng = np.random.default_rng([int(run_seed), int(iteration), 7])
    for g in range(n):
        if rng.random() < fraction:
            moves = recent_games[int(rng.integers(len(recent_games)))]
            cut = int(rng.integers(int(lo * len(moves)), max(int(hi * len(moves)), int(lo * len(moves)) + 1)))
            starts[g] = moves[:cut]
    return starts


def choose_opponents(args, n, run_seed, iteration):
    """
    (checkpoint, colors): this iteration's earlier checkpoint (taking turns
    through --opponent_checkpoints) and, for each game, the color it plays
    or None for ordinary self-play. Seeded by (run_seed, iteration).
    """
    if not args.opponent_fraction:
        return None, [None] * n
    from chaturanga.types import Color

    paths = [p for p in args.opponent_checkpoints.split(",") if p]
    rng = np.random.default_rng([int(run_seed), int(iteration), 11])
    colors = [
        (Color.BLACK if g % 2 == 0 else Color.WHITE) if rng.random() < args.opponent_fraction else None
        for g in range(n)
    ]
    return paths[iteration % len(paths)], colors


def run_self_play_batched(model, args, pool, iteration, run_seed, recent_games=None,
                          reanalyse_pool=None):  # fmt: skip
    """
    Batched self-play: the iteration's games are dealt out to the workers
    (or all played here without a pool), and each worker plays its games
    side by side with one network call per step on --selfplay_device.
    Uses the same per-game seeds as run_self_play_parallel.
    """
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    settings = {k: getattr(args, k) for k in _SELF_PLAY_KEYS}
    config = model_config(model)
    n = args.games_per_iter
    starts = choose_start_moves(recent_games, n, run_seed, iteration, args.late_start_fraction,
                                args.late_start_from, args.late_start_to)  # fmt: skip
    record_moves = args.late_start_fraction > 0 or args.reanalyse_games > 0
    reanalyse = []
    if args.reanalyse_games and reanalyse_pool:
        rng = np.random.default_rng([int(run_seed), int(iteration), 13])
        picks = rng.choice(len(reanalyse_pool), size=min(args.reanalyse_games, len(reanalyse_pool)),
                           replace=False)  # fmt: skip
        reanalyse = [(n + k, reanalyse_pool[int(i)]) for k, i in enumerate(picks)]
    opponent_path, opponent_colors = choose_opponents(args, n, run_seed, iteration)
    chunks = [list(range(w, n, max(1, args.workers))) for w in range(max(1, args.workers))]
    tasks = [
        (
            chunk,
            [game_seed(run_seed, iteration, g) for g in chunk],
            state_dict,
            config,
            settings,
            args.selfplay_device,
            args.cpu_below,
            args.batch_groups,
            n,
            [starts[g] for g in chunk],
            record_moves,
            opponent_path,
            [opponent_colors[g] for g in chunk],
            reanalyse[w::len(chunks)],
        )
        for w, chunk in enumerate(chunks)
        if chunk
    ]
    results = [None] * (n + len(reanalyse))
    outputs = pool.imap_unordered(_play_game_batch, tasks) if pool else map(_play_game_batch, tasks)
    for chunk_result in outputs:
        for g, examples, info in chunk_result:
            results[g] = (examples, info)
    return [r[0] for r in results], [r[1] for r in results]


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
    config = model_config(model)
    tasks = [
        (iteration, g, game_seed(run_seed, iteration, g), state_dict, config, settings)
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
    settings = {k: getattr(args, k) for k in _SELF_PLAY_KEYS}
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
            q_mix=args.q_mix,
            fast_simulations=args.fast_simulations,
            full_fraction=args.full_fraction,
            gumbel=args.gumbel,
            gumbel_actions=args.gumbel_actions,
            gumbel_scale=args.gumbel_scale,
            **_resign_settings(settings, rng),
        )
        examples_per_game.append(examples)
        infos.append(info)
        print(
            f"  game {g + 1}/{args.games_per_iter}: {info['outcome']:10s} "
            f"{info['plies']:4d} plies  ({time.time() - t0:.1f}s)",
            flush=True,
        )
    return examples_per_game, infos


def run_training(model, optimizer, loss_fn, buffer, args, aux=None):
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    for _ in range(args.train_steps):
        batch = buffer.sample(args.batch_size)
        metrics = train_step(model, optimizer, loss_fn, batch, device=args.device, aux=aux)
        for k in totals:
            totals[k] += metrics[k]
    return {k: v / args.train_steps for k, v in totals.items()}


def main(argv=None):
    args = parse_args(argv)

    # Seed before building the network, so a fresh run's initial weights
    # come from --seed too (they used to depend on whatever torch's random
    # state happened to be).
    torch.manual_seed(args.seed)
    model = ChaturangaNet(num_res_blocks=args.res_blocks, num_filters=args.filters)
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

    bootstrap = None
    if args.bootstrap_iters > start_iter:
        bootstrap, _, _, _ = load_checkpoint(args.bootstrap_model, device=args.device)
        bootstrap.eval()
        print(
            f"Self-play of iterations {start_iter + 1}..{args.bootstrap_iters} "
            f"is played by {args.bootstrap_model}."
        )
    cfg = model_config(model)
    print(
        f"Network: {cfg['num_res_blocks']} residual blocks, {cfg['num_filters']} "
        f"filters, {sum(t.numel() for t in model.parameters()):,} parameters."
    )

    loss_fn = AlphaZeroLoss()
    buffer = ReplayBuffer(capacity=args.buffer_capacity, seed=run_seed,
                          decisive_weight=args.decisive_weight, mirror=args.mirror,
                          moves_left=args.moves_left_weight > 0)  # fmt: skip
    aux = None
    if args.moves_left_weight > 0:
        from network.model import MovesLeftHead

        head = MovesLeftHead(model.num_filters).to(args.device)
        aux = (head, torch.optim.Adam(head.parameters(), lr=args.lr), args.moves_left_weight)

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
        _train_loop(
            model, optimizer, loss_fn, buffer, args, rng, run_seed, start_iter, pool,
            bootstrap, aux,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _train_loop(model, optimizer, loss_fn, buffer, args, rng, run_seed, start_iter, pool,
                bootstrap=None, aux=None):  # fmt: skip
    run_start = time.time()
    recent_games = []  # move lists of the last iteration's games, for --late_start_fraction
    reanalyse_window = []  # move lists of the last few iterations' games, for --reanalyse_games
    for done, iteration in enumerate(
        range(start_iter + 1, args.iterations + 1), start=1
    ):
        player = model
        if bootstrap is not None and iteration <= args.bootstrap_iters:
            player = bootstrap
        who = "" if player is model else " (bootstrap model)"
        print(f"\nIteration {iteration}/{args.iterations}: self-play{who}")
        t0 = time.time()
        if args.selfplay_device:
            compact_games, infos = run_self_play_batched(
                player, args, pool, iteration, run_seed, recent_games,
                [m for games in reanalyse_window for m in games],
            )
            reanalysed = sum(1 for i in infos if i.get("reanalysed"))
            infos = [i for i in infos if not i.get("reanalysed")]
            recent_games = [i.pop("moves") for i in infos if len(i.get("moves", ())) >= 20]
            if args.reanalyse_games:
                reanalyse_window = (reanalyse_window + [recent_games])[-args.reanalyse_window:]
                print(f"  reanalysed {reanalysed} earlier games", flush=True)
            for entries in compact_games:
                buffer.add_compact_game(entries)
            del compact_games
        elif pool is None:
            examples_per_game, infos = run_self_play(player, args, rng)
            for examples in examples_per_game:
                buffer.add_game(examples)
        else:
            compact_games, infos = run_self_play_parallel(
                player, args, pool, iteration, run_seed
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
                    model, optimizer, loss_fn, buffer, args, aux
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
            **resign_counts(infos),
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
            + (
                f"resigned {row['resigned']} (wrong in {row['resign_wrong']} of "
                f"{row['resign_checks']} checks). "
                if row["resigned"] != ""
                else ""
            )
            + f"Self-play {selfplay_seconds:.0f}s, train {train_seconds:.0f}s. "
            f"Saved {iter_path}. Estimated time left: {remaining / 60:.1f} min"
        )


if __name__ == "__main__":
    main()
