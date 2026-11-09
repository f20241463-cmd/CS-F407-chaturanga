"""
evaluation/parallel_match.py -- play a match across several CPU cores.

The same games as evaluation.players.play_match (colors alternate, A is
White in even-numbered games), but each game runs in a worker process and
gets its own seed, derived from (seed, game number). The results therefore
do not depend on the number of workers, and every checkpoint benchmarked
with the same seed faces the same sequence of game seeds.

Players are described by plain dicts ("player specs") so they can be sent
to worker processes; each worker builds a player once and reuses it:

    {"kind": "random"}
    {"kind": "greedy"}
    {"kind": "mcts", "network": "<checkpoint path>" or "untrained",
     "seed": 0, "simulations": 50, "c_puct": 1.5, "temperature": 0.3,
     "avoid_repetition": True}

Head to head under the benchmark protocol (from the repo root):
    python -m evaluation.parallel_match --a checkpoints/long/latest.pt \
        --b checkpoints/final_B/latest.pt --games 100 --workers 6
"""

import argparse
import multiprocessing
import time

import numpy as np

from evaluation.players import (
    greedy_capture_player,
    make_mcts_player,
    play_game,
    play_game_steps,
    random_player,
)

# Per-process cache of built players, keyed by their spec.
_players = {}


def mcts_spec(network, seed=0, simulations=50, c_puct=1.5, temperature=0.3,
              avoid_repetition=True, reuse_tree=False):  # fmt: skip
    """Spec of the benchmark-protocol MCTS agent for a checkpoint path or 'untrained'."""
    return {
        "kind": "mcts",
        "network": network,
        "seed": seed,
        "simulations": simulations,
        "c_puct": c_puct,
        "temperature": temperature,
        "avoid_repetition": avoid_repetition,
        "reuse_tree": reuse_tree,
    }


def _key(spec):
    return tuple(sorted(spec.items()))


def _models(spec):
    """The networks of an MCTS spec; several checkpoints joined by '+' play as one."""
    return [_model({**spec, "network": n}) for n in spec["network"].split("+")]


def _model(spec):
    """The network of an MCTS spec: a checkpoint, or a seeded fresh network."""
    import torch

    from network.checkpoint import load_checkpoint
    from network.model import ChaturangaNet

    if spec["network"] == "untrained":
        # Same initialization as evaluation.benchmark.untrained_network_fn.
        torch.manual_seed(spec["seed"])
        return ChaturangaNet()
    model, _, _, _ = load_checkpoint(spec["network"])
    return model


def build_player(spec):
    """The player function for a spec (built once per process, then cached)."""
    key = _key(spec)
    if key in _players:
        return _players[key]
    kind = spec["kind"]
    if kind == "random":
        player = random_player
    elif kind == "greedy":
        player = greedy_capture_player
    elif kind == "mcts":
        from network.agent import make_ensemble_network_fn, make_network_fn

        models = _models(spec)
        network_fn = make_network_fn(models[0]) if len(models) == 1 else make_ensemble_network_fn(models)
        player = make_mcts_player(
            network_fn,
            simulations=spec["simulations"],
            c_puct=spec["c_puct"],
            temperature=spec["temperature"],
            avoid_repetition=spec["avoid_repetition"],
            reuse_tree=spec.get("reuse_tree", False),
        )
    else:
        raise ValueError(f"unknown player kind: {kind!r}")
    _players[key] = player
    return player


def _init_worker():
    import torch

    torch.set_num_threads(1)


def _play(task):
    g, spec_a, spec_b, seed, max_plies = task
    a_is_white = g % 2 == 0
    a, b = build_player(spec_a), build_player(spec_b)
    white, black = (a, b) if a_is_white else (b, a)
    t0 = time.time()
    outcome, plies = play_game(
        white, black, np.random.default_rng([int(seed), int(g)]), max_plies
    )
    return g, a_is_white, outcome, plies, time.time() - t0


def _play_batch(task):
    """
    Several games side by side, with each MCTS side's positions evaluated
    in one batch on device. The same games as _play, game by game.
    """
    from network.agent import EnsembleNetwork, make_batch_network_fn
    from training.batched_self_play import play_keyed_games_batched

    indices, spec_a, spec_b, seed, max_plies, device = task
    a, b = build_player(spec_a), build_player(spec_b)
    networks = {}
    for key, spec in (("a", spec_a), ("b", spec_b)):
        if spec["kind"] == "mcts":
            nets = [make_batch_network_fn(m, device=device, cpu_below=4) for m in _models(spec)]
            networks[key] = nets[0] if len(nets) == 1 else EnsembleNetwork(nets)
    games = []
    for g in indices:
        a_is_white = g % 2 == 0
        white, black = (a, b) if a_is_white else (b, a)
        keys = ("a", "b") if a_is_white else ("b", "a")
        rng = np.random.default_rng([int(seed), int(g)])
        games.append(play_game_steps(white, black, rng, max_plies, keys))
    results = play_keyed_games_batched(games, networks)
    return [
        (g, g % 2 == 0, outcome, plies, 0.0)
        for g, (outcome, plies) in zip(indices, results)
    ]


def make_pool(workers):
    """A worker pool for play_match_parallel ('spawn' starts clean processes)."""
    return multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker)


def _batched_results(spec_a, spec_b, games, seed, max_plies, pool, device):
    chunks = getattr(pool, "_processes", 1) if pool is not None else 1
    tasks = [
        (list(range(c, games, chunks)), spec_a, spec_b, seed, max_plies, device)
        for c in range(chunks)
        if c < games
    ]
    outputs = map(_play_batch, tasks) if pool is None else pool.imap_unordered(_play_batch, tasks)
    for chunk in outputs:
        yield from chunk


def play_match_parallel(spec_a, spec_b, games, seed, max_plies=800, pool=None,
                        on_game=None, device=None):  # fmt: skip
    """
    Same result dict as players.play_match: A's wins, losses, draws,
    unfinished, and the game lengths (in game order). Without a pool the
    games run here, one after another, with the same per-game seeds.
    on_game(game_index, a_is_white, outcome, plies) is called as games finish.

    With device (mps or cuda), each worker plays its share of the games
    side by side and evaluates their positions in batches on that device.
    The games are the same, except where GPU rounding changes a close choice.
    """
    if device:
        results = _batched_results(spec_a, spec_b, games, seed, max_plies, pool, device)
    else:
        tasks = [(g, spec_a, spec_b, seed, max_plies) for g in range(games)]
        results = map(_play, tasks) if pool is None else pool.imap_unordered(_play, tasks)
    counts = {"wins": 0, "losses": 0, "draws": 0, "unfinished": 0}
    plies_by_game = [0] * games
    for g, a_is_white, outcome, plies, _ in results:
        plies_by_game[g] = plies
        if outcome == "draw":
            counts["draws"] += 1
        elif outcome == "unfinished":
            counts["unfinished"] += 1
        elif (outcome == "white") == a_is_white:
            counts["wins"] += 1
        else:
            counts["losses"] += 1
        if on_game is not None:
            on_game(g, a_is_white, outcome, plies)
    counts["plies"] = plies_by_game
    return counts


def main(argv=None):
    from compare_checkpoints import score_summary

    p = argparse.ArgumentParser(
        description="Checkpoint A vs checkpoint B under the benchmark protocol, in parallel"
    )
    p.add_argument("--a", required=True, help="checkpoint path, or 'untrained'; join several "
                   "checkpoints with '+' to play them as one averaged network")
    p.add_argument("--b", required=True, help="checkpoint path, or 'untrained'")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--simulations_a", type=int, default=None,
                   help="simulations per move for A only (default --simulations)")
    p.add_argument("--simulations_b", type=int, default=None,
                   help="simulations per move for B only (default --simulations)")
    p.add_argument("--c_puct_a", type=float, default=1.5, help="search exploration constant for A")
    p.add_argument("--c_puct_b", type=float, default=1.5, help="search exploration constant for B")
    p.add_argument("--reuse_a", action="store_true",
                   help="A keeps its search tree between moves")
    p.add_argument("--reuse_b", action="store_true",
                   help="B keeps its search tree between moves")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max_plies", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="mps or cuda: play each worker's games side by side and "
                   "evaluate positions in batches on this device (much faster "
                   "for big networks)")
    args = p.parse_args(argv)

    sims_a = args.simulations_a or args.simulations
    sims_b = args.simulations_b or args.simulations

    def spec(network, simulations, reuse, c_puct):
        return mcts_spec(network, seed=args.seed, simulations=simulations, c_puct=c_puct,
                         temperature=args.temperature, reuse_tree=reuse)  # fmt: skip

    sims = f"{sims_a} sims" if sims_a == sims_b else f"A {sims_a} sims, B {sims_b} sims"
    for side, reuse in (("A", args.reuse_a), ("B", args.reuse_b)):
        if reuse:
            sims += f", {side} keeps its tree"
    if (args.c_puct_a, args.c_puct_b) != (1.5, 1.5):
        sims += f", c_puct A {args.c_puct_a} B {args.c_puct_b}"
    print(
        f"A = {args.a}\nB = {args.b}\n{args.games} games, {sims}, "
        f"temperature {args.temperature}, repetition avoidance on, "
        f"max {args.max_plies} plies, {args.workers} workers"
    )
    start = time.time()

    def show(g, a_is_white, outcome, plies):
        print(f"  game {g + 1:3d}: A {'White' if a_is_white else 'Black'} -> "
              f"{outcome:10s} ({plies} plies)", flush=True)  # fmt: skip

    pool = make_pool(args.workers) if args.workers > 1 else None
    try:
        c = play_match_parallel(spec(args.a, sims_a, args.reuse_a, args.c_puct_a),
                                spec(args.b, sims_b, args.reuse_b, args.c_puct_b), args.games, args.seed,
                                args.max_plies, pool, show, device=args.device)  # fmt: skip
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    finished = c["wins"] + c["losses"] + c["draws"]
    text = "no finished games"
    if finished:
        score, margin = score_summary(c["wins"], c["losses"], c["draws"])
        text = f"A scores {score:.3f} +/- {margin:.3f}"
    print(
        f"\nA: {c['wins']} wins, {c['losses']} losses, {c['draws']} draws, "
        f"{c['unfinished']} unfinished. {text}. "
        f"Average {np.mean(c['plies']):.0f} plies, {time.time() - start:.0f}s."
    )


if __name__ == "__main__":
    main()
