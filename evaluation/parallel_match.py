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
    random_player,
)

# Per-process cache of built players, keyed by their spec.
_players = {}


def mcts_spec(network, seed=0, simulations=50, c_puct=1.5, temperature=0.3,
              avoid_repetition=True):  # fmt: skip
    """Spec of the benchmark-protocol MCTS agent for a checkpoint path or 'untrained'."""
    return {
        "kind": "mcts",
        "network": network,
        "seed": seed,
        "simulations": simulations,
        "c_puct": c_puct,
        "temperature": temperature,
        "avoid_repetition": avoid_repetition,
    }


def _key(spec):
    return tuple(sorted(spec.items()))


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
        import torch

        from network.agent import make_network_fn
        from network.checkpoint import load_checkpoint
        from network.model import ChaturangaNet

        if spec["network"] == "untrained":
            # Same initialization as evaluation.benchmark.untrained_network_fn.
            torch.manual_seed(spec["seed"])
            network_fn = make_network_fn(ChaturangaNet())
        else:
            model, _, _, _ = load_checkpoint(spec["network"])
            network_fn = make_network_fn(model)
        player = make_mcts_player(
            network_fn,
            simulations=spec["simulations"],
            c_puct=spec["c_puct"],
            temperature=spec["temperature"],
            avoid_repetition=spec["avoid_repetition"],
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


def make_pool(workers):
    """A worker pool for play_match_parallel ('spawn' starts clean processes)."""
    return multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker)


def play_match_parallel(spec_a, spec_b, games, seed, max_plies=800, pool=None,
                        on_game=None):  # fmt: skip
    """
    Same result dict as players.play_match: A's wins, losses, draws,
    unfinished, and the game lengths (in game order). Without a pool the
    games run here, one after another, with the same per-game seeds.
    on_game(game_index, a_is_white, outcome, plies) is called as games finish.
    """
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
    p.add_argument("--a", required=True, help="checkpoint path, or 'untrained'")
    p.add_argument("--b", required=True, help="checkpoint path, or 'untrained'")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--simulations_a", type=int, default=None,
                   help="simulations per move for A only (default --simulations)")
    p.add_argument("--simulations_b", type=int, default=None,
                   help="simulations per move for B only (default --simulations)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max_plies", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    sims_a = args.simulations_a or args.simulations
    sims_b = args.simulations_b or args.simulations

    def spec(network, simulations):
        return mcts_spec(network, seed=args.seed, simulations=simulations,
                         temperature=args.temperature)  # fmt: skip

    sims = f"{sims_a} sims" if sims_a == sims_b else f"A {sims_a} sims, B {sims_b} sims"
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
        c = play_match_parallel(spec(args.a, sims_a), spec(args.b, sims_b), args.games, args.seed,
                                args.max_plies, pool, show)  # fmt: skip
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
