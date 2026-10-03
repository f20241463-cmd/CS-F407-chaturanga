#!/usr/bin/env python3
"""
Playable loop for the Chaturanga environment.

Not part of the frozen interfaces - this is a debugging/sanity tool for
Student 1, and later doubles as the manual precursor to Week 4's
"complete one legal AI-vs-AI game" milestone (swap choose_random_move
for MCTS+network once those exist).

Modes:
  human   (default) - you play White by typing moves, a random agent
                       plays Black. Best way to actually *feel* whether
                       the rules are right, not just pass assertions.
  random             - random vs random. With --games N and --quiet,
                        this is a stress test: it will surface bugs
                        like games that never terminate, or crashes on
                        rare board configurations, that hand-written
                        unit tests won't stumble into by chance.

Examples:
    python play.py
    python play.py --mode random
    python play.py --mode random --games 500 --quiet --seed 0
"""

import argparse
import random
import sys

from chaturanga import Color, GameState
from chaturanga.notation import move_to_algebraic


def print_board(state: GameState) -> None:
    print()
    print(f"  {'-' * 23}")
    board = state.board
    for row in range(7, -1, -1):
        cells = []
        for col in range(8):
            sq = row * 8 + col
            p = board.piece_at(sq)
            cells.append(repr(p) if p else ".")
        print(f"{row + 1} | {'  '.join(cells)} |")
    print(f"  {'-' * 23}")
    print("    " + "  ".join("abcdefgh"))
    print(f"{state.side_to_move.name} to move.  In check: {state.is_in_check()}")


def choose_random_move(state: GameState):
    return random.choice(state.legal_moves())


def prompt_human_move(state: GameState):
    legal = state.legal_moves()
    legal_alg = {move_to_algebraic(m): m for m in legal}
    while True:
        raw = input(
            "Your move (e.g. 'e2e4', 'moves' to list legal moves, 'quit' to exit): "
        ).strip().lower()
        if raw in ("quit", "exit"):
            print("Goodbye.")
            sys.exit(0)
        if raw == "moves":
            print(", ".join(sorted(legal_alg)))
            continue
        if raw in legal_alg:
            return legal_alg[raw]
        # also accept raw square indices, e.g. "12 20", to match the
        # exact (from_square, to_square) format from the contract doc
        parts = raw.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            mv = (int(parts[0]), int(parts[1]))
            if mv in legal:
                return mv
        print("Not a legal move. Type 'moves' to see the legal moves for this position.")


def result_label(state: GameState) -> str:
    r = state.result()
    if r is None:
        return "Game not over."
    if r == 0:
        return "Draw."
    # r == -1: state.side_to_move is the player with no legal moves,
    # so the OTHER color is the winner (see contract section 2.10).
    loser = state.side_to_move
    winner = loser.opponent
    reason = "checkmate" if state.is_in_check() else "no legal moves"
    return f"{reason.title()} - {loser.name.title()} has no legal moves. {winner.name.title()} wins!"


def play_human_vs_random(seed=None) -> None:
    if seed is not None:
        random.seed(seed)
    state = GameState()
    half_moves = 0
    while not state.is_game_over():
        print_board(state)
        if state.side_to_move == Color.WHITE:
            move = prompt_human_move(state)
        else:
            move = choose_random_move(state)
            print(f"Black (random) plays: {move_to_algebraic(move)}")
        state = state.apply_move(move)
        half_moves += 1
    print_board(state)
    print(result_label(state))
    print(f"Total half-moves: {half_moves}")


def play_random_vs_random(games: int, quiet: bool, max_moves: int, seed=None) -> None:
    if seed is not None:
        random.seed(seed)

    outcomes = {"white_win": 0, "black_win": 0, "draw": 0, "unfinished": 0}
    lengths = []

    for g in range(games):
        state = GameState()
        n = 0
        while not state.is_game_over() and n < max_moves:
            move = choose_random_move(state)
            if not quiet:
                print(f"[game {g + 1}] move {n + 1}: {state.side_to_move.name} plays "
                      f"{move_to_algebraic(move)}")
            state = state.apply_move(move)
            n += 1
            if not quiet:
                print_board(state)
        lengths.append(n)

        if not state.is_game_over():
            outcomes["unfinished"] += 1
            summary = f"hit --max-moves cap ({max_moves}) without a result"
        else:
            r = state.result()
            if r == 0:
                outcomes["draw"] += 1
                summary = "draw"
            else:
                winner = state.side_to_move.opponent
                outcomes["white_win" if winner == Color.WHITE else "black_win"] += 1
                summary = f"{winner.name.title()} wins"

        if not quiet:
            print(f"[game {g + 1}] finished: {n} half-moves, {summary}")
        elif games <= 20:
            print(f"Game {g + 1}: {n} half-moves, {summary}")

    print()
    print(f"=== {games} game(s) complete ===")
    print(f"White wins:  {outcomes['white_win']}")
    print(f"Black wins:  {outcomes['black_win']}")
    print(f"Draws:       {outcomes['draw']}")
    print(f"Unfinished:  {outcomes['unfinished']}  "
          f"(hit --max-moves; investigate if this is ever > 0)")
    print(f"Average length: {sum(lengths) / len(lengths):.1f} half-moves")
    print(f"Longest game:   {max(lengths)} half-moves")
    print(f"Shortest game:  {min(lengths)} half-moves")


def play_human_vs_network(checkpoint_path=None, seed=None) -> None:
    from network import ChaturangaNet, choose_network_move, load_checkpoint
    import torch
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if checkpoint_path:
        net, _, step, _ = load_checkpoint(checkpoint_path)
        print(f"Loaded network checkpoint from {checkpoint_path} (step {step})")
    else:
        net = ChaturangaNet()
        print("Using initialized ChaturangaNet architecture.")

    state = GameState()
    half_moves = 0
    while not state.is_game_over():
        print_board(state)
        if state.side_to_move == Color.WHITE:
            move = prompt_human_move(state)
        else:
            move, val = choose_network_move(state, net, temperature=0.2)
            print(f"Black (Neural Network) plays: {move_to_algebraic(move)} (NN value estimate: {val:+.2f})")
        state = state.apply_move(move)
        half_moves += 1
    print_board(state)
    print(result_label(state))
    print(f"Total half-moves: {half_moves}")


def main():
    parser = argparse.ArgumentParser(
        description="Play or stress-test the Chaturanga environment."
    )
    parser.add_argument(
        "--mode", choices=["human", "random", "network"], default="human",
        help="'human': you (White) vs a random agent (Black). "
             "'network': you (White) vs the Neural Network agent (Black). "
             "'random': random vs random, for stress-testing.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to neural network checkpoint (.pt) for --mode network.",
    )
    parser.add_argument(
        "--games", type=int, default=1,
        help="Number of games to play (random mode only).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-move/board output in random mode; print only per-game "
             "and summary lines. Recommended for --games > 1.",
    )
    parser.add_argument(
        "--max-moves", type=int, default=1000,
        help="Safety cap on half-moves per game, in case a rules bug ever prevents "
             "termination. Random play rarely stumbles into checkmate by chance and "
             "mostly ends via the no-progress/repetition draw rules, which can take "
             "several hundred half-moves - this is expected, not a bug.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed, for reproducible runs.",
    )
    args = parser.parse_args()

    if args.mode == "human":
        play_human_vs_random(seed=args.seed)
    elif args.mode == "network":
        play_human_vs_network(checkpoint_path=args.checkpoint, seed=args.seed)
    else:
        play_random_vs_random(
            games=args.games, quiet=args.quiet, max_moves=args.max_moves, seed=args.seed
        )


if __name__ == "__main__":
    main()

