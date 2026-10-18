"""
play_vs_ai.py

Play Chaturanga against your own MCTS, guided by either network source.
Student 3's "Agent vs. Human" evaluation mode.

Usage:
    python play_vs_ai.py                              # you play White, AI uses stub network
    python play_vs_ai.py --network real                # AI uses Student 2's real (untrained) network
    python play_vs_ai.py --human_color black           # you play Black instead
    python play_vs_ai.py --simulations 50 --verbose     # stronger AI, shows its move reasoning
    python play_vs_ai.py --checkpoint checkpoints/final_B/latest.pt --player Avisha
    python play_vs_ai.py --summary                     # totals so far from the game log

Moves are entered as four characters, from-square then to-square, no
space: e.g. "e2e4" moves whatever is on e2 to e4. Type "legal" to see
all legal moves in that format, "board" to reprint the board, "resign"
to give up (counts as an agent win), "quit" to stop (counts as
unfinished).

Every game is appended to results/human_games.csv (change with --log,
turn off with --no_log), so the Agent vs. Human score can be reported.
"""

import argparse

import numpy as np

from chaturanga.game import GameState
from chaturanga.types import Color, PieceType, square
from chaturanga import encoding
from chaturanga.notation import (
    algebraic_to_square,
    square_to_algebraic,
    move_to_algebraic,
)
from chaturanga.stub_network import random_stub_network

from mcts.search import run_mcts
from evaluate_vs_random import make_real_network_fn
from evaluation.game_endings import classify_ending
from evaluation.human_log import append_game, format_summary, summarize
from evaluation.players import filter_repeating_moves

PIECE_LETTERS = {
    PieceType.RAJA: "K",
    PieceType.MANTRI: "Q",
    PieceType.RATHA: "R",
    PieceType.GAJA: "E",
    PieceType.ASHVA: "N",
    PieceType.PADATI: "P",
}


def print_board(state):
    print()
    for row in range(7, -1, -1):
        rank = f"{row + 1}  "
        for col in range(8):
            sq = square(row, col)
            piece = state.board.piece_at(sq)
            if piece is None:
                rank += ". "
            else:
                letter = PIECE_LETTERS.get(piece.piece_type, "?")
                rank += (letter if piece.color == Color.WHITE else letter.lower()) + " "
        print(rank)
    print("   a b c d e f g h")
    print(f"   {state.side_to_move.name} to move\n")


def make_stub_network_fn():
    import numpy as np
    from functools import partial

    return partial(random_stub_network, rng=np.random.default_rng())


def parse_move_input(text, legal_moves):
    text = text.strip().lower()
    if len(text) != 4:
        return None
    try:
        from_sq = algebraic_to_square(text[0:2])
        to_sq = algebraic_to_square(text[2:4])
    except Exception:
        return None
    move = (from_sq, to_sq)
    return move if move in legal_moves else None


def human_turn(state):
    legal = state.legal_moves()
    while True:
        text = input(
            "Your move (e.g. e2e4, or 'legal'/'board'/'resign'/'quit'): "
        ).strip()
        if text.lower() in ("quit", "resign"):
            return text.lower()
        if text.lower() == "board":
            print_board(state)
            continue
        if text.lower() == "legal":
            print("  " + ", ".join(move_to_algebraic(m) for m in legal))
            continue
        move = parse_move_input(text, legal)
        if move is None:
            print("  not a legal move, try again (type 'legal' to see options)")
            continue
        return move


def ai_turn(state, network_fn, num_simulations, c_puct, verbose, avoid_repetition):
    visit_counts, move = run_mcts(
        root_state=state,
        network_fn=network_fn,
        num_simulations=num_simulations,
        c_puct=c_puct,
        state_to_tensor_fn=encoding.state_to_tensor,
        legal_action_mask_fn=encoding.legal_action_mask,
        action_to_move_fn=encoding.action_to_move,
    )
    if avoid_repetition:
        # Same rule as the benchmark: skip a move that causes a threefold
        # repetition draw unless every searched move does.
        visit_counts, _ = filter_repeating_moves(state, visit_counts)
        move = encoding.action_to_move(int(np.argmax(visit_counts)))
    if verbose:
        top = sorted(
            (
                (encoding.action_to_move(a), int(v))
                for a, v in enumerate(visit_counts)
                if v > 0
            ),
            key=lambda x: -x[1],
        )[:5]
        print("  AI considered (top 5 by visits):")
        for m, v in top:
            print(f"    {move_to_algebraic(m):6s} visits={v}")
    print(f"  AI plays: {move_to_algebraic(move)}")
    return move


def main():
    parser = argparse.ArgumentParser(description="Play Chaturanga against MCTS")
    parser.add_argument("--network", choices=["stub", "real"], default="stub")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to a trained checkpoint (.pt); implies --network real",
    )
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--human_color", choices=["white", "black"], default="white")
    parser.add_argument(
        "--verbose", action="store_true", help="show the AI's top candidate moves"
    )
    parser.add_argument(
        "--no_avoid_repetition",
        action="store_true",
        help="let the AI repeat positions (by default it avoids threefold repetition, "
        "as in the benchmark)",
    )
    parser.add_argument("--player", default="human", help="your name, for the game log")
    parser.add_argument("--log", default="results/human_games.csv")
    parser.add_argument("--no_log", action="store_true", help="do not record this game")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print the totals from --log (all checkpoints, or only --checkpoint) and exit",
    )
    args = parser.parse_args()

    if args.summary:
        print(f"{args.log}: " + format_summary(summarize(args.log, args.checkpoint)))
        return

    if args.checkpoint:
        args.network = "real"
    network_fn = (
        make_stub_network_fn()
        if args.network == "stub"
        else make_real_network_fn(checkpoint=args.checkpoint)
    )
    human_color = Color.WHITE if args.human_color == "white" else Color.BLACK

    print(
        f"Playing against MCTS ({args.checkpoint or args.network + ' network'}, "
        f"{args.simulations} simulations/move)."
    )
    print(f"You are {human_color.name}. Moves: four characters, e.g. e2e4.\n")

    state = GameState()
    moves = []
    print_board(state)

    stopped = None
    while not state.is_game_over():
        if state.side_to_move == human_color:
            move = human_turn(state)
            if move in ("quit", "resign"):
                stopped = move
                break
        else:
            move = ai_turn(
                state,
                network_fn,
                args.simulations,
                args.c_puct,
                args.verbose,
                not args.no_avoid_repetition,
            )

        state = state.apply_move(move)
        moves.append(move_to_algebraic(move))
        print_board(state)

    if stopped == "resign":
        result, ending = "agent_win", "resigned"
        print("You resigned: the AI wins.")
    elif stopped == "quit":
        result, ending = "unfinished", "quit"
        print("Game stopped early (recorded as unfinished).")
    else:
        ending = classify_ending(state)
        if state.result() == 0:
            result = "draw"
            print(f"Game over: draw ({ending}).")
        else:
            loser = state.side_to_move
            winner = Color.BLACK if loser == Color.WHITE else Color.WHITE
            result = "human_win" if winner == human_color else "agent_win"
            who = "You win" if result == "human_win" else "The AI wins"
            print(f"Game over: {who} ({winner.name}, {ending})!")

    if not args.no_log:
        append_game(
            args.log,
            player=args.player,
            checkpoint=args.checkpoint or f"{args.network} network (untrained)",
            simulations=args.simulations,
            human_color=args.human_color,
            result=result,
            ending=ending,
            plies=len(moves),
            moves=" ".join(moves),
        )
        print(f"Recorded in {args.log}. " + format_summary(summarize(args.log)))


if __name__ == "__main__":
    main()
