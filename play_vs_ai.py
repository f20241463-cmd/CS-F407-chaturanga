"""
play_vs_ai.py

Play Chaturanga against your own MCTS, guided by either network source.
Student 3's "Agent vs. Human" evaluation mode.

Usage:
    python play_vs_ai.py                              # you play White, AI uses stub network
    python play_vs_ai.py --network real                # AI uses Student 2's real (untrained) network
    python play_vs_ai.py --human_color black           # you play Black instead
    python play_vs_ai.py --simulations 50 --verbose     # stronger AI, shows its move reasoning

Moves are entered as four characters, from-square then to-square, no
space: e.g. "e2e4" moves whatever is on e2 to e4. Type "legal" to see
all legal moves in that format, "board" to reprint the board, "quit"
to exit.

Swap in a trained checkpoint later with:
    from network.checkpoint import load_checkpoint
    net, _, _, _ = load_checkpoint("path/to/checkpoint.pt")
    network_fn = make_network_fn(net)
and pass that in place of make_real_network_fn() below -- nothing else
in this file needs to change.
"""

import argparse

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


def make_real_network_fn():
    from network.model import ChaturangaNet
    from network.agent import make_network_fn

    net = ChaturangaNet()
    return make_network_fn(net)


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
        text = input("Your move (e.g. e2e4, or 'legal'/'board'/'quit'): ").strip()
        if text.lower() == "quit":
            return None
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


def ai_turn(state, network_fn, num_simulations, c_puct, verbose):
    visit_counts, move = run_mcts(
        root_state=state,
        network_fn=network_fn,
        num_simulations=num_simulations,
        c_puct=c_puct,
        state_to_tensor_fn=encoding.state_to_tensor,
        legal_action_mask_fn=encoding.legal_action_mask,
        action_to_move_fn=encoding.action_to_move,
    )
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
    parser.add_argument("--simulations", type=int, default=30)
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--human_color", choices=["white", "black"], default="white")
    parser.add_argument(
        "--verbose", action="store_true", help="show the AI's top candidate moves"
    )
    args = parser.parse_args()

    network_fn = (
        make_stub_network_fn() if args.network == "stub" else make_real_network_fn()
    )
    human_color = Color.WHITE if args.human_color == "white" else Color.BLACK

    print(
        f"Playing against MCTS ({args.network} network, {args.simulations} simulations/move)."
    )
    print(f"You are {human_color.name}. Moves: four characters, e.g. e2e4.\n")

    state = GameState()
    print_board(state)

    while not state.is_game_over():
        if state.side_to_move == human_color:
            move = human_turn(state)
            if move is None:
                print("Game ended early.")
                return
        else:
            move = ai_turn(
                state, network_fn, args.simulations, args.c_puct, args.verbose
            )

        state = state.apply_move(move)
        print_board(state)

    result = state.result()
    if result == 0 or result is None:
        print("Game over: draw.")
    else:
        loser = state.side_to_move
        winner = Color.BLACK if loser == Color.WHITE else Color.WHITE
        who = "You" if winner == human_color else "The AI"
        print(f"Game over: {who} ({winner.name}) win{'s' if who == 'The AI' else ''}!")


if __name__ == "__main__":
    main()
