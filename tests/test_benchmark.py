"""
tests/test_benchmark.py

Tests for the evaluation suite: baseline players, the MCTS player, the
match runner, and checkpoint discovery. Only TestFindIterations needs
torch (skipped without it).
"""

import os
import tempfile
import unittest

import numpy as np

from chaturanga.board import Board
from chaturanga.game import GameState
from chaturanga.stub_network import zero_stub_network
from chaturanga.types import Color, Piece, PieceType, square
from evaluation.players import (
    greedy_capture_player,
    make_mcts_player,
    play_match,
    random_player,
)

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


def _rook_can_take_knight_or_pawn():
    """White rook on a4 can capture a Black knight on f4 or a Black pawn on a7."""
    state = GameState(board=Board(), side_to_move=Color.WHITE)
    state.board.set_piece(square(0, 4), Piece(PieceType.RAJA, Color.WHITE))
    state.board.set_piece(square(7, 4), Piece(PieceType.RAJA, Color.BLACK))
    state.board.set_piece(square(3, 0), Piece(PieceType.RATHA, Color.WHITE))
    state.board.set_piece(square(3, 5), Piece(PieceType.ASHVA, Color.BLACK))
    state.board.set_piece(square(6, 0), Piece(PieceType.PADATI, Color.BLACK))
    return state


class TestBaselinePlayers(unittest.TestCase):
    def test_greedy_takes_the_most_valuable_piece(self):
        state = _rook_can_take_knight_or_pawn()
        # The rook could also take the pawn, which is worth less.
        self.assertIn((square(3, 0), square(6, 0)), state.legal_moves())
        for seed in range(5):
            move = greedy_capture_player(state, np.random.default_rng(seed))
            self.assertEqual(move, (square(3, 0), square(3, 5)))

    def test_greedy_without_captures_plays_a_legal_move(self):
        state = GameState()
        move = greedy_capture_player(state, np.random.default_rng(0))
        self.assertIn(move, state.legal_moves())

    def test_random_player_plays_legal_moves(self):
        state = GameState()
        rng = np.random.default_rng(0)
        for _ in range(10):
            self.assertIn(random_player(state, rng), state.legal_moves())

    def test_mcts_player_plays_a_legal_move(self):
        state = GameState()
        player = make_mcts_player(zero_stub_network, simulations=8, temperature=0.0)
        self.assertIn(player(state, np.random.default_rng(0)), state.legal_moves())


class TestPlayMatch(unittest.TestCase):
    def test_colors_alternate_and_counts_add_up(self):
        seen = []
        player = make_mcts_player(zero_stub_network, simulations=4)
        counts = play_match(
            player,
            random_player,
            games=3,
            rng=np.random.default_rng(0),
            max_plies=6,
            on_game=lambda g, a_is_white, outcome, plies: seen.append(
                (a_is_white, outcome, plies)
            ),
        )
        self.assertEqual([s[0] for s in seen], [True, False, True])
        self.assertEqual(counts["unfinished"], 3)  # nothing ends in 6 plies
        self.assertEqual(counts["plies"], [6, 6, 6])
        self.assertEqual(
            counts["wins"] + counts["losses"] + counts["draws"] + counts["unfinished"],
            3,
        )


@unittest.skipIf(torch is None, "torch not installed")
class TestFindIterations(unittest.TestCase):
    def test_finds_and_sorts_iteration_checkpoints(self):
        from evaluation.benchmark import find_iterations

        with tempfile.TemporaryDirectory() as tmp:
            for name in ["iter_0010.pt", "iter_0002.pt", "latest.pt", "notes.txt"]:
                open(os.path.join(tmp, name), "w").close()
            found = find_iterations(tmp)
        self.assertEqual([it for it, _ in found], [2, 10])


if __name__ == "__main__":
    unittest.main()
