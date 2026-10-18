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
from evaluation.human_log import append_game, format_summary, summarize, wilson_interval
from evaluation.game_endings import classify_ending, count_endings
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


def _kings_and_rooks(rook_squares, side_to_move=Color.BLACK):
    """Black king on a8, White king on h1, White rooks on the given squares."""
    state = GameState(board=Board(), side_to_move=side_to_move)
    state.board.set_piece(square(7, 0), Piece(PieceType.RAJA, Color.BLACK))
    state.board.set_piece(square(0, 7), Piece(PieceType.RAJA, Color.WHITE))
    for sq in rook_squares:
        state.board.set_piece(sq, Piece(PieceType.RATHA, Color.WHITE))
    return state


class TestGameEndings(unittest.TestCase):
    def test_each_rule_is_named(self):
        # Rooks on a1 and b1: the a-file rook checks, the b-file rook covers b7/b8.
        self.assertEqual(
            classify_ending(_kings_and_rooks([square(0, 0), square(0, 1)])),
            "checkmate",
        )
        # Rooks on b1 and h7: no check, but a7, b7 and b8 are all covered.
        self.assertEqual(
            classify_ending(_kings_and_rooks([square(0, 1), square(6, 7)])),
            "stalemate",
        )
        self.assertEqual(
            classify_ending(GameState(halfmove_clock=100)), "no-progress rule"
        )
        state = GameState()
        for move in [(6, 21), (62, 45), (21, 6), (45, 62)] * 2:
            state = state.apply_move(move)
        self.assertEqual(classify_ending(state), "threefold repetition")
        no_black_king = GameState(board=Board(), side_to_move=Color.BLACK)
        no_black_king.board.set_piece(square(0, 7), Piece(PieceType.RAJA, Color.WHITE))
        self.assertEqual(classify_ending(no_black_king), "king captured")
        self.assertEqual(classify_ending(GameState()), "unfinished (ply cap)")

    def test_count_endings_respects_the_ply_cap(self):
        counts = count_endings(
            random_player, random_player, 2, np.random.default_rng(0), max_plies=5
        )
        self.assertEqual(counts, {"unfinished (ply cap)": 2})


class TestHumanLog(unittest.TestCase):
    def test_append_and_summarize(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "human.csv")
            self.assertEqual(summarize(path)["games"], 0)
            for result, ckpt in [
                ("agent_win", "a.pt"),
                ("human_win", "a.pt"),
                ("draw", "a.pt"),
                ("unfinished", "a.pt"),
                ("agent_win", "b.pt"),
            ]:
                append_game(path, player="p", checkpoint=ckpt, result=result)
            s = summarize(path, checkpoint="a.pt")
            self.assertEqual(
                (s["games"], s["agent_win"], s["human_win"], s["draw"]), (4, 1, 1, 1)
            )
            self.assertAlmostEqual(s["score"], 0.5)  # (1 + 0.5) / 3 finished
            self.assertEqual(summarize(path)["games"], 5)
            self.assertIn("Agent score 0.500", format_summary(s))

    def test_wilson_interval_stays_wide_when_one_side_wins_everything(self):
        low, high = wilson_interval(0.0, 6)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.3)  # the +/- margin would claim exactly 0
        low, high = wilson_interval(0.25, 6)
        self.assertAlmostEqual(low, 0.060, places=3)
        self.assertAlmostEqual(high, 0.635, places=3)
        self.assertEqual(wilson_interval(0.5, 0), (None, None))

    def test_rejects_unknown_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                append_game(os.path.join(tmp, "h.csv"), result="win")


if __name__ == "__main__":
    unittest.main()
