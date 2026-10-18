"""
The fast rules engine must behave exactly like the original one.

tests/reference_rules.py is a frozen copy of the engine before the speed-up.
These tests play many seeded random games and build many random positions,
and check that both engines agree on everything the rest of the code relies
on: the legal moves (same list, same order), check, every attacked square,
and the game result, at every ply.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

import reference_rules as ref  # noqa: E402

from chaturanga.board import Board  # noqa: E402
from chaturanga.game import GameState  # noqa: E402
from chaturanga.moves import is_square_attacked  # noqa: E402
from chaturanga.types import Color, Piece, PieceType  # noqa: E402


def as_reference(state):
    """The same position as a ReferenceGameState."""
    r = ref.ReferenceGameState(
        board=Board(state.board._squares),
        side_to_move=state.side_to_move,
        halfmove_clock=state.halfmove_clock,
    )
    r.history = tuple(
        (tuple((sq, p) for sq, p in enumerate(squares) if p is not None), side)
        for squares, side in state.history
    )
    return r


class TestEngineEquivalence(unittest.TestCase):
    def assert_same_position(self, fast, slow, where):
        self.assertEqual(fast.legal_moves(), slow.legal_moves(), where)
        self.assertEqual(fast.result(), slow.result(), where)
        for color in Color:
            self.assertEqual(fast.is_in_check(color), slow.is_in_check(color), where)
            for sq in range(64):
                self.assertEqual(
                    is_square_attacked(fast.board, sq, color),
                    ref.is_square_attacked(slow.board, sq, color),
                    f"{where}: square {sq} attacked by {color.name}",
                )

    def test_random_games_agree_at_every_ply(self):
        rng = random.Random(0)
        positions = 0
        for game in range(40):
            fast, slow = GameState(), ref.ReferenceGameState()
            plies = 0
            while True:
                self.assert_same_position(fast, slow, f"game {game} ply {plies}")
                positions += 1
                if fast.is_game_over() or plies >= 400:
                    break
                move = rng.choice(fast.legal_moves())
                fast, slow = fast.apply_move(move), slow.apply_move(move)
                plies += 1
        self.assertGreater(positions, 5000)

    def test_random_positions_agree(self):
        rng = random.Random(1)
        others = [pt for pt in PieceType if pt != PieceType.RAJA]
        for trial in range(3000):
            board = Board()
            free = list(range(64))
            rng.shuffle(free)
            # Usually both kings, sometimes one missing, to cover that case too.
            for color in Color:
                if rng.random() < 0.95:
                    board.set_piece(free.pop(), Piece(PieceType.RAJA, color))
            for _ in range(rng.randint(0, 14)):
                board.set_piece(
                    free.pop(), Piece(rng.choice(others), rng.choice(list(Color)))
                )
            fast = GameState(
                board=board,
                side_to_move=rng.choice(list(Color)),
                halfmove_clock=rng.choice([0, 50, 99, 100]),
            )
            self.assert_same_position(fast, as_reference(fast), f"position {trial}")

    def test_cache_follows_board_edits(self):
        state = GameState(board=Board())
        state.board.set_piece(0, Piece(PieceType.RAJA, Color.WHITE))
        state.board.set_piece(63, Piece(PieceType.RAJA, Color.BLACK))
        before = state.legal_moves()
        state.board.set_piece(9, Piece(PieceType.RATHA, Color.BLACK))
        self.assertNotEqual(state.legal_moves(), before)
        self.assertEqual(state.legal_moves(), as_reference(state).legal_moves())

    def test_returned_moves_cannot_change_the_cache(self):
        state = GameState()
        state.legal_moves().clear()
        self.assertEqual(len(state.legal_moves()), 16)

    def test_threefold_repetition_agrees(self):
        # Knights out and back twice: the start position appears three times.
        shuffle = [(1, 18), (57, 42), (18, 1), (42, 57)] * 2
        fast, slow = GameState(), ref.ReferenceGameState()
        for move in shuffle:
            fast, slow = fast.apply_move(move), slow.apply_move(move)
            self.assertEqual(fast.result(), slow.result())
        self.assertEqual(fast.result(), 0)
        self.assertTrue(fast._is_threefold_repetition())


if __name__ == "__main__":
    unittest.main()
