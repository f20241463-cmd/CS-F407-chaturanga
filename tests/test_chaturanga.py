import unittest

import numpy as np

from chaturanga.board import Board
from chaturanga.encoding import (
    NUM_ACTIONS,
    action_to_move,
    board_to_planes,
    legal_action_mask,
    move_to_action,
    state_to_tensor,
)
from chaturanga.game import GameState
from chaturanga.stub_network import random_stub_network, zero_stub_network
from chaturanga.types import Color, Piece, PieceType, square


def empty_state(side_to_move=Color.WHITE, **kwargs) -> GameState:
    return GameState(board=Board(), side_to_move=side_to_move, **kwargs)


class TestStartingPosition(unittest.TestCase):
    def test_piece_counts(self):
        b = Board.starting_position()
        white = list(b.pieces_of(Color.WHITE))
        black = list(b.pieces_of(Color.BLACK))
        self.assertEqual(len(white), 16)
        self.assertEqual(len(black), 16)

    def test_kings_present_on_expected_squares(self):
        b = Board.starting_position()
        self.assertEqual(b.find_king_square(Color.WHITE), square(0, 3))
        self.assertEqual(b.find_king_square(Color.BLACK), square(7, 3))

    def test_white_to_move_first(self):
        gs = GameState()
        self.assertEqual(gs.side_to_move, Color.WHITE)
        self.assertFalse(gs.is_game_over())
        self.assertGreater(len(gs.legal_moves()), 0)


class TestActionEncoding(unittest.TestCase):
    def test_spec_example(self):
        # from the frozen contract doc: (12,20) -> 788
        self.assertEqual(move_to_action((12, 20)), 788)
        self.assertEqual(action_to_move(788), (12, 20))

    def test_roundtrip_all_actions(self):
        for action_id in range(0, NUM_ACTIONS, 137):  # sample, not exhaustive
            mv = action_to_move(action_id)
            self.assertEqual(move_to_action(mv), action_id)

    def test_corners(self):
        self.assertEqual(move_to_action((0, 0)), 0)
        self.assertEqual(move_to_action((63, 63)), 4095)

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            move_to_action((64, 0))
        with self.assertRaises(ValueError):
            action_to_move(4096)


class TestTensorEncoding(unittest.TestCase):
    def test_shape_and_dtype(self):
        gs = GameState()
        t = state_to_tensor(gs)
        self.assertEqual(t.shape, (13, 8, 8))
        self.assertEqual(str(t.dtype), "float32")

    def test_turn_plane_convention(self):
        # section 2.11: 0 for Player 1 (White) to move, 1 for Black
        white_state = GameState(side_to_move=Color.WHITE)
        black_state = GameState(side_to_move=Color.BLACK)
        self.assertTrue((state_to_tensor(white_state)[12] == 0.0).all())
        self.assertTrue((state_to_tensor(black_state)[12] == 1.0).all())

    def test_single_king_plane_has_exactly_one_hot(self):
        b = Board.starting_position()
        planes = board_to_planes(b)
        # plane 0 = White Raja
        self.assertEqual(planes[0].sum(), 1.0)
        # plane 6 = Black Raja
        self.assertEqual(planes[6].sum(), 1.0)

    def test_padati_plane_has_eight_at_start(self):
        b = Board.starting_position()
        planes = board_to_planes(b)
        self.assertEqual(planes[5].sum(), 8.0)   # white padati
        self.assertEqual(planes[11].sum(), 8.0)  # black padati


class TestRajaMovement(unittest.TestCase):
    def test_one_square_any_direction(self):
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(4, 4)}
        expected = {
            square(5, 4), square(3, 4), square(4, 5), square(4, 3),
            square(5, 5), square(5, 3), square(3, 5), square(3, 3),
        }
        self.assertEqual(targets, expected)

    def test_cannot_move_into_check(self):
        gs = empty_state()
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        # Black Ratha sweeps the whole of row 1, so White king can't step there.
        gs.board.set_piece(square(1, 7), Piece(PieceType.RATHA, Color.BLACK))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(0, 0)}
        self.assertNotIn(square(1, 0), targets)
        self.assertIn(square(0, 1), targets)


class TestMantriMovement(unittest.TestCase):
    def test_diagonal_only(self):
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.MANTRI, Color.WHITE))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(4, 4)}
        expected = {square(5, 5), square(5, 3), square(3, 5), square(3, 3)}
        self.assertEqual(targets, expected)
        self.assertNotIn(square(5, 4), targets)  # not orthogonal


class TestRathaMovement(unittest.TestCase):
    def test_slides_and_stops_at_blocker(self):
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.RATHA, Color.WHITE))
        gs.board.set_piece(square(4, 6), Piece(PieceType.PADATI, Color.BLACK))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(4, 4)}
        # can reach and capture the blocker at (4,6) but not go past it
        self.assertIn(square(4, 6), targets)
        self.assertNotIn(square(4, 7), targets)
        self.assertIn(square(4, 5), targets)
        self.assertNotIn(square(5, 5), targets)  # no diagonal movement


class TestGajaMovement(unittest.TestCase):
    def test_jumps_over_occupied_intermediate_square(self):
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.GAJA, Color.WHITE))
        # occupy the square directly between (4,4) and (6,6)
        gs.board.set_piece(square(5, 5), Piece(PieceType.PADATI, Color.BLACK))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(4, 4)}
        self.assertIn(square(6, 6), targets)  # lands despite blocker between


class TestAshvaMovement(unittest.TestCase):
    def test_knight_jumps(self):
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.ASHVA, Color.WHITE))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(4, 4)}
        expected = {
            square(5, 6), square(5, 2), square(3, 6), square(3, 2),
            square(6, 5), square(6, 3), square(2, 5), square(2, 3),
        }
        self.assertEqual(targets, expected)


class TestPadatiMovement(unittest.TestCase):
    def test_forward_push_only_when_empty(self):
        gs = empty_state()
        gs.board.set_piece(square(3, 3), Piece(PieceType.PADATI, Color.WHITE))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(3, 3)}
        self.assertEqual(targets, {square(4, 3)})

    def test_no_double_step(self):
        gs = empty_state()
        gs.board.set_piece(square(1, 3), Piece(PieceType.PADATI, Color.WHITE))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(1, 3)}
        self.assertEqual(targets, {square(2, 3)})

    def test_diagonal_capture_only_with_enemy_present(self):
        gs = empty_state()
        gs.board.set_piece(square(3, 3), Piece(PieceType.PADATI, Color.WHITE))
        gs.board.set_piece(square(4, 4), Piece(PieceType.PADATI, Color.BLACK))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(3, 3)}
        self.assertIn(square(4, 4), targets)   # diagonal capture allowed
        self.assertIn(square(4, 3), targets)   # straight push still legal
        self.assertNotIn(square(4, 2), targets)  # empty diagonal, no capture

    def test_promotion_to_mantri(self):
        gs = empty_state()
        gs.board.set_piece(square(6, 3), Piece(PieceType.PADATI, Color.WHITE))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        new_state = gs.apply_move((square(6, 3), square(7, 3)))
        promoted = new_state.board.piece_at(square(7, 3))
        self.assertEqual(promoted.piece_type, PieceType.MANTRI)
        self.assertEqual(promoted.color, Color.WHITE)


class TestApplyMoveImmutability(unittest.TestCase):
    def test_original_state_unchanged(self):
        gs = GameState()
        original_repr = repr(gs.board)
        mv = gs.legal_moves()[0]
        _ = gs.apply_move(mv)
        self.assertEqual(repr(gs.board), original_repr)

    def test_halfmove_clock_resets_on_capture_or_padati_move(self):
        gs = empty_state()
        gs.board.set_piece(square(3, 3), Piece(PieceType.PADATI, Color.WHITE))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        gs = gs.__class__(board=gs.board, side_to_move=Color.WHITE, halfmove_clock=10)
        next_state = gs.apply_move((square(3, 3), square(4, 3)))
        self.assertEqual(next_state.halfmove_clock, 0)


class TestTerminalDetection(unittest.TestCase):
    def test_checkmate_is_loss_for_mated_side(self):
        # Black king boxed in the corner (7,0). White Ratha on column 0
        # gives check and covers the (6,0) escape square; a second
        # White Ratha on column 1 covers (7,1) and (6,1). No block or
        # capture is available -> checkmate.
        gs = empty_state(side_to_move=Color.BLACK)
        gs.board.set_piece(square(7, 0), Piece(PieceType.RAJA, Color.BLACK))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RATHA, Color.WHITE))
        gs.board.set_piece(square(0, 1), Piece(PieceType.RATHA, Color.WHITE))
        gs.board.set_piece(square(4, 4), Piece(PieceType.RAJA, Color.WHITE))
        self.assertTrue(gs.is_in_check())
        self.assertEqual(gs.legal_moves(), [])
        self.assertEqual(gs.result(), -1)

    def test_stalemate_is_draw(self):
        # Black king in the corner (7,7), NOT currently in check, but
        # every escape square is covered: the White king (at (5,6))
        # covers (6,6) and (6,7) without itself checking (7,7) - too
        # far away (distance (2,1)). A White Mantri at (6,5) covers
        # (7,6) via its one-step diagonal, without lining up on the
        # king itself and without being blocked by anything.
        gs = empty_state(side_to_move=Color.BLACK)
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        gs.board.set_piece(square(5, 6), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(6, 5), Piece(PieceType.MANTRI, Color.WHITE))
        self.assertFalse(gs.is_in_check())
        self.assertEqual(gs.legal_moves(), [])
        self.assertEqual(gs.result(), 0)

    def test_no_progress_limit_forces_draw(self):
        gs = empty_state()
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 7), Piece(PieceType.RAJA, Color.BLACK))
        gs = gs.__class__(board=gs.board, side_to_move=Color.WHITE, halfmove_clock=100)
        self.assertEqual(gs.result(), 0)

    def test_ongoing_game_returns_none(self):
        gs = GameState()
        self.assertIsNone(gs.result())
        self.assertFalse(gs.is_game_over())


class TestActionMask(unittest.TestCase):
    def test_mask_matches_legal_moves(self):
        gs = GameState()
        mask = legal_action_mask(gs)
        self.assertEqual(mask.shape, (NUM_ACTIONS,))
        self.assertEqual(str(mask.dtype), "bool")
        legal_actions = {move_to_action(m) for m in gs.legal_moves()}
        self.assertEqual(set(mask.nonzero()[0].tolist()), legal_actions)

    def test_mask_all_false_when_no_legal_moves(self):
        # reuse the checkmate position: no legal moves -> empty mask
        gs = empty_state(side_to_move=Color.BLACK)
        gs.board.set_piece(square(7, 0), Piece(PieceType.RAJA, Color.BLACK))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RATHA, Color.WHITE))
        gs.board.set_piece(square(0, 1), Piece(PieceType.RATHA, Color.WHITE))
        gs.board.set_piece(square(4, 4), Piece(PieceType.RAJA, Color.WHITE))
        mask = legal_action_mask(gs)
        self.assertFalse(mask.any())


class TestPinsAndChecks(unittest.TestCase):
    def test_pinned_piece_can_only_move_along_the_pin_line(self):
        # White Raja (4,4), White Ratha (4,5), Black Ratha (4,7), all
        # on row 4. The White Ratha is pinned: moving it off row 4
        # would expose the king to the Black Ratha along that row.
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(4, 5), Piece(PieceType.RATHA, Color.WHITE))
        gs.board.set_piece(square(4, 7), Piece(PieceType.RATHA, Color.BLACK))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.BLACK))

        pinned_targets = {mv[1] for mv in gs.legal_moves() if mv[0] == square(4, 5)}
        # staying on the pin line (including capturing the attacker) is fine
        self.assertIn(square(4, 6), pinned_targets)
        self.assertIn(square(4, 7), pinned_targets)  # captures the pinning Ratha
        # leaving the pin line, even though it's a normal Ratha move,
        # would expose the king - must be excluded
        self.assertNotIn(square(5, 5), pinned_targets)
        self.assertNotIn(square(3, 5), pinned_targets)

    def test_double_check_forces_a_king_move(self):
        # White Raja (4,4) is attacked simultaneously by a Black Ashva
        # (knight-jump from (2,5)) and a Black Ratha sweeping row 4
        # from (4,0). White also has a Ratha at (7,2) that COULD slide
        # down to block the Ratha's line at (4,2) - but blocking only
        # one of two simultaneous checks is still illegal.
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(7, 2), Piece(PieceType.RATHA, Color.WHITE))
        gs.board.set_piece(square(2, 5), Piece(PieceType.ASHVA, Color.BLACK))
        gs.board.set_piece(square(4, 0), Piece(PieceType.RATHA, Color.BLACK))
        gs.board.set_piece(square(0, 7), Piece(PieceType.RAJA, Color.BLACK))

        self.assertTrue(gs.is_in_check())
        legal = gs.legal_moves()
        self.assertGreater(len(legal), 0)  # not checkmate - king has an escape
        # every legal move must originate from the king; the Ratha's
        # blocking move addresses only one attacker and must be filtered out
        self.assertTrue(all(mv[0] == square(4, 4) for mv in legal))
        self.assertNotIn((square(7, 2), square(4, 2)), legal)

    def test_capturing_the_checking_piece_resolves_check(self):
        # Black Padati at (5,5) checks the White king at (4,4) via its
        # diagonal-forward attack. The king can simply capture it.
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(5, 5), Piece(PieceType.PADATI, Color.BLACK))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.BLACK))

        self.assertTrue(gs.is_in_check())
        legal = gs.legal_moves()
        self.assertIn((square(4, 4), square(5, 5)), legal)
        after = gs.apply_move((square(4, 4), square(5, 5)))
        self.assertFalse(after.is_in_check(Color.WHITE))

    def test_interposing_a_piece_resolves_check(self):
        # Black Ratha at (4,7) checks along row 4. A White Ratha at
        # (0,5) can slide up to (4,5), interposing between king and
        # attacker, which blocks the check without capturing anything.
        gs = empty_state()
        gs.board.set_piece(square(4, 4), Piece(PieceType.RAJA, Color.WHITE))
        gs.board.set_piece(square(0, 5), Piece(PieceType.RATHA, Color.WHITE))
        gs.board.set_piece(square(4, 7), Piece(PieceType.RATHA, Color.BLACK))
        gs.board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.BLACK))

        self.assertTrue(gs.is_in_check())
        legal = gs.legal_moves()
        block_move = (square(0, 5), square(4, 5))
        self.assertIn(block_move, legal)
        after = gs.apply_move(block_move)
        self.assertFalse(after.is_in_check(Color.WHITE))


class TestStubNetwork(unittest.TestCase):
    def test_zero_stub_shapes(self):
        gs = GameState()
        policy, value = zero_stub_network(state_to_tensor(gs))
        self.assertEqual(policy.shape, (NUM_ACTIONS,))
        self.assertEqual(str(policy.dtype), "float32")
        self.assertEqual(value, 0.0)

    def test_random_stub_shapes_and_value_range(self):
        gs = GameState()
        rng = np.random.default_rng(0)
        policy, value = random_stub_network(state_to_tensor(gs), rng=rng)
        self.assertEqual(policy.shape, (NUM_ACTIONS,))
        self.assertEqual(str(policy.dtype), "float32")
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_stub_rejects_wrong_shape(self):
        bad_tensor = np.zeros((13, 8, 7), dtype=np.float32)
        with self.assertRaises(ValueError):
            zero_stub_network(bad_tensor)


if __name__ == "__main__":
    unittest.main()
