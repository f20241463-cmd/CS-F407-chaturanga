"""
GameState = Board + whose turn it is + enough history to detect draws.

This is the class Students 2/3/4 actually talk to. Kept immutable:
apply_move() returns a *new* GameState rather than mutating in place,
which is the safe default for MCTS (different tree branches must never
share mutable state).

Terminal-state convention (documented assumption - confirm with team,
the original doc didn't fix this):
    - Checkmate: the side to move has no legal moves and is in check
      -> that side loses.
    - Stalemate: the side to move has no legal moves and is NOT in
      check -> draw.
    - No-progress rule: 100 half-moves (no capture, no Padati move)
      -> draw, to guarantee self-play games terminate.
    - Threefold repetition of the same (board, side_to_move) -> draw.
"""

from typing import List, Optional, Tuple

from .board import Board
from .moves import generate_pseudo_legal_moves, is_square_attacked
from .types import Color, PieceType, row_of

Move = Tuple[int, int]

NO_PROGRESS_LIMIT = 100  # half-moves


class GameState:
    def __init__(
        self,
        board: Optional[Board] = None,
        side_to_move: Color = Color.WHITE,
        halfmove_clock: int = 0,
        history: Optional[Tuple[Tuple, ...]] = None,
    ):
        self.board = board if board is not None else Board.starting_position()
        self.side_to_move = side_to_move
        self.halfmove_clock = halfmove_clock
        # history: tuple of (board_squares_tuple, side_to_move) snapshots,
        # used only for threefold-repetition detection.
        self.history: Tuple[Tuple, ...] = history if history is not None else ()

    # ---- check / legality -------------------------------------------------

    def is_in_check(self, color: Optional[Color] = None) -> bool:
        color = self.side_to_move if color is None else color
        king_sq = self.board.find_king_square(color)
        if king_sq is None:
            # King already captured (shouldn't happen in a well-formed
            # game reached only through legal moves, but don't crash).
            return False
        return is_square_attacked(self.board, king_sq, color.opponent)

    def legal_moves(self) -> List[Move]:
        legal: List[Move] = []
        for mv in generate_pseudo_legal_moves(self.board, self.side_to_move):
            if not self._leaves_own_king_in_check(mv):
                legal.append(mv)
        return legal

    def _leaves_own_king_in_check(self, move: Move) -> bool:
        # Cheap simulate-and-check-and-discard; fine at this scale
        # (Chaturanga has few enough pieces/moves per position that
        # this doesn't need to be optimized before it's a proven
        # bottleneck).
        next_board = self._board_after(move)
        king_sq = next_board.find_king_square(self.side_to_move)
        if king_sq is None:
            return True  # own king was just captured - illegal
        return is_square_attacked(next_board, king_sq, self.side_to_move.opponent)

    # ---- applying moves -----------------------------------------------

    def _board_after(self, move: Move) -> Board:
        from_sq, to_sq = move
        new_board = self.board.clone()
        piece = new_board.remove_piece(from_sq)
        if piece is None:
            raise ValueError(f"No piece on source square {from_sq}")
        # Promotion: Padati reaching the far rank becomes Mantri.
        if piece.piece_type == PieceType.PADATI:
            dest_row = row_of(to_sq)
            if (piece.color == Color.WHITE and dest_row == 7) or (
                piece.color == Color.BLACK and dest_row == 0
            ):
                piece = piece._replace(piece_type=PieceType.MANTRI)
        new_board.set_piece(to_sq, piece)
        return new_board

    def apply_move(self, move: Move) -> "GameState":
        if move not in self.legal_moves():
            raise ValueError(f"Illegal move: {move}")
        from_sq, to_sq = move
        moved_piece = self.board.piece_at(from_sq)
        is_capture = self.board.piece_at(to_sq) is not None
        is_padati_move = moved_piece is not None and moved_piece.piece_type == PieceType.PADATI

        new_board = self._board_after(move)
        new_clock = 0 if (is_capture or is_padati_move) else self.halfmove_clock + 1
        new_history = self.history + (self._snapshot(),)

        return GameState(
            board=new_board,
            side_to_move=self.side_to_move.opponent,
            halfmove_clock=new_clock,
            history=new_history,
        )

    def _snapshot(self) -> Tuple:
        return (tuple(self.board.to_dict().items()), self.side_to_move)

    # ---- terminal detection ------------------------------------------

    def is_game_over(self) -> bool:
        return self.result() is not None

    def result(self) -> Optional[int]:
        """Terminal result from the perspective of self.side_to_move
        (the player who would move next, if the game isn't over).
        +1 = that player has won, -1 = lost, 0 = draw, None = ongoing.
        Matches contract section 2.10."""
        if self.board.find_king_square(self.side_to_move) is None:
            return -1  # own king already gone -> loss

        if not self.legal_moves():
            if self.is_in_check():
                return -1  # checkmate: side to move has lost
            return 0  # stalemate -> draw

        if self.halfmove_clock >= NO_PROGRESS_LIMIT:
            return 0

        if self._is_threefold_repetition():
            return 0

        return None

    def _is_threefold_repetition(self) -> bool:
        current = self._snapshot()
        count = 1  # current position counts once
        for past in self.history:
            if past == current:
                count += 1
        return count >= 3

    def __repr__(self) -> str:
        return f"GameState(side_to_move={self.side_to_move.name})\n{self.board!r}"


if __name__ == "__main__":
    # Run with: python -m chaturanga.game  (from the repo root)
    import random

    from .notation import move_to_algebraic

    print("=== chaturanga.game demo ===\n")

    state = GameState()
    print(state)
    print(f"\nside_to_move: {state.side_to_move.name}")
    print(f"legal_moves(): {len(state.legal_moves())} options, "
          f"e.g. {[move_to_algebraic(m) for m in state.legal_moves()[:4]]}")
    print(f"is_game_over(): {state.is_game_over()}   result(): {state.result()}")

    move = state.legal_moves()[0]
    next_state = state.apply_move(move)
    print(f"\nAfter {move_to_algebraic(move)}:")
    print(f"  original state unchanged? {state.side_to_move.name == 'WHITE'}")
    print(f"  new side_to_move: {next_state.side_to_move.name}")

    print("\nPlaying a full random game to completion...")
    random.seed(0)
    s = GameState()
    half_moves = 0
    while not s.is_game_over():
        s = s.apply_move(random.choice(s.legal_moves()))
        half_moves += 1
    print(f"Finished after {half_moves} half-moves.")
    print(f"result() from {s.side_to_move.name}'s perspective: {s.result()}")
