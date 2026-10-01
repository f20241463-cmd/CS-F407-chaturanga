"""
Piece movement rules and pseudo-legal move generation.

"Pseudo-legal" = obeys how the piece moves and can't land on your own
piece, but does NOT check whether it leaves your own Raja in check.
game.py filters pseudo-legal moves down to fully legal ones.

Piece rules implemented here (see project brief for the definitions):
    RAJA   - one square, any of 8 directions
    MANTRI - one square, diagonal only (4 directions)
    RATHA  - any distance, orthogonal, blocked by pieces (rook-like)
    GAJA   - exactly two squares diagonally, JUMPS over the square
             in between (blocking piece there, if any, is irrelevant)
    ASHVA  - knight-style L-jump, ignores blocking
    PADATI - one square straight forward (only if empty, no capture);
             one square diagonally forward ONLY to capture an enemy
             piece; no initial double-step; promotes to MANTRI on
             reaching the far rank (row 7 for White, row 0 for Black)
"""

from typing import List, Tuple

from .board import Board
from .types import Color, Piece, PieceType, col_of, on_board, row_of, square

_ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_ALL8 = _ORTHO + _DIAG
_KNIGHT = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]


def _step_targets(board: Board, sq: int, color: Color, deltas) -> List[int]:
    """One-step moves (Raja, Mantri): land on empty or enemy square."""
    out = []
    r, c = row_of(sq), col_of(sq)
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if not on_board(nr, nc):
            continue
        target = square(nr, nc)
        occ = board.piece_at(target)
        if occ is None or occ.color != color:
            out.append(target)
    return out


def _sliding_targets(board: Board, sq: int, color: Color, deltas) -> List[int]:
    """Ratha: slide until a piece blocks the path (capture that piece
    if it's an enemy, stop either way)."""
    out = []
    r, c = row_of(sq), col_of(sq)
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        while on_board(nr, nc):
            target = square(nr, nc)
            occ = board.piece_at(target)
            if occ is None:
                out.append(target)
            else:
                if occ.color != color:
                    out.append(target)
                break
            nr, nc = nr + dr, nc + dc
    return out


def _jump_targets(board: Board, sq: int, color: Color, deltas) -> List[int]:
    """Ashva / Gaja: fixed-offset jumps that ignore anything in between."""
    out = []
    r, c = row_of(sq), col_of(sq)
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if not on_board(nr, nc):
            continue
        target = square(nr, nc)
        occ = board.piece_at(target)
        if occ is None or occ.color != color:
            out.append(target)
    return out


_GAJA_DELTAS = [(2, 2), (2, -2), (-2, 2), (-2, -2)]


def _padati_forward_dir(color: Color) -> int:
    return 1 if color == Color.WHITE else -1


def _padati_moves(board: Board, sq: int, color: Color) -> List[int]:
    out = []
    r, c = row_of(sq), col_of(sq)
    d = _padati_forward_dir(color)
    fr = r + d
    if on_board(fr, c) and board.is_empty(square(fr, c)):
        out.append(square(fr, c))
    for dc in (-1, 1):
        nc = c + dc
        if on_board(fr, nc):
            target = square(fr, nc)
            occ = board.piece_at(target)
            if occ is not None and occ.color != color:
                out.append(target)
    return out


def _padati_attack_squares(sq: int, color: Color) -> List[int]:
    """Squares a Padati threatens (diagonal-forward), regardless of
    what currently occupies them. Used for check detection only."""
    out = []
    r, c = row_of(sq), col_of(sq)
    d = _padati_forward_dir(color)
    fr = r + d
    for dc in (-1, 1):
        nc = c + dc
        if on_board(fr, nc):
            out.append(square(fr, nc))
    return out


def piece_targets(board: Board, sq: int, piece: Piece) -> List[int]:
    """Pseudo-legal destination squares for the piece at `sq`."""
    color = piece.color
    pt = piece.piece_type
    if pt == PieceType.RAJA:
        return _step_targets(board, sq, color, _ALL8)
    if pt == PieceType.MANTRI:
        return _step_targets(board, sq, color, _DIAG)
    if pt == PieceType.RATHA:
        return _sliding_targets(board, sq, color, _ORTHO)
    if pt == PieceType.GAJA:
        return _jump_targets(board, sq, color, _GAJA_DELTAS)
    if pt == PieceType.ASHVA:
        return _jump_targets(board, sq, color, _KNIGHT)
    if pt == PieceType.PADATI:
        return _padati_moves(board, sq, color)
    raise ValueError(f"Unknown piece type: {pt}")


def attack_squares(board: Board, sq: int, piece: Piece) -> List[int]:
    """Squares this piece threatens to capture on, for check detection.
    Identical to piece_targets() except for Padati, whose forward push
    is not an attack and whose diagonal attack doesn't require an
    enemy to already be standing there."""
    if piece.piece_type == PieceType.PADATI:
        return _padati_attack_squares(sq, piece.color)
    return piece_targets(board, sq, piece)


def generate_pseudo_legal_moves(board: Board, color: Color) -> List[Tuple[int, int]]:
    moves: List[Tuple[int, int]] = []
    for sq, piece in board.pieces_of(color):
        for target in piece_targets(board, sq, piece):
            moves.append((sq, target))
    return moves


def is_square_attacked(board: Board, sq: int, by_color: Color) -> bool:
    for from_sq, piece in board.pieces_of(by_color):
        if sq in attack_squares(board, from_sq, piece):
            return True
    return False


if __name__ == "__main__":
    # Run with: python -m chaturanga.moves  (from the repo root)
    from .notation import square_to_algebraic

    print("=== chaturanga.moves demo ===\n")

    board = Board.starting_position()
    white_moves = generate_pseudo_legal_moves(board, Color.WHITE)
    print(f"White pseudo-legal moves from the starting position: {len(white_moves)}")

    # isolate one Ratha on an otherwise empty board to show pure sliding behavior
    empty = Board()
    empty.set_piece(square(4, 4), Piece(PieceType.RATHA, Color.WHITE))
    empty.set_piece(square(4, 6), Piece(PieceType.PADATI, Color.BLACK))  # a blocker
    targets = piece_targets(empty, square(4, 4), Piece(PieceType.RATHA, Color.WHITE))
    print(f"\nRatha alone on e5, enemy Padati on g5 (blocks the slide):")
    print("  reachable squares:", sorted(square_to_algebraic(t) for t in targets))

    # Gaja jumping clean over an occupied intermediate square
    empty2 = Board()
    empty2.set_piece(square(4, 4), Piece(PieceType.GAJA, Color.WHITE))
    empty2.set_piece(square(5, 5), Piece(PieceType.PADATI, Color.BLACK))  # sits in between
    g_targets = piece_targets(empty2, square(4, 4), Piece(PieceType.GAJA, Color.WHITE))
    print(f"\nGaja on e5, enemy Padati on f6 directly in its jump path:")
    print("  reachable squares (jump ignores the blocker):",
          sorted(square_to_algebraic(t) for t in g_targets))

    # check detection
    check_board = Board()
    check_board.set_piece(square(0, 0), Piece(PieceType.RAJA, Color.WHITE))
    check_board.set_piece(square(0, 7), Piece(PieceType.RATHA, Color.BLACK))
    attacked = is_square_attacked(check_board, square(0, 0), Color.BLACK)
    print(f"\nWhite Raja on a1, Black Ratha on h1 (clear rank): is a1 attacked? {attacked}")
