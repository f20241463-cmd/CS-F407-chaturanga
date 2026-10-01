"""
Raw board representation: a 64-square array of pieces, with no game
rules attached (no turn tracking, no legality). GameState (in game.py)
wraps this with the rest of the position (side to move, move counters).

Starting arrangement (documented assumption, flag to team if you want
the historically asymmetric layout instead):

    rank 0 (White): Ratha Ashva Gaja Raja Mantri Gaja Ashva Ratha
    rank 1 (White): Padati x8
    rank 6 (Black): Padati x8
    rank 7 (Black): Ratha Ashva Gaja Raja Mantri Gaja Ashva Ratha

This mirrors the chess starting layout (King and Minister facing each
other), which is simpler and unambiguous compared to reconstructions
where White and Black kings sit on different files.
"""

from typing import Dict, Iterator, List, Optional, Tuple

from .types import Color, Piece, PieceType, square

_BACK_RANK: Tuple[PieceType, ...] = (
    PieceType.RATHA,
    PieceType.ASHVA,
    PieceType.GAJA,
    PieceType.RAJA,
    PieceType.MANTRI,
    PieceType.GAJA,
    PieceType.ASHVA,
    PieceType.RATHA,
)


class Board:
    __slots__ = ("_squares",)

    def __init__(self, squares: Optional[List[Optional[Piece]]] = None):
        self._squares: List[Optional[Piece]] = (
            list(squares) if squares is not None else [None] * 64
        )

    @classmethod
    def starting_position(cls) -> "Board":
        b = cls()
        for col, ptype in enumerate(_BACK_RANK):
            b.set_piece(square(0, col), Piece(ptype, Color.WHITE))
            b.set_piece(square(7, col), Piece(ptype, Color.BLACK))
        for col in range(8):
            b.set_piece(square(1, col), Piece(PieceType.PADATI, Color.WHITE))
            b.set_piece(square(6, col), Piece(PieceType.PADATI, Color.BLACK))
        return b

    def piece_at(self, sq: int) -> Optional[Piece]:
        return self._squares[sq]

    def set_piece(self, sq: int, piece: Optional[Piece]) -> None:
        self._squares[sq] = piece

    def remove_piece(self, sq: int) -> Optional[Piece]:
        p = self._squares[sq]
        self._squares[sq] = None
        return p

    def is_empty(self, sq: int) -> bool:
        return self._squares[sq] is None

    def clone(self) -> "Board":
        return Board(self._squares)

    def pieces_of(self, color: Color) -> Iterator[Tuple[int, Piece]]:
        for sq, p in enumerate(self._squares):
            if p is not None and p.color == color:
                yield sq, p

    def find_king_square(self, color: Color) -> Optional[int]:
        for sq, p in self.pieces_of(color):
            if p.piece_type == PieceType.RAJA:
                return sq
        return None

    def to_dict(self) -> Dict[int, Piece]:
        return {sq: p for sq, p in enumerate(self._squares) if p is not None}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return self._squares == other._squares

    def __repr__(self) -> str:
        rows = []
        for row in range(7, -1, -1):
            cells = []
            for col in range(8):
                p = self._squares[square(row, col)]
                cells.append(repr(p) if p else ".")
            rows.append(" ".join(cells))
        return "\n".join(rows)


if __name__ == "__main__":
    # Run with: python -m chaturanga.board  (from the repo root)
    print("=== chaturanga.board demo ===\n")

    b = Board.starting_position()
    print(b)

    white = list(b.pieces_of(Color.WHITE))
    black = list(b.pieces_of(Color.BLACK))
    print(f"\nWhite pieces: {len(white)}   Black pieces: {len(black)}")
    print(f"White king square: {b.find_king_square(Color.WHITE)}")
    print(f"Black king square: {b.find_king_square(Color.BLACK)}")

    print(f"\nis_empty(square 20) [an empty middle square]: {b.is_empty(20)}")
    print(f"piece_at(square 3) [White's back rank]: {b.piece_at(3)!r}")

    clone = b.clone()
    clone.remove_piece(3)
    print(f"\nAfter removing a piece from the CLONE only:")
    print(f"  original piece_at(3): {b.piece_at(3)!r}   (unaffected)")
    print(f"  clone piece_at(3):    {clone.piece_at(3)!r}")
