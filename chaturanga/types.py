"""
Core types shared by the whole engine.

Square numbering (frozen contract, section 2.2):
    square = row * 8 + column
    (0,0) -> 0   (0,7) -> 7   (1,0) -> 8   (7,7) -> 63

Row 0 is White's back rank (absolute board orientation - not flipped
per side to move). This is a Week-0 gap that was left open in the
team doc; flag it to Student 2/3 if you want relative orientation
instead, since it changes the tensor encoding and nothing else.
"""

from enum import IntEnum
from typing import NamedTuple


class Color(IntEnum):
    WHITE = 0
    BLACK = 1

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self == Color.WHITE else Color.WHITE


class PieceType(IntEnum):
    """Order here fixes the plane order used in state_to_tensor.
    This order must be communicated to Student 2 (network) since the
    contract doc does not pin it down explicitly."""
    RAJA = 0    # King: one square any direction
    MANTRI = 1  # Counselor: one square diagonally only
    RATHA = 2   # Chariot: any distance orthogonally (rook)
    GAJA = 3    # Elephant: exactly two squares diagonally, jumps
    ASHVA = 4   # Horse: knight-style L-jump
    PADATI = 5  # Foot soldier: one step forward, diagonal capture


class Piece(NamedTuple):
    piece_type: PieceType
    color: Color

    def __repr__(self) -> str:
        symbols = {
            PieceType.RAJA: "K",
            PieceType.MANTRI: "M",
            PieceType.RATHA: "R",
            PieceType.GAJA: "G",
            PieceType.ASHVA: "A",
            PieceType.PADATI: "P",
        }
        s = symbols[self.piece_type]
        return s if self.color == Color.WHITE else s.lower()


def square(row: int, col: int) -> int:
    return row * 8 + col


def row_of(sq: int) -> int:
    return sq // 8


def col_of(sq: int) -> int:
    return sq % 8


def on_board(row: int, col: int) -> bool:
    return 0 <= row < 8 and 0 <= col < 8


if __name__ == "__main__":
    # Run with: python -m chaturanga.types  (from the repo root)
    print("=== chaturanga.types demo ===\n")

    print("Color.WHITE =", int(Color.WHITE), " Color.BLACK =", int(Color.BLACK))
    print("Color.WHITE.opponent =", Color.WHITE.opponent.name)

    print("\nPieceType order (this fixes the tensor plane order in encoding.py):")
    for pt in PieceType:
        print(f"  {pt.value}: {pt.name}")

    white_ashva = Piece(PieceType.ASHVA, Color.WHITE)
    black_ashva = Piece(PieceType.ASHVA, Color.BLACK)
    print(f"\nExample pieces: White Ashva = {white_ashva!r}   Black Ashva = {black_ashva!r}")

    sq = square(3, 4)
    print(f"\nsquare(row=3, col=4) = {sq}")
    print(f"row_of({sq}) = {row_of(sq)}   col_of({sq}) = {col_of(sq)}")
    print(f"on_board(3, 4) = {on_board(3, 4)}   on_board(8, 4) = {on_board(8, 4)}")
