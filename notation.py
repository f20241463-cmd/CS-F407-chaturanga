"""
Human-readable square notation, e.g. "e2" <-> square 12.

Purely a convenience layer for CLIs/debugging - nothing else in the
engine depends on this, and no frozen contract requires it. Column
letters a-h map to columns 0-7, rank numbers 1-8 map to rows 0-7.
"""

from .types import col_of, row_of, square

_FILES = "abcdefgh"


def square_to_algebraic(sq: int) -> str:
    return f"{_FILES[col_of(sq)]}{row_of(sq) + 1}"


def algebraic_to_square(text: str) -> int:
    text = text.strip().lower()
    if len(text) != 2 or text[0] not in _FILES or not text[1].isdigit():
        raise ValueError(f"Not a valid square: {text!r} (expected e.g. 'e2')")
    col = _FILES.index(text[0])
    row = int(text[1]) - 1
    if not (0 <= row < 8):
        raise ValueError(f"Rank out of range: {text!r}")
    return square(row, col)


def move_to_algebraic(move) -> str:
    from_sq, to_sq = move
    return f"{square_to_algebraic(from_sq)}{square_to_algebraic(to_sq)}"
