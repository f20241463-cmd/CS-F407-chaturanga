"""
The two numeric contracts Student 1 owns the source-of-truth for:

  - Move <-> action_id, section 2.4:
        action_id = from_square * 64 + to_square
        from_square = action_id // 64 ; to_square = action_id % 64

  - GameState -> tensor [13, 8, 8], float32, section 2.1 / 2.11:
        planes 0-5  : White pieces, order RAJA, MANTRI, RATHA, GAJA,
                      ASHVA, PADATI (fixes PieceType enum order - this
                      exact order must match what Student 2's network
                      is told to expect)
        planes 6-11 : Black pieces, same order
        plane 12    : all 0.0 if White to move, all 1.0 if Black to
                      move (section 2.11's proposed convention)

Board orientation is absolute: row 0 is always White's back rank,
regardless of side to move. (Flagged gap - confirm with team if you
want a relative/flipped encoding instead; that would only change this
file, nothing else.)
"""

import numpy as np

from .board import Board
from .game import GameState
from .types import Color, PieceType, col_of, row_of

NUM_ACTIONS = 4096  # 64 * 64

_PLANE_ORDER = (
    PieceType.RAJA,
    PieceType.MANTRI,
    PieceType.RATHA,
    PieceType.GAJA,
    PieceType.ASHVA,
    PieceType.PADATI,
)
_PIECE_TO_PLANE = {pt: i for i, pt in enumerate(_PLANE_ORDER)}


def move_to_action(move) -> int:
    from_sq, to_sq = move
    if not (0 <= from_sq < 64 and 0 <= to_sq < 64):
        raise ValueError(f"Square out of range in move {move}")
    return from_sq * 64 + to_sq


def action_to_move(action_id: int):
    if not (0 <= action_id < NUM_ACTIONS):
        raise ValueError(f"action_id out of range: {action_id}")
    return (action_id // 64, action_id % 64)


def board_to_planes(board: Board) -> np.ndarray:
    """The 12 piece planes only (no turn plane) - split out so tests
    can check board encoding independent of whose turn it is."""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.to_dict().items():
        plane_idx = _PIECE_TO_PLANE[piece.piece_type]
        if piece.color == Color.BLACK:
            plane_idx += 6
        planes[plane_idx, row_of(sq), col_of(sq)] = 1.0
    return planes


def legal_action_mask(state: GameState) -> np.ndarray:
    """Boolean [4096] mask: True at action_id's that are legal moves
    in this position, False everywhere else.

    This is the single shared implementation of "legal moves -> mask"
    referenced by contract section 2.6 (masking sits between the
    network's raw logits and the softmax). Students 3 and 4 should
    both import this rather than each writing their own conversion
    from state.legal_moves() to a [4096] array - two independent
    implementations of the same formula is exactly the kind of thing
    that silently drifts apart.

    Usage against raw policy logits from the network:
        mask = legal_action_mask(state)
        logits[~mask] = -np.inf   # or a large negative number
        probs = softmax(logits)
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in state.legal_moves():
        mask[move_to_action(move)] = True
    return mask


def state_to_tensor(state: GameState) -> np.ndarray:
    piece_planes = board_to_planes(state.board)
    turn_plane = np.full(
        (1, 8, 8),
        fill_value=(1.0 if state.side_to_move == Color.BLACK else 0.0),
        dtype=np.float32,
    )
    return np.concatenate([piece_planes, turn_plane], axis=0)


if __name__ == "__main__":
    # Run with: python -m chaturanga.encoding  (from the repo root)
    print("=== chaturanga.encoding demo ===\n")

    # action_id <-> move, matching the contract doc's worked example
    mv = (12, 20)
    action_id = move_to_action(mv)
    print(f"move_to_action({mv}) = {action_id}   (contract doc example: should be 788)")
    print(f"action_to_move({action_id}) = {action_to_move(action_id)}")

    state = GameState()
    tensor = state_to_tensor(state)
    print(f"\nstate_to_tensor(starting position): shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"  plane 0 (White Raja) nonzero cell: {tensor[0].nonzero()}")
    print(f"  plane 12 (turn plane) unique values: {np.unique(tensor[12])}  "
          f"(all 0.0 = White to move)")

    mask = legal_action_mask(state)
    print(f"\nlegal_action_mask(starting position): shape={mask.shape}, dtype={mask.dtype}")
    print(f"  legal actions: {mask.sum()}   (matches len(state.legal_moves()) = "
          f"{len(state.legal_moves())})")
