# chaturanga_env — Environment & Rules module (Student 1)

This package implements the Chaturanga board, movement rules, legal-move
generation, terminal detection, and the two numeric contracts
(`GameState -> [13,8,8]` tensor, `Move <-> action_id`) that the rest of
the team builds on top of.

Status: Week 2 deliverable complete. 36/36 tests passing.

## Quick start

```bash
python -m unittest discover -s tests -v     # run the test suite
python play.py                               # you (White) vs random (Black)
python play.py --mode random --games 300 --quiet   # stress test
python integration_preview.py                # env -> tensor -> stub network -> move
```

## Public API, by file

### `chaturanga/game.py` — `GameState`
The class everyone actually talks to.

| Method / attribute | Returns | Notes |
|---|---|---|
| `GameState()` | new game, White to move | starting position |
| `.side_to_move` | `Color.WHITE` / `Color.BLACK` | |
| `.legal_moves()` | `List[(from_sq, to_sq)]` | fully legal, king safety included |
| `.apply_move(move)` | new `GameState` | **immutable** — does not mutate `self`, raises `ValueError` on illegal moves |
| `.is_in_check(color=None)` | `bool` | defaults to side to move |
| `.is_game_over()` | `bool` | |
| `.result()` | `+1` / `-1` / `0` / `None` | from the perspective of `.side_to_move`; `None` = not over |
| `.board` | `Board` | see below |

### `chaturanga/board.py` — `Board`
Raw piece placement, no rules attached. `Board.starting_position()`,
`.piece_at(sq)`, `.pieces_of(color)`, `.find_king_square(color)`.

### `chaturanga/encoding.py` — the two frozen numeric contracts
| Function | Signature | Contract section |
|---|---|---|
| `move_to_action(move)` | `(from,to) -> int` | 2.4, `from*64+to` |
| `action_to_move(action_id)` | `int -> (from,to)` | 2.4, inverse |
| `state_to_tensor(state)` | `GameState -> float32[13,8,8]` | 2.1 / 2.11 |
| `legal_action_mask(state)` | `GameState -> bool[4096]` | 2.6 — **use this**, don't reimplement |
| `NUM_ACTIONS` | `4096` | |

### `chaturanga/stub_network.py` — fake network, real contract
`zero_stub_network(tensor)` and `random_stub_network(tensor, rng=None)`,
both returning `(policy_logits: float32[4096], value: float in [-1,1])`.
For Student 3/4 to build and test against before Student 2's real
network exists. Swapping in the real network later should not require
changing any calling code — only which function gets passed in.

### `chaturanga/notation.py` — convenience only, not part of the contract
`square_to_algebraic(sq)`, `algebraic_to_square("e2")`,
`move_to_algebraic(move)`. Used by `play.py`; safe to ignore otherwise.

### `chaturanga/types.py`
`Color` (`WHITE=0, BLACK=1`), `PieceType` (`RAJA, MANTRI, RATHA, GAJA,
ASHVA, PADATI` — **this exact order fixes the plane order in
`state_to_tensor`**), `Piece`, `square(row,col)`.

## What's confirmed vs. what still needs a team yes/no

These four were left open by the Week 0 doc. The code picks a default
for each so Week 2 work isn't blocked, but Students 2/3 are building
against these defaults right now — worth a two-minute team confirmation
before Week 3 integration, since changing any of them later means
changing `encoding.py` (and possibly retraining).

| Decision | Current default | Where it's implemented |
|---|---|---|
| Board orientation | **absolute** (row 0 = White's back rank always, no flipping by side to move) | `encoding.py` |
| Piece plane order | `Raja, Mantri, Ratha, Gaja, Ashva, Padati`, White (0–5) then Black (6–11) | `types.py` `PieceType` enum order |
| Starting layout | symmetric, chess-like (`Ratha Ashva Gaja Raja Mantri Gaja Ashva Ratha`) | `board.py` `_BACK_RANK` |
| Terminal rules | checkmate = loss, stalemate = draw, 100-half-move no-progress cap, threefold repetition = draw | `game.py` `NO_PROGRESS_LIMIT`, `result()` |

## What to hand each teammate

**Student 2 (Network)** — `encoding.py` (`state_to_tensor`, `NUM_ACTIONS`)
for the exact I/O shape; `stub_network.py` as a reference contract to
match.

**Student 3 (MCTS)** — `game.py` for the whole game-tree interface;
`encoding.py` (`move_to_action`/`action_to_move`/`legal_action_mask`);
`stub_network.py` to start building search today; `integration_preview.py`
as a worked example of the loop they're embedding search into.

**Student 4 (Self-Play/Training)** — `game.py` and `encoding.py` (same
as Student 3) for assembling `(state, policy_target, outcome)` training
triples; `play.py --mode random --games N --quiet` as a ready-made
stress test for their self-play loop before real MCTS is plugged in.

**Everyone** — the full `tests/` suite as evidence the rules are
correct enough to build on: 36 tests including piece movement, pins,
double check, checkmate/stalemate, and the action/tensor encoding
contracts.
