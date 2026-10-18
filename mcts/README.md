# mcts: Search module (Student 3)

Guided Monte Carlo Tree Search with the PUCT formula: selection,
expansion, network evaluation of leaf nodes (no random rollouts), and
backpropagation. Given a state, it returns `visit_counts[4096]` and the
selected move, matching the MCTS output contract in the technical
interfaces document.

Status: complete, integrated with the real `chaturanga` environment and
the real `ChaturangaNet`, and used by self-play training
(`training/`) and the evaluation suite (`evaluation/`). Tested in
`tests/test_mcts.py` (13 tests, toy Nim game with a known correct
answer, no `chaturanga` dependency) and through self-play and benchmark
tests in `tests/test_training.py` and `tests/test_benchmark.py`.

## Quick start

```
python -m unittest discover -s tests -v     # full test suite
python mcts_preview.py                      # env -> MCTS -> move with the stub network
```

## Public API

### `mcts/node.py`: `MCTSNode`

| Method / attribute | Returns | Notes |
|---|---|---|
| `MCTSNode(state=None, parent=None, prior=0.0, action_taken=None, move=None)` | new node | the root gets `state`; children get `parent` + `move` |
| `.state` | `GameState` | children compute it lazily from the parent on first access, then cache it |
| `.value` | `float` | mean value, from this node's own side-to-move's perspective |
| `.expand(policy_probs, legal_actions, action_to_move_fn)` | `None` | creates one child per legal action, without calling `apply_move` |
| `.select_child(c_puct)` | `(action_id, MCTSNode)` | PUCT-best child |

### `mcts/search.py`: `run_mcts` and `select_action_from_visits`

```python
visit_counts, best_move = run_mcts(
    root_state,            # a GameState
    network_fn,            # tensor -> (policy_logits[4096], value)
    num_simulations,       # int, 30-50 in training per the project plan
    c_puct,                # float, exploration constant (1.5 used throughout)
    state_to_tensor_fn,    # encoding.state_to_tensor
    legal_action_mask_fn,  # encoding.legal_action_mask
    action_to_move_fn,     # encoding.action_to_move
    dirichlet_alpha=0.3,   # root noise concentration
    dirichlet_eps=0.0,     # root noise weight; 0 = off (evaluation), 0.25 in self-play
    rng=None,              # np.random.Generator, for reproducible noise
)

action_id = select_action_from_visits(visit_counts, temperature=0.0, rng=None)
```

- `visit_counts` is `float32[4096]`, the un-normalized policy target.
  Self-play divides it by its sum to get the training target.
- `best_move` is `(from_square, to_square)`, or `None` if the root
  state was already game-over.
- `select_action_from_visits` picks the most visited action when
  `temperature <= 1e-3`, and otherwise samples with probability
  proportional to `visits ** (1 / temperature)`. It never picks an
  unvisited action, and raises `ValueError` if nothing was visited.

## Design notes

- **Perspective convention.** `value` is always "how good for whoever's
  turn it is at this exact node", matching the network contract.
  `select_child` negates the child's value, and backpropagation flips
  the sign at every step up the tree. Don't remove either flip.
- **Lazy children.** Children store only the move that leads to them.
  Their `GameState` is built on first access of `.state`. Most children
  are never visited, and building every child's state up front took
  about 98% of search time. With lazy children the search output is
  identical (same visit counts, checked by md5 over several moves) and
  about 4x faster.
- **Root Dirichlet noise.** `P'(a) = (1 - eps) * P(a) + eps * Dir(alpha)`
  over the root's legal children, applied right after root expansion.
  Off by default (`dirichlet_eps=0.0`), so evaluation and play are
  deterministic given the network. Self-play turns it on with
  `eps=0.25, alpha=0.3`.
- **c_puct.** 1.5 is used everywhere. Higher values explore more, lower
  values trust the network's prior more.
- **Dependency direction.** `node.py` and `search.py` take
  `action_to_move_fn` / `state_to_tensor_fn` / `legal_action_mask_fn`
  as parameters instead of importing `chaturanga.encoding`, which is
  why `tests/test_mcts.py` can test the search on a toy game.
- **Repetition avoidance** lives outside this module
  (`training.self_play.filter_repeating_moves`). It works on the
  visit counts after the search and uses `GameState`'s repetition rule,
  which this environment-independent module doesn't know about.
