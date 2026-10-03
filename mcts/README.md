# mcts — Search module (Student 3)

Guided Monte Carlo Tree Search, implementing selection, expansion,
evaluation and backpropagation with the PUCT formula, matching the
MCTS output contract from section 2.7 of the execution-plan document:
given a state, returns `visit_counts[4096]` and a selected move.

Status: tree mechanics complete and unit-tested against a toy game
independent of the real environment (8/8 tests passing). Not yet
integration-tested against the real `chaturanga` package — do that
first, before anything else.

## Quick start

```
python -m unittest discover -s tests -v     # run the MCTS-only test suite
python mcts_preview.py                       # env -> MCTS -> move, once chaturanga/ is available
```

## Public API

### `mcts/node.py` — `MCTSNode`

| Method / attribute | Returns | Notes |
|---|---|---|
| `MCTSNode(state, parent=None, prior=0.0, action_taken=None)` | new node | wraps one `GameState` |
| `.value` | `float` | mean value, from this node's own side-to-move's perspective |
| `.expand(policy_probs, legal_actions, action_to_move_fn)` | `None` | creates one child per legal action |
| `.select_child(c_puct)` | `(action_id, MCTSNode)` | PUCT-best child |

### `mcts/search.py` — `run_mcts`

```python
visit_counts, best_move = run_mcts(
    root_state,          # a GameState
    network_fn,          # tensor -> (policy_logits[4096], value)
    num_simulations,      # int, 30-50 per the execution plan
    c_puct,               # float, exploration constant (try 1.4-2.0)
    state_to_tensor_fn,    # encoding.state_to_tensor
    legal_action_mask_fn,  # encoding.legal_action_mask
    action_to_move_fn,     # encoding.action_to_move
)
```

`visit_counts` is `float32[4096]` — this is the un-normalized policy
target. Student 4 divides by its sum before storing it in a training
example, per section 2.8 of the execution plan.

`best_move` is `(from_square, to_square)`, or `None` if the root state
was already game-over.

## Integration steps, once you have the real repo

1. Copy this `mcts/` folder and `mcts_preview.py` into the repo root,
   alongside `chaturanga/`, `tests/`, and `play.py`.
2. Copy `tests/test_mcts.py` into the existing `tests/` folder (it
   won't collide with Student 1's test files — different test classes,
   no shared imports).
3. Run `python -m unittest discover -s tests -v` — you should see her
   36 tests plus these 8 all pass together.
4. Run `python mcts_preview.py` to confirm MCTS actually runs against
   the real `GameState`, `encoding`, and `stub_network`.
5. Once it runs cleanly, you're ready for Week 3 integration with
   Student 2's real network — nothing in this file needs to change,
   only which `network_fn` gets passed in.

## Design notes worth knowing before you extend this

- **Perspective convention.** `value` is always "how good for whoever's
  turn it is at this exact node", matching the network contract
  (section 2.5). Backpropagation flips the sign at every step up the
  tree for this reason — don't remove that flip when refactoring.
- **c_puct.** Not fixed by the execution plan. 1.4-2.0 is a reasonable
  starting range; higher values explore more, lower values trust the
  network's prior more. Worth a short experiment once the real network
  exists rather than guessing.
- **Dirichlet noise at the root.** Not implemented here. Real AlphaZero
  adds Dirichlet noise to the root's policy during self-play (not
  during evaluation matches) to guarantee some exploration even when
  the network is confident. This matters once Student 4 starts
  generating self-play games, not before — flag it to them rather than
  building it preemptively into this file.
- **Dependency direction.** `node.py` and `search.py` take
  `action_to_move_fn` / `state_to_tensor_fn` / `legal_action_mask_fn`
  as parameters rather than importing `chaturanga.encoding` directly.
  That's deliberate — it's what let this whole module be written and
  tested before the real repo was even cloned, and it's also why the
  toy-game tests in `tests/test_mcts.py` don't need the `chaturanga`
  package installed to run.