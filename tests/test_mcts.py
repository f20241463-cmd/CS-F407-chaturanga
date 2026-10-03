"""
tests/test_mcts.py

These tests deliberately do NOT import anything from the `chaturanga`
package. They test the MCTS logic in isolation using a tiny toy game
(a Nim variant with a known correct answer), so you can verify
selection/expansion/backprop/PUCT are correct before ever touching the
real environment. Once you're confident this passes, swap in the real
GameState/encoding/stub_network from the chaturanga package -- see
mcts_preview.py for that wiring.

Run with:
    python -m unittest discover -s tests -v
"""

import math
import unittest

import numpy as np

from mcts.node import MCTSNode
from mcts.search import run_mcts, _softmax_masked


# ---------------------------------------------------------------------
# Toy game: take-1,2,or-3 Nim. Player to move picks k in {1,2,3} with
# k <= n stones remaining; n -= k. A player with n == 0 stones and the
# move has no legal move and loses (so the player who took the last
# stone wins). Theory: n % 4 == 0 is a LOSING position for the player
# to move; any other n has a winning move (reduce n to a multiple of 4).
# This gives every test an unambiguous correct answer to check against.
# ---------------------------------------------------------------------
class NimState:
    def __init__(self, n):
        self.n = n

    def legal_moves(self):
        return [k for k in (1, 2, 3) if k <= self.n]

    def apply_move(self, move):
        return NimState(self.n - move)

    def is_game_over(self):
        return self.n == 0

    def result(self):
        return -1.0 if self.n == 0 else None


def nim_state_to_tensor(state):
    # MCTS never inspects the tensor's contents directly, only passes
    # it straight to network_fn, so any placeholder is fine here.
    return state.n


def nim_legal_action_mask(state):
    mask = [False] * 4096
    for k in state.legal_moves():
        mask[k] = True
    return mask


def nim_action_to_move(action_id):
    return action_id  # moves ARE action ids in this toy game


def neutral_network_fn(tensor):
    """Uniform policy, value 0 -- deliberately uninformative, so a
    passing test proves the SEARCH mechanics (not a lucky prior) found
    the right answer by exploring terminal outcomes."""
    return np.zeros(4096, dtype=np.float32), 0.0


class TestSoftmaxMasked(unittest.TestCase):
    def test_zeros_out_illegal_entries(self):
        logits = np.zeros(4096, dtype=np.float32)
        mask = [False] * 4096
        mask[1] = True
        mask[2] = True
        probs = _softmax_masked(logits, mask)
        self.assertAlmostEqual(probs.sum(), 1.0, places=6)
        self.assertEqual(probs[0], 0.0)
        self.assertGreater(probs[1], 0.0)
        self.assertGreater(probs[2], 0.0)

    def test_single_legal_move_gets_all_probability(self):
        logits = np.random.randn(4096).astype(np.float32)
        mask = [False] * 4096
        mask[7] = True
        probs = _softmax_masked(logits, mask)
        self.assertAlmostEqual(probs[7], 1.0, places=6)


class TestMCTSNode(unittest.TestCase):
    def test_value_is_zero_before_any_visits(self):
        node = MCTSNode(state=NimState(4))
        self.assertEqual(node.value, 0.0)

    def test_expand_creates_one_child_per_legal_action(self):
        root = MCTSNode(state=NimState(4))
        policy = np.ones(4096, dtype=np.float32) / 3
        root.expand(policy, [1, 2, 3], nim_action_to_move)
        self.assertEqual(set(root.children.keys()), {1, 2, 3})
        self.assertTrue(root.is_expanded)
        self.assertEqual(root.children[1].state.n, 3)

    def test_select_child_prefers_unvisited_over_already_explored(self):
        root = MCTSNode(state=NimState(4))
        policy = np.full(4096, 0.5, dtype=np.float32)
        root.expand(policy, [1, 2], nim_action_to_move)
        root.visit_count = 1
        root.children[1].visit_count = 1
        root.children[1].value_sum = 1.0  # looks great for the opponent
        # child 2 is unvisited -- its exploration bonus should still let
        # it win against an already-explored, opponent-favoring child.
        action, _ = root.select_child(c_puct=2.0)
        self.assertEqual(action, 2)


class TestRunMCTS(unittest.TestCase):
    def test_finds_correct_move_from_a_winning_position(self):
        # n=5 is NOT a multiple of 4, so a winning move exists: take 1
        # stone, leaving n=4 (a losing position for the opponent).
        state = NimState(5)
        visit_counts, best_move = run_mcts(
            root_state=state,
            network_fn=neutral_network_fn,
            num_simulations=500,
            c_puct=1.4,
            state_to_tensor_fn=nim_state_to_tensor,
            legal_action_mask_fn=nim_legal_action_mask,
            action_to_move_fn=nim_action_to_move,
        )
        self.assertEqual(best_move, 1)

    def test_root_children_visit_counts_sum_to_num_simulations(self):
        state = NimState(6)
        num_sims = 200
        visit_counts, _ = run_mcts(
            root_state=state,
            network_fn=neutral_network_fn,
            num_simulations=num_sims,
            c_puct=1.4,
            state_to_tensor_fn=nim_state_to_tensor,
            legal_action_mask_fn=nim_legal_action_mask,
            action_to_move_fn=nim_action_to_move,
        )
        self.assertEqual(int(visit_counts.sum()), num_sims)

    def test_handles_already_terminal_root_gracefully(self):
        state = NimState(0)
        visit_counts, best_move = run_mcts(
            root_state=state,
            network_fn=neutral_network_fn,
            num_simulations=50,
            c_puct=1.4,
            state_to_tensor_fn=nim_state_to_tensor,
            legal_action_mask_fn=nim_legal_action_mask,
            action_to_move_fn=nim_action_to_move,
        )
        self.assertIsNone(best_move)
        self.assertEqual(visit_counts.sum(), 0)


if __name__ == "__main__":
    unittest.main()
