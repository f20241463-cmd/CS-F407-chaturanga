"""
tests/test_evaluation.py

Checks for the benchmark scripts' network loading: --checkpoint loads
the saved weights, and an untrained network is pinned by its seed.
"""

import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from chaturanga import encoding
from chaturanga.game import GameState
from network.agent import make_network_fn
from network.checkpoint import save_checkpoint
from network.model import ChaturangaNet

from compare_checkpoints import filter_repeating_moves, score_summary
from evaluate_vs_random import make_real_network_fn


def _start_tensor():
    return encoding.state_to_tensor(GameState())


class TestNetworkLoading(unittest.TestCase):
    def test_same_seed_gives_same_untrained_network(self):
        a, va = make_real_network_fn(seed=3)(_start_tensor())
        b, vb = make_real_network_fn(seed=3)(_start_tensor())
        np.testing.assert_array_equal(a, b)
        self.assertEqual(va, vb)

    def test_different_seeds_give_different_networks(self):
        a, _ = make_real_network_fn(seed=3)(_start_tensor())
        b, _ = make_real_network_fn(seed=4)(_start_tensor())
        self.assertFalse(np.array_equal(a, b))

    def test_checkpoint_loads_saved_weights(self):
        torch.manual_seed(7)
        net = ChaturangaNet()
        expected, expected_value = make_network_fn(net)(_start_tensor())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "iter_001.pt")
            save_checkpoint(path, net, step=1)
            got, got_value = make_real_network_fn(checkpoint=path)(_start_tensor())
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(got_value, expected_value, places=5)


class TestScoreSummary(unittest.TestCase):
    def test_draws_count_half(self):
        score, _ = score_summary(a_wins=2, b_wins=0, draws=2)
        self.assertAlmostEqual(score, 0.75)

    def test_all_draws_is_even_with_no_spread(self):
        score, margin = score_summary(0, 0, 10)
        self.assertEqual((score, margin), (0.5, 0.0))


def _one_move_from_threefold():
    """Knights shuffle out and back; Black's g8 knight returning to g8
    (square 45 -> 62) would now repeat the start position a third time."""
    state = GameState()
    for move in [(6, 21), (62, 45), (21, 6), (45, 62), (6, 21), (62, 45), (21, 6)]:
        state = state.apply_move(move)
    return state


class TestRepetitionAvoidance(unittest.TestCase):
    def test_repeating_move_is_removed_when_alternatives_exist(self):
        state = _one_move_from_threefold()
        repeat = encoding.move_to_action((45, 62))
        other = encoding.move_to_action((45, 28))
        visits = np.zeros(4096, dtype=np.float32)
        visits[[repeat, other]] = [10, 3]
        filtered, avoided = filter_repeating_moves(state, visits)
        self.assertTrue(avoided)
        self.assertEqual(filtered[repeat], 0)
        self.assertEqual(filtered[other], 3)
        self.assertEqual(visits[repeat], 10)  # input left untouched

    def test_repeating_move_kept_when_it_is_the_only_visited_move(self):
        state = _one_move_from_threefold()
        repeat = encoding.move_to_action((45, 62))
        visits = np.zeros(4096, dtype=np.float32)
        visits[repeat] = 10
        filtered, avoided = filter_repeating_moves(state, visits)
        self.assertFalse(avoided)
        np.testing.assert_array_equal(filtered, visits)


class TestPlayVsAiLog(unittest.TestCase):
    def _play(self, inputs, log):
        import play_vs_ai

        argv = ["play_vs_ai.py", "--simulations", "4", "--player", "t", "--log", log]
        with mock.patch("sys.argv", argv), mock.patch(
            "builtins.input", side_effect=inputs
        ), mock.patch("builtins.print"):
            play_vs_ai.main()

    def test_resign_and_quit_are_recorded(self):
        from evaluation.human_log import summarize

        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "human.csv")
            self._play(["resign"], log)
            self._play(["e2e3", "quit"], log)
            s = summarize(log)
            self.assertEqual((s["games"], s["agent_win"], s["unfinished"]), (2, 1, 1))
            with open(log) as f:
                last = list(csv.DictReader(f))[-1]
            self.assertEqual((last["ending"], last["plies"]), ("quit", "2"))
            self.assertTrue(last["moves"].startswith("e2e3 "))


if __name__ == "__main__":
    unittest.main()
