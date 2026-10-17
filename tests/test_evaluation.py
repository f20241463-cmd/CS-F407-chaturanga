"""
tests/test_evaluation.py

Checks for the benchmark scripts' network loading: --checkpoint loads
the saved weights, and an untrained network is pinned by its seed.
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from chaturanga import encoding
from chaturanga.game import GameState
from network.agent import make_network_fn
from network.checkpoint import save_checkpoint
from network.model import ChaturangaNet

from compare_checkpoints import score_summary
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
        score, _ = score_summary(new_wins=2, old_wins=0, draws=2)
        self.assertAlmostEqual(score, 0.75)

    def test_all_draws_is_even_with_no_spread(self):
        score, margin = score_summary(0, 0, 10)
        self.assertEqual((score, margin), (0.5, 0.0))


if __name__ == "__main__":
    unittest.main()
