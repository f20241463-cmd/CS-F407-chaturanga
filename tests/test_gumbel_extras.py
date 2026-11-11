"""Tests for the Gumbel play option, the Gumbel value scale and mirror averaging."""

import unittest
from functools import partial

import numpy as np
import torch

from chaturanga import encoding
from chaturanga.board import Board
from chaturanga.game import GameState
from chaturanga.stub_network import random_stub_network
from evaluation.parallel_match import mcts_spec, play_match_parallel
from evaluation.players import make_mcts_player, play_game, random_player
from mcts.gumbel import gumbel_search
from mcts.search import drive
from network.agent import MIRROR_ACTION, BatchNetwork, MirrorNetwork
from network.model import ChaturangaNet


def stub(seed=0):
    return partial(random_stub_network, rng=np.random.default_rng(seed))


def fixed_net(state_tensor):
    """The same (random-looking) answer every time for the same position."""
    seed = int(np.flatnonzero(state_tensor.ravel()).sum())
    return random_stub_network(state_tensor, rng=np.random.default_rng(seed))


def middlegame(seed=0, plies=30):
    rng = np.random.default_rng(seed)
    state = GameState()
    for _ in range(plies):
        moves = state.legal_moves()
        state = state.apply_move(moves[int(rng.integers(len(moves)))])
    return state


def mirrored(state):
    squares = state.board._squares
    return GameState(Board([squares[(sq // 8) * 8 + 7 - sq % 8] for sq in range(64)]),
                     state.side_to_move, state.halfmove_clock)  # fmt: skip


def gumbel(state, sims, net, rng, **kw):
    return drive(gumbel_search(state, sims, 1.5, encoding.state_to_tensor,
                               encoding.legal_action_mask, encoding.action_to_move, rng, **kw),
                 net)  # fmt: skip


class MirrorNetworkTest(unittest.TestCase):
    def test_answers_for_mirror_images_are_mirror_images(self):
        torch.manual_seed(0)
        net = MirrorNetwork(BatchNetwork(ChaturangaNet()))
        state = middlegame()
        a = encoding.state_to_tensor(state)
        b = encoding.state_to_tensor(mirrored(state))
        logits, values = net(np.stack([a, b]))
        np.testing.assert_allclose(logits[0], logits[1][MIRROR_ACTION], atol=1e-5)
        self.assertAlmostEqual(float(values[0]), float(values[1]), places=5)

    def test_mirror_action_is_its_own_inverse(self):
        np.testing.assert_array_equal(MIRROR_ACTION[MIRROR_ACTION], np.arange(4096))


class GumbelOptionsTest(unittest.TestCase):
    def test_without_noise_the_move_does_not_depend_on_the_rng(self):
        state = middlegame(1)
        moves = {gumbel(state, 32, stub(), np.random.default_rng(s), noise=False)[1] for s in range(4)}
        self.assertEqual(len(moves), 1)

    def test_scale_zero_gives_the_prior_back(self):
        state = middlegame(2)
        policy, _ = gumbel(state, 32, fixed_net, np.random.default_rng(0), c_scale=0.0)
        logits, _ = fixed_net(encoding.state_to_tensor(state))
        legal = encoding.legal_action_mask(state).astype(bool)
        prior = np.exp(logits[legal] - logits[legal].max())
        np.testing.assert_allclose(policy[legal], prior / prior.sum(), atol=1e-5)

    def test_gumbel_player_finishes_legal_games(self):
        player = make_mcts_player(stub(), simulations=16, gumbel=True)
        for seed in range(2):
            outcome, plies = play_game(player, random_player, np.random.default_rng(seed), max_plies=60)
            self.assertIn(outcome, ("white", "black", "draw", "unfinished"))
            self.assertGreater(plies, 0)

    def test_match_runs_with_gumbel_and_mirror_sides(self):
        a = mcts_spec("untrained", seed=1, simulations=8, gumbel=True, mirror=True)
        b = mcts_spec("untrained", seed=2, simulations=8)
        c = play_match_parallel(a, b, games=2, seed=0, max_plies=30)
        self.assertEqual(c["wins"] + c["losses"] + c["draws"] + c["unfinished"], 2)


class TrainOptionTest(unittest.TestCase):
    def test_gumbel_scale_reaches_the_self_play_settings(self):
        from training import train

        self.assertIn("gumbel_scale", train._SELF_PLAY_KEYS)
        args = train.parse_args(["--gumbel", "--gumbel_scale", "0.5"])
        self.assertEqual(args.gumbel_scale, 0.5)


if __name__ == "__main__":
    unittest.main()
