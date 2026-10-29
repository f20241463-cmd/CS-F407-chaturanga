"""
Batched self-play: many games side by side, one network call per step.
The games must be exactly the games play_self_play_game plays alone.
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
from mcts.search import drive, mcts_search, run_mcts
from network.agent import make_batch_network_fn, make_network_fn
from network.model import ChaturangaNet
from training.batched_self_play import play_games_batched
from training.self_play import play_self_play_game, self_play_game_steps


def hashed_network(tensor):
    """A deterministic stand-in network: outputs depend only on the position."""
    seed = int(np.frombuffer(np.ascontiguousarray(tensor).tobytes(), np.uint8).sum())
    rng = np.random.default_rng(seed)
    return rng.normal(size=4096).astype(np.float32), float(rng.uniform(-0.5, 0.5))


def hashed_batch(states):
    outputs = [hashed_network(s) for s in states]
    return np.stack([p for p, _ in outputs]), np.array([v for _, v in outputs])


class SplitHashedNetwork:
    """hashed_batch with submit/fetch, recording the size of every batch."""

    def __init__(self):
        self.sizes = []

    def submit(self, states):
        self.sizes.append(len(states))
        return hashed_batch(states)

    def fetch(self, handle):
        return handle


SETTINGS = dict(
    num_simulations=6, max_plies=24, temperature_moves=1000, avoid_repetition=True
)


def same_games(test, a, b):
    test.assertEqual(len(a), len(b))
    for (ex_a, info_a), (ex_b, info_b) in zip(a, b):
        test.assertEqual(info_a, info_b)
        test.assertEqual(len(ex_a), len(ex_b))
        for (t1, p1, v1), (t2, p2, v2) in zip(ex_a, ex_b):
            test.assertTrue(np.array_equal(t1, t2))
            test.assertTrue(np.array_equal(p1, p2))
            test.assertEqual(v1, v2)


class TestSearchGenerator(unittest.TestCase):
    def test_generator_search_matches_run_mcts(self):
        state = GameState()
        args = (encoding.state_to_tensor, encoding.legal_action_mask,
                encoding.action_to_move)  # fmt: skip
        for seed in range(3):
            v1, m1 = run_mcts(state, hashed_network, 20, 1.5, *args,
                              dirichlet_eps=0.25, rng=np.random.default_rng(seed))  # fmt: skip
            v2, m2 = drive(
                mcts_search(state, 20, 1.5, *args, dirichlet_eps=0.25,
                            rng=np.random.default_rng(seed)),  # fmt: skip
                hashed_network,
            )
            self.assertTrue(np.array_equal(v1, v2))
            self.assertEqual(m1, m2)


class TestPlayGamesBatched(unittest.TestCase):
    def _alone(self, seeds):
        return [
            play_self_play_game(hashed_network, rng=np.random.default_rng(s), **SETTINGS)
            for s in seeds
        ]

    def _together(self, seeds, network, groups=1):
        games = [
            self_play_game_steps(rng=np.random.default_rng(s), **SETTINGS)
            for s in seeds
        ]
        return play_games_batched(games, network, groups=groups)

    def test_batched_games_are_the_games_played_alone(self):
        seeds = [0, 1, 2, 3, 4]
        same_games(self, self._together(seeds, hashed_batch), self._alone(seeds))

    def test_groups_take_turns_and_still_play_the_same_games(self):
        seeds = [5, 6, 7, 8, 9]
        network = SplitHashedNetwork()
        same_games(self, self._together(seeds, network, groups=2), self._alone(seeds))
        # Two groups of 3 and 2 games: no batch holds more than one group.
        self.assertLessEqual(max(network.sizes), 3)

    def test_games_of_different_lengths_and_on_done(self):
        lengths = [3, 9, 1]
        done = []
        games = [
            self_play_game_steps(num_simulations=2, max_plies=n,
                                 rng=np.random.default_rng(n))  # fmt: skip
            for n in lengths
        ]
        results = play_games_batched(
            games, hashed_batch, on_done=lambda i, r: done.append(i)
        )
        self.assertEqual([info["plies"] for _, info in results], lengths)
        self.assertEqual(sorted(done), [0, 1, 2])
        self.assertEqual(done[0], 2)  # the 1-ply game finishes first


class TestBatchNetwork(unittest.TestCase):
    def test_matches_the_single_position_network(self):
        torch.manual_seed(0)
        model = ChaturangaNet().eval()
        single = make_network_fn(model)
        states = np.stack(
            [encoding.state_to_tensor(GameState())]
            + [np.random.default_rng(i).random((13, 8, 8), np.float32) for i in range(4)]
        )
        for cpu_below in (0, 10):
            batch = make_batch_network_fn(model, cpu_below=cpu_below)
            logits, values = batch(states)
            self.assertEqual(logits.shape, (5, 4096))
            self.assertEqual(values.shape, (5,))
            for row, state in enumerate(states):
                p, v = single(state)
                np.testing.assert_allclose(logits[row], p, atol=1e-4)
                self.assertAlmostEqual(float(values[row]), v, places=4)
            # submit/fetch gives the same as one call.
            p2, v2 = batch.fetch(batch.submit(states))
            np.testing.assert_array_equal(p2, logits)
            np.testing.assert_array_equal(v2, values)

    def test_rejects_wrong_shape(self):
        batch = make_batch_network_fn(ChaturangaNet())
        with self.assertRaises(ValueError):
            batch(np.zeros((13, 8, 8), np.float32))


class TestTrainWithBatchedSelfPlay(unittest.TestCase):
    def _run(self, tmp, name, *extra):
        from training import train

        ckpt_dir = os.path.join(tmp, name)
        log = os.path.join(tmp, name + ".csv")
        with mock.patch("builtins.print"):
            train.main(
                [
                    "--iterations", "2", "--games_per_iter", "3",
                    "--simulations", "3", "--max_plies", "8",
                    "--min_buffer", "1", "--train_steps", "2", "--batch_size", "4",
                    "--checkpoint_dir", ckpt_dir, "--log", log, *extra,
                ]
            )  # fmt: skip
        with open(log) as f:
            return list(csv.DictReader(f))

    def test_batched_self_play_plays_the_same_games_as_parallel(self):
        keys = ["white_wins", "black_wins", "draws", "unfinished", "avg_plies",
                "buffer_size"]  # fmt: skip
        with tempfile.TemporaryDirectory() as tmp:
            parallel = self._run(tmp, "parallel", "--workers", "2")
            batched = self._run(tmp, "batched", "--workers", "2",
                                "--selfplay_device", "cpu")  # fmt: skip
            here = self._run(tmp, "here", "--selfplay_device", "cpu")
        expected = [[r[k] for k in keys] for r in parallel]
        self.assertEqual([[r[k] for k in keys] for r in batched], expected)
        self.assertEqual([[r[k] for k in keys] for r in here], expected)
        self.assertEqual([r["iteration"] for r in batched], ["1", "2"])


class TestBatchedMatch(unittest.TestCase):
    def test_batched_match_plays_the_same_games(self):
        from evaluation.parallel_match import make_pool, mcts_spec, play_match_parallel

        agent = mcts_spec("untrained", seed=0, simulations=3)
        other = mcts_spec("untrained", seed=1, simulations=4)
        for opponent in ({"kind": "random"}, other):
            plain = play_match_parallel(agent, opponent, 6, seed=5, max_plies=30)
            batched = play_match_parallel(
                agent, opponent, 6, seed=5, max_plies=30, device="cpu"
            )
            self.assertEqual(plain, batched)
        pool = make_pool(2)
        try:
            batched = play_match_parallel(
                agent, other, 6, seed=5, max_plies=30, pool=pool, device="cpu"
            )
        finally:
            pool.close()
            pool.join()
        self.assertEqual(plain, batched)

    def test_mcts_player_is_its_generator_driven_one_position_at_a_time(self):
        from evaluation.players import make_mcts_player

        player = make_mcts_player(hashed_network, simulations=8)
        state = GameState()
        move = player(state, np.random.default_rng(0))
        again = drive(player.steps(state, np.random.default_rng(0)), hashed_network)
        self.assertEqual(move, again)


class TestResignation(unittest.TestCase):
    @staticmethod
    def gloomy_for(side):
        """A network that rates every position as lost for `side` (0 White, 1 Black)."""

        def net(tensor):
            mover = int(tensor[12, 0, 0])  # the turn plane: 0 White, 1 Black
            return np.zeros(4096, np.float32), (-0.99 if mover == side else 0.99)

        return net

    def test_side_that_rates_itself_lost_resigns(self):
        trace = []
        examples, info = play_self_play_game(
            self.gloomy_for(1), num_simulations=4, max_plies=50,
            rng=np.random.default_rng(0), resign_threshold=0.9, resign_after=2,
            value_trace=trace,
        )  # fmt: skip
        self.assertTrue(info["resigned"])
        self.assertEqual(info["outcome"], "white")
        self.assertEqual(info["would_resign"], "black")
        self.assertLess(info["plies"], 50)
        # Every position gets the real result: +1 for White, -1 for Black.
        for tensor, _, value in examples:
            self.assertEqual(value, 1.0 if tensor[12, 0, 0] == 0 else -1.0)
        self.assertEqual(len(trace), len(examples))

    def test_check_games_play_on_and_record_who_would_resign(self):
        _, info = play_self_play_game(
            self.gloomy_for(0), num_simulations=4, max_plies=12,
            rng=np.random.default_rng(0), resign_threshold=0.9,
            resign_disabled=True,
        )  # fmt: skip
        self.assertFalse(info["resigned"])
        self.assertEqual(info["would_resign"], "white")
        self.assertEqual(info["plies"], 12)

    def test_no_resignation_keys_when_off(self):
        _, info = play_self_play_game(
            self.gloomy_for(0), num_simulations=4, max_plies=6,
            rng=np.random.default_rng(0),
        )  # fmt: skip
        self.assertEqual(info, {"plies": 6, "outcome": "unfinished"})

    def test_resign_counts(self):
        from training.train import resign_counts

        infos = [
            {"outcome": "white", "resigned": True, "would_resign": "black"},
            {"outcome": "black", "resigned": False, "would_resign": "white"},
            {"outcome": "draw", "resigned": False, "would_resign": "black"},
            {"outcome": "draw", "resigned": False, "would_resign": None},
        ]
        self.assertEqual(
            resign_counts(infos), {"resigned": 1, "resign_checks": 2, "resign_wrong": 1}
        )
        self.assertEqual(resign_counts([{"outcome": "draw"}])["resigned"], "")

    def test_train_with_resignation_and_an_old_log(self):
        from training import train

        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "log.csv")
            with open(log, "w") as f:  # a log written before the resign columns
                f.write(",".join(train.LOG_FIELDS[:-3]) + "\n")
            with mock.patch("builtins.print"):
                train.main(
                    [
                        "--iterations", "1", "--games_per_iter", "2",
                        "--simulations", "3", "--max_plies", "8", "--min_buffer", "1",
                        "--train_steps", "1", "--batch_size", "4",
                        "--resign_threshold", "0.5", "--selfplay_device", "cpu",
                        "--checkpoint_dir", os.path.join(tmp, "c"), "--log", log,
                    ]
                )  # fmt: skip
            with open(log) as f:
                rows = list(csv.reader(f))
        self.assertEqual(len(rows[1]), len(rows[0]))


class TestResignCheck(unittest.TestCase):
    def test_summary_counts_right_and_wrong_resignations(self):
        from evaluation.resign_check import summarize, would_resign

        lost = [("white", 0.0), ("black", -0.95), ("white", 0.1), ("black", -0.97)]
        self.assertEqual(would_resign(lost, 0.9, 2), ("black", 3))
        self.assertEqual(would_resign(lost, 0.9, 3), (None, None))
        games = [("white", lost), ("draw", lost), ("draw", [("white", 0.0)])]
        self.assertEqual(summarize(games, [0.9], [2]), [(0.9, 2, 2, 1)])


if __name__ == "__main__":
    unittest.main()
