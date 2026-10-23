"""
tests/test_training.py

Tests for the self-play / training pipeline: replay buffer, value
targets, move selection from visit counts, root Dirichlet noise, lazy
MCTS children, and a short self-play game.

Everything except TestTrainStep runs without torch (it is skipped when
torch is not installed).

Run with:
    python -m unittest discover -s tests -v
"""

import unittest
from unittest import mock

import numpy as np

from chaturanga import encoding
from chaturanga.board import Board
from chaturanga.game import GameState
from chaturanga.stub_network import zero_stub_network
from chaturanga.types import Color, Piece, PieceType, square
from mcts.node import MCTSNode
from mcts.search import _add_dirichlet_noise, run_mcts, select_action_from_visits
from training import self_play
from training.replay_buffer import ReplayBuffer
from training.self_play import (
    filter_repeating_moves,
    material_balance,
    material_values,
    outcome_values,
    play_self_play_game,
)

try:
    from tests.test_mcts import (
        NimState,
        neutral_network_fn,
        nim_action_to_move,
        nim_legal_action_mask,
        nim_state_to_tensor,
    )
except ImportError:  # run via `discover -s tests`, where tests/ itself is on sys.path
    from test_mcts import (
        NimState,
        neutral_network_fn,
        nim_action_to_move,
        nim_legal_action_mask,
        nim_state_to_tensor,
    )

try:
    import torch
except ImportError:
    torch = None


def _example(fill, actions, value):
    tensor = np.full((13, 8, 8), fill, dtype=np.float32)
    policy = np.zeros(4096, dtype=np.float32)
    policy[actions] = 1.0 / len(actions)
    return tensor, policy, value


class TestReplayBuffer(unittest.TestCase):
    def test_sample_shapes_and_dense_policy_roundtrip(self):
        buf = ReplayBuffer(capacity=10, seed=0)
        tensor, policy, value = _example(0.5, [3, 700, 4095], -1.0)
        buf.add(tensor, policy, value)
        self.assertEqual(len(buf), 1)

        # One example, batch of 4: sampled with replacement.
        states, policies, values = buf.sample(4)
        self.assertEqual(states.shape, (4, 13, 8, 8))
        self.assertEqual(policies.shape, (4, 4096))
        self.assertEqual(values.shape, (4,))
        for arr in (states, policies, values):
            self.assertEqual(arr.dtype, np.float32)
        for row in range(4):
            np.testing.assert_array_equal(states[row], tensor)
            np.testing.assert_array_equal(policies[row], policy)
            self.assertEqual(values[row], -1.0)

    def test_samples_without_replacement_when_large_enough(self):
        buf = ReplayBuffer(capacity=10, seed=0)
        buf.add_game([_example(float(i), [i], 0.0) for i in range(5)])
        states, _, _ = buf.sample(5)
        self.assertEqual(sorted(states[:, 0, 0, 0].tolist()), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_oldest_examples_are_evicted(self):
        buf = ReplayBuffer(capacity=3, seed=0)
        buf.add_game([_example(float(i), [i], 0.0) for i in range(5)])
        self.assertEqual(len(buf), 3)
        states, _, _ = buf.sample(3)
        self.assertEqual(sorted(states[:, 0, 0, 0].tolist()), [2.0, 3.0, 4.0])

    def test_sampling_empty_buffer_raises(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=3).sample(1)


class TestOutcomeValues(unittest.TestCase):
    def test_loss_for_final_side_to_move(self):
        # Black is to move at the end and has lost (result -1 for Black).
        movers = [Color.WHITE, Color.BLACK, Color.WHITE, Color.BLACK]
        self.assertEqual(
            outcome_values(movers, -1, Color.BLACK), [1.0, -1.0, 1.0, -1.0]
        )

    def test_draw_is_zero_everywhere(self):
        movers = [Color.WHITE, Color.BLACK, Color.WHITE]
        self.assertEqual(outcome_values(movers, 0, Color.BLACK), [0.0, 0.0, 0.0])


class TestSelectActionFromVisits(unittest.TestCase):
    def test_greedy_picks_most_visited(self):
        visits = np.zeros(4096, dtype=np.float32)
        visits[[5, 9, 100]] = [3, 10, 2]
        self.assertEqual(select_action_from_visits(visits, temperature=0.0), 9)

    def test_sampling_only_picks_visited_actions(self):
        visits = np.zeros(4096, dtype=np.float32)
        visits[[5, 9, 100]] = [3, 10, 2]
        rng = np.random.default_rng(0)
        picks = {select_action_from_visits(visits, 1.0, rng) for _ in range(300)}
        self.assertEqual(picks, {5, 9, 100})

    def test_zero_visits_raises(self):
        with self.assertRaises(ValueError):
            select_action_from_visits(np.zeros(4096, dtype=np.float32), 1.0)


class TestRootNoise(unittest.TestCase):
    def test_noise_keeps_priors_summing_to_one_but_changes_them(self):
        root = MCTSNode(NimState(5))
        probs = np.zeros(4096)
        probs[[1, 2, 3]] = 1 / 3
        root.expand(probs, [1, 2, 3], nim_action_to_move)
        _add_dirichlet_noise(root, 0.3, 0.25, np.random.default_rng(0))
        priors = [child.prior for child in root.children.values()]
        self.assertAlmostEqual(sum(priors), 1.0, places=6)
        self.assertFalse(np.allclose(priors, 1 / 3))

    def test_noisy_search_still_finds_winning_nim_move(self):
        # From 5 stones the only winning move is to take 1 (leaving 4).
        _, best = run_mcts(
            NimState(5),
            neutral_network_fn,
            500,
            1.4,
            nim_state_to_tensor,
            nim_legal_action_mask,
            nim_action_to_move,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25,
            rng=np.random.default_rng(3),
        )
        self.assertEqual(best, 1)


class TestLazyChildren(unittest.TestCase):
    def test_expand_does_not_apply_moves_until_state_is_read(self):
        calls = []

        class CountingNim(NimState):
            def apply_move(self, move):
                calls.append(move)
                return CountingNim(self.n - move)

        root = MCTSNode(CountingNim(5))
        probs = np.zeros(4096)
        probs[[1, 2, 3]] = 1 / 3
        root.expand(probs, [1, 2, 3], nim_action_to_move)
        self.assertEqual(len(root.children), 3)
        self.assertEqual(calls, [])

        child = root.children[2]
        self.assertEqual(child.state.n, 3)
        self.assertEqual(child.state.n, 3)  # cached: second access is free
        self.assertEqual(calls, [2])


class TestSelfPlay(unittest.TestCase):
    def _game(self, **kwargs):
        return play_self_play_game(
            zero_stub_network,
            num_simulations=4,
            max_plies=10,
            rng=np.random.default_rng(0),
            **kwargs,
        )

    def test_short_game_returns_well_formed_unfinished_examples(self):
        examples, info = self._game()
        self.assertEqual(info, {"plies": 10, "outcome": "unfinished"})
        self.assertEqual(len(examples), 10)
        for tensor, policy, value in examples:
            self.assertEqual(tensor.shape, (13, 8, 8))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertEqual(policy.shape, (4096,))
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
            self.assertEqual(value, 0.0)
        # The turn plane alternates White (0) / Black (1).
        turns = [float(t[12, 0, 0]) for t, _, _ in examples]
        self.assertEqual(turns, [0.0, 1.0] * 5)

    def test_draw_value_applies_to_unfinished_games(self):
        examples, info = self._game(draw_value=-0.25)
        self.assertEqual(info["outcome"], "unfinished")
        self.assertEqual({value for _, _, value in examples}, {-0.25})

    def test_policy_target_is_zero_on_illegal_moves(self):
        roots = []

        def recording_run_mcts(root_state, *args, **kwargs):
            roots.append(root_state)
            return run_mcts(root_state, *args, **kwargs)

        with mock.patch.object(self_play, "run_mcts", side_effect=recording_run_mcts):
            examples, _ = self._game()

        self.assertEqual(len(roots), len(examples))
        for state, (_, policy, _) in zip(roots, examples):
            legal = encoding.legal_action_mask(state)
            self.assertTrue(np.all(policy[~legal] == 0))
            self.assertTrue(np.any(policy[legal] > 0))
        self.assertIsInstance(roots[0], GameState)


class TestMaterialAdjudication(unittest.TestCase):
    def _white_up_a_rook(self):
        state = GameState(board=Board(), side_to_move=Color.BLACK)
        state.board.set_piece(square(0, 4), Piece(PieceType.RAJA, Color.WHITE))
        state.board.set_piece(square(7, 4), Piece(PieceType.RAJA, Color.BLACK))
        state.board.set_piece(square(0, 0), Piece(PieceType.RATHA, Color.WHITE))
        return state

    def test_material_balance(self):
        self.assertEqual(material_balance(GameState(), Color.WHITE), 0.0)
        state = self._white_up_a_rook()
        self.assertEqual(material_balance(state, Color.WHITE), 5.0)
        self.assertEqual(material_balance(state, Color.BLACK), -5.0)

    def test_material_values_are_from_each_movers_perspective(self):
        values = material_values(
            [Color.WHITE, Color.BLACK], self._white_up_a_rook(), material_scale=5.0
        )
        self.assertAlmostEqual(values[0], float(np.tanh(1.0)))
        self.assertAlmostEqual(values[1], -float(np.tanh(1.0)))

    def test_self_play_uses_material_for_unfinished_games_when_enabled(self):
        with mock.patch.object(self_play, "material_balance", return_value=3.0):
            examples, info = play_self_play_game(
                zero_stub_network,
                num_simulations=4,
                max_plies=10,
                rng=np.random.default_rng(0),
                material_scale=5.0,
            )
        self.assertEqual(info["outcome"], "unfinished")
        expected = float(np.tanh(0.6))
        values = [value for _, _, value in examples]
        for i, value in enumerate(values):
            # Ply 0 is White to move, ply 1 Black, and so on.
            self.assertAlmostEqual(value, expected if i % 2 == 0 else -expected)


class TestSelfPlayRepetitionAvoidance(unittest.TestCase):
    def _one_move_from_threefold(self):
        # Knights shuffle out and back; Black's knight returning f6 -> g8
        # (45 -> 62) would repeat the start position a third time.
        state = GameState()
        for move in [(6, 21), (62, 45), (21, 6), (45, 62), (6, 21), (62, 45), (21, 6)]:
            state = state.apply_move(move)
        return state

    def test_filter_removes_repeating_move(self):
        state = self._one_move_from_threefold()
        repeat = encoding.move_to_action((45, 62))
        other = encoding.move_to_action((45, 28))
        visits = np.zeros(4096, dtype=np.float32)
        visits[[repeat, other]] = [10, 3]
        filtered, avoided = filter_repeating_moves(state, visits)
        self.assertTrue(avoided)
        self.assertEqual((filtered[repeat], filtered[other]), (0, 3))

    def test_self_play_avoids_repetition_draws_when_enabled(self):
        # A network that only wants to shuffle the g-file knights out and
        # back draws by threefold repetition after 8 plies; with avoidance
        # the game continues and no policy target includes the repeating move.
        shuffle = [
            encoding.move_to_action(m) for m in [(6, 21), (21, 6), (62, 45), (45, 62)]
        ]

        def knight_shuffler(tensor):
            logits = np.zeros(4096, dtype=np.float32)
            logits[shuffle] = 20.0
            return logits, 0.0

        def game(avoid):
            roots = []

            def recording_run_mcts(root_state, *args, **kwargs):
                roots.append(root_state)
                return run_mcts(root_state, *args, **kwargs)

            with mock.patch.object(
                self_play, "run_mcts", side_effect=recording_run_mcts
            ):
                examples, info = play_self_play_game(
                    knight_shuffler,
                    num_simulations=4,
                    max_plies=20,
                    temperature_moves=0,
                    dirichlet_eps=0.0,
                    rng=np.random.default_rng(0),
                    avoid_repetition=avoid,
                )
            return roots, examples, info

        _, _, info = game(avoid=False)
        self.assertEqual(info, {"plies": 8, "outcome": "draw"})

        roots, examples, info = game(avoid=True)
        self.assertEqual(info["outcome"], "unfinished")
        for state, (_, policy, _) in zip(roots, examples):
            for action in np.flatnonzero(policy):
                child = state.apply_move(encoding.action_to_move(int(action)))
                self.assertFalse(child._is_threefold_repetition())


@unittest.skipIf(torch is None, "torch not installed")
class TestTrainStep(unittest.TestCase):
    def test_train_step_reduces_loss_on_a_fixed_batch(self):
        from network.loss import AlphaZeroLoss
        from network.model import ChaturangaNet
        from training.train import train_step

        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        model = ChaturangaNet(num_res_blocks=1, num_filters=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = AlphaZeroLoss()

        batch_size = 8
        states = rng.integers(0, 2, size=(batch_size, 13, 8, 8)).astype(np.float32)
        policies = np.zeros((batch_size, 4096), dtype=np.float32)
        for row in range(batch_size):
            actions = rng.choice(4096, size=3, replace=False)
            policies[row, actions] = 1 / 3
        values = rng.choice([-1.0, 1.0], size=batch_size).astype(np.float32)
        batch = (states, policies, values)

        losses = [
            train_step(model, optimizer, loss_fn, batch)["loss"] for _ in range(40)
        ]
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(model.training)


@unittest.skipIf(torch is None, "torch is not installed")
class TestTrainLoop(unittest.TestCase):
    def test_tiny_run_writes_log_and_checkpoints_then_resumes(self):
        import csv
        import os
        import tempfile

        from network.checkpoint import load_checkpoint
        from training import train

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = os.path.join(tmp, "ckpt")
            log = os.path.join(tmp, "log.csv")
            common = [
                "--games_per_iter", "1", "--simulations", "2", "--max_plies", "6",
                "--min_buffer", "1", "--train_steps", "1", "--batch_size", "4",
                "--checkpoint_dir", ckpt_dir, "--log", log,
            ]  # fmt: skip
            with mock.patch("builtins.print"):
                train.main(["--iterations", "2", *common])
                train.main(
                    [
                        "--iterations", "3", *common,
                        "--resume", os.path.join(ckpt_dir, "latest.pt"),
                    ]
                )  # fmt: skip

            with open(log) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["iteration"] for r in rows], ["1", "2", "3"])
            self.assertTrue(all(r["train_steps"] == "1" for r in rows))
            for i in (1, 2, 3):
                self.assertTrue(
                    os.path.exists(os.path.join(ckpt_dir, f"iter_{i:04d}.pt"))
                )
            _, _, step, metadata = load_checkpoint(os.path.join(ckpt_dir, "latest.pt"))
            self.assertEqual(step, 3)
            self.assertEqual(metadata["log_row"]["iteration"], 3)

    def test_parallel_self_play_does_not_depend_on_worker_count(self):
        import csv
        import os
        import tempfile

        import torch

        from network.checkpoint import load_checkpoint
        from training import train

        def run(tmp, workers):
            ckpt_dir = os.path.join(tmp, f"ckpt{workers}")
            log = os.path.join(tmp, f"log{workers}.csv")
            with mock.patch("builtins.print"):
                train.main(
                    [
                        "--iterations", "2", "--games_per_iter", "3",
                        "--simulations", "3", "--max_plies", "8",
                        "--min_buffer", "1", "--train_steps", "2", "--batch_size", "4",
                        "--checkpoint_dir", ckpt_dir, "--log", log,
                        "--workers", str(workers),
                    ]
                )  # fmt: skip
            with open(log) as f:
                rows = list(csv.DictReader(f))
            model, _, step, _ = load_checkpoint(os.path.join(ckpt_dir, "latest.pt"))
            return rows, model, step

        with tempfile.TemporaryDirectory() as tmp:
            rows2, model2, step2 = run(tmp, 2)
            rows3, model3, step3 = run(tmp, 3)

        self.assertEqual((step2, step3), (2, 2))
        self.assertEqual([r["games"] for r in rows2], ["3", "3"])
        keys = ["white_wins", "black_wins", "draws", "unfinished", "avg_plies",
                "buffer_size", "loss"]  # fmt: skip
        self.assertEqual(
            [[r[k] for k in keys] for r in rows2], [[r[k] for k in keys] for r in rows3]
        )
        for a, b in zip(model2.state_dict().values(), model3.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_workers_must_be_positive(self):
        from training import train

        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            train.parse_args(["--workers", "0"])

    def _tiny(self, tmp, name, *extra):
        import os

        from training import train

        ckpt_dir = os.path.join(tmp, name)
        with mock.patch("builtins.print"):
            train.main(
                [
                    "--games_per_iter", "2", "--simulations", "3", "--max_plies", "6",
                    "--min_buffer", "1", "--train_steps", "1", "--batch_size", "4",
                    "--checkpoint_dir", ckpt_dir, "--log", os.path.join(tmp, name + ".csv"),
                    *extra,
                ]
            )  # fmt: skip
        return os.path.join(ckpt_dir, "latest.pt")

    def test_bigger_network_bootstrapped_from_a_smaller_one(self):
        import tempfile

        from network.checkpoint import load_checkpoint
        from training import train

        with tempfile.TemporaryDirectory() as tmp:
            small = self._tiny(tmp, "small", "--iterations", "1")

            # Serial: record which network plays each iteration's games.
            players = []
            real = train.run_self_play

            def spy(model, args, rng):
                players.append(train.model_config(model))
                return real(model, args, rng)

            with mock.patch.object(train, "run_self_play", spy):
                big = self._tiny(
                    tmp, "big", "--iterations", "3", "--res_blocks", "5",
                    "--filters", "96", "--bootstrap_model", small,
                    "--bootstrap_iters", "2",
                )  # fmt: skip
            # Parallel: workers switch between the two sizes without errors.
            self._tiny(
                tmp, "big_parallel", "--iterations", "3", "--res_blocks", "5",
                "--filters", "96", "--bootstrap_model", small,
                "--bootstrap_iters", "2", "--workers", "2",
            )  # fmt: skip
            model, _, step, _ = load_checkpoint(big)

        small_cfg = {"in_channels": 13, "num_res_blocks": 4, "num_filters": 64}
        big_cfg = {"in_channels": 13, "num_res_blocks": 5, "num_filters": 96}
        self.assertEqual(players, [small_cfg, small_cfg, big_cfg])
        self.assertEqual(train.model_config(model), big_cfg)
        self.assertEqual(step, 3)

    def test_network_size_and_bootstrap_arguments_are_checked(self):
        from training import train

        for bad in (
            ["--res_blocks", "7"],
            ["--filters", "32"],
            ["--bootstrap_iters", "2"],
        ):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                train.parse_args(bad)


if __name__ == "__main__":
    unittest.main()
