"""Tests for the last round of training ideas (late starts, mixed search sizes, ...)."""

import csv
import os
import tempfile
import unittest
from functools import partial
from unittest import mock

import numpy as np

from chaturanga import encoding
from chaturanga.game import GameState
from chaturanga.stub_network import random_stub_network
from training.self_play import play_self_play_game


def stub(seed=0):
    return partial(random_stub_network, rng=np.random.default_rng(seed))


def train_run(tmp, name, *extra):
    from training import train

    log = os.path.join(tmp, name + ".csv")
    with mock.patch("builtins.print"):
        train.main(
            [
                "--iterations", "2", "--games_per_iter", "3", "--simulations", "3",
                "--max_plies", "30", "--min_buffer", "1", "--train_steps", "2",
                "--batch_size", "4", "--selfplay_device", "cpu",
                "--checkpoint_dir", os.path.join(tmp, name), "--log", log, *extra,
            ]
        )  # fmt: skip
    with open(log) as f:
        return list(csv.DictReader(f))


class LateStartTest(unittest.TestCase):
    def test_game_continues_from_the_given_moves(self):
        _, info = play_self_play_game(stub(), num_simulations=3, max_plies=12,
                                      rng=np.random.default_rng(1), record_moves=True)  # fmt: skip
        prefix = info["moves"][:6]
        examples, info2 = play_self_play_game(stub(), num_simulations=3, max_plies=12,
                                              rng=np.random.default_rng(2), start_moves=prefix,
                                              record_moves=True)  # fmt: skip
        self.assertEqual(info2["moves"][:6], prefix)
        # The replayed moves count towards max_plies but are not training examples.
        self.assertEqual(len(examples), info2["plies"] - 6)
        state = GameState()
        for a in prefix:
            state = state.apply_move(encoding.action_to_move(a))
        np.testing.assert_array_equal(examples[0][0], encoding.state_to_tensor(state))

    def test_choose_start_moves(self):
        from training.train import choose_start_moves

        games = [list(range(100)), list(range(1000, 1040))]
        self.assertEqual(choose_start_moves(games, 4, 0, 1, 0.0, 0.5, 0.9), [None] * 4)
        self.assertEqual(choose_start_moves([], 4, 0, 1, 1.0, 0.5, 0.9), [None] * 4)
        starts = choose_start_moves(games, 50, 0, 1, 0.5, 0.5, 0.9)
        self.assertEqual(starts, choose_start_moves(games, 50, 0, 1, 0.5, 0.5, 0.9))
        used = [s for s in starts if s is not None]
        self.assertTrue(10 < len(used) < 40)
        for s in used:
            full = games[0] if s[0] == 0 else games[1]
            self.assertTrue(0.5 * len(full) <= len(s) < 0.9 * len(full))

    def test_training_with_late_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = train_run(tmp, "late", "--late_start_fraction", "1.0")
        self.assertEqual([r["iteration"] for r in rows], ["1", "2"])


class MixedSearchTest(unittest.TestCase):
    def test_only_full_searches_are_recorded(self):
        examples, info = play_self_play_game(stub(), num_simulations=4, max_plies=40,
                                             rng=np.random.default_rng(3), fast_simulations=2,
                                             full_fraction=0.25)  # fmt: skip
        self.assertGreater(info["plies"], len(examples))
        self.assertGreater(len(examples), 0)
        all_full, info2 = play_self_play_game(stub(), num_simulations=4, max_plies=40,
                                              rng=np.random.default_rng(3), fast_simulations=2,
                                              full_fraction=1.0)  # fmt: skip
        self.assertEqual(len(all_full), info2["plies"])

    def test_training_with_mixed_searches(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = train_run(tmp, "mixed", "--fast_simulations", "2", "--full_fraction", "0.5")
        self.assertEqual(len(rows), 2)



class BufferIdeasTest(unittest.TestCase):
    def test_mirror_is_a_symmetry_of_the_rules(self):
        from chaturanga.board import Board
        from training.replay_buffer import MIRROR_ACTION

        rng = np.random.default_rng(0)
        state = GameState()
        for _ in range(30):  # a random middlegame position
            moves = state.legal_moves()
            state = state.apply_move(moves[int(rng.integers(len(moves)))])
        squares = state.board._squares
        mirrored = GameState(Board([squares[(sq // 8) * 8 + 7 - sq % 8] for sq in range(64)]),
                             state.side_to_move, state.halfmove_clock)  # fmt: skip
        np.testing.assert_array_equal(encoding.state_to_tensor(state)[..., ::-1],
                                      encoding.state_to_tensor(mirrored))  # fmt: skip
        legal = np.flatnonzero(encoding.legal_action_mask(state))
        self.assertEqual(sorted(MIRROR_ACTION[legal]),
                         sorted(np.flatnonzero(encoding.legal_action_mask(mirrored))))  # fmt: skip

    def test_buffer_mirrors_state_and_policy_together(self):
        from training.replay_buffer import MIRROR_ACTION, ReplayBuffer

        state = encoding.state_to_tensor(GameState())
        policy = np.zeros(4096, np.float32)
        policy[8 * 64 + 16] = 1.0  # a2a3
        buf = ReplayBuffer(10, seed=0, mirror=True)
        buf.add(state, policy, 0.0)
        flipped = 0
        for _ in range(40):
            s, p, _ = buf.sample(1)
            if p[0, 8 * 64 + 16] == 0:
                flipped += 1
                self.assertEqual(p[0, MIRROR_ACTION[8 * 64 + 16]], 1.0)  # h2h3
                np.testing.assert_array_equal(s[0], state[..., ::-1])
        self.assertTrue(5 < flipped < 35)

    def test_decisive_positions_are_sampled_more(self):
        from training.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(1000, seed=0, decisive_weight=9.0)
        state, policy = np.zeros((13, 8, 8), np.float32), np.eye(4096, dtype=np.float32)[0]
        for i in range(1000):
            buf.add(state, policy, 1.0 if i < 100 else 0.0)  # 10% decisive
        values = np.concatenate([buf.sample(20)[2] for _ in range(50)])
        self.assertTrue(0.38 < np.mean(values != 0) < 0.62)  # about 9 x 10% / (9 x 10% + 90%)

    def test_training_with_buffer_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = train_run(tmp, "buf", "--decisive_weight", "3", "--mirror")
        self.assertEqual(len(rows), 2)



class OpponentTest(unittest.TestCase):
    def test_only_the_current_side_is_recorded(self):
        from chaturanga.types import Color
        from training.batched_self_play import play_keyed_games_batched
        from training.self_play import self_play_game_steps

        seen = {"cur": 0, "old": 0}

        def counting(key):
            def net(states):
                seen[key] += len(states)
                return np.zeros((len(states), 4096), np.float32), np.zeros(len(states), np.float32)
            return net

        game = self_play_game_steps(num_simulations=3, max_plies=10, rng=np.random.default_rng(0),
                                    opponent_color=Color.BLACK)  # fmt: skip
        [(examples, info)] = play_keyed_games_batched([game], {"cur": counting("cur"), "old": counting("old")})
        self.assertEqual(len(examples), (info["plies"] + 1) // 2)  # White's moves only
        self.assertTrue(seen["cur"] > 0 and seen["old"] > 0)
        for tensor, _, _ in examples:
            self.assertEqual(tensor[12].max(), 0.0)  # White to move in every example

    def test_training_against_an_earlier_checkpoint(self):
        import torch

        from network.checkpoint import save_checkpoint
        from network.model import ChaturangaNet

        with tempfile.TemporaryDirectory() as tmp:
            old = os.path.join(tmp, "old.pt")
            torch.manual_seed(5)
            save_checkpoint(old, ChaturangaNet())
            rows = train_run(tmp, "opp", "--opponent_checkpoints", old, "--opponent_fraction", "0.7")
        self.assertEqual(len(rows), 2)


def run_gumbel(network_fn, state, sims, m=16, seed=0):
    from mcts.gumbel import gumbel_search
    from mcts.search import drive

    stats = {}
    policy, action = drive(gumbel_search(state, sims, 1.5, encoding.state_to_tensor,
                                         encoding.legal_action_mask, encoding.action_to_move,
                                         np.random.default_rng(seed), m, stats), network_fn)  # fmt: skip
    return policy, action, stats


class GumbelTest(unittest.TestCase):
    def test_completed_q(self):
        from mcts.gumbel import completed_q

        prior = np.array([0.5, 0.3, 0.2])
        q = np.array([0.8, -0.4, 0.0])
        visits = np.array([2.0, 2.0, 0.0])
        cq = completed_q(prior, q, visits, root_value=0.1)
        # unvisited move: (0.1 + 4 * (0.5*0.8 + 0.3*-0.4) / 0.8) / 5 = 0.3, then rescaled.
        np.testing.assert_allclose(cq, [1.0, 0.0, (0.3 + 0.4) / 1.2], atol=1e-9)

    def test_policy_and_move_are_well_formed(self):
        state = GameState()
        policy, action, stats = run_gumbel(stub(), state, 50)
        legal = encoding.legal_action_mask(state)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertTrue((policy[legal] > 0).all() and (policy[~legal] == 0).all())
        self.assertTrue(legal[action])
        self.assertIn("root_value", stats)

    def test_finds_the_move_the_network_rates_as_winning(self):
        state = GameState()
        target = state.apply_move(state.legal_moves()[7])
        target_tensor = encoding.state_to_tensor(target)

        def network(tensor):  # uniform priors; the opponent is lost after one move
            lost = np.array_equal(tensor, target_tensor)
            return np.zeros(4096, np.float32), (-1.0 if lost else 0.0)

        best = encoding.move_to_action(state.legal_moves()[7])
        for seed in range(3):
            policy, action, _ = run_gumbel(network, state, 50, m=20, seed=seed)
            self.assertEqual(action, best)
            self.assertEqual(int(np.argmax(policy)), best)

    def test_training_with_gumbel_self_play(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = train_run(tmp, "gumbel", "--gumbel", "--gumbel_actions", "4")
        self.assertEqual(len(rows), 2)


class MovesLeftTest(unittest.TestCase):
    def test_buffer_keeps_plies_left(self):
        from training.replay_buffer import ReplayBuffer, compact_example

        state, policy = np.zeros((13, 8, 8), np.float32), np.eye(4096, dtype=np.float32)[3]
        buf = ReplayBuffer(10, seed=0, moves_left=True)
        buf.add_compact_game([compact_example(state, policy, float(i)) for i in range(4)])
        _, _, values, left = buf.sample(4)
        self.assertEqual(sorted(zip(values, left)), [(0, 4), (1, 3), (2, 2), (3, 1)])

    def test_backbone_and_heads_match_forward(self):
        import torch

        from network.model import ChaturangaNet

        torch.manual_seed(0)
        net = ChaturangaNet().eval()
        x = torch.rand(3, 13, 8, 8)
        with torch.no_grad():
            p1, v1 = net(x)
            p2, v2 = net.heads(net.features(x))
        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(v1, v2)

    def test_head_learns_plies_left(self):
        import torch

        from network.loss import AlphaZeroLoss
        from network.model import ChaturangaNet, MovesLeftHead
        from training.train import train_step

        torch.manual_seed(0)
        net, head = ChaturangaNet(), MovesLeftHead(64)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        aux = (head, torch.optim.Adam(head.parameters(), lr=1e-2), 1.0)
        states = np.random.default_rng(0).random((8, 13, 8, 8)).astype(np.float32)
        batch = (states, np.full((8, 4096), 1 / 4096, np.float32), np.zeros(8, np.float32),
                 np.full(8, 150.0, np.float32))  # fmt: skip
        first = train_step(net, opt, AlphaZeroLoss(), batch, aux=aux)["moves_left_loss"]
        for _ in range(30):
            last = train_step(net, opt, AlphaZeroLoss(), batch, aux=aux)["moves_left_loss"]
        self.assertLess(last, first / 4)

    def test_training_with_moves_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = train_run(tmp, "ml", "--moves_left_weight", "0.1")
            from network.checkpoint import load_checkpoint

            load_checkpoint(os.path.join(tmp, "ml", "iter_0002.pt"))  # saved network unchanged
        self.assertEqual(len(rows), 2)


class ReanalyseTest(unittest.TestCase):
    def test_replays_the_game_with_new_targets(self):
        from mcts.search import drive
        from training.self_play import reanalyse_game_steps

        original, info = play_self_play_game(stub(), num_simulations=3, max_plies=16,
                                             rng=np.random.default_rng(4), record_moves=True)  # fmt: skip
        examples, info2 = drive(reanalyse_game_steps(info["moves"], num_simulations=6), stub(9))
        self.assertEqual(len(examples), len(original))
        self.assertEqual(info2["outcome"], info["outcome"])
        for (t1, _, v1), (t2, p2, v2) in zip(original, examples):
            np.testing.assert_array_equal(t1, t2)
            self.assertEqual(v1, v2)
            self.assertAlmostEqual(float(p2.sum()), 1.0, places=5)

    def test_training_with_reanalyse(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = train_run(tmp, "re", "--reanalyse_games", "2", "--max_plies", "40")
        # Reanalysed games are not counted as games played.
        self.assertEqual([r["games"] for r in rows], ["3", "3"])
        self.assertGreater(int(rows[1]["buffer_size"]) - int(rows[0]["buffer_size"]), 0)


class EnsembleTest(unittest.TestCase):
    def test_ensemble_of_one_network_keeps_its_choices(self):
        import torch

        from network.agent import EnsembleNetwork, make_batch_network_fn
        from network.model import ChaturangaNet

        torch.manual_seed(0)
        a, b = ChaturangaNet(), ChaturangaNet()
        states = np.stack([encoding.state_to_tensor(GameState())] * 2)
        one = make_batch_network_fn(a)
        logits, values = EnsembleNetwork([one])(states)
        ref_logits, ref_values = one(states)
        np.testing.assert_allclose(values, ref_values, atol=1e-6)
        # Log-probabilities differ from logits by a constant, so the move order is the same.
        np.testing.assert_allclose(logits - logits.max(), ref_logits - ref_logits.max(), atol=1e-4)
        both_logits, both_values = EnsembleNetwork([one, make_batch_network_fn(b)])(states)
        np.testing.assert_allclose(both_values, (ref_values + make_batch_network_fn(b)(states)[1]) / 2,
                                   atol=1e-6)  # fmt: skip

    def test_match_with_an_ensemble(self):
        from evaluation.parallel_match import mcts_spec, play_match_parallel

        with tempfile.TemporaryDirectory() as tmp:
            import torch

            from network.checkpoint import save_checkpoint
            from network.model import ChaturangaNet

            paths = []
            for i in range(2):
                torch.manual_seed(i)
                paths.append(os.path.join(tmp, f"n{i}.pt"))
                save_checkpoint(paths[-1], ChaturangaNet())
            ens = mcts_spec("+".join(paths), simulations=3)
            for device in (None, "cpu"):
                c = play_match_parallel(ens, {"kind": "random"}, 2, seed=0, max_plies=20, device=device)
                self.assertEqual(c["wins"] + c["losses"] + c["draws"] + c["unfinished"], 2)


if __name__ == "__main__":
    unittest.main()
