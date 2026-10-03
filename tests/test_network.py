"""
Unit tests for the Neural Network module (Student 2 Deliverables).

Verifies:
  - Input/Output tensor shapes conforming to Technical Interfaces Section 5
  - Policy logits shape [4096] and Value range [-1.0, 1.0]
  - Single state and batched forward passes
  - Backward pass and gradient flow
  - Loss function calculation
  - Checkpoint saving and loading
  - Stub network contract compatibility for MCTS (Student 3)
  - Legal move selection via masked policy
"""

import os
import tempfile
import unittest
import numpy as np
import torch

from chaturanga import GameState, state_to_tensor
from network import (
    AlphaZeroLoss,
    ChaturangaNet,
    choose_network_move,
    load_checkpoint,
    make_network_fn,
    save_checkpoint,
)


class TestChaturangaNetArchitecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.net = ChaturangaNet(in_channels=13, num_res_blocks=2, num_filters=32)

    def test_forward_pass_single(self):
        state = GameState()
        tensor = state_to_tensor(state)  # [13, 8, 8]
        logits, value = self.net.predict_numpy(tensor)

        self.assertEqual(logits.shape, (4096,))
        self.assertEqual(logits.dtype, np.float32)
        self.assertIsInstance(value, float)
        self.assertTrue(-1.0 <= value <= 1.0)

    def test_forward_pass_batch(self):
        batch = torch.randn(4, 13, 8, 8, dtype=torch.float32)
        logits, values = self.net(batch)

        self.assertEqual(logits.shape, (4, 4096))
        self.assertEqual(values.shape, (4, 1))
        self.assertTrue((values >= -1.0).all() and (values <= 1.0).all())

    def test_gradient_flow_backward(self):
        self.net.train()
        x = torch.randn(2, 13, 8, 8, requires_grad=True)
        logits, values = self.net(x)

        dummy_target_policy = torch.softmax(torch.randn(2, 4096), dim=-1)
        dummy_target_value = torch.tensor([[1.0], [-1.0]])

        loss_fn = AlphaZeroLoss()
        loss, _ = loss_fn(logits, values, dummy_target_policy, dummy_target_value)
        loss.backward()

        # Check gradients exist and contain no NaNs
        for name, param in self.net.named_parameters():
            self.assertIsNotNone(param.grad, f"Param {name} has no gradient")
            self.assertFalse(torch.isnan(param.grad).any(), f"Param {name} gradient contains NaN")


class TestLossFunction(unittest.TestCase):
    def test_loss_computation(self):
        loss_fn = AlphaZeroLoss(value_weight=1.0, policy_weight=1.0)

        pred_logits = torch.zeros(2, 4096)
        pred_values = torch.zeros(2, 1)

        # Uniform target policy
        target_policies = torch.full((2, 4096), 1.0 / 4096)
        target_values = torch.zeros(2, 1)

        total_loss, metrics = loss_fn(pred_logits, pred_values, target_policies, target_values)

        self.assertAlmostEqual(metrics["value_loss"], 0.0, places=4)
        # Uniform log-softmax of 4096 elements is ln(4096) ~= 8.3177
        self.assertAlmostEqual(metrics["policy_loss"], float(np.log(4096)), places=3)
        self.assertAlmostEqual(metrics["loss"], metrics["value_loss"] + metrics["policy_loss"], places=3)


class TestCheckpointing(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        net = ChaturangaNet(in_channels=13, num_res_blocks=2, num_filters=32)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        state = GameState()
        tensor = state_to_tensor(state)
        logits_before, val_before = net.predict_numpy(tensor)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_model.pt")
            save_checkpoint(ckpt_path, net, optimizer, step=42, metadata={"version": "1.0"})

            loaded_net, loaded_opt, step, meta = load_checkpoint(ckpt_path)

            self.assertEqual(step, 42)
            self.assertEqual(meta.get("version"), "1.0")

            logits_after, val_after = loaded_net.predict_numpy(tensor)
            np.testing.assert_allclose(logits_before, logits_after, rtol=1e-5, atol=1e-5)
            self.assertAlmostEqual(val_before, val_after, places=5)


class TestIntegrationCompatibility(unittest.TestCase):
    def test_network_fn_contract(self):
        """Verifies make_network_fn matches the stub_network contract for Student 3."""
        net = ChaturangaNet(in_channels=13, num_res_blocks=2, num_filters=32)
        network_fn = make_network_fn(net)

        state = GameState()
        tensor = state_to_tensor(state)
        logits, value = network_fn(tensor)

        self.assertEqual(logits.shape, (4096,))
        self.assertIsInstance(value, float)
        self.assertTrue(-1.0 <= value <= 1.0)

    def test_choose_network_move_is_always_legal(self):
        """Verifies masked move selection never chooses an illegal move."""
        net = ChaturangaNet(in_channels=13, num_res_blocks=2, num_filters=32)
        state = GameState()

        # Deterministic
        move, _ = choose_network_move(state, net, temperature=0.0)
        self.assertIn(move, state.legal_moves())

        # Stochastic temperature sampling
        move_sampled, _ = choose_network_move(state, net, temperature=1.0)
        self.assertIn(move_sampled, state.legal_moves())


if __name__ == "__main__":
    unittest.main()
