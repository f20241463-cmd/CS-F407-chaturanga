"""
training/replay_buffer.py -- fixed-size store of self-play training examples.

Each example is one position from a self-play game:
    (state_tensor float32[13, 8, 8], policy_target float32[4096], value float)

The policy target is a probability distribution over all 4096 actions,
but only the handful of moves MCTS actually visited are nonzero. Storing
it densely would cost 16 KB per position, so the buffer keeps just the
nonzero action indices and their probabilities, and rebuilds the dense
[4096] vector when a batch is sampled.

It is a ring buffer: a plain list plus a write index. Once the list is
full, each new example overwrites the oldest one.

No torch import: the buffer hands out numpy arrays, and the training
loop turns them into tensors.
"""

import numpy as np

POLICY_SIZE = 4096


class ReplayBuffer:
    def __init__(self, capacity=20000, seed=None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._data = []
        self._write_index = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self._data)

    def add(self, tensor, policy, value):
        policy = np.asarray(policy, dtype=np.float32)
        if policy.shape != (POLICY_SIZE,):
            raise ValueError(
                f"policy must have shape ({POLICY_SIZE},), got {policy.shape}"
            )
        indices = np.flatnonzero(policy).astype(np.int32)
        entry = (
            np.array(tensor, dtype=np.float32),  # copy, so callers can't mutate it
            indices,
            policy[indices].copy(),
            float(value),
        )
        if len(self._data) < self.capacity:
            self._data.append(entry)
        else:
            self._data[self._write_index] = entry
        self._write_index = (self._write_index + 1) % self.capacity

    def add_game(self, examples):
        for tensor, policy, value in examples:
            self.add(tensor, policy, value)

    def sample(self, batch_size):
        """
        Returns (states [B,13,8,8], policies [B,4096], values [B]), all float32.
        Samples without replacement when the buffer holds at least
        batch_size examples, and with replacement otherwise.
        """
        n = len(self._data)
        if n == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")
        picks = self._rng.choice(n, size=batch_size, replace=n < batch_size)

        states = np.stack([self._data[i][0] for i in picks]).astype(np.float32)
        policies = np.zeros((batch_size, POLICY_SIZE), dtype=np.float32)
        values = np.empty(batch_size, dtype=np.float32)
        for row, i in enumerate(picks):
            _, indices, probs, value = self._data[i]
            policies[row, indices] = probs
            values[row] = value
        return states, policies, values
