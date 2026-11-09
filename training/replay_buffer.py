"""
training/replay_buffer.py -- fixed-size store of self-play training examples.

Each example is one position from a self-play game:
    (state_tensor float32[13, 8, 8], policy_target float32[4096], value float)

The policy target is a probability distribution over all 4096 actions,
but only the handful of moves MCTS actually visited are nonzero. Storing
it densely would cost 16 KB per position, so the buffer keeps just the
nonzero action indices and their probabilities, and rebuilds the dense
[4096] vector when a batch is sampled.

Board planes are always 0 or 1, so states are kept as uint8 (exact, and a
quarter of the memory of float32); any other state is kept as float32.

It is a ring buffer: a plain list plus a write index. Once the list is
full, each new example overwrites the oldest one.

No torch import: the buffer hands out numpy arrays, and the training
loop turns them into tensors.
"""

import numpy as np

POLICY_SIZE = 4096


def _mirror_square(sq):
    return (sq // 8) * 8 + (7 - sq % 8)


# Action id of each move's left-right mirror image (file a <-> h, b <-> g, ...).
MIRROR_ACTION = np.array(
    [_mirror_square(a // 64) * 64 + _mirror_square(a % 64) for a in range(POLICY_SIZE)],
    dtype=np.int64,
)


def compact_example(tensor, policy, value):
    """
    The stored form of one example: (state, action indices, probabilities,
    value). Self-play workers send games back in this form, which is much
    smaller to pass between processes than the dense arrays.
    """
    policy = np.asarray(policy, dtype=np.float32)
    if policy.shape != (POLICY_SIZE,):
        raise ValueError(f"policy must have shape ({POLICY_SIZE},), got {policy.shape}")
    state = np.array(tensor, dtype=np.float32)  # copy, so callers can't mutate it
    if np.all((state == 0) | (state == 1)):
        state = state.astype(np.uint8)
    indices = np.flatnonzero(policy).astype(np.int16)
    return (state, indices, policy[indices].copy(), float(value))


class ReplayBuffer:
    """
    decisive_weight: 1 (default) samples every position equally. With w > 1,
    positions from won or lost games (|value| > 0.5) are sampled w times as
    often as the rest.

    mirror: False (default). If True, each sampled position is flipped left
    to right (files a <-> h) with probability 1/2, with its policy target
    flipped to match. Every Chaturanga rule is the same mirrored, so the
    flipped position has the same value; it is a symmetry of the rules, not
    game knowledge.
    """

    def __init__(self, capacity=20000, seed=None, decisive_weight=1.0, mirror=False,
                 moves_left=False):  # fmt: skip
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._data = []
        self._write_index = 0
        self._rng = np.random.default_rng(seed)
        self.decisive_weight = float(decisive_weight)
        self.mirror = bool(mirror)
        # With moves_left, each position also keeps how many recorded plies of
        # its game came after it, and sample() returns those as a fourth array.
        self.moves_left = bool(moves_left)
        self._weights = np.ones(self.capacity) if self.decisive_weight != 1.0 else None

    def __len__(self):
        return len(self._data)

    def add(self, tensor, policy, value):
        self.add_compact(compact_example(tensor, policy, value))

    def add_compact(self, entry):
        """Add one example already in compact_example form."""
        if self._weights is not None:
            self._weights[self._write_index] = self.decisive_weight if abs(entry[3]) > 0.5 else 1.0
        if len(self._data) < self.capacity:
            self._data.append(entry)
        else:
            self._data[self._write_index] = entry
        self._write_index = (self._write_index + 1) % self.capacity

    def add_compact_game(self, entries):
        n = len(entries)
        for i, entry in enumerate(entries):
            self.add_compact((*entry[:4], n - i) if self.moves_left else entry)

    def add_game(self, examples):
        self.add_compact_game([compact_example(t, p, v) for t, p, v in examples])

    def sample(self, batch_size):
        """
        Returns (states [B,13,8,8], policies [B,4096], values [B]), all float32.
        Samples without replacement when the buffer holds at least
        batch_size examples, and with replacement otherwise.
        """
        n = len(self._data)
        if n == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")
        if self._weights is None:
            picks = self._rng.choice(n, size=batch_size, replace=n < batch_size)
        else:
            w = self._weights[:n]
            picks = self._rng.choice(n, size=batch_size, replace=n < batch_size, p=w / w.sum())

        states = np.stack([self._data[i][0] for i in picks]).astype(np.float32)
        policies = np.zeros((batch_size, POLICY_SIZE), dtype=np.float32)
        values = np.empty(batch_size, dtype=np.float32)
        left = np.zeros(batch_size, dtype=np.float32)
        for row, i in enumerate(picks):
            entry = self._data[i]
            _, indices, probs, value = entry[:4]
            policies[row, indices] = probs
            values[row] = value
            if self.moves_left:
                left[row] = entry[4] if len(entry) > 4 else 0
        if self.mirror:
            flip = self._rng.random(batch_size) < 0.5
            states[flip] = states[flip][..., ::-1]
            policies[flip] = policies[flip][:, MIRROR_ACTION]
        if self.moves_left:
            return states, policies, values, left
        return states, policies, values
