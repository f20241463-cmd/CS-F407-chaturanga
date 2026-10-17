"""
training -- the self-play / training loop (Student 3).

replay_buffer and self_play do not import torch, so they can be tested
and used with the stub networks. train.py (run as python -m training.train)
is the only module that needs torch, and it is not imported here.
"""

from training.replay_buffer import ReplayBuffer
from training.self_play import outcome_values, play_self_play_game

__all__ = ["ReplayBuffer", "outcome_values", "play_self_play_game"]
