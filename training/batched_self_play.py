"""
training/batched_self_play.py -- play many self-play games at once and
evaluate their positions in one network call.

Each game is a self_play_game_steps generator. All games run side by side:
every game advances until its search needs the network, the waiting
positions are stacked into one batch, the network evaluates the batch, and
each game gets its own answer back. Every game makes exactly the moves it
would make alone with the same seed; only the network calls are grouped.
This matters on a GPU, which is fast on a batch of positions but slow on
one at a time.

With groups > 1 the games are split into groups that take turns: while the
GPU evaluates one group's batch, the CPU advances the games of another.
"""

from collections import deque

import numpy as np


def play_games_batched(games, network, groups=1, on_done=None):
    """
    Run generators side by side to the end and return their results in order.

    games: list of generators that yield [13, 8, 8] tensors, take
        (policy_logits, value) back, and return a result (for example
        self_play_game_steps(...)).
    network: fn(float32[B, 13, 8, 8]) -> (logits [B, 4096], values [B]), or
        an object with submit(states) -> handle and fetch(handle) -> the same
        outputs (network.agent.BatchNetwork), which lets groups overlap.
    groups: how many groups to split the games into (only used with submit).
    on_done: optional fn(index, result), called as each game finishes. If
        it returns something other than None, that is kept as the game's
        result instead (for example a compact copy, to save memory).
    """
    if hasattr(network, "submit"):
        submit, fetch = network.submit, network.fetch
    else:
        submit, fetch = network, lambda outputs: outputs
        groups = 1
    results = [None] * len(games)
    waiting = {}  # game index -> the tensor it is waiting on

    def advance(i, answer):
        try:
            waiting[i] = games[i].send(answer)
        except StopIteration as stop:
            waiting.pop(i, None)
            results[i] = stop.value
            if on_done is not None:
                kept = on_done(i, stop.value)
                if kept is not None:
                    results[i] = kept

    for i in range(len(games)):
        advance(i, None)  # a generator's first send must be None

    def start(members):
        members = [i for i in members if i in waiting]
        if members:
            queue.append((members, submit(np.stack([waiting[i] for i in members]))))

    queue = deque()
    for k in range(max(1, groups)):
        start(range(k, len(games), max(1, groups)))
    while queue:
        members, handle = queue.popleft()
        logits, values = fetch(handle)
        for row, i in enumerate(members):
            advance(i, (logits[row], float(values[row])))
        start(members)
    return results


def play_keyed_games_batched(games, networks, on_done=None):
    """
    Like play_games_batched, for games where positions go to different
    networks (a match between two agents): each generator yields
    (key, tensor), and networks[key] evaluates it. Each step submits one
    batch per network before waiting on any of them, so the networks'
    batches overlap.
    """
    results = [None] * len(games)
    waiting = {}  # game index -> (key, tensor)

    def advance(i, answer):
        try:
            waiting[i] = games[i].send(answer)
        except StopIteration as stop:
            waiting.pop(i, None)
            results[i] = stop.value
            if on_done is not None:
                kept = on_done(i, stop.value)
                if kept is not None:
                    results[i] = kept

    for i in range(len(games)):
        advance(i, None)
    while waiting:
        by_key = {}
        for i, (key, _) in waiting.items():
            by_key.setdefault(key, []).append(i)
        started = []
        for key, members in by_key.items():
            network = networks[key]
            batch = np.stack([waiting[i][1] for i in members])
            if hasattr(network, "submit"):
                started.append((members, network.fetch, network.submit(batch)))
            else:
                started.append((members, lambda out: out, network(batch)))
        for members, fetch, handle in started:
            logits, values = fetch(handle)
            for row, i in enumerate(members):
                advance(i, (logits[row], float(values[row])))
    return results
