"""
mcts/gumbel.py -- Gumbel root search (Danihelka et al., "Policy improvement
by planning with Gumbel", ICLR 2022), for self-play with few simulations.

Plain AlphaZero self-play adds Dirichlet noise at the root and trains on the
visit counts. With only 50 simulations the visit counts are a coarse target,
and a move that was never visited gets no credit at all. The Gumbel search
instead:

1. samples Gumbel noise g(a) for every legal move and considers the top m
   moves by g(a) + logit(a) (sampling without replacement from the prior);
2. splits the simulations between the considered moves by Sequential
   Halving: each phase visits every remaining move equally, then keeps the
   better half by g(a) + logit(a) + sigma(q(a));
3. plays the move that is left, and trains on the improved policy
   softmax(logit + sigma(completed q)), which gives every legal move a
   target from its prior and, if visited, its searched value.

Below the root the ordinary PUCT selection is used. Values follow the rest
of the search: a node's value is from its own side to move's point of view.
Constants follow DeepMind's mctx defaults (maxvisit_init 50, value_scale
0.1, values rescaled to [0, 1]).
"""

import math

import numpy as np

from mcts.node import MCTSNode
from mcts.search import _evaluate_and_expand

C_VISIT = 50.0
C_SCALE = 0.1


def _sigma(q, max_visits):
    return (C_VISIT + max_visits) * C_SCALE * q


def completed_q(prior, q, visits, root_value):
    """
    q for every legal move: the searched value where a move was visited,
    otherwise a mix of the network's root value and the visited moves' values
    (weighted by prior), then rescaled to [0, 1] over the legal moves.
    """
    visited = visits > 0
    total = visits.sum()
    if visited.any():
        weighted = (prior[visited] * q[visited]).sum() / max(prior[visited].sum(), 1e-12)
        v_mix = (root_value + total * weighted) / (1 + total)
    else:
        v_mix = root_value
    cq = np.where(visited, q, v_mix)
    lo, hi = cq.min(), cq.max()
    return (cq - lo) / max(hi - lo, 1e-8)


def gumbel_search(root_state, num_simulations, c_puct, state_to_tensor_fn,
                  legal_action_mask_fn, action_to_move_fn, rng, max_considered=16,
                  stats=None):  # fmt: skip
    """
    Generator like mcts_search (yields state tensors, takes (logits, value)).
    Returns (policy_target float32[4096], action id to play), or
    (zeros, None) if the game is already over. stats["root_value"] is set
    as in mcts_search.
    """
    root = MCTSNode(root_state)
    raw_value = yield from _evaluate_and_expand(root, state_to_tensor_fn, legal_action_mask_fn,
                                                action_to_move_fn)  # fmt: skip
    children = root.children
    if not children:
        return np.zeros(4096, dtype=np.float32), None
    actions = np.array(list(children), dtype=np.int64)
    prior = np.array([children[a].prior for a in actions], dtype=np.float64)
    logits = np.log(np.maximum(prior, 1e-12))  # log-prior: the logits up to a constant
    gumbel = rng.gumbel(size=len(actions))

    def simulate(index):
        node = children[int(actions[index])]
        path = [root, node]
        while node.is_expanded and node.has_children:
            _, node = node.select_child(c_puct)
            path.append(node)
        if node.state.is_game_over():
            value = node.state.result() or 0.0
        else:
            value = yield from _evaluate_and_expand(node, state_to_tensor_fn,
                                                    legal_action_mask_fn, action_to_move_fn)  # fmt: skip
        for ancestor in reversed(path):
            ancestor.visit_count += 1
            ancestor.value_sum += value
            value = -value

    def q_and_visits():
        visits = np.array([children[a].visit_count for a in actions], dtype=np.float64)
        q = np.array([-children[a].value for a in actions], dtype=np.float64)  # root's view
        return q, visits

    m = min(max_considered, len(actions))
    considered = list(np.argsort(-(gumbel + logits))[:m])
    phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
    used = 0
    for phase in range(phases):
        last = phase == phases - 1 or len(considered) == 1
        if last:
            per = max(1, (num_simulations - used) // len(considered))
        else:
            per = max(1, num_simulations // (phases * len(considered)))
        for _ in range(per):
            for i in considered:
                yield from simulate(i)
                used += 1
        if last:
            break
        q, visits = q_and_visits()
        score = gumbel + logits + _sigma(completed_q(prior, q, visits, raw_value), visits.max())
        considered = sorted(considered, key=lambda i: -score[i])[: max(1, len(considered) // 2)]

    q, visits = q_and_visits()
    cq = completed_q(prior, q, visits, raw_value)
    improved = logits + _sigma(cq, visits.max())
    improved = np.exp(improved - improved.max())
    improved /= improved.sum()
    score = gumbel + logits + _sigma(cq, visits.max())
    chosen = max(considered, key=lambda i: score[i])

    policy = np.zeros(4096, dtype=np.float32)
    policy[actions] = improved
    if stats is not None:
        stats["root_value"] = root.value
        stats["improved_order"] = [int(actions[i]) for i in np.argsort(-improved)]
    return policy, int(actions[chosen])
