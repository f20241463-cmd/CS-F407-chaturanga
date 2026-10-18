"""
mcts/search.py -- the search() entry point.

run_mcts(...) and select_action_from_visits(...) are the public API.
Everything else in this file is a private helper.

network_fn must match the frozen contract from the execution-plan doc
(section 2.5 / stub_network.py): given a [13,8,8] float32 tensor, it
returns (policy_logits: float32[4096], value: float in [-1, 1]).
During development, pass in chaturanga.stub_network.random_stub_network
or zero_stub_network. Later, swap in the real trained network -- this
file never needs to change, only which function gets passed in.

Output: (visit_counts, best_move)
  visit_counts -- float32[4096], matching section 2.7/2.8 of the
                  execution plan. This IS the un-normalized policy
                  target; Student 4 divides by its sum to get
                  policy_target for the replay buffer.
  best_move    -- (from_square, to_square), the single move actually
                  chosen (highest visit count).
"""

import math
import numpy as np

from mcts.node import MCTSNode


def _softmax_masked(logits, legal_mask):
    """Softmax over only the legal entries of a [4096] logit vector.
    Illegal entries get probability 0."""
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(legal_mask, dtype=bool)

    if not mask.any():
        # No legal moves at all -- caller (run_mcts) checks for this
        # before it matters, but stay safe rather than divide by zero.
        return np.zeros_like(logits)

    masked_logits = np.where(mask, logits, -np.inf)
    max_logit = np.max(logits[mask])
    exp = np.where(mask, np.exp(masked_logits - max_logit), 0.0)
    total = exp.sum()

    if total <= 0:
        # Degenerate case (e.g. exactly one legal move with a huge
        # negative logit) -- fall back to a uniform distribution over
        # legal moves so the search can still proceed.
        probs = mask.astype(np.float64)
        probs /= probs.sum()
        return probs

    return exp / total


def _legal_actions_from_mask(legal_mask):
    return np.flatnonzero(legal_mask).tolist()


def _evaluate_and_expand(
    node, network_fn, state_to_tensor_fn, legal_action_mask_fn, action_to_move_fn
):
    """
    Run the network/stub on node.state, turn its output into a masked
    policy distribution, and expand node with one child per legal
    move. Returns the value estimate for this node's side-to-move, to
    be backpropagated.
    """
    tensor = state_to_tensor_fn(node.state)
    policy_logits, value = network_fn(tensor)

    legal_mask = legal_action_mask_fn(node.state)
    legal_actions = _legal_actions_from_mask(legal_mask)
    policy_probs = _softmax_masked(policy_logits, legal_mask)

    node.expand(policy_probs, legal_actions, action_to_move_fn)
    return value


def _add_dirichlet_noise(root, alpha, eps, rng):
    """
    Mix Dirichlet noise into the priors of the root's legal children, in place:

        P'(a) = (1 - eps) * P(a) + eps * eta_a,  eta ~ Dir(alpha)

    This is AlphaZero's exploration trick for self-play. Without it, an
    untrained network's fixed (and arbitrary) priors make the search
    pick the same few moves every game, so self-play keeps generating
    near-identical games. Only ever applied at the root, and only
    during self-play; evaluation and play_vs_ai leave it off.
    """
    actions = list(root.children)
    if not actions:
        return
    noise = rng.dirichlet([alpha] * len(actions))
    for action, eta in zip(actions, noise):
        child = root.children[action]
        child.prior = (1 - eps) * child.prior + eps * float(eta)


def select_action_from_visits(visit_counts, temperature=0.0, rng=None):
    """
    Pick an action id from a [4096] visit-count vector.

    temperature <= 1e-3: greedy, the most-visited action.
    otherwise: sample with probability proportional to
               visits ** (1 / temperature). Actions with zero visits are
               never chosen.
    Raises ValueError if no action has any visits.
    """
    visits = np.asarray(visit_counts, dtype=np.float64)
    if visits.sum() <= 0:
        raise ValueError("select_action_from_visits: no action has any visits")

    if temperature <= 1e-3:
        return int(np.argmax(visits))

    # Work in log space so large exponents (small temperatures) don't overflow.
    visited = visits > 0
    log_w = np.full_like(visits, -np.inf)
    log_w[visited] = np.log(visits[visited]) / temperature
    w = np.exp(log_w - log_w[visited].max())
    probs = w / w.sum()

    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.choice(len(probs), p=probs))


def run_mcts(
    root_state,
    network_fn,
    num_simulations,
    c_puct,
    state_to_tensor_fn,
    legal_action_mask_fn,
    action_to_move_fn,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.0,
    rng=None,
):
    """
    state_to_tensor_fn, legal_action_mask_fn, action_to_move_fn are
    passed in explicitly (rather than imported from chaturanga.encoding
    directly in this file) so mcts/ has no hard dependency on the
    environment package and tests can swap in a toy implementation.
    In real use you'll call it like:

        from chaturanga import encoding
        from chaturanga.stub_network import random_stub_network
        visit_counts, move = run_mcts(
            state, random_stub_network, 50, 1.5,
            encoding.state_to_tensor, encoding.legal_action_mask,
            encoding.action_to_move,
        )

    Root Dirichlet noise is off by default (dirichlet_eps=0.0), so
    evaluation and play behave exactly as before this option existed.
    Self-play turns it on with dirichlet_eps=0.25 (AlphaZero's value)
    and dirichlet_alpha=0.3 (AlphaZero's chess value; Chaturanga has a
    similar number of legal moves per position), plus a seeded
    np.random.Generator as rng so the noisy games are reproducible.
    """
    root = MCTSNode(root_state)

    # Root is always expanded immediately, even before the simulation
    # loop starts, so there's something to select among on simulation 1.
    _evaluate_and_expand(
        root, network_fn, state_to_tensor_fn, legal_action_mask_fn, action_to_move_fn
    )

    if not root.children:
        # No legal moves at all from the root -- game is already over.
        return np.zeros(4096, dtype=np.float32), None

    if dirichlet_eps > 0:
        _add_dirichlet_noise(
            root,
            dirichlet_alpha,
            dirichlet_eps,
            rng if rng is not None else np.random.default_rng(),
        )

    for _ in range(num_simulations):
        node = root
        path = [node]

        # Selection: descend while the node has already been expanded.
        while node.is_expanded and node.children:
            _, node = node.select_child(c_puct)
            path.append(node)

        # node is now a leaf: either unexpanded, or a terminal state.
        if node.state.is_game_over():
            value = node.state.result()
            # result() returning None would mean "not over", which
            # contradicts is_game_over() -- treat as a draw defensively.
            if value is None:
                value = 0.0
        else:
            value = _evaluate_and_expand(
                node,
                network_fn,
                state_to_tensor_fn,
                legal_action_mask_fn,
                action_to_move_fn,
            )

        # Backpropagation: value is from `node`'s side-to-move
        # perspective. Flip sign at each step up, since every step up
        # the tree swaps whose turn it is.
        for ancestor in reversed(path):
            ancestor.visit_count += 1
            ancestor.value_sum += value
            value = -value

    visit_counts = np.zeros(4096, dtype=np.float32)
    for action, child in root.children.items():
        visit_counts[action] = child.visit_count

    best_action = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    best_move = action_to_move_fn(best_action)

    return visit_counts, best_move
