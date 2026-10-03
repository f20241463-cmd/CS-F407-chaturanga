"""
mcts/node.py -- MCTSNode

One node = one GameState. A node knows its parent, its children (keyed
by action_id, not by move tuple, so it lines up directly with the
[4096]-wide policy/visit-count contract everyone else uses), and the
running statistics PUCT needs: visit_count (N), value_sum (W), and the
prior probability (P) the network/stub assigned to the move that led
here.

Perspective convention (this matters more than anything else in this
file): .value always means "how good this position is for the player
whose turn it is at THIS node". That's the same convention the network
contract uses (section 2.5: value is from the current player's
perspective). Because Chaturanga alternates turns, a child's "good for
me" is automatically "bad for my parent", which is why select_child
negates the child's value before comparing it to its siblings.
"""

import math


class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0, action_taken=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.action_taken = (
            action_taken  # action_id that led to this node, None for root
        )

        self.children = {}  # action_id -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def value(self):
        """Mean value from this node's own side-to-move's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, policy_probs, legal_actions, action_to_move_fn):
        """
        Create one child per legal action, each wrapping the resulting
        GameState. policy_probs is the full [4096] array (only the
        entries at legal_actions are ever read). action_to_move_fn is
        encoding.action_to_move -- passed in rather than imported so
        this file has zero dependency on the chaturanga package and
        can be unit-tested with any toy state that implements
        .apply_move().
        """
        for action in legal_actions:
            if action in self.children:
                continue
            move = action_to_move_fn(action)
            child_state = self.state.apply_move(move)
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                prior=float(policy_probs[action]),
                action_taken=action,
            )
        self.is_expanded = True

    def select_child(self, c_puct):
        """
        PUCT selection. Returns (action_id, child_node) for the child
        with the highest UCB score:

            score = -child.value + c_puct * child.prior * sqrt(N_parent) / (1 + N_child)

        The leading minus sign on child.value is the perspective flip
        described at the top of this file: a high value for the child
        (good for the opponent) should look bad from the parent's point
        of view, not good.
        """
        best_score = -math.inf
        best_action = None
        best_child = None
        sqrt_parent_visits = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            q = -child.value
            u = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child
