"""
mcts/reuse.py -- keep the search tree between moves.

After the agent moves, the part of its tree below that move is still
useful: it already holds searches of the opponent's likely replies. A
TreeCache keeps that part, and when the agent is to move again it hands
back the subtree for the reply actually played, so the next search starts
from those visits instead of from nothing. It uses nothing but the search
itself.

One cache can serve several games played side by side (batched matches):
each kept tree is filed under the position it belongs to, including the
game's history, so a game only ever gets back its own tree.
"""


class TreeCache:
    def __init__(self, max_trees=64):
        self.max_trees = max_trees  # old trees (from finished games) are dropped
        self._trees = {}

    @staticmethod
    def _history_after(state):
        # The history a position one move later will have.
        return state.history + (state._snapshot(),)

    def keep(self, root, action):
        """After the agent plays `action` from `root`, keep the tree below it."""
        child = root._made.get(action)
        if child is None or not child.is_expanded:
            return
        self._trees[self._history_after(child.state)] = child
        while len(self._trees) > self.max_trees:
            self._trees.pop(next(iter(self._trees)))

    def take(self, state):
        """The searched node for `state`, if the kept tree reached it; else None."""
        node = self._trees.pop(state.history, None)
        if node is None:
            return None
        target = (state._snapshot(), state.halfmove_clock)
        for child in node._made.values():
            if child.is_expanded and child._state is not None and \
                    (child._state._snapshot(), child._state.halfmove_clock) == target:  # fmt: skip
                child.parent = None  # detach, so the rest of the old tree can be freed
                return child
        return None

    def clear(self):
        self._trees.clear()
