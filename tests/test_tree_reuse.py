"""Tests for keeping the search tree between moves (mcts/reuse.py)."""

import unittest
from functools import partial

import numpy as np

from chaturanga import encoding
from chaturanga.game import GameState
from chaturanga.notation import move_to_algebraic
from chaturanga.stub_network import random_stub_network
from chaturanga.types import Color
from evaluation.players import make_mcts_player, play_game, random_player
from mcts.reuse import TreeCache
from mcts.search import drive, mcts_search
from play_web import WebGame


def stub(seed=0):
    return partial(random_stub_network, rng=np.random.default_rng(seed))


def search(state, sims, network_fn, root=None):
    stats = {}
    visits, move = drive(
        mcts_search(state, sims, 1.5, encoding.state_to_tensor,
                    encoding.legal_action_mask, encoding.action_to_move,
                    stats=stats, root=root),  # fmt: skip
        network_fn,
    )
    return visits, move, stats["root"]


def most_visited_reply(root, action):
    """The opponent reply the kept tree searched most after `action`."""
    child = root._made[action]
    reply = max(child._made.values(), key=lambda c: c.visit_count)
    return reply


class TreeCacheTest(unittest.TestCase):
    def test_reply_gets_back_its_searched_subtree(self):
        net = stub()
        state = GameState()
        visits, move, root = search(state, 200, net)
        action = encoding.move_to_action(move)
        cache = TreeCache()
        cache.keep(root, action)

        reply = most_visited_reply(root, action)
        before = reply.visit_count
        self.assertGreater(before, 0)
        next_state = state.apply_move(move).apply_move(reply._move)

        node = cache.take(next_state)
        self.assertIs(node, reply)
        self.assertIsNone(node.parent)
        # The new search adds its simulations to the visits already there.
        _, _, new_root = search(next_state, 50, net, root=node)
        self.assertIs(new_root, reply)
        self.assertEqual(new_root.visit_count, before + 50)

    def test_unsearched_reply_or_other_game_gets_nothing(self):
        net = stub()
        state = GameState()
        _, move, root = search(state, 30, net)
        action = encoding.move_to_action(move)
        cache = TreeCache()
        cache.keep(root, action)
        after = state.apply_move(move)
        searched = {c._move for c in root._made[action]._made.values() if c.is_expanded}
        unsearched = [m for m in after.legal_moves() if m not in searched]
        self.assertTrue(unsearched)
        self.assertIsNone(cache.take(after.apply_move(unsearched[0])))
        # A position from a different game history is never matched.
        cache.keep(root, action)
        other = GameState().apply_move(GameState().legal_moves()[0])
        self.assertIsNone(cache.take(other))

    def test_old_trees_are_dropped(self):
        net = stub()
        cache = TreeCache(max_trees=2)
        state = GameState()
        for move in state.legal_moves()[:3]:
            s = state.apply_move(move)
            _, m, root = search(s, 20, net)
            cache.keep(root, encoding.move_to_action(m))
        self.assertEqual(len(cache._trees), 2)


class ReusePlayerTest(unittest.TestCase):
    def test_reusing_player_finishes_legal_games(self):
        player = make_mcts_player(stub(), simulations=16, reuse_tree=True)
        for seed in range(2):
            outcome, plies = play_game(player, random_player,
                                       np.random.default_rng(seed), max_plies=60)  # fmt: skip
            self.assertIn(outcome, ("white", "black", "draw", "unfinished"))
            self.assertGreater(plies, 0)

    def test_web_game_builds_on_its_last_search(self):
        game = WebGame(stub(), simulations=40)
        game.new(Color.BLACK)  # the agent plays White
        game.ai_move()
        kept = next(iter(game.tree_cache._trees.values()))
        reply = max(kept._made.values(), key=lambda c: c.visit_count)
        before = reply.visit_count
        game.human_move(move_to_algebraic(reply._move))
        game.ai_move()
        self.assertEqual(reply.visit_count, before + 40)


if __name__ == "__main__":
    unittest.main()
