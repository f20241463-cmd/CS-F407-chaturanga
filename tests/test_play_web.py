"""Tests for the browser game (play_web.py), with the random stub network."""

import csv
import json
import threading
import unittest
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from chaturanga.stub_network import random_stub_network
from chaturanga.types import Color
from play_web import WebGame, make_handler


def stub():
    return partial(random_stub_network, rng=np.random.default_rng(0))


class WebGameTest(unittest.TestCase):
    def test_moves_alternate_and_are_checked(self):
        game = WebGame(stub(), simulations=8)
        view = game.view()
        self.assertEqual(view["legal"]["e2"], ["e3"])
        with self.assertRaises(ValueError):
            game.human_move("e2e5")
        game.human_move("e2e3")
        with self.assertRaises(ValueError):
            game.human_move("d2d3")  # agent's turn now
        game.ai_move()
        view = game.view()
        self.assertEqual(len(view["moves"]), 2)
        self.assertEqual(view["side_to_move"], "white")
        self.assertIsNotNone(view["thinking"])

    def test_agent_moves_first_when_human_is_black(self):
        game = WebGame(stub(), simulations=8)
        game.new(Color.BLACK)
        self.assertEqual(game.view()["legal"], {})
        game.ai_move()
        self.assertTrue(game.view()["legal"])

    def test_games_are_logged(self):
        with TemporaryDirectory() as tmp:
            log = str(Path(tmp) / "games.csv")
            game = WebGame(stub(), simulations=8, checkpoint="x.pt", player="P", log=log)
            game.human_move("e2e3")
            game.resign()
            game.new(Color.WHITE)  # finished game: nothing more to record
            game.human_move("d2d3")
            game.new(Color.BLACK)  # unfinished game is recorded as such
            rows = list(csv.DictReader(open(log)))
        self.assertEqual([r["result"] for r in rows], ["agent_win", "unfinished"])
        self.assertEqual(rows[0]["player"], "P")
        self.assertEqual(rows[0]["moves"], "e2e3")
        self.assertEqual(rows[1]["human_color"], "white")

    def test_server(self):
        game = WebGame(stub(), simulations=8)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(game, threading.Lock()))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def post(path, body):
            req = urllib.request.Request(base + path, json.dumps(body).encode(), method="POST")
            with urllib.request.urlopen(req) as res:
                return json.load(res)

        try:
            with urllib.request.urlopen(base + "/") as res:
                self.assertIn(b"ChaturangZero", res.read())
            self.assertEqual(post("/api/move", {"move": "e2e3"})["side_to_move"], "black")
            self.assertEqual(len(post("/api/ai", {})["moves"]), 2)
            with self.assertRaises(urllib.error.HTTPError):
                post("/api/move", {"move": "a1a5"})
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
