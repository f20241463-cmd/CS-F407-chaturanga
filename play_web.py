"""
play_web.py -- play Chaturanga against the agent in the browser.

Same game, search and result log as play_vs_ai.py, with a clickable board
instead of typed moves. It runs on this computer only (no internet needed):

    python play_web.py --player YOUR_NAME
    python play_web.py --player YOUR_NAME --simulations 50 --port 8001

then open the address it prints (it tries to open the browser itself).
Click one of your pieces to see where it can go, then click the square.
Every finished game is appended to results/human_games.csv, like the
terminal version (turn off with --no_log).
"""

import argparse
import json
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from chaturanga import encoding
from chaturanga.game import GameState
from chaturanga.notation import move_to_algebraic, square_to_algebraic
from chaturanga.stub_network import random_stub_network
from chaturanga.types import Color, PieceType
from evaluation.game_endings import classify_ending
from evaluation.human_log import append_game, format_summary, summarize
from mcts.reuse import TreeCache
from mcts.search import drive, mcts_search
from training.self_play import filter_repeating_moves

PAGE = Path(__file__).with_name("web") / "play.html"

PIECE_NAMES = {
    PieceType.RAJA: "raja",
    PieceType.MANTRI: "mantri",
    PieceType.RATHA: "ratha",
    PieceType.GAJA: "gaja",
    PieceType.ASHVA: "ashva",
    PieceType.PADATI: "padati",
}


class WebGame:
    """One game between a person and the agent, driven by the web page."""

    def __init__(self, network_fn, simulations, c_puct=1.5, checkpoint=None,
                 player="human", log=None, avoid_repetition=True, reuse_tree=True):  # fmt: skip
        self.network_fn = network_fn
        self.simulations = simulations
        self.c_puct = c_puct
        self.checkpoint = checkpoint
        self.player = player
        self.log = log
        self.avoid_repetition = avoid_repetition
        # Keep the search tree between the agent's moves (mcts/reuse.py).
        self.tree_cache = TreeCache() if reuse_tree else None
        self.new(Color.WHITE)

    # ---- game flow --------------------------------------------------

    def new(self, human_color):
        if self._in_progress():
            self._record("unfinished", "quit")
        self.human_color = human_color
        if self.tree_cache is not None:
            self.tree_cache.clear()
        self.state = GameState()
        self.moves = []
        self.last_move = None
        self.thinking = None  # what the agent's last search found
        self.finished = None  # (result, ending) once the game is over

    def human_move(self, text):
        if self.finished or self.state.side_to_move != self.human_color:
            raise ValueError("not your turn")
        move = next((m for m in self.state.legal_moves()
                     if move_to_algebraic(m) == text), None)  # fmt: skip
        if move is None:
            raise ValueError(f"illegal move {text}")
        self._play(move)

    def ai_move(self):
        if self.finished or self.state.side_to_move == self.human_color:
            raise ValueError("not the agent's turn")
        stats = {}
        visit_counts, move = drive(
            mcts_search(
                root_state=self.state,
                num_simulations=self.simulations,
                c_puct=self.c_puct,
                state_to_tensor_fn=encoding.state_to_tensor,
                legal_action_mask_fn=encoding.legal_action_mask,
                action_to_move_fn=encoding.action_to_move,
                stats=stats,
                root=self.tree_cache.take(self.state) if self.tree_cache else None,
            ),
            self.network_fn,
        )
        if self.avoid_repetition:
            visit_counts, _ = filter_repeating_moves(self.state, visit_counts)
            move = encoding.action_to_move(int(np.argmax(visit_counts)))
        if self.tree_cache is not None and "root" in stats:
            self.tree_cache.keep(stats["root"], encoding.move_to_action(move))
        total = float(np.sum(visit_counts)) or 1.0
        top = np.argsort(visit_counts)[::-1][:3]
        self.thinking = {
            "value": float(stats.get("root_value", 0.0)),
            "top": [
                {"move": move_to_algebraic(encoding.action_to_move(int(a))),
                 "share": float(visit_counts[a]) / total}  # fmt: skip
                for a in top if visit_counts[a] > 0
            ],
        }
        self._play(move)

    def resign(self):
        if self.finished:
            return
        self.finished = ("agent_win", "resigned")
        self._record(*self.finished)

    # ---- helpers ----------------------------------------------------

    def _in_progress(self):
        return getattr(self, "moves", None) and not self.finished

    def _play(self, move):
        self.state = self.state.apply_move(move)
        self.moves.append(move_to_algebraic(move))
        self.last_move = move_to_algebraic(move)
        if self.state.is_game_over():
            ending = classify_ending(self.state)
            if self.state.result() == 0:
                result = "draw"
            else:
                winner = self.state.side_to_move.opponent
                result = "human_win" if winner == self.human_color else "agent_win"
            self.finished = (result, ending)
            self._record(result, ending)

    def _record(self, result, ending):
        if not self.log:
            return
        append_game(
            self.log,
            player=self.player,
            checkpoint=self.checkpoint or "stub network",
            simulations=self.simulations,
            human_color=self.human_color.name.lower(),
            result=result,
            ending=ending,
            plies=len(self.moves),
            moves=" ".join(self.moves),
        )

    def view(self):
        """Everything the page needs to draw the position, as plain data."""
        board = []
        for sq in range(64):
            piece = self.state.board.piece_at(sq)
            board.append(None if piece is None else
                         [PIECE_NAMES[piece.piece_type], piece.color.name.lower()])  # fmt: skip
        legal = {}
        if not self.finished and self.state.side_to_move == self.human_color:
            for m in self.state.legal_moves():
                legal.setdefault(square_to_algebraic(m[0]), []).append(
                    square_to_algebraic(m[1]))
        summary = None
        if self.log and Path(self.log).exists():
            summary = format_summary(summarize(self.log, self.checkpoint))
        return {
            "board": board,
            "human_color": self.human_color.name.lower(),
            "side_to_move": self.state.side_to_move.name.lower(),
            "in_check": (not self.finished) and self.state.is_in_check(),
            "legal": legal,
            "last_move": self.last_move,
            "moves": self.moves,
            "no_progress": self.state.halfmove_clock,
            "thinking": self.thinking,
            "finished": None if not self.finished else
                        {"result": self.finished[0], "ending": self.finished[1]},  # fmt: skip
            "simulations": self.simulations,
            "checkpoint": self.checkpoint or "stub network",
            "player": self.player,
            "summary": summary,
        }


def make_handler(game, lock):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # keep the terminal quiet

        def _send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                with lock:
                    self._send(200, game.view())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                body = {}
            with lock:
                try:
                    if self.path == "/api/new":
                        color = Color.BLACK if body.get("color") == "black" else Color.WHITE
                        game.new(color)
                    elif self.path == "/api/move":
                        game.human_move(str(body.get("move", "")))
                    elif self.path == "/api/ai":
                        game.ai_move()
                    elif self.path == "/api/resign":
                        game.resign()
                    else:
                        return self._send(404, {"error": "not found"})
                except ValueError as err:
                    return self._send(400, {"error": str(err), **game.view()})
                self._send(200, game.view())

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description="Play Chaturanga against the agent in the browser")
    p.add_argument("--checkpoint", default="models/chaturangzero_final.pt",
                   help="trained model to play against ('stub' for random play)")  # fmt: skip
    p.add_argument("--simulations", type=int, default=1600,
                   help="search steps per agent move (1600: about half a second)")  # fmt: skip
    p.add_argument("--c_puct", type=float, default=1.5)
    p.add_argument("--no_tree_reuse", action="store_true",
                   help="start every search from nothing instead of keeping the last tree")
    p.add_argument("--player", default="human", help="your name, for the game log")
    p.add_argument("--log", default="results/human_games.csv")
    p.add_argument("--no_log", action="store_true", help="do not record games")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no_browser", action="store_true", help="do not open a browser")
    args = p.parse_args(argv)

    if args.checkpoint == "stub":
        network_fn, checkpoint = partial(random_stub_network, rng=np.random.default_rng()), None
    else:
        from evaluate_vs_random import make_real_network_fn

        network_fn, checkpoint = make_real_network_fn(checkpoint=args.checkpoint), args.checkpoint
    game = WebGame(network_fn, args.simulations, args.c_puct, checkpoint,
                   args.player, None if args.no_log else args.log,
                   reuse_tree=not args.no_tree_reuse)  # fmt: skip
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game, threading.Lock()))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"ChaturangZero is ready at {url}  (press Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if game._in_progress():
            game._record("unfinished", "quit")
        server.server_close()


if __name__ == "__main__":
    main()
