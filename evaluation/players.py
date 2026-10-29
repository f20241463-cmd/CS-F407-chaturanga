"""
evaluation/players.py -- the players a benchmark can put on the board.

A player is any function (state, rng) -> move, where rng is a
np.random.Generator. Three kinds:

  random_player          uniformly random legal move (the weakest baseline)
  greedy_capture_player  a simple heuristic baseline: captures the most
                         valuable piece it can, otherwise moves at random.
                         It uses piece values, which is fine for an
                         OPPONENT; the agent itself never sees them.
  make_mcts_player(...)  MCTS guided by a network_fn (the agent)

play_game / play_match run games between two players with colors
alternating, so neither side gets the first-move edge every time.
"""

import numpy as np

from chaturanga import encoding
from chaturanga.game import GameState
from chaturanga.types import Color
from mcts.search import drive, mcts_search, select_action_from_visits
from training.self_play import PIECE_VALUES, filter_repeating_moves  # noqa: F401


def random_player(state, rng):
    moves = state.legal_moves()
    return moves[int(rng.integers(len(moves)))]


def greedy_capture_player(state, rng):
    """Capture the most valuable enemy piece available (ties broken at random); else move at random."""
    best_value, best_moves = 0.0, []
    for move in state.legal_moves():
        target = state.board.piece_at(move[1])
        if target is None:
            continue
        value = PIECE_VALUES[target.piece_type]
        if value > best_value:
            best_value, best_moves = value, [move]
        elif value == best_value:
            best_moves.append(move)
    if best_moves:
        return best_moves[int(rng.integers(len(best_moves)))]
    return random_player(state, rng)


def mcts_move_steps(state, rng, simulations=50, c_puct=1.5, temperature=0.3,
                    avoid_repetition=True):  # fmt: skip
    """
    The MCTS player's move choice as a generator: yields each position the
    network must evaluate, takes (policy_logits, value) back, and returns
    the move. make_mcts_player drives it one position at a time; batched
    matches drive many at once.
    """
    visit_counts, _ = yield from mcts_search(
        state,
        simulations,
        c_puct,
        encoding.state_to_tensor,
        encoding.legal_action_mask,
        encoding.action_to_move,
    )
    if avoid_repetition:
        visit_counts, _ = filter_repeating_moves(state, visit_counts)
    action = select_action_from_visits(visit_counts, temperature, rng)
    return encoding.action_to_move(action)


def make_mcts_player(
    network_fn, simulations=50, c_puct=1.5, temperature=0.3, avoid_repetition=True
):
    """
    MCTS player with no Dirichlet noise. The move is sampled from the
    visit counts at `temperature` (0 = always the most visited move).
    avoid_repetition skips moves that would cause a threefold-repetition
    draw unless every considered move does; without it, near-greedy play
    between these networks ends most games in a quick repetition draw.

    The player also carries .steps(state, rng), the same move choice as a
    generator (mcts_move_steps), for batched matches.
    """

    def steps(state, rng):
        return mcts_move_steps(state, rng, simulations, c_puct, temperature,
                               avoid_repetition)  # fmt: skip

    def player(state, rng):
        return drive(steps(state, rng), network_fn)

    player.steps = steps
    return player


def _tagged(key, steps):
    """Pass a generator's requests on as (key, request), and the answers back."""
    try:
        request = next(steps)
        while True:
            request = steps.send((yield key, request))
    except StopIteration as stop:
        return stop.value


def play_game_steps(white, black, rng, max_plies=800, keys=("white", "black")):
    """
    play_game as a generator, for batched matches. A player with .steps
    (an MCTS player) yields (key, position) for each position its network
    must evaluate, with keys[0] for White and keys[1] for Black, and takes
    (policy_logits, value) back; other players move directly. Returns the
    same as play_game.
    """
    state = GameState()
    plies = 0
    while not state.is_game_over() and plies < max_plies:
        white_to_move = state.side_to_move == Color.WHITE
        player = white if white_to_move else black
        if hasattr(player, "steps"):
            key = keys[0] if white_to_move else keys[1]
            move = yield from _tagged(key, player.steps(state, rng))
        else:
            move = player(state, rng)
        state = state.apply_move(move)
        plies += 1
    return _game_outcome(state, plies)


def _game_outcome(state, plies):
    if not state.is_game_over():
        return "unfinished", plies
    result = state.result()
    if result == 0:
        return "draw", plies
    winner = state.side_to_move if result > 0 else state.side_to_move.opponent
    return ("white" if winner == Color.WHITE else "black"), plies


def play_game(white, black, rng, max_plies=800):
    """Returns ("white" | "black" | "draw" | "unfinished", plies)."""
    state = GameState()
    plies = 0
    while not state.is_game_over() and plies < max_plies:
        player = white if state.side_to_move == Color.WHITE else black
        state = state.apply_move(player(state, rng))
        plies += 1
    return _game_outcome(state, plies)


def play_match(player_a, player_b, games, rng, max_plies=800, on_game=None):
    """
    Play `games` games, A as White in even-numbered games. Returns a dict
    with A's wins/losses/draws/unfinished and the game lengths.
    on_game(game_index, a_is_white, outcome, plies) is called after each game.
    """
    counts = {"wins": 0, "losses": 0, "draws": 0, "unfinished": 0, "plies": []}
    for g in range(games):
        a_is_white = g % 2 == 0
        white, black = (player_a, player_b) if a_is_white else (player_b, player_a)
        outcome, plies = play_game(white, black, rng, max_plies)
        counts["plies"].append(plies)
        if outcome == "draw":
            counts["draws"] += 1
        elif outcome == "unfinished":
            counts["unfinished"] += 1
        elif (outcome == "white") == a_is_white:
            counts["wins"] += 1
        else:
            counts["losses"] += 1
        if on_game is not None:
            on_game(g, a_is_white, outcome, plies)
    return counts
