"""
training/self_play.py -- generate training data by having MCTS play itself.

One call to play_self_play_game plays a full game where both sides use
MCTS guided by the same network, with Dirichlet noise at the root so
games differ from each other. Every position becomes one training
example:

    (state_tensor, policy_target, value)

    policy_target = MCTS visit counts at that position, normalized to sum to 1
    value         = the final game result from the point of view of the
                    player who was to move at that position (+1 won,
                    -1 lost, draw_value for a draw or an unfinished game)

No torch import: network_fn is any callable honoring the
tensor -> (policy_logits, value) contract, so this also runs with the
stub networks. self_play_game_steps is the same game as a generator that
yields each position the network must evaluate, so many games can share
one batched network call (training/batched_self_play.py).
"""

import numpy as np

from chaturanga import encoding
from chaturanga.game import GameState
from chaturanga.types import Color, PieceType
from mcts.gumbel import gumbel_search
from mcts.search import drive, mcts_search, select_action_from_visits

# Rough piece values for material adjudication. Mantri (one step
# diagonally) and gaja (two-square diagonal jump) are weak short-range
# pieces in Chaturanga, far below a modern queen or bishop.
PIECE_VALUES = {
    PieceType.RAJA: 0.0,
    PieceType.MANTRI: 1.5,
    PieceType.RATHA: 5.0,
    PieceType.GAJA: 1.5,
    PieceType.ASHVA: 3.0,
    PieceType.PADATI: 1.0,
}


def filter_repeating_moves(state, visit_counts):
    """
    Return (visit_counts, avoided): a copy of visit_counts with every
    visited move that would cause a threefold-repetition draw set to 0.
    If every visited move would repeat, the counts come back unchanged so
    a move can still be chosen. avoided is True if anything was removed.
    Uses only the rules of the game (repetition detection), no game knowledge.
    """
    filtered = np.array(visit_counts, dtype=np.float32)
    for action in np.flatnonzero(filtered):
        child = state.apply_move(encoding.action_to_move(int(action)))
        # GameState has no public repetition check; result() == 0 alone
        # would also match stalemate and the no-progress rule.
        if child._is_threefold_repetition():
            filtered[action] = 0.0
    if filtered.sum() == 0:
        return np.asarray(visit_counts, dtype=np.float32), False
    return filtered, bool((filtered != visit_counts).any())


def _tagged(key, steps):
    """Pass a search's requests on as (key, request), and the answers back."""
    try:
        request = next(steps)
        while True:
            request = steps.send((yield key, request))
    except StopIteration as stop:
        return stop.value


def material_balance(state, color):
    """Material of color minus material of its opponent, using PIECE_VALUES."""
    mine = sum(PIECE_VALUES[p.piece_type] for _, p in state.board.pieces_of(color))
    theirs = sum(
        PIECE_VALUES[p.piece_type] for _, p in state.board.pieces_of(color.opponent)
    )
    return mine - theirs


def material_values(movers, final_state, material_scale):
    """
    Value targets for a drawn or unfinished game, from the material left
    on the board at the end: tanh(balance / material_scale), from each
    position's mover's perspective. A rook up with scale 5 gives about
    +0.76 for that side and -0.76 for the other; equal material gives 0.
    """
    white_value = float(
        np.tanh(material_balance(final_state, Color.WHITE) / material_scale)
    )
    return [white_value if mover == Color.WHITE else -white_value for mover in movers]


def outcome_values(movers, result, final_side_to_move):
    """
    Turn one final result into a value for every recorded position.

    result is from final_side_to_move's perspective (GameState.result()
    convention). A position where that same side was to move gets
    result; a position where the other side was to move gets -result.
    """
    return [
        float(result) if mover == final_side_to_move else -float(result)
        for mover in movers
    ]


def play_self_play_game(
    network_fn,
    num_simulations=30,
    c_puct=1.5,
    temperature_moves=30,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    max_plies=300,
    draw_value=0.0,
    rng=None,
    material_scale=None,
    avoid_repetition=False,
    resign_threshold=None,
    resign_after=2,
    resign_disabled=False,
    value_trace=None,
    q_mix=0.0,
    **options,
):
    """
    Returns (examples, info).

    examples: list of (state_tensor float32[13,8,8], policy float32[4096], value float)
    info:     {"plies": int, "outcome": "white" | "black" | "draw" | "unfinished"}

    For the first temperature_moves plies the move is sampled in
    proportion to visit counts (temperature 1); after that the most
    visited move is played. The policy target is always the normalized
    visit counts, whichever way the move was chosen.

    material_scale=None (default): drawn and unfinished games give every
    position draw_value. If set (for example 5.0), they instead get
    material_values(...), so the value head still learns that being
    ahead in material is good when few games end decisively. Decisive
    games always use the real +1/-1 result.

    avoid_repetition=False (default): unchanged. If True, moves that would
    cause a threefold-repetition draw are dropped from the visit counts
    (unless every visited move would repeat) before the move is chosen
    and before the policy target is made, so the network learns the moves
    the agent actually plays. Most self-play games otherwise end in
    repetition draws, which leaves the value head almost nothing but 0
    targets. This needs only the rules, no game knowledge.

    resign_threshold=None (default): no resignation. If set (for example
    0.9), a side whose search rates its own position below -resign_threshold
    on resign_after of its moves in a row resigns, and the game counts as
    its loss. Hopeless games then end early with a real result instead of
    dragging on to a draw by the 100-ply rule, so the value head sees more
    wins and losses. It uses only the agent's own evaluation, no game
    knowledge (AlphaZero does the same). With resign_disabled=True the game
    is played out anyway, which measures how often a resignation would have
    been wrong; info then also says which side would have resigned
    ("would_resign": "white", "black" or None).

    value_trace: optional list; (side to move, root value) is appended for
    every move, for analysis.

    q_mix: 0 (default) uses only the game result as each position's value
    target. With q_mix = w > 0 the target is (1 - w) x result + w x the
    search's own value of that position (the root value, from the mover's
    point of view). A drawn game then still says which positions the
    search found good or bad. It uses only the agent's own search, no game
    knowledge.

    options: start_moves, fast_simulations, full_fraction, record_moves; see
    self_play_game_steps.
    """
    return drive(
        self_play_game_steps(
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature_moves=temperature_moves,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            max_plies=max_plies,
            draw_value=draw_value,
            rng=rng,
            material_scale=material_scale,
            avoid_repetition=avoid_repetition,
            resign_threshold=resign_threshold,
            resign_after=resign_after,
            resign_disabled=resign_disabled,
            value_trace=value_trace,
            q_mix=q_mix,
            **options,
        ),
        network_fn,
    )


def self_play_game_steps(
    num_simulations=30,
    c_puct=1.5,
    temperature_moves=30,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    max_plies=300,
    draw_value=0.0,
    rng=None,
    material_scale=None,
    avoid_repetition=False,
    resign_threshold=None,
    resign_after=2,
    resign_disabled=False,
    value_trace=None,
    q_mix=0.0,
    start_moves=None,
    fast_simulations=None,
    full_fraction=1.0,
    record_moves=False,
    opponent_color=None,
    gumbel=False,
    gumbel_actions=16,
):
    """
    play_self_play_game as a generator: yields a state tensor whenever the
    search needs the network, expects (policy_logits, value) sent back,
    and returns (examples, info). Arguments as in play_self_play_game, plus:

    start_moves: optional list of action ids. The game starts from the
    position after these moves (for example a late position of an earlier
    self-play game) instead of the opening; they count towards max_plies
    but are not recorded as examples.

    fast_simulations, full_fraction: with fast_simulations set, each move
    gets a full search (num_simulations, root noise, recorded as an
    example) with probability full_fraction, and otherwise a fast search
    (fast_simulations, no root noise, not recorded). More moves per hour,
    and every training example still comes from a full search
    ("playout cap randomization", from KataGo).

    record_moves: if True, info["moves"] is the list of action ids of the
    whole game (used to start later games from its late positions).

    opponent_color: None (default) for normal self-play. If set, the
    generator yields ("old", tensor) for that side's searches and
    ("cur", tensor) for the other side's, so two networks can play each
    other (training.batched_self_play.play_keyed_games_batched), and only
    the "cur" side's moves become examples.

    gumbel: False (default) uses the AlphaZero root (Dirichlet noise, visit
    counts as the policy target, moves sampled at the temperature). If True,
    full searches use mcts.gumbel.gumbel_search instead: the move is the
    one its Sequential Halving picks among gumbel_actions sampled moves,
    and the policy target is its improved policy. No Dirichlet noise or
    temperature: the Gumbel sampling is the exploration.
    """
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")
    rng = rng if rng is not None else np.random.default_rng()

    state = GameState()
    moves = []
    for action in start_moves or ():
        state = state.apply_move(encoding.action_to_move(int(action)))
        moves.append(int(action))
    records = []  # (tensor, policy, side_to_move) per ply
    root_values = []  # the search's value of each recorded position
    plies = len(moves)
    low_streak = {Color.WHITE: 0, Color.BLACK: 0}
    would_resign = None  # first side whose evaluation crossed the threshold
    resigned = None

    while not state.is_game_over() and plies < max_plies:
        full = fast_simulations is None or rng.random() < full_fraction
        use_gumbel = gumbel and full and opponent_color is None
        stats = {}
        search = gumbel_search(
            state, num_simulations, c_puct, encoding.state_to_tensor, encoding.legal_action_mask,
            encoding.action_to_move, rng, max_considered=gumbel_actions, stats=stats,
        ) if use_gumbel else mcts_search(
            state,
            num_simulations if full else fast_simulations,
            c_puct,
            encoding.state_to_tensor,
            encoding.legal_action_mask,
            encoding.action_to_move,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps if full else 0.0,
            rng=rng,
            stats=stats,
        )
        if opponent_color is None:
            visit_counts, gumbel_action = yield from search
        else:
            old = state.side_to_move == opponent_color
            visit_counts, _ = yield from _tagged("old" if old else "cur", search)
            full = full and not old
        if avoid_repetition:
            visit_counts, _ = filter_repeating_moves(state, visit_counts)
        if not full:
            temperature = 1.0 if plies < temperature_moves else 0.0
            action = select_action_from_visits(visit_counts, temperature, rng)
            state = state.apply_move(encoding.action_to_move(action))
            moves.append(int(action))
            plies += 1
            continue
        policy = (visit_counts / visit_counts.sum()).astype(np.float32)
        mover = state.side_to_move
        records.append((encoding.state_to_tensor(state), policy, mover))
        root_values.append(float(stats["root_value"]))
        if value_trace is not None:
            value_trace.append((mover, stats["root_value"]))

        if resign_threshold is not None:
            if stats["root_value"] < -resign_threshold:
                low_streak[mover] += 1
            else:
                low_streak[mover] = 0
            if low_streak[mover] >= resign_after and would_resign is None:
                would_resign = mover
                if not resign_disabled:
                    resigned = mover
                    break

        if use_gumbel:
            # The Gumbel choice, unless repetition avoidance removed it.
            action = gumbel_action if visit_counts[gumbel_action] > 0 else int(np.argmax(visit_counts))
        else:
            temperature = 1.0 if plies < temperature_moves else 0.0
            action = select_action_from_visits(visit_counts, temperature, rng)
        state = state.apply_move(encoding.action_to_move(action))
        moves.append(int(action))
        plies += 1

    movers = [mover for _, _, mover in records]
    result = state.result() if state.is_game_over() else None
    if resigned is not None:
        # A loss for the side that resigned (it was to move at the end).
        values = outcome_values(movers, -1.0, resigned)
        outcome = "white" if resigned.opponent == Color.WHITE else "black"
    elif result is None or result == 0:
        outcome = "unfinished" if result is None else "draw"
        if material_scale is not None:
            values = material_values(movers, state, material_scale)
        else:
            values = [float(draw_value)] * len(records)
    else:
        values = outcome_values(movers, result, state.side_to_move)
        winner = state.side_to_move if result > 0 else state.side_to_move.opponent
        outcome = "white" if winner == Color.WHITE else "black"

    if q_mix:
        values = [(1.0 - q_mix) * z + q_mix * q for z, q in zip(values, root_values)]
    examples = [
        (tensor, policy, value) for (tensor, policy, _), value in zip(records, values)
    ]
    info = {"plies": plies, "outcome": outcome}
    if record_moves:
        info["moves"] = moves
    if resign_threshold is not None:
        info["resigned"] = resigned is not None
        info["would_resign"] = (
            None if would_resign is None
            else "white" if would_resign == Color.WHITE else "black"
        )  # fmt: skip
    return examples, info


def reanalyse_game_steps(moves, num_simulations=50, c_puct=1.5, draw_value=0.0,
                         avoid_repetition=False):  # fmt: skip
    """
    Reanalyse (from MuZero): replay an earlier self-play game, given as its
    list of action ids, and search every position again with the current
    network (no root noise). Returns (examples, info) like
    self_play_game_steps: the policy targets are the new search's visit
    counts, the value targets the game's original result. A generator in
    the same way, so it can share batched network calls with self-play.
    """
    states = [GameState()]
    for action in moves:
        states.append(states[-1].apply_move(encoding.action_to_move(int(action))))
    final = states[-1]
    records = []
    for state in states[:-1]:
        visit_counts, _ = yield from mcts_search(
            state, num_simulations, c_puct, encoding.state_to_tensor,
            encoding.legal_action_mask, encoding.action_to_move,
        )
        if avoid_repetition:
            visit_counts, _ = filter_repeating_moves(state, visit_counts)
        policy = (visit_counts / visit_counts.sum()).astype(np.float32)
        records.append((encoding.state_to_tensor(state), policy, state.side_to_move))
    movers = [mover for _, _, mover in records]
    result = final.result() if final.is_game_over() else None
    if result is None or result == 0:
        values = [float(draw_value)] * len(records)
        outcome = "unfinished" if result is None else "draw"
    else:
        values = outcome_values(movers, result, final.side_to_move)
        winner = final.side_to_move if result > 0 else final.side_to_move.opponent
        outcome = "white" if winner == Color.WHITE else "black"
    examples = [(t, p, v) for (t, p, _), v in zip(records, values)]
    return examples, {"plies": len(moves), "outcome": outcome, "reanalysed": True}
