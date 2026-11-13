import sys, json, random, math
sys.path.insert(0, '.')
import numpy as np
from chaturanga.game import GameState
from chaturanga import encoding
from evaluation.game_endings import classify_ending
from mcts.search import drive, mcts_search
from training.self_play import filter_repeating_moves
out = {'games': [], 'search': []}
random.seed(11)
for g in range(300):
    s = GameState(); plies = []; moves = []
    for ply in range(400):
        legal = sorted(encoding.move_to_action(m) for m in s.legal_moves())
        plies.append([legal, s.result(), bool(s.is_in_check()), classify_ending(s) if s.is_game_over() else None, s.halfmove_clock])
        if s.is_game_over(): break
        # bias toward captures and repetitions to hit every rule
        ms = s.legal_moves()
        m = random.choice(ms)
        moves.append(list(m)); s = s.apply_move(m)
    out['games'].append({'moves': moves, 'plies': plies})
def fake(t):
    x = t.reshape(13, 64).T.reshape(-1)  # NHWC order, like JS
    h = float(np.dot(x.astype(np.float64), np.arange(x.size, dtype=np.float64) % 97))
    logits = np.array([math.sin(a * 0.37 + h * 0.001) * 3 for a in range(4096)], dtype=np.float32)
    return logits, float(np.float32(math.tanh(math.sin(h * 0.013))))
random.seed(5)
for k in range(8):
    s = GameState()
    for _ in range(random.randint(0, 40)):
        if s.is_game_over(): break
        s = s.apply_move(random.choice(s.legal_moves()))
    hist = []
    st = GameState()
    stats = {}
    vc, mv = drive(mcts_search(s, 300, 1.5, encoding.state_to_tensor, encoding.legal_action_mask, encoding.action_to_move, stats=stats), fake)
    f, _ = filter_repeating_moves(s, vc)
    out['search'].append({'seq_len': len(s.history), 'visits': {int(a): int(vc[a]) for a in np.flatnonzero(vc)}, 'choice': int(np.argmax(f)), 'root_value': stats['root_value']})
# store the move sequences used to reach each search position
random.seed(5); seqs = []
for k in range(8):
    s = GameState(); seq = []
    for _ in range(random.randint(0, 40)):
        if s.is_game_over(): break
        m = random.choice(s.legal_moves()); seq.append(list(m)); s = s.apply_move(m)
    seqs.append(seq)
for o, sq in zip(out['search'], seqs): o['seq'] = sq
json.dump(out, open(sys.argv[1], 'w'))
print('games', len(out['games']), 'plies', sum(len(g['plies']) for g in out['games']))
