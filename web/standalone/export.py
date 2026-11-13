import sys, json, base64, copy, random
sys.path.insert(0, '.')
import numpy as np, torch
from network.checkpoint import load_checkpoint
from network.agent import _fuse_conv_bn, make_network_fn
from chaturanga.game import GameState
from chaturanga import encoding
OUT = sys.argv[1]
r = load_checkpoint('models/chaturangzero_final.pt', device='cpu')
model = r[0] if isinstance(r, tuple) else r
model.eval()
net = _fuse_conv_bn(copy.deepcopy(model).eval())
W = {}
def conv(name, m):  # torch (out,in,kh,kw) -> tfjs (kh,kw,in,out)
    W[name + '.w'] = m.weight.detach().numpy().transpose(2, 3, 1, 0)
    W[name + '.b'] = m.bias.detach().numpy()
conv('in', net.input_conv)
for i, b in enumerate(net.res_blocks):
    conv(f'r{i}a', b.conv1); conv(f'r{i}b', b.conv2)
conv('pc', net.policy_conv); conv('vc', net.value_conv)
def fc_nhwc(name, lin, C):  # torch (out, C*64) flatten NCHW -> (64*C in NHWC order, out)
    w = lin.weight.detach().numpy(); out = w.shape[0]
    W[name + '.w'] = w.reshape(out, C, 8, 8).transpose(2, 3, 1, 0).reshape(64 * C, out)
    W[name + '.b'] = lin.bias.detach().numpy()
fc_nhwc('pf', net.policy_fc, 2); fc_nhwc('v1', net.value_fc1, 1)
W['v2.w'] = net.value_fc2.weight.detach().numpy().T; W['v2.b'] = net.value_fc2.bias.detach().numpy()
enc = {k: [list(v.shape), base64.b64encode(np.ascontiguousarray(v, dtype='<f4').tobytes()).decode()] for k, v in W.items()}
# reference outputs on positions from a random game
fn = make_network_fn(model)
random.seed(3); s = GameState(); refs = []
for ply in range(60):
    if ply % 12 == 0:
        logits, v = fn(encoding.state_to_tensor(s))
        legal = [encoding.move_to_action(m) for m in s.legal_moves()]
        refs.append({'moves_from_start': ply, 'value': v, 'legal': legal, 'logits': [float(logits[a]) for a in legal]})
    if s.is_game_over(): break
    s = s.apply_move(random.choice(s.legal_moves()))
random.seed(3); seq = []; s = GameState()
for ply in range(60):
    m = random.choice(s.legal_moves()); seq.append(m); s = s.apply_move(m)
with open(OUT, 'w') as f:
    f.write('window.CZ_WEIGHTS=' + json.dumps(enc) + ';\nwindow.CZ_REFS=' + json.dumps({'seq': seq, 'refs': refs}) + ';\n')
print('params', sum(v.size for v in W.values()))
