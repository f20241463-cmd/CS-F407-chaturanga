"""Average the weights of several 4x64 checkpoints into one (a "model soup")."""
import sys, torch
from network.checkpoint import load_checkpoint, save_checkpoint

out, paths = sys.argv[1], sys.argv[2:]
model, _, _, _ = load_checkpoint(paths[0])
states = [torch.load(p, map_location="cpu")["model_state_dict"] for p in paths]
avg = {}
for k, v in states[0].items():
    if v.is_floating_point():
        avg[k] = sum(s[k] for s in states) / len(states)
    else:
        avg[k] = v  # batch-norm step counters
model.load_state_dict(avg)
save_checkpoint(out, model, metadata={"soup_of": paths})
print("averaged", len(paths), "checkpoints into", out)
