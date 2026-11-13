# Standalone web version

A single HTML page that plays the final model (iteration 1220) entirely in the
browser, with no Python and no server. It was built so that a player in another
city could play against the agent from a Windows laptop with nothing installed.
The 14 games against iteration 1220 in `results/human_games.csv` were played on
it (Section 6.4 of the report).

It is a port, not a new agent:

- `engine.js`: the Chaturanga rules, the PUCT search (c_puct 1.5, tree reuse)
  and the move choice (most visited move, skipping moves that repeat a
  position three times), written to match `chaturanga/`, `mcts/` and
  `play_web.py`.
- `net.c` compiled to `net.wasm`: the 4x64 network with batch norm folded into
  the convolutions, in WebAssembly with SIMD (about 0.6 ms per position).
  `netjs.js` loads it, and also has a plain JavaScript version as a fallback.
- `export.py`: writes the weights of `models/chaturangzero_final.pt` to
  `weights.js`, with reference outputs the page checks itself against on load.
- `app.js`, `head.html`, `body.html`: the board, 800 simulations per move, and
  saving each finished game (date in UTC, colors, result, ending, moves).

## Checks

- `ref.py` writes `ref.json` (300 random games and 8 searches from the Python
  code); `node test.js` replays them in JavaScript. Result: 114,767 positions
  with the same legal moves, results and endings, 0 mismatches, and the same
  visit counts in all 8 searches.
- The page compares its network outputs with the Python ones when it loads
  (largest difference 5.7e-6, saved with every game as `net_check`).

## Build

From the repository root:

```bash
python web/standalone/export.py web/standalone/weights.js
# only if net.c changes (needs clang and lld, e.g. brew install llvm lld):
clang --target=wasm32 -O3 -msimd128 -ffast-math -fno-math-errno -nostdlib \
  -Wl,--no-entry -Wl,--export=get_w -Wl,--export=get_in -Wl,--export=get_out \
  -Wl,--export=get_off -Wl,--export=get_v1 -Wl,--export=init -Wl,--export=forward \
  -Wl,--initial-memory=8388608 -o web/standalone/net.wasm web/standalone/net.c
zsh web/standalone/build.sh   # writes chaturangzero.html and local.html (about 4.5 MB)
```

Open `local.html` in a browser to play offline (games then stay in the
browser and can be copied out). `chaturangzero.html` is the version that was
published as a claude.ai page, where finished games were saved to the page's
database.
