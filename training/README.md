# Training and evaluation (ChaturangZero)

How to train a network by self-play, measure it, and plot the results.
Everything runs from the repo root, on the CPU or the Apple GPU (`--device mps`). Results and their discussion are in
`results/README.md`.

## The training loop

`python -m training.train` runs the AlphaZero loop. Each iteration:

1. **Self-play.** The current network plays `--games_per_iter` games against
   itself. Every move comes from MCTS (`--simulations` per move, PUCT with
   `--c_puct`) with Dirichlet noise mixed into the root priors
   (`--dirichlet_alpha`, `--dirichlet_eps`). Moves are sampled in proportion to
   visit counts for the first `--temperature_moves` plies, then the most
   visited move is played. Each position becomes one training example: the
   encoded state, the MCTS visit distribution (policy target) and the final
   game result from the mover's point of view (value target: +1, -1, or
   `--draw_value` for draws and games cut off at `--max_plies`).
2. **Training.** Once the replay buffer (`--buffer_capacity` positions, oldest
   dropped first) holds `--min_buffer` positions, run `--train_steps` Adam
   steps on random minibatches of `--batch_size`. The loss is policy
   cross-entropy plus value MSE, with L2 through `--weight_decay`.
3. **Checkpoint.** Save `iter_NNNN.pt` and `latest.pt` in `--checkpoint_dir`
   and append one row to the CSV `--log`.

The command used for the final runs (20 iterations x 20 games, about 7 hours
each on one CPU core):

```
OMP_NUM_THREADS=1 python -m training.train --iterations 20 --games_per_iter 20 \
    --simulations 50 --avoid_repetition --temperature_moves 1000 \
    --max_plies 800 --train_steps 400 \
    --checkpoint_dir checkpoints/final_B --log checkpoints/final_B/log.csv
```

`OMP_NUM_THREADS=1` matters when running several jobs at once: the network is
small, so extra torch threads only compete with each other.

### Faster runs: parallel self-play and GPU training

```
OMP_NUM_THREADS=1 python -m training.train --resume checkpoints/final_B/latest.pt \
    --iterations 220 --games_per_iter 100 --simulations 50 --avoid_repetition \
    --temperature_moves 1000 --max_plies 800 --buffer_capacity 250000 \
    --batch_size 256 --train_steps 500 --device mps --workers 6 --seed 1 \
    --checkpoint_dir checkpoints/long --log checkpoints/long/log.csv
```

- `--workers N` plays the iteration's games in N processes (about one per CPU
  core). Each game's seed comes from (`--seed`, iteration, game number), so the
  games do not depend on N. Workers always run on the CPU.
- `--device mps` trains on the Apple GPU, about 8 to 10 times faster than one
  CPU core for this network. Self-play still runs on the CPU workers.

Where the speed comes from, measured on one game with 50 simulations per move
(an Apple A18 Pro, 2 performance and 4 efficiency cores):

| Change | ms per move |
|---|---|
| Before | 259 |
| Faster rules engine: legal moves computed once per position and cached, check detection that looks outward from the king, precomputed move tables | 48 |
| Network copy for search with batch norm folded into the convolutions and channels-last memory layout | 19 |

With 6 workers that is about 13 self-play games a minute, against about 1.2
before. The rules are unchanged: `tests/test_engine_equivalence.py` compares
the fast engine with a frozen copy of the original (`tests/reference_rules.py`)
on every ply of 40 random games and on 3000 random positions.

To continue a run, pass `--resume checkpoints/final_B/latest.pt` and a larger
`--iterations` (the total to reach). The model and optimizer are restored;
the replay buffer starts empty. The `--lr` given on the command line replaces
the saved one, so a resumed run can continue at a lower learning rate.

The final model came from this command run to iteration 420 (with one restart
at 220), followed by three resumed stretches with the same options except:

| Iterations | Resume from | `--lr` | `--seed` | `--checkpoint_dir` |
|---|---|---|---|---|
| 421 to 460 | `checkpoints/long/iter_0420.pt` | 1e-4 | 1 | `checkpoints/long_lowlr` |
| 461 to 500 | `checkpoints/long_lowlr/iter_0460.pt` | 1e-4 | 3 | `checkpoints/small_more` |
| 501 to 560 | `checkpoints/small_more/iter_0500.pt` | 3e-5 | 4 | `checkpoints/small_more2` |

Each resumed stretch used `--min_buffer 150000`, so the first two iterations
only refill the replay buffer before training starts again. Iteration 560 is
saved in the repo as `models/chaturangzero_final.pt`.

### A bigger network, started from an earlier agent's games

```
OMP_NUM_THREADS=1 python -m training.train --res_blocks 6 --filters 128 \
    --bootstrap_model checkpoints/long/iter_0300.pt --bootstrap_iters 5 \
    --iterations 300 --games_per_iter 100 --simulations 50 --avoid_repetition \
    --temperature_moves 1000 --max_plies 800 --buffer_capacity 250000 \
    --min_buffer 150000 --batch_size 256 --train_steps 500 --device mps \
    --workers 6 --seed 2 --checkpoint_dir checkpoints/big --log checkpoints/big/log.csv
```

- `--res_blocks` (4 to 6) and `--filters` (64 to 128) set the size of a new
  network. 6 blocks of 128 filters is the largest the official plan allows,
  about 4 times the parameters of the default 4 x 64. With `--resume` the
  checkpoint's size is kept.
- `--bootstrap_model` and `--bootstrap_iters N` let an earlier checkpoint play
  the self-play games of iterations 1 to N while the new network trains on
  them. A new network then starts from games of a trained self-play agent
  instead of random play. From iteration N + 1 on it plays its own games as
  usual. Only our own self-play agent is used, no outside knowledge.

### Options worth knowing

| Option | Default | Notes |
|---|---|---|
| `--simulations` | 30 | The official plan allows 30 to 50. |
| `--temperature_moves` | 1000 | Sample from visit counts for this many plies. 1000 means the whole game. |
| `--avoid_repetition` | off | Never pick a move that causes a threefold-repetition draw unless every considered move does. Rules only, so it fits the no-domain-knowledge requirement. Used in the final runs. |
| `--max_plies` | 300 | Games longer than this stop and count as unfinished. The final runs used 800. |
| `--material_scale` | off | Value targets from material balance for drawn games. Uses piece values, which is human domain knowledge, so it is **outside the official plan** and kept only as a labeled experiment. |
| `--draw_value` | 0.0 | Value target for drawn and unfinished games. |

## Evaluating checkpoints

### Benchmark every iteration against a fixed opponent

```
python -m evaluation.benchmark --checkpoint_dir checkpoints/final_B --opponent untrained --every 4
python -m evaluation.benchmark --checkpoint_dir checkpoints/final_B --opponent random
python -m evaluation.benchmark --checkpoint_dir checkpoints/final_B --opponent greedy
```

Opponents: `random` (uniform random legal move), `greedy` (captures the most
valuable piece it can, otherwise random) and `untrained` (the same MCTS agent
with a freshly initialized network). Iteration 0 is the untrained network
itself (left out against `untrained`). One CSV row per iteration is appended to
`--out` (default `results/benchmark_<opponent>.csv`).

`--workers N` plays the games in N processes, with one seed per game, so
`--games 100` takes minutes instead of hours; `--iterations 0 100 220` picks
iterations. Rows made with `--workers` use per-game seeds and rows made
without it share one random stream, so compare rows made the same way.

The protocol is fixed so that iterations are comparable:

- 20 games per iteration, colors alternating, the same seed for every iteration
- the agent uses 50 simulations, no root noise, temperature 0.3 for the whole
  game, and repetition avoidance
- games stop at 800 plies; unfinished games are reported but left out of the score
- score = (wins + 0.5 x draws) / finished games, with a rough 95% margin

### Two checkpoints head to head

```
python compare_checkpoints.py --a checkpoints/final_B/latest.pt --b checkpoints/final_A/latest.pt \
    --games 20 --simulations 50 --temp_plies 1000 --temperature 0.3 \
    --avoid_repetition --max_plies 800
```

Use `--b untrained` to compare against a fresh network. The same match under
the benchmark protocol, on several cores:

```
python -m evaluation.parallel_match --a checkpoints/long/latest.pt \
    --b checkpoints/final_B/latest.pt --games 100 --workers 6
```

Keep `--avoid_repetition` on: with the defaults (sampling for only 8 plies, then
greedy), two deterministic players repeat positions and nearly every game ends
in a threefold-repetition draw, which says nothing about strength.

### How games end

```
python -m evaluation.game_endings --checkpoint checkpoints/final_B/latest.pt --opponent self --games 10
```

Plays games under the benchmark protocol and counts which rule ended each one
(checkmate, stalemate, the 100-ply no-progress rule, threefold repetition, or
the ply cap). Opponents: `random`, `greedy`, `untrained` or `self`.

### Against random

```
python evaluate_vs_random.py --checkpoint checkpoints/final_B/latest.pt
```

### Against a human

```
python play_vs_ai.py --checkpoint checkpoints/final_B/latest.pt --player YOUR_NAME
python play_vs_ai.py --checkpoint checkpoints/final_B/latest.pt --player YOUR_NAME --human_color black
python play_vs_ai.py --summary --checkpoint checkpoints/final_B/latest.pt
```

The AI plays like the benchmark agent: 50 simulations and repetition
avoidance, always choosing its most visited move. Enter moves as `e2e4`; type
`legal` to list legal moves, `resign` to give up (an agent win) or `quit` to
stop (unfinished). Every game is appended to `results/human_games.csv` with the
result, the rule that ended it and the full move list (`--no_log` to skip).
`--summary` prints the agent's score against humans, scored like the benchmark
(win 1, draw 0.5, unfinished left out), with a 95% Wilson range because the
number of human games is small. Alternate colors between games.

## Plotting

```
python plot_training_curves.py --logs results/training_log_finalA.csv results/training_log_finalB.csv \
    --labels "Final A" "Final B" --out results/training_curves.png
python plot_benchmark.py --csvs results/benchmark_finalB_untrained.csv results/benchmark_finalB_random.csv \
    --out results/benchmark_curve.png
```

`plot_training_curves.py` shows policy loss, value loss, decisive-game share
and average game length per iteration (up to 3 runs). `plot_benchmark.py`
shows benchmark score per iteration with its margin, against a 0.5 line.

## Tests

```
python -m pytest -q
```

`tests/test_training.py` covers the replay buffer, self-play examples and
value targets, repetition avoidance, one training step and a tiny end-to-end
run of `train.py` including resume, and that parallel self-play gives the same
run for any number of workers, and that a bigger network can train on games
played by a smaller bootstrap checkpoint. `tests/test_benchmark.py` and
`tests/test_evaluation.py` cover the baseline players, match scoring, the
benchmark CSV (also with workers), the game-ending classifier and the
human-game log. `tests/test_engine_equivalence.py` checks the fast rules
engine against the original one.
