# Training and evaluation (ChaturangZero)

How to train a network by self-play, measure it, and plot the results.
Everything runs from the repo root on CPU. Results and their discussion are in
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

To continue a run, pass `--resume checkpoints/final_B/latest.pt` and a larger
`--iterations` (the total to reach). The model and optimizer are restored;
the replay buffer starts empty.

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

Use `--b untrained` to compare against a fresh network. Keep
`--avoid_repetition` on: with the defaults (sampling for only 8 plies, then
greedy), two deterministic players repeat positions and nearly every game ends
in a threefold-repetition draw, which says nothing about strength.

### How games end

```
python -m evaluation.game_endings --checkpoint checkpoints/final_B/latest.pt --opponent self --games 10
```

Plays games under the benchmark protocol and counts which rule ended each one
(checkmate, stalemate, the 100-ply no-progress rule, threefold repetition, or
the ply cap). Opponents: `random`, `greedy`, `untrained` or `self`.

### Against random, or play it yourself

```
python evaluate_vs_random.py --checkpoint checkpoints/final_B/latest.pt
python play_vs_ai.py --checkpoint checkpoints/final_B/latest.pt
```

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
run of `train.py` including resume. `tests/test_benchmark.py` and
`tests/test_evaluation.py` cover the baseline players, match scoring, the
benchmark CSV and the game-ending classifier.
