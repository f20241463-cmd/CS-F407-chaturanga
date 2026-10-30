# ChaturangZero: a self-play AlphaZero agent for Chaturanga

CS F407 course project. An agent that learns Chaturanga, the ancient Indian
ancestor of chess, by playing against itself. Like AlphaZero, it gets no human
knowledge beyond the rules: a residual network suggests moves and judges
positions, Monte Carlo Tree Search (MCTS) looks ahead with it, and the network
is trained on the games it plays.

Everything runs on a CPU; training can also use an Apple GPU (`--device mps`).
Python 3 with `numpy`, `torch` and `matplotlib`.

## How it fits together

```
chaturanga/   rules engine: board, legal moves, game end, tensor + action encoding
network/      dual-head ResNet (policy + value), loss, checkpoints
mcts/         guided MCTS with PUCT, root Dirichlet noise, lazy child states
training/     self-play games, replay buffer, training loop
evaluation/   baseline players, per-iteration benchmark, game endings, human-game log
results/      training logs, benchmark CSVs, plots, and the results write-up
models/       the final trained network (chaturangzero_final.pt)
tests/        123 automated tests
```

| Part | What it does | Details |
|---|---|---|
| Rules engine | 8x8 board, all piece moves, check, checkmate (loss), stalemate (draw), 100-ply no-progress and threefold-repetition draws. The state is encoded as 13 planes of 8x8; a move is `from * 64 + to` in a 4096-wide action space, with illegal moves masked out. | `chaturanga/README.md` |
| Network | Residual network, 4 blocks of 64 filters (about 836,000 parameters). Policy head: 4096 move logits. Value head: one number in [-1, 1] for the side to move. Loss: value MSE + policy cross-entropy, with L2 regularization. | `network/` |
| Search | MCTS guided by the network: PUCT selection, expansion with the network's move priors, leaf values from the value head (no random rollouts), backup with the sign flipped each ply. | `mcts/README.md` |
| Training | Self-play with MCTS (30 to 50 simulations per move), root noise for exploration, a replay buffer of recent positions, Adam updates, a checkpoint per iteration. | `training/README.md` |
| Evaluation | Fixed-protocol benchmark of every checkpoint against random, greedy-capture and untrained players; head-to-head matches between checkpoints; play against a human with automatic result logging; plots. | `training/README.md`, `results/README.md` |

## Quick start

```bash
python -m pytest -q                                  # run all tests
python play.py                                       # you vs a random player
python play_vs_ai.py --checkpoint models/chaturangzero_final.pt --player YOUR_NAME --simulations 800
python -m training.train --iterations 2 --games_per_iter 2 --checkpoint_dir checkpoints/demo --log checkpoints/demo/log.csv
python -m evaluation.benchmark --checkpoint_dir checkpoints/demo --opponent random --games 4 --out checkpoints/demo/bench.csv
```

The last two lines are a tiny training run (about 1 minute) and its benchmark
(about 5 minutes). Training checkpoints are not stored in git (`checkpoints/` is
ignored); only the final model is, in `models/`. The training commands and
every evaluation command are in `training/README.md`.

## Results in brief

**First runs.** Two runs of 20 iterations x 20 self-play games (400 games, about
7 hours each on one CPU core) did not make the agent measurably stronger than
an untrained network, a random player or a greedy-capture player, and a human
new to chess beat it. About 80% of games were drawn, and 400 games was far too
little data.

**Long run (final model).** After making self-play about 11 times faster, we
continued training for 540 more iterations of 100 games (54,000 games, about
144 hours on one laptop), lowering the learning rate for the last 140
iterations. Under the same official settings (4 x 64 network, 50 simulations,
no domain knowledge), the final model:

- beats the model it started from: 27 wins, 0 losses, 59 draws in 100 games
  (score 0.657),
- beats random play (16 wins, 0 losses) and the untrained network (17 wins,
  4 losses), where the first runs lost more than they won,
- still draws almost every game against greedy capture (0 wins, 1 loss).

Its main weakness is finishing games: it gets ahead but rarely checkmates, so
most games end by the 100-ply no-progress rule. A bigger 6 x 128 network was
also tried and stopped, since it stayed behind after 61 iterations. Thinking
longer when playing helps: with 800 simulations per move the final model beats
itself at 50 by 23 wins to 0 (0.616). Training on with draws counted as
slightly bad made it weaker (0.454), and 140 more iterations at a smaller
learning rate gave no measurable gain (0.515), so iteration 560 stays the
final model. Self-play later moved to the GPU (about
2.2 to 2.4 times faster); with it, the bigger network reached the final
model's level by learning from its games but did not pass it, and a
measurement showed that resigning hopeless games cannot work yet, because the
model does not see a loss coming.
`results/README.md` has every number, the plots, the game endings, the human
games and the diagnosis in full.

## Milestones

| Milestone | Status |
|---|---|
| 1. Rules engine verified by unit tests | Done: 36 rules tests |
| 2. MCTS with an untrained network | Done: search tests plus a run on the real untrained network |
| 3. Self-play pipeline, loss tracked over 50 to 100+ games | Done: 54,000 games in the long run, loss curves in `results/` |
| 4. Benchmark against earlier iterations and baseline players, learning curves | Done: `results/README.md` |

## Team

| Role | Area |
|---|---|
| Student 1: Environment and logic | `chaturanga/`, `play.py` |
| Student 2: Deep learning and pipeline | `network/`, training loop, replay buffer |
| Student 3: Search and evaluation | `mcts/`, `evaluation/`, benchmarks, plots, `play_vs_ai.py` |
