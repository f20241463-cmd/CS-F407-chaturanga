# ChaturangZero: a self-play AlphaZero agent for Chaturanga

CS F407 course project. An agent that learns Chaturanga, the ancient Indian
ancestor of chess, by playing against itself. Like AlphaZero, it gets no human
knowledge beyond the rules: a residual network suggests moves and judges
positions, Monte Carlo Tree Search (MCTS) looks ahead with it, and the network
is trained on the games it plays.

Everything runs on a CPU. Python 3 with `numpy`, `torch` and `matplotlib`.

## How it fits together

```
chaturanga/   rules engine: board, legal moves, game end, tensor + action encoding
network/      dual-head ResNet (policy + value), loss, checkpoints
mcts/         guided MCTS with PUCT, root Dirichlet noise, lazy child states
training/     self-play games, replay buffer, training loop
evaluation/   baseline players, per-iteration benchmark, game endings, human-game log
results/      training logs, benchmark CSVs, plots, and the results write-up
tests/        96 automated tests
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
python play_vs_ai.py --checkpoint checkpoints/final_B/latest.pt --player YOUR_NAME
python -m training.train --iterations 2 --games_per_iter 2 --checkpoint_dir checkpoints/demo --log checkpoints/demo/log.csv
python -m evaluation.benchmark --checkpoint_dir checkpoints/demo --opponent random --games 4 --out checkpoints/demo/bench.csv
```

The last two lines are a tiny training run (about 1 minute) and its benchmark
(about 5 minutes). Checkpoints are not stored in git (`checkpoints/` is ignored). The training
command for the final run, and every evaluation command, is in
`training/README.md`.

## Results in brief

Two final runs of 20 iterations x 20 self-play games each (400 games, about
7 hours per run on one CPU core), 50 simulations per move, the 4 x 64
network. The policy loss fell from about 5 to 2.2 and then flattened.

The trained agent is **not measurably stronger** than where it started. Under
a fixed 20-game benchmark, every checkpoint scores within its margin of 0.5
against the untrained network, a random player and a greedy-capture player,
and the two final runs drew each other. About 80% of games are drawn, mostly by
the 100-ply no-progress rule, so the value head sees almost only 0 targets and
search cannot tell good positions from bad ones. 400 games is very little data
for this method. `results/README.md` has every number, the plots, the
measured game endings, two exploratory runs, and this diagnosis in full.

## Milestones

| Milestone | Status |
|---|---|
| 1. Rules engine verified by unit tests | Done: 36 rules tests |
| 2. MCTS with an untrained network | Done: search tests plus a run on the real untrained network |
| 3. Self-play pipeline, loss tracked over 50 to 100+ games | Done: 400 games per final run, loss curves in `results/` |
| 4. Benchmark against earlier iterations and baseline players, learning curves | Done: `results/README.md` |

## Team

| Role | Area |
|---|---|
| Student 1: Environment and logic | `chaturanga/`, `play.py` |
| Student 2: Deep learning and pipeline | `network/`, training loop, replay buffer |
| Student 3: Search and evaluation | `mcts/`, `evaluation/`, benchmarks, plots, `play_vs_ai.py` |
