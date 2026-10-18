---
title: "ChaturangZero: Learning Chaturanga by Self-Play"
subtitle: "CS F407 course project report"
date: "October 2026"
---

# Abstract

We built ChaturangZero, an AlphaZero-style agent for Chaturanga, the ancient
Indian ancestor of chess. The agent knows only the rules. A residual network
with a policy head and a value head guides Monte Carlo Tree Search (MCTS) with
PUCT selection, and the network is trained on the games the agent plays against
itself. The full system runs on a single CPU: a rules engine checked by unit
tests, the network, search, a self-play training loop with a replay buffer, and
an evaluation suite. We trained two final runs of 400 self-play games each
(50 simulations per move, a 4-block, 64-filter network, about 7 hours per run).
The policy loss fell from about 5 to 2.2, but the trained agent is not
measurably stronger than the untrained network. Against an untrained network, a
random player and a greedy-capture player, every checkpoint scores within its
95% margin of 0.5. In six games against a human new to chess, the agent lost 3
and drew 3. We measured why: about 80% of self-play games are drawn, mostly by
the 100-ply no-progress rule, so the value head is trained almost only on
targets of 0 and search cannot tell good positions from bad ones. We report
this negative result in full, along with the diagnosis and what we expect would
fix it.

# 1. Introduction

AlphaZero (Silver et al., 2018) showed that one algorithm can learn chess,
shogi and Go to superhuman level from the rules alone, by combining tree search
with a neural network trained on self-play. The goal of this project was to
build that method from scratch for Chaturanga and to measure how far it gets on
the compute of a course project.

The course plan set these constraints:

- no human domain knowledge beyond the rules (no opening books, no piece
  values, no hand-written evaluation);
- 30 to 50 MCTS simulations per move during training;
- a residual network with 4 to 6 blocks and 64 to 128 filters;
- a 4096-wide action space (every from-square, to-square pair).

The project had four milestones: (1) a rules engine verified by unit tests,
(2) MCTS working with an untrained network, (3) a self-play pipeline with the
loss tracked over at least 50 to 100 games, and (4) benchmarks against earlier
iterations and baseline players, with learning curves. All four are complete.

# 2. The game

Chaturanga is played on an 8x8 board with the same starting layout as chess,
but most pieces are weaker:

| Piece | Letter | Move |
|---|---|---|
| Raja (king) | K | one square in any direction |
| Mantri (counselor) | Q | one square diagonally |
| Ratha (chariot) | R | any distance orthogonally, like a rook |
| Gaja (elephant) | E | exactly two squares diagonally, jumping the square between |
| Ashva (horse) | N | an L-shaped jump, like a knight |
| Padati (foot soldier) | P | one square forward, captures one square diagonally forward, promotes to Mantri on the last rank |

There is no castling, no pawn double step and no en passant. Checkmate wins.
Stalemate, 100 plies with no capture or pawn move (the no-progress rule) and
threefold repetition are draws. Because the Mantri and Gaja are short-range,
material advantages are much harder to convert into checkmate than in chess,
which turns out to matter for learning (Section 7).

# 3. Method

## 3.1 State and action encoding

A position is encoded as 13 planes of 8x8: one plane per piece type and color
(12) and one plane for the side to move. Squares are absolute (the board is not
flipped for Black). A move is the action `from * 64 + to`, giving 4096 actions;
illegal actions are masked out of the policy before the softmax. We checked
that the encoding and the action indices agree for both colors.

## 3.2 Network

`ChaturangaNet` is a residual network:

- an input 3x3 convolution to 64 filters with batch normalization and ReLU;
- 4 residual blocks, each two 3x3 convolutions with batch normalization and a
  skip connection;
- a policy head (1x1 convolution to 2 planes, then a linear layer to 4096
  logits);
- a value head (1x1 convolution to 1 plane, a 64-unit hidden layer, then a
  tanh output in [-1, 1], the expected result for the side to move).

It has 836,359 parameters. The loss is value mean squared error plus policy
cross-entropy against the MCTS visit distribution, with L2 regularization
(weight decay 1e-4).

## 3.3 Monte Carlo Tree Search

Each simulation descends the tree from the root by choosing the child that
maximizes

$$ -Q(s') + c_{puct} \, P(s, a) \, \frac{\sqrt{N(s)}}{1 + N(s')} $$

where $Q(s')$ is the child's mean value from the child's side (negated to give
the parent's view), $P(s, a)$ is the network's prior for the move, $N$ are
visit counts and $c_{puct} = 1.5$. At a new leaf, the network gives the priors
for its children and a value, which is backed up the path with its sign
flipped at each ply. There are no random rollouts. Child states are created
only when first visited. During self-play, Dirichlet noise
($\alpha = 0.3$, weight 0.25) is mixed into the root priors so that the agent
explores moves the network does not yet favor.

## 3.4 Self-play and training

Each training iteration:

1. **Self-play.** The current network plays 20 games against itself with 50
   simulations per move. Moves are sampled in proportion to visit counts.
   Every position becomes an example (state, visit distribution, result from
   the mover's view: +1, -1 or 0 for a draw). Games longer than 800 plies stop
   and count as draws.
2. **Training.** Examples go into a replay buffer of the latest 20,000
   positions. The network takes 400 Adam steps (learning rate 1e-3) on random
   batches of 64.
3. **Checkpoint.** The network is saved, so every iteration can be evaluated
   later.

One rules-only addition: with `--avoid_repetition`, the agent never picks a
move that causes a threefold-repetition draw unless every searched move does.
This uses only the rules, so it fits the no-domain-knowledge requirement. We
added it because early self-play games often ended in repetition loops.

# 4. Implementation and testing

The code is plain Python with NumPy and PyTorch, organized by component:

| Folder | Contents |
|---|---|
| `chaturanga/` | rules engine, move generation, game end, encoding |
| `network/` | the network, loss and checkpoints |
| `mcts/` | tree nodes, PUCT search, root noise |
| `training/` | self-play, replay buffer, training loop |
| `evaluation/` | baseline players, benchmark, game-ending classifier, human-game log |

97 automated tests cover the rules (36 tests, including check, checkmate,
stalemate, promotion and the draw rules), the search (including a toy game
with a known best move), the network, value-target signs, policy targets, a
complete small training run including resuming from a checkpoint, the
benchmark scoring and the evaluation tools. All runs are seeded and repeatable.

# 5. Experiments

## 5.1 Training runs

| Run | Simulations | Games | Move choice in self-play | Decisive games | Policy loss | Time |
|---|---|---|---|---|---|---|
| Run 1 (early) | 30 | 10 x 10 | earlier settings | 16 of 100 (16%) | 6.70 to 2.64 | 0.5 h |
| Final A | 50 | 20 x 20 | sample 30 plies, then best | 71 of 400 (18%) | 5.15 to 2.35 | 7.1 h |
| Final B | 50 | 20 x 20 | sample the whole game | 90 of 400 (22%) | 4.89 to 2.23 | 6.8 h |

Final B is the model we kept.

## 5.2 Benchmark protocol

To compare checkpoints fairly, every checkpoint plays 20 games against a fixed
opponent under the same conditions: 50 simulations, no root noise, temperature
0.3, repetition avoidance, alternating colors, the same random seed for every
checkpoint, and an 800-ply cap (unfinished games are left out). The score is
(wins + 0.5 x draws) / finished games, with a 95% margin of
$1.96 \sqrt{p(1-p)/n}$. The opponents are:

- **untrained**: the same MCTS agent with a freshly initialized network;
- **random**: a uniformly random legal move;
- **greedy-capture**: captures the most valuable piece it can, otherwise
  plays randomly. Its piece values are used only by this baseline, never by
  our agent.

# 6. Results

## 6.1 Learning curves

![Policy loss, value loss, share of decisive games and game length per iteration for the final runs and run 1.](../results/training_curves.png)

The policy loss falls quickly over the first 5 iterations and then flattens
around 2.2 to 2.4: the network learns to predict its own search, but search is
not getting better. The value loss stays below 0.05 and noisy, because most
targets are 0. The share of decisive games does not trend upward, and games
stay long (400 to 600 plies on average).

## 6.2 Benchmarks against fixed opponents

![Benchmark score per iteration for final run B against the untrained network, random and greedy-capture players. The dashed line is an even score.](../results/benchmark_curve_finalB.png)

| Iteration (final B) | vs untrained | vs random | vs greedy-capture |
|---|---|---|---|
| 0 (untrained) | 0.5 by definition | 0.550 +/- 0.066 | 0.550 +/- 0.066 |
| 4 | 0.500 +/- 0.103 | | |
| 8 | 0.500 +/- 0.115 | | |
| 10 | | 0.444 +/- 0.106 | 0.474 +/- 0.050 |
| 12 | 0.423 +/- 0.098 | | |
| 16 | 0.393 +/- 0.108 | | |
| 20 | 0.500 +/- 0.087 | 0.472 +/- 0.093 | 0.425 +/- 0.078 |

Final A against the untrained network scored 0.444, 0.433, 0.382, 0.393 and
0.500 at iterations 4, 8, 12, 16 and 20. Every score in both runs is within its
margin of 0.5.

![Final A and final B against the untrained network.](../results/benchmark_curve_final_vs_untrained.png)

## 6.3 Against earlier iterations

| Match (20 games) | Wins | Losses | Draws | Unfinished | Score |
|---|---|---|---|---|---|
| Final B iter 20 vs final A iter 20 | 1 | 1 | 15 | 3 | 0.500 |
| Final B iter 20 vs final B iter 10 | 0 | 0 | 15 | 5 | 0.500 |

Ten more iterations of training did not give a single win over the older
network.

## 6.4 Agent vs. human

A team member who is new to chess played six games against final B, alternating
colors.

| Game | Human plays | Result | Ending | Plies | Pieces left (human / agent) |
|---|---|---|---|---|---|
| 1 | White | Human win | Checkmate | 91 | 11 / 2 |
| 2 | Black | Human win | Checkmate | 128 | 12 / 2 |
| 3 | White | Draw | Stalemate | 137 | 12 / 1 |
| 4 | Black | Human win | Checkmate | 90 | 16 / 1 |
| 5 | White | Draw | Stalemate | 161 | 8 / 1 |
| 6 | Black | Draw | Stalemate | 260 | 7 / 1 |

The agent won none of the six games. Its score is 0.250, with a 95% Wilson
range of 0.060 to 0.635. Six games are too few to separate that from 0.5
statistically, but the final positions are one-sided: the agent always ended
with at most two pieces, and in each draw it had only its king left, which the
human stalemated while going for checkmate.

## 6.5 How games end

Final B at iteration 16, 10 games against each opponent under the benchmark
protocol:

| Ending | vs itself | vs random | vs greedy-capture |
|---|---|---|---|
| No-progress rule | 5 | 5 | 7 |
| Threefold repetition | 2 | 3 | 3 |
| Checkmate | 2 | 1 | 0 |
| 800-ply cap | 1 | 1 | 0 |

## 6.6 Exploratory runs (outside the course constraints)

We ran two short experiments that break the constraints, to test the
diagnosis below. They are not the final agent.

- **Material value targets** (30 simulations, 10 x 10 games): drawn games get a
  value target from the final material balance, which uses piece values. The
  value head did learn material (correlation +0.45 with the material balance,
  against +0.07 in run 1), but strength did not improve (0.474 and 0.469
  against untrained at iterations 5 and 10).
- **100 simulations per move** (10 x 10 games, also with material targets):
  decisive games peaked at 60% in iteration 3, but strength against untrained
  was 0.421 at iteration 5 and 0.346 at iteration 10.

![Exploratory runs: loss, decisive share and game length.](../results/training_curves_exploratory.png)

# 7. Discussion

We first checked for a bug. The tests confirm the value targets have the right
sign for both colors, the policy targets match the visit counts, and the
encoding and actions agree for Black, so we believe the result reflects the
method at this scale, not an error. Three causes stand out:

1. **Too little data.** 400 games (about 200,000 positions) per run is tiny
   for this method. AlphaZero used millions of games, with far more search per
   move.
2. **The value head learns almost nothing.** With about 80% of games drawn,
   nearly every value target is 0. In run 1, 81% of value predictions were
   within 0.05 of 0, and their correlation with the material balance was only
   +0.07. Without a useful value, 50-simulation search cannot tell losing a
   piece from a quiet move, so trained play is no sharper than untrained play.
3. **Games are long and rarely decisive.** Chaturanga's short-range pieces make
   checkmate hard to force. Games average 400 to 600 plies, and most end by the
   no-progress rule (Section 6.5), so the rare win or loss signal is spread over
   hundreds of quiet positions.

The human games show the same weakness from the outside: the agent gives away
material steadily and cannot defend, and a beginner can win.

The exploratory runs suggest that a better value signal alone is not enough at
this scale: the value head learned material, but 30 to 100 simulations with a
weak policy still did not produce stronger play in 100 games.

# 8. Limitations and future work

- Every benchmark uses 20 games per checkpoint, so differences smaller than
  about 0.1 cannot be resolved.
- The human evaluation is six games by one beginner.
- Compute was the main limit: about 20 minutes per iteration of 20 games on
  one CPU core.

What we expect would help most, in order:

1. Many more self-play games: parallel self-play workers and runs of hundreds
   of iterations.
2. A shorter game cap or resignation rule, so results reach the value head
   more often.
3. More simulations once the value head is useful, and a larger network.

# 9. Conclusion

We implemented every part of an AlphaZero-style system for Chaturanga under
the course constraints and verified it with 97 tests. Training ran as designed,
and the network learned to imitate its own search, but in 400 self-play games
per run the agent did not become measurably stronger than its untrained start,
and a beginner beat it. The cause is a value signal starved by draws: most
games end by the no-progress rule, so the value head sees almost only 0. The
evaluation tools we built (a fixed benchmark protocol, head-to-head matches,
game-ending counts and human-game logging) made this result measurable and
explainable, and they are ready for a longer run.

# References

1. Silver, D. et al. (2018). A general reinforcement learning algorithm that
   masters chess, shogi, and Go through self-play. *Science*, 362(6419),
   1140-1144.
2. Silver, D. et al. (2017). Mastering the game of Go without human knowledge.
   *Nature*, 550, 354-359.

# Appendix A: Team roles

| Role | Area |
|---|---|
| Student 1: Environment and logic | rules engine (`chaturanga/`), `play.py` |
| Student 2: Deep learning and pipeline | network, training loop, replay buffer |
| Student 3: Search and evaluation | MCTS, evaluation suite, benchmarks, plots, `play_vs_ai.py` |

# Appendix B: Reproducing the results

```
python -m pytest -q
OMP_NUM_THREADS=1 python -m training.train --iterations 20 --games_per_iter 20 \
    --simulations 50 --avoid_repetition --temperature_moves 1000 \
    --max_plies 800 --train_steps 400 \
    --checkpoint_dir checkpoints/final_B --log checkpoints/final_B/log.csv
python -m evaluation.benchmark --checkpoint_dir checkpoints/final_B --opponent untrained --every 4
python -m evaluation.game_endings --checkpoint checkpoints/final_B/iter_0016.pt --opponent self --games 10
python play_vs_ai.py --checkpoint checkpoints/final_B/latest.pt --player NAME
```

Every log, benchmark CSV, match log and plot is in `results/`.
