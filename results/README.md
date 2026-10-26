# MCTS and Search Evaluation (Student 3)

## Implementation

The search module implements guided Monte Carlo Tree Search following the AlphaZero approach: selection via the PUCT formula, expansion using network-predicted move priors, leaf evaluation via the network's value head rather than random rollouts, and backpropagation with perspective-flipping at each level of the tree.

## Correctness verification

The module passes 44 automated tests: 36 covering the Chaturanga rules engine and 8 covering MCTS itself. The MCTS tests include an independent check against a toy game with a known correct answer, confirming the search mechanics (selection, expansion, backpropagation) work correctly in isolation from the real environment. A separate sanity check against the actual Chaturanga environment, using an untrained network with uniform policy output, confirmed that simulation visits spread evenly across all legal moves as expected when no learned signal is present.

Once the real network module (Student 2) was merged, a further integration check (`network_integration_preview.py`) confirmed MCTS runs correctly end to end against the actual `ChaturangaNet`, with deterministic, reproducible output across repeated runs on the same network weights. Combined with the network module's own 7 tests, the full suite stood at 51 automated tests passing. Tests added later for root Dirichlet noise (5) and checkpoint loading and match scoring (5) bring it to 61. With the training pipeline and the evaluation suite added, the full suite is now 97 tests (`python -m pytest -q`).

## Reproducibility

All benchmark runs are fully seeded, including the stub network's internal randomness, not just move selection. Running an identical command twice produces byte-identical results, verified directly by comparing two independent runs of the same sweep configuration. Real-network runs are seeded too: the untrained network's random initialization is pinned to `--seed` (via `torch.manual_seed`), so repeated runs play against the same network. The 37.5 percent baseline below was recorded before this seeding was added, so rerunning it now plays a different (but fixed) set of initial weights. Trained networks are loaded with `--checkpoint path/to/file.pt` in `evaluate_vs_random.py`, `play_vs_ai.py` and `compare_checkpoints.py`.

## Results against a random-move baseline (stub network)

Using an untrained stub network (random policy and value, resampled fresh on every call, carrying no structured information), MCTS-guided play was benchmarked against pure random move selection across three separate runs, alternating which side MCTS played to cancel out first-move advantage:

- 60 games: MCTS won 62.5 percent of decided games (5 wins, 3 losses, 8 decided out of 60)
- 200 games: MCTS won 73.7 percent of decided games (14 wins, 5 losses, 19 decided out of 200)
- Simulation-depth sweep (5/15/30/50 simulations, 20 games per depth): MCTS won 8 of 9 decided games across all depths combined

Pooling the two larger runs gives 19 wins against 8 losses across 27 decided games, a win rate of roughly 70 percent in MCTS's favor. A binomial check against a 50 percent null (no advantage from search) puts this comfortably outside what chance alone would typically produce, suggesting a real effect from search rather than noise, though the sample remains modest.

## Baseline before training (real, untrained network)

Once Student 2's actual `ChaturangaNet` was available, the same benchmark was rerun using the real (but still untrained) network in place of the stub, to establish a "before training" reference point.

- 60 games: MCTS won 37.5 percent of decided games (3 wins, 5 losses, 8 decided out of 60), 86.7 percent draws, 0 unfinished

This is a notably different result from the stub-network baseline above, and the difference is itself informative rather than concerning. The stub network's policy and value outputs are unbiased random noise, resampled fresh on every call, so MCTS guided by it is effectively relying on genuine lookahead into real game outcomes. An untrained real network, by contrast, produces a *fixed* set of predictions, structured but arbitrary, since the weights have never been updated. There is no guarantee that structure happens to favor good moves, and it can just as easily steer search consistently toward worse ones. This is a known characteristic of AlphaZero-style systems: before training, a real network's confident-looking but meaningless biases can actively mislead search in a way unbiased randomness cannot.

With only 8 decided games, this result is not statistically strong evidence that the real network is worse than random, the sample is too small either way. What it does establish is a clean, honest "before training" data point: at initialization, the network provides no demonstrated advantage over random play. This is the gap the self-play training loop (Student 2, in progress) exists to close, and this result gives a concrete number to compare future trained checkpoints against.

## Self-play training (Milestones 3 and 4)

The training loop and its options are documented in `training/README.md`. All
runs below use the default network (4 residual blocks, 64 filters) and only the
rules of the game, as the official plan requires, except the two runs marked
exploratory.

### Final runs

Two runs of 20 iterations x 20 self-play games (400 games each), 50 MCTS
simulations per move, root Dirichlet noise (alpha 0.3, weight 0.25), 800-ply
cap, draws valued 0, 400 Adam steps of batch 64 per iteration, replay buffer of
20000 positions, and repetition avoidance on. They differ only in how moves are
picked during self-play:

| Run | Move choice in self-play | Decisive games | Policy loss (iter 1 to 20) | Time |
|---|---|---|---|---|
| Final A | sample for 30 plies, then most visited | 71 of 400 (18%) | 5.15 to 2.35 | 7.1 h |
| Final B | sample the whole game | 90 of 400 (22%) | 4.89 to 2.23 | 6.8 h |

`training_curves.png` shows both, plus the earlier 30-simulation run 1. The
policy loss falls quickly for 5 iterations and then flattens; value loss stays
small (under 0.05) and noisy, because about 80% of games are drawn and every
position in a drawn game has target 0. The share of decisive games does not trend upward.

### Benchmark protocol

`python -m evaluation.benchmark` plays 20 games per checkpoint with a fixed
protocol so iterations can be compared: the agent uses 50 simulations, no
root noise, temperature 0.3 for the whole game and repetition avoidance;
colors alternate; every checkpoint faces the same seeded game sequence; games
stop at 800 plies and unfinished games are left out of the score. Score =
(wins + 0.5 x draws) / finished games, with a rough 95% margin. Opponents are
`random`, `greedy` (captures the most valuable piece it can, otherwise random)
and `untrained` (the same MCTS agent with a fresh network).

### Results: no measurable strength gain

Final run B (`benchmark_curve_finalB.png`):

| Iteration | vs untrained | vs random | vs greedy-capture |
|---|---|---|---|
| 0 (untrained) | (0.5 by definition) | 0.550 +/- 0.066 | 0.550 +/- 0.066 |
| 4 | 0.500 +/- 0.103 | | |
| 8 | 0.500 +/- 0.115 | | |
| 10 | | 0.444 +/- 0.106 | 0.474 +/- 0.050 |
| 12 | 0.423 +/- 0.098 | | |
| 16 | 0.393 +/- 0.108 | | |
| 20 | 0.500 +/- 0.087 | 0.472 +/- 0.093 | 0.425 +/- 0.078 |

Final run A against the untrained network (`benchmark_curve_final_vs_untrained.png`):
0.444, 0.433, 0.382, 0.393 and 0.500 at iterations 4, 8, 12, 16 and 20, with
margins of 0.09 to 0.18.

Head to head, final B iteration 20 against final A iteration 20 (20 games, same
protocol): 1 win each, 15 draws, 3 unfinished, score 0.500 +/- 0.082
(`finalB_vs_finalA_iter20.log`).

Against its own earlier self, final B iteration 20 played iteration 10 (20
games, same protocol): 0 wins, 0 losses, 15 draws, 5 unfinished, score 0.500
(`finalB_iter20_vs_iter10.log`). Ten more iterations of training did not
produce a single win over the older network.

Every score is within its margin of 0.5. After 400 self-play games, neither
run is measurably stronger than the untrained network, than random play or than
greedy capture, and the two runs are even with each other. Final B is the one
kept (`checkpoints/final_B/latest.pt`): it ties A head to head and at iteration
20, and is never behind A at the earlier benchmarked iterations, but the gap is
within noise. Run 1 (30 simulations, 100 games) was also flat against random
(0.469 to 0.550) and greedy (0.450 to 0.550), in `benchmark_run1_*.csv`.

We checked for a pipeline error before accepting this. The tests cover the
value-target signs, the policy targets and a full train-and-resume run, and the
state encoding and the action indices both use absolute board squares (with a
side-to-move plane), so they agree for Black.

### Agent vs. Human

Six games against final B (`checkpoints/final_B/latest.pt`, 50 simulations,
repetition avoidance, always its most visited move), played by a team member
who is new to chess, alternating colors. Every game is in `human_games.csv`
with its full move list.

| Game | Human plays | Result | Ending | Plies | Pieces left (human / agent) |
|---|---|---|---|---|---|
| 1 | White | Human win | Checkmate | 91 | 11 / 2 |
| 2 | Black | Human win | Checkmate | 128 | 12 / 2 |
| 3 | White | Draw | Stalemate | 137 | 12 / 1 |
| 4 | Black | Human win | Checkmate | 90 | 16 / 1 |
| 5 | White | Draw | Stalemate | 161 | 8 / 1 |
| 6 | Black | Draw | Stalemate | 260 | 7 / 1 |

The agent won none of the six: 3 human wins and 3 draws, an agent score of
0.250 (95% Wilson range 0.060 to 0.635; `python play_vs_ai.py --summary`).
Six games cannot separate that score from 0.5 statistically, but the final
positions are one-sided. In every game the agent was left with at most two
pieces, and in all three draws it had only its king, which the human
stalemated while trying to checkmate it. This matches the benchmark: the agent
gives material away and cannot tell a lost position from a quiet one, so even a
beginner beats it.

### Why most games are draws (measured)

How games end, for final B iteration 16 under the benchmark protocol, 10 games
each (`python -m evaluation.game_endings --checkpoint checkpoints/final_B/iter_0016.pt --opponent random --games 10`):

| Ending | vs itself | vs random | vs greedy-capture |
|---|---|---|---|
| No-progress rule (100 plies without capture or pawn move) | 5 | 5 | 7 |
| Threefold repetition | 2 | 3 | 3 |
| Checkmate | 2 | 1 | 0 |
| 800-ply cap reached | 1 | 1 | 0 |

The main cause is the no-progress rule, not repetition: the agent reaches
positions it could win but cannot convert them, and shuffles until the
100-ply rule ends the game. Repetition still happens because random and greedy
do not avoid it, and because avoidance gives way when every move would repeat.

### Diagnosis

- **Too little data.** 400 games (about 200,000 positions) per run is very
  small for AlphaZero-style learning; the original work used millions of games.
- **The value head learns almost nothing.** With about 80% of games drawn, most
  value targets are 0. In run 1, 81% of value predictions were within 0.05 of 0
  and their correlation with material balance was +0.07. Without a useful value,
  50-simulation search cannot tell a lost piece from a quiet move, so trained
  play is not sharper than untrained play.
- **Long games dilute the signal.** Games average 400 to 600 plies, so a single
  result at the end is spread over hundreds of mostly quiet positions.

### Exploratory runs (outside the official plan)

Both are kept only as labeled experiments (`training_curves_exploratory.png`).

- **Material value targets** (`--material_scale 5`, 30 simulations, 10 iterations x 10 games):
  drawn games get value targets from the final material balance. This uses
  piece values, which is human domain knowledge. The value head did learn
  material (correlation +0.45 on self-play positions), but strength did not
  improve: 0.474 +/- 0.089 and 0.469 +/- 0.105 against untrained at iterations
  5 and 10. With a uniform policy and this value head, the agent scored 0.500
  (0 wins, 0 losses) against untrained, versus 0.625 +/- 0.137 for the
  untrained value head with a uniform policy.
- **100 simulations per move** (10 iterations x 10 games, also with
  `--material_scale 5`), above the 30 to 50 allowed: decisive games peaked at 60% in iteration 3, but strength against
  untrained was 0.421 +/- 0.082 at iteration 5 and 0.346 +/- 0.125 at
  iteration 10.

### What we changed next

The diagnosis pointed at too little data, and the compute was the limit (about
20 minutes per iteration of 20 games on one CPU core). We made self-play about
11 times faster without changing the rules or the search (a faster rules
engine, a faster network copy for search, and 6 parallel self-play workers;
see `training/README.md`), and then trained much longer.

## The long run (final model)

### Setup

The run continues from final B (iteration 20) under the same official
settings: 4 residual blocks and 64 filters, 50 simulations, root noise,
repetition avoidance, 800-ply cap, draws valued 0, no domain knowledge. Per
iteration: 100 self-play games, then 500 Adam steps of batch 256 from a replay
buffer of 250,000 positions.

| Iterations | Learning rate | Notes |
|---|---|---|
| 21 to 420 | 1e-3 | restarted once at 220; after a restart the buffer refills before training resumes |
| 421 to 500 | 1e-4 | two stretches of 40 |
| 501 to 560 | 3e-5 | final stretch |

In total 540 iterations, 54,000 self-play games, about 23.7 million positions
and about 144 hours on one laptop (Apple A18 Pro, 6 CPU cores for self-play,
its GPU for training). `training_curves_long.png` shows the curves and
`training_log_long.csv` has every iteration (1 to 20 are final B).

### Progress against the starting model

Every so often a checkpoint played final B (iteration 20) for 100 games under
the benchmark protocol (`long_run_matches/`):

| Iteration | Wins | Losses | Draws | Unfinished | Score |
|---|---|---|---|---|---|
| 40 | 10 | 3 | 61 | 26 | 0.547 +/- 0.047 |
| 60 | 11 | 5 | 62 | 22 | 0.538 +/- 0.050 |
| 74 | 15 | 3 | 72 | 10 | 0.567 +/- 0.044 |
| 100 | 18 | 4 | 61 | 17 | 0.584 +/- 0.052 |
| 140 | 27 | 2 | 57 | 14 | 0.645 +/- 0.053 |
| 180 | 18 | 3 | 68 | 11 | 0.584 +/- 0.047 |
| 220 | 21 | 5 | 57 | 17 | 0.596 +/- 0.057 |
| 300 | 27 | 5 | 52 | 16 | 0.631 +/- 0.060 |
| 380 | 11 | 5 | 63 | 21 | 0.538 +/- 0.049 |
| 500 | 27 | 3 | 60 | 10 | 0.633 +/- 0.053 |
| **560 (final)** | **27** | **0** | **59** | **14** | **0.657 +/- 0.049** |

Unlike the 400-game runs, training now clearly helps: every checkpoint from
iteration 40 on beats the starting model, and the final one won 27 games and
lost none. Most of the gain came in the first 100 to 140 iterations. From there
to iteration 420 the score only moved up and down by about the size of its margin.

### Lowering the learning rate

To pick a model, iterations 140, 300, 420 and 460 (the first 40 at learning
rate 1e-4) played each other, 100 games per pair:

| | vs 140 | vs 300 | vs 420 | vs 460 |
|---|---|---|---|---|
| 140 | | 0.511 | 0.488 | 0.413 |
| 300 | 0.489 | | 0.500 | 0.463 |
| 420 | 0.512 | 0.500 | | 0.453 |
| 460 | **0.587** | **0.537** | **0.547** | |

(Score of the row model against the column model.) Iterations 140, 300 and 420
are even with each other, and 460 beats all three (41 wins, 9 losses in
total). At learning rate 1e-3 the network kept moving around one level without
settling; the lower rate let it settle, which is why the last 140 iterations
use 1e-4 and then 3e-5. Each further stretch still helped a little:

| Match | Games | Wins | Losses | Score |
|---|---|---|---|---|
| 500 vs 460 | 100 | 8 | 3 | 0.526 +/- 0.033 |
| 530 vs 500 | 100 | 3 | 2 | 0.505 +/- 0.022 |
| 560 vs 500 | 100 | 8 | 2 | 0.531 +/- 0.031 |
| 560 vs 500, second seed | 200 | 18 | 10 | 0.520 +/- 0.026 |

Over 300 games iteration 560 scores 0.524 against 500, so it is the final model
(`models/chaturangzero_final.pt` in the repo).

### Benchmarks of the final model

Each point is 100 games under the benchmark protocol, the same seeded game
sequence for every checkpoint (`benchmark_long_{random,greedy,untrained}.csv`,
plotted in `benchmark_curve_long.png`):

| Iteration | vs random | vs untrained | vs greedy |
|---|---|---|---|
| 0 (untrained) | 11 W, 2 L: 0.547 | | 3 W, 1 L: 0.510 |
| 20 (final B) | 4 W, 6 L: 0.489 | 4 W, 18 L: 0.419 | 0 W, 6 L: 0.469 |
| 100 | 6 W, 1 L: 0.525 | 7 W, 9 L: 0.487 | 1 W, 2 L: 0.495 |
| 220 | 15 W, 2 L: 0.566 | 23 W, 13 L: 0.560 | 0 W, 0 L: 0.500 |
| 420 | 10 W, 2 L: 0.540 | 13 W, 8 L: 0.533 | 1 W, 5 L: 0.480 |
| 500 | 20 W, 0 L: 0.600 | 16 W, 3 L: 0.577 | 0 W, 4 L: 0.480 |
| **560 (final)** | **16 W, 0 L: 0.581** | **17 W, 4 L: 0.572** | **0 W, 1 L: 0.495** |

Margins are about +/- 0.04 against random and +/- 0.05 against untrained. The
final model beats random and untrained play clearly and almost never loses to
them (0 and 4 losses in 100 games), where final B lost more than it won. It
does not beat greedy capture: 99 of 100 games are draws.

### How the final model's games end

`python -m evaluation.game_endings --checkpoint checkpoints/final.pt`, 30 games each
(`final_endings_{random,greedy}.log`):

| Ending | vs random | vs greedy |
|---|---|---|
| Checkmate | 3 | 1 |
| Stalemate | 0 | 2 |
| No-progress rule | 12 | 27 |
| Threefold repetition | 14 | 0 |
| 800-ply cap | 1 | 0 |

The model has learned not to lose and to win material, but it still rarely
converts: most games end by the 100-ply no-progress rule or, against random, by
repetitions the random player causes. Checkmating needs a long precise plan,
and with about 80% of self-play games drawn the value head gets few examples of
how a win is finished. This is the main limitation of the final model.

### A bigger network (stopped)

We also tried the largest network the plan allows: 6 residual blocks and 128
filters (2.3 million parameters, about 2.8 times the default). It was trained
from scratch with the same settings, but iterations 1 to 5 used self-play games
played by iteration 460, so it did not start from random play
(`--bootstrap_model`, see `training/README.md`). Each iteration took about 25 to
30 minutes, against about 9 for the default network.

| Match | Wins | Losses | Score |
|---|---|---|---|
| big iteration 30 vs 460 | 4 | 21 | 0.411 +/- 0.048 |
| big iteration 60 vs 460 | 3 | 17 | 0.427 +/- 0.043 |
| big iteration 60 vs 420 | 8 | 14 | 0.468 +/- 0.049 |

After 61 iterations (about 31 hours) it was still behind the default network and
improving slowly, so we stopped it and spent the time on the lower learning
rate instead (`training_log_big_exploratory.csv`).

## Limitations

The stub-network results measure what search depth alone contributes, with no learned signal involved. The simulation-depth sweep's per-depth samples (1 to 4 decided games each) are too small to support a confident claim about how win rate scales with simulation count specifically; the pooled across-depth result is the more defensible number from that data. The benchmarks of the 400-game runs use 20 games per checkpoint, so their margins are about +/- 0.1; the long-run matches and benchmarks use 100 games (200 for the last comparison), with margins of about +/- 0.03 to 0.06. Differences smaller than that cannot be resolved, and with most games drawn a score says little about how often the model can actually win. The human games so far were played against final B, not the final model.

## Files in this folder

- `mcts_vs_random_60games.log`: first stub-network baseline run
- `mcts_vs_random_200games.log`: larger stub-network baseline run
- `sweep_results.csv`: simulation-depth sweep, raw numbers (stub network)
- `sweep_win_rate_vs_sims.png`: simulation-depth sweep, plotted (stub network)
- `mcts_vs_real_untrained_60games.log`: real (untrained) network baseline run
- `training_log_finalA.csv`, `training_log_finalB.csv`: per-iteration logs of the final runs
- `training_log_run1.csv`: first run (30 simulations, 10 iterations x 10 games)
- `training_log_material_exploratory.csv`, `training_log_sims100_exploratory.csv`: exploratory runs
- `training_curves.png`, `training_curves_exploratory.png`: loss, decisive share and game length per iteration
- `benchmark_finalA_untrained.csv`, `benchmark_finalB_{untrained,random,greedy}.csv`: final-run benchmarks
- `benchmark_run1_{random,greedy}.csv`, `benchmark_material_untrained.csv`, `benchmark_sims100_untrained.csv`: earlier benchmarks
- `benchmark_curve_finalB.png`, `benchmark_curve_final_vs_untrained.png`: benchmark score per iteration
- `finalB_vs_finalA_iter20.log`: head-to-head match of the two final runs
- `finalB_iter20_vs_iter10.log`: final B iteration 20 against its own iteration 10
- `human_games.csv`: Agent vs. Human games against final B, with full move lists
- `training_log_long.csv`: every iteration of the long run (1 to 20 are final B); `training_curves_long.png` plots it
- `training_log_big_exploratory.csv`: the stopped 6 x 128 network
- `benchmark_long_{random,greedy,untrained}.csv`, `benchmark_curve_long.png`: long-run benchmarks, 100 games per checkpoint
- `long_run_matches/`: every head-to-head match of the long run (`rN` is iteration N, `big_rN` the bigger network)
- `final_endings_{random,greedy}.log`: how the final model's games end
