# MCTS and Search Evaluation (Student 3)

## Implementation

The search module implements guided Monte Carlo Tree Search following the AlphaZero approach: selection via the PUCT formula, expansion using network-predicted move priors, leaf evaluation via the network's value head rather than random rollouts, and backpropagation with perspective-flipping at each level of the tree.

## Correctness verification

The module passes 44 automated tests: 36 covering the Chaturanga rules engine and 8 covering MCTS itself. The MCTS tests include an independent check against a toy game with a known correct answer, confirming the search mechanics (selection, expansion, backpropagation) work correctly in isolation from the real environment. A separate sanity check against the actual Chaturanga environment, using an untrained network with uniform policy output, confirmed that simulation visits spread evenly across all legal moves as expected when no learned signal is present.

Once the real network module (Student 2) was merged, a further integration check (`network_integration_preview.py`) confirmed MCTS runs correctly end to end against the actual `ChaturangaNet`, with deterministic, reproducible output across repeated runs on the same network weights. Combined with the network module's own 7 tests, the full suite stands at 51 automated tests passing.

## Reproducibility

All benchmark runs are fully seeded, including the stub network's internal randomness, not just move selection. Running an identical command twice produces byte-identical results, verified directly by comparing two independent runs of the same sweep configuration. Real-network runs are deterministic within a run (the network's weights are fixed once loaded), though not currently seeded across separate runs, since the network's random initialization is not yet pinned to `--seed`.

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

## Why most games are draws

Across every run, 85 to 90 percent of games ended in a draw rather than a decisive result. This matches the project's own random-vs-random baseline (see `chaturanga`'s `play.py --mode random` output) and appears to be a property of weak play in this environment generally, rather than anything specific to MCTS or to which network is guiding it. It limits how many decided games any given run size can produce, which is the main constraint on tightening the confidence interval on any of the results above.

One related note from testing with the real network specifically: games against the untrained network required a notably higher `--max_plies` cap (1500, versus 600-900 for the stub) to resolve naturally rather than being cut off. This is consistent with the same underlying cause described above, a fixed biased prior can lead MCTS into repetitive, non-progressing move patterns more readily than fresh random noise does.

## Limitations and next steps

The stub-network results measure what search depth alone contributes, with no learned signal involved. The real-network result measures the opposite: what an untrained but structurally real network contributes, which so far is no better than random. The simulation-depth sweep's per-depth samples (1 to 4 decided games each) are too small to support a confident claim about how win rate scales with simulation count specifically; the pooled across-depth result is the more defensible number from that data.

As Student 2's training loop produces checkpoints, rerunning this same benchmark (`evaluate_vs_random.py --network real`) against each one, and eventually checkpoint-vs-checkpoint rather than checkpoint-vs-random, is the natural way to produce the learning-curve evidence Milestone 4 calls for. The "before training" number above (37.5 percent, 8 decided games) is the baseline that progress should be measured against.

## Files in this folder

- `mcts_vs_random_60games.log` — first stub-network baseline run
- `mcts_vs_random_200games.log` — larger stub-network baseline run
- `sweep_results.csv` — simulation-depth sweep, raw numbers (stub network)
- `sweep_win_rate_vs_sims.png` — simulation-depth sweep, plotted (stub network)
- `mcts_vs_real_untrained_60games.log` — real (untrained) network baseline run