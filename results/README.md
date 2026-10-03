# MCTS and Search Evaluation (Student 3)

## Implementation

The search module implements guided Monte Carlo Tree Search following the AlphaZero approach: selection via the PUCT formula, expansion using network-predicted move priors, leaf evaluation via the network's value head rather than random rollouts, and backpropagation with perspective-flipping at each level of the tree.

## Correctness verification

The module passes 44 automated tests: 36 covering the Chaturanga rules engine and 8 covering MCTS itself. The MCTS tests include an independent check against a toy game with a known correct answer, confirming the search mechanics (selection, expansion, backpropagation) work correctly in isolation from the real environment. A separate sanity check against the actual Chaturanga environment, using an untrained network with uniform policy output, confirmed that simulation visits spread evenly across all legal moves as expected when no learned signal is present.

## Reproducibility

All benchmark runs are fully seeded, including the stub network's internal randomness, not just move selection. Running an identical command twice produces byte-identical results, verified directly by comparing two independent runs of the same sweep configuration (see `sweep_run1.csv` / `sweep_run2.csv` in this folder).

## Results against a random-move baseline

Using an untrained (random-prior) network, MCTS-guided play was benchmarked against pure random move selection across three separate runs, alternating which side MCTS played to cancel out first-move advantage:

- 60 games (`mcts_vs_random_60games.log`): MCTS won 62.5 percent of decided games (5 wins, 3 losses, 8 decided out of 60)
- 200 games (`mcts_vs_random_200games.log`): MCTS won 73.7 percent of decided games (14 wins, 5 losses, 19 decided out of 200)
- Simulation-depth sweep (5/15/30/50 simulations, 20 games per depth, `sweep_results.csv`): MCTS won 8 of 9 decided games across all depths combined

Pooling the two larger runs gives 19 wins against 8 losses across 27 decided games, a win rate of roughly 70 percent in MCTS's favor. A binomial check against a 50 percent null (no advantage from search) puts this comfortably outside what chance alone would typically produce, suggesting a real effect from search rather than noise, though the sample remains modest.

## Why most games are draws

Across every run, 85 to 90 percent of games ended in a draw rather than a decisive result. This matches the project's own random-vs-random baseline (see `chaturanga`'s `play.py --mode random` output) and appears to be a property of weak play in this environment generally, rather than anything specific to MCTS. It limits how many decided games any given run size can produce, which is the main constraint on tightening the confidence interval above.

## Limitations and next steps

These results use an untrained network with random priors, so they measure what search depth alone contributes, not what the final trained system will achieve. The simulation-depth sweep's per-depth samples (1 to 4 decided games each) are too small to support a confident claim about how win rate scales with simulation count specifically; the pooled across-depth result is the more defensible number from the current data. A larger per-depth sweep (50+ games per depth) would be needed to speak to that question specifically.

Once the trained network from the Deep Learning module is available, this same benchmarking script can be rerun by swapping in the real network as `network_fn`, with no other changes required.

## Files in this folder

- `mcts_vs_random_60games.log` — first baseline run
- `mcts_vs_random_200games.log` — larger baseline run
- `sweep_results.csv` — simulation-depth sweep, raw numbers
- `sweep_win_rate_vs_sims.png` — simulation-depth sweep, plotted
