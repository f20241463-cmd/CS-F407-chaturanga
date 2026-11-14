# Ratings of all models

Made with `python -m evaluation.elo_table` (see the docstring there). Elo-style ratings from every standard head-to-head match (50 simulations, temperature 0.3, repetition avoidance, 800-ply cap), draws as half a win, iteration 560 fixed at 0. The 90% ranges come from 100 resamples of the games. `--all` lists every checkpoint.

120 matches, 20040 games
| Player | Rating | 90% range | Games |
|---|---|---|---|
| Iteration 1280, 32 root moves (final model) | 275 | 249 to 309 | 798 |
| Tweak: no repetition avoidance (1280) | 259 | 229 to 301 | 397 |
| Averaged weights of the 1220 chain (soup) | 256 | 209 to 290 | 200 |
| Tweak: weight decay 1e-5 (1280) | 247 | 225 to 274 | 795 |
| Iteration 1220 (second final model) | 237 | 221 to 254 | 2990 |
| Tweak: 150 games x 32 simulations (1280) | 235 | 208 to 275 | 397 |
| Iteration 1280, 16 root moves | 229 | 200 to 261 | 798 |
| Gumbel, learning rate 3e-5, iteration 1160 | 204 | 184 to 228 | 996 |
| Tweak: weight decay 1e-3 (1280) | 198 | 165 to 228 | 395 |
| Tweak: replay buffer 100k (1280) | 187 | 157 to 218 | 399 |
| Tweak: 1000 training steps (1280) | 181 | 153 to 218 | 397 |
| Gumbel, iteration 1100 | 179 | 164 to 199 | 2191 |
| Tweak: learning rate 3e-4 (1280) | 117 | 85 to 146 | 400 |
| Gumbel, iteration 800 | 110 | 95 to 127 | 1596 |
| Gumbel self-play, iteration 620 | 34 | 10 to 61 | 599 |
| Iteration 560 (first final model) | 0 | 0 to 0 | 9203 |
| First long run, iteration 420 | -25 | -133 to 55 | 365 |
