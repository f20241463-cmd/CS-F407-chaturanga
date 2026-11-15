# Ratings of all models

Made with `python -m evaluation.elo_table` (see the docstring there). Elo-style ratings from every standard head-to-head match (50 simulations, temperature 0.3, repetition avoidance, 800-ply cap), draws as half a win, iteration 560 fixed at 0. The 90% ranges come from 100 resamples of the games. `--all` lists every checkpoint.

130 matches, 22030 games
| Player | Rating | 90% range | Games |
|---|---|---|---|
| Iteration 1280, 32 root moves (final model) | 276 | 254 to 294 | 1396 |
| Tweak: Gumbel value scale 0.03 (1280) | 267 | 245 to 292 | 796 |
| Final model trained 60 more rounds (1340) | 260 | 217 to 297 | 399 |
| Averaged weights of the 1220 chain (soup) | 256 | 217 to 302 | 200 |
| Tweak: weight decay 1e-5 (1280) | 248 | 222 to 274 | 795 |
| Tweak: no repetition avoidance (1280) | 242 | 216 to 268 | 794 |
| Iteration 1220 (second final model) | 236 | 223 to 255 | 3389 |
| Tweak: 150 games x 32 simulations (1280) | 235 | 207 to 270 | 397 |
| Iteration 1280, 16 root moves | 233 | 213 to 256 | 1194 |
| Gumbel, learning rate 3e-5, iteration 1160 | 204 | 186 to 227 | 996 |
| Tweak: weight decay 1e-3 (1280) | 198 | 163 to 232 | 395 |
| Tweak: replay buffer 100k (1280) | 186 | 155 to 219 | 399 |
| Tweak: 1000 training steps (1280) | 181 | 153 to 219 | 397 |
| Gumbel, iteration 1100 | 179 | 166 to 196 | 2191 |
| Tweak: Gumbel value scale 0.3 (1280) | 173 | 142 to 208 | 398 |
| Tweak: learning rate 3e-4 (1280) | 117 | 88 to 152 | 400 |
| Gumbel, iteration 800 | 110 | 91 to 126 | 1596 |
| Gumbel self-play, iteration 620 | 34 | 8 to 59 | 599 |
| Iteration 560 (first final model) | 0 | 0 to 0 | 9800 |
| First long run, iteration 420 | -25 | -97 to 55 | 365 |
