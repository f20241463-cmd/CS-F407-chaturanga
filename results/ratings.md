# Ratings of all models

Made with `python -m evaluation.elo_table` (see the docstring there). Elo-style ratings from every standard head-to-head match (50 simulations, temperature 0.3, repetition avoidance, 800-ply cap), draws as half a win, iteration 560 fixed at 0. The 90% ranges come from 100 resamples of the games. `--all` lists every checkpoint.

124 matches, 20835 games
| Player | Rating | 90% range | Games |
|---|---|---|---|
| Iteration 1280, 32 root moves (final model) | 283 | 263 to 310 | 997 |
| Averaged weights of the 1220 chain (soup) | 258 | 221 to 313 | 200 |
| Tweak: weight decay 1e-5 (1280) | 252 | 236 to 278 | 795 |
| Tweak: no repetition avoidance (1280) | 246 | 224 to 273 | 794 |
| Iteration 1220 (second final model) | 239 | 227 to 257 | 3190 |
| Tweak: 150 games x 32 simulations (1280) | 237 | 203 to 272 | 397 |
| Iteration 1280, 16 root moves | 236 | 214 to 260 | 996 |
| Gumbel, learning rate 3e-5, iteration 1160 | 206 | 186 to 226 | 996 |
| Tweak: weight decay 1e-3 (1280) | 200 | 165 to 230 | 395 |
| Tweak: replay buffer 100k (1280) | 188 | 161 to 222 | 399 |
| Tweak: 1000 training steps (1280) | 182 | 152 to 217 | 397 |
| Gumbel, iteration 1100 | 180 | 163 to 195 | 2191 |
| Tweak: Gumbel value scale 0.3 (1280) | 175 | 150 to 202 | 398 |
| Tweak: learning rate 3e-4 (1280) | 119 | 92 to 148 | 400 |
| Gumbel, iteration 800 | 110 | 90 to 129 | 1596 |
| Gumbel self-play, iteration 620 | 34 | 11 to 56 | 599 |
| Iteration 560 (first final model) | 0 | 0 to 0 | 9401 |
| First long run, iteration 420 | -25 | -118 to 70 | 365 |
