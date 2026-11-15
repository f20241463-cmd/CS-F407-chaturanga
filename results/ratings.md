# Ratings of all models

Made with `python -m evaluation.elo_table` (see the docstring there). Elo-style ratings from every standard head-to-head match (50 simulations, temperature 0.3, repetition avoidance, 800-ply cap), draws as half a win, iteration 560 fixed at 0. The 90% ranges come from 100 resamples of the games. `--all` lists every checkpoint.

126 matches, 21234 games
| Player | Rating | 90% range | Games |
|---|---|---|---|
| Iteration 1280, 32 root moves (final model) | 274 | 252 to 296 | 1197 |
| Final model trained 60 more rounds (1340) | 259 | 227 to 295 | 399 |
| Averaged weights of the 1220 chain (soup) | 256 | 210 to 302 | 200 |
| Tweak: weight decay 1e-5 (1280) | 247 | 227 to 278 | 795 |
| Tweak: no repetition avoidance (1280) | 241 | 221 to 268 | 794 |
| Iteration 1220 (second final model) | 237 | 224 to 252 | 3190 |
| Tweak: 150 games x 32 simulations (1280) | 235 | 204 to 272 | 397 |
| Iteration 1280, 16 root moves | 231 | 216 to 259 | 996 |
| Gumbel, learning rate 3e-5, iteration 1160 | 204 | 185 to 221 | 996 |
| Tweak: weight decay 1e-3 (1280) | 198 | 167 to 226 | 395 |
| Tweak: replay buffer 100k (1280) | 187 | 155 to 222 | 399 |
| Tweak: 1000 training steps (1280) | 181 | 150 to 215 | 397 |
| Gumbel, iteration 1100 | 179 | 164 to 198 | 2191 |
| Tweak: Gumbel value scale 0.3 (1280) | 173 | 151 to 206 | 398 |
| Tweak: learning rate 3e-4 (1280) | 117 | 83 to 148 | 400 |
| Gumbel, iteration 800 | 110 | 93 to 131 | 1596 |
| Gumbel self-play, iteration 620 | 34 | 11 to 58 | 599 |
| Iteration 560 (first final model) | 0 | 0 to 0 | 9600 |
| First long run, iteration 420 | -25 | -109 to 57 | 365 |
