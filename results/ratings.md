# Ratings of all models

Made with `python -m evaluation.elo_table` (see the docstring there). Elo-style ratings from every standard head-to-head match (50 simulations, temperature 0.3, repetition avoidance, 800-ply cap), draws as half a win, iteration 560 fixed at 0. The 90% ranges come from 100 resamples of the games. `--all` lists every checkpoint.

141 matches, 24217 games
| Player | Rating | 90% range | Games |
|---|---|---|---|
| Iteration 1340 (final model) | 298 | 277 to 324 | 998 |
| Iteration 1280, 32 root moves (third final model) | 272 | 252 to 287 | 1795 |
| Tweak: Gumbel value scale 0.03 (1280) | 265 | 244 to 291 | 796 |
| Final model trained 60 more rounds (1340) | 257 | 232 to 291 | 399 |
| Averaged weights of the 1220 chain (soup) | 255 | 220 to 300 | 200 |
| Tweak: weight decay 1e-5 (1280) | 247 | 223 to 272 | 795 |
| Tweak: no repetition avoidance (1280) | 241 | 219 to 261 | 794 |
| Iteration 1220 (second final model) | 236 | 227 to 249 | 3788 |
| Tweak: 150 games x 32 simulations (1280) | 235 | 205 to 273 | 397 |
| Iteration 1280, 16 root moves | 233 | 218 to 255 | 1394 |
| Gumbel, learning rate 3e-5, iteration 1160 | 204 | 186 to 225 | 996 |
| Tweak: weight decay 1e-3 (1280) | 198 | 169 to 227 | 395 |
| Tweak: replay buffer 100k (1280) | 186 | 163 to 215 | 399 |
| Tweak: 1000 training steps (1280) | 181 | 150 to 211 | 397 |
| Gumbel, iteration 1100 | 179 | 164 to 196 | 2191 |
| Tweak: Gumbel value scale 0.3 (1280) | 173 | 149 to 198 | 398 |
| Tweak: learning rate 3e-4 (1280) | 117 | 85 to 141 | 400 |
| Gumbel, iteration 800 | 110 | 92 to 132 | 1596 |
| From scratch with every idea, iteration 200 (experiment) | 57 | 35 to 87 | 791 |
| Bigger network (6 x 128) from scratch, iteration 200 (experiment) | 34 | 12 to 56 | 594 |
| Gumbel self-play, iteration 620 | 33 | 13 to 56 | 799 |
| Iteration 560 (first final model) | 0 | 0 to 0 | 10395 |
| First long run, iteration 420 | -25 | -112 to 66 | 365 |
| First long run, iteration 200 | -37 | -73 to 10 | 198 |
