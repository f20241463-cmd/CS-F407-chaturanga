# Ratings of all models

Made with `python -m evaluation.elo_table` (see the docstring there). Elo-style ratings from every standard head-to-head match (50 simulations, temperature 0.3, repetition avoidance, 800-ply cap), draws as half a win, iteration 560 fixed at 0. The 90% ranges come from 100 resamples of the games. `--all` lists every checkpoint.

135 matches, 23028 games
| Player | Rating | 90% range | Games |
|---|---|---|---|
| Iteration 1340 (final model) | 297 | 271 to 320 | 998 |
| Iteration 1280, 32 root moves (third final model) | 271 | 251 to 292 | 1795 |
| Tweak: Gumbel value scale 0.03 (1280) | 265 | 239 to 290 | 796 |
| Final model trained 60 more rounds (1340) | 257 | 225 to 289 | 399 |
| Averaged weights of the 1220 chain (soup) | 254 | 215 to 298 | 200 |
| Tweak: weight decay 1e-5 (1280) | 246 | 219 to 272 | 795 |
| Tweak: no repetition avoidance (1280) | 240 | 216 to 267 | 794 |
| Iteration 1220 (second final model) | 235 | 225 to 249 | 3588 |
| Tweak: 150 games x 32 simulations (1280) | 234 | 199 to 270 | 397 |
| Iteration 1280, 16 root moves | 232 | 211 to 254 | 1394 |
| Gumbel, learning rate 3e-5, iteration 1160 | 203 | 184 to 225 | 996 |
| Tweak: weight decay 1e-3 (1280) | 197 | 168 to 225 | 395 |
| Tweak: replay buffer 100k (1280) | 186 | 153 to 213 | 399 |
| Tweak: 1000 training steps (1280) | 180 | 155 to 223 | 397 |
| Gumbel, iteration 1100 | 179 | 167 to 194 | 2191 |
| Tweak: Gumbel value scale 0.3 (1280) | 172 | 146 to 207 | 398 |
| Tweak: learning rate 3e-4 (1280) | 117 | 92 to 143 | 400 |
| Gumbel, iteration 800 | 110 | 90 to 128 | 1596 |
| Gumbel self-play, iteration 620 | 34 | 15 to 62 | 599 |
| Iteration 560 (first final model) | 0 | 0 to 0 | 10000 |
| First long run, iteration 420 | -25 | -129 to 47 | 365 |
