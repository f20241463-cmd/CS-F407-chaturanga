# Ratings of all models

Made with `python -m evaluation.elo_table` (see the docstring there). Elo-style ratings from every standard head-to-head match (50 simulations, temperature 0.3, repetition avoidance, 800-ply cap), draws as half a win, iteration 560 fixed at 0. The 90% ranges come from 100 resamples of the games. `--all` lists every checkpoint.

138 matches, 23623 games
| Player | Rating | 90% range | Games |
|---|---|---|---|
| Iteration 1340 (final model) | 297 | 278 to 317 | 998 |
| Iteration 1280, 32 root moves (third final model) | 271 | 255 to 292 | 1795 |
| Tweak: Gumbel value scale 0.03 (1280) | 265 | 243 to 286 | 796 |
| Final model trained 60 more rounds (1340) | 257 | 224 to 297 | 399 |
| Averaged weights of the 1220 chain (soup) | 254 | 207 to 290 | 200 |
| Tweak: weight decay 1e-5 (1280) | 246 | 227 to 270 | 795 |
| Tweak: no repetition avoidance (1280) | 240 | 218 to 270 | 794 |
| Iteration 1220 (second final model) | 235 | 223 to 249 | 3588 |
| Tweak: 150 games x 32 simulations (1280) | 234 | 198 to 277 | 397 |
| Iteration 1280, 16 root moves | 232 | 218 to 252 | 1394 |
| Gumbel, learning rate 3e-5, iteration 1160 | 203 | 184 to 225 | 996 |
| Tweak: weight decay 1e-3 (1280) | 197 | 170 to 232 | 395 |
| Tweak: replay buffer 100k (1280) | 186 | 154 to 216 | 399 |
| Tweak: 1000 training steps (1280) | 180 | 154 to 213 | 397 |
| Gumbel, iteration 1100 | 179 | 164 to 199 | 2191 |
| Tweak: Gumbel value scale 0.3 (1280) | 172 | 150 to 201 | 398 |
| Tweak: learning rate 3e-4 (1280) | 117 | 86 to 148 | 400 |
| Gumbel, iteration 800 | 109 | 96 to 131 | 1596 |
| From scratch with every idea, iteration 200 (experiment) | 53 | 28 to 85 | 595 |
| Gumbel self-play, iteration 620 | 32 | 10 to 57 | 799 |
| Iteration 560 (first final model) | 0 | 0 to 0 | 10197 |
| First long run, iteration 420 | -25 | -100 to 66 | 365 |
| First long run, iteration 200 | -40 | -80 to 32 | 198 |
