---
title: "ChaturangZero: A Beginner's Guide"
subtitle: "Our project explained simply, as if to a friend"
date: "November 2026"
---

# Before you start

This guide explains our project in the simplest possible way. You do not need
to know anything about computers, programming or artificial intelligence.
Whenever a technical word is needed, it is explained right there.

If you want every detail later, the long document
(`ChaturangZero_Full_Explanation.docx`) has everything. This guide is the
friendly version: read it first.

# The project in one minute

We built a computer program that plays an ancient Indian board game called
**Chaturanga**. The special thing about it is how it learned:

- **We never taught it how to play well.** We only told it the rules: how the
  pieces move and how a game is won or drawn.
- **It taught itself** by playing tens of thousands of games against itself,
  slowly figuring out which moves tend to work.

This is the same idea Google DeepMind used for **AlphaZero**, a famous
program that taught itself chess and became better than any human. We built
our own small version from scratch and called it **ChaturangZero**.

The short story of what happened:

1. We built all the parts.
2. Our first practice sessions did not make it any better. We worked out why.
3. We made practice about 11 times faster and let it practice for about six
   days straight.
4. After that, it became clearly stronger than where it started.
5. We tried several more ideas to make it even stronger. Most did not help,
   but letting it "think longer" before each move did.
6. A rated club chess player (FIDE blitz rating 1641) played 10 games against
   it and won 5, lost 2 and drew 3.

# The game: Chaturanga

Chaturanga is about 1,500 years old and comes from India. It is the
**great-grandparent of chess**. It spread to Persia, then the Arab world, then
Europe, and slowly turned into the chess people play today.

It is played on the same 8 by 8 board as chess, and the pieces start in the
same places. But most of the pieces are **weaker** than in chess:

| Piece | Like which chess piece | How it moves |
|---|---|---|
| Raja (king) | King | One step in any direction |
| Mantri (minister) | Queen's spot, but far weaker | One step diagonally only |
| Ratha (chariot) | Rook | Any distance in a straight line |
| Gaja (elephant) | Bishop's spot, but different | Jumps exactly two squares diagonally |
| Ashva (horse) | Knight | L-shaped jump, same as a knight |
| Padati (soldier) | Pawn | One step forward; captures one step diagonally; becomes a Mantri at the end of the board |

**How a game ends:**

- **Checkmate** (the king is attacked and cannot escape): that side loses.
- **Stalemate** (no legal move, but not in check): a draw.
- **100 moves in a row with no capture and no soldier move**: a draw. (We
  call this the "no-progress rule".)
- **The same position happening three times**: a draw.

**Why this matters for us.** Because most pieces are short-range, it is very
hard to actually checkmate someone, even when you are winning. So **lots of
games end in a draw**. Remember this: it turns out to be the biggest problem
in the whole project.

# How can a computer teach itself a game?

Think about how a person gets good at a game without a teacher. They play a
lot, notice which moves led to wins and which led to losses, and slowly build
up a feeling for good and bad positions.

Our program does exactly that, in three parts.

## Part 1: The rulebook

First, the program needs to know the rules perfectly: where each piece can
move, when a king is in check, when the game is over. We wrote this part by
hand, because the rules are allowed knowledge. We tested it carefully with
dozens of small checks (for example, "a horse in the corner has exactly two
moves").

## Part 2: The "gut feeling" (the neural network)

A strong player looking at a board gets an instant feeling: "these two moves
look interesting" and "I think I'm a bit better here". Our program has a
similar gut feeling, called a **neural network**.

A neural network is a big mathematical formula with about 800,000 adjustable
knobs. You give it a board position, and it gives back two things:

- **Which moves look promising** (we call this the **policy**).
- **Who is probably winning**, as a number from -1 (I'm losing) to +1 (I'm
  winning), with 0 meaning even (we call this the **value**).

At the start, all the knobs are set randomly, so its gut feeling is
nonsense. Learning means slowly turning the knobs so its feelings get better.

## Part 3: Thinking ahead (the search)

A good player does not just trust their gut. They also think: "If I go here,
they might go there, then I go here...". Our program does this too, using a
method called **tree search** (the full name is Monte Carlo Tree Search, or
MCTS).

Before each move, the program imagines possible futures 50 times. Each time,
it follows the moves that look best, using its gut feeling to decide which
moves are worth looking at and to judge how good the position at the end
looks. After 50 of these "imaginings", it picks the move it looked at most,
because that is the move it trusted most after thinking.

**Gut feeling plus thinking ahead is much stronger than either alone.** That
is the core idea of AlphaZero.

## How it learns: practice games

Now the clever part. The program plays a game against itself, thinking ahead
before every move. When the game ends, it looks back at every position from
that game and learns two lessons:

1. **"Here is what my thinking chose."** The gut feeling is adjusted to agree
   more with what the thinking-ahead decided. So the gut feeling gets
   smarter.
2. **"Here is who actually won."** The "who is winning" feeling is adjusted
   toward the real result. So if a position led to a loss, next time it will
   feel worse about similar positions.

Then it plays more games with its improved gut feeling, learns again, and so
on. One cycle of "play a batch of games, then learn from them" is called a
**round** (or an **iteration**). Over hundreds of rounds, it should get
steadily better.

**A simple comparison:** it is like a student who solves practice problems,
checks their answers, and adjusts. The thinking-ahead is the careful working;
the gut feeling is the quick intuition that slowly improves from all that
careful working.

# The rules we had to follow

Our course set some rules for the project, and we followed all of them:

- **No outside knowledge.** We could not tell the program anything about
  good play, like "a chariot is worth more than a soldier" or "control the
  center". Only the rules of the game.
- **50 "imaginings" at most per move during practice** (the official limit
  was 30 to 50). When playing for real afterward, it may think longer.
- **A small network** (the size was fixed by the course).
- **Every move is written as "from square, to square"**, which gives 64 x 64
  = 4,096 possible move codes. Most are not legal in any given position, so
  the illegal ones are simply blocked.

# What we built

In plain terms, the finished project has:

1. **The rulebook** (the game engine).
2. **The gut feeling** (the neural network).
3. **The thinking-ahead** (the tree search).
4. **The practice system** that makes it play itself and learn.
5. **Measuring tools** that play it against simple opponents and against
   older versions of itself, so we can tell whether it is improving.
6. **Two ways to play against it**: by typing moves in a text window, and by
   clicking on a board in the web browser.
7. **157 automatic checks** (called **tests**) that make sure every part
   works correctly. If anyone breaks something, a test fails and shows
   exactly what.

# How we measured "better"

To know if the program improved, we made it play many games and counted
results. We use a **score**:

score = (wins + half of the draws) / games

- **0.5 means even.** Both sides are equally strong.
- **Above 0.5** means our program is better.
- **Below 0.5** means it is worse.

**Example:** 27 wins, 0 losses and 59 draws gives (27 + 29.5) / 86 = 0.657.
That is well above 0.5.

Every score has a bit of luck in it, so we also give a **margin**, like
"0.657 plus or minus 0.049". That means the true score is very likely
somewhere between 0.608 and 0.706. Since that whole range is above 0.5, we
can say for sure our program was better. If a range includes 0.5, we cannot
tell the difference.

We measured against three simple opponents:

- **Random**: picks any legal move at random.
- **Untrained**: our own program, but with the gut feeling still random (as
  if it had never practiced).
- **Greedy capture**: always grabs the most valuable piece it can, otherwise
  moves randomly.

# First attempt: it did not work

Our first two practice sessions were 400 games each, taking about 7 hours
each on the laptop.

**Result: no improvement at all.** After practicing, the program was no
better than the untrained version, than random moves, or than the greedy
grabber. Ten extra rounds of practice did not produce a single win against
its own earlier self. And a team member who is new to chess beat it easily:
3 wins and 3 draws in 6 games.

**We did not just give up or assume a mistake.** First, we checked carefully
for bugs (using our tests) and found none. Then we investigated why it was
not learning, and found three reasons:

1. **Far too few practice games.** 400 games is tiny. The real AlphaZero
   played about 44 million games of chess.
2. **Almost every game was a draw.** About 80% of games ended in a draw.
   Remember lesson 2 from the practice games: "here is who actually won".
   If almost every game is a draw, the answer is almost always "nobody", so
   the "who is winning" feeling learns nothing. It is like a student who
   gets back every test marked "no grade": they cannot tell what they did
   right or wrong.
3. **Games were very long.** Games lasted 400 to 600 moves, so even when
   someone did win, it was hard to tell which of those hundreds of moves
   mattered.

# Making practice 11 times faster

The fixable problem was the number of games, and the limit was speed. So we
made the program much faster **without changing what it does**: the same
rules, the same thinking, the same moves, just quicker.

- **Remembering answers.** The rulebook used to recalculate the legal moves
  of a position every time it was asked. Now it works them out once and
  remembers them.
- **Smarter check detection.** To see if a king is attacked, it now looks
  outward from the king along the few lines an attacker could use, instead
  of checking every possible enemy move.
- **A faster copy of the network** for thinking ahead.
- **Playing 6 games at once**, one on each of the laptop's processor cores.
- **Using the laptop's graphics chip** (GPU) for the learning part, which is
  much faster at the kind of math networks need.

Together, one move went from 259 thousandths of a second to 19 thousandths.
We also wrote a test that compares the new rulebook with the old one on
thousands of positions, to prove they give exactly the same answers.

# The long practice: it worked

Then we let the program practice for a long time: **540 more rounds of 100
games each, so 54,000 games**, which took about **144 hours (six days)** on
the laptop.

One important trick: halfway through, the program stopped improving and kept
moving around one level. We made its learning steps smaller (like taking
smaller, more careful steps instead of big jumps), and it immediately got
better. This was the single most helpful change.

**The final version** (from round 560, saved as
`models/chaturangzero_final.pt`) gave these results:

| Against | Wins | Losses | Draws | What it means |
|---|---|---|---|---|
| Its starting self (before the long practice) | 27 | **0** | 59 | Clearly stronger |
| Random moves | 16 | **0** | 83 | Clearly stronger, never lost |
| Untrained version | 17 | 4 | 69 | Clearly stronger |
| Greedy capture | 0 | 1 | 99 | Almost every game a draw |

So practice really worked this time. It learned to **keep its pieces safe
and not lose**. But it still **rarely wins by checkmate**: most of its games
end with the 100-move no-progress rule, because it gets ahead but cannot
finish the job.

# Trying to make it even stronger

We tried several more ideas. Here is each one, simply.

**1. Let it think longer when playing.** The 50-imaginings limit is only for
practice. When playing for real, we let it imagine 800 times per move instead
of 50. Against its normal self, it won **23 games and lost 0**. Later we
tried 1600 against 800: 1600 won **15 games and lost 3**. So thinking longer
still helps, but each doubling helps less than the one before. Then 3200
against 1600 was even (7 wins, 6 losses), so thinking longer than 1600 no
longer helps at all. A move at 1600 takes about half a second, so **1600 is now the setting when people
play against it** in the browser.

**2. Tell it draws are slightly bad.** We hoped this would make it try harder
to win. It did play more games to a result, but it took bad risks and got
**weaker**. We stopped.

**3. A bigger network** (a bigger gut feeling, the largest the course
allowed). We tried twice. The second time, it first learned by studying our
best version's games, and became just as good. But practicing on its own
afterward, it **never got better** than our final version.

**4. Give up hopeless games early** (resigning), like the real AlphaZero
does. Before trying it, we checked whether our program can tell when a game
is lost. It **cannot**: even in games it went on to lose, it felt the
position was about even until just before checkmate. So it would have given
up games that were really draws. We did not use it.

**5. Even smaller learning steps, for 140 more rounds.** **No improvement.**
We kept the round-560 version.

**5b. Learning from its own look-ahead.** Most practice games are draws, so
the program is mostly told "this position was even". We tried also training
on the search's own guess of how good each position was, so even drawn games
teach it something. After 100 more rounds it was about even with round 560
(21 wins, 14 losses, 162 draws), so **no improvement**. The guesses come from
the same weak "who is winning" feeling we were trying to fix.

**5c. Shorter practice games.** Games with a winner tend to be short, and
draws go on for hundreds of moves. So we stopped practice games at 300 moves
instead of 800, which gave about 2.4 times more games with a winner per hour.
After 100 more rounds it was exactly even with round 560 (10 wins, 10 losses,
177 draws), so **no improvement** again.

**5d. Averaging several versions.** We took seven trained versions that were
all about as strong as round 560 and averaged their numbers into one. This
needs no practice at all, only a few seconds. It was about even with round
560 (16 wins, 10 losses, 173 draws), so **no improvement**.

**5e. A last round of ideas.** We then tried every other idea we know of
that keeps to the rules, one at a time (see the full explanation for the
list). First: changing how adventurous the search is (its "c_puct"
setting) from 1.5 to 1.0, 2.0 or 3.0. All three were about even with 1.5,
so **no improvement**. Next: letting three of our best
networks vote on every move. It won a few more games than it lost, but
so few that it could be luck, and it thinks three times slower, so we
**kept the single model**. Next: adding more randomness to the practice games
so the agent tries more new openings. It came out slightly weaker, so
**no improvement** either. Then: starting some practice games from the
middle of earlier games, so the agent practises more endgames. It came out
exactly as strong, so **no improvement**. Then: thinking quickly on half
of the practice moves, so it could play more practice games. Again **no
improvement**. Then: studying the games that had a winner three times as
often as the draws. Even again, so **no improvement**. Then: also learning from mirror images
of the practice positions. A tiny bit above even, but too small to count,
so **no improvement**. Then: playing a quarter of the practice games against
older versions of itself. Even, so **no improvement**. Then: a smarter way to spend the 50 thoughts
per move in practice games (called Gumbel search), which also gives the
agent a better lesson from each move. This one **beat the final model**: 17
wins against 5 losses. 200 more games went even better (23 wins against
3 losses), so it was not luck. After 60 more rounds of
practice it was stronger still: **40 wins against 3 losses** against the
final model. We are letting it practise more. Adding
the other ideas on top of Gumbel made **no difference**. Then: also teaching the agent to guess how many moves a game has
left. **No improvement**. Last: going back over recent practice games with
the newer network to give fresher lessons. **No improvement** either.

**6. Faster practice on the graphics chip.** We rebuilt the practice system
so many games share the graphics chip at once. Practice got about 2.2 to 2.4
times faster again, with exactly the same games.

**The common thread:** every idea ran into the same wall. The program's "who
is winning" feeling never learned to recognize a won or lost position,
because almost all of its practice games were draws.

# Playing against people

For the presentation, we built a version you can play **in a web browser by
clicking**. It runs on the laptop and needs no internet:

```
python play_web.py --player YOUR_NAME
```

You click a piece, it shows where the piece can go, you click the square,
and the AI answers in under a second. A side panel shows **what the AI
thinks**: who it believes is winning, and the three moves it considered most.

A rated club chess player from outside the team, with a **FIDE blitz rating of
1641** (the official international rating; 1641 is a solid club player, well
above a beginner), played **10 games** against the final
version (thinking 800 times per move), 5 as White and 5 as Black:

| | Human won | AI won | Draw |
|---|---|---|---|
| Human as White | 3 | 0 | 2 |
| Human as Black | 2 | 2 | 1 |
| **Total** | **5** | **2** | **3** |

The first version never beat even a beginner. The final version won 2
games and drew 3 against a much stronger player, and kept more of its
pieces. But the club player still won half the games, four of them quickly
(in under 110 moves).

**What a typical loss looks like for the AI.** In one game, the AI jumped its
horse right next to the human's king, where the king simply captured it.
Later it lost its elephant and chariot for nothing. At the very end, while
the human brought both chariots toward its king, the AI kept moving one
elephant back and forth between the same two squares, not realizing it was
in danger, until it was checkmated.

**Why does this happen?** The AI learned mostly from slow, drawn games against
itself. It almost never saw a direct attack on its king during practice, so
it does not recognize the danger in time.

# The big lessons

1. **Our first failure was about practice, not a mistake in the code.**
   With 135 times more practice games, the same program clearly improved.
2. **Smaller learning steps at the right time matter a lot.**
3. **Draws were the real enemy.** Because this game is so drawish, the
   program got very little "you won" or "you lost" feedback, and so its sense
   of who is winning stayed weak.
4. **On one laptop, within the course rules, the program has reached about
   its limit.** More of the same practice does not help. Thinking longer at
   play time does.

# How AlphaZero and our project compare

| | AlphaZero (chess) | ChaturangZero |
|---|---|---|
| Practice games | About 44 million | 54,000 |
| Imaginings per move in practice | 800 | 50 |
| Computer | Thousands of special chips in a data center | One laptop |
| Knowledge given | Only the rules | Only the rules |
| Result | Better than any human | Better than random and its earlier selves; a rated club player (FIDE blitz 1641) still wins half the games |

Same method, a tiny fraction of the resources. Given that, the fact that the
long practice clearly worked, and that we could measure and explain its
limits, is the main achievement.

# Showing it in the presentation

**A 30-second version you can say:**

"We built an AlphaZero-style AI for Chaturanga, the ancestor of chess. It only
knows the rules and teaches itself by playing against itself. At first it did
not improve, because almost every game was a draw and it had too little
practice. We made it 11 times faster and trained it for 54,000 games. It then
beat its starting version 27 to 0. It still struggles to checkmate, and a
rated club player beat it in half of ten games, because it never learned to recognize when a game is
won or lost."

**Running the live demo:**

1. Open the Terminal in the project folder.
2. Type `python play_web.py --player Demo --no_log` and press Enter.
   (`--no_log` means demo games are not saved to our results.)
3. The browser opens the game. If it does not, go to http://127.0.0.1:8000.
4. Play a few moves. Point out how pieces show their possible moves, how fast
   the AI answers, and the "What the agent thinks" panel.
5. Press Ctrl+C in the Terminal to stop.

# Questions you might be asked, with simple answers

**Did you give it any chess knowledge?** No, only the rules. That was a
course requirement.

**Why didn't it improve at first?** Too few practice games, and about 80% of
them were draws, so it had almost nothing to learn from.

**How do you know it got better later?** It played 100 games against its
earlier self and won 27, lost 0, drew 59. That is far beyond luck.

**Why does it draw so much?** In Chaturanga most pieces are short-range, so
checkmate is hard to force. Our AI learned not to lose, but not how to finish
a won game.

**Why can a club player beat it?** It practiced mostly through slow, drawn games,
so it never learned to defend against a direct attack on its king.

**Why not just practice more?** We did: 140 more rounds gave no improvement.
We also tried a bigger network and other ideas; none helped.

**Did making it faster change how it plays?** No. Tests prove the faster
version makes exactly the same moves.

**What would you do with more time or computers?** Many more practice games
with the bigger network, and a way to get more decisive (non-draw) games
within the rules, for example by making practice games shorter.

**What is the difference between thinking 50 times and 800 times?** It
imagines more possible futures before each move, so it makes fewer mistakes.
At 800 it beat its 50 self 23 to 0, and at 1600 it beat its 800 self 15 to 3.
At 3200 it was no better than at 1600, so 1600 is where the gain stops.

**Is the program you're showing the same one in the report?** Yes, the same
final version. The browser game now thinks 1600 times per move; the ten
games against the chess player were played at 800, before that change.

# Small word list

| Word | What it means |
|---|---|
| AI | Artificial intelligence: a program that does something that seems smart |
| AlphaZero | Google DeepMind's famous self-taught game program, which ours copies in a small way |
| Neural network | The program's "gut feeling": a big formula with adjustable knobs |
| Policy | The network's sense of which moves look promising |
| Value | The network's sense of who is winning (from -1 to +1) |
| Tree search (MCTS) | Thinking ahead by imagining possible futures before each move |
| Imagining (simulation) | One look into a possible future during the search |
| Round (iteration) | One batch of practice games followed by learning from them |
| Practice game (self-play) | A game the program plays against itself to learn |
| Learning step (learning rate) | How big each adjustment to the network is |
| Score | (wins + half the draws) / games; 0.5 is even |
| Margin | How much luck could be in a score |
| GPU (graphics chip) | Part of the computer that is very fast at the math networks need |
| Test | An automatic check that a part of the program works |
| Checkpoint | A saved copy of the network after a round |
| No-progress rule | 100 moves with no capture or soldier move is a draw |
| Resign | Give up a game you think is lost |
