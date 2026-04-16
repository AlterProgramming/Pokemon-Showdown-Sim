# Random Baseline Ceiling

## Question

What is the practical upper limit against the local random opponent in this benchmark harness, independent of any one model family?

## Short Answer

The ceiling against this local random baseline is likely very high, but not literally `100%`.

A strong policy should be expected to win the large majority of games, because the opponent is not making structured competitive decisions. But a perfect sweep over an unlimited sample is implausible because:

- teams are random on both sides
- the game itself contains accuracy, damage-roll, crit, and status variance
- hidden information still exists early in games
- some generated team and lead configurations will be structurally awkward even for a strong agent

So the right research framing is not "can a good agent reach 100%?" but "how much irreducible loss remains once the agent stops making strategic mistakes?"

## What The Local Random Bot Actually Does

The most important discovery is that the benchmark opponent is weaker, and also narrower, than the phrase "random AI" might suggest.

In [random-player-ai.js](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/dist/sim/tools/random-player-ai.js):

- legal moves are sampled uniformly from available legal moves
- forced switches are sampled uniformly from legal switch targets
- voluntary switches happen only if `prng.random() > this.move`

That last point matters. In the benchmark path used here, the statistical runner creates the opponent as:

- `new RandomPlayerAI(streams.p1)`

with no custom options in [statistical-runner.js](/Users/AI-CCORE/alter-programming/pokemon-showdown-model-feature/dist/sim/examples/statistical-runner.js).

Inside `RandomPlayerAI`, the default is:

- `this.move = options.move || 1`

and the voluntary-switch condition is:

- switch only if `prng.random() > this.move`

Since `this.move` defaults to `1`, and `prng.random()` is in `[0, 1)`, the random bot effectively never voluntarily switches in this benchmark harness. It attacks randomly, and it only switches when forced.

This means the benchmark baseline is not "random over all legal actions." It is much closer to:

- random legal move selection
- no strategic repositioning
- random forced replacement only

That makes the opponent very exploitable by any policy that can do three things consistently:

1. avoid unnecessary switching
2. choose stronger attacks than chance
3. convert winning positions instead of looping

## Why The Ceiling Is Very High

Against a move-random, no-voluntary-switch opponent, a competent agent gains advantage from structure alone.

First, Pokemon move quality is highly uneven. Legal move sets are not flat action spaces. Some moves are clearly better than others in the current state:

- stronger STAB attacks
- super-effective attacks
- accurate cleanup attacks
- recovery at the correct time
- setup only in safe windows

A random mover throws this structure away. Even simple tactical competence should dominate it over time.

Second, voluntary switching is one of the main ways to recover from bad matchups, absorb threats, or preserve win conditions. A random opponent that never voluntarily switches is strategically brittle. It leaves bad actives in play, fails to pivot out of losing exchanges, and often lets a stronger side press the same local advantage repeatedly.

Third, random play wastes information. As turns pass, a stronger agent accumulates revealed moves, role clues, HP information, and endgame structure. The random bot does not use that information coherently, so the information gap widens over time instead of narrowing.

All of this implies that once an agent reaches a basic threshold of tactical competence, win rate should rise quickly and then flatten only near the point where losses are driven mostly by variance rather than bad decisions.

## Why The Ceiling Is Not 100%

Even against this weak baseline, some losses are unavoidable in practice.

### 1. Team-generation variance

This is `gen9randombattle`, so both sides receive randomly generated teams. Some games will start from materially worse conditions:

- poor internal synergy
- awkward lead matchups
- weak answers to specific threats
- bad role compression

A strong player can recover from many such states, but not all of them.

### 2. Simulator variance

Pokemon is not deterministic. Even correct play can lose to:

- misses
- crits
- low or high damage rolls
- status proc timing
- secondary effects
- speed ties where relevant

This creates an irreducible loss floor.

### 3. Hidden information

Even a strong policy begins each game with incomplete knowledge:

- exact opposing moves
- item and ability details
- some role distinctions

Against a random opponent, this matters less than against a strong one, but it still matters. A policy can make the correct decision in expectation and still lose the realized game.

### 4. Action-space imperfections

Any real controller will still make some mistakes:

- wrong move among several attacks
- premature recovery
- bad closeout sequencing
- failure to trade correctly

As performance rises, these mistakes get rarer, but the remaining ones become expensive. That is why the curve tends to flatten.

## A Better Ceiling Estimate

For this exact baseline, the practical ceiling should be thought of in bands.

### Below `60%`

This usually means the agent is still making large structural mistakes:

- excessive switching
- poor move ranking
- bad legality handling
- unstable policy modes

### `60%` to `80%`

This is strong exploitation of the random bot, but still with significant tactical leakage. The agent is winning because it is more coherent than the opponent, not because it converts nearly every advantage.

### `80%` to `90%`

This is where the interesting ceiling question begins. An agent here is no longer merely "better than random." It is exploiting the baseline reliably. Remaining losses are increasingly a mixture of:

- narrow tactical errors
- awkward team-generation states
- irreducible simulator variance

### `90%+`

This should be possible only if the policy is extremely stable and converts advantages very cleanly. At that point, most remaining losses should look like:

- severe matchup disadvantage
- heavy variance
- very narrow tactical misses

### `100%`

This is not a realistic expectation for large samples in a stochastic random-battle environment.

## What This Means Conceptually

The upper limit against random is not mainly a question of raw model intelligence. It is a question of how much of the game’s exploitable structure the policy captures before variance takes over as the dominant remaining source of loss.

Against this baseline, the random opponent is giving away value in several ways:

- it does not rank moves intelligently
- it does not reposition voluntarily
- it does not protect long-term resources
- it does not convert information into cleaner later turns

So a high-performing agent should not be judged only by whether it beats random. It should be judged by where its losses come from once it already beats random a lot.

If the losses are still mostly strategic, the policy is far from the ceiling.
If the losses are mostly variance and awkward team-generation states, the policy is near the ceiling.

## Working Conclusion

Independent of the current word-model project, the local random baseline appears to permit a very high ceiling, probably well above ordinary "good enough" performance, because the opponent is effectively a move-random, no-voluntary-switch bot.

The realistic research conclusion is:

- the ceiling is high
- the ceiling is not perfect
- once a policy reaches the high-80s, progress becomes expensive because the remaining loss pool is increasingly narrow and variance-heavy

That makes the right question:

- not "why is the agent failing to crush random at all?"
- but "what fraction of the remaining losses are still strategic and therefore still recoverable?"
