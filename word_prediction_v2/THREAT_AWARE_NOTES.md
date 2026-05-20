# Threat-Aware Decoder Notes

## Purpose

This document explains the architectural shift after the `78.02%` benchmarked
profile: the decoder no longer scores the current move only as "my action into a
target." It also estimates "what this revealed opponent can do back to me."

This is the first step toward a bidirectional tactical decoder.

## 1. Why this layer was added

The `78.02%` scale profile proved that:

- semantic control through battle words works
- voluntary switch suppression works
- type-aware move scoring works

But it also exposed the next ceiling. The decoder was still mostly one-sided:

- it understood my move quality
- it understood opponent typing
- it did not understand opponent threat strongly enough

That meant the policy could still take lines that were locally strong in attack
value but poor in retaliation risk.

## 2. New information source

The battle snapshot already contains:

- opponent species
- opponent status
- opponent boosts
- opponent `observed_moves`

The important unused field was `observed_moves`.

This field lets the policy estimate threat using information actually revealed
in the battle, instead of pretending the opponent is a blank typed target.

## 3. What the new decoder does

The current decoder now combines four classes of information:

1. semantic word prior
2. move-quality metadata
3. outbound matchup value
4. inbound opponent threat

### 3.1 Semantic word prior

Source:

- ranked word predictions from the battle inquiry model

Examples:

- `attack`
- `finish`
- `stabilize`
- `status`

This remains the coarse control signal.

### 3.2 Move-quality metadata

Source:

- local Showdown move data via [pokemon_move_metadata.py](/Users/AI-CCORE/alter-programming/word_prediction_model/pokemon_move_metadata.py)

Fields used:

- move category
- base power
- accuracy
- priority
- target type
- self-switch flag
- side condition / pseudo-weather flags

This lets the decoder distinguish a strong damaging move from a hazard or other
utility move using canonical simulator data rather than string-name guesses.

### 3.3 Outbound matchup value

Source:

- local type utilities

Signals used:

- type effectiveness
- STAB bonus

This measures how good my move is into the opponent.

### 3.4 Inbound opponent threat

Source:

- opponent revealed `observed_moves`

Signals used:

- revealed move type
- revealed move category
- revealed move base power
- revealed move priority
- effectiveness into my current typing

This produces a threat estimate representing how dangerous the revealed
opponent's strongest known retaliation is against my active Pokemon.

## 4. Strategic effect

The threat estimate is used to reshape the action scores:

- recovery is rewarded more when known threat is high
- setup is penalized more under high threat
- status and scouting become less attractive under strong revealed pressure
- attacking lines can be discounted slightly when the opponent is known to
  threaten a strong return hit and the position is not yet convertible

The key point is that this still does not create a full lookahead policy.
It simply stops the decoder from evaluating the turn as if only one side has
teeth.

## 5. Why this matters

This layer is important because it changes the philosophical role of the
decoder.

Before:

- semantic word -> best local move for my side

Now:

- semantic word -> best legal move under local bidirectional pressure

That is a large conceptual improvement even though the code change is still
relatively small.

## 6. Relation to the semantic controller

The threat-aware layer does not replace the word model.

The word model still decides the coarse regime:

- attack now
- finish now
- stabilize now
- spread status

The threat-aware decoder then decides whether that regime is actually safe or
correct in the present revealed tactical context.

This is exactly the architecture the project has been converging toward:

- semantic compression above
- tactical correction below

## 7. Current evidence

Qualification results after this layer was added:

- `50` games: `44 / 50` -> `88.00%`
- `200` games: `171 / 200` -> `85.50%`
- failed games: `0`

These are qualification results, not yet the final scale benchmark for this new
profile. The `10000`-game run is currently underway in:

- [word_policy_10000_threat_aware.log](/Users/AI-CCORE/alter-programming/artifacts/word_prediction_model/benchmarks/word_policy_10000_threat_aware.log)

## 8. Main interpretation

The project did not reach the `85%` qualification threshold by making the word
model broader or deeper. It did so by making the decoder more realistic about
the opponent.

That reinforces the central lesson of the project:

"The word model supplies semantic intent. The decoder determines whether that
intent survives contact with the actual tactical position."
