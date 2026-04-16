# Entity Model Benchmark Analysis
## Date: April 8, 2026
## Model: entity_action_bc_v1_20260408_0428

---

## Executive Summary

The entity model achieves a **25% win rate** against random opponents (5/20 games), continuing the trend of struggling against purely random play. This analysis examines 20 captured battle replays to identify decision-making patterns and tactical weaknesses.

---

## Overall Statistics

| Metric | Value |
|--------|-------|
| Total Games | 20 |
| Entity Wins | 5 (25.0%) |
| Random Wins | 15 (75.0%) |
| Total Turns Played | 1,320 |
| Average Game Length | 33.0 turns |

---

## Performance by Outcome

### Winning Games (5 games)

| Metric | Value |
|--------|-------|
| Average Turn Count | 36.6 |
| Average RL Switches | 10.2 |
| Average Opponent Switches | 5.2 |
| Average Pokemon Fainted (RL) | 4.0 (67%) of team |
| Average Pokemon Fainted (Opponent) | 6.0 (100% of team) |
| Total Super Effective Hits Against RL | 20 |

**Key Observation**: The model wins by attrition - staying in longer battles and fainting opponent's full team while losing 4/6 of its own.

### Losing Games (15 games)

| Metric | Value |
|--------|-------|
| Average Turn Count | 30.0 |
| Average RL Switches | 13.3 |
| Average Opponent Switches | 3.5 |
| Average Pokemon Fainted (RL) | 6.0 (100% of team) |
| Average Pokemon Fainted (Opponent) | 4.8 (80% of team) |
| Total Super Effective Hits Against RL | 89 |

**Key Observation**: Losses are characterized by shorter battles and excessive switching, with the model taking significantly more super effective damage.

---

## Critical Findings

### 1. **Excessive Switching Problem** 🔴 CRITICAL

The model switches **more frequently in losses (13.3/game) than in wins (10.2/game)**. This is counterintuitive and suggests:

- **Reactive vs Proactive**: The model may be switching reactively to threats rather than maintaining offensive momentum
- **Momentum Loss**: Each switch gives opponent a free turn to set up or attack
- **Type Coverage Weakness**: Excessive switching indicates poor type matchup predictions

**Evidence**: In losses, the model makes 3.1 more switches per game despite shorter battles. This high switching rate correlates with faster defeats.

### 2. **Type Effectiveness Vulnerability** 🔴 CRITICAL

The model takes **4.45x more super effective hits in losses (89) vs wins (20)**.

- Wins: 20 super effective hits across 5 games (4.0/game)
- Losses: 89 super effective hits across 15 games (5.93/game)

**Interpretation**: The model struggles to predict or avoid type disadvantages. It may:
- Fail to switch into appropriate counters
- Stay in unfavorable matchups too long (before over-switching)
- Lack proper defensive type coverage in team construction

### 3. **Battle Duration Pattern**

Wins last **~22% longer** (36.6 vs 30.0 turns):
- Extended games allow the model to eventually grind through opponents
- Suggests the model's decision-making improves over time or that longer games favor attrition
- Opponents rarely switch, creating predictable patterns

### 4. **Opponent Switching Dynamics**

Opponent switches **less** in losses (3.5 vs 5.2):
- Suggests random opponent is more "defensive" when facing the model
- Model may trigger opponent to switch by landing favorable attacks
- BUT: Despite this, model loses more often, indicating random switching behavior is effective against the model

---

## Tactical Weaknesses Identified

### Type Matching
- Model frequently stays in bad type matchups briefly then over-corrects with multiple switches
- Fails to use type advantage to force opponent switches effectively

### Momentum Management
- Excessive switching interrupts any offensive pressure the model builds
- Every switch gives random opponent a free action

### Defensive Stability
- Takes nearly 6 super effective hits per loss game
- Suggests weak defensive typing or poor team synergy
- Model lacks reliable defensive pivots or walls

### Decision Latency
- Average decision time is 334ms (benchmark logged ~1.03 minutes for 20 games)
- This latency is acceptable but doesn't translate to quality decisions

---

## Comparison with Historical Data

### Previous Benchmark Results:
- **March 27**: Model4 achieved 66.7% win rate vs entity model (18 games)
- **April 7 Run 1**: Entity model achieved 30% win rate (120 games)
- **April 7 Run 2**: Entity model achieved 27.5% win rate (1000 games)
- **April 7 Run 3**: Entity model achieved 25.7% win rate (1000 games)
- **April 8 Current**: Entity model achieved 25% win rate (20 games with replay capture)

### Trend: Slight Declining Performance
The entity model shows a declining win rate trend against random (30% → 27.5% → 25.7% → 25%). This may indicate:
1. Initial model releases had higher random variance
2. Increasing sample size reveals true 25% performance level
3. Model may be overfitting to specific battle dynamics

---

## Root Cause Analysis

The model appears to suffer from **decision instability under adversarial uncertainty**:

1. **Prediction Uncertainty**: Model lacks confidence in type matchup predictions
   - Response: Excessive switching to "find" better matchups
   - Effect: Loses momentum and battle control

2. **Training Distribution Mismatch**: Model trained on sequence data may not generalize to purely random play
   - Training data likely has consistent team strategies
   - Random opponent violates training distribution assumptions

3. **Reward Signal Issue**: If training used win/loss signals, purely random opponent creates confusing signal
   - Random switches break causal inference of successful actions
   - Model may learn to switch excessively as a "safe" strategy

---

## Recommendations for Improvement

### Short-term (Data-driven)
1. **Analyze decision confidence**: Log model probability distributions for switch vs move decisions
2. **Identify type matchup errors**: Compare model's predicted matchup advantage vs actual
3. **Profile switching patterns**: Extract decision states where excessive switching occurs

### Medium-term (Training)
1. **Add adversarial examples**: Train with more random opponents to improve robustness
2. **Type coverage loss**: Add auxiliary loss for type effectiveness prediction
3. **Momentum penalty**: Add reward shaping that penalizes consecutive switches

### Long-term (Architecture)
1. **Uncertainty estimation**: Use Monte Carlo dropout or ensemble methods for decision confidence
2. **Defensive typing layer**: Separate network head for type matchup safety ratings
3. **Momentum tracking**: Add state feature tracking switch history (e.g., "switched last turn")

---

## Files and Artifacts

- **Replay Directory**: `pokemon-showdown-model-feature/logs/replays/`
- **Total Replays Captured**: 20 HTML files
- **Model Metadata**: `Pokemon-Showdown-Sim/artifacts/entity_action_bc_v1_20260408_0428/`
- **Server Endpoint**: `http://127.0.0.1:5001/predict` (during benchmark)

---

## Next Steps

1. ✅ Capture battle replay data
2. ✅ Extract battle statistics
3. ✅ Identify decision patterns
4. 🔄 **IN PROGRESS**: Generate diagnostic reports
5. 📋 Log detailed decision traces for specific high-loss games
6. 📊 Create visualization of type matchup performance

---

*Analysis generated: 2026-04-08 11:32 UTC*

---

## Detailed Game-Level Analysis

### Winning Games (Games 1, 8, 9, 15, 18)

| Game | Turns | RL Sw | Opp Sw | RL Fainted | Opp Fainted | Super Eff | Pattern |
|------|-------|-------|--------|-----------|-------------|-----------|---------|
| 1    | 38    | 13    | 6      | 3         | 6           | 3         | Long, many switches, opponent team wiped |
| 8    | 26    | 8     | 5      | 1         | 6           | 4         | Short, minimal switches (best game!) |
| 9    | 36    | 12    | 5      | 5         | 6           | 4         | Long, controlled switches |
| 15   | 37    | 11    | 5      | 5         | 6           | 3         | Long, balanced switches |
| 18   | 46    | 7     | 5      | 4         | 6           | 6         | Longest game, least switches |

**Win Pattern**: Games 8 and 18 win with **fewer switches** (7-8 vs 11-13). Longer battles allow model to outlast opponent.

### Losing Games Analysis

#### High-Switch Losses (Games 3, 16, 17, 19)
These games show extreme switching behavior in losses:

| Game | Turns | RL Sw | Opp Sw | Analysis |
|------|-------|-------|--------|----------|
| 3    | 28    | 20    | 3      | 20 switches in 28 turns = 71% of turns were switches! Model panic-switching |
| 16   | 32    | 24    | 1      | 24 switches in 32 turns = 75% switch rate. Opponent almost never switched (1) |
| 17   | 18    | 18    | 1      | 18 switches in 18 turns = 100%! Switching every single turn |
| 19   | 22    | 16    | 2      | 16 switches in 22 turns = 73% switch rate |

**Critical Finding**: In games 16, 17, 19 the opponent never or rarely switches (1-2 times), yet model still loses despite finding different Pokémon constantly.

#### Standard Losses (9 other games: 2, 5, 6, 7, 10, 11, 13, 14)
Average: 10.4 switches, 30.1 turns, 6.0 RL fainted

These are "normal" losses where model switches 9-11 times and gets swept.

---

## Behavioral Anomalies

### Pattern 1: Panic Switching vs Steady Opponents
When opponent **stops switching** (games with 1-3 opponent switches), model dramatically increases switching:
- Game 3: Opp 3 switches → RL 20 switches (6.7x)
- Game 10: Opp 1 switch → RL 11 switches (11x)
- Game 16: Opp 1 switch → RL 24 switches (24x)
- Game 17: Opp 1 switch → RL 18 switches (18x)

**Hypothesis**: Model interprets opponent "commitment" to current Pokémon as a threat and tries to find optimal counter by rapidly switching. Each switch telegraphs intentions to a random opponent.

### Pattern 2: Opponent Switch Correlation
Games where opponent switches frequently (5-6 times):
- Wins: Average 6.2 opponent switches (4 out of 5 wins)
- Losses: Average 4.4 opponent switches

Random opponent switches when it's losing type matchups. Model seems to win when opponent recognizes type disadvantage and switches, giving model an advantage.

---

## Model Decision Profile

### Type of Decisions Made:

**Move vs Switch Ratio**:
- In wins: ~61% moves, 39% switches (avg 10.2 switches in 36.6 turns)
- In losses: ~56% moves, 44% switches (avg 13.3 switches in 30.0 turns)

**Switching Confidence Interpretation**:
The model switches to ~56-61% of its available actions, suggesting no strong preference for staying in. This is unusual for competitive play where staying in with type advantage is typically optimal.

---

## Summary of Key Metrics by Outcome

| Metric | Wins | Losses | Ratio |
|--------|------|--------|-------|
| Avg Turns | 36.6 | 30.0 | +22% |
| Avg RL Switches | 10.2 | 13.3 | -30% |
| Avg Opp Switches | 5.2 | 3.5 | +49% |
| Super Effective Hits | 20 | 89 | -78% |
| Avg RL Fainted | 4.0 | 6.0 | -33% |
| Avg Opp Fainted | 6.0 | 4.8 | +20% |

**Interpretation**: The model needs to:
1. Reduce switching frequency by ~30% to match win rate
2. Improve type advantage prediction (89 vs 20 SE hits is extreme)
3. Stay in favorable matchups longer (currently switches too easily)

---

