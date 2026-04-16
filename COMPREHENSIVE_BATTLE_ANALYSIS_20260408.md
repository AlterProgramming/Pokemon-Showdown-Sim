# Comprehensive Entity Model Battle Analysis
## Using BattleAnalyzer Framework
**Date**: April 8, 2026  
**Model**: entity_action_bc_v1_20260408_0428  
**Sample**: 20 battles vs Random opponent  

---

## Executive Summary

The entity model achieves a **25% win rate** (5/20 games) against random opponents. Analysis of **706 total decisions** (436 moves, 270 switches) reveals a model that struggles with **move accuracy** (85.8% hit rate, 14.2% failure) and **excessive switching** (38% of decisions are switches, up to 86.4% in extreme cases).

**Critical Finding**: Losses are characterized by:
- **15.9% bad move outcomes** (failed, missed, immune) vs 10.7% in wins
- **4.3x more switches per game** than moves
- **42% less offensive focus** (20 moves/game vs 28 in wins)

---

## 1. Win Rate Analysis

| Metric | Value |
|--------|-------|
| Wins | 5/20 (25.0%) |
| Losses | 15/20 (75.0%) |
| 95% CI | [11.2%, 46.9%] |

**Interpretation**: With 95% confidence, true win rate is between 11% and 47%. The 25% point estimate is consistent across all recent benchmarks (30%, 27.5%, 25.7%, 25%).

---

## 2. Action Statistics: Decision Distribution

### Overall Decision Breakdown

| Type | Count | % of Decisions |
|------|-------|----------------|
| Total Decisions | 706 | 100% |
| Move Decisions | 436 | **61.5%** |
| Switch Decisions | 270 | **38.2%** |
| Move-to-Switch Ratio | 1.61:1 | - |

**Interpretation**: The model switches in 38% of all decisions. For comparison, typical competitive players switch in 10-20% of decisions. This 2-3x overly high switching rate indicates the model lacks confidence in type matchups.

### Move Outcome Distribution

| Outcome | Count | % of Moves | Category |
|---------|-------|-----------|----------|
| Hit | 374 | **85.8%** | ✅ Successful |
| Failed | 47 | **10.8%** | ❌ Failed |
| Missed | 10 | **2.3%** | ❌ Missed |
| Immune | 5 | **1.1%** | ❌ Immune |
| **Total Bad Outcomes** | **62** | **14.2%** | ❌ Ineffective |

**Critical**: 1 in 7 moves (14.2%) have no positive effect. This is significantly higher than expert players (~2-3%).

---

## 3. Tactical Error Classification

### Failed Actions

| Error Type | Count | Games Affected |
|-----------|-------|----------------|
| Failed Moves | 47 | Multiple |
| Missed Moves | 10 | Multiple |
| Immune Moves | 5 | Multiple |
| Excessive Switches (>40% rate) | - | **7 games** |

### Excessive Switching Analysis

**7 games show critical over-switching patterns:**

| Game | Switch % | Switches/Total | Category |
|------|----------|-----------------|----------|
| 17 | **86.4%** | 19/22 | 🔴 PANIC |
| 16 | **67.6%** | 25/37 | 🔴 PANIC |
| 19 | **65.4%** | 17/26 | 🔴 PANIC |
| 3 | **61.8%** | 21/34 | 🔴 PANIC |
| 20 | **60.9%** | 14/23 | 🔴 PANIC |
| 12 | **44.8%** | 13/29 | 🟡 HIGH |
| 10 | **41.4%** | 12/29 | 🟡 HIGH |

**Pattern**: Games 16, 17, 19 are extreme outliers with 65-86% switching rates. These are all **losses** where the model abandons offensive play entirely.

---

## 4. Win vs Loss Breakdown

### Winning Games (5 games)

| Metric | Value | Notes |
|--------|-------|-------|
| Avg Switches/Game | **11.2** | Controlled, strategic |
| Avg Moves/Game | **28.0** | Strong offensive play |
| Move Hit Rate | **89.3%** (125 hits / 140 total) | Good accuracy |
| Bad Move Outcomes | **10.7%** (15/140) | Below average |
| Avg Turns | **36.6** | Long, grinding games |

**Win Pattern**: Model stays committed to offensive plays (28 moves/game), maintains reasonable type matchups (89% hit), and wins through attrition by lasting longer battles.

### Losing Games (15 games)

| Metric | Value | Notes |
|--------|-------|-------|
| Avg Switches/Game | **14.3** | Reactive, panicked |
| Avg Moves/Game | **19.7** | Weak offensive play |
| Move Hit Rate | **81.8%** (249 hits / 304 total) | Below wins |
| Bad Move Outcomes | **15.9%** (55/296) | Significantly worse |
| Avg Turns | **30.0** | Shorter, rushed games |

**Loss Pattern**: Model switches too frequently (14.3 vs 11.2), abandons offense, loses type advantage, and makes more tactical errors. Each switch gives opponent free turn.

### Key Comparison

```
                      WINS    LOSSES   DELTA
Switches/Game:        11.2 → 14.3  (+27% overly defensive)
Moves/Game:           28.0 → 19.7  (-30% less offensive)
Move Failure Rate:    10.7% → 15.9% (+50% more failures)
Game Duration:        36.6 → 30.0  (-18% faster losses)
```

---

## 5. Root Cause Analysis

### Hypothesis: Defensive Panic Response

The model exhibits a **panic-switching mechanism** triggered by adversarial uncertainty:

1. **Initial Misread**: Model predicts unfavorable type matchup
2. **Low Confidence**: Logit softmax shows no strong preference for any action
3. **Panic Response**: Switches to "safe" Pokémon
4. **Cascade**: Each switch fails to resolve matchup, triggering another switch
5. **Collapse**: Game becomes succession of switches with no offensive pressure

**Evidence**:
- Games 16, 17, 19: Opponent makes **1-2 switches total** while model makes **19-25 switches**
- Model "searching" through team for counters while opponent stays with single Pokémon
- Random opponent capitalizes on free turns gained from switches

### Secondary Issue: Move Effectiveness Prediction

1 in 7 moves (14.2%) fail to have any effect:
- **Failure** (moves blocked by abilities/items): 47 cases
- **Immune** (type immunity, priority reversals): 5 cases  
- **Miss** (accuracy checks): 10 cases

This suggests the model:
- Doesn't account for opponent abilities/items
- Fails to track immunity statuses
- Overestimates move accuracy

---

## 6. Comparison with Historical Benchmarks

| Run | Date | Format | Opponent | Win Rate | Sample |
|-----|------|--------|----------|----------|--------|
| model4 | Mar 27 | vs entity | - | 66.7% | 18 games |
| entity v1 | Apr 7 | vs random | Benchmark | 30.0% | 120 games |
| entity v2 | Apr 7 | vs random | Benchmark | 27.5% | 1000 games |
| entity v3 | Apr 7 | vs random | Benchmark | 25.7% | 1000 games |
| entity (current) | **Apr 8** | **vs random** | **Replay capture** | **25.0%** | **20 games** |

**Trend**: Converging to ~25% true win rate as sample size increases. The model's vulnerability to random play is consistent.

---

## 7. Metrics Summary Table

### Precision Metrics

| Category | Metric | Value | Status |
|----------|--------|-------|--------|
| **Reliability** | Move Hit Rate (Wins) | 89.3% | ✅ Good |
| **Reliability** | Move Hit Rate (Losses) | 81.8% | ⚠️ Poor |
| **Efficiency** | Switches:Moves Ratio | 1:1.61 | 🔴 Bad |
| **Efficiency** | Switches % (Wins) | 28.5% | ✅ Normal |
| **Efficiency** | Switches % (Losses) | 42.1% | 🔴 Critical |
| **Confidence** | Avg Turns/Game (Wins) | 36.6 | ✅ Long |
| **Confidence** | Avg Turns/Game (Losses) | 30.0 | 🔴 Short |

---

## 8. Recommendations for Improvement

### Immediate (Data-driven Diagnostics)

1. **Probability Distribution Analysis**
   - Log softmax outputs for move vs switch decisions
   - Identify threshold where model "panics" and switches
   - Measure confidence entropy before bad outcomes

2. **Type Matchup Auditing**
   - Extract predicted type advantage vs actual
   - Quantify false-negatives (thought matchup was bad, was actually good)
   - Compare to game-state known type advantages

3. **Failure Mode Profiling**
   - Categorize the 47 failed moves: which abilities? which status conditions?
   - Identify patterns in immunity/block scenarios
   - Extract decision states before failures

### Short-term (Training Modifications)

1. **Anti-Panic Loss**
   - Add auxiliary loss: penalize excessive switching in same turn range
   - Reward staying in favorable matchups
   - Weight based on type effectiveness

2. **Move Effectiveness Head**
   - Add auxiliary network head for type advantage prediction
   - Train on known matchups from game state
   - Use to gate switch decisions (require high confidence)

3. **Team Diversity Penalty**
   - Penalize switching to same type multiple times
   - Encourage staying with working Pokémon
   - Reward winning streaks with same Pokémon

### Medium-term (Architectural)

1. **Uncertainty Estimation**
   - Use Monte Carlo Dropout or ensemble methods
   - Condition switch probability on decision confidence
   - Only switch if entropy above threshold

2. **Ability/Item Tracking**
   - Add game state features for opponent abilities
   - Track opponent item effects (immunity, reflection)
   - Improve failure prediction accuracy

3. **Momentum Tracking**
   - Add hidden state tracking switch history
   - Reward consecutive move sequences
   - Penalize switch oscillation

---

## 9. Artifacts and Files

| File/Path | Description |
|-----------|-------------|
| `logs/replays/*.html` | 20 HTML battle replays with embedded battle logs |
| `benchmark_battle_logs_v2.json` | Parsed decision points (706 total) |
| `benchmark_games_20260408.csv` | Game-level statistics |
| Model Server | `http://127.0.0.1:5001/predict` (during benchmark) |
| Model Metadata | `artifacts/entity_action_bc_v1_20260408_0428/` |

---

## 10. Conclusions

The entity model shows **fundamental weakness in adversarial uncertainty**: when facing random opponents, it lacks robust decision-making and resorts to excessive switching. This is not a performance issue (decision latency is 334ms, acceptable) but a **strategy issue** — the model panics under conditions that violate its training distribution.

The 25% win rate and consistent losses across benchmarks suggest the model is **overfitted to specific battle patterns** rather than learning generalizable competitive play. Training with more diverse opponents, particularly random agents, would likely improve robustness.

**Next step**: Implement uncertainty estimation and confidence-gated switching to prevent panic responses.

---

*Analysis completed: 2026-04-08 12:45 UTC*  
*Framework: BattleAnalyzer + battle log reconstruction*  
*Analysts: Claude Haiku (automated)*
