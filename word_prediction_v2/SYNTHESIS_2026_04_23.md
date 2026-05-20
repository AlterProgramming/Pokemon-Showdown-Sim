# Research Synthesis — 2026-04-23

Synthesis across: FINDINGS.md, RESEARCH_SNAPSHOT_2026_04_09.md, BENCHMARK_CHRONOLOGY.md,
THREAT_AWARE_NOTES.md, ROADMAP_90_PERCENT.md, PARAMETERS.md, DECISION_FLOW.md,
RANDOM_BASELINE_CEILING.md (Pokemon-Showdown-Agents-Go-Brrrr/docs),
FINDINGS_HISTORY_V1.md (artifacts/entity_history_v1_20260422_2300),
RUNBOOK_7802_OF_10000.md.

## Conflicts Resolved

- [CONFIRMED] Baseline timeline is sequential, not contradictory:
  78.02% (n=10000, Phase 11, type-aware + switch suppression) →
  85.50% (n=200, threat-aware decoder) →
  87.00% (n=1000, Phase 12 plateau) →
  88.40% (n=2000, sticky_move_run_003, current head).
- [CONFIRMED] 88.40% is the current large-sample baseline. 78.02% remains the
  only 10 000-game anchor.
- [CONFIRMED] The plateau is local, not the random-baseline floor. The 80–90%
  band is "reliable exploitation with narrow tactical errors"; 90%+ is feasible.
- [OVERTURNED] Assumption that broad opponent-role modeling is the next lever.
  Prior attempt regressed to 84.60%.
- [OVERTURNED] ROADMAP_90_PERCENT's A→B→C→D→E order. Evidence requires B and
  narrow-E to go first, A and C to wait for a decoder-isolation refactor, D to
  be skipped.
- [CONFIRMED] History encoder (Cell C 40.50% on n=1000) is a separate track,
  cannot close the word-policy gap.

## Root Causes

- `repeat_loop` (907), `generic_attack_loop` (262) → SYMPTOM of: decoder has
  no memory of own-action / opponent-response dynamics; sticky-move patch is
  a local mask.
- `priority_choice` (320) → SYMPTOM of: closeout heuristic conflates
  "opponent low HP" with "I can safely finish"; no retaliation discount.
- Retune 87.00→86.30%, opponent-role →84.60%, estimated-lethal →86.70%,
  risky-closeout 89.60→86.90% at n=500→1000 → SYMPTOMS of: decoder is a
  co-adapted sum of heuristics with no per-layer isolation. Any global change
  breaks implicit balances; n=500 spikes are variance.
- Voluntary-switch suppression, type-awareness, threat-awareness = ROOT-CAUSE
  fixes (action-space constraint, symbolic prior, bidirectional scoring).

**Core root cause of plateau:** scoring layers are entangled. New correct-sounding
rules compete in the same additive score space and degrade the whole.

## Blocking Order

1. [Unblocked] B — endgame / anti-throw gate.
2. [Unblocked] E-narrow — strict risky-closeout with safety gate.
3. [Blocked by B, E] Decoder-isolation refactor (per-layer attribution).
4. [Blocked by isolation] A — gated opponent-set inference.
5. [Blocked by A] C — narrow voluntary switching.
6. [Terminal / Skip] D — hazard economy.

## Action List

1. [Fix now] Implement endgame gate in `policy_adapter.py` (healthy-count,
   HP-diff, demote setup/risk, boost finish, no-greed clause). Smoke 50 →
   qualify 200 → validate 2000 via
   `./word_prediction_model/scripts/record_benchmark_run.sh endgame_gate_001`.
   — Directly targets ahead-and-looping losses dominating current bucket stats.

2. [Fix now] Strict safety gate on risky-closeout in `policy_adapter.py`:
   require KO-range confirmation, no revealed opponent priority, no
   outspeed-dependence. Same benchmark ladder as `risky_closeout_strict_001`.
   — The n=500 spike to 89.60% is real signal; collapse was missing safety.

3. [Opportunistic] Extend sticky-move family beyond the five patched
   (earthquake, shadowball, playrough, icebeam, drainpunch). Pick top 3
   remaining `repeat_loop` offenders from latest loss-bucket artifact.

4. [Plan sprint] Decoder-isolation refactor design sketch in
   `DECODER_ISOLATION_PLAN.md`. Named scoring layers returning per-layer
   contributions; per-decision logging; confidence gates. Write sketch before
   editing code.

5. [Plan sprint] Gated opponent-set inference (Phase A redux). Only fire when
   confidence > τ on {offensive, setup, utility}; else no-op. Hold until (4).

6. [Plan sprint] Narrow voluntary switching (Phase C). Requires (5) for role
   signal plus endgame signal from (1) to avoid reintroducing the weakness
   that original suppression fixed.

7. [Skip] Phase D hazard economy — loss buckets don't point here.
   [Skip] Broad opponent-role modeling (previous form) — regressed to 84.60%.
   [Skip] Global coefficient retune — regressed to 86.30%; co-adaptation makes
   it strictly harmful.
   [Skip] Replacing word policy with history encoder track — far below baseline.

8. [Pre-work question] Before (4): grep `policy_adapter.py` scoring composition.
   If scoring is already additive but unlogged, (4) collapses to a logging
   patch, not a refactor — large cost reduction.
