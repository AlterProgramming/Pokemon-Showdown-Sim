# Advisor Synthesis — FINDINGS_HISTORY_V1.md — 2026-04-23

Advisor-produced /synthesis of
`/Users/AI-CCORE/alter-programming/artifacts/entity_history_v1_20260422_2300/FINDINGS_HISTORY_V1.md`.
Directed scope: history-encoder research track only. Independent of the
word-policy synthesis (`SYNTHESIS_2026_04_23.md`).

## Conflicts Resolved

- [OVERTURNED] Sub-claim #3 ("history encoder pays off via rerank") as
  *confirmed*. B→C lift (+5.9pp, p=0.006) is confirmed; content-vs-pipeline
  attribution is not. C-vs-D p=0.089, B-vs-D p=0.305 — D is statistically
  indistinguishable from both. The pre-registered decision rule (§90) resolves
  to inconclusive, not confirmed. To isolate content, Cell D needs either
  ~4× sample (~4000 games/cell) or paired-session design across cells.
- [OVERTURNED] Framing of "Cell C recovers to no-history baseline" as a win.
  A→C at p=0.314 is null. The honest headline is: the history encoder is
  *net-neutral* vs no-history at n=1000 — not evidence for "compensating for
  information loss."
- [OVERTURNED] Cell B as a clean "policy-only, training-pressure-only" serve.
  §104 flags `has_sequence_head: true` on the v1 server may nudge logits at
  serve time. Until resolved, the A→B regression narrative is unsafe.
- [CONFIRMED] The attention distribution's learned shape (recency bias,
  entropy 77% of max) — pending the turn-index ≥ 8 conditioned rerun that the
  author themselves proposed (§62) to remove the PAD-slot-0 artifact.
- [CONFIRMED] Entity family loses to random (34–40%) on this harness;
  word_policy_v1 is 78.02% on the same harness. Claims must be scoped to
  relative lift within the entity family, not absolute strength.
- [METHODS GAP] Rerank composition in Cell C is under-specified: the
  checkpoint epoch used at serving, and whether the regressed value head
  contributes to rerank, are not stated in the file. The "aux-head fusion"
  story cannot be evaluated without this.

## Root Causes

- +5.9pp B→C lift → SYMPTOM. Root cause **unresolved**: rerank-pipeline vs
  history-content. D ablation is underpowered.
- A→B −3.7pp regression → SYMPTOM. Candidate root cause: sequence-head
  contamination of Cell B (§104). Cannot attribute to "training pressure on
  shared trunk" until contamination is ruled out.
- Attention recency peak → SYMPTOM of a network being trained. Root cause of
  the peak shape needs the turn-index ≥ 8 conditioned re-analysis; slot-7 may
  partly be the PAD-symmetric mirror of the slot-0 artifact the author
  already flagged.
- Value-head 3× val-loss regression → ROOT CAUSE (capacity × data-volume;
  known pre-history issue per §38). Independent of the history encoder but
  contaminates any rerank that includes it.
- v1 switches 32% more → SYMPTOM of the same contamination concern as the
  A-vs-B regression.
- Entity family 34–40% vs random → SYMPTOM of a prior architectural gap in
  the entity-action family; ROOT CAUSE of this paper's weak load-bearing
  (small effects on a weak floor are hard to detect).

**Core root cause of the paper's current state:** n=1000 per cell is
under-powered for the effect sizes claimed, and the A↔B↔C↔D cells are not
cleanly isolated (sequence-head wire-through, value-head regression,
under-specified rerank composition). The ablation design is correct; the
execution is under-resolved.

## Blocking Order

1. [Unblocked] Resolve §104: is the sequence-head actually dormant in Cell B?
2. [Unblocked] Document Cell C rerank composition + checkpoint epoch.
3. [Unblocked] Re-run attention analysis conditioned on turn-index ≥ 8.
4. [Blocked by 1] Any A-vs-B narrative.
5. [Blocked by 1, 2] Larger-n Cell D (n≈4000/cell or paired-session).
6. [Blocked by 5] Any "history content is the mechanism" claim.
7. [Terminal / Opportunistic] Seed sweep (3 × 30 ep, ~$10).
8. [Terminal / Skip] K ∈ {4, 8, 16} ablation — premature.

## Action List

1. [Fix now] Inspect `serve_entity_model_benchmark.py` + Cell B run config
   for how `has_sequence_head: true` affects logits in policy-only mode.
   If the sequence head contributes, re-run Cell B with it disabled.
   Decision rule: if A→B closes by >2pp when truly dormant, §104 was the
   cause and the paper's A-vs-B narrative must change.

2. [Fix now] Edit the file to state explicitly: (a) checkpoint epoch used at
   serving for Cell C, (b) the rerank score formula and head weights,
   (c) whether the regressed value head contributes. If value is in and
   epoch is 30, add a supplementary Cell C' with value frozen at ep 3–11 or
   removed from rerank.

3. [Fix now] Re-run `attention_analysis` on validation conditioned on
   turn-index ≥ 8. Strengthens or softens §54's "learned recency prior"
   claim. No retrain required.

4. [Fix now — editorial] Rewrite §124–133 Interpretation and §5 abstract.
   Replace "confirmed" on sub-claim #3 with:
   "B→C lift confirmed; content-vs-pipeline attribution underpowered at
   n=1000 (C-vs-D p=0.089)." The file's own §90 already contains this
   honesty; the headline got ahead of it. Zero-cost integrity fix.

5. [Plan sprint] After 1–4 land: run Cell D at n≈4000 against Cell C at
   n≈4000, ideally paired-session (identical opponent seeds). Pre-register
   the significance threshold. This is the experiment that resolves the
   paper's central question.

6. [Opportunistic] Seed sweep (3 seeds × 30 ep, ~$10) to confirm
   attention-shape reproducibility. Can run in parallel with (5).

7. [Skip — for now] K ablation, fusing history into the policy pre-logits.
   Both are follow-up, not resolutions of the current ablation.

8. [Pre-work question before (5)] Does `statistical-runner.js` support a
   seed-replay / paired-session mode? If yes, paired n≈1500/cell may
   replace independent n≈4000 — large compute saving. Check before
   committing.

## Bottom line

The single highest-leverage action is item 4 — aligning the headline claims
with the ablation the author already ran. The paper's own §90 contains the
honest interpretation; the abstract and §131 Interpretation overclaim
relative to it. That fix is free and restores integrity before any further
experiment is worth running.
