"""Analyze JSONL auxiliary logs to measure history encoder influence on action selection.

Usage:
    python scripts/analyze_sequence_influence.py \\
        --log path/to/aux_log.jsonl \\
        [--output-csv results.csv] \\
        [--min-candidates 2]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

EVENT_CATEGORIES = {
    "move", "switch", "damage", "heal",
    "status_start", "status_end", "boost", "unboost",
    "faint", "weather", "field", "side_condition", "forme_change", "turn_end",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _rank_shift(record: dict[str, Any]) -> bool:
    """Return True if reranking changed the top action."""
    analyses = record.get("analyses") or []
    if len(analyses) < 2:
        return False
    top_base = min(analyses, key=lambda a: a.get("base_rank", 999))
    top_comb = min(analyses, key=lambda a: a.get("combined_rank", 999))
    return top_base.get("action_token") != top_comb.get("action_token")


def _score_zero_rate(analyses: list[dict]) -> float:
    if not analyses:
        return 0.0
    zero = sum(1 for a in analyses if not a.get("sequence_score"))
    return zero / len(analyses)


def _event_categories(analyses: list[dict]) -> Counter:
    cats: Counter = Counter()
    for a in analyses:
        for tok in (a.get("sequence_tokens") or []):
            cat = str(tok).split(":")[0]
            if cat in EVENT_CATEGORIES:
                cats[cat] += 1
    return cats


def _attn_peak_turn(analyses: list[dict]) -> int | None:
    weights_list = [a.get("history_attention") for a in analyses if a.get("history_attention")]
    if not weights_list:
        return None
    K = len(weights_list[0])
    avg = [sum(w[k] for w in weights_list) / len(weights_list) for k in range(K)]
    return avg.index(max(avg))


def _influence_magnitude(record: dict[str, Any]) -> float | None:
    selected = record.get("selected_action_token")
    for a in (record.get("analyses") or []):
        if a.get("action_token") == selected:
            combined = a.get("combined_score") or 0.0
            base = a.get("base_logit") or 0.0
            return abs(combined - base)
    return None


def run_analysis(
    log_path: Path,
    min_candidates: int = 2,
    output_csv: Path | None = None,
) -> None:
    records = load_jsonl(log_path)
    print(f"Loaded {len(records)} records from {log_path}")

    eligible = [r for r in records if len(r.get("analyses") or []) >= min_candidates]
    print(f"Eligible (>= {min_candidates} candidates): {len(eligible)}")

    if not eligible:
        print("No eligible records — nothing to analyze.")
        return

    total = len(eligible)
    shifted = sum(1 for r in eligible if _rank_shift(r))
    zero_rates = [_score_zero_rate(r.get("analyses") or []) for r in eligible]
    magnitudes = [m for r in eligible if (m := _influence_magnitude(r)) is not None]
    global_cats: Counter = Counter()
    for r in eligible:
        global_cats.update(_event_categories(r.get("analyses") or []))
    peak_turns = [p for r in eligible if (p := _attn_peak_turn(r.get("analyses") or [])) is not None]

    print("\n=== Semi-Observed Sequence Influence Analysis ===")
    print(f"Rank-shift rate:           {shifted / total:.3f}  ({shifted}/{total})")
    print(f"Score-zero rate (avg):     {sum(zero_rates)/len(zero_rates):.3f}")
    if magnitudes:
        print(f"Influence magnitude (avg): {sum(magnitudes)/len(magnitudes):.4f}")
    print("\nEvent category breakdown (sequence tokens across all candidates):")
    for cat, count in global_cats.most_common():
        print(f"  {cat:22s}: {count}")
    if peak_turns:
        peak_dist = Counter(peak_turns)
        print("\nAttention peak-turn distribution (0=oldest, K-1=most recent):")
        for idx in sorted(peak_dist):
            print(f"  turn[{idx}]: {peak_dist[idx]}")

    if output_csv:
        rows = []
        for r in eligible:
            for a in (r.get("analyses") or []):
                attn = a.get("history_attention") or []
                peak_idx = attn.index(max(attn)) if attn else None
                rows.append({
                    "battle_id": r.get("battle_id", ""),
                    "turn_number": r.get("turn_number", ""),
                    "action_token": a.get("action_token", ""),
                    "base_rank": a.get("base_rank", ""),
                    "combined_rank": a.get("combined_rank", ""),
                    "base_logit": a.get("base_logit", ""),
                    "combined_score": a.get("combined_score", ""),
                    "sequence_score": a.get("sequence_score", ""),
                    "history_attention_peak_turn": peak_idx,
                    "rank_shifted": _rank_shift(r),
                })
        if rows:
            with open(output_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nCSV written to {output_csv}  ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze auxiliary JSONL log for sequence influence.")
    parser.add_argument("--log", required=True, help="Path to JSONL auxiliary log file.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--min-candidates", type=int, default=2,
                        help="Min candidates per record to include (default: 2).")
    args = parser.parse_args()
    run_analysis(
        log_path=Path(args.log),
        min_candidates=args.min_candidates,
        output_csv=Path(args.output_csv) if args.output_csv else None,
    )


if __name__ == "__main__":
    main()
