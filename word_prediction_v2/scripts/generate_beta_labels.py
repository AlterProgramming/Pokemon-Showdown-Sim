#!/usr/bin/env python3.12
"""Beta-1: offline label generation for the symbolic-to-neural distillation
methodology proposed in the 2026-05-01 paper.

For each training example produced by the entity model's loader, this script
emits two scalar labels derived from the word_policy_v1 symbolic decoder:

    threat_score      : float, replicating the local threat_score computation
                        in word_prediction_model/policy_adapter.py (lines 602-623).
                        Aggregates per-opponent-move type effectiveness, base
                        power, and priority into a single threat scalar.
    action_type_eff   : float, the type-effectiveness multiplier of the
                        actor's chosen-move type vs. the opponent active's
                        types, computed via pokemon_type_utils.type_effectiveness.
                        None for switch actions.

These are the offline targets that would supervise the two new auxiliary
heads described in section 3.3 of the paper. Generating them does not
require modifying the entity model or its training code — that is the
training-time step (out of scope for beta-1).

Output:
    <out-dir>/labels.jsonl   one JSON record per example
    <out-dir>/summary.json   counts, distributions, timing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

POKEMON_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, POKEMON_REPO)

from core.BattleStateTracker import BattleStateTracker  # noqa: E402
from core.TrainingSplit import ingest_battles_to_examples  # noqa: E402
from word_prediction_v2.pokemon_move_metadata import move_metadata  # noqa: E402
from word_prediction_v2.pokemon_type_utils import (  # noqa: E402
    move_type,
    species_types,
    type_effectiveness,
)


def find_active_uid(side: Dict[str, Any], mons: Dict[str, Any]) -> Optional[str]:
    uid = side.get("active_uid")
    if uid and isinstance(uid, str):
        m = mons.get(uid)
        if m and not m.get("fainted"):
            return uid
    return None


def compute_threat_score(
    my_types: List[str],
    my_hp: float,
    opp_observed_moves: List[str],
) -> float:
    """Replicates the threat_score computation in policy_adapter._score_attack
    (lines 602-623). Pure function; no side effects."""
    threat_score = 0.0
    for opp_move_name in opp_observed_moves:
        opp_meta = move_metadata(opp_move_name)
        opp_move_type = str(opp_meta.get("type") or move_type(opp_move_name) or "")
        opp_category = str(opp_meta.get("category") or "")
        opp_power = float(opp_meta.get("basePower") or 0.0)
        opp_priority = float(opp_meta.get("priority") or 0.0)
        if opp_category not in {"Physical", "Special"} or not opp_move_type:
            continue
        multiplier = type_effectiveness(opp_move_type, my_types)
        local = min(opp_power / 70.0, 2.0)
        if multiplier == 0.0:
            local -= 0.5
        elif multiplier >= 4.0:
            local += 3.0
        elif multiplier > 1.0:
            local += 1.5
        elif multiplier < 1.0:
            local -= 0.5
        if opp_priority > 0 and my_hp is not None and my_hp <= 0.35:
            local += 0.8
        threat_score = max(threat_score, local)
    return threat_score


def extract_labels(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    state = example.get("state") or {}
    player = example.get("player") or ""
    opp = "p2" if player == "p1" else "p1"
    mons = state.get("mons") or {}

    my_active_uid = find_active_uid(state.get(player) or {}, mons)
    opp_active_uid = find_active_uid(state.get(opp) or {}, mons)
    if not my_active_uid or not opp_active_uid:
        return None

    my_mon = mons.get(my_active_uid) or {}
    opp_mon = mons.get(opp_active_uid) or {}

    my_species = str(my_mon.get("species") or "")
    opp_species = str(opp_mon.get("species") or "")
    my_types = species_types(my_species) if my_species else []
    opp_types = species_types(opp_species) if opp_species else []
    my_hp = my_mon.get("hp_frac")
    opp_observed = [m for m in (opp_mon.get("observed_moves") or []) if m]

    threat = compute_threat_score(
        my_types,
        float(my_hp) if my_hp is not None else 1.0,
        opp_observed,
    )

    action = example.get("action") or ()
    action_type_eff: Optional[float] = None
    if isinstance(action, (list, tuple)) and len(action) >= 2 and action[0] == "move":
        m_id = str(action[1])
        m_meta = move_metadata(m_id)
        m_type = str(m_meta.get("type") or move_type(m_id) or "")
        if m_type and opp_types:
            action_type_eff = float(type_effectiveness(m_type, opp_types))

    return {
        "battle_id": example.get("battle_id"),
        "turn_number": example.get("turn_number"),
        "player": player,
        "my_species": my_species,
        "opp_species": opp_species,
        "my_hp_frac": my_hp,
        "n_opp_observed_moves": len(opp_observed),
        "threat_score": threat,
        "action_kind": (action[0] if isinstance(action, (list, tuple)) and action else None),
        "action_type_eff": action_type_eff,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-battles", type=int, default=1000)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(REPO, "artifacts/beta_labels_v1"),
    )
    ap.add_argument(
        "--kaggle-dir",
        default="/Users/AI-CCORE/.cache/kagglehub/datasets/"
        "thephilliplin/pokemon-showdown-battles-gen9-randbats/versions/1",
    )
    ap.add_argument("--include-switches", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "labels.jsonl")
    out_summary = os.path.join(args.out_dir, "summary.json")

    files = sorted(os.listdir(args.kaggle_dir))
    paths = [os.path.join(args.kaggle_dir, f) for f in files if f.endswith(".json")]
    print(f"replay files available: {len(paths)}; using up to {args.max_battles}", flush=True)

    t0 = time.time()
    tracker = BattleStateTracker()
    examples = ingest_battles_to_examples(
        tracker,
        paths,
        max_battles=args.max_battles,
        verbose_every=200,
        include_switches=args.include_switches,
    )
    t_load = time.time() - t0
    print(f"loaded {len(examples)} examples in {t_load:.1f}s", flush=True)

    t0 = time.time()
    n_written = 0
    n_skipped = 0
    threat_sum = 0.0
    type_eff_counts: Dict[str, int] = {}
    action_kind_counts: Dict[str, int] = {}
    with open(out_jsonl, "w") as f:
        for ex in examples:
            label = extract_labels(ex)
            if label is None:
                n_skipped += 1
                continue
            f.write(json.dumps(label) + "\n")
            n_written += 1
            threat_sum += float(label["threat_score"] or 0.0)
            ak = label.get("action_kind") or "none"
            action_kind_counts[ak] = action_kind_counts.get(ak, 0) + 1
            ate = label.get("action_type_eff")
            if ate is not None:
                bucket = f"{ate:.2f}"
                type_eff_counts[bucket] = type_eff_counts.get(bucket, 0) + 1
    t_gen = time.time() - t0

    summary = {
        "max_battles_requested": args.max_battles,
        "include_switches": args.include_switches,
        "examples_loaded": len(examples),
        "labels_written": n_written,
        "examples_skipped": n_skipped,
        "load_time_s": round(t_load, 1),
        "gen_time_s": round(t_gen, 1),
        "avg_threat_score": round(threat_sum / max(n_written, 1), 4),
        "action_kind_distribution": dict(
            sorted(action_kind_counts.items(), key=lambda kv: -kv[1])
        ),
        "action_type_eff_distribution": dict(
            sorted(type_eff_counts.items(), key=lambda kv: -kv[1])
        ),
        "output_jsonl": out_jsonl,
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
