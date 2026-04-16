from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from BattleStateTracker import BattleStateTracker
from ModelRegistry import resolve_artifact_path as resolve_registry_artifact_path
from RewardSignals import (
    RewardConfig,
    compute_reward_targets,
)
from StateVectorization import encode_state_v0


MAJOR_STATUS_MOVES: Dict[str, str] = {
    "glare": "par",
    "hypnosis": "slp",
    "poisonpowder": "psn",
    "sleeppowder": "slp",
    "spore": "slp",
    "stunspore": "par",
    "thunderwave": "par",
    "toxic": "tox",
    "toxicthread": "psn",
    "willowisp": "brn",
    "yawn": "slp",
}


def discover_json_paths(inputs: Sequence[str]) -> List[Path]:
    json_paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            json_paths.extend(sorted(path.rglob("*.json")))
        elif path.suffix.lower() == ".json" and path.exists():
            json_paths.append(path)
    return json_paths


def resolve_artifact_path(metadata_path: Path, raw_path: str) -> Path:
    repo_path = metadata_path.parent.parent
    return resolve_registry_artifact_path(repo_path, metadata_path, raw_path)


def normalize_policy_token(raw_token: str, label_format: str) -> str:
    if label_format == "move_ids" and raw_token != "<UNK>":
        return f"move:{raw_token}"
    return raw_token


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 0:
        return np.zeros_like(logits)
    return exp / denom


def extract_model_outputs(model: Any, predictions: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(predictions, dict):
        policy_logits = np.asarray(predictions["policy"], dtype=np.float32)
        value = np.asarray(predictions["value"], dtype=np.float32)
        return policy_logits, value.reshape(-1)

    if isinstance(predictions, (list, tuple)):
        outputs = {
            name: np.asarray(pred, dtype=np.float32)
            for name, pred in zip(getattr(model, "output_names", []), predictions)
        }
        if "policy" in outputs and "value" in outputs:
            return outputs["policy"], outputs["value"].reshape(-1)

    raise ValueError("policy+value model must return named 'policy' and 'value' outputs")


def predict_policy_and_value(model: Any, state_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(state_vectors, verbose=0)
    return extract_model_outputs(model, predictions)


def build_policy_topk(
    logits: np.ndarray,
    idx_to_token: Dict[int, str],
    *,
    label_format: str,
    top_k: int,
) -> List[Dict[str, float | str]]:
    probs = softmax(logits)
    k = max(1, min(top_k, logits.shape[0]))
    top_indices = np.argsort(logits)[-k:][::-1]
    rows: List[Dict[str, float | str]] = []
    for idx in top_indices:
        raw_token = idx_to_token.get(int(idx), "<UNK>")
        rows.append(
            {
                "token": normalize_policy_token(raw_token, label_format),
                "logit": float(logits[idx]),
                "prob": float(probs[idx]),
            }
        )
    return rows


def action_family(action_token: str) -> str:
    if action_token.startswith("move:"):
        return "move"
    if action_token.startswith("switch:"):
        return "switch"
    return "other"


def is_forced_switch_state(state: Dict[str, Any], player: str) -> bool:
    side = state.get(player, {}) or {}
    active_uid = side.get("active_uid")
    if not active_uid:
        return True
    mon = (state.get("mons", {}) or {}).get(active_uid, {}) or {}
    return bool(mon.get("fainted"))


def get_opponent_active_uid(state: Dict[str, Any], player: str) -> Optional[str]:
    other = "p2" if player == "p1" else "p1"
    side = state.get(other, {}) or {}
    active_uid = side.get("active_uid")
    return str(active_uid) if active_uid else None


def get_mon_status(state: Dict[str, Any], uid: Optional[str]) -> Optional[str]:
    if not uid:
        return None
    mon = (state.get("mons", {}) or {}).get(uid, {}) or {}
    status = mon.get("status")
    return str(status) if status else None


def get_mon_species(state: Dict[str, Any], uid: Optional[str]) -> Optional[str]:
    if not uid:
        return None
    mon = (state.get("mons", {}) or {}).get(uid, {}) or {}
    species = mon.get("species")
    return str(species) if species else None


def build_turn_lookup(battle: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for turn in battle.get("turns", []) or []:
        turn_number = turn.get("turn_number")
        if turn_number is None:
            continue
        lookup[int(turn_number)] = turn
    return lookup


def find_first_decision_event_index(turn_events: Sequence[Dict[str, Any]], player: str) -> Optional[int]:
    for idx, ev in enumerate(turn_events):
        if ev.get("player") != player:
            continue
        if ev.get("type") in {"move", "switch"}:
            return idx
    return None


def infer_move_effectiveness(
    turn_events: Sequence[Dict[str, Any]],
    *,
    move_index: Optional[int],
    actor_player: str,
    target_uid: Optional[str],
) -> Optional[str]:
    if move_index is None or not target_uid:
        return None
    if target_uid.startswith(actor_player):
        return None

    saw_damage_to_target = False
    for ev in turn_events[move_index + 1 :]:
        if ev.get("type") == "move":
            break

        if ev.get("type") == "damage" and ev.get("target_uid") == target_uid and ev.get("source") == "move":
            saw_damage_to_target = True

        if ev.get("type") != "effect":
            continue

        effect_type = ev.get("effect_type")
        if effect_type == "immune":
            return "immune"
        if effect_type == "supereffective":
            return "super"
        if effect_type == "resisted":
            return "resisted"

    if saw_damage_to_target:
        return "neutral"
    return None


def build_turn_diagnostics(
    ex: Dict[str, Any],
    turn_lookup: Dict[int, Dict[str, Any]],
    *,
    previous_action_token: Optional[str],
    previous_move_target_uid: Optional[str],
    previous_opponent_active_uid: Optional[str],
    previous_move_target_streak: int,
    previous_voluntary_switch: bool,
) -> Dict[str, Any]:
    state_before = ex["state"]
    player = str(ex["player"])
    action_token = str(ex["action_token"])
    family = action_family(action_token)
    forced_switch = is_forced_switch_state(state_before, player)
    opponent_active_uid_before = get_opponent_active_uid(state_before, player)
    opponent_active_species_before = get_mon_species(state_before, opponent_active_uid_before)

    turn_obj = turn_lookup.get(int(ex["turn_number"]), {}) or {}
    turn_events = turn_obj.get("events", []) or []
    decision_idx = find_first_decision_event_index(turn_events, player)
    decision_event = turn_events[decision_idx] if decision_idx is not None else {}

    move_id = decision_event.get("move_id") if family == "move" else None
    target_uid = decision_event.get("target_uid") if family == "move" else None
    target_status_before = get_mon_status(state_before, target_uid)
    expected_major_status = MAJOR_STATUS_MOVES.get(move_id or "")

    status_move_into_already_statused_target = bool(expected_major_status and target_status_before)
    known_status_violation = status_move_into_already_statused_target

    move_effectiveness = infer_move_effectiveness(
        turn_events,
        move_index=decision_idx if family == "move" else None,
        actor_player=player,
        target_uid=target_uid,
    )

    targets_opponent = bool(target_uid and not str(target_uid).startswith(player))
    if family == "move" and targets_opponent:
        if (
            previous_action_token == action_token
            and previous_move_target_uid == target_uid
            and previous_opponent_active_uid == opponent_active_uid_before
        ):
            same_move_target_streak = previous_move_target_streak + 1
        else:
            same_move_target_streak = 1
    else:
        same_move_target_streak = 0

    switch_chain = bool(
        family == "switch"
        and not forced_switch
        and previous_voluntary_switch
    )

    return {
        "public": {
            "action_family": family,
            "forced_switch_state": forced_switch,
            "opponent_active_uid_before": opponent_active_uid_before,
            "opponent_active_species_before": opponent_active_species_before,
            "target_uid": target_uid,
            "target_status_before": target_status_before,
            "status_move_into_already_statused_target": status_move_into_already_statused_target,
            "known_status_violation": known_status_violation,
            "switch_chain": switch_chain,
            "same_action_streak": 0,
            "same_move_target_streak": same_move_target_streak,
            "same_move_spam": bool(family == "move" and same_move_target_streak >= 3),
        },
        "retrospective": {
            "move_effectiveness": move_effectiveness,
            "used_super_effective_move": move_effectiveness == "super",
            "used_not_super_effective_move": move_effectiveness in {"neutral", "resisted", "immune"},
            "used_neutral_move": move_effectiveness == "neutral",
            "used_resisted_move": move_effectiveness == "resisted",
            "used_immune_move": move_effectiveness == "immune",
        },
    }


def generate_battle_trace_rows(
    battle: Dict[str, Any],
    *,
    tracker: BattleStateTracker,
    policy_value_model: Any,
    idx_to_token: Dict[int, str],
    include_switches: bool,
    label_format: str,
    top_k: int,
    reward_config: RewardConfig,
    move_reward_profile: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    per_player_rows: List[Dict[str, Any]] = []
    turn_lookup = build_turn_lookup(battle)

    for player in ("p1", "p2"):
        examples = list(tracker.iter_turn_examples(battle, player=player, include_switches=include_switches))
        if not examples:
            continue

        player_examples = [
            ex if "player" in ex else {**ex, "player": player}
            for ex in examples
        ]

        pre_states = np.asarray(
            [encode_state_v0(ex["state"], perspective_player=player) for ex in player_examples],
            dtype=np.float32,
        )
        post_states = np.asarray(
            [encode_state_v0(ex["next_state"], perspective_player=player) for ex in player_examples],
            dtype=np.float32,
        )
        policy_logits_pre, values_pre = predict_policy_and_value(policy_value_model, pre_states)
        _, values_post = predict_policy_and_value(policy_value_model, post_states)
        reward_components_seq, reward_totals, returns_to_go = compute_reward_targets(
            player_examples,
            reward_config,
            move_reward_profile,
        )

        cumulative_reward = 0.0
        previous_action_token: Optional[str] = None
        previous_move_target_uid: Optional[str] = None
        previous_opponent_active_uid: Optional[str] = None
        previous_move_target_streak = 0
        previous_voluntary_switch = False
        previous_same_action_streak = 0
        for idx, ex in enumerate(player_examples):
            reward_components = reward_components_seq[idx]
            reward_total = reward_totals[idx]
            cumulative_reward += reward_total
            diagnostics = build_turn_diagnostics(
                ex,
                turn_lookup,
                previous_action_token=previous_action_token,
                previous_move_target_uid=previous_move_target_uid,
                previous_opponent_active_uid=previous_opponent_active_uid,
                previous_move_target_streak=previous_move_target_streak,
                previous_voluntary_switch=previous_voluntary_switch,
            )

            public_diag = diagnostics["public"]
            if previous_action_token == ex["action_token"]:
                public_diag["same_action_streak"] = previous_same_action_streak + 1
            else:
                public_diag["same_action_streak"] = 1

            per_player_rows.append(
                {
                    "battle_id": ex["battle_id"],
                    "turn_number": int(ex["turn_number"]),
                    "player": player,
                    "chosen_action_token": ex["action_token"],
                    "policy_topk": build_policy_topk(
                        policy_logits_pre[idx],
                        idx_to_token,
                        label_format=label_format,
                        top_k=top_k,
                    ),
                    "value_pre": float(values_pre[idx]),
                    "value_post": float(values_post[idx]),
                    "value_delta": float(values_post[idx] - values_pre[idx]),
                    "reward_components": reward_components,
                    "reward_total": float(reward_total),
                    "cumulative_reward": float(cumulative_reward),
                    "return_to_go": float(returns_to_go[idx]),
                    "terminal_result": ex.get("terminal_result"),
                    "diagnostics": diagnostics,
                }
            )

            previous_action_token = ex["action_token"]
            previous_move_target_uid = public_diag["target_uid"]
            previous_opponent_active_uid = public_diag["opponent_active_uid_before"]
            previous_move_target_streak = int(public_diag["same_move_target_streak"])
            previous_voluntary_switch = bool(
                public_diag["action_family"] == "switch" and not public_diag["forced_switch_state"]
            )
            previous_same_action_streak = int(public_diag["same_action_streak"])

    rows.extend(sorted(per_player_rows, key=lambda row: (row["turn_number"], row["player"])))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay battle logs with a policy+value model and emit JSONL traces.")
    parser.add_argument("data_paths", nargs="+", help="Battle JSON files or directories containing battle logs.")
    parser.add_argument("--metadata-path", required=True, help="Training metadata JSON generated by train_policy.py.")
    parser.add_argument("--output-path", required=True, help="Destination JSONL file for per-turn traces.")
    parser.add_argument("--max-battles", type=int, default=200, help="Maximum number of battles to replay.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of policy actions to include per turn.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = discover_json_paths(args.data_paths)
    if not json_paths:
        raise SystemExit("No JSON files found in the provided path(s).")

    metadata_path = Path(args.metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    policy_value_model_path = metadata.get("policy_value_model_path")
    if not policy_value_model_path:
        raise SystemExit("Metadata does not contain policy_value_model_path. Train with --predict-value first.")

    reward_profile_path = metadata.get("reward_profile_path")
    if not reward_profile_path:
        raise SystemExit("Metadata does not contain reward_profile_path.")

    model_path = resolve_artifact_path(metadata_path, str(policy_value_model_path))
    vocab_path = resolve_artifact_path(metadata_path, str(metadata["policy_vocab_path"]))
    reward_profile_path_resolved = resolve_artifact_path(metadata_path, str(reward_profile_path))

    import tensorflow as tf

    policy_value_model = tf.keras.models.load_model(model_path)
    action_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    move_reward_profile = json.loads(reward_profile_path_resolved.read_text(encoding="utf-8"))
    reward_config = RewardConfig.from_dict(metadata.get("reward_config"))
    idx_to_token = {int(idx): token for token, idx in action_vocab.items()}

    tracker = BattleStateTracker(form_change_species={"Palafin"})
    include_switches = bool(metadata.get("include_switches", False))
    label_format = str(metadata.get("policy_label_format", "action_tokens"))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        battles_written = 0
        for path in json_paths:
            if battles_written >= args.max_battles:
                break

            try:
                battle = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if "turns" not in battle:
                continue

            rows = generate_battle_trace_rows(
                battle,
                tracker=tracker,
                policy_value_model=policy_value_model,
                idx_to_token=idx_to_token,
                include_switches=include_switches,
                label_format=label_format,
                top_k=args.top_k,
                reward_config=reward_config,
                move_reward_profile=move_reward_profile,
            )
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            battles_written += 1

    print(f"wrote_trace_rows_to={output_path.resolve()}")


if __name__ == "__main__":
    main()
