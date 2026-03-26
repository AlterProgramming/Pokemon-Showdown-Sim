from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from BattleStateTracker import BattleStateTracker
from ModelRegistry import resolve_artifact_path as resolve_registry_artifact_path
from RewardSignals import (
    RewardConfig,
    compute_reward_components,
    reward_total_from_components,
)
from StateVectorization import encode_state_v0


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

    for player in ("p1", "p2"):
        examples = list(tracker.iter_turn_examples(battle, player=player, include_switches=include_switches))
        if not examples:
            continue

        pre_states = np.asarray(
            [encode_state_v0(ex["state"], perspective_player=player) for ex in examples],
            dtype=np.float32,
        )
        post_states = np.asarray(
            [encode_state_v0(ex["next_state"], perspective_player=player) for ex in examples],
            dtype=np.float32,
        )
        policy_logits_pre, values_pre = predict_policy_and_value(policy_value_model, pre_states)
        _, values_post = predict_policy_and_value(policy_value_model, post_states)

        cumulative_reward = 0.0
        for idx, ex in enumerate(examples):
            reward_components = compute_reward_components(
                ex["state"],
                ex["next_state"],
                ex["action"],
                player,
                move_reward_profile,
                is_terminal_example=(idx == len(examples) - 1),
                terminal_result=ex.get("terminal_result"),
            )
            reward_total = reward_total_from_components(reward_components, reward_config)
            cumulative_reward += reward_total

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
                    "terminal_result": ex.get("terminal_result"),
                }
            )

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
