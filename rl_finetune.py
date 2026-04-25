"""REINFORCE RL fine-tuning for any entity-action baseline.

Reads per-decision sharded training examples (produced by the model-league
training pipeline) or legacy per-game ``rl_examples_*.jsonl`` files, encodes
each decision through the v1 entity pipeline, and does one online gradient
update per example using a REINFORCE policy-gradient loss with an advantage
baseline (``outcome - value_pred``) plus a value-head MSE term when a value
head is present.

Usage (new sharded pipeline, via a training-job manifest)::

    python3 rl_finetune.py \\
        --base-model entity_action_v2_20260409_1811 \\
        --job-manifest databases/model-league/training/bundles/.../manifest.json \\
        --use-sharded-adapter

Usage (new sharded pipeline, explicit paths)::

    python3 rl_finetune.py \\
        --base-model entity_action_v2_20260409_1811 \\
        --data-paths path/a.jsonl path/b.jsonl \\
        --use-sharded-adapter

Usage (legacy, preserved)::

    python3 rl_finetune.py --base-model entity_action_bc_v1_20260408_0428 \\
                           --examples-dir training/examples
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: repo root must be first on sys.path so the top-level shims
# (EntityTensorization.py -> core.EntityTensorization, etc.) resolve correctly.
# The server/ sub-package is NOT added — EntityServerRuntime lives in server/
# but we import directly from the root-level shims here.
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
ROOT = REPO.parent
SIM_REPO = Path(os.environ.get("POKEMON_SHOWDOWN_REPO", str(ROOT / "pokemon-showdown-model-feature")))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EXAMPLES_DIR = SIM_REPO / "training" / "examples"
RL_ROUND_PATTERN = re.compile(r".*_ft_(\d{8}_\d{4})$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_policy_logits(outputs: Any) -> Any:
    """Extract policy logits tensor from whatever shape the model returns."""
    if isinstance(outputs, dict):
        return outputs.get("policy") or next(iter(outputs.values()))
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def _extract_value(outputs: Any) -> Any | None:
    """Extract value tensor if present, else None."""
    if isinstance(outputs, dict) and "value" in outputs:
        return outputs["value"]
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        return outputs[1]
    return None


def _resolve_manifest_path(rel_or_abs: str) -> Path:
    """Resolve a path from manifest.buffer.exampleFiles.

    Manifest entries are written relative to the simulator repo root; absolute
    paths are honoured as-is.
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    # Try simulator-repo-relative first.
    candidate = SIM_REPO / rel_or_abs
    if candidate.exists():
        return candidate
    # Then trainer-repo-relative as a last resort.
    return REPO / rel_or_abs


def _discover_example_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve the list of shard paths from CLI flags.

    Precedence (highest first):
        1. --job-manifest  — read ``buffer.exampleFiles``
        2. --data-paths    — explicit absolute paths
        3. --examples-dir  — legacy glob ``rl_examples_*.jsonl``
    """
    if args.job_manifest is not None:
        manifest_path = Path(args.job_manifest)
        if not manifest_path.exists():
            sys.exit(f"[error] --job-manifest not found: {manifest_path}")
        manifest = _load_json(manifest_path)
        files = (manifest.get("buffer") or {}).get("exampleFiles") or []
        resolved = [_resolve_manifest_path(rel) for rel in files]
        missing = [p for p in resolved if not p.exists()]
        if missing:
            print(
                f"[warn] {len(missing)} shard paths from manifest not found; "
                f"skipping them. First: {missing[0]}"
            )
        return [p for p in resolved if p.exists()]

    if args.data_paths:
        return [Path(p) for p in args.data_paths]

    # Legacy: glob rl_examples_*.jsonl under --examples-dir
    examples_dir = args.examples_dir
    if examples_dir is None or not examples_dir.exists():
        return []
    return sorted(examples_dir.glob("rl_examples_*.jsonl"))


# ---------------------------------------------------------------------------
# Data collection — legacy per-game JSONL
# ---------------------------------------------------------------------------

def _iter_legacy_games(
    jsonl_paths: list[Path], *, base_model_id: str
) -> list[dict]:
    """Read ``rl_examples_*.jsonl`` files and yield game dicts.

    Only games whose ``model_id`` matches ``base_model_id`` are kept.
    """
    games: list[dict] = []
    skipped_model = 0
    for fpath in jsonl_paths:
        with open(fpath, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    print(f"[warn] Skipping malformed JSON line in {fpath.name}")
                    continue
                if record.get("model_id") != base_model_id:
                    skipped_model += 1
                    continue
                games.append(record)
    if skipped_model:
        print(f"[data] Legacy games skipped (wrong model): {skipped_model}")
    return games


# ---------------------------------------------------------------------------
# Data collection — encode decisions into training tuples
# ---------------------------------------------------------------------------

def _encode_games(
    games: list[dict],
    *,
    policy_vocab: dict[str, int],
    token_vocabs: dict[str, Any],
    min_games: int,
    dry_run: bool,
    drop_counts: dict[str, int],
    history_dims: dict[str, Any] | None = None,
) -> list[tuple[dict, int, float]]:
    """Encode ``games`` into ``(batched_inputs, action_index, outcome)`` tuples.

    Updates ``drop_counts`` with ``vocab_miss`` and ``total_seen``.
    """
    from EntityTensorization import (  # noqa: PLC0415
        encode_entity_state,
        to_single_example_entity_inputs,
    )

    history_dims = history_dims or {"enabled": False}

    all_examples: list[tuple[dict, int, float]] = []
    games_loaded = 0
    games_skipped_min = 0
    encode_failures = 0

    for record in games:
        outcome = float(record.get("outcome", 0))
        perspective_player = str(record.get("perspective_player", "p1"))
        decisions = record.get("decisions") or []

        valid_examples_for_game: list[tuple[dict | None, int, float]] = []
        for decision in decisions:
            drop_counts["total_seen"] += 1
            state_json = decision.get("state_json")
            model_response = decision.get("modelResponse") or {}
            action_token = model_response.get("action_token")

            if state_json is None:
                continue
            if not action_token:
                continue
            if action_token not in policy_vocab:
                drop_counts["vocab_miss"] += 1
                continue

            action_index = policy_vocab[action_token]

            try:
                encoded = encode_entity_state(
                    state_json,
                    perspective_player=perspective_player,
                    token_vocabs=token_vocabs,
                )
                if dry_run:
                    # Validate encoding but drop tensor data to save RAM.
                    valid_examples_for_game.append((None, action_index, outcome))
                else:
                    batched = to_single_example_entity_inputs(encoded)
                    # If history-decoding is active, the model expects
                    # event_history_tokens/mask inputs. The sharded training
                    # records don't carry proper per-turn event streams, so we
                    # supply zero-filled placeholders here. At init the policy
                    # head's history-column weights are zero-spliced, so the
                    # zero history input produces no change vs baseline — the
                    # decoder variant trains as if the architecture were the
                    # same as the baseline until we plumb real history tokens.
                    # TODO: reconstruct per-turn events from sharded records or
                    # pull from captured replay logs for a meaningful decoder
                    # experiment. See TurnEventTokenizer.encode_event_history.
                    history_turns = history_dims.get("turns", 8)
                    history_events = history_dims.get("events_per_turn", 24)
                    if history_dims.get("enabled"):
                        import numpy as _np_hist  # noqa: PLC0415
                        batched["event_history_tokens"] = _np_hist.zeros(
                            (1, history_turns, history_events), dtype=_np_hist.int32
                        )
                        batched["event_history_mask"] = _np_hist.zeros(
                            (1, history_turns), dtype=_np_hist.float32
                        )
                    valid_examples_for_game.append((batched, action_index, outcome))
            except Exception as exc:  # noqa: BLE001
                encode_failures += 1
                warnings.warn(
                    f"[warn] Encoding failed for game "
                    f"{record.get('game_id', '?')} turn "
                    f"{decision.get('turn', '?')}: {exc}"
                )

        if len(valid_examples_for_game) < 3:
            games_skipped_min += 1
            continue

        all_examples.extend(valid_examples_for_game)  # type: ignore[arg-type]
        games_loaded += 1

    print(f"[data] Games accepted  : {games_loaded}")
    print(f"[data] Games skipped (<3 valid decisions): {games_skipped_min}")
    print(f"[data] Encoding failures: {encode_failures}")
    print(f"[data] Decisions total  : {len(all_examples)}")

    if games_loaded < min_games:
        print(
            f"[warn] Only {games_loaded} games loaded (min_games={min_games}). "
            "Pass --min-games to lower the threshold or collect more game records."
        )

    return all_examples


# ---------------------------------------------------------------------------
# Artifact scaffolding
# ---------------------------------------------------------------------------

def _materialise_output_artifact_dir(
    *,
    base_artifact_dir: Path,
    base_model_id: str,
    output_model_id: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Create the new artifact directory and copy the vocab files.

    Returns a dict of the copied paths (keyed by logical filename) so the
    trainer can refer to them when writing metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_pairs = {
        "policy_vocab": f"{base_model_id}.policy_vocab.json",
        "entity_token_vocabs": f"{base_model_id}.entity_token_vocabs.json",
    }
    copied: dict[str, Path] = {}
    for key, fname in vocab_pairs.items():
        src = base_artifact_dir / fname
        if not src.exists():
            # Non-fatal: trainer will log if a downstream step needs it.
            print(f"[warn] Baseline vocab missing, not copied: {src.name}")
            continue
        # Rename to reference the new model id so serving can load by id.
        renamed = fname.replace(base_model_id, output_model_id)
        dst = output_dir / renamed
        shutil.copy2(src, dst)
        copied[key] = dst
    return copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _default_output_model_id(base_model_id: str) -> str:
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M")
    return f"{base_model_id}_ft_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REINFORCE RL fine-tuning with advantage baseline",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Model id of the baseline (e.g. entity_action_bc_v1_20260408_0428)",
    )
    parser.add_argument(
        "--job-manifest",
        type=Path,
        default=None,
        help=(
            "Path to bundles/<id>/manifest.json. When set, training data is "
            "read from manifest.buffer.exampleFiles[]. Takes precedence over "
            "--data-paths and --examples-dir."
        ),
    )
    parser.add_argument(
        "--data-paths",
        nargs="*",
        default=None,
        help="Explicit shard paths (space-separated). Used if --job-manifest not set.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=DEFAULT_EXAMPLES_DIR,
        help="Legacy: directory containing rl_examples_*.jsonl files",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        help="Output model id (default: <base-model>_ft_<YYYYMMDD_HHMM>)",
    )
    parser.add_argument(
        "--use-sharded-adapter",
        action="store_true",
        help=(
            "Read sharded per-decision JSONL records via core.format_adapter "
            "instead of the legacy per-game rl_examples_*.jsonl format."
        ),
    )
    parser.add_argument(
        "--drop-fallback-mode",
        choices=("decision", "game"),
        default="decision",
        help=(
            "How to handle records where usedFallback=true: drop only that "
            "decision (default) or drop the whole game."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of full passes over the training data (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Adam learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=50,
        help="Warn if fewer than this many games are loaded (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate data only; skip model loading and gradient steps",
    )
    parser.add_argument(
        "--policy-reads-history",
        action="store_true",
        help=(
            "Build the decoder variant: the policy head reads the history-fused "
            "representation (Concatenate([shared, history_context])) instead of "
            "the history-free backbone. At init, the expanded policy head is "
            "spliced so it is numerically equivalent to the baseline (zeros on "
            "the new attn_dim columns). Requires a baseline with use_history=True."
        ),
    )
    parser.add_argument(
        "--kl-beta",
        type=float,
        default=0.0,
        help=(
            "KL-regularization coefficient. When >0, the loss gains a term "
            "beta * (log_p_new - log_p_baseline) on the chosen action, "
            "anchoring the new policy toward the frozen baseline to prevent "
            "divergence on small/noisy data. 0.05-0.5 are typical. Default 0 "
            "(off) preserves vanilla REINFORCE behavior."
        ),
    )
    args = parser.parse_args()

    base_model_id: str = args.base_model
    output_model_id: str = args.output_model or _default_output_model_id(base_model_id)

    # ------------------------------------------------------------------
    # 1. Load baseline metadata and vocabs
    # ------------------------------------------------------------------
    base_artifact_dir = REPO / "artifacts" / base_model_id
    metadata_path = (
        base_artifact_dir / f"training_metadata_{base_model_id}.local.json"
    )
    if not metadata_path.exists():
        sys.exit(f"[error] Baseline metadata not found: {metadata_path}")

    metadata: dict[str, Any] = _load_json(metadata_path)

    policy_vocab_path = Path(metadata["policy_vocab_path"])
    token_vocab_path = Path(metadata["entity_token_vocab_path"])
    if not policy_vocab_path.exists():
        sys.exit(f"[error] Policy vocab not found: {policy_vocab_path}")
    if not token_vocab_path.exists():
        sys.exit(f"[error] Token vocabs not found: {token_vocab_path}")
    policy_vocab: dict[str, int] = _load_json(policy_vocab_path)
    token_vocabs: dict[str, Any] = _load_json(token_vocab_path)

    print(f"[config] Base model     : {base_model_id}")
    print(f"[config] Output model   : {output_model_id}")
    print(f"[config] Rounds         : {args.rounds}")
    print(f"[config] Learning rate  : {args.lr}")
    print(f"[config] Min games      : {args.min_games}")
    print(f"[config] Dry run        : {args.dry_run}")
    print(f"[config] Policy vocab   : {len(policy_vocab)} tokens")
    print(f"[config] Sharded adapter: {args.use_sharded_adapter}")
    print(f"[config] Drop fallback  : {args.drop_fallback_mode}")

    # ------------------------------------------------------------------
    # 2. Resolve shard paths and load game dicts
    # ------------------------------------------------------------------
    shard_paths = _discover_example_paths(args)
    if not shard_paths:
        sys.exit("[error] No example shards discovered.")
    print(f"[data] Shards          : {len(shard_paths)}")

    drop_counts: dict[str, int] = {
        "used_fallback": 0,
        "ties": 0,
        "vocab_miss": 0,
        "total_seen": 0,   # decisions inspected inside _encode_games
        "records_seen": 0,  # raw shard records inspected by the adapter
    }

    if args.use_sharded_adapter:
        from core.format_adapter import sharded_to_per_game  # noqa: PLC0415

        games, adapter_drops = sharded_to_per_game(
            shard_paths,
            drop_fallback_mode=args.drop_fallback_mode,
            base_model_id=base_model_id,
        )
        drop_counts["used_fallback"] = adapter_drops.get("used_fallback", 0)
        drop_counts["ties"] = adapter_drops.get("ties", 0)
        drop_counts["records_seen"] = adapter_drops.get("records_seen", 0)
        print(
            f"[data] Sharded adapter: {len(games)} games ready; "
            f"dropped {drop_counts['used_fallback']} fallback decisions, "
            f"{drop_counts['ties']} ties"
        )
    else:
        games = _iter_legacy_games(shard_paths, base_model_id=base_model_id)
        print(f"[data] Legacy games    : {len(games)}")

    if not games:
        sys.exit("[error] No games available after filtering.")

    _history_dims_for_encode = {
        "enabled": bool(args.policy_reads_history),
        "turns": int(metadata.get("history_turns", 8)),
        "events_per_turn": int(metadata.get("history_events_per_turn", 24)),
    }
    all_examples = _encode_games(
        games,
        policy_vocab=policy_vocab,
        token_vocabs=token_vocabs,
        min_games=args.min_games,
        dry_run=args.dry_run,
        drop_counts=drop_counts,
        history_dims=_history_dims_for_encode,
    )

    print(
        f"[data] Drop counts     : {drop_counts}  "
        f"(total_seen includes every decision the adapter emitted)"
    )

    if args.dry_run:
        print("[dry-run] Data validation complete. Skipping model load and training.")
        return

    if not all_examples:
        sys.exit("[error] No training examples collected. Cannot proceed.")

    # ------------------------------------------------------------------
    # 3. Pick the output artifact dir and copy vocab files now — we'd
    #    rather fail fast than after a 30-minute training loop.
    # ------------------------------------------------------------------
    output_dir = REPO / "artifacts" / output_model_id
    print(f"[config] Output dir     : {output_dir}")
    copied_vocabs = _materialise_output_artifact_dir(
        base_artifact_dir=base_artifact_dir,
        base_model_id=base_model_id,
        output_model_id=output_model_id,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # 4. Load model — rebuild architecture then load weights (same pattern
    #    as EntityServerRuntime.py to avoid Lambda deserialization failures).
    # ------------------------------------------------------------------
    print("[model] Rebuilding entity_action_bc architecture...")
    from core.EntityModelV1 import build_entity_action_models  # noqa: PLC0415

    baseline_use_history = bool(metadata.get("use_history", False))

    # Decide which weights file to load BEFORE building, so the build's
    # predict_value setting matches the file's actual contents. metadata may
    # declare predict_value=True while only shipping a policy-only .keras; in
    # that case we must build WITHOUT value layers or load_weights will crash
    # on shape-mismatch for the value dense head.
    _pv_path = metadata.get("policy_value_model_path")
    _policy_path = metadata.get("policy_model_path")
    if args.policy_reads_history:
        # Decoder variant: needs the richer graph; the build+skip_mismatch load
        # below handles the splice. Force predict_value so history + value +
        # aux layers all materialize for named-weight loading.
        _predict_value = True
        baseline_has_value_head = True  # from-file perspective
    elif _pv_path and Path(_pv_path).exists():
        _predict_value = True
        baseline_has_value_head = True
    elif _policy_path and Path(_policy_path).exists():
        _predict_value = False
        baseline_has_value_head = False
    else:
        sys.exit(
            "[error] No loadable weights file found. Checked "
            f"policy_value_model_path={_pv_path} and policy_model_path={_policy_path}."
        )

    # When the decoder variant is requested, the rebuilt graph must match the
    # baseline's history encoder (so the conv/LSTM/attention weights load by name)
    # AND must include policy_reads_history=True so the policy head reads the
    # fused representation.
    _build_kwargs: dict[str, Any] = dict(
        vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
        num_policy_classes=int(metadata["num_action_classes"]),
        hidden_dim=int(metadata["hidden_dim"]),
        depth=int(metadata["depth"]),
        dropout=float(metadata["dropout"]),
        learning_rate=float(metadata["learning_rate"]),
        token_embed_dim=int(metadata.get("token_embed_dim", 24)),
        predict_value=_predict_value,
        value_hidden_dim=int(metadata.get("value_hidden_dim", 128)),
        value_weight=float(metadata.get("value_weight", 0.25)),
    )
    if args.policy_reads_history:
        if not baseline_use_history:
            sys.exit(
                "[error] --policy-reads-history requires the baseline to have "
                "use_history=True. Baseline metadata.use_history="
                f"{baseline_use_history}."
            )
        # Mirror the baseline's history encoder config so weights load by name.
        _sequence_vocab_size = int(metadata.get("sequence_vocab_size") or 0)
        if _sequence_vocab_size <= 0:
            sys.exit(
                "[error] --policy-reads-history requires metadata.sequence_vocab_size > 0."
            )
        _build_kwargs.update(
            use_history=True,
            history_vocab_size=_sequence_vocab_size,
            history_embed_dim=int(metadata.get("history_embed_dim", 32)),
            history_lstm_dim=int(metadata.get("history_lstm_dim", 64)),
            history_turns=int(metadata.get("history_turns", 8)),
            history_events_per_turn=int(metadata.get("history_events_per_turn", 24)),
            policy_reads_history=True,
        )
        print("[model] Decoder variant: policy_reads_history=True")
    _, policy_only_model, policy_value_model, _ = build_entity_action_models(
        **_build_kwargs
    )
    # When the baseline has no value head, the training-model slot comes back
    # equal to (or wrapping) the policy-only model — use it uniformly so the
    # rest of this function can reference policy_value_model without branching.
    if not baseline_has_value_head:
        policy_value_model = policy_only_model

    # For the decoder variant we always need a weights source that carries the
    # full history encoder + value head + (optionally) aux heads. Prefer an
    # explicit policy_value_model_path; fall back to the on-disk convention
    # ``policy_value_model_<id>.keras`` alongside the other artifacts; and if
    # nothing else works, fall back to training_model_path (the richest graph).
    if args.policy_reads_history:
        candidates: list[Path] = []
        if "policy_value_model_path" in metadata:
            candidates.append(Path(metadata["policy_value_model_path"]))
        candidates.append(
            base_artifact_dir / f"policy_value_model_{base_model_id}.keras"
        )
        if "training_model_path" in metadata:
            candidates.append(Path(metadata["training_model_path"]))
        weights_src = next((p for p in candidates if p.exists()), None)
        if weights_src is None:
            sys.exit(
                "[error] --policy-reads-history: no baseline weights file found "
                "with history encoder. Looked for policy_value_model_*.keras "
                "and training_model_*.keras under "
                f"{base_artifact_dir}."
            )
        src_key = "policy_value_or_training_model_path"
    else:
        # Prefer policy_value_model_path when the path key is actually present
        # (not merely because predict_value=True is recorded — some metadata
        # files, e.g. entity_history_v1, declare the flag but omit the path).
        if "policy_value_model_path" in metadata:
            src_key = "policy_value_model_path"
        else:
            src_key = "policy_model_path"
        weights_src = Path(metadata[src_key])
        if not weights_src.exists():
            sys.exit(f"[error] {src_key} not found: {weights_src}")

    print(f"[model] Loading weights from {weights_src.name} ({src_key}) ...")
    # For the decoder variant, the policy Dense layer has a wider kernel
    # ([hidden_dim + attn_dim, num_actions]) than the baseline
    # ([hidden_dim, num_actions]), so strict load_weights() fails. We load with
    # skip_mismatch=True to populate every matching layer, then splice the policy
    # head manually below so step 0 is numerically equivalent to the baseline.
    if args.policy_reads_history:
        policy_value_model.load_weights(str(weights_src), skip_mismatch=True)
    else:
        policy_value_model.load_weights(str(weights_src))
    print("[model] Weights loaded.")

    # Probe output shape so we can log it once and handle it correctly below.
    import tensorflow as tf  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if args.policy_reads_history:
        # Splice the policy head so that step 0 matches the baseline policy
        # outputs exactly: the first `hidden_dim` rows of the new kernel take
        # the baseline's [hidden_dim, num_actions] weights; the remaining
        # `attn_dim` rows are zero. The bias copies straight across.
        #
        # We read the baseline policy weights by reloading the baseline file
        # via a throwaway build with policy_reads_history=False.
        from core.EntityModelV1 import build_entity_action_models as _bm  # noqa: PLC0415

        _baseline_build_kwargs = dict(_build_kwargs)
        _baseline_build_kwargs["policy_reads_history"] = False
        # The baseline-shape graph still includes the history encoder so weights
        # load by name; only the policy head shape differs.
        _, _baseline_policy_model, _baseline_policy_value_model, _ = _bm(
            **_baseline_build_kwargs
        )
        # Use the policy_value artifact from the baseline build (it has both
        # the history encoder and the value head, matching our weights_src).
        _baseline_reload_target = _baseline_policy_value_model or _baseline_policy_model
        _baseline_reload_target.load_weights(str(weights_src))

        _baseline_policy_layer = _baseline_reload_target.get_layer("policy")
        _base_kernel, _base_bias = _baseline_policy_layer.get_weights()
        _new_policy_layer = policy_value_model.get_layer("policy")
        _new_kernel, _new_bias = _new_policy_layer.get_weights()
        _hidden_dim = int(metadata["hidden_dim"])
        _attn_dim_expected = _new_kernel.shape[0] - _hidden_dim
        if _attn_dim_expected <= 0:
            sys.exit(
                f"[error] decoder policy kernel shape {_new_kernel.shape} does "
                f"not exceed hidden_dim={_hidden_dim}; cannot splice."
            )
        # Splice: first hidden_dim rows = baseline kernel; last attn_dim rows = 0
        _spliced_kernel = np.zeros_like(_new_kernel)
        _spliced_kernel[:_hidden_dim, :] = _base_kernel
        # The remaining rows stay at 0 (zeros_like) — this is what makes step 0
        # numerically equivalent to the baseline: history_context multiplies zero.
        _new_policy_layer.set_weights([_spliced_kernel, _base_bias])
        print(
            f"[model] Spliced policy head: baseline kernel "
            f"{_base_kernel.shape} -> decoder kernel {_spliced_kernel.shape} "
            f"(first {_hidden_dim} rows=baseline, last {_attn_dim_expected} rows=0)"
        )
        # Free the baseline throwaway graph.
        del _baseline_reload_target, _baseline_policy_model, _baseline_policy_value_model

    _dummy_inputs = {
        "global_conditions":      np.zeros((1, 5),      dtype=np.int32),
        "global_numeric":         np.zeros((1, 17),     dtype=np.float32),
        "pokemon_ability":        np.zeros((1, 12),     dtype=np.int32),
        "pokemon_item":           np.zeros((1, 12),     dtype=np.int32),
        "pokemon_numeric":        np.zeros((1, 12, 13), dtype=np.float32),
        "pokemon_observed_moves": np.zeros((1, 12, 4),  dtype=np.int32),
        "pokemon_side":           np.zeros((1, 12),     dtype=np.int32),
        "pokemon_slot":           np.zeros((1, 12),     dtype=np.int32),
        "pokemon_species":        np.zeros((1, 12),     dtype=np.int32),
        "pokemon_status":         np.zeros((1, 12),     dtype=np.int32),
        "pokemon_tera":           np.zeros((1, 12),     dtype=np.int32),
        "weather":                np.zeros((1, 1),      dtype=np.int32),
    }
    _probe = policy_value_model(_dummy_inputs, training=False)
    _policy_probe = _extract_policy_logits(_probe)
    _value_probe = _extract_value(_probe)
    print(
        f"[model] Policy logits shape : {_policy_probe.shape}  "
        f"(expected [{metadata['num_action_classes']}])"
    )
    print(f"[model] Has value output    : {_value_probe is not None}")
    has_value_head = _value_probe is not None
    if not has_value_head:
        warnings.warn(
            "[warn] Baseline has no value head; advantage baseline will fall "
            "back to (outcome - 0.5)."
        )

    # ------------------------------------------------------------------
    # 5. REINFORCE training loop — online (one example at a time)
    #    Loss: -log pi(a|s) * (outcome - stop_gradient(value_pred))
    #          + 0.5 * (outcome - value_pred)**2   when value head present
    # ------------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    num_examples = len(all_examples)

    # Cache baseline log-prob of the chosen action per example. At this point
    # the model still holds the baseline-init weights (for the decoder variant,
    # zero-init on the new history columns makes it numerically equivalent to
    # the baseline). The cache becomes the KL reference distribution.
    baseline_log_probs: list[float] = []
    if args.kl_beta > 0:
        print(f"[kl] Caching baseline log-probs (beta={args.kl_beta}) ...")
        for batched_inputs, action_index, _ in all_examples:
            out = policy_value_model(batched_inputs, training=False)
            logits = tf.reshape(_extract_policy_logits(out), [-1])
            baseline_log_probs.append(
                float(tf.nn.log_softmax(logits)[action_index])
            )
        print(f"[kl] Cached {len(baseline_log_probs)} baseline log-probs.")

    for round_idx in range(args.rounds):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0

        for ex_idx, (batched_inputs, action_index, outcome) in enumerate(
            all_examples
        ):
            try:
                with tf.GradientTape() as tape:
                    outputs = policy_value_model(batched_inputs, training=True)
                    policy_logits_raw = _extract_policy_logits(outputs)
                    policy_logits = tf.reshape(policy_logits_raw, [-1])
                    log_probs = tf.nn.log_softmax(policy_logits)
                    chosen_log_prob = log_probs[action_index]

                    if has_value_head:
                        value_raw = _extract_value(outputs)
                        value_scalar = tf.reshape(value_raw, [-1])[0]
                        advantage = float(outcome) - tf.stop_gradient(value_scalar)
                        policy_loss = -chosen_log_prob * advantage
                        value_loss = tf.square(value_scalar - float(outcome))
                        loss = policy_loss + 0.5 * value_loss
                    else:
                        # Fall back to a numeric baseline of the mean outcome.
                        advantage = float(outcome) - 0.5
                        policy_loss = -chosen_log_prob * advantage
                        value_loss = tf.constant(0.0)
                        loss = policy_loss

                    if args.kl_beta > 0:
                        kl_term = chosen_log_prob - baseline_log_probs[ex_idx]
                        kl_loss = args.kl_beta * kl_term
                        loss = loss + kl_loss
                    else:
                        kl_loss = tf.constant(0.0)

                grads = tape.gradient(loss, policy_value_model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                optimizer.apply_gradients(
                    zip(grads, policy_value_model.trainable_variables)
                )
                total_loss += float(loss)
                total_policy_loss += float(policy_loss)
                total_value_loss += float(value_loss) if has_value_head else 0.0
                total_kl += float(kl_loss) if args.kl_beta > 0 else 0.0

            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Gradient step failed for example {ex_idx}: {exc}")
                traceback.print_exc()

        avg_loss = total_loss / max(num_examples, 1)
        avg_policy = total_policy_loss / max(num_examples, 1)
        avg_value = total_value_loss / max(num_examples, 1)
        avg_kl = total_kl / max(num_examples, 1)
        kl_str = f"  kl={avg_kl:.4f}" if args.kl_beta > 0 else ""
        print(
            f"[round {round_idx + 1}/{args.rounds}] "
            f"loss={avg_loss:.4f}  policy={avg_policy:.4f}  value={avg_value:.4f}"
            f"{kl_str}"
        )

    # ------------------------------------------------------------------
    # 6. Save checkpoint (vocab files already copied in step 3)
    # ------------------------------------------------------------------
    policy_value_out = output_dir / f"policy_value_model_{output_model_id}.keras"
    policy_only_out = output_dir / f"{output_model_id}.keras"

    if baseline_has_value_head:
        print(f"[save] Writing policy-value model -> {policy_value_out.name}")
        policy_value_model.save(str(policy_value_out))

    # policy_only_model shares weights with policy_value_model (same backbone),
    # so saving it after the training loop captures the updated weights.
    print(f"[save] Writing policy-only model  -> {policy_only_out.name}")
    policy_only_model.save(str(policy_only_out))

    # Build output metadata by extending the base metadata.
    out_metadata = dict(metadata)
    out_metadata["model_name"] = output_model_id
    out_metadata["model_release_id"] = output_model_id
    out_metadata["parent_release_id"] = base_model_id
    out_metadata["base_model_id"] = base_model_id
    out_metadata["policy_model_path"] = str(policy_only_out)
    if baseline_has_value_head:
        out_metadata["policy_value_model_path"] = str(policy_value_out)
    else:
        out_metadata.pop("policy_value_model_path", None)
    regime = "reinforce_v1_advantage"
    if args.kl_beta > 0:
        regime += "_kl"
    if args.policy_reads_history:
        regime += "_decode"
    out_metadata["training_regime"] = regime
    out_metadata["kl_beta"] = float(args.kl_beta)
    out_metadata["policy_reads_history"] = bool(args.policy_reads_history)
    out_metadata["drop_counts"] = drop_counts
    # Point at the copied vocab files (keep hyperparameters intact).
    if "policy_vocab" in copied_vocabs:
        out_metadata["policy_vocab_path"] = str(copied_vocabs["policy_vocab"])
    if "entity_token_vocabs" in copied_vocabs:
        out_metadata["entity_token_vocab_path"] = str(
            copied_vocabs["entity_token_vocabs"]
        )

    # Write the serving-local metadata (absolute paths) and a portable copy.
    local_meta_path = output_dir / f"training_metadata_{output_model_id}.local.json"
    with open(local_meta_path, "w", encoding="utf-8") as fh:
        json.dump(out_metadata, fh, indent=2)
    print(f"[save] Metadata (.local) -> {local_meta_path.name}")

    portable_meta_path = output_dir / f"training_metadata_{output_model_id}.json"
    with open(portable_meta_path, "w", encoding="utf-8") as fh:
        json.dump(out_metadata, fh, indent=2)
    print(f"[save] Metadata         -> {portable_meta_path.name}")

    print(f"\nCheckpoint saved to: {output_dir}")
    print("Add to league with ps-league add:")
    print(f"  /ps-league add {output_model_id}")


if __name__ == "__main__":
    main()
