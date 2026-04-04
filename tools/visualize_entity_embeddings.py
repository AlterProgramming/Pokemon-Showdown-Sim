from __future__ import annotations

"""HTML viewer for learned entity embedding tables.

This is an observability tool, not a training dependency. It gives us a quick
qualitative way to inspect what the learned token spaces start to resemble:
    - are similar species clustering?
    - do move embeddings develop neighborhoods?
    - is the unknown token drifting toward sensible regions?

The projection is intentionally lightweight so it can be run right after a model
finishes training.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from EntityModelV1 import build_entity_action_models
from ModelRegistry import resolve_artifact_path


LAYER_SPECS: List[Tuple[str, str, str]] = [
    ("species_embedding", "species", "Species Embedding"),
    ("move_embedding", "move", "Observed Move Embedding"),
    ("item_embedding", "item", "Item Embedding"),
    ("ability_embedding", "ability", "Ability Embedding"),
    ("tera_embedding", "tera", "Tera Type Embedding"),
    ("status_embedding", "status", "Status Embedding"),
    ("weather_embedding", "weather", "Weather Embedding"),
    ("global_condition_embedding", "global_condition", "Global Condition Embedding"),
]


def resolve_artifact_paths(metadata_path: Path) -> tuple[dict[str, Any], Path, Path]:
    """Resolve the saved policy model and token vocab bundle from training metadata."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_ref = metadata.get("policy_model_path")
    vocab_ref = metadata.get("entity_token_vocab_path")
    if not model_ref or not vocab_ref:
        raise SystemExit("Metadata must contain policy_model_path and entity_token_vocab_path.")
    repo_path = metadata_path.parent.parent.resolve()
    model = resolve_artifact_path(repo_path, metadata_path, str(model_ref))
    vocab = resolve_artifact_path(repo_path, metadata_path, str(vocab_ref))
    return metadata, model, vocab


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """Convert token->id mappings into id->token mappings for display."""
    return {int(idx): str(token) for token, idx in vocab.items()}


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if matrix.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    # PCA is enough for a quick qualitative view; we mainly want a stable first look at
    # whether nearby tokens begin to cluster semantically during training.
    components = vt[:2].T
    projected = centered @ components
    if projected.shape[1] == 1:
        projected = np.concatenate([projected, np.zeros((projected.shape[0], 1), dtype=projected.dtype)], axis=1)
    return projected[:, :2].astype(np.float32)


def normalize_points(points: np.ndarray, width: float = 840.0, height: float = 320.0) -> np.ndarray:
    """Scale projected coordinates into the fixed SVG viewport."""
    if len(points) == 0:
        return points
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    norm = (points - mins) / spans
    norm[:, 0] = 30.0 + norm[:, 0] * (width - 60.0)
    norm[:, 1] = 30.0 + norm[:, 1] * (height - 60.0)
    return norm


def cosine_neighbors(weights: np.ndarray, top_k: int = 8) -> Dict[int, List[int]]:
    if len(weights) == 0:
        return {}
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = weights / norms
    sims = normalized @ normalized.T
    np.fill_diagonal(sims, -np.inf)
    neighbors: Dict[int, List[int]] = {}
    for idx in range(weights.shape[0]):
        order = np.argsort(-sims[idx])[:top_k]
        neighbors[idx] = [int(item) for item in order if np.isfinite(sims[idx, item])]
    return neighbors


def build_embedding_payload(model_path: Path, vocab_path: Path) -> Dict[str, Any]:
    """Extract embedding layers plus nearest-neighbor summaries into a display payload."""
    try:
        from tensorflow import keras
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorFlow is required to visualize learned entity embeddings.") from exc

    model = keras.models.load_model(model_path)
    token_vocabs = json.loads(vocab_path.read_text(encoding="utf-8"))

    panels: List[Dict[str, Any]] = []
    for layer_name, vocab_key, title in LAYER_SPECS:
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        matrix = np.asarray(weights[0], dtype=np.float32)
        vocab = token_vocabs.get(vocab_key) or {}
        id_to_token = invert_vocab(vocab)
        keep_ids = [
            idx
            for idx in sorted(id_to_token)
            if id_to_token[idx] not in {"<PAD>"}
        ]
        # Skip only PAD here. Keeping <UNK> visible is useful because it shows whether
        # unknown-token geometry drifts toward a meaningful neighborhood.
        if not keep_ids:
            continue
        keep_matrix = matrix[keep_ids]
        labels = [id_to_token[idx] for idx in keep_ids]
        projected = normalize_points(pca_2d(keep_matrix))
        neighbors_local = cosine_neighbors(keep_matrix)
        # Store neighbors as labels, not ids, so the HTML view remains understandable
        # without cross-referencing the raw vocab file.
        neighbor_labels = {
            labels[idx]: [labels[nbr] for nbr in neighbor_ids]
            for idx, neighbor_ids in neighbors_local.items()
        }
        points = [
            {
                "label": labels[idx],
                "x": float(projected[idx, 0]),
                "y": float(projected[idx, 1]),
                "neighbors": neighbor_labels.get(labels[idx], []),
            }
            for idx in range(len(labels))
        ]
        panels.append(
            {
                "title": title,
                "layer_name": layer_name,
                "vocab_key": vocab_key,
                "count": len(points),
                "points": points,
            }
        )

    return {
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "panels": panels,
    }


def build_embedding_payload_from_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Rebuild the entity policy architecture from metadata and inspect its learned embeddings."""
    metadata, model_path, vocab_path = resolve_artifact_paths(metadata_path)
    token_vocabs = json.loads(vocab_path.read_text(encoding="utf-8"))
    _, model, _ = build_entity_action_models(
        vocab_sizes={key: len(value) for key, value in token_vocabs.items()},
        num_policy_classes=int(metadata["num_action_classes"]),
        hidden_dim=int(metadata["hidden_dim"]),
        depth=int(metadata["depth"]),
        dropout=float(metadata["dropout"]),
        learning_rate=float(metadata["learning_rate"]),
        token_embed_dim=int(metadata.get("token_embed_dim", 24)),
    )
    model.load_weights(model_path)

    panels: List[Dict[str, Any]] = []
    for layer_name, vocab_key, title in LAYER_SPECS:
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        matrix = np.asarray(weights[0], dtype=np.float32)
        vocab = token_vocabs.get(vocab_key) or {}
        id_to_token = invert_vocab(vocab)
        keep_ids = [idx for idx in sorted(id_to_token) if id_to_token[idx] not in {"<PAD>"}]
        if not keep_ids:
            continue
        keep_matrix = matrix[keep_ids]
        labels = [id_to_token[idx] for idx in keep_ids]
        projected = normalize_points(pca_2d(keep_matrix))
        neighbors_local = cosine_neighbors(keep_matrix)
        neighbor_labels = {
            labels[idx]: [labels[nbr] for nbr in neighbor_ids]
            for idx, neighbor_ids in neighbors_local.items()
        }
        points = [
            {
                "label": labels[idx],
                "x": float(projected[idx, 0]),
                "y": float(projected[idx, 1]),
                "neighbors": neighbor_labels.get(labels[idx], []),
            }
            for idx in range(len(labels))
        ]
        panels.append(
            {
                "title": title,
                "layer_name": layer_name,
                "vocab_key": vocab_key,
                "count": len(points),
                "points": points,
            }
        )

    return {
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "panels": panels,
    }


def render_html(payload: Dict[str, Any]) -> str:
    """Render the standalone HTML viewer so it can be opened locally in a browser."""
    payload_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Entity Embedding Viewer</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: #0f1218;
      color: #eef2ff;
    }}
    .page {{
      display: grid;
      grid-template-columns: 1fr 380px;
      min-height: 100vh;
    }}
    .main {{
      padding: 24px;
      background:
        radial-gradient(circle at top, rgba(46, 74, 126, 0.35), transparent 38%),
        linear-gradient(180deg, #161b25 0%, #0f1218 100%);
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 22px;
    }}
    .subtitle {{
      margin: 0 0 18px 0;
      color: #9fb0d3;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(420px, 1fr));
      gap: 16px;
    }}
    .panel-card, .sidebar-card {{
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      padding: 14px;
      box-sizing: border-box;
    }}
    .panel-title {{
      margin: 0 0 10px 0;
      font-size: 14px;
      font-weight: 700;
    }}
    svg {{
      width: 100%;
      height: 320px;
      background: rgba(255,255,255,0.02);
      border-radius: 12px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    }}
    .point {{
      fill: rgba(124, 184, 255, 0.82);
      cursor: pointer;
    }}
    .point:hover {{
      fill: rgba(255, 220, 135, 0.95);
    }}
    .sidebar {{
      padding: 24px 20px;
      border-left: 1px solid rgba(255,255,255,0.08);
      background: #0d1016;
      overflow: auto;
    }}
    .meta-row {{
      padding: 7px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      font-size: 12px;
    }}
    .meta-key {{
      color: #9fb0d3;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      color: #dbe5fb;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="main">
      <h1>Entity Embedding Viewer</h1>
      <div class="subtitle">{payload["model_path"]}</div>
      <div class="grid" id="grid"></div>
    </div>
    <div class="sidebar">
      <div class="sidebar-card">
        <div class="panel-title">Selection</div>
        <pre id="details">Click a point to inspect its nearest neighbors.</pre>
      </div>
    </div>
  </div>
  <script>
    const payload = {payload_json};
    const grid = document.getElementById("grid");
    const details = document.getElementById("details");

    for (const panel of payload.panels) {{
      const card = document.createElement("div");
      card.className = "panel-card";

      const title = document.createElement("div");
      title.className = "panel-title";
      title.textContent = panel.title + " (" + panel.count + ")";
      card.appendChild(title);

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 840 320");

      const border = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      border.setAttribute("x", "30");
      border.setAttribute("y", "30");
      border.setAttribute("width", "780");
      border.setAttribute("height", "260");
      border.setAttribute("fill", "none");
      border.setAttribute("stroke", "rgba(255,255,255,0.10)");
      border.setAttribute("stroke-width", "1");
      svg.appendChild(border);

      for (const point of panel.points) {{
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", point.x);
        circle.setAttribute("cy", point.y);
        circle.setAttribute("r", "4.5");
        circle.setAttribute("class", "point");
        circle.addEventListener("click", () => {{
          details.textContent = JSON.stringify({{
            panel: panel.title,
            token: point.label,
            nearest_neighbors: point.neighbors,
          }}, null, 2);
        }});

        const tooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
        tooltip.textContent = point.label;
        circle.appendChild(tooltip);
        svg.appendChild(circle);
      }}

      card.appendChild(svg);
      grid.appendChild(card);
    }}
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Define the CLI for embedding inspection."""
    parser = argparse.ArgumentParser(description="Visualize learned entity embedding layers from an entity_action_bc_v1 model.")
    parser.add_argument("--metadata-path", default=None, help="Training metadata JSON for an entity model run.")
    parser.add_argument("--model-path", default=None, help="Direct path to a saved Keras model.")
    parser.add_argument("--token-vocab-path", default=None, help="Direct path to the entity token vocab JSON.")
    parser.add_argument("--output-path", default="out/entity_embedding_view.html", help="Destination HTML path.")
    parser.add_argument("--json-output-path", default=None, help="Optional payload JSON sidecar.")
    return parser.parse_args()


def main() -> None:
    """Resolve artifacts, build the payload, and write HTML + JSON sidecars."""
    args = parse_args()
    if args.metadata_path:
        payload = build_embedding_payload_from_metadata(Path(args.metadata_path).resolve())
    else:
        if not args.model_path or not args.token_vocab_path:
            raise SystemExit("Provide either --metadata-path or both --model-path and --token-vocab-path.")
        model_path = Path(args.model_path).resolve()
        vocab_path = Path(args.token_vocab_path).resolve()
        payload = build_embedding_payload(model_path, vocab_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(payload), encoding="utf-8")

    json_output_path = Path(args.json_output_path) if args.json_output_path else output_path.with_suffix(".json")
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote_entity_embedding_html={output_path.resolve()}")
    print(f"wrote_entity_embedding_json={json_output_path.resolve()}")


if __name__ == "__main__":
    main()
