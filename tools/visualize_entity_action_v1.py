from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from BattleStateTracker import BattleStateTracker
from EntityActionV1 import build_entity_action_graph


def load_example(
    battle_path: Path,
    *,
    player: str,
    turn_number: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    battle = json.loads(battle_path.read_text(encoding="utf-8"))
    tracker = BattleStateTracker(form_change_species={"Palafin"})
    examples = list(tracker.iter_turn_examples(battle, player=player, include_switches=True))
    if not examples:
        raise SystemExit(f"No turn examples found for player {player} in {battle_path}.")

    if turn_number is None:
        return battle, examples[0]

    for ex in examples:
        if int(ex["turn_number"]) == int(turn_number):
            return battle, ex
    raise SystemExit(f"Turn {turn_number} not found for player {player} in {battle_path}.")


def graph_heads_from_metadata(metadata: dict[str, Any], graph: dict[str, Any]) -> list[dict[str, Any]]:
    objective_set = list(metadata.get("objective_set") or [])
    if not objective_set:
        return graph["heads"]

    head_map = {entry["head_id"]: entry for entry in graph["heads"]}
    heads: list[dict[str, Any]] = []
    for head_id in objective_set:
        if head_id in head_map:
            heads.append(head_map[head_id])
            continue
        heads.append(
            {
                "head_id": str(head_id),
                "head_type": "unknown",
                "description": f"{head_id} head from metadata",
            }
        )
    return heads


def layout_graph(graph: dict[str, Any]) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[str, float]] = {}
    positions["global:battle"] = {"x": 520.0, "y": 40.0, "w": 200.0, "h": 72.0}

    for entity in graph["entities"]:
        entity_id = entity["id"]
        entity_type = entity["entity_type"]
        if entity_type == "pokemon":
            side = entity["side"]
            slot_index = int(entity["slot_index"])
            x = 120.0 if side == "self" else 900.0
            y = 140.0 + (slot_index - 1) * 82.0
            positions[entity_id] = {"x": x, "y": y, "w": 220.0, "h": 64.0}
        elif entity_type == "move_candidate":
            idx = int(entity["state_features"]["candidate_index"]) - 1
            count = max(1, graph["summary"]["move_candidate_count"])
            total_width = (count - 1) * 170.0
            start_x = 520.0 - total_width / 2.0
            positions[entity_id] = {"x": start_x + idx * 170.0, "y": 680.0, "w": 150.0, "h": 56.0}
    return positions


def metadata_summary_rows(metadata: dict[str, Any] | None) -> list[tuple[str, Any]]:
    if not metadata:
        return []
    rows: list[tuple[str, Any]] = []
    for key in [
        "family_id",
        "family_version",
        "training_regime",
        "objective_set",
        "reward_definition_id",
        "value_target_definition",
        "transition_target_definition",
    ]:
        value = metadata.get(key)
        if value is None or value == [] or value == "":
            continue
        rows.append((key, value))
    return rows


def render_svg(
    *,
    graph: dict[str, Any],
    battle_name: str,
    player: str,
    turn_number: int,
) -> str:
    positions = layout_graph(graph)
    width = 1130
    height = 780

    def center_for(position: dict[str, float]) -> tuple[float, float]:
        return position["x"] + position["w"] / 2.0, position["y"] + position["h"] / 2.0

    edge_lines: list[str] = []
    for edge in graph["edges"]:
        source_pos = positions.get(edge["source"])
        target_pos = positions.get(edge["target"])
        if source_pos is None or target_pos is None:
            continue
        x1, y1 = center_for(source_pos)
        x2, y2 = center_for(target_pos)
        stroke = "#ffb184" if edge["edge_type"] == "active_matchup" else "#8ea5d7"
        stroke_width = "3" if edge["edge_type"] == "active_matchup" else "2"
        stroke_opacity = "0.92" if edge["edge_type"] == "active_matchup" else "0.32"
        edge_lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-opacity="{stroke_opacity}" stroke-width="{stroke_width}" />'
        )

    node_blocks: list[str] = []
    for entity in graph["entities"]:
        position = positions.get(entity["id"])
        if position is None:
            continue
        entity_type = entity["entity_type"]
        fill = "#586f9f"
        fill_opacity = "0.50"
        stroke = "#8ea5d7"
        stroke_opacity = "0.22"
        if entity_type == "pokemon" and entity["side"] == "self":
            fill = "#315fbc"
            fill_opacity = "0.34"
        elif entity_type == "pokemon" and entity["side"] == "opponent":
            fill = "#9f4848"
            fill_opacity = "0.30"
        elif entity_type == "move_candidate":
            fill = "#3d7e52"
            fill_opacity = "0.34"
        if entity_type == "pokemon" and entity["state_features"]["active"] > 0.5:
            stroke = "#ffdf92"
            stroke_opacity = "0.92"
        if entity_type == "move_candidate" and entity["state_features"]["is_chosen"] > 0.5:
            stroke = "#86ffb6"
            stroke_opacity = "0.92"

        title = html.escape(str(entity["display"]["title"]))
        subtitle = html.escape(str(entity["display"]["subtitle"]))
        if entity_type == "pokemon":
            meta = (
                f'species={entity["token_inputs"]["species"]} | '
                f'hp={entity["state_features"]["hp_frac"]:.2f} | '
                f'status={entity["token_inputs"]["status"]}'
            )
        elif entity_type == "move_candidate":
            meta = f'move={entity["token_inputs"]["move"]}'
        else:
            meta = f'weather={entity["token_inputs"]["weather"]}'

        node_blocks.append(
            f'<g>'
            f'<rect x="{position["x"]:.1f}" y="{position["y"]:.1f}" width="{position["w"]:.1f}" height="{position["h"]:.1f}" '
            f'rx="14" ry="14" fill="{fill}" fill-opacity="{fill_opacity}" stroke="{stroke}" stroke-opacity="{stroke_opacity}" stroke-width="2" />'
            f'<text x="{position["x"] + 12:.1f}" y="{position["y"] + 22:.1f}" fill="#eef2ff" font-size="14" font-weight="700">{title}</text>'
            f'<text x="{position["x"] + 12:.1f}" y="{position["y"] + 38:.1f}" fill="#c4d0ec" font-size="11">{subtitle}</text>'
            f'<text x="{position["x"] + 12:.1f}" y="{position["y"] + 54:.1f}" fill="#e3e9f8" font-size="10">{html.escape(meta)}</text>'
            f'</g>'
        )

    summary = (
        f'family={graph["family_id"]}_v{graph["family_version"]} | '
        f'battle={battle_name} | player={player} | turn={turn_number} | '
        f'actions={len(graph["action_candidates"])}'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<defs>'
        f'<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#171d28" />'
        f'<stop offset="100%" stop-color="#0f1218" />'
        f'</linearGradient>'
        f'</defs>'
        f'<rect width="{width}" height="{height}" fill="url(#bg)" />'
        f'<text x="24" y="28" fill="#eef2ff" font-size="20" font-weight="700">Entity Action v1 Preview</text>'
        f'<text x="24" y="50" fill="#9fb0d3" font-size="12">{html.escape(summary)}</text>'
        f'{"".join(edge_lines)}'
        f'{"".join(node_blocks)}'
        f'</svg>'
    )


def render_html(
    *,
    graph: dict[str, Any],
    battle_name: str,
    player: str,
    turn_number: int,
    metadata: dict[str, Any] | None,
) -> str:
    positions = layout_graph(graph)
    heads = graph_heads_from_metadata(metadata or {}, graph)
    metadata_rows = metadata_summary_rows(metadata)

    payload = {
        "battle_name": battle_name,
        "player": player,
        "turn_number": turn_number,
        "graph": graph,
        "positions": positions,
        "heads": heads,
        "metadata": metadata or {},
        "metadata_summary_rows": metadata_rows,
    }
    payload_json = json.dumps(payload)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Entity Action v1 Preview</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: #111318;
      color: #eef2ff;
    }}
    .page {{
      display: grid;
      grid-template-columns: 1180px 360px;
      min-height: 100vh;
    }}
    .canvas-wrap {{
      position: relative;
      padding: 24px;
      background:
        radial-gradient(circle at top, rgba(46, 74, 126, 0.35), transparent 38%),
        linear-gradient(180deg, #161b25 0%, #0f1218 100%);
      border-right: 1px solid rgba(255,255,255,0.08);
    }}
    .title {{
      margin: 0 0 12px 4px;
      font-size: 20px;
      font-weight: 700;
    }}
    .subtitle {{
      margin: 0 0 18px 4px;
      color: #9fb0d3;
      font-size: 13px;
    }}
    #graph {{
      position: relative;
      width: 1130px;
      height: 780px;
      border-radius: 16px;
      background: rgba(255,255,255,0.02);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
      overflow: hidden;
    }}
    #edges {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }}
    .edge {{
      stroke: rgba(173, 197, 255, 0.28);
      stroke-width: 2;
    }}
    .edge.active_matchup {{
      stroke: rgba(255, 164, 122, 0.8);
      stroke-width: 3;
    }}
    .node {{
      position: absolute;
      border-radius: 14px;
      padding: 10px 12px;
      box-sizing: border-box;
      cursor: pointer;
      border: 1px solid rgba(255,255,255,0.14);
      box-shadow: 0 10px 24px rgba(0,0,0,0.24);
      backdrop-filter: blur(4px);
    }}
    .node.global {{
      background: rgba(99, 125, 189, 0.35);
    }}
    .node.pokemon.self {{
      background: rgba(76, 142, 255, 0.2);
    }}
    .node.pokemon.opponent {{
      background: rgba(221, 86, 86, 0.18);
    }}
    .node.move_candidate {{
      background: rgba(90, 182, 120, 0.18);
    }}
    .node.active {{
      border-color: rgba(255, 221, 135, 0.75);
      box-shadow: 0 0 0 1px rgba(255, 221, 135, 0.25), 0 10px 24px rgba(0,0,0,0.24);
    }}
    .node.chosen {{
      outline: 2px solid rgba(133, 255, 181, 0.9);
    }}
    .node-title {{
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .node-subtitle {{
      font-size: 11px;
      color: #c4d0ec;
      margin-bottom: 6px;
    }}
    .node-meta {{
      font-size: 11px;
      color: #e3e9f8;
      line-height: 1.35;
      white-space: pre-line;
    }}
    .panel {{
      padding: 24px 20px;
      overflow: auto;
      background: #0e1117;
    }}
    .card {{
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 14px;
      background: rgba(255,255,255,0.03);
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 14px;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 10px;
      font-size: 12px;
    }}
    .kv .k {{
      color: #9fb0d3;
    }}
    .head-pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(132, 162, 255, 0.16);
      border: 1px solid rgba(132, 162, 255, 0.24);
      margin: 0 8px 8px 0;
      font-size: 12px;
    }}
    .candidate {{
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.03);
      margin-bottom: 8px;
      font-size: 12px;
    }}
    .candidate.chosen {{
      border: 1px solid rgba(133, 255, 181, 0.8);
    }}
    pre {{
      margin: 0;
      font-size: 11px;
      white-space: pre-wrap;
      word-break: break-word;
      color: #dbe5fb;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="canvas-wrap">
      <div class="title">Entity Action v1 Preview</div>
      <div class="subtitle">{battle_name} | player={player} | turn={turn_number}</div>
      <div id="graph">
        <svg id="edges"></svg>
      </div>
    </div>
    <div class="panel">
      <div class="card">
        <h2>Graph Summary</h2>
        <div class="kv" id="summary"></div>
      </div>
      <div class="card">
        <h2>Training Heads</h2>
        <div id="heads"></div>
      </div>
      <div class="card">
        <h2>Training Metadata</h2>
        <div class="kv" id="metadata-summary"></div>
      </div>
      <div class="card">
        <h2>Action Candidates</h2>
        <div id="candidates"></div>
      </div>
      <div class="card">
        <h2>Selected Node</h2>
        <pre id="details">Click a node to inspect its tokens and explicit state.</pre>
      </div>
    </div>
  </div>
  <script>
    const payload = {payload_json};
    const graphEl = document.getElementById("graph");
    const edgesEl = document.getElementById("edges");
    const detailsEl = document.getElementById("details");
    const summaryEl = document.getElementById("summary");
    const headsEl = document.getElementById("heads");
    const metadataSummaryEl = document.getElementById("metadata-summary");
    const candidatesEl = document.getElementById("candidates");

    function addSummary(key, value) {{
      const k = document.createElement("div");
      k.className = "k";
      k.textContent = key;
      const v = document.createElement("div");
      v.textContent = String(value);
      summaryEl.appendChild(k);
      summaryEl.appendChild(v);
    }}

    addSummary("family", payload.graph.family_id + "_v" + payload.graph.family_version);
    addSummary("state_schema", payload.graph.state_schema_version);
    addSummary("pokemon_entities", payload.graph.summary.pokemon_entity_count);
    addSummary("move_candidates", payload.graph.summary.move_candidate_count);
    addSummary("switch_candidates", payload.graph.summary.switch_candidate_count);
    addSummary("edges", payload.graph.summary.edge_count);
    addSummary("chosen_action", payload.graph.summary.chosen_action_token || "(none)");

    for (const head of payload.heads) {{
      const pill = document.createElement("div");
      pill.className = "head-pill";
      pill.textContent = head.head_id + ": " + head.description;
      headsEl.appendChild(pill);
    }}

    for (const [key, value] of payload.metadata_summary_rows || []) {{
      const k = document.createElement("div");
      k.className = "k";
      k.textContent = key;
      const v = document.createElement("div");
      v.textContent = Array.isArray(value) ? value.join(", ") : String(value);
      metadataSummaryEl.appendChild(k);
      metadataSummaryEl.appendChild(v);
    }}

    for (const candidate of payload.graph.action_candidates) {{
      const row = document.createElement("div");
      row.className = "candidate" + (candidate.is_chosen ? " chosen" : "");
      row.textContent = candidate.token + (candidate.is_chosen ? "  <- chosen" : "");
      candidatesEl.appendChild(row);
    }}

    function centerFor(position) {{
      return {{
        x: position.x + position.w / 2,
        y: position.y + position.h / 2,
      }};
    }}

    const byId = Object.fromEntries(payload.graph.entities.map(entity => [entity.id, entity]));

    for (const edge of payload.graph.edges) {{
      const sourcePos = payload.positions[edge.source];
      const targetPos = payload.positions[edge.target];
      if (!sourcePos || !targetPos) continue;
      const sourceCenter = centerFor(sourcePos);
      const targetCenter = centerFor(targetPos);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", sourceCenter.x);
      line.setAttribute("y1", sourceCenter.y);
      line.setAttribute("x2", targetCenter.x);
      line.setAttribute("y2", targetCenter.y);
      line.setAttribute("class", "edge " + edge.edge_type);
      edgesEl.appendChild(line);
    }}

    for (const entity of payload.graph.entities) {{
      const position = payload.positions[entity.id];
      if (!position) continue;
      const node = document.createElement("div");
      const classes = ["node", entity.entity_type];
      if (entity.entity_type === "pokemon") {{
        classes.push(entity.side);
        if (entity.state_features.active > 0.5) classes.push("active");
      }}
      if (entity.entity_type === "move_candidate" && entity.state_features.is_chosen > 0.5) {{
        classes.push("chosen");
      }}
      node.className = classes.join(" ");
      node.style.left = position.x + "px";
      node.style.top = position.y + "px";
      node.style.width = position.w + "px";
      node.style.height = position.h + "px";

      const title = document.createElement("div");
      title.className = "node-title";
      title.textContent = entity.display.title;
      node.appendChild(title);

      const subtitle = document.createElement("div");
      subtitle.className = "node-subtitle";
      subtitle.textContent = entity.display.subtitle;
      node.appendChild(subtitle);

      const meta = document.createElement("div");
      meta.className = "node-meta";
      if (entity.entity_type === "pokemon") {{
        meta.textContent =
          "species=" + entity.token_inputs.species +
          "\\nhp=" + entity.state_features.hp_frac.toFixed(2) +
          " known=" + entity.state_features.hp_known.toFixed(0) +
          "\\nstatus=" + entity.token_inputs.status;
      }} else if (entity.entity_type === "move_candidate") {{
        meta.textContent = "move=" + entity.token_inputs.move;
      }} else {{
        meta.textContent = "weather=" + entity.token_inputs.weather;
      }}
      node.appendChild(meta);

      node.addEventListener("click", () => {{
        detailsEl.textContent = JSON.stringify(entity, null, 2);
      }});
      graphEl.appendChild(node);
    }}
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an Entity Action v1 preview from a logged battle turn.")
    parser.add_argument("battle_path", help="Battle JSON file.")
    parser.add_argument("--player", default="p1", choices=["p1", "p2"])
    parser.add_argument("--turn-number", type=int, default=None, help="Turn number to preview. Defaults to the first available example.")
    parser.add_argument("--metadata-path", default=None, help="Optional training metadata JSON to display current objective heads.")
    parser.add_argument("--output-path", default="out/entity_action_v1_preview.html", help="Destination HTML path.")
    parser.add_argument("--json-output-path", default=None, help="Optional JSON dump for the entity graph.")
    parser.add_argument("--svg-output-path", default=None, help="Optional static SVG path. Defaults to the HTML path with a .svg suffix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    battle_path = Path(args.battle_path).resolve()
    _, example = load_example(
        battle_path,
        player=args.player,
        turn_number=args.turn_number,
    )
    graph = build_entity_action_graph(
        state=example["state"],
        perspective_player=args.player,
        chosen_action=example.get("action"),
        chosen_action_token=example.get("action_token"),
    )

    metadata = None
    if args.metadata_path:
        metadata = json.loads(Path(args.metadata_path).resolve().read_text(encoding="utf-8"))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(
            graph=graph,
            battle_name=battle_path.name,
            player=args.player,
            turn_number=int(example["turn_number"]),
            metadata=metadata,
        ),
        encoding="utf-8",
    )

    json_output_path = Path(args.json_output_path) if args.json_output_path else output_path.with_suffix(".json")
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    svg_output_path = Path(args.svg_output_path) if args.svg_output_path else output_path.with_suffix(".svg")
    svg_output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_output_path.write_text(
        render_svg(
            graph=graph,
            battle_name=battle_path.name,
            player=args.player,
            turn_number=int(example["turn_number"]),
        ),
        encoding="utf-8",
    )

    print(f"wrote_entity_preview_html={output_path.resolve()}")
    print(f"wrote_entity_preview_json={json_output_path.resolve()}")
    print(f"wrote_entity_preview_svg={svg_output_path.resolve()}")


if __name__ == "__main__":
    main()
