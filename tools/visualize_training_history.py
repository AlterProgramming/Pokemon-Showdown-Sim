from __future__ import annotations

"""Render saved Keras history artifacts into a local HTML dashboard.

This viewer exists because the project puts a lot of weight on observability.
The raw JSON history is fine for tooling, but this page gives a faster read on:
    - whether training stabilized
    - whether validation tracks training sensibly
    - which heads are improving or stalling
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ModelRegistry import resolve_artifact_path


def load_history(args: argparse.Namespace) -> tuple[dict[str, list[float]], str]:
    """Load a training-history payload either directly or through metadata indirection."""
    if args.history_path:
        history_path = Path(args.history_path).resolve()
        return json.loads(history_path.read_text(encoding="utf-8")), history_path.name

    if not args.metadata_path:
        raise SystemExit("Provide either --history-path or --metadata-path.")

    metadata_path = Path(args.metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    history_ref = metadata.get("training_history_path")
    if not history_ref:
        raise SystemExit(f"No training_history_path found in {metadata_path}.")
    repo_path = metadata_path.parent.parent.resolve()
    history_path = resolve_artifact_path(repo_path, metadata_path, str(history_ref))
    return json.loads(history_path.read_text(encoding="utf-8")), history_path.name


def render_html(history: dict[str, list[float]], title: str) -> str:
    """Render a compact self-contained training dashboard."""
    payload = json.dumps({"title": title, "history": history})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Training History Viewer</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: #0f1218;
      color: #eef2ff;
    }}
    .page {{
      display: grid;
      grid-template-columns: 1fr 360px;
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
    .charts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(420px, 1fr));
      gap: 16px;
    }}
    .chart-card, .panel-card {{
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      padding: 14px;
      box-sizing: border-box;
    }}
    .chart-title {{
      margin: 0 0 10px 0;
      font-size: 14px;
      font-weight: 700;
    }}
    svg {{
      width: 100%;
      height: 220px;
      background: rgba(255,255,255,0.02);
      border-radius: 12px;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    }}
    .panel {{
      padding: 24px 20px;
      border-left: 1px solid rgba(255,255,255,0.08);
      background: #0d1016;
      overflow: auto;
    }}
    .metric-row {{
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      font-size: 12px;
    }}
    .metric-name {{
      color: #9fb0d3;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 11px;
      color: #cfd8ef;
      margin-bottom: 10px;
    }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: middle;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="main">
      <h1>Training History Viewer</h1>
      <div class="subtitle">{title}</div>
      <div class="charts" id="charts"></div>
    </div>
    <div class="panel">
      <div class="panel-card">
        <div class="chart-title">Metric Summary</div>
        <div id="summary"></div>
      </div>
    </div>
  </div>
  <script>
    const payload = {payload};
    const chartsEl = document.getElementById("charts");
    const summaryEl = document.getElementById("summary");
    const colors = ["#7cb8ff", "#8ef0a4", "#ffb57c", "#ff8db4", "#d5a6ff", "#ffe084"];

    function linePath(values, width, height, padding) {{
      const finite = values.filter(v => Number.isFinite(v));
      const minV = Math.min(...finite);
      const maxV = Math.max(...finite);
      const span = Math.max(maxV - minV, 1e-6);
      return values.map((v, idx) => {{
        const x = padding + (idx * (width - padding * 2) / Math.max(values.length - 1, 1));
        const y = height - padding - ((v - minV) / span) * (height - padding * 2);
        return (idx === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      }}).join(" ");
    }}

    function metricGroupName(metricName) {{
      // Group val_* and train metrics onto the same panel so trend comparison is immediate.
      if (metricName.startsWith("val_")) return metricName.slice(4);
      return metricName;
    }}

    const grouped = new Map();
    for (const [name, values] of Object.entries(payload.history)) {{
      const group = metricGroupName(name);
      if (!grouped.has(group)) grouped.set(group, []);
      grouped.get(group).push([name, values]);
    }}

    let chartIndex = 0;
    for (const [groupName, entries] of grouped.entries()) {{
      const card = document.createElement("div");
      card.className = "chart-card";

      const title = document.createElement("div");
      title.className = "chart-title";
      title.textContent = groupName;
      card.appendChild(title);

      const legend = document.createElement("div");
      legend.className = "legend";
      card.appendChild(legend);

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      const width = 460;
      const height = 220;
      const padding = 22;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);

      const grid = document.createElementNS("http://www.w3.org/2000/svg", "g");
      for (let i = 0; i < 4; i++) {{
        const y = padding + i * ((height - padding * 2) / 3);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", padding);
        line.setAttribute("y1", y);
        line.setAttribute("x2", width - padding);
        line.setAttribute("y2", y);
        line.setAttribute("stroke", "rgba(255,255,255,0.10)");
        line.setAttribute("stroke-width", "1");
        grid.appendChild(line);
      }}
      svg.appendChild(grid);

      entries.forEach(([name, values], idx) => {{
        const color = colors[(chartIndex + idx) % colors.length];
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", linePath(values, width, height, padding));
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", color);
        path.setAttribute("stroke-width", "3");
        svg.appendChild(path);

        const item = document.createElement("div");
        item.innerHTML = `<span class="swatch" style="background:${{color}}"></span>${{name}}`;
        legend.appendChild(item);
      }});

      card.appendChild(svg);
      chartsEl.appendChild(card);
      chartIndex += entries.length;
    }}

    const sortedMetricNames = Object.keys(payload.history).sort();
    for (const name of sortedMetricNames) {{
      const values = payload.history[name] || [];
      if (!values.length) continue;
      const row = document.createElement("div");
      row.className = "metric-row";
      const first = values[0];
      const last = values[values.length - 1];
      const best = name.toLowerCase().includes("loss") || name.toLowerCase().includes("mae") || name.toLowerCase().includes("brier")
        ? Math.min(...values)
        : Math.max(...values);
      row.innerHTML = `
        <div class="metric-name">${{name}}</div>
        <div>epochs=${{values.length}} | first=${{first.toFixed(4)}} | last=${{last.toFixed(4)}} | best=${{best.toFixed(4)}}</div>
      `;
      summaryEl.appendChild(row);
    }}
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Define the CLI for the training-history viewer."""
    parser = argparse.ArgumentParser(description="Render an HTML view of a saved Keras training history.")
    parser.add_argument("--history-path", default=None, help="Path to a raw history JSON file.")
    parser.add_argument("--metadata-path", default=None, help="Training metadata JSON that points to the history artifact.")
    parser.add_argument("--output-path", default="out/training_history_view.html", help="Destination HTML path.")
    return parser.parse_args()


def main() -> None:
    """Load the history artifact and write the HTML viewer."""
    args = parse_args()
    history, title = load_history(args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(history, title), encoding="utf-8")
    print(f"wrote_training_history_html={output_path.resolve()}")


if __name__ == "__main__":
    main()
