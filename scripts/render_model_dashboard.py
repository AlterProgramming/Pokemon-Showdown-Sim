#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def service_models(service: dict[str, Any]) -> str:
    payload = service.get("payload") or {}
    supported = payload.get("supported_model_ids")
    if isinstance(supported, list) and supported:
        return ", ".join(str(item) for item in supported)
    model_id = payload.get("model_id")
    if model_id:
        return str(model_id)
    return "n/a"


def render() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    league_latest = load_json(repo_root / "docs" / "model_league_latest.json")
    league_history = load_json(repo_root / "docs" / "model_league_history.json")
    service_latest = load_json(repo_root / "docs" / "model_service_health_latest.json")

    ranking_rows = "\n".join(
        f"<tr><td>{idx}</td><td><code>{entry['model']}</code></td><td>{entry['elo']}</td>"
        f"<td>{entry.get('wins', 'n/a')}-{entry.get('losses', 'n/a')}</td></tr>"
        for idx, entry in enumerate(league_latest.get("ranking") or [], start=1)
    )

    history_rows = []
    for run in league_history.get("runs") or []:
        ranking = run.get("ranking") or []
        leader = ranking[0] if ranking else {}
        history_rows.append(
            "<tr>"
            f"<td><code>{run.get('run_id', 'n/a')}</code></td>"
            f"<td><span class='pill {'ok' if run.get('status') == 'validated' else 'bad'}'>{run.get('status', 'unknown')}</span></td>"
            f"<td>{run.get('scheduledGames', 0)} scheduled, {run.get('completed', 0)} completed</td>"
            f"<td class='{'ok' if int(run.get('failures', 0)) == 0 else 'bad'}'>{run.get('failures', 0)}</td>"
            f"<td><code>{leader.get('model', 'n/a')}</code> {leader.get('elo', 'n/a')}</td>"
            f"<td>{' '.join(run.get('notes') or [])}</td>"
            "</tr>"
        )
    history_rows_html = "\n".join(history_rows)

    services = service_latest.get("services") or []
    if services:
        service_rows_html = "\n".join(
            "<tr>"
            f"<td><code>{service.get('name', 'n/a')}</code></td>"
            f"<td class='{'ok' if service.get('healthy') else 'bad'}'>{service.get('status', 'unknown')}</td>"
            f"<td>{service.get('latencyMs', 'n/a')}</td>"
            f"<td><code>{service_models(service)}</code></td>"
            f"<td>{service.get('error') or service.get('url', '')}</td>"
            "</tr>"
            for service in services
        )
        service_summary = (
            f"{service_latest.get('summary', {}).get('healthyServices', 0)}/"
            f"{service_latest.get('summary', {}).get('totalServices', 0)} healthy"
        )
    else:
        service_rows_html = (
            "<tr><td colspan='5' class='muted'>No service snapshot recorded yet. "
            "Run <code>python scripts/collect_model_service_health.py</code>.</td></tr>"
        )
        service_summary = "no snapshot"

    reliability_runs = league_history.get("runs") or []
    failure_before = next((run.get("failures", 0) for run in reliability_runs if run.get("status") != "validated"), "n/a")
    failure_after = next((run.get("failures", 0) for run in reversed(reliability_runs) if run.get("status") == "validated"), "n/a")
    champion = (league_latest.get("ranking") or [{}])[0]

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Model League Dashboard</title>
    <style>
      :root {{
        --bg: #f2efe8;
        --panel: #fffaf0;
        --ink: #1e2a26;
        --muted: #5b6a63;
        --accent: #1d6b57;
        --warning: #a46012;
        --danger: #a0342a;
        --border: #d9d1c4;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Menlo, Monaco, Consolas, monospace;
        background: linear-gradient(180deg, #ece7dc 0%, var(--bg) 100%);
        color: var(--ink);
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 24px;
      }}
      h1, h2, h3 {{ margin: 0 0 12px; }}
      p {{ margin: 0; line-height: 1.5; }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 16px;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04);
      }}
      .metric {{
        font-size: 28px;
        font-weight: 700;
        margin-top: 6px;
      }}
      .muted {{ color: var(--muted); }}
      .ok {{ color: var(--accent); }}
      .warn {{ color: var(--warning); }}
      .bad {{ color: var(--danger); }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        text-align: left;
        padding: 10px 8px;
        border-bottom: 1px solid var(--border);
        vertical-align: top;
      }}
      th {{ color: var(--muted); font-weight: 700; }}
      .section {{ margin-top: 24px; }}
      .pill {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        font-size: 12px;
      }}
      .small {{ font-size: 13px; }}
      code {{ font-family: inherit; }}
    </style>
  </head>
  <body>
    <main>
      <h1>Model League Dashboard</h1>
      <p class="muted">Canonical league reporting for standings, stability, and service health.</p>

      <div class="grid">
        <section class="panel">
          <h2>Champion</h2>
          <div class="metric">{champion.get('model', 'n/a')}</div>
          <p class="muted">Current validated leader</p>
        </section>
        <section class="panel">
          <h2>Current Elo</h2>
          <div class="metric">{champion.get('elo', 'n/a')}</div>
          <p class="muted">From the latest validated mixed-model league</p>
        </section>
        <section class="panel">
          <h2>Latest Run</h2>
          <div class="metric">{league_latest.get('completed', 0)}</div>
          <p class="muted">games, {league_latest.get('failures', 0)} failures, concurrency {league_latest.get('concurrency', 0)}</p>
        </section>
        <section class="panel">
          <h2>Reliability Shift</h2>
          <div class="metric ok">{failure_before} → {failure_after}</div>
          <p class="muted">failures from the pre-stabilization long run to the current validated baseline</p>
        </section>
      </div>

      <section class="panel section">
        <h2>Current Ranking</h2>
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th>Elo</th>
              <th>Record</th>
            </tr>
          </thead>
          <tbody>
            {ranking_rows}
          </tbody>
        </table>
      </section>

      <section class="panel section">
        <h2>Service Health</h2>
        <p class="muted">Latest snapshot: {service_latest.get('generated_at', 'n/a')} · {service_summary}</p>
        <table>
          <thead>
            <tr>
              <th>Service</th>
              <th>Status</th>
              <th>Latency ms</th>
              <th>Models</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {service_rows_html}
          </tbody>
        </table>
      </section>

      <section class="panel section">
        <h2>Run History</h2>
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Shape</th>
              <th>Failures</th>
              <th>Leader</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {history_rows_html}
          </tbody>
        </table>
      </section>

      <section class="grid section">
        <section class="panel">
          <h2>Storage Contract</h2>
          <p class="small">Latest league pointer: <code>docs/model_league_latest.json</code></p>
          <p class="small">League history: <code>docs/model_league_history.json</code></p>
          <p class="small">Latest service snapshot: <code>docs/model_service_health_latest.json</code></p>
          <p class="small">Service history: <code>docs/model_service_health_history.json</code></p>
        </section>
        <section class="panel">
          <h2>Operational Use</h2>
          <p class="small">Record both clean runs and failed runs. Reliability regressions are part of the benchmark signal.</p>
        </section>
        <section class="panel">
          <h2>Maintenance</h2>
          <p class="small">After new league data or service data arrives, regenerate this file with <code>python scripts/render_model_dashboard.py</code>.</p>
        </section>
      </section>
    </main>
  </body>
</html>
"""


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "docs" / "MODEL_LEAGUE_DASHBOARD.html"
    output_path.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
