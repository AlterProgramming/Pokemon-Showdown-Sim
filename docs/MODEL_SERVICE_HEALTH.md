# Model Service Health

This file is the landing page for HTTP service health snapshots that support the league system.

Use these paths:

- Latest service snapshot: [model_service_health_latest.json](/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Sim/docs/model_service_health_latest.json)
- Service snapshot history: [model_service_health_history.json](/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Sim/docs/model_service_health_history.json)
- Combined dashboard: [MODEL_LEAGUE_DASHBOARD.html](/Users/AI-CCORE/alter-programming/Pokemon-Showdown-Sim/docs/MODEL_LEAGUE_DASHBOARD.html)

## Scope

These snapshots are for HTTP-serving points such as:

- the shared vector-model server on `127.0.0.1:5000`
- the entity benchmark server on `127.0.0.1:5001`

They are intended to answer:

- which services were up at run time
- how fast health checks responded
- whether a degraded serving state correlates with league failures

## Update Flow

Collect a fresh snapshot:

```bash
python scripts/collect_model_service_health.py
```

Regenerate the dashboard after service or league data changes:

```bash
python scripts/render_model_dashboard.py
```
