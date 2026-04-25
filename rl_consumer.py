#!/usr/bin/env python3
"""RL fine-tuning consumer daemon (Phase 1).

Polls the model-league training/pending directory every 30s, launches
rl_finetune.py on each new job, spawns a serving process on the next free
port >= 5004, and POSTs a completion payload to the daemon webhook on
127.0.0.1:3410.

Safety properties:
  * Jobs present at startup are snapshotted and skipped (stale from earlier
    sessions per operator guidance).
  * Jobs with createdAt older than 24h are skipped as "stale".
  * Inner try/except around each job so one bad job doesn't poison the batch.
  * Outer try/except around the poll loop so the daemon never dies on a
    transient error.
  * On training-subprocess failure: log stderr, leave pending file in place
    for retry, do NOT POST completion.
  * On success: move pending -> processed/ (webhook handler ultimately deletes).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AGENT_REPO = Path(__file__).resolve().parent
ROOT = AGENT_REPO.parent
SIM_REPO = Path(os.environ.get("POKEMON_SHOWDOWN_REPO", str(ROOT / "pokemon-showdown-model-feature")))
LEAGUE_ROOT = SIM_REPO / "databases" / "model-league"
PENDING_DIR = LEAGUE_ROOT / "training" / "pending"
PROCESSED_DIR = PENDING_DIR.parent / "processed"
SECRET_FILE = LEAGUE_ROOT / "webhook-secret.txt"

VENV_PY = AGENT_REPO / ".venv" / "bin" / "python3"
FINETUNE_SCRIPT = AGENT_REPO / "rl_finetune.py"
SERVE_SCRIPT = AGENT_REPO / "serve_entity_model_benchmark.py"

WEBHOOK_URL = "http://127.0.0.1:3410/training/completed"
POLL_INTERVAL_SEC = 30
PORT_MIN = 5004
PORT_MAX = 5020
SERVE_HEALTH_TIMEOUT_SEC = 60
STALE_AGE_SECONDS = 24 * 60 * 60  # 24h

LOG_PATH = Path("/tmp/rl_consumer.log")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("rl_consumer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_iso_utc(s: str) -> dt.datetime:
    # Accepts "...Z" and "...+00:00" forms.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def _read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_sim_path(rel_or_abs: str) -> Path:
    """Job JSON paths like 'databases/model-league/...' are relative to the
    simulator repo root."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return SIM_REPO / p


def _read_secret() -> str | None:
    try:
        if SECRET_FILE.exists():
            return SECRET_FILE.read_text(encoding="utf-8").strip()
    except Exception as exc:  # pragma: no cover
        log.warning("failed to read webhook-secret.txt: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Serving launcher
# ---------------------------------------------------------------------------

def _port_free(port: int) -> bool:
    """True if nothing is listening on TCP:port (via lsof)."""
    try:
        result = subprocess.run(
            ["lsof", "-iTCP:%d" % port, "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        log.warning("lsof probe failed for port %d: %s", port, exc)
        return False
    return not result.stdout.strip()


def _find_free_port() -> int | None:
    for port in range(PORT_MIN, PORT_MAX + 1):
        if _port_free(port):
            return port
    return None


def _metadata_path_for(model_id: str) -> Path | None:
    """Prefer .local.json; fall back to .json with a warning."""
    base_dir = AGENT_REPO / "artifacts" / model_id
    local = base_dir / f"training_metadata_{model_id}.local.json"
    plain = base_dir / f"training_metadata_{model_id}.json"
    if local.exists():
        return local
    if plain.exists():
        log.warning("%s missing; falling back to non-local metadata %s",
                    local, plain)
        return plain
    log.error("no metadata found for model %s (looked for %s and %s)",
              model_id, local, plain)
    return None


def launch_serving(model_id: str, artifact_dir: Path) -> tuple[int, int]:
    """Spawn serve_entity_model_benchmark.py on the next free port >= 5004.

    Returns (port, pid). Waits up to SERVE_HEALTH_TIMEOUT_SEC for /health.
    Raises RuntimeError on failure to find a port, start the process, or
    see /health respond.
    """
    port = _find_free_port()
    if port is None:
        raise RuntimeError(
            f"no free port in [{PORT_MIN},{PORT_MAX}] for serving {model_id}"
        )

    meta_path = _metadata_path_for(model_id)
    if meta_path is None:
        raise RuntimeError(f"metadata not found for {model_id}")

    log_path = Path(f"/tmp/flask_server_{port}.log")
    log_fh = open(log_path, "ab")
    cmd = [
        str(VENV_PY),
        str(SERVE_SCRIPT),
        "--metadata-path",
        str(meta_path),
        "--port",
        str(port),
    ]
    log.info("launching serving: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(AGENT_REPO),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + SERVE_HEALTH_TIMEOUT_SEC
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"serving process exited early (rc={proc.returncode}); "
                f"see {log_path}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    log.info(
                        "serving healthy on port %d (pid %d)", port, proc.pid
                    )
                    return port, proc.pid
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError(
        f"serving /health did not respond on {health_url} within "
        f"{SERVE_HEALTH_TIMEOUT_SEC}s; see {log_path}"
    )


# ---------------------------------------------------------------------------
# Completion POST
# ---------------------------------------------------------------------------

def post_completion(
    job_id: str,
    new_model_id: str,
    port: int,
    parent: str,
    secret: str | None,
) -> None:
    """POST the completion payload. Expects 2xx; raises on error."""
    payload = {
        "jobId": job_id,
        "parentCheckpointId": parent,
        "newModelId": new_model_id,
        "name": new_model_id,
        "endpoint": f"http://127.0.0.1:{port}/predict",
        "modelProfile": "joint-policy-value",
        "allowVoluntarySwitches": True,
        "lineageId": parent,
        "parentModelId": parent,
        "metadata": {"training_regime": "reinforce_v1_advantage"},
        "activate": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    if secret:
        req.add_header("x-model-league-secret", secret)
    else:
        log.warning(
            "webhook-secret.txt missing; POSTing %s without auth header",
            WEBHOOK_URL,
        )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if 200 <= resp.status < 300:
                log.info(
                    "completion POSTed for job %s (status %d)",
                    job_id, resp.status,
                )
                return
            raise RuntimeError(f"webhook returned HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"webhook HTTPError {exc.code}: {exc.read()!r}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"webhook URLError: {exc.reason}") from exc


# ---------------------------------------------------------------------------
# Training subprocess
# ---------------------------------------------------------------------------

def _run_finetune(
    base_model: str,
    manifest_path: Path,
    output_model: str,
) -> subprocess.CompletedProcess:
    cmd = [
        str(VENV_PY),
        str(FINETUNE_SCRIPT),
        "--base-model",
        base_model,
        "--job-manifest",
        str(manifest_path),
        "--output-model",
        output_model,
        "--use-sharded-adapter",
    ]
    log.info("training started: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(AGENT_REPO),
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Per-job processing
# ---------------------------------------------------------------------------

def _is_stale(job: dict[str, Any]) -> bool:
    created = job.get("createdAt")
    if not created:
        return False
    try:
        age = (_now_utc() - _parse_iso_utc(created)).total_seconds()
    except Exception:
        return False
    return age >= STALE_AGE_SECONDS


def _new_model_id(base: str) -> str:
    ts = _now_utc().strftime("%Y%m%d_%H%M")
    return f"{base}_ft_{ts}"


def process_job(pending_file: Path, secret: str | None) -> None:
    """Process a single pending job file. Errors are logged but not raised."""
    log.info("job found: %s", pending_file.name)
    try:
        job = _read_json(pending_file)
    except Exception as exc:
        log.error("failed to read %s: %s", pending_file, exc)
        return

    job_id = job.get("jobId") or pending_file.stem
    base_model = job.get("modelCheckpointId")
    manifest_rel = job.get("manifestPath")
    if not base_model or not manifest_rel:
        log.error(
            "job %s missing modelCheckpointId or manifestPath; skipping",
            job_id,
        )
        return

    if _is_stale(job):
        log.info(
            "stale, skipping: job %s createdAt=%s (older than 24h)",
            job_id, job.get("createdAt"),
        )
        return

    manifest_path = _resolve_sim_path(manifest_rel)
    if not manifest_path.exists():
        log.error(
            "job %s manifest not found at %s; skipping",
            job_id, manifest_path,
        )
        return

    # Sanity: manifest should have buffer.exampleFiles[]
    try:
        manifest = _read_json(manifest_path)
        example_files = (
            manifest.get("buffer", {}).get("exampleFiles")
            or manifest.get("exampleFiles")
            or []
        )
        log.info(
            "job %s manifest ok: %d example files",
            job_id, len(example_files),
        )
    except Exception as exc:
        log.error("job %s manifest read failed: %s", job_id, exc)
        return

    output_model = _new_model_id(base_model)
    log.info(
        "training started: job=%s base=%s output=%s",
        job_id, base_model, output_model,
    )
    try:
        result = _run_finetune(base_model, manifest_path, output_model)
    except Exception as exc:
        log.error("job %s training subprocess raised: %s", job_id, exc)
        return

    if result.returncode != 0:
        log.error(
            "job %s training finished with rc=%d; stderr (last 4KB): %s",
            job_id, result.returncode, (result.stderr or "")[-4096:],
        )
        return
    log.info("training finished: job=%s rc=0 output=%s", job_id, output_model)

    # Spawn serving.
    try:
        port, pid = launch_serving(
            output_model, AGENT_REPO / "artifacts" / output_model,
        )
    except Exception as exc:
        log.error("job %s serving launch failed: %s", job_id, exc)
        return
    log.info("serving launched: job=%s port=%d pid=%d", job_id, port, pid)

    # POST completion.
    try:
        post_completion(
            job_id=job_id,
            new_model_id=output_model,
            port=port,
            parent=base_model,
            secret=secret,
        )
    except Exception as exc:
        log.error("job %s completion POST failed: %s", job_id, exc)
        return

    # Move pending -> processed.
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = PROCESSED_DIR / pending_file.name
        shutil.move(str(pending_file), str(dest))
        log.info("moved pending -> processed: %s", dest)
    except Exception as exc:
        log.error(
            "job %s: completion posted but failed to move pending file: %s",
            job_id, exc,
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("rl_consumer starting (pid=%d)", os.getpid())
    log.info("PENDING_DIR=%s", PENDING_DIR)
    log.info("PROCESSED_DIR=%s", PROCESSED_DIR)

    # Snapshot existing pending files at startup — per operator guidance,
    # these are stale from earlier sessions and must not be processed.
    if PENDING_DIR.exists():
        startup_snapshot: set[str] = {
            p.name for p in PENDING_DIR.glob("*.json")
        }
    else:
        startup_snapshot = set()
    log.info(
        "startup snapshot: %d existing pending file(s) will be skipped: %s",
        len(startup_snapshot), sorted(startup_snapshot),
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            if not PENDING_DIR.exists():
                log.warning(
                    "pending dir does not exist: %s (will retry)",
                    PENDING_DIR,
                )
            else:
                files = sorted(PENDING_DIR.glob("*.json"))
                log.info(
                    "polling pending: %d file(s) present",
                    len(files),
                )
                secret = _read_secret()
                for f in files:
                    if f.name in startup_snapshot:
                        log.info(
                            "startup-snapshot skip: %s (stale from earlier "
                            "session)", f.name,
                        )
                        continue
                    try:
                        process_job(f, secret)
                    except Exception as exc:
                        # Inner guard: one bad job must not poison the batch.
                        log.exception(
                            "unhandled error on job %s: %s", f.name, exc,
                        )
        except Exception as exc:
            # Outer guard: never let the daemon die.
            log.exception("poll loop error (continuing): %s", exc)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("rl_consumer stopped by KeyboardInterrupt")
        sys.exit(0)
