#!/usr/bin/env python3
"""Run deterministic Smart Locker demo scenarios against a live backend."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Any, Sequence

import httpx


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from config import RAPID_ACCESS_LIMIT
except Exception:
    RAPID_ACCESS_LIMIT = 5


class DemoError(RuntimeError):
    """Raised when a demo scenario does not produce the expected result."""


@dataclass
class ScenarioState:
    created_user_ids: list[int] = field(default_factory=list)


@dataclass
class DemoContext:
    client: httpx.Client
    base_url: str
    locker_id: str
    base_seed: int | None
    verbose: bool
    cleanup_enabled: bool
    session_tag: str


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Smart Locker backend demo scenarios against a live server."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL where the backend is running (default: %(default)s).",
    )
    parser.add_argument(
        "--scenario",
        choices=("grant", "deny", "anomaly", "all"),
        default="all",
        help="Which demo scenario to run (default: %(default)s).",
    )
    parser.add_argument(
        "--locker-id",
        default="L001",
        help="Locker ID to use in requests (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible demo embeddings.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full JSON payloads and responses during the demo.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary users after the run for debugging.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: %(default)s).",
    )
    return parser


def build_rng(base_seed: int | None, salt: str) -> Random:
    seed_material = f"{base_seed}:{salt}" if base_seed is not None else secrets.token_hex(16)
    return Random(seed_material)


def generate_embedding(base_seed: int | None, salt: str, dim: int = 128) -> list[float]:
    rng = build_rng(base_seed, salt)
    return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(dim)]


def make_temp_name(prefix: str, session_tag: str) -> str:
    return f"{prefix}-{session_tag}"


def log_step(message: str) -> None:
    print(message)


def print_payload(label: str, payload: Any, enabled: bool) -> None:
    if not enabled or payload is None:
        return
    print(f"\n[{label} PAYLOAD]")
    print(json.dumps(payload, indent=2))


def print_response(label: str, response: httpx.Response, enabled: bool) -> None:
    if not enabled:
        return
    print(f"\n[{label} RESPONSE]")
    print(f"{response.request.method} {response.request.url}")
    print(f"Status: {response.status_code}")
    print(json.dumps(safe_json(response), indent=2))


def request_json(
    ctx: DemoContext,
    method: str,
    path: str,
    label: str,
    *,
    expected_status: int | tuple[int, ...] = 200,
    payload: Any | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    request_preview = payload if payload is not None else params
    print_payload(label, request_preview, ctx.verbose)
    response = ctx.client.request(
        method,
        f"{ctx.base_url}{path}",
        json=payload,
        params=params,
    )
    print_response(label, response, ctx.verbose)

    expected = (
        expected_status
        if isinstance(expected_status, tuple)
        else (expected_status,)
    )
    if response.status_code not in expected:
        raise DemoError(
            f"{label} failed with status {response.status_code}. "
            f"Expected {expected}. Body: {safe_json(response)}"
        )

    return safe_json(response)


def latest_id(items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    return max(int(item["id"]) for item in items)


def fetch_latest_log_id(ctx: DemoContext) -> int:
    body = request_json(
        ctx,
        "GET",
        "/api/logs",
        "BASELINE LOGS",
        params={"limit": 1, "offset": 0},
    )
    return latest_id(body.get("data", []))


def fetch_latest_alert_id(ctx: DemoContext) -> int:
    body = request_json(
        ctx,
        "GET",
        "/api/alerts",
        "BASELINE ALERTS",
        params={"limit": 1, "offset": 0},
    )
    return latest_id(body.get("data", []))


def fetch_new_logs(ctx: DemoContext, baseline_log_id: int) -> list[dict[str, Any]]:
    body = request_json(
        ctx,
        "GET",
        "/api/logs",
        "VERIFY LOGS",
        params={"limit": 100, "offset": 0},
    )
    return [row for row in body.get("data", []) if int(row["id"]) > baseline_log_id]


def fetch_new_alerts(ctx: DemoContext, baseline_alert_id: int) -> list[dict[str, Any]]:
    body = request_json(
        ctx,
        "GET",
        "/api/alerts",
        "VERIFY ALERTS",
        params={"limit": 100, "offset": 0},
    )
    return [row for row in body.get("data", []) if int(row["id"]) > baseline_alert_id]


def register_temp_user(ctx: DemoContext, name: str, embedding: list[float]) -> int:
    body = request_json(
        ctx,
        "POST",
        "/api/users",
        "REGISTER USER",
        expected_status=201,
        payload={"name": name, "face_embedding": embedding},
    )
    user_id = body.get("id")
    if not isinstance(user_id, int):
        raise DemoError(f"Registration response missing integer id: {body}")
    return user_id


def delete_user(ctx: DemoContext, user_id: int) -> None:
    body = request_json(
        ctx,
        "DELETE",
        f"/api/users/{user_id}",
        "DELETE USER",
        expected_status=(200, 204, 404),
    )
    if not ctx.verbose and body:
        log_step(f"[cleanup] user_id={user_id}")


def recognize(
    ctx: DemoContext,
    *,
    embedding: list[float],
    confidence_score: float,
    label: str,
) -> dict[str, Any]:
    payload = {
        "face_embedding": embedding,
        "confidence_score": confidence_score,
        "locker_id": ctx.locker_id,
        "timestamp": utc_now_iso(),
    }
    body = request_json(
        ctx,
        "POST",
        "/api/auth/recognize",
        label,
        payload=payload,
    )
    if not isinstance(body, dict):
        raise DemoError(f"{label} returned unexpected body: {body}")
    return body


def verify_health(ctx: DemoContext) -> None:
    body = request_json(ctx, "GET", "/api/health", "HEALTH CHECK")
    status = body.get("status")
    if status != "ok":
        raise DemoError(f"Health check returned unexpected payload: {body}")
    if not ctx.verbose:
        log_step("[ok] backend health check passed")


def run_grant_scenario(ctx: DemoContext, state: ScenarioState) -> None:
    baseline_log_id = fetch_latest_log_id(ctx)
    embedding = generate_embedding(ctx.base_seed, "grant")
    name = make_temp_name("DemoGrant", ctx.session_tag)
    created_user_id = register_temp_user(ctx, name, embedding)
    state.created_user_ids.append(created_user_id)

    body = recognize(
        ctx,
        embedding=embedding,
        confidence_score=0.96,
        label="GRANT RECOGNIZE",
    )

    if body.get("access_granted") is not True:
        raise DemoError(f"Grant scenario expected access_granted=true, got: {body}")
    if body.get("locker_action") != "OPEN":
        raise DemoError(f"Grant scenario expected locker_action=OPEN, got: {body}")
    if body.get("user_id") != created_user_id:
        raise DemoError(
            "Grant scenario matched an unexpected user id. "
            f"Expected {created_user_id}, got {body.get('user_id')}."
        )

    new_logs = fetch_new_logs(ctx, baseline_log_id)
    matching_log = next((row for row in new_logs if row["id"] == body.get("log_id")), None)
    if not matching_log:
        raise DemoError("Grant scenario did not produce a new access log.")
    if matching_log.get("result") != "SUCCESS":
        raise DemoError(f"Grant scenario expected SUCCESS log, got: {matching_log}")

    if not ctx.verbose:
        log_step(f"[pass] grant: user_id={created_user_id}, log_id={body['log_id']}")


def run_deny_scenario(ctx: DemoContext) -> None:
    baseline_log_id = fetch_latest_log_id(ctx)
    embedding = generate_embedding(ctx.base_seed, "deny")

    body = recognize(
        ctx,
        embedding=embedding,
        confidence_score=0.31,
        label="DENY RECOGNIZE",
    )

    if body.get("access_granted") is not False:
        raise DemoError(f"Deny scenario expected access_granted=false, got: {body}")
    if body.get("locker_action") != "ACCESS_DENIED":
        raise DemoError(
            f"Deny scenario expected locker_action=ACCESS_DENIED, got: {body}"
        )

    new_logs = fetch_new_logs(ctx, baseline_log_id)
    matching_log = next((row for row in new_logs if row["id"] == body.get("log_id")), None)
    if not matching_log:
        raise DemoError("Deny scenario did not produce a new access log.")
    if matching_log.get("result") != "FAILURE":
        raise DemoError(f"Deny scenario expected FAILURE log, got: {matching_log}")

    if not ctx.verbose:
        log_step(f"[pass] deny: log_id={body['log_id']}")


def run_anomaly_scenario(ctx: DemoContext, state: ScenarioState) -> None:
    baseline_log_id = fetch_latest_log_id(ctx)
    baseline_alert_id = fetch_latest_alert_id(ctx)
    embedding = generate_embedding(ctx.base_seed, "anomaly")
    name = make_temp_name("DemoAnomaly", ctx.session_tag)
    created_user_id = register_temp_user(ctx, name, embedding)
    state.created_user_ids.append(created_user_id)

    last_body: dict[str, Any] | None = None
    for attempt in range(1, RAPID_ACCESS_LIMIT + 1):
        last_body = recognize(
            ctx,
            embedding=embedding,
            confidence_score=0.99,
            label=f"ANOMALY RECOGNIZE {attempt}/{RAPID_ACCESS_LIMIT}",
        )
        if last_body.get("access_granted") is not True:
            raise DemoError(
                f"Anomaly scenario expected successful recognize on attempt {attempt}: {last_body}"
            )

    new_logs = fetch_new_logs(ctx, baseline_log_id)
    if len(new_logs) < RAPID_ACCESS_LIMIT:
        raise DemoError(
            "Anomaly scenario did not create the expected number of new success logs."
        )

    new_alerts = fetch_new_alerts(ctx, baseline_alert_id)
    rapid_access_alert = next(
        (row for row in new_alerts if row.get("type") == "RAPID_ACCESS"),
        None,
    )
    if rapid_access_alert is None:
        raise DemoError(
            f"Anomaly scenario expected a RAPID_ACCESS alert, got: {new_alerts}"
        )

    if not ctx.verbose and last_body is not None:
        log_step(
            f"[pass] anomaly: alert_id={rapid_access_alert['id']}, "
            f"last_log_id={last_body['log_id']}"
        )


def cleanup_users(ctx: DemoContext, state: ScenarioState) -> None:
    if not ctx.cleanup_enabled:
        if state.created_user_ids:
            log_step(f"[info] no-cleanup enabled; kept users {state.created_user_ids}")
        return

    for user_id in state.created_user_ids:
        delete_user(ctx, user_id)


def run_selected_scenarios(ctx: DemoContext, scenario: str) -> None:
    state = ScenarioState()
    try:
        verify_health(ctx)
        if scenario in ("grant", "all"):
            run_grant_scenario(ctx, state)
        if scenario in ("deny", "all"):
            run_deny_scenario(ctx)
        if scenario in ("anomaly", "all"):
            run_anomaly_scenario(ctx, state)
    finally:
        cleanup_users(ctx, state)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_url = args.base_url.rstrip("/")
    session_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        with httpx.Client(timeout=args.timeout) as client:
            ctx = DemoContext(
                client=client,
                base_url=base_url,
                locker_id=args.locker_id,
                base_seed=args.seed,
                verbose=args.verbose,
                cleanup_enabled=not args.no_cleanup,
                session_tag=session_tag,
            )
            run_selected_scenarios(ctx, args.scenario)
    except (httpx.HTTPError, DemoError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected demo failure: {exc}", file=sys.stderr)
        return 1

    if not args.verbose:
        log_step("[done] demo completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
