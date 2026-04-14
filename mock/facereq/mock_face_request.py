#!/usr/bin/env python3
"""
Mock client for Smart Locker backend contract testing.

Use this script to test /api/auth/recognize without the face module.
It supports two scenarios:
1) deny  -> send an unknown/random embedding
2) grant -> register a temporary user with an embedding, then recognize it
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from typing import Any

import httpx


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def generate_embedding(dim: int = 128, seed: int | None = None) -> list[float]:
    """Generate a deterministic mock embedding when seed is provided."""
    rng = random.Random(seed)
    return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(dim)]


def safe_json(response: httpx.Response) -> Any:
    """Return JSON body if available, otherwise return raw text."""
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def print_response(label: str, response: httpx.Response) -> None:
    """Pretty-print response status and payload."""
    print(f"\n[{label}] {response.request.method} {response.request.url}")
    print(f"Status: {response.status_code}")
    print(json.dumps(safe_json(response), indent=2))


def register_user(
    client: httpx.Client,
    base_url: str,
    name: str,
    embedding: list[float],
) -> int:
    """Register a user and return user ID."""
    response = client.post(
        f"{base_url}/api/users/register",
        json={"name": name, "face_embedding": embedding},
    )
    print_response("REGISTER", response)
    response.raise_for_status()
    body = response.json()
    user_id = body.get("id")
    if not isinstance(user_id, int):
        raise ValueError("Registration succeeded but response did not contain integer 'id'.")
    return user_id


def recognize(
    client: httpx.Client,
    base_url: str,
    embedding: list[float],
    confidence: float,
    locker_id: str,
    print_payload: bool,
) -> httpx.Response:
    """Send recognize request."""
    payload = {
        "face_embedding": embedding,
        "confidence_score": confidence,
        "locker_id": locker_id,
        "timestamp": utc_now_iso(),
    }
    if print_payload:
        print("\n[PAYLOAD] /api/auth/recognize")
        print(json.dumps(payload, indent=2))

    response = client.post(f"{base_url}/api/auth/recognize", json=payload)
    print_response("RECOGNIZE", response)
    return response


def delete_user(client: httpx.Client, base_url: str, user_id: int) -> None:
    """Delete temporary test user."""
    response = client.delete(f"{base_url}/api/users/{user_id}")
    print_response("CLEANUP", response)
    if response.status_code not in (200, 204, 404):
        response.raise_for_status()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mock Smart Locker client for backend contract testing."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL where backend is running (default: %(default)s)",
    )
    parser.add_argument(
        "--scenario",
        choices=("deny", "grant"),
        default="deny",
        help="Test scenario: deny (unknown embedding) or grant (register + recognize).",
    )
    parser.add_argument(
        "--name",
        default=f"MockUser-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Name used when --scenario grant (default: timestamped mock user).",
    )
    parser.add_argument(
        "--locker-id",
        default="L001",
        help="Locker ID in recognize payload (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.92,
        help="confidence_score value sent in payload (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic embeddings.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="When scenario is grant, delete the created user after recognize.",
    )
    parser.add_argument(
        "--print-payload",
        action="store_true",
        help="Print full recognize payload before request.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not (0.0 <= args.confidence <= 1.0):
        print("Error: --confidence must be between 0.0 and 1.0", file=sys.stderr)
        return 2

    embedding = generate_embedding(seed=args.seed)
    created_user_id: int | None = None

    try:
        with httpx.Client(timeout=args.timeout) as client:
            if args.scenario == "grant":
                created_user_id = register_user(client, args.base_url, args.name, embedding)
                print(f"Registered user_id={created_user_id} for grant scenario.")

            response = recognize(
                client=client,
                base_url=args.base_url,
                embedding=embedding,
                confidence=args.confidence,
                locker_id=args.locker_id,
                print_payload=args.print_payload,
            )
            response.raise_for_status()

            result = response.json()
            granted = result.get("access_granted")
            print(f"\nResult summary: access_granted={granted}")

            if args.scenario == "grant" and granted is not True:
                print("Warning: Expected grant scenario but access was denied.")
            if args.scenario == "deny" and granted is not False:
                print("Warning: Expected deny scenario but access was granted.")

    except httpx.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # Keep explicit top-level failure reporting for script users
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        # Cleanup is best-effort and should run even if recognize step failed.
        if args.cleanup and created_user_id is not None:
            try:
                with httpx.Client(timeout=args.timeout) as cleanup_client:
                    delete_user(cleanup_client, args.base_url, created_user_id)
            except Exception as cleanup_exc:
                print(f"Cleanup warning: {cleanup_exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
