#!/usr/bin/env python3
"""First-run bootstrap for a Smart Locker device.

Generates per-device secrets and writes them into ``backend/.env``:

    - SECRET_KEY               (64-hex random)
    - EMBEDDING_ENCRYPTION_KEY (32-byte base64, AES-256)
    - OPERATOR_PIN_HASH        (bcrypt hash of an operator PIN)

Run once on each physical locker device after install:

    python backend/scripts/bootstrap_device.py --operator-pin 837291

If any secret is already set to a real value, it is left alone. Placeholder
values (``change_this_to_a_random_secret``, empty, or missing) are replaced.
"""
from __future__ import annotations

import argparse
import base64
import os
import secrets
import sys
from pathlib import Path

import bcrypt

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
PLACEHOLDERS = {"", "change_this_to_a_random_secret", "dev_secret", "changeme"}


def _parse_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        env[key.strip()] = value.strip()
    return env


def _upsert_env(text: str, key: str, value: str) -> str:
    """Return env text with ``key=value`` inserted or replaced in-place."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    # Not found → append
    newline = "" if text.endswith("\n") else "\n"
    return text + newline + f"{key}={value}\n"


def _needs_rotation(current: str | None) -> bool:
    return current is None or current in PLACEHOLDERS


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a Smart Locker device")
    parser.add_argument(
        "--operator-pin",
        help="4-8 digit operator PIN (required unless OPERATOR_PIN_HASH already set)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rotate all secrets even if already set",
    )
    args = parser.parse_args()

    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} does not exist. Create it first.", file=sys.stderr)
        return 2

    text = ENV_PATH.read_text(encoding="utf-8")
    env = _parse_env(text)
    updated: list[str] = []

    # SECRET_KEY
    if args.force or _needs_rotation(env.get("SECRET_KEY")):
        text = _upsert_env(text, "SECRET_KEY", secrets.token_hex(32))
        updated.append("SECRET_KEY")

    # EMBEDDING_ENCRYPTION_KEY (32 random bytes, base64)
    if args.force or _needs_rotation(env.get("EMBEDDING_ENCRYPTION_KEY")):
        key = base64.b64encode(os.urandom(32)).decode("ascii")
        text = _upsert_env(text, "EMBEDDING_ENCRYPTION_KEY", key)
        text = _upsert_env(text, "EMBEDDING_ENCRYPTION_ENABLED", "true")
        updated.append("EMBEDDING_ENCRYPTION_KEY")
        updated.append("EMBEDDING_ENCRYPTION_ENABLED=true")

    # OPERATOR_PIN_HASH
    needs_pin = args.force or _needs_rotation(env.get("OPERATOR_PIN_HASH"))
    if needs_pin:
        if not args.operator_pin:
            print(
                "ERROR: --operator-pin required to generate OPERATOR_PIN_HASH. "
                "Provide a 4-8 digit PIN.",
                file=sys.stderr,
            )
            return 2
        if not args.operator_pin.isdigit() or not (4 <= len(args.operator_pin) <= 8):
            print("ERROR: operator PIN must be 4-8 digits.", file=sys.stderr)
            return 2
        h = bcrypt.hashpw(args.operator_pin.encode("utf-8"), bcrypt.gensalt(rounds=12))
        text = _upsert_env(text, "OPERATOR_PIN_HASH", h.decode("ascii"))
        updated.append("OPERATOR_PIN_HASH")

    # ENV flag (default production)
    if env.get("ENV") is None:
        text = _upsert_env(text, "ENV", "production")
        updated.append("ENV=production")

    if not updated:
        print("All secrets already set. Use --force to rotate.")
        return 0

    ENV_PATH.write_text(text, encoding="utf-8")
    print(f"Updated {ENV_PATH}:")
    for item in updated:
        print(f"  - {item}")
    print("\nDevice bootstrap complete. Restart the backend to pick up changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
