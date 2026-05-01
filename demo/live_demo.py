#!/usr/bin/env python3
r"""Live webcam enroll & recognize against the Smart Locker backend.

Usage (from repo root):
    .\.venv\Scripts\python.exe demo\live_demo.py enroll --name "Aakash"
    .\.venv\Scripts\python.exe demo\live_demo.py recognize
    .\.venv\Scripts\python.exe demo\live_demo.py live
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
ENROLL_CAPTURE_COUNT = 7
RECOGNIZE_CAPTURE_COUNT = 5
CAPTURE_DELAY_SEC = 0.4
LIVE_COOLDOWN_SEC = 3.0


def _find_user_for_locker(base_url: str, locker_id: str) -> dict | None:
    """Look up the user currently assigned to `locker_id` via GET /api/users."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base_url}/api/users")
    except httpx.HTTPError as exc:
        print(f"[users] Could not reach {base_url}/api/users: {exc}")
        return None

    if resp.status_code != 200:
        print(f"[users] GET /api/users failed ({resp.status_code}): {resp.text}")
        return None

    users = resp.json()
    for u in users:
        if u.get("assigned_locker_id") == locker_id:
            return u
    return None


def capture_frames(count: int, window_title: str = "Smart Locker") -> list[bytes]:
    """Open webcam, auto-detect face, capture `count` frames as JPEG bytes."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Close other apps using the camera.")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    frames: list[bytes] = []
    print(f"[camera] Capturing {count} frames — look at the camera...")

    while len(frames) < count:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        progress = f"{len(frames)}/{count}"
        cv2.putText(
            display, progress, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
        cv2.imshow(window_title, display)

        if len(faces) > 0:
            _, buf = cv2.imencode(".jpg", frame)
            frames.append(buf.tobytes())
            print(f"  captured {len(frames)}/{count}")
            time.sleep(CAPTURE_DELAY_SEC)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) < count:
        print(f"WARNING: Only captured {len(frames)}/{count} frames.")
    return frames


def enroll(name: str, base_url: str) -> dict | None:
    """Capture faces and enroll a new user."""
    print(f"\n=== ENROLL: {name} ===")
    frames = capture_frames(ENROLL_CAPTURE_COUNT)
    if not frames:
        print("No frames captured. Aborting.")
        return None

    files = [("images", (f"frame_{i}.jpg", f, "image/jpeg")) for i, f in enumerate(frames)]
    print(f"[enroll] Sending {len(frames)} images to {base_url}/api/enroll ...")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{base_url}/api/enroll", data={"name": name}, files=files)

    if resp.status_code == 200:
        body = resp.json()
        print(f"[enroll] SUCCESS — user_id={body['user_id']}, embeddings={body['embedding_count']}")
        return body
    else:
        print(f"[enroll] FAILED ({resp.status_code}): {resp.text}")
        return None


def recognize(base_url: str, locker_id: str = "L001") -> dict | None:
    """Capture faces and run recognition against the user assigned to `locker_id`."""
    print("\n=== RECOGNIZE ===")

    user = _find_user_for_locker(base_url, locker_id)
    if user is None:
        print(f"[recognize] No user assigned to locker '{locker_id}'. Enroll one first.")
        return None
    print(f"[recognize] Verifying against user_id={user['id']} ({user['name']}) on locker {locker_id}")

    frames = capture_frames(RECOGNIZE_CAPTURE_COUNT)
    if not frames:
        print("No frames captured. Aborting.")
        return None

    files = [("images", (f"frame_{i}.jpg", f, "image/jpeg")) for i, f in enumerate(frames)]
    print(f"[recognize] Sending {len(frames)} images to {base_url}/api/auth/recognize-multi ...")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{base_url}/api/auth/recognize-multi",
            params={
                "user_id": user["id"],
                "locker_id": locker_id,
                "check_liveness": "true",
            },
            files=files,
        )

    if resp.status_code == 200:
        body = resp.json()
        granted = body.get("access_granted")
        user_name = body.get("user_name", "unknown")
        score = body.get("final_score", 0)
        action = body.get("locker_action", "?")

        if granted:
            print(f"[recognize] ACCESS GRANTED — Welcome, {user_name}! (score: {score:.3f})")
        else:
            print(f"[recognize] ACCESS DENIED (score: {score:.3f})")
            # Diagnostic: show per-frame gate results so we can see what rejected
            print(f"[recognize] frames_processed={body.get('frames_processed')} "
                  f"passed_quality={body.get('frames_passed_quality')} "
                  f"passed_anomaly={body.get('frames_passed_anomaly')}")
            for fr in body.get("frame_results", []):
                print(f"  frame {fr.get('frame_index')}: "
                      f"quality={fr.get('quality_passed')} "
                      f"reasons={fr.get('rejection_reasons')} "
                      f"antispoof={fr.get('antispoof_passed')} "
                      f"anomaly={fr.get('anomaly_passed')} "
                      f"score={fr.get('score')}")
            live = body.get("liveness")
            if live:
                print(f"  liveness: passed={live.get('passed')} "
                      f"blink={live.get('blink_detected')} "
                      f"head_movement={live.get('head_movement_detected')} "
                      f"reason={live.get('reason')}")
        print(f"[recognize] locker_action={action}, log_id={body.get('log_id')}")
        return body
    else:
        print(f"[recognize] FAILED ({resp.status_code}): {resp.text}")
        return None


def live_mode(base_url: str, locker_id: str = "L001") -> None:
    """Continuous kiosk mode — auto-scan when a face is detected."""
    print("\n=== LIVE KIOSK MODE (press Q on camera window to quit) ===")

    user = _find_user_for_locker(base_url, locker_id)
    if user is None:
        print(f"[live] No user assigned to locker '{locker_id}'. Enroll one first.")
        return
    print(f"[live] Verifying against user_id={user['id']} ({user['name']}) on locker {locker_id}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    last_scan_time = 0.0
    status_text = "Waiting for face..."
    status_color = (200, 200, 200)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        now = time.time()
        cooldown_remaining = LIVE_COOLDOWN_SEC - (now - last_scan_time)

        if len(faces) > 0 and cooldown_remaining <= 0:
            # Capture a burst of frames for recognition
            burst: list[bytes] = []
            for _ in range(RECOGNIZE_CAPTURE_COUNT):
                ret2, f2 = cap.read()
                if ret2:
                    _, buf = cv2.imencode(".jpg", f2)
                    burst.append(buf.tobytes())
                    time.sleep(0.1)

            if burst:
                files = [("images", (f"f_{i}.jpg", b, "image/jpeg")) for i, b in enumerate(burst)]
                try:
                    with httpx.Client(timeout=15.0) as client:
                        resp = client.post(
                            f"{base_url}/api/auth/recognize-multi",
                            params={
                                "user_id": user["id"],
                                "locker_id": locker_id,
                                "check_liveness": "true",
                            },
                            files=files,
                        )
                    if resp.status_code == 200:
                        body = resp.json()
                        if body.get("access_granted"):
                            name = body.get("user_name", "?")
                            score = body.get("final_score", 0)
                            status_text = f"GRANTED: {name} ({score:.2f})"
                            status_color = (0, 255, 0)
                        else:
                            score = body.get("final_score", 0)
                            status_text = f"DENIED ({score:.2f})"
                            status_color = (0, 0, 255)
                    else:
                        status_text = f"Error {resp.status_code}"
                        status_color = (0, 0, 255)
                except httpx.HTTPError as exc:
                    status_text = f"Connection error: {exc}"
                    status_color = (0, 0, 255)

                last_scan_time = time.time()

        # Draw status bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(
            display, status_text, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2,
        )
        if cooldown_remaining > 0:
            cv2.putText(
                display, f"Next scan in {cooldown_remaining:.1f}s",
                (display.shape[1] - 280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1,
            )

        cv2.imshow("Smart Locker — Live", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[live] Exited.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Locker live webcam demo")
    parser.add_argument("--api", default=DEFAULT_BASE_URL, help="Backend URL")
    parser.add_argument("--locker", default="L001", help="Locker ID")
    sub = parser.add_subparsers(dest="command", required=True)

    enroll_p = sub.add_parser("enroll", help="Enroll a new user via webcam")
    enroll_p.add_argument("--name", required=True, help="User name")

    sub.add_parser("recognize", help="Single recognition scan")
    sub.add_parser("live", help="Continuous kiosk mode")

    args = parser.parse_args()

    if args.command == "enroll":
        result = enroll(args.name, args.api)
        return 0 if result else 1
    elif args.command == "recognize":
        result = recognize(args.api, args.locker)
        return 0 if result else 1
    elif args.command == "live":
        live_mode(args.api, args.locker)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
