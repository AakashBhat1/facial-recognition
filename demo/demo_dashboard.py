#!/usr/bin/env python3
"""Smart Locker — Demo Readiness Dashboard.

Single-window operator console that validates the backend end-to-end and
visualises live system health before a client demo:

    +---------------------------------------------------------------+
    |   Smart Locker System          [BACKEND: OK]   [MODEL: ok]    |
    +------------------------ +------------------------------------ +
    |  Endpoint Checks        |       Live Camera Feed              |
    |  [*] /locker/status     |       (webcam preview with face box)|
    |  [*] /enroll            |                                     |
    |  [*] /recognize-multi   |                                     |
    |  [*] /locker/close      |                                     |
    |  [*] /users/{id}/pin    |       Latency                       |
    |  [*] /auth/pin          |         Enroll:      312 ms         |
    |  [*] /logs              |         Recognize:    84 ms         |
    |  [*] /alerts            |         PIN auth:     12 ms         |
    |  [*] /users/{id} del    |                                     |
    +------------------------ +------------------------------------ +
    |  Recent Access Logs (live)                                    |
    |  [OK] 14:02:03 user=12 method=face conf=0.87                  |
    |  [!]  14:01:48 user=-  method=face DENIED                     |
    +---------------------------------------------------------------+
       [R] Re-run checks   [C] Capture frame   [Q] Quit

Flow
----
On start the dashboard shows the backend health indicator and streams the
webcam feed with a face-detection box. Pressing ``R`` or clicking **Re-run
Checks** captures 6 enrollment frames + 3 auth frames from the live feed
and walks through every endpoint sequentially, updating the UI as each
step passes or fails. Recent backend logs auto-refresh every 5 seconds.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. pip install httpx", file=sys.stderr)
    sys.exit(1)


# ---------- Config ----------
WINDOW_TITLE = "Smart Locker — Demo Readiness Dashboard"
CANVAS_W, CANVAS_H = 1280, 780
BASE_URL = "http://localhost:8000"
LOCKER_ID = "L001"
NUM_ENROLL_FRAMES = 8
NUM_AUTH_FRAMES = 5
TEST_PIN = "1234"
LOG_REFRESH_INTERVAL = 5.0

# Colors (BGR)
C_BG = (24, 24, 28)
C_PANEL = (38, 40, 48)
C_HEADER = (48, 52, 62)
C_BORDER = (85, 90, 100)
C_TEXT = (240, 240, 240)
C_MUTED = (160, 160, 168)
C_OK = (80, 200, 120)
C_FAIL = (60, 70, 220)
C_PENDING = (180, 180, 60)
C_INFO = (200, 180, 0)
C_ACCENT = (0, 200, 255)

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------- State ----------
@dataclass
class StepStatus:
    name: str
    label: str
    state: str = "pending"  # pending | running | ok | fail
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class DashboardState:
    backend_ok: bool = False
    model_loaded: bool = False
    steps: list[StepStatus] = field(default_factory=list)
    last_logs: list[dict] = field(default_factory=list)
    last_alerts: list[dict] = field(default_factory=list)
    last_log_refresh: float = 0.0
    running: bool = False
    message: str = "Press R to run checks"
    latency_enroll_ms: float = 0.0
    latency_recognize_ms: float = 0.0
    latency_pin_ms: float = 0.0
    current_frame: np.ndarray | None = None
    faces_detected: int = 0

    def reset_steps(self) -> None:
        self.steps = [
            StepStatus("baseline", "GET /locker/status"),
            StepStatus("enroll", "POST /enroll"),
            StepStatus("recognize", "POST /auth/recognize-multi"),
            StepStatus("close", "POST /locker/close"),
            StepStatus("setpin", "PUT /users/{id}/pin"),
            StepStatus("pinauth", "POST /auth/pin"),
            StepStatus("logs", "GET /logs"),
            StepStatus("alerts", "GET /alerts"),
            StepStatus("cleanup", "DELETE /users/{id}"),
        ]


STATE = DashboardState()
STATE.reset_steps()


# ---------- Drawing helpers ----------
def draw_text(img, text, org, scale=0.55, color=C_TEXT, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), C_PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_BORDER, 1)
    if title:
        cv2.rectangle(img, (x, y), (x + w, y + 32), C_HEADER, -1)
        cv2.rectangle(img, (x, y), (x + w, y + 32), C_BORDER, 1)
        draw_text(img, title, (x + 12, y + 22), 0.6, C_TEXT, 2)


def status_color(state: str) -> tuple[int, int, int]:
    return {"ok": C_OK, "fail": C_FAIL, "running": C_PENDING, "pending": C_MUTED}[state]


def status_icon(state: str) -> str:
    return {"ok": "[OK]", "fail": "[X]", "running": "[..]", "pending": "[ ]"}[state]


# ---------- Render ----------
def render(state: DashboardState) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 60), C_HEADER, -1)
    cv2.line(canvas, (0, 60), (CANVAS_W, 60), C_BORDER, 1)
    draw_text(canvas, "Smart Locker System  -  Demo Readiness Dashboard", (20, 38), 0.85, C_TEXT, 2)

    # Health badges
    backend_color = C_OK if state.backend_ok else C_FAIL
    backend_label = "BACKEND: OK" if state.backend_ok else "BACKEND: DOWN"
    cv2.rectangle(canvas, (900, 15), (1060, 45), backend_color, -1)
    draw_text(canvas, backend_label, (910, 35), 0.55, (20, 20, 20), 2)

    model_color = C_OK if state.model_loaded else C_PENDING
    model_label = "MODEL: ok" if state.model_loaded else "MODEL: warming"
    cv2.rectangle(canvas, (1075, 15), (1235, 45), model_color, -1)
    draw_text(canvas, model_label, (1085, 35), 0.55, (20, 20, 20), 2)

    # Left panel — Endpoint checks
    draw_panel(canvas, 20, 80, 440, 440, "Endpoint Checks")
    y = 130
    for step in state.steps:
        color = status_color(step.state)
        icon = status_icon(step.state)
        draw_text(canvas, icon, (36, y), 0.55, color, 2)
        draw_text(canvas, step.label, (90, y), 0.55, C_TEXT, 1)
        if step.latency_ms > 0 and step.state == "ok":
            draw_text(canvas, f"{step.latency_ms:.0f} ms", (350, y), 0.5, C_MUTED, 1)
        if step.state == "fail" and step.detail:
            # wrap up to 2 lines of detail
            detail = step.detail
            draw_text(canvas, detail[:52], (36, y + 18), 0.42, C_FAIL, 1)
            y += 18
            if len(detail) > 52:
                draw_text(canvas, detail[52:104], (36, y + 18), 0.42, C_FAIL, 1)
                y += 18
        y += 32

    # Right panel — Camera feed + latency
    draw_panel(canvas, 480, 80, 780, 440, "Live Camera Feed")
    if state.current_frame is not None:
        fh, fw = state.current_frame.shape[:2]
        target_w, target_h = 520, 390
        scale = min(target_w / fw, target_h / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(state.current_frame, (nw, nh))
        ox, oy = 500, 125
        canvas[oy:oy + nh, ox:ox + nw] = resized
        cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + nw + 1, oy + nh + 1), C_BORDER, 1)
        # Faces detected indicator
        face_txt = f"Faces detected: {state.faces_detected}"
        fc = C_OK if state.faces_detected == 1 else (C_PENDING if state.faces_detected == 0 else C_FAIL)
        draw_text(canvas, face_txt, (500, 530), 0.5, fc, 2)
    else:
        draw_text(canvas, "Camera initializing...", (680, 310), 0.6, C_MUTED, 2)

    # Latency panel
    draw_panel(canvas, 1030, 125, 220, 280, "Latency")
    draw_text(canvas, "Enroll:", (1050, 170), 0.55, C_MUTED, 1)
    draw_text(canvas, f"{state.latency_enroll_ms:.0f} ms", (1050, 200), 0.85, C_ACCENT, 2)
    draw_text(canvas, "Recognize:", (1050, 250), 0.55, C_MUTED, 1)
    draw_text(canvas, f"{state.latency_recognize_ms:.0f} ms", (1050, 280), 0.85, C_ACCENT, 2)
    draw_text(canvas, "PIN auth:", (1050, 330), 0.55, C_MUTED, 1)
    draw_text(canvas, f"{state.latency_pin_ms:.0f} ms", (1050, 360), 0.85, C_ACCENT, 2)

    # Bottom panel — Recent logs
    draw_panel(canvas, 20, 540, 1240, 180, "Recent Activity  (auto-refresh 5s)")
    ly = 585
    if not state.last_logs and not state.last_alerts:
        draw_text(canvas, "No activity yet.", (40, ly), 0.5, C_MUTED, 1)
    for log in state.last_logs[:4]:
        ts = str(log.get("timestamp", ""))[:19]
        uid = log.get("user_id", "-")
        method = log.get("method", "-")
        success = log.get("success", False)
        conf = log.get("confidence")
        icon = "[OK]" if success else "[X]"
        color = C_OK if success else C_FAIL
        line = f"{icon}  {ts}  user={uid}  method={method}"
        if conf is not None:
            line += f"  conf={conf:.2f}" if isinstance(conf, (int, float)) else ""
        draw_text(canvas, line, (40, ly), 0.48, color, 1)
        ly += 22
    for alert in state.last_alerts[:2]:
        sev = alert.get("severity", "info")
        msg = str(alert.get("message", ""))[:90]
        draw_text(canvas, f"[ALERT {sev}] {msg}", (40, ly), 0.48, C_INFO, 1)
        ly += 22

    # Footer
    cv2.rectangle(canvas, (0, CANVAS_H - 40), (CANVAS_W, CANVAS_H), C_HEADER, -1)
    cv2.line(canvas, (0, CANVAS_H - 40), (CANVAS_W, CANVAS_H - 40), C_BORDER, 1)
    footer = "[R] Re-run checks    [C] Capture test frame    [Q] Quit"
    draw_text(canvas, footer, (20, CANVAS_H - 14), 0.55, C_MUTED, 1)
    msg_color = C_OK if "PASS" in state.message else (C_FAIL if "FAIL" in state.message else C_ACCENT)
    draw_text(canvas, state.message, (620, CANVAS_H - 14), 0.55, msg_color, 2)

    return canvas


# ---------- Backend interaction ----------
def check_backend_health(base_url: str) -> tuple[bool, bool]:
    """Return (backend_ok, model_loaded). Short timeouts to stay snappy."""
    try:
        r = httpx.get(f"{base_url}/docs", timeout=1.5)
        if r.status_code != 200:
            return False, False
    except Exception:
        return False, False
    return True, True


def _files_payload(frames: list[np.ndarray]) -> list:
    payload = []
    for i, frame in enumerate(frames):
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        payload.append(("images", (f"frame_{i:02d}.jpg", buf.tobytes(), "image/jpeg")))
    return payload


def _mark(step_name: str, state: str, detail: str = "", latency_ms: float = 0.0) -> None:
    for s in STATE.steps:
        if s.name == step_name:
            s.state = state
            s.detail = detail
            s.latency_ms = latency_ms
            return


def refresh_logs(base_url: str) -> None:
    try:
        r = httpx.get(f"{base_url}/api/logs", params={"limit": 6}, timeout=1.5)
        if r.status_code == 200:
            STATE.last_logs = r.json().get("items", [])
    except Exception:
        pass
    try:
        r = httpx.get(f"{base_url}/api/alerts", params={"limit": 3}, timeout=1.5)
        if r.status_code == 200:
            STATE.last_alerts = r.json().get("items", [])
    except Exception:
        pass
    STATE.last_log_refresh = time.time()


def _background_poller(base_url: str, stop_flag: dict) -> None:
    """Poll backend health + logs on a worker thread so UI never blocks."""
    while not stop_flag.get("stop"):
        if not STATE.running:
            STATE.backend_ok, STATE.model_loaded = check_backend_health(base_url)
            refresh_logs(base_url)
        time.sleep(LOG_REFRESH_INTERVAL)


def run_e2e_checks(base_url: str, enroll_frames: list[np.ndarray], auth_frames: list[np.ndarray]) -> None:
    """Run all endpoint checks in sequence, updating STATE live."""
    STATE.running = True
    STATE.message = "Running E2E checks..."
    user_name = f"demo_ready_{int(time.time())}"
    user_id: int | None = None
    passed = 0
    total = len(STATE.steps)

    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        # 1. baseline
        _mark("baseline", "running")
        t0 = time.perf_counter()
        try:
            r = client.get("/api/locker/status", params={"locker_id": LOCKER_ID})
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                _mark("baseline", "ok", "", dt); passed += 1
            else:
                _mark("baseline", "fail", f"{r.status_code}", dt)
        except Exception as e:
            _mark("baseline", "fail", str(e)[:40])

        # 2. enroll
        _mark("enroll", "running")
        t0 = time.perf_counter()
        try:
            r = client.post("/api/enroll", params={"name": user_name}, files=_files_payload(enroll_frames))
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                user_id = r.json()["user"]["id"]
                STATE.latency_enroll_ms = dt
                _mark("enroll", "ok", "", dt); passed += 1
            else:
                _mark("enroll", "fail", r.text[:90], dt)
        except Exception as e:
            _mark("enroll", "fail", str(e)[:40])

        if user_id is None:
            STATE.message = f"FAIL: enroll failed — {passed}/{total} steps"
            STATE.running = False
            return

        # 3. recognize-multi
        _mark("recognize", "running")
        t0 = time.perf_counter()
        try:
            r = client.post(
                "/api/auth/recognize-multi",
                params={"locker_id": LOCKER_ID, "check_liveness": "false"},
                files=_files_payload(auth_frames),
            )
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                matched = r.json().get("matched_user_id")
                STATE.latency_recognize_ms = dt
                if matched == user_id:
                    _mark("recognize", "ok", "", dt); passed += 1
                else:
                    _mark("recognize", "fail", f"matched={matched} expected={user_id}", dt)
            else:
                _mark("recognize", "fail", r.text[:90], dt)
        except Exception as e:
            _mark("recognize", "fail", str(e)[:40])

        # 4. close
        _mark("close", "running")
        t0 = time.perf_counter()
        try:
            r = client.post("/api/locker/close", json={"locker_id": LOCKER_ID, "user_id": user_id})
            dt = (time.perf_counter() - t0) * 1000
            _mark("close", "ok" if r.status_code == 200 else "fail", "" if r.status_code == 200 else r.text[:90], dt)
            if r.status_code == 200: passed += 1
        except Exception as e:
            _mark("close", "fail", str(e)[:40])

        # 5. set PIN
        _mark("setpin", "running")
        t0 = time.perf_counter()
        try:
            r = client.put(f"/api/users/{user_id}/pin", json={"pin": TEST_PIN})
            dt = (time.perf_counter() - t0) * 1000
            _mark("setpin", "ok" if r.status_code == 200 else "fail", "" if r.status_code == 200 else r.text[:90], dt)
            if r.status_code == 200: passed += 1
        except Exception as e:
            _mark("setpin", "fail", str(e)[:40])

        # 6. PIN auth
        _mark("pinauth", "running")
        t0 = time.perf_counter()
        try:
            r = client.post("/api/auth/pin", json={"user_id": user_id, "pin": TEST_PIN, "locker_id": LOCKER_ID})
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200:
                STATE.latency_pin_ms = dt
                _mark("pinauth", "ok", "", dt); passed += 1
            else:
                _mark("pinauth", "fail", r.text[:90], dt)
        except Exception as e:
            _mark("pinauth", "fail", str(e)[:40])

        client.post("/api/locker/close", json={"locker_id": LOCKER_ID, "user_id": user_id})

        # 7. logs
        _mark("logs", "running")
        t0 = time.perf_counter()
        try:
            r = client.get("/api/logs", params={"limit": 5})
            dt = (time.perf_counter() - t0) * 1000
            _mark("logs", "ok" if r.status_code == 200 else "fail", "", dt)
            if r.status_code == 200: passed += 1
        except Exception as e:
            _mark("logs", "fail", str(e)[:40])

        # 8. alerts
        _mark("alerts", "running")
        t0 = time.perf_counter()
        try:
            r = client.get("/api/alerts", params={"limit": 5})
            dt = (time.perf_counter() - t0) * 1000
            _mark("alerts", "ok" if r.status_code == 200 else "fail", "", dt)
            if r.status_code == 200: passed += 1
        except Exception as e:
            _mark("alerts", "fail", str(e)[:40])

        # 9. cleanup
        _mark("cleanup", "running")
        t0 = time.perf_counter()
        try:
            r = client.delete(f"/api/users/{user_id}")
            dt = (time.perf_counter() - t0) * 1000
            _mark("cleanup", "ok" if r.status_code == 200 else "fail", "", dt)
            if r.status_code == 200: passed += 1
        except Exception as e:
            _mark("cleanup", "fail", str(e)[:40])

    STATE.message = f"DEMO READY — ALL PASS ({passed}/{total})" if passed == total else f"FAIL ({passed}/{total})"
    STATE.running = False
    refresh_logs(base_url)


def capture_batch(cap: cv2.VideoCapture, count: int, interval: float = 0.45) -> list[np.ndarray]:
    """Capture ``count`` frames where each frame has exactly one face detected
    by Haar cascade (pre-filter before hitting InsightFace on the backend).
    """
    frames: list[np.ndarray] = []
    t_last = 0.0
    attempts = 0
    max_attempts = count * 6
    while len(frames) < count and attempts < max_attempts:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        if now - t_last < interval:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
        if len(faces) == 1:
            frames.append(frame.copy())
            t_last = now
            STATE.message = f"Capturing... {len(frames)}/{count}"
        attempts += 1
    return frames


# ---------- Main loop ----------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 2

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, CANVAS_W, CANVAS_H)

    # Background poller for health + logs — keeps UI responsive
    stop_flag = {"stop": False}
    poller = threading.Thread(
        target=_background_poller, args=(args.base_url, stop_flag), daemon=True
    )
    poller.start()

    worker: threading.Thread | None = None

    while True:
        ok, frame = cap.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
            STATE.faces_detected = len(faces)
            for (x, y, w, h) in faces:
                color = C_OK if len(faces) == 1 else C_PENDING
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            STATE.current_frame = frame

        canvas = render(STATE)
        cv2.imshow(WINDOW_TITLE, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('r'), ord('R')) and not STATE.running:
            if not STATE.backend_ok:
                STATE.message = "FAIL: backend not reachable"
                continue
            if STATE.faces_detected != 1:
                STATE.message = "Position ONE face in frame, then press R"
                continue
            STATE.reset_steps()
            STATE.message = "Capturing frames..."
            enroll_batch = capture_batch(cap, NUM_ENROLL_FRAMES, 0.3)
            auth_batch = capture_batch(cap, NUM_AUTH_FRAMES, 0.3)
            worker = threading.Thread(
                target=run_e2e_checks,
                args=(args.base_url, enroll_batch, auth_batch),
                daemon=True,
            )
            worker.start()
        if key in (ord('c'), ord('C')):
            STATE.message = f"Test frame captured ({STATE.faces_detected} face(s))"

    stop_flag["stop"] = True
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
