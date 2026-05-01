"""Smart Locker — Interactive Kiosk Simulation.

Single-window state-machine kiosk that mirrors a real locker bank:

    GRID  --click locker-->  MENU  --[Login / Sign Up / Back]-->  CAPTURE
                                                                    |
                                                                  RESULT
                                                                    |
                                                                   GRID

Locker <-> user mapping is persisted locally in
`demo/locker_assignments.json` so sign-up binds a user to a specific slot
and subsequent logins only succeed on that slot.

Controls
--------
  Mouse click   : choose locker / button
  Keys 1-6      : choose locker by number (on GRID screen)
  L / S         : Login / Sign-up shortcut (on MENU screen)
  B  or  ESC    : Back
  Q             : Quit

Usage
-----
    python demo/locker_simulation.py
    python demo/locker_simulation.py --api http://localhost:8000 --capacity 6
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Smart Locker — Kiosk"
CANVAS_W, CANVAS_H = 1280, 720
GRID_COLS = 2
GRID_ROWS = 3
LOCKER_PAD = 18
ENROLL_PHOTO_COUNT = 8
CAPTURE_STABLE_THRESHOLD = 8
CAPTURE_INTERVAL_SEC = 0.5
LOGIN_FRAME_COUNT = 8            # frames POSTed for liveness-checked login
LOGIN_CAPTURE_INTERVAL = 0.25    # seconds between login frames (~2s total)
RESULT_DISPLAY_SECONDS = 3.5

ASSIGNMENTS_FILE = Path(__file__).parent / "locker_assignments.json"

# Colors (BGR)
C_BG = (24, 24, 28)
C_PANEL = (34, 36, 44)
C_DOOR_CLOSED = (70, 72, 82)
C_DOOR_EMPTY = (48, 50, 58)
C_DOOR_OPEN = (46, 160, 90)
C_DOOR_DENIED = (52, 52, 190)
C_BORDER = (110, 115, 125)
C_TEXT = (240, 240, 240)
C_MUTED = (170, 170, 175)
C_ACCENT = (0, 200, 255)
C_BTN = (70, 130, 180)
C_BTN_HOVER = (100, 160, 210)
C_BTN_BACK = (80, 80, 88)
C_BTN_DANGER = (60, 60, 180)

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.7,
    color: tuple = C_TEXT,
    thickness: int = 2,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_text_centered(
    img: np.ndarray,
    text: str,
    center: tuple[int, int],
    scale: float = 0.7,
    color: tuple = C_TEXT,
    thickness: int = 2,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = (center[0] - tw // 2, center[1] + th // 2)
    draw_text(img, text, org, scale, color, thickness)


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class Button:
    rect: Rect
    label: str
    action: str
    color: tuple = C_BTN


def draw_button(img: np.ndarray, btn: Button, hovered: bool) -> None:
    color = C_BTN_HOVER if hovered and btn.color == C_BTN else btn.color
    cv2.rectangle(img, (btn.rect.x, btn.rect.y),
                  (btn.rect.x + btn.rect.w, btn.rect.y + btn.rect.h), color, -1)
    cv2.rectangle(img, (btn.rect.x, btn.rect.y),
                  (btn.rect.x + btn.rect.w, btn.rect.y + btn.rect.h), C_BORDER, 2)
    draw_text_centered(img, btn.label, btn.rect.center, scale=0.75, color=C_TEXT, thickness=2)


# ---------------------------------------------------------------------------
# Locker assignment persistence
# ---------------------------------------------------------------------------
def load_assignments() -> dict[int, str]:
    if not ASSIGNMENTS_FILE.exists():
        return {}
    try:
        raw = json.loads(ASSIGNMENTS_FILE.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError):
        return {}


def save_assignments(assignments: dict[int, str]) -> None:
    ASSIGNMENTS_FILE.write_text(
        json.dumps({str(k): v for k, v in assignments.items()}, indent=2),
        encoding="utf-8",
    )


def bootstrap_assignments_from_backend(api_base: str, capacity: int) -> dict[int, str]:
    """If no local assignments yet, seed from already-enrolled users (by id)."""
    current = load_assignments()
    if current:
        return current
    try:
        resp = requests.get(f"{api_base}/api/users", timeout=4)
        resp.raise_for_status()
        users = sorted(resp.json(), key=lambda u: u.get("id", 0))
    except requests.exceptions.RequestException:
        return {}
    mapping: dict[int, str] = {}
    for i, u in enumerate(users[:capacity]):
        mapping[i + 1] = u["name"]
    if mapping:
        save_assignments(mapping)
    return mapping


# ---------------------------------------------------------------------------
# Face detect + camera capture
# ---------------------------------------------------------------------------
def detect_face(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------
def api_list_users(api_base: str) -> list[str]:
    try:
        resp = requests.get(f"{api_base}/api/users", timeout=4)
        resp.raise_for_status()
        return [u["name"] for u in resp.json()]
    except requests.exceptions.RequestException:
        return []


def api_find_user_id(api_base: str, name: str) -> int | None:
    try:
        resp = requests.get(f"{api_base}/api/users", timeout=4)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    target = name.strip().lower()
    for user in resp.json():
        if str(user.get("name", "")).strip().lower() == target:
            user_id = user.get("id")
            return int(user_id) if user_id is not None else None
    return None


def api_enroll(api_base: str, name: str, frames: list[bytes]) -> tuple[bool, dict | str]:
    files = [("images", (f"face{i}.jpg", buf, "image/jpeg")) for i, buf in enumerate(frames)]
    try:
        resp = requests.post(f"{api_base}/api/enroll", data={"name": name}, files=files, timeout=90)
    except requests.exceptions.RequestException as exc:
        return False, f"Network error: {exc}"
    if resp.status_code == 200:
        data = resp.json()
        return True, {
            "message": f"Enrolled {name} ({data.get('embedding_count', 0)} embeddings)",
            "user_id": data.get("user_id"),
        }
    return False, f"Enroll failed: HTTP {resp.status_code} {resp.text[:120]}"


def api_recognize_multi(api_base: str, user_id: int, locker_id: str, frames: list[bytes]) -> tuple[bool, dict | str]:
    """Call the multi-frame recognize endpoint with liveness checks enabled."""
    files = [("images", (f"f{i}.jpg", buf, "image/jpeg")) for i, buf in enumerate(frames)]
    try:
        resp = requests.post(
            f"{api_base}/api/auth/recognize-multi",
            files=files,
            params={"user_id": user_id, "locker_id": locker_id, "check_liveness": "true"},
            timeout=60,
        )
    except requests.exceptions.RequestException as exc:
        return False, f"Network error: {exc}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:160]}"
    return True, resp.json()


def api_login(api_base: str, name: str, locker_id: str, frames: list[bytes]) -> tuple[bool, dict | str]:
    user_id = api_find_user_id(api_base, name)
    if user_id is None:
        return False, f"Could not find enrolled user '{name}' on backend"
    return api_recognize_multi(api_base, user_id, locker_id, frames)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class KioskState:
    screen: str = "grid"  # grid | menu | capture_enroll | capture_login | result | text_input
    assignments: dict[int, str] = field(default_factory=dict)
    capacity: int = 6
    selected_slot: int | None = None
    mouse: tuple[int, int] = (0, 0)
    click: tuple[int, int] | None = None
    buttons: list[Button] = field(default_factory=list)
    # capture
    captured_frames: list[bytes] = field(default_factory=list)
    stable_count: int = 0
    last_capture_ts: float = 0.0
    # text input (enrollment name)
    input_text: str = ""
    input_prompt: str = ""
    pending_action: str = ""   # "signup" etc.
    # result
    result_title: str = ""
    result_detail: str = ""
    result_color: tuple = C_TEXT
    result_until: float = 0.0
    # async work
    busy_kind: str = ""                      # "" | "enroll" | "login" | "name_check"
    busy_label: str = ""
    busy_started_at: float = 0.0
    busy_result: tuple | None = None
    busy_thread: threading.Thread | None = None
    pending_name: str = ""

    def clear_buttons(self) -> None:
        self.buttons = []

    def consume_click(self) -> tuple[int, int] | None:
        c = self.click
        self.click = None
        return c


def start_busy(state: KioskState, kind: str, label: str, fn: Callable[[], tuple]) -> None:
    """Run an API call in a background thread so the UI stays responsive."""
    state.busy_kind = kind
    state.busy_label = label
    state.busy_started_at = time.time()
    state.busy_result = None

    def worker() -> None:
        try:
            result = fn()
        except Exception as exc:   # defensive — should be caught inside api_* helpers
            result = (False, f"Unexpected error: {exc}")
        state.busy_result = result

    thread = threading.Thread(target=worker, daemon=True)
    state.busy_thread = thread
    thread.start()


def render_busy_overlay(state: KioskState, frame: np.ndarray) -> np.ndarray:
    """Draw a live camera feed with a spinner + label while a thread runs."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)
    view_w, view_h = 960, 540
    cam = cv2.resize(frame, (view_w, view_h))
    vx = (CANVAS_W - view_w) // 2
    vy = 90
    canvas[vy:vy + view_h, vx:vx + view_w] = cam
    cv2.rectangle(canvas, (vx, vy), (vx + view_w, vy + view_h), C_BORDER, 2)

    draw_text(canvas, "PLEASE WAIT...", (24, 50), scale=0.9, color=C_ACCENT)

    # Spinner
    elapsed = time.time() - state.busy_started_at
    angle = (elapsed * 360.0) % 360.0
    center = (CANVAS_W // 2, vy + view_h + 60)
    r = 22
    cv2.ellipse(canvas, center, (r, r), angle, 0, 270, C_ACCENT, 4)
    cv2.circle(canvas, center, r, C_BORDER, 1)

    draw_text_centered(canvas, f"{state.busy_label}  ({elapsed:.1f}s)",
                       (CANVAS_W // 2, vy + view_h + 120), scale=0.75, color=C_TEXT, thickness=2)
    draw_text_centered(canvas, "Server can take a few seconds on first call (model warm-up).",
                       (CANVAS_W // 2, CANVAS_H - 30), scale=0.5, color=C_MUTED, thickness=1)
    return canvas


def mouse_callback(event, x, y, flags, state: KioskState) -> None:
    state.mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state.click = (x, y)


# ---------------------------------------------------------------------------
# Screen: GRID
# ---------------------------------------------------------------------------
def compute_locker_rects(capacity: int) -> list[Rect]:
    top = 130
    bottom_margin = 60
    avail_w = CANVAS_W - LOCKER_PAD * (GRID_COLS + 1)
    avail_h = CANVAS_H - top - bottom_margin - LOCKER_PAD * (GRID_ROWS + 1)
    cell_w = avail_w // GRID_COLS
    cell_h = avail_h // GRID_ROWS
    rects: list[Rect] = []
    for idx in range(capacity):
        row, col = divmod(idx, GRID_COLS)
        x = LOCKER_PAD + col * (cell_w + LOCKER_PAD)
        y = top + LOCKER_PAD + row * (cell_h + LOCKER_PAD)
        rects.append(Rect(x, y, cell_w, cell_h))
    return rects


def render_grid(state: KioskState) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)
    draw_text(canvas, "SMART LOCKER  —  SELECT A LOCKER", (24, 50), scale=1.0, color=C_ACCENT)
    draw_text(canvas, "Click a locker, or press keys 1-6.  Q to quit.",
              (24, 90), scale=0.55, color=C_MUTED, thickness=1)

    rects = compute_locker_rects(state.capacity)
    mx, my = state.mouse
    for idx, rect in enumerate(rects):
        slot = idx + 1
        user = state.assignments.get(slot)
        occupied = user is not None
        hovered = rect.contains(mx, my)
        color = C_DOOR_CLOSED if occupied else C_DOOR_EMPTY
        if hovered:
            color = tuple(min(c + 30, 255) for c in color)
        cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), color, -1)
        cv2.rectangle(canvas, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), C_BORDER, 2)
        # handle
        hx = rect.x + rect.w - 34
        cv2.rectangle(canvas, (hx, rect.center[1] - 20),
                      (hx + 14, rect.center[1] + 20), C_BORDER, -1)
        draw_text(canvas, f"#{slot:02d}", (rect.x + 16, rect.y + 36), scale=0.7, color=C_TEXT)
        label = user if occupied else "Empty — tap to register"
        label_color = C_TEXT if occupied else C_MUTED
        draw_text(canvas, label, (rect.x + 16, rect.y + rect.h - 22),
                  scale=0.6, color=label_color, thickness=2)
        status = "REGISTERED" if occupied else "AVAILABLE"
        status_color = (180, 255, 200) if occupied else C_MUTED
        draw_text(canvas, status, (rect.x + rect.w - 140, rect.y + 36),
                  scale=0.5, color=status_color, thickness=1)

    # Store rects so click handler can hit-test
    state.buttons = [
        Button(rect=r, label=str(i + 1), action=f"select:{i + 1}") for i, r in enumerate(rects)
    ]
    return canvas


def handle_grid(state: KioskState, key: int) -> None:
    click = state.consume_click()
    if click is not None:
        for btn in state.buttons:
            if btn.rect.contains(*click):
                _, slot_s = btn.action.split(":")
                state.selected_slot = int(slot_s)
                goto_menu(state)
                return
    if ord("1") <= key <= ord("9"):
        slot = key - ord("0")
        if 1 <= slot <= state.capacity:
            state.selected_slot = slot
            goto_menu(state)


# ---------------------------------------------------------------------------
# Screen: MENU
# ---------------------------------------------------------------------------
def goto_menu(state: KioskState) -> None:
    state.screen = "menu"
    state.clear_buttons()


def render_menu(state: KioskState) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)
    slot = state.selected_slot or 0
    user = state.assignments.get(slot)
    draw_text(canvas, f"LOCKER #{slot:02d}", (CANVAS_W // 2 - 140, 120),
              scale=1.3, color=C_ACCENT, thickness=3)
    if user:
        draw_text_centered(canvas, f"Registered to: {user}", (CANVAS_W // 2, 180),
                           scale=0.8, color=C_TEXT)
    else:
        draw_text_centered(canvas, "This locker has no registered user.",
                           (CANVAS_W // 2, 180), scale=0.7, color=C_MUTED)

    # Build buttons
    btn_w, btn_h = 320, 90
    cx = CANVAS_W // 2
    top_y = 260
    login = Button(rect=Rect(cx - btn_w - 20, top_y, btn_w, btn_h),
                   label="LOGIN  (scan face)", action="login", color=C_BTN)
    signup = Button(rect=Rect(cx + 20, top_y, btn_w, btn_h),
                    label="SIGN UP  (register)", action="signup", color=C_BTN)
    back = Button(rect=Rect(cx - btn_w // 2, top_y + btn_h + 40, btn_w, btn_h),
                  label="BACK", action="back", color=C_BTN_BACK)
    state.buttons = [login, signup, back]

    mx, my = state.mouse
    for btn in state.buttons:
        draw_button(canvas, btn, hovered=btn.rect.contains(mx, my))

    draw_text_centered(canvas, "Shortcuts: L = Login   S = Sign Up   B/Esc = Back",
                       (CANVAS_W // 2, CANVAS_H - 50), scale=0.55, color=C_MUTED, thickness=1)
    return canvas


def handle_menu(state: KioskState, key: int) -> None:
    click = state.consume_click()
    action = None
    if click is not None:
        for btn in state.buttons:
            if btn.rect.contains(*click):
                action = btn.action
                break
    if key in (ord("l"), ord("L")):
        action = "login"
    elif key in (ord("s"), ord("S")):
        action = "signup"
    elif key in (ord("b"), ord("B"), 27):
        action = "back"

    if action == "back":
        state.screen = "grid"
        state.selected_slot = None
        return
    if action == "login":
        slot = state.selected_slot or 0
        user = state.assignments.get(slot)
        if user is None:
            show_result(state, "No user on this locker",
                        "Sign up first to register a face to this locker.",
                        C_DOOR_DENIED)
            return
        begin_login(state)
        return
    if action == "signup":
        slot = state.selected_slot or 0
        if slot in state.assignments:
            show_result(state, "Locker already registered",
                        f"Slot #{slot:02d} belongs to {state.assignments[slot]}.",
                        C_DOOR_DENIED)
            return
        begin_signup_name_entry(state)
        return


# ---------------------------------------------------------------------------
# Screen: TEXT INPUT (enrollment name)
# ---------------------------------------------------------------------------
def begin_signup_name_entry(state: KioskState) -> None:
    state.screen = "text_input"
    state.input_text = ""
    state.input_prompt = "Enter your name, then press ENTER"
    state.pending_action = "signup"
    state.clear_buttons()


def render_text_input(state: KioskState) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)
    draw_text(canvas, f"SIGN UP  —  Locker #{state.selected_slot:02d}", (24, 50),
              scale=0.9, color=C_ACCENT)
    draw_text_centered(canvas, state.input_prompt, (CANVAS_W // 2, 220),
                       scale=0.8, color=C_TEXT)
    # Input box
    box = Rect(CANVAS_W // 2 - 360, 270, 720, 90)
    cv2.rectangle(canvas, (box.x, box.y), (box.x + box.w, box.y + box.h), C_PANEL, -1)
    cv2.rectangle(canvas, (box.x, box.y), (box.x + box.w, box.y + box.h), C_ACCENT, 2)
    typed = state.input_text + ("_" if int(time.time() * 2) % 2 == 0 else " ")
    draw_text(canvas, typed, (box.x + 20, box.y + 60), scale=1.1, color=C_TEXT, thickness=2)

    draw_text_centered(canvas, "Type your name  |  ENTER = confirm  |  ESC/B = cancel  |  BACKSPACE = delete",
                       (CANVAS_W // 2, CANVAS_H - 60), scale=0.55, color=C_MUTED, thickness=1)
    return canvas


# (text-input key handling is performed inline in the main loop,
# since it needs the api_base to check for duplicate names)


# ---------------------------------------------------------------------------
# Screen: CAPTURE (enroll)  — auto face capture
# ---------------------------------------------------------------------------
def begin_enroll_capture(state: KioskState) -> None:
    state.screen = "capture_enroll"
    state.captured_frames = []
    state.stable_count = 0
    state.last_capture_ts = 0.0
    state.clear_buttons()


def render_capture(state: KioskState, frame: np.ndarray, bbox,
                   needed: int, progress: float, header: str,
                   subtitle: str = "") -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)
    draw_text(canvas, header, (24, 50), scale=0.9, color=C_ACCENT)

    # Large camera view centered
    view_w, view_h = 960, 540
    cam = cv2.resize(frame, (view_w, view_h))
    if bbox is not None:
        fx, fy, fw, fh = bbox
        sx = view_w / frame.shape[1]
        sy = view_h / frame.shape[0]
        x, y, w, h = int(fx * sx), int(fy * sy), int(fw * sx), int(fh * sy)
        color = (0, int(255 * progress), int(255 * (1 - progress)))
        cv2.rectangle(cam, (x, y), (x + w, y + h), color, 3)
    vx = (CANVAS_W - view_w) // 2
    vy = 90
    canvas[vy:vy + view_h, vx:vx + view_w] = cam
    cv2.rectangle(canvas, (vx, vy), (vx + view_w, vy + view_h), C_BORDER, 2)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = vx, vy + view_h + 20, view_w, 24
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), C_PANEL, -1)
    filled = int(bar_w * (len(state.captured_frames) / max(needed, 1)))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), C_DOOR_OPEN, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), C_BORDER, 2)
    draw_text_centered(canvas, f"Captured {len(state.captured_frames)}/{needed}",
                       (bar_x + bar_w // 2, bar_y + bar_h // 2 + 1),
                       scale=0.55, color=C_TEXT, thickness=2)

    if subtitle:
        draw_text_centered(canvas, subtitle, (CANVAS_W // 2, vy + view_h + 75),
                           scale=0.75, color=C_ACCENT, thickness=2)
    draw_text_centered(canvas, "Hold steady. B/Esc to cancel.",
                       (CANVAS_W // 2, CANVAS_H - 30), scale=0.55, color=C_MUTED, thickness=1)
    return canvas


def handle_capture_enroll(state: KioskState, frame: np.ndarray, key: int) -> bool:
    """Returns True when capture is complete and caller should enroll."""
    if key in (27, ord("b"), ord("B")):
        state.screen = "menu"
        return False
    bbox = detect_face(frame)
    now = time.time()
    if bbox is not None:
        state.stable_count += 1
        if (state.stable_count >= CAPTURE_STABLE_THRESHOLD
                and (now - state.last_capture_ts) >= CAPTURE_INTERVAL_SEC):
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                state.captured_frames.append(buf.tobytes())
                state.last_capture_ts = now
    else:
        state.stable_count = 0
    return len(state.captured_frames) >= ENROLL_PHOTO_COUNT


# ---------------------------------------------------------------------------
# Screen: CAPTURE (login)  — single confident frame
# ---------------------------------------------------------------------------
def begin_login(state: KioskState) -> None:
    state.screen = "capture_login"
    state.stable_count = 0
    state.captured_frames = []
    state.last_capture_ts = 0.0
    state.clear_buttons()


def handle_capture_login(state: KioskState, frame: np.ndarray, key: int) -> list[bytes] | None:
    """Capture N spaced frames, return them when enough are collected."""
    if key in (27, ord("b"), ord("B")):
        state.screen = "menu"
        return None
    bbox = detect_face(frame)
    now = time.time()
    if bbox is not None:
        state.stable_count += 1
        if (now - state.last_capture_ts) >= LOGIN_CAPTURE_INTERVAL:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                state.captured_frames.append(buf.tobytes())
                state.last_capture_ts = now
    else:
        state.stable_count = 0
    if len(state.captured_frames) >= LOGIN_FRAME_COUNT:
        return list(state.captured_frames)
    return None


# ---------------------------------------------------------------------------
# Screen: RESULT
# ---------------------------------------------------------------------------
def show_result(state: KioskState, title: str, detail: str, color: tuple) -> None:
    state.screen = "result"
    state.result_title = title
    state.result_detail = detail
    state.result_color = color
    state.result_until = time.time() + RESULT_DISPLAY_SECONDS
    state.clear_buttons()


def render_result(state: KioskState) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)
    # Coloured banner
    banner_h = 260
    cv2.rectangle(canvas, (0, 200), (CANVAS_W, 200 + banner_h), state.result_color, -1)
    draw_text_centered(canvas, state.result_title, (CANVAS_W // 2, 320),
                       scale=1.6, color=C_TEXT, thickness=4)
    draw_text_centered(canvas, state.result_detail, (CANVAS_W // 2, 400),
                       scale=0.75, color=C_TEXT, thickness=2)

    remaining = max(state.result_until - time.time(), 0.0)
    draw_text_centered(canvas, f"Returning to menu in {remaining:.1f}s  (press any key to continue)",
                       (CANVAS_W // 2, CANVAS_H - 60), scale=0.6, color=C_MUTED, thickness=1)

    back_btn = Button(rect=Rect(CANVAS_W // 2 - 140, CANVAS_H - 140, 280, 60),
                      label="BACK TO LOCKERS", action="back", color=C_BTN_BACK)
    state.buttons = [back_btn]
    mx, my = state.mouse
    draw_button(canvas, back_btn, hovered=back_btn.rect.contains(mx, my))
    return canvas


def handle_result(state: KioskState, key: int) -> None:
    click = state.consume_click()
    if key != -1 or click is not None or time.time() >= state.result_until:
        state.screen = "grid"
        state.selected_slot = None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_kiosk(api_base: str, capacity: int) -> None:
    state = KioskState(capacity=capacity)
    state.assignments = bootstrap_assignments_from_backend(api_base, capacity)

    # Stash api_base for handler that needs it
    setattr(state, "api_base", api_base)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WINDOW_TITLE, mouse_callback, state)

    print(f">> Kiosk running against {api_base}. Press Q to quit.")
    print(f">> Assignments: {state.assignments}")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            key = -1
        if key in (ord("q"), ord("Q")) and state.screen == "grid" and state.busy_thread is None:
            break

        # ----- Busy dispatch: thread is running, show spinner + await result -----
        if state.busy_thread is not None:
            canvas = render_busy_overlay(state, frame)
            if state.busy_result is not None:
                _handle_busy_complete(state, api_base)
            cv2.imshow(WINDOW_TITLE, canvas)
            continue

        if state.screen == "grid":
            canvas = render_grid(state)
            handle_grid(state, key)

        elif state.screen == "menu":
            canvas = render_menu(state)
            handle_menu(state, key)

        elif state.screen == "text_input":
            canvas = render_text_input(state)
            if key == 27:
                state.screen = "menu"
            elif key in (13, 10):
                name = state.input_text.strip()
                if len(name) < 2:
                    state.input_prompt = "Name must be at least 2 characters"
                elif name in state.assignments.values():
                    show_result(state, "Profile already exists",
                                f"'{name}' is bound to another locker.", C_DOOR_DENIED)
                else:
                    state.pending_name = name
                    # Thread the duplicate-name check so the UI stays live
                    start_busy(state, "name_check", "Checking name availability...",
                               lambda: (True, [n.lower() for n in api_list_users(api_base)]))
            elif key in (8, 127):
                state.input_text = state.input_text[:-1]
            elif 32 <= key < 127:
                if len(state.input_text) < 30:
                    state.input_text += chr(key)

        elif state.screen == "capture_enroll":
            bbox = detect_face(frame)
            progress = min(state.stable_count / CAPTURE_STABLE_THRESHOLD, 1.0) if bbox is not None else 0.0
            canvas = render_capture(state, frame, bbox, ENROLL_PHOTO_COUNT, progress,
                                    f"SIGN UP — capturing face for '{state.pending_name}'")
            done = handle_capture_enroll(state, frame, key)
            if done:
                # Kick off enroll on a worker thread
                _name = state.pending_name
                _frames = list(state.captured_frames)
                start_busy(state, "enroll",
                           f"Uploading {len(_frames)} photos & building embedding...",
                           lambda: api_enroll(api_base, _name, _frames))

        elif state.screen == "capture_login":
            bbox = detect_face(frame)
            progress = len(state.captured_frames) / LOGIN_FRAME_COUNT
            canvas = render_capture(
                state, frame, bbox, LOGIN_FRAME_COUNT, progress,
                f"LOGIN — Locker #{state.selected_slot:02d}",
                subtitle="BLINK or SLOWLY TURN YOUR HEAD — liveness check active",
            )
            frames_bundle = handle_capture_login(state, frame, key)
            if frames_bundle is not None:
                _frames = frames_bundle
                _slot = state.selected_slot or 0
                _name = state.assignments.get(_slot, "")
                _locker_id = f"L{_slot:03d}"
                start_busy(state, "login",
                           f"Verifying {len(_frames)} frames with liveness + anti-spoof...",
                           lambda: api_login(api_base, _name, _locker_id, _frames))

        elif state.screen == "result":
            canvas = render_result(state)
            handle_result(state, key)

        else:
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        cv2.imshow(WINDOW_TITLE, canvas)

    cap.release()
    cv2.destroyAllWindows()


def _handle_busy_complete(state: KioskState, api_base: str) -> None:
    """Drain a completed busy thread and transition to the appropriate screen."""
    result = state.busy_result
    kind = state.busy_kind
    state.busy_thread = None
    state.busy_result = None
    state.busy_kind = ""
    state.busy_label = ""

    if result is None:
        return
    ok, payload = result

    if kind == "name_check":
        if not ok:
            show_result(state, "Server error", str(payload), C_DOOR_DENIED)
            return
        existing = payload  # list of lowercased names
        name = state.pending_name
        if name.lower() in existing:
            show_result(state, "Profile already exists",
                        f"'{name}' is already enrolled.", C_DOOR_DENIED)
        else:
            begin_enroll_capture(state)
        return

    if kind == "enroll":
        if ok and state.selected_slot is not None:
            state.assignments[state.selected_slot] = state.pending_name
            save_assignments(state.assignments)
            show_result(state, "Registered!",
                        f"{state.pending_name} is now bound to Locker #{state.selected_slot:02d}.",
                        C_DOOR_OPEN)
        else:
            show_result(state, "Enrollment failed", str(payload), C_DOOR_DENIED)
        state.pending_name = ""
        return

    if kind == "login":
        if not ok:
            show_result(state, "Server error", str(payload), C_DOOR_DENIED)
            return
        data = payload
        granted = bool(data.get("access_granted"))
        name = data.get("user_name") or "Unknown"
        score = float(data.get("final_score") or 0.0)
        liveness = data.get("liveness") or {}
        liveness_passed = bool(liveness.get("passed", True))
        bound_user = state.assignments.get(state.selected_slot or 0)

        frame_results = data.get("frame_results") or []
        quality_ok = [f for f in frame_results if f.get("quality_passed")]
        spoof_hits = [f for f in quality_ok if not f.get("antispoof_passed", True)]
        spoof_dominant = (
            len(quality_ok) >= 3 and len(spoof_hits) >= max(2, len(quality_ok) // 2)
        )

        if spoof_dominant:
            show_result(state, "Spoof suspected",
                        f"Anti-spoof model rejected {len(spoof_hits)}/{len(quality_ok)} frames.",
                        C_DOOR_DENIED)
        elif not liveness_passed:
            reason = liveness.get("reason") or "No blink or head movement detected."
            show_result(state, "Spoof suspected",
                        f"Liveness failed: {reason}", C_DOOR_DENIED)
        elif granted and bound_user and name.lower() == bound_user.lower():
            show_result(state, "LOCKER OPEN",
                        f"Welcome, {name}. Locker #{state.selected_slot:02d} unlocked. (score {score:.2f})",
                        C_DOOR_OPEN)
        elif granted:
            show_result(state, "Wrong locker",
                        f"{name} is registered to a different locker.", C_DOOR_DENIED)
        else:
            prompt_obj = data.get("prompt") or {}
            prompt_msg = prompt_obj.get("message") if isinstance(prompt_obj, dict) else None
            show_result(state, "Access denied",
                        f"{prompt_msg or 'Face did not match.'} (score {score:.2f})",
                        C_DOOR_DENIED)
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Locker interactive kiosk")
    parser.add_argument("--api", default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument("--capacity", type=int, default=GRID_COLS * GRID_ROWS,
                        help=f"Number of lockers to show (default {GRID_COLS * GRID_ROWS})")
    args = parser.parse_args()
    run_kiosk(args.api, args.capacity)


if __name__ == "__main__":
    main()
