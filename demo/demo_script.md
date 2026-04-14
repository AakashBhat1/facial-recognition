# Smart Locker Demo Script

Last updated: 2026-04-04

## Goal
Run the demo in two modes:
1. **Backend API demo** — 3 automated scenarios (grant, deny, anomaly) via `demo/run_demo.py`
2. **Live face recognition demo** — real webcam enrollment + unlock via `backend/scripts/live_demo.py`

## Startup
From the repo root, start the backend:

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Wait for `GET /api/health` to return `{"status":"ok"}`.

Expected startup logs:
- "ML anomaly scoring is active" (model loaded from `models_ml/anomaly_model.joblib`, threshold=0.27)
- InsightFace model preloaded (buffalo_l, ~300MB first time)

---

## Part 1: Backend API Demo

### Main Demo Command
```powershell
.venv\Scripts\python.exe demo\run_demo.py --base-url http://127.0.0.1:8000 --scenario all --seed 42
```

Shows:
- `grant`: temp user created, recognized, logged as `SUCCESS`
- `deny`: unknown face denied, logged as `FAILURE`
- `anomaly`: rapid repeated access triggers `RAPID_ACCESS` alert + ML anomaly scoring

### Verbose Mode (show JSON)
```powershell
.venv\Scripts\python.exe demo\run_demo.py --base-url http://127.0.0.1:8000 --scenario all --seed 42 --verbose
```

### Keep temp user for inspection
```powershell
.venv\Scripts\python.exe demo\run_demo.py --base-url http://127.0.0.1:8000 --scenario grant --seed 42 --no-cleanup
```

---

## Part 2: Live Face Recognition Demo

### Enroll a user (auto-capture 10 photos)
```powershell
cd backend
..\\.venv\Scripts\python.exe scripts\live_demo.py enroll --name "Aakash"
```
- Camera opens, auto-detects face, captures 10 photos
- Sends to `/api/enroll` — stores 512-d embeddings + centroid

### Recognize (single scan)
```powershell
..\\.venv\Scripts\python.exe scripts\live_demo.py recognize
```
- Camera opens, detects face, sends to `/api/auth/recognize-image`
- Shows ACCESS GRANTED / DENIED with similarity score

### Kiosk Mode (continuous scanning)
```powershell
..\\.venv\Scripts\python.exe scripts\live_demo.py live
```
- Continuous auto-scan, 3s cooldown between scans
- Press `Q` on the window to quit

---

## Live Narration Guide

**Scenario 1: Authorized access**
- Backend registers a user with face embeddings (enrollment)
- Same person's face is recognized with similarity score
- `access_granted=true`, `locker_action=OPEN`

**Scenario 2: Unauthorized access**
- Unknown face has no matching embeddings
- `access_granted=false`, `locker_action=ACCESS_DENIED`

**Scenario 3: Anomaly alert**
- Rapid repeated access triggers rule-based `RAPID_ACCESS` alert
- ML anomaly scoring runs in parallel (RandomForest, threshold=0.27)
- Show alert in `/api/alerts`

**Scenario 4 (optional): Live webcam**
- Enroll yourself with `live_demo.py enroll --name "Your Name"`
- Unlock with `live_demo.py recognize`
- Show kiosk mode with `live_demo.py live`

---

## Rate-Limit Note
`POST /api/auth/recognize` is rate-limited to 10 requests per 60 seconds per IP.

One full `--scenario all` run sends 7 recognition requests — safe for a single run.

If you immediately rerun, wait 60 seconds or restart the backend.

## Fallback Sequence
If the all-in-one demo fails, run scenarios separately:

```powershell
.venv\Scripts\python.exe demo\run_demo.py --base-url http://127.0.0.1:8000 --scenario grant --seed 42
.venv\Scripts\python.exe demo\run_demo.py --base-url http://127.0.0.1:8000 --scenario deny --seed 43
.venv\Scripts\python.exe demo\run_demo.py --base-url http://127.0.0.1:8000 --scenario anomaly --seed 44
```
