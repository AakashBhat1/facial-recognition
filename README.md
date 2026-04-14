# Smart Locker — AI-Powered Facial Recognition Access Control

Backend system for an AI-powered smart locker that uses face recognition, PIN fallback, anomaly detection, and anti-spoof checks to control physical locker access.

## Tech Stack

- **Python 3.12**, **FastAPI**, **SQLAlchemy**, **SQLite**
- **InsightFace** (RetinaFace + ArcFace) — 512-d face embeddings
- **PyTorch / torchvision** — custom `best.pt` face embedding backend
- **scikit-learn** — RandomForest anomaly detection
- **OpenCV** — webcam capture and image processing
- **ONNX Runtime** — legacy MiniFASNetV2 anti-spoof inference
- **Ultralytics YOLOv8** — custom `l_version_1_300.pt` real/fake anti-spoof detector

## Prerequisites

- Python 3.12+
- Webcam (for live demo)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd "facial recognition"

# Create virtual environment and install dependencies
uv venv
uv pip install -r backend/requirements.txt

# Or with pip
python -m venv .venv
.venv\Scripts\pip install -r backend/requirements.txt
```

### Environment

Copy the example env or create `backend/.env`:

```env
DATABASE_URL=sqlite:///./data/smart_locker.db
SECRET_KEY=change_this_to_a_random_secret
USE_BUFFALO_MODEL=true
USE_CUSTOM_FACE_MODEL=false
FACE_MODEL_NAME=buffalo_s
# CUSTOM_FACE_MODEL_NAME=your_custom_pack_name
# CUSTOM_FACE_MODEL_ROOT=path/to/insightface/models
# CUSTOM_PT_EMBEDDER_ENABLED=true
# CUSTOM_PT_CHECKPOINT_PATH=../face_model_scratch/models/best.pt
SIMILARITY_THRESHOLD=0.45
ML_SCORING_ENABLED=true
ML_MODEL_PATH=models_ml/anomaly_model.joblib
ML_ANOMALY_THRESHOLD=0.27
ANTISPOOF_ENABLED=true
ANTISPOOF_BACKEND=yolo               # "onnx" (legacy MiniFASNet) or "yolo" (custom real/fake)
ANTISPOOF_THRESHOLD=0.4
# ONNX backend (legacy)
ANTISPOOF_MODEL_PATH=models_ml/antispoof.onnx
# YOLO backend (custom trained real/fake detector)
ANTISPOOF_YOLO_MODEL_PATH=models_ml/l_version_1_300.pt
ANTISPOOF_YOLO_IMGSZ=640
ANTISPOOF_YOLO_CONF=0.25
MULTI_FRAME_MIN_REQUIRED=5
```

Face backend selection rules:
- Set exactly one of `USE_BUFFALO_MODEL` or `USE_CUSTOM_FACE_MODEL` to `true`.
- If `USE_CUSTOM_FACE_MODEL=true`, set `CUSTOM_FACE_MODEL_NAME` to your custom InsightFace pack name.
- Set `CUSTOM_PT_EMBEDDER_ENABLED=true` to extract embeddings from your `best.pt` checkpoint.
- If `CUSTOM_PT_EMBEDDER_ENABLED=true`, set `CUSTOM_PT_CHECKPOINT_PATH` to a valid checkpoint path.
- If both are `true` (or both `false`), backend startup fails fast with a clear config error.
- Buffalo remains the default path; rollback is an env switch pair:
  - `USE_BUFFALO_MODEL=true`
  - `USE_CUSTOM_FACE_MODEL=false`

Anti-spoof backend selection rules:
- `ANTISPOOF_ENABLED=false` disables spoof checks entirely — every face is treated as live.
- `ANTISPOOF_BACKEND=onnx` uses the legacy MiniFASNetV2 face-crop classifier (`models_ml/antispoof.onnx`).
- `ANTISPOOF_BACKEND=yolo` uses the custom YOLOv8 real/fake detector (`models_ml/l_version_1_300.pt`). The detector runs on the **full frame**, picks the detection that best overlaps the target face bbox by IoU, and maps `fake → spoof_score`, `real → 1 − conf`. It requires `ultralytics` to be installed.
- If the selected backend's model file is missing at startup, backend init fails fast with a clear config error (for YOLO) or a warning + skipped checks (for ONNX).
- `ANTISPOOF_THRESHOLD` applies to both backends — the face is flagged as spoof when `spoof_score ≥ threshold`.

Run with custom `best.pt` embedding (PowerShell example):

```powershell
$env:USE_BUFFALO_MODEL='false'
$env:USE_CUSTOM_FACE_MODEL='true'
$env:CUSTOM_FACE_MODEL_NAME='buffalo_s'
$env:CUSTOM_FACE_MODEL_ROOT='C:/Users/zewan/.insightface'
$env:CUSTOM_PT_EMBEDDER_ENABLED='true'
$env:CUSTOM_PT_CHECKPOINT_PATH='../face_model_scratch/models/best.pt'
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

### Download Models

```bash
# Anti-spoof ONNX model (legacy MiniFASNetV2 backend)
.venv\Scripts\python.exe backend\scripts\download_antispoof_model.py

# InsightFace model downloads automatically on first run
```

The custom YOLO anti-spoof model (`backend/models_ml/l_version_1_300.pt`) is committed to the repo. If you want to swap in your own checkpoint, point `ANTISPOOF_YOLO_MODEL_PATH` at it — any Ultralytics-compatible YOLOv8 detection weights with classes named `fake` and `real` will work.

## Running

### Start the Backend

```bash
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Wait for these log lines:
- `ML anomaly scoring is active`
- `Face backend active: buffalo (buffalo_s)` (or your custom backend/model)
- `Custom PT embedder enabled: yes` when checkpoint embedding is active

### API Docs

Open http://127.0.0.1:8000/docs for the interactive Swagger UI (disabled in production mode).

## Demo Scripts

All commands run from the **repo root**.

### Live Webcam — Enroll, Recognize, Kiosk

```bash
# Enroll a user (captures 7 frames via webcam)
.\.venv\Scripts\python.exe demo\live_demo.py enroll --name "YourName"

# Single recognition scan (captures 5 frames)
.\.venv\Scripts\python.exe demo\live_demo.py recognize

# Continuous kiosk mode (auto-scans, press Q to quit)
.\.venv\Scripts\python.exe demo\live_demo.py live
```

### Locker Kiosk GUI

```bash
# Interactive kiosk with locker grid, login/signup, webcam capture
.\.venv\Scripts\python.exe demo\locker_simulation.py
```

### Demo Dashboard (Operator Console)

```bash
# Health checks + live camera feed + latency metrics
.\.venv\Scripts\python.exe demo\demo_dashboard.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/enroll` | Enroll user with face images (multipart/form-data) |
| `PUT` | `/api/enroll/{id}/re-enroll` | Re-enroll existing user (multipart/form-data) |
| `POST` | `/api/auth/recognize-multi` | 1:1 multi-frame face verification — requires `user_id` query param |
| `PUT` | `/api/users/{id}/pin` | Set PIN for a user |
| `POST` | `/api/auth/pin` | PIN fallback authentication |
| `GET` | `/api/users` | List all enrolled users (id, name, assigned locker) |
| `DELETE` | `/api/users/{id}` | Remove a user and all their embeddings/logs/locker |
| `GET` | `/api/locker/status` | Locker state |
| `POST` | `/api/locker/close` | Close locker |

### `POST /api/enroll` — request shape

`multipart/form-data` with two fields:

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | form string | yes | Name of the person being enrolled |
| `images` | file list | yes | 5–10 JPEG/PNG frames captured from the device camera |

```bash
curl -X POST http://127.0.0.1:8000/api/enroll \
  -F "name=Aakash" \
  -F "images=@frame_0.jpg" -F "images=@frame_1.jpg" -F "images=@frame_2.jpg" \
  -F "images=@frame_3.jpg" -F "images=@frame_4.jpg"
```

The response includes the assigned `user_id` and `assigned_locker_id` — the same `user_id` must be supplied when calling `recognize-multi` later.

### `POST /api/auth/recognize-multi` — request shape

As of the latest refactor this endpoint is a **1:1 face verify**, not a 1:N search. The caller claims a `user_id`, proves it with face frames, and the backend restricts matching to that user's enrolled embeddings only.

Query parameters (not form fields):

| Param | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `user_id` | int | yes | — | The claimed user, obtained from `GET /api/users` |
| `locker_id` | string | no | user's assigned locker | Override the locker being unlocked |
| `check_liveness` | bool | no | `true` | Enable blink/head-movement liveness check |

Body: `multipart/form-data` with a single `images` field containing at least `MULTI_FRAME_MIN_REQUIRED` frames (default 5).

```bash
curl -X POST "http://127.0.0.1:8000/api/auth/recognize-multi?user_id=1&locker_id=L001&check_liveness=true" \
  -F "images=@f0.jpg" -F "images=@f1.jpg" -F "images=@f2.jpg" \
  -F "images=@f3.jpg" -F "images=@f4.jpg"
```

The typical client flow is:
1. `GET /api/users` → find the user whose `assigned_locker_id` matches the kiosk's locker.
2. Capture 5 frames from the camera.
3. `POST /api/auth/recognize-multi?user_id=<that user's id>` with those frames.

`demo/live_demo.py recognize` and the `live` kiosk mode already implement this flow — see `demo/live_demo.py:_find_user_for_locker`.

## Face Embedding & Similarity Search

The recognition pipeline converts faces into numerical vectors and matches them against stored profiles:

1. **Embedding extraction** — InsightFace ArcFace model maps each detected face to a **512-dimensional embedding vector**, a compact numerical fingerprint of the face's geometry.
2. **Storage** — Embeddings are encrypted with AES-256-GCM and stored in SQLite alongside the user profile. They are decrypted only at match time.
3. **Similarity scoring** — At recognition time the system computes **cosine similarity** between the live embedding and every stored embedding. Cosine similarity ranges from −1 (opposite) to 1 (identical).
4. **Threshold decision** — A match is accepted when similarity ≥ `SIMILARITY_THRESHOLD` (default **0.45**). The threshold is intentionally conservative to reduce false accepts on a shared locker.
5. **Multi-frame voting** — The `/api/auth/recognize-multi` endpoint captures **multiple frames** (default 5). Each frame is scored independently; the user is authenticated only if a **majority** of frames agree on the same identity above the threshold, making the system robust against single-frame noise or partial occlusion.

## Security Features

- **Rate limiting** — 10 req/60s per IP, 30s cooldown after 5 failures
- **PIN fallback** — bcrypt-hashed PINs
- **Embedding encryption** — AES-256-GCM at rest (optional via `EMBEDDING_ENCRYPTION_ENABLED`)
- **Anti-spoof detection** — pluggable backend: legacy MiniFASNetV2 ONNX **or** custom YOLOv8 real/fake detector, selected via `ANTISPOOF_BACKEND`
- **1:1 verify** — `recognize-multi` only compares frames against the claimed user's embeddings, making cross-user confusion impossible
- **Quality filtering** — blur, brightness, face size, pose angle
- **Liveness detection** — blink + head movement analysis
- **CORS lockdown** — backend only accepts localhost origins (single-instance kiosk deployment)

## Anomaly Detection

**Layer 1 — Rule-Based:**
- Brute force (3 failures in 60s)
- Off-hours access (10 PM - 6 AM)
- Rapid repeated access (5 in 10 min)
- Repeated unknown faces (3 in 5 min)

**Layer 2 — Machine Learning:**
- RandomForest classifier (300 trees)
- 14 behavioral features
- Auto-retrain from production logs: `python backend/scripts/retrain_model.py`

## Project Structure

```
backend/
  api/            # FastAPI route handlers
  middleware/     # Rate limiter, operator auth, error handler
  models/         # SQLAlchemy models and Pydantic schemas
  models_ml/      # Trained ML models (.joblib, .onnx, .pt)
  scripts/        # Training, bootstrapping, model download
  services/       # Business logic (face pipeline, enrollment, anomaly, etc.)
  config.py       # All configuration from .env
  main.py         # FastAPI app entrypoint
demo/
  live_demo.py          # Webcam enroll/recognize/kiosk
  locker_simulation.py  # Interactive kiosk GUI
  demo_dashboard.py     # Operator dashboard
  run_demo.py           # Scripted demo runner
docs/                   # Project documentation and blueprints
```

## Project Working Session Docs

Living context for whoever picks the project up next lives in `docs/project_working_session/`:

- `REPO_CONTEXT.md` — current architecture, backend switches, and key modules
- `CURRENT_STEP.md` — what is being worked on right now
- `changes.md` — running changelog of the session

Update these after any non-trivial code change (see `.codex/skills/repo-doc-pass/SKILL.md`).
