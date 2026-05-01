> **Links:** [[../Welcome]] | [[Style DNA]] | [[Welcome]] | [[changes]]
> Last updated: 2026-05-01

# Smart Locker — AI-Powered Facial Recognition Access Control

Backend system for an AI-powered smart locker that uses face recognition, PIN fallback, anomaly detection, and anti-spoof checks to control physical locker access.

---

## 🚀 Quick Start — Run & Test

> Run from the **repo root**: `C:\Users\zewan\OneDrive\Documents\Claude\Projects\facial recognition`

### 1️⃣ Start the backend (Terminal A)

```powershell
cd backend
.venv\Scripts\activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Wait for: `Smart Locker API v2.0 is running.`

Sanity check:

```powershell
curl http://127.0.0.1:8000/api/health
```

### 2️⃣ Enroll yourself (Terminal B)

```powershell
backend\.venv\Scripts\python.exe demo\live_demo.py enroll --name "YourName"
```

Captures 7 webcam frames. First user gets locker `L001` automatically.

### 3️⃣ Recognize (Terminal B)

```powershell
backend\.venv\Scripts\python.exe demo\live_demo.py recognize
```

Look at the camera and **blink once** during the 5-frame capture. Expected: `ACCESS GRANTED — Welcome, YourName! (score: 0.6+)`.

### 4️⃣ Other demos

```powershell
# Continuous kiosk mode (auto-scans, press Q to quit)
backend\.venv\Scripts\python.exe demo\live_demo.py live

# Interactive locker grid GUI
backend\.venv\Scripts\python.exe demo\locker_simulation.py

# Operator dashboard with health checks + camera feed + latency
backend\.venv\Scripts\python.exe demo\demo_dashboard.py
```

### 🔄 Reset everything (start clean)

```powershell
# Stop the backend (Ctrl+C in Terminal A), then:
del backend\data\smart_locker.db
# Restart the backend — it recreates an empty schema on startup
```

### ⚙️ Tuning notes for laptop webcams

If `recognize` denies with `score: 0.000` and frames fail for `blur`, your `backend/.env` should have:

```env
QUALITY_BLUR_THRESHOLD=40         # default 100 is too strict for webcams
MULTI_FRAME_MIN_REQUIRED=3        # default 3 — don't raise to 5 unless ANTISPOOF_ENABLED=true
```

---

## Tech Stack

- **Python 3.12**, **FastAPI**, **SQLAlchemy**, **SQLite**
- **InsightFace** (RetinaFace + ArcFace) — 512-d face embeddings
- **PyTorch / torchvision** — custom `best.pt` face embedding backend
- **scikit-learn** — RandomForest anomaly detection
- **OpenCV** — webcam capture and image processing (use `opencv-python`, not `opencv-python-headless`, on the demo client — `imshow` requires GUI support)
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
ANTISPOOF_ENABLED=false
ANTISPOOF_BACKEND=yolo               # "onnx" (legacy MiniFASNet) or "yolo" (custom real/fake)
ANTISPOOF_THRESHOLD=0.4
# ONNX backend (legacy)
ANTISPOOF_MODEL_PATH=models_ml/antispoof.onnx
# YOLO backend (custom trained real/fake detector)
ANTISPOOF_YOLO_MODEL_PATH=models_ml/l_version_1_300.pt
ANTISPOOF_YOLO_IMGSZ=640
ANTISPOOF_YOLO_CONF=0.25
# Webcam-friendly tuning (loosen for low-quality cameras)
QUALITY_BLUR_THRESHOLD=40
MULTI_FRAME_MIN_REQUIRED=3
```

Face backend selection rules:
- Set exactly one of `USE_BUFFALO_MODEL` or `USE_CUSTOM_FACE_MODEL` to `true`.
- If `USE_CUSTOM_FACE_MODEL=true`, set `CUSTOM_FACE_MODEL_NAME` to your custom InsightFace pack name.
- Set `CUSTOM_PT_EMBEDDER_ENABLED=true` to extract embeddings from your `best.pt` checkpoint.
- If `CUSTOM_PT_EMBEDDER_ENABLED=true`, set `CUSTOM_PT_CHECKPOINT_PATH` to a valid checkpoint path.
- If both are `true` (or both `false`), backend startup fails fast with a clear config error.

Anti-spoof backend selection rules:
- `ANTISPOOF_ENABLED=false` disables spoof checks entirely — every face is treated as live.
- `ANTISPOOF_BACKEND=onnx` uses the legacy MiniFASNetV2 face-crop classifier (`models_ml/antispoof.onnx`).
- `ANTISPOOF_BACKEND=yolo` uses the custom YOLOv8 real/fake detector (`models_ml/l_version_1_300.pt`).
- `ANTISPOOF_THRESHOLD` applies to both backends — the face is flagged as spoof when `spoof_score ≥ threshold`.

Run with custom `best.pt` embedding (PowerShell):

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

## API Docs

Open http://127.0.0.1:8000/docs for the interactive Swagger UI (disabled in production mode).

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

### `POST /api/enroll`

`multipart/form-data` with `name` (form string) and `images` (5–10 JPEG/PNG files).

```bash
curl -X POST http://127.0.0.1:8000/api/enroll \
  -F "name=Aakash" \
  -F "images=@frame_0.jpg" -F "images=@frame_1.jpg" -F "images=@frame_2.jpg" \
  -F "images=@frame_3.jpg" -F "images=@frame_4.jpg"
```

### `POST /api/auth/recognize-multi`

1:1 face verify. Caller claims `user_id` and proves it with face frames.

| Param | Type | Required | Default |
|-------|------|----------|---------|
| `user_id` | int (query) | yes | — |
| `locker_id` | string (query) | no | user's assigned locker |
| `check_liveness` | bool (query) | no | `true` |

Body: `multipart/form-data` with `images` field, ≥ `MULTI_FRAME_MIN_REQUIRED` frames.

```bash
curl -X POST "http://127.0.0.1:8000/api/auth/recognize-multi?user_id=1&locker_id=L001&check_liveness=true" \
  -F "images=@f0.jpg" -F "images=@f1.jpg" -F "images=@f2.jpg" \
  -F "images=@f3.jpg" -F "images=@f4.jpg"
```

## Face Embedding & Similarity Search

1. **Embedding extraction** — InsightFace ArcFace model maps each detected face to a **512-d embedding**.
2. **Storage** — Embeddings are encrypted with AES-256-GCM and stored in SQLite.
3. **Similarity scoring** — Cosine similarity between live and stored embeddings.
4. **Threshold decision** — Match accepted when similarity ≥ `SIMILARITY_THRESHOLD` (default **0.45**).
5. **Multi-frame voting** — `/api/auth/recognize-multi` aggregates per-frame scores via top-K median; recognition succeeds when the aggregate ≥ threshold and liveness passes.

## Security Features

- **Rate limiting** — 10 req/60s per IP, 30s cooldown after 5 failures
- **PIN fallback** — bcrypt-hashed PINs
- **Embedding encryption** — AES-256-GCM at rest (optional via `EMBEDDING_ENCRYPTION_ENABLED`)
- **Anti-spoof detection** — MiniFASNetV2 ONNX or YOLOv8 real/fake detector
- **1:1 verify** — `recognize-multi` only compares against the claimed user's embeddings
- **Quality filtering** — blur, brightness, face size, pose angle
- **Liveness detection** — blink (EAR over InsightFace 106-pt eye landmarks 33-42 + 87-95) + head movement
- **CORS lockdown** — backend only accepts localhost origins

## Anomaly Detection

**Layer 1 — Rule-Based:**
- Brute force (3 failures in 60s)
- Off-hours access (10 PM - 6 AM)
- Rapid repeated access (5 in 10 min)
- Repeated unknown faces (3 in 5 min)

**Layer 2 — Machine Learning:**
- RandomForest classifier (300 trees), 14 behavioral features
- Auto-retrain: `python backend/scripts/retrain_model.py`

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
docs/                   # Project documentation and blueprints
```

## Project Working Session Docs

Living context for whoever picks the project up next lives in the second_brain vault at `C:\dev\second_brain\backend\`:

- `changes.md` — running changelog of the session
- `REPO_CONTEXT.md` — current architecture, backend switches, key modules (when present)
- `CURRENT_STEP.md` — what is being worked on right now (when present)

Update these after any non-trivial code change.
