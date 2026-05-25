# Smart Locker — Full Project Documentation

> **AI-Powered Facial Recognition Access Control System**
>
> Version **2.0.0** · Python 3.12 · FastAPI · InsightFace · SQLite

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Tech Stack](#4-tech-stack)
5. [Configuration Reference](#5-configuration-reference)
6. [Data Models](#6-data-models)
7. [API Reference](#7-api-reference)
8. [Service Layer](#8-service-layer)
9. [Recognition Pipeline](#9-recognition-pipeline)
10. [Security Design](#10-security-design)
11. [Anomaly Detection](#11-anomaly-detection)
12. [Demo Applications](#12-demo-applications)
13. [Deployment](#13-deployment)
14. [Development Guide](#14-development-guide)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Overview

Smart Locker is a backend system for an AI-powered physical locker that uses **facial recognition** as the primary authentication mechanism, with **PIN fallback** for accessibility. The system is designed for **single-instance, on-device deployment** — the backend and kiosk UI run on the same hardware (e.g. a Raspberry Pi or edge device).

### Key Capabilities

| Capability | Description |
|---|---|
| **Face Enrollment** | Multi-image enrollment with augmentation, centroid computation, and per-user threshold calibration |
| **1:1 Face Verification** | Caller claims a `user_id` and proves identity with webcam frames |
| **Multi-Frame Voting** | Top-K median aggregation across 3–10 frames for robust matching |
| **Liveness Detection** | Blink (EAR-based) and head-movement checks to reject photos/videos |
| **Anti-Spoof Detection** | Pluggable backends — ONNX (MiniFASNetV2 face crop) or YOLOv8 (full-frame fake/real) |
| **PIN Fallback** | bcrypt-hashed 4–8 digit PINs with rate limiting |
| **Anomaly Detection** | Rule-based (brute force, off-hours, rapid access, repeated unknown) + ML (Isolation Forest) |
| **Embedding Encryption** | Optional AES-256-GCM encryption of stored face embeddings |
| **Quality Filtering** | Blur, brightness, face size, yaw, and pitch checks before embedding extraction |
| **Locker Control** | Per-user locker assignment with LOCK/UNLOCK state management |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐   │
│  │ live_demo.py │  │ locker_sim.py    │  │ Swagger UI    │   │
│  │ (CLI + cam)  │  │ (GUI kiosk)      │  │ /docs         │   │
│  └──────┬───────┘  └───────┬──────────┘  └───────┬───────┘   │
│         │                  │                     │           │
│         └──────────────────┼─────────────────────┘           │
│                            │ HTTP (localhost:8000)            │
├────────────────────────────┼─────────────────────────────────┤
│                     FastAPI Backend                           │
│  ┌─────────────────────────┴───────────────────────────────┐ │
│  │                    API Layer (api/)                      │ │
│  │  enroll.py · recognize_multi.py · pin_auth.py           │ │
│  │  users.py · locker.py                                   │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │                 Middleware Layer (middleware/)            │ │
│  │  rate_limiter.py · error_handler.py                     │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │                 Service Layer (services/)                │ │
│  │  face_pipeline · multi_frame_recognizer · enrollment    │ │
│  │  face_comparator · quality_filter · liveness_detector   │ │
│  │  antispoof_detector · anomaly_detector · embedding_crypto│ │
│  │  ml_anomaly_service · locker_controller · logger_service│ │
│  │  model_manager · version_service · threshold_service    │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │                Data Layer (models/)                      │ │
│  │  database.py (SQLAlchemy ORM)                           │ │
│  │  schemas.py  (Pydantic request/response models)         │ │
│  ├─────────────────────────────────────────────────────────┤ │
│  │                 ML Models (models_ml/)                   │ │
│  │  anomaly_model.joblib · antispoof.onnx                  │ │
│  │  l_version_1_300.pt (YOLOv8)                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                   SQLite (data/smart_locker.db)               │
└──────────────────────────────────────────────────────────────┘
```

### Design Principles

- **Single-instance**: Backend + kiosk run on the same device; CORS locked to `localhost`.
- **Layered**: API → Service → Data. No business logic in route handlers.
- **Fail-safe**: Every gate (quality, liveness, anti-spoof, anomaly) defaults to pass-through when its model is unavailable.
- **Configuration-driven**: Every threshold, toggle, and path is controlled via `.env`.
- **Docs hidden in production**: Swagger/ReDoc/OpenAPI endpoints are suppressed when `ENV=production`.

---

## 3. Repository Structure

```
.
├── backend/
│   ├── api/                      # FastAPI route handlers
│   │   ├── __init__.py
│   │   ├── enroll.py             # POST /api/enroll, PUT /api/enroll/{id}/re-enroll
│   │   ├── recognize_multi.py    # POST /api/auth/recognize-multi
│   │   ├── pin_auth.py           # POST /api/auth/pin, PUT /api/users/{id}/pin
│   │   ├── users.py              # GET /api/users, DELETE /api/users/{id}
│   │   └── locker.py             # GET /api/locker/status
│   │
│   ├── middleware/               # Request processing middleware
│   │   ├── __init__.py
│   │   ├── error_handler.py      # Standardized error envelope
│   │   └── rate_limiter.py       # Sliding-window + cooldown rate limiting
│   │
│   ├── models/                   # Data layer
│   │   ├── __init__.py
│   │   ├── database.py           # SQLAlchemy ORM models, init_db(), migrations
│   │   └── schemas.py            # Pydantic request/response schemas
│   │
│   ├── models_ml/                # ML model binaries (gitignored)
│   │   ├── anomaly_model.joblib  # Isolation Forest for behavioral anomaly
│   │   ├── antispoof.onnx        # MiniFASNetV2 anti-spoof
│   │   ├── l_version_1_300.pt    # YOLOv8 real/fake classifier (~84 MB)
│   │   └── training_metrics.json # Recommended threshold from last training
│   │
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py   # Rule-based + ML anomaly checks
│   │   ├── antispoof_detector.py # ONNX/YOLO anti-spoof backends
│   │   ├── custom_pt_embedder.py # Custom PyTorch checkpoint embedder
│   │   ├── embedding_anomaly.py  # Embedding-space Isolation Forest
│   │   ├── embedding_crypto.py   # AES-256-GCM encrypt/decrypt for embeddings
│   │   ├── enrollment_service.py # Multi-image enrollment + re-enrollment
│   │   ├── face_comparator.py    # Cosine similarity, weighted scoring, multi-user margin
│   │   ├── face_pipeline.py      # InsightFace wrapper (detect, align, embed)
│   │   ├── feature_extractor.py  # Feature extraction for ML anomaly model
│   │   ├── image_augmentation.py # Brightness/rotation augmentation for enrollment
│   │   ├── liveness_detector.py  # Blink (EAR) + head movement liveness
│   │   ├── locker_controller.py  # Lock/unlock state machine
│   │   ├── logger_service.py     # JSONL access + security log writer
│   │   ├── ml_anomaly_service.py # Isolation Forest model loader + scorer
│   │   ├── ml_model_scoring.py   # Scoring pipeline for ML anomaly model
│   │   ├── model_manager.py      # InsightFace model singleton manager
│   │   ├── multi_frame_recognizer.py # Multi-frame recognition orchestrator
│   │   ├── quality_filter.py     # Blur, brightness, face size, pose checks
│   │   ├── threshold_service.py  # Per-user personal threshold computation
│   │   ├── user_prompt.py        # Adaptive rejection message generator
│   │   └── version_service.py    # Embedding model version stamping
│   │
│   ├── data/                     # Runtime data (gitignored)
│   │   └── smart_locker.db       # SQLite database
│   │
│   ├── config.py                 # Centralized configuration (reads .env)
│   ├── main.py                   # FastAPI app entrypoint
│   ├── requirements.txt          # Pip dependencies (pinned)
│   └── .env                      # Environment variables (gitignored)
│
├── demo/
│   ├── live_demo.py              # CLI webcam enroll/recognize/live-kiosk
│   ├── locker_simulation.py      # OpenCV GUI locker simulation (1280×720)
│   └── locker_assignments.json   # Persistent locker→user mapping for sim
│
├── mock/
│   └── facereq/                  # Mock HTTP request scripts for testing
│
├── face_model_scratch/           # Custom model training workspace (gitignored)
│   └── models/                   # Checkpoints (best.pt)
│
├── logs/                         # Runtime logs (gitignored)
│   ├── access.jsonl              # JSON Lines access log
│   └── security.log              # Human-readable security log
│
├── Dockerfile                    # CPU-only production image
├── .dockerignore                 # Docker context exclusions
├── .gitignore                    # Git exclusions
├── pyproject.toml                # Project metadata + uv dependencies
├── uv.lock                       # uv lockfile
├── README.md                     # Quick-start guide
├── HOW_TO_RUN.md                 # Step-by-step beginner guide
└── CLAUDE.md                     # AI assistant project instructions
```

---

## 4. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Web Framework** | FastAPI 0.115 | Async REST API with OpenAPI docs |
| **ASGI Server** | Uvicorn 0.30 | Production-grade HTTP server |
| **ORM** | SQLAlchemy 2.0 | Database abstraction |
| **Database** | SQLite | Single-file, zero-config persistence |
| **Validation** | Pydantic 2.9 | Request/response schema enforcement |
| **Face Detection** | InsightFace (RetinaFace) | Multi-face detection with bounding boxes |
| **Face Embedding** | InsightFace (ArcFace/MobileFaceNet) | 512-dimensional face embeddings |
| **Anti-Spoof (ONNX)** | MiniFASNetV2 | Face-crop spoof probability |
| **Anti-Spoof (YOLO)** | Ultralytics YOLOv8 | Full-frame real/fake detection |
| **Deep Learning** | PyTorch 2.11 / torchvision 0.26 | Custom model inference |
| **ONNX** | onnxruntime ≥1.17 | ONNX model inference |
| **Computer Vision** | OpenCV ≥4.9 | Image decode, Laplacian blur, drawing |
| **ML** | scikit-learn 1.5, pandas, joblib | Anomaly detection (Isolation Forest) |
| **Encryption** | cryptography ≥42 | AES-256-GCM embedding encryption |
| **PIN Hashing** | bcrypt 5.0 | 12-round bcrypt for PIN storage |
| **HTTP Client** | httpx 0.27 | Demo → backend communication |
| **Config** | python-dotenv 1.0 | `.env` file loading |
| **Testing** | pytest 8.3, pytest-asyncio | Unit/integration tests |

---

## 5. Configuration Reference

All configuration is via `backend/.env`. Defaults are built into `config.py`.

### Core Settings

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/smart_locker.db` | SQLAlchemy DB URL |
| `SECRET_KEY` | `dev_secret` | Used for encryption key derivation |
| `ENV` | `development` | `production` hides /docs, /redoc, /openapi.json |
| `LOG_DIR` | `../logs` | Directory for access.jsonl and security.log |
| `OPERATOR_PIN_HASH` | *(empty)* | bcrypt hash for admin endpoints (production) |

### Face Recognition

| Variable | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.45` | Cosine similarity threshold for a match (tuned via LFW — EER=0.11, FAR=0.0%) |
| `EMBEDDING_DIM` | `512` | Expected embedding dimensionality |
| `CENTROID_WEIGHT` | `0.7` | Weight of centroid similarity in weighted scoring |
| `MAX_SCORE_WEIGHT` | `0.3` | Weight of best individual similarity |
| `MIN_ENROLLMENT_IMAGES` | `5` | Minimum images for enrollment |
| `MAX_ENROLLMENT_IMAGES` | `10` | Maximum images for enrollment |
| `FACE_DETECTION_THRESHOLD` | `0.5` | InsightFace detection confidence threshold |
| `FACE_DET_SIZE` | `320` | Detection input resolution (lower = faster) |

### Face Backend Selection

| Variable | Default | Description |
|---|---|---|
| `USE_BUFFALO_MODEL` | `true` | Use InsightFace buffalo_s model |
| `USE_CUSTOM_FACE_MODEL` | `false` | Use a custom InsightFace model pack |
| `FACE_MODEL_NAME` | `buffalo_s` | InsightFace model name (~16 MB, MobileFaceNet backbone) |
| `CUSTOM_FACE_MODEL_NAME` | *(empty)* | Custom model pack name (required when `USE_CUSTOM_FACE_MODEL=true`) |
| `CUSTOM_FACE_MODEL_ROOT` | *(empty)* | Root directory for custom model packs |
| `CUSTOM_PT_EMBEDDER_ENABLED` | `false` | Use a custom PyTorch checkpoint for embedding extraction |
| `CUSTOM_PT_CHECKPOINT_PATH` | `../face_model_scratch/models/best.pt` | Path to custom `.pt` checkpoint |

> **Exactly one** of `USE_BUFFALO_MODEL` or `USE_CUSTOM_FACE_MODEL` must be `true`. Setting both to the same value is a startup error.

### Quality Filtering

| Variable | Default | Description |
|---|---|---|
| `QUALITY_BLUR_THRESHOLD` | `100.0` | Minimum Laplacian variance (lower = more blur allowed) |
| `QUALITY_BRIGHTNESS_MIN` | `50.0` | Minimum mean pixel intensity |
| `QUALITY_BRIGHTNESS_MAX` | `200.0` | Maximum mean pixel intensity |
| `QUALITY_MIN_FACE_WIDTH` | `80` px | Minimum face bounding box width |
| `QUALITY_MAX_YAW` | `30.0°` | Maximum horizontal head rotation |
| `QUALITY_MAX_PITCH` | `25.0°` | Maximum vertical head rotation |

### Multi-Frame Recognition

| Variable | Default | Description |
|---|---|---|
| `MULTI_FRAME_MIN_REQUIRED` | `3` | Minimum frames that must pass all gates |
| `MULTI_FRAME_TOP_K` | `3` | Number of top scores used in median aggregation |
| `DETECTION_INTERVAL_FRAMES` | `3` | Frames between detection passes |

### Liveness Detection

| Variable | Default | Description |
|---|---|---|
| `LIVENESS_EAR_THRESHOLD` | `0.21` | Eye Aspect Ratio threshold for blink detection |
| `LIVENESS_EAR_CONSEC_FRAMES` | `2` | Consecutive below-threshold frames to register a blink |
| `LIVENESS_HEAD_MOVEMENT_THRESHOLD` | `5.0` px | Mean landmark displacement to detect head movement |

### Anti-Spoof Detection

| Variable | Default | Description |
|---|---|---|
| `ANTISPOOF_ENABLED` | `false` | Master toggle for anti-spoof checks |
| `ANTISPOOF_BACKEND` | `onnx` | `onnx` (MiniFASNetV2 face crop) or `yolo` (YOLOv8 full-frame) |
| `ANTISPOOF_THRESHOLD` | `0.5` | Score ≥ this → spoof |
| `ANTISPOOF_MODEL_PATH` | `models_ml/antispoof.onnx` | Path to ONNX model |
| `ANTISPOOF_YOLO_MODEL_PATH` | `models_ml/l_version_1_300.pt` | Path to YOLOv8 weights |
| `ANTISPOOF_YOLO_IMGSZ` | `640` | YOLO input resolution |
| `ANTISPOOF_YOLO_CONF` | `0.25` | YOLO minimum detection confidence |

### Anomaly Detection

| Variable | Default | Description |
|---|---|---|
| `BRUTE_FORCE_LIMIT` | `3` | Failed attempts before brute force alert |
| `BRUTE_FORCE_WINDOW_SECONDS` | `60` | Window for brute force counting |
| `OFF_HOURS_START` | `22` (10 PM) | Off-hours window start |
| `OFF_HOURS_END` | `6` (6 AM) | Off-hours window end |
| `RAPID_ACCESS_LIMIT` | `5` | Successful accesses before rapid-access alert |
| `RAPID_ACCESS_WINDOW_MINUTES` | `10` | Window for rapid-access counting |
| `ML_SCORING_ENABLED` | `true` | Enable ML anomaly scoring |
| `ML_MODEL_PATH` | `models_ml/anomaly_model.joblib` | Path to Isolation Forest model |
| `ML_ANOMALY_THRESHOLD` | *(from training_metrics.json or 0.0)* | Score ≥ this → anomalous |
| `ML_CONTAMINATION` | `0.05` | Isolation Forest contamination parameter |

### Rate Limiting

| Variable | Default | Description |
|---|---|---|
| `RATE_LIMIT_RECOGNIZE_MAX` | `10` | Max recognition requests per window per IP |
| `RATE_LIMIT_RECOGNIZE_WINDOW_SECONDS` | `60` | Sliding window size |
| `RATE_LIMIT_COOLDOWN_SECONDS` | `30` | Cooldown after failure threshold |
| `RATE_LIMIT_FAILURE_TRIGGER` | `5` | Failures before cooldown |

### Embedding Security

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_ENCRYPTION_ENABLED` | `false` | Encrypt embeddings at rest with AES-256-GCM |
| `EMBEDDING_ENCRYPTION_KEY` | *(empty — falls back to SECRET_KEY)* | Key material (SHA-256 hashed to 256 bits) |
| `EMBEDDING_MODEL_VERSION` | `buffalo_s_v1` | Version tag stored on each embedding row |

### Multi-User & Thresholds

| Variable | Default | Description |
|---|---|---|
| `MULTI_USER_MARGIN` | `0.08` | Minimum score gap between top-2 users to accept (ambiguity rejection) |
| `PERSONAL_THRESHOLD_K` | `1.5` | Standard deviations below centroid mean for personal threshold |
| `EMBEDDING_DRIFT_CAP` | `50` | Max embeddings before oldest are pruned |

---

## 6. Data Models

### SQLAlchemy ORM Tables

#### `users`

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `name` | String | User display name |
| `embedding` | Text | JSON centroid embedding (legacy compat) |
| `created_at` | DateTime | UTC |
| `personal_threshold` | Float (nullable) | Per-user similarity threshold |
| `pin_hash` | String (nullable) | bcrypt hash of PIN |
| `assigned_locker_id` | String (nullable) | e.g. `L001` |

#### `face_embeddings`

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `user_id` | Integer FK | References `users.id` |
| `embedding` | Text | JSON float array or AES-256-GCM base64 blob |
| `is_centroid` | Boolean | True for the centroid row |
| `model_name` | String | `arcface`, `mobilefacenet` |
| `model_version` | String | e.g. `buffalo_s_v1` |
| `created_at` | DateTime | UTC |

#### `access_logs`

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | |
| `user_id` | Integer (nullable) | Null for unknown faces |
| `user_name` | String | `UNKNOWN` if unrecognized |
| `action` | String | `OPEN`, `CLOSE`, `ACCESS_DENIED` |
| `result` | String | `SUCCESS`, `FAILURE` |
| `confidence_score` | Float | Score minus threshold |
| `similarity_score` | Float | Raw cosine similarity |
| `locker_id` | String | e.g. `L001` |
| `ip_address` | String | Client IP |
| `anomaly_flag` | Boolean | True if any anomaly rule fired |
| `ml_anomaly_score` | Float (nullable) | Isolation Forest score |
| `timestamp` | DateTime | UTC |

#### `alerts`

| Column | Type | Notes |
|---|---|---|
| `id` | Integer PK | |
| `type` | String | `BRUTE_FORCE`, `OFF_HOURS`, `RAPID_ACCESS`, `REPEATED_UNKNOWN`, `ML_ANOMALY` |
| `severity` | String | `LOW`, `MEDIUM`, `HIGH` |
| `user_id` | Integer (nullable) | |
| `description` | Text | Human-readable alert message |
| `resolved` | Boolean | Default `False` |
| `timestamp` | DateTime | UTC |

#### `locker_state`

| Column | Type | Notes |
|---|---|---|
| `locker_id` | String PK | e.g. `L001` |
| `status` | String | `LOCKED`, `UNLOCKED` |
| `last_user_id` | Integer (nullable) | Last user who interacted |
| `updated_at` | DateTime | UTC |

### Database Initialization

`init_db()` in `database.py`:
1. Creates all tables via `Base.metadata.create_all()`.
2. Runs idempotent migration: adds `users.assigned_locker_id` column if missing.
3. Seeds `LockerState(locker_id='L001')` if not present.
4. Backfills `assigned_locker_id` for existing users (pattern: `L{user.id:03d}`).

---

## 7. API Reference

Base URL: `http://127.0.0.1:8000`

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Returns `{"status": "ok", "service": "Smart Locker API", "version": "2.0.0"}` |

---

### Enroll

#### `POST /api/enroll`

Enroll a new user by uploading 5–10 face images.

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string (form) | ✅ | User display name |
| `images` | file[] | ✅ | 5–10 JPEG/PNG face images (max 50 MB total) |

**Response** (`200 OK`):
```json
{
  "user_id": 1,
  "user_name": "Aakash",
  "embedding_count": 16,
  "assigned_locker_id": "L001",
  "created_at": "2025-05-25T12:00:00"
}
```

**Pipeline**:
1. Validate image count (5–10) and MIME types.
2. Decode each image → detect face → extract 512-dim embedding.
3. Augment images (flip, brightness, rotation) if < 7 supplied.
4. Compute L2-normalized centroid of all embeddings.
5. Compute personal threshold = centroid mean − K × std dev of pairwise distances.
6. Persist `User` row + individual `FaceEmbedding` rows + centroid row.
7. Auto-assign locker `L{user_id:03d}` and seed `LockerState`.

---

#### `PUT /api/enroll/{user_id}/re-enroll`

Replace all embeddings for an existing user.

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `images` | file[] | ✅ | 5–10 face images |

**Response** (`200 OK`):
```json
{
  "user_id": 1,
  "user_name": "Aakash",
  "embedding_count": 16,
  "personal_threshold": 0.3842,
  "assigned_locker_id": "L001",
  "updated_at": "2025-05-25T14:00:00"
}
```

---

### Recognition

#### `POST /api/auth/recognize-multi`

1:1 face verification — caller claims a `user_id` and proves identity with frames.

**Content-Type**: `multipart/form-data`

| Param | Type | Required | Description |
|---|---|---|---|
| `images` | file[] | ✅ | ≥ `MULTI_FRAME_MIN_REQUIRED` face images |
| `user_id` | int (query) | ✅ | ID of the claimed user |
| `locker_id` | string (query) | ❌ | If omitted, uses user's assigned locker |
| `check_liveness` | bool (query) | ❌ | Default `true` |

**Rate limited**: 10 requests / 60 seconds per IP (sliding window).

**Response** (`200 OK`):
```json
{
  "access_granted": true,
  "user_id": 1,
  "user_name": "Aakash",
  "locker_action": "OPEN",
  "final_score": 0.5832,
  "confidence": 0.1332,
  "liveness": {
    "passed": true,
    "blink_detected": true,
    "head_movement_detected": false,
    "reason": "Liveness confirmed"
  },
  "prompt": null,
  "frame_results": [
    {
      "frame_index": 0,
      "quality_passed": true,
      "rejection_reasons": [],
      "antispoof_passed": true,
      "anomaly_passed": true,
      "score": 0.58
    }
  ],
  "frames_processed": 5,
  "frames_passed_quality": 5,
  "frames_passed_anomaly": 5,
  "log_id": 42,
  "anomaly_flag": false
}
```

---

### PIN Authentication

#### `PUT /api/users/{user_id}/pin`

Set or update a user's fallback PIN.

**Body**:
```json
{
  "pin": "1234",
  "old_pin": "0000"   // required only when updating existing PIN
}
```

**Validation**: PIN must be 4–8 digits only.

---

#### `POST /api/auth/pin`

Authenticate via PIN fallback.

**Rate limited**: 3 attempts / 60 seconds per IP.

**Body**:
```json
{
  "user_id": 1,
  "pin": "1234",
  "locker_id": "L001"  // optional — defaults to user's assigned locker
}
```

**Response** (`200 OK`):
```json
{
  "access_granted": true,
  "user_id": 1,
  "user_name": "Aakash",
  "locker_action": "OPEN",
  "method": "PIN",
  "log_id": 43
}
```

---

### Users

#### `GET /api/users`

List all enrolled users (without embedding data).

**Response** (`200 OK`):
```json
[
  {
    "id": 1,
    "name": "Aakash",
    "assigned_locker_id": "L001",
    "has_pin": true,
    "created_at": "2025-05-25T12:00:00"
  }
]
```

---

#### `DELETE /api/users/{user_id}`

Delete a user and all associated data (embeddings, access logs, locker state).

**Response** (`200 OK`):
```json
{
  "user_id": 1,
  "user_name": "Aakash",
  "embeddings_deleted": 16,
  "access_logs_deleted": 5,
  "locker_freed": "L001",
  "message": "User 'Aakash' (id=1) deleted with all associated data."
}
```

---

### Locker

#### `GET /api/locker/status`

Get current locker state (read-only).

| Query | Type | Default | Description |
|---|---|---|---|
| `locker_id` | string | `L001` | Locker to query |

**Response** (`200 OK`):
```json
{
  "locker_id": "L001",
  "status": "LOCKED",
  "last_user_id": 1,
  "updated_at": "2025-05-25T14:30:00"
}
```

---

### Error Envelope

All error responses follow a standardized format:

```json
{
  "error": {
    "code": "BAD_REQUEST",
    "message": "Expected 5-10 images, got 2",
    "details": null
  }
}
```

| HTTP Status | Code |
|---|---|
| 400 | `BAD_REQUEST` |
| 401 | `UNAUTHORIZED` |
| 403 | `FORBIDDEN` |
| 404 | `NOT_FOUND` |
| 422 | `VALIDATION_ERROR` |
| 429 | `RATE_LIMITED` |
| 500 | `INTERNAL_ERROR` |

---

## 8. Service Layer

### `face_pipeline.py` — InsightFace Wrapper

Central module that wraps InsightFace. **No other module imports InsightFace directly.**

| Function | Description |
|---|---|
| `load_image_from_bytes(data)` | Decode raw bytes to BGR numpy array |
| `detect_faces(image)` | Run RetinaFace detection; raises on no face |
| `extract_embedding(face, image)` | 512-dim L2-normalized embedding |
| `get_bbox(face)` | Bounding box as `(x1, y1, x2, y2)` |
| `get_face_landmarks(face)` | 106-point or 5-point landmarks |
| `get_face_pose(face)` | `(yaw, pitch, roll)` in degrees |
| `process_image(image)` | Convenience: detect → embed → return `(embedding, confidence)` |
| `process_image_full(image)` | Like above, also returns the face object |

### `enrollment_service.py` — Enrollment Pipeline

| Function | Description |
|---|---|
| `enroll_user(name, images, db)` | Full enrollment: decode → detect → embed → augment → centroid → persist |
| `re_enroll_user(user_id, images, db)` | Replace all embeddings atomically |
| `get_user_embeddings(user_id, db)` | Load embeddings + centroid for a user |
| `get_all_users_with_embeddings(db)` | Load all users for 1:N matching |

### `multi_frame_recognizer.py` — Recognition Orchestrator

10-step pipeline per recognition request:
1. **Detect face** in each frame
2. **Quality filter** (blur, brightness, size, pose)
3. **Anti-spoof check** (if enabled)
4. **Extract embedding**
5. **Embedding anomaly check** (Isolation Forest)
6. **Compute weighted similarity** against claimed user
7. **Check minimum valid frames** (≥ `MULTI_FRAME_MIN_REQUIRED`)
8. **Top-K median aggregation** of scores
9. **Liveness check** (blink or head movement)
10. **Generate adaptive prompt** if rejected

### `face_comparator.py` — Similarity Scoring

| Function | Description |
|---|---|
| `cosine_similarity(emb1, emb2)` | Raw cosine similarity [0, 1] |
| `compute_weighted_score(query, embeddings, centroid)` | `0.7 × centroid_sim + 0.3 × max_sim` |
| `compute_confidence(score)` | `score - threshold`, clamped to [-1, 1] |
| `find_best_match_v2(query, users)` | 1:N match with personal threshold |
| `find_best_match_multi_user(query, users)` | Like above + ambiguity rejection (margin check) |

### `quality_filter.py` — Image Quality Gates

Checks run before embedding extraction:
- **Blur**: Laplacian variance ≥ threshold
- **Brightness**: Mean intensity in [50, 200]
- **Face size**: Bounding box width ≥ 80px
- **Yaw**: |yaw| ≤ 30°
- **Pitch**: |pitch| ≤ 25°

### `liveness_detector.py` — Liveness Detection

- **Blink detection**: EAR (Eye Aspect Ratio) from 106-point landmarks; detects a dip below threshold for N consecutive frames followed by recovery.
- **Head movement**: Mean Euclidean displacement of landmarks between consecutive frames ≥ threshold.
- **Pass condition**: Blink **OR** head movement.

### `antispoof_detector.py` — Anti-Spoof

Two pluggable backends:
- **ONNX**: MiniFASNetV2 — crops the face, resizes to model input, returns spoof probability.
- **YOLO**: YOLOv8 `{0: fake, 1: real}` — runs on full frame, matches detection to face bbox by IoU.

### `embedding_crypto.py` — Encryption at Rest

- Uses AES-256-GCM with a 96-bit random nonce per encryption.
- Key derived by SHA-256 hashing `EMBEDDING_ENCRYPTION_KEY` or `SECRET_KEY`.
- Storage format: `base64(nonce ‖ ciphertext ‖ tag)`.
- Transparently decrypts; falls back to plain JSON for legacy unencrypted rows.

### `anomaly_detector.py` — Rule-Based + ML Anomaly

Runs after every access attempt:
1. **Brute force**: ≥ N failures from same IP in window.
2. **Off-hours**: Access during 10 PM – 6 AM UTC.
3. **Rapid access**: ≥ N successes by same user in window.
4. **Repeated unknown**: ≥ 2 unknown-face failures from same IP in 5 min.
5. **ML anomaly**: Isolation Forest score on behavioral features.

### `rate_limiter.py` — Request Throttling

Two classes:
- `SlidingWindowLimiter`: Per-IP sliding window with configurable max requests and window seconds. Used for recognize and PIN endpoints.
- `CooldownLimiter`: Locks out a client for N seconds after M consecutive failures. Used for brute-force protection.

---

## 9. Recognition Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                    POST /api/auth/recognize-multi                  │
│                    (user_id, locker_id, images[])                  │
└────────────────────────┬───────────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     Rate Limit Check        │ 429 if exceeded
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Resolve User & Locker      │ 404 / 400
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Load User Embeddings       │ 400 if none enrolled
          └──────────────┬──────────────┘
                         │
           ┌─────────────▼─────────────┐
           │   FOR EACH FRAME:         │
           │                           │
           │  1. Face Detection        │──→ skip if no face
           │  2. Quality Filter        │──→ skip if blur/dark/small/pose
           │  3. Anti-Spoof Check      │──→ skip if spoof detected
           │  4. Extract Embedding     │──→ skip if dimension mismatch
           │  5. Embedding Anomaly     │──→ skip if outlier
           │  6. Weighted Similarity   │──→ record score
           └─────────────┬─────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Min Valid Frames Check     │ DENY if < 3 passed
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Top-K Median Aggregation   │ final_score
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Threshold Comparison       │ DENY if < 0.45
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Liveness Check             │ DENY if no blink/movement
          │  (blink OR head movement)   │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │  Log Access Event           │
          │  Run Anomaly Detection      │
          │  Open Locker (if granted)   │
          └──────────────┬──────────────┘
                         │
                  ┌──────▼──────┐
                  │   RESPONSE  │
                  └─────────────┘
```

---

## 10. Security Design

### Authentication Layers

| Layer | Mechanism | Notes |
|---|---|---|
| **Primary** | Face verification (1:1) | 512-dim ArcFace embeddings, threshold=0.45 |
| **Fallback** | PIN (bcrypt, 12 rounds) | 4–8 digits, 3 attempts / 60s |
| **Liveness** | Blink + head movement | Rejects photos and static screens |
| **Anti-spoof** | ONNX or YOLO classifiers | Rejects printed photos, video replays |

### Data Protection

| What | How |
|---|---|
| Face embeddings at rest | Optional AES-256-GCM encryption (nonce ‖ ciphertext ‖ tag) |
| PIN storage | bcrypt with 12 rounds |
| API surface in production | /docs, /redoc, /openapi.json disabled |
| CORS | Locked to localhost origins only |
| Upload limits | 50 MB total, 5 MB per image |

### Rate Limiting & Brute-Force Protection

- **Recognition**: 10 requests / 60s per IP (sliding window).
- **PIN**: 3 attempts / 60s per IP.
- **Cooldown**: Lock out for 30s after 5 consecutive failures.
- **Anomaly alerts**: Persist to `alerts` table with severity levels.

### Error Handling

- All errors return a standardized `{error: {code, message, details}}` envelope.
- Unhandled exceptions return `500 INTERNAL_ERROR` without leaking stack traces.
- Validation errors (422) include per-field location and message.

---

## 11. Anomaly Detection

### Rule-Based (5 Rules)

| Rule | Trigger | Severity | Description |
|---|---|---|---|
| **Brute Force** | ≥3 failures from same IP in 60s | HIGH | IP-based attack detection |
| **Off-Hours** | Access during 22:00–06:00 UTC | MEDIUM | Unusual access timing |
| **Rapid Access** | ≥5 successes by same user in 10 min | LOW | Suspicious repetition |
| **Repeated Unknown** | ≥2 unknown-face failures from IP in 5 min | HIGH | Unauthorized person probing |
| **ML Anomaly** | Isolation Forest score above threshold | MEDIUM | Behavioral pattern anomaly |

### ML-Based (Isolation Forest)

- Model: `anomaly_model.joblib` (RandomForest / IsolationForest).
- Features: Behavioral access patterns (timing, frequency, confidence scores).
- Threshold: Loaded from `training_metrics.json` or configured via `ML_ANOMALY_THRESHOLD`.
- Score persisted on `access_logs.ml_anomaly_score` for audit trail.

### Alerts Table

All detected anomalies are persisted to the `alerts` table with type, severity, description, and optional user_id. Alerts are also written to `logs/security.log`.

---

## 12. Demo Applications

### `live_demo.py` — CLI Webcam Tool

```
python demo/live_demo.py <command> [--api URL] [--locker ID]
```

| Command | Description |
|---|---|
| `enroll --name "Name"` | Capture 7 webcam frames, enroll as new user |
| `recognize` | Capture 5 frames, run 1:1 verification (blink once for liveness) |
| `live` | Continuous kiosk mode — auto-scan on face detection with 3s cooldown |

**How it works**:
- Opens webcam via OpenCV.
- Uses Haar cascade for local face detection (drawing green boxes).
- Captures JPEG frames and POST them to the backend API.
- Displays progress counter (`3/7`) on the video window.
- In `live` mode, shows status bar with GRANTED/DENIED and cooldown timer.

### `locker_simulation.py` — GUI Kiosk

```
python demo/locker_simulation.py
```

- Opens a 1280×720 OpenCV window showing 6 lockers.
- Click a locker → choose **Sign Up** or **Login**.
- Sign Up triggers enrollment via webcam → backend.
- Login triggers recognition → backend.
- Press **Q** to quit, **ESC** to go back.
- Locker assignments persisted in `demo/locker_assignments.json`.

---

## 13. Deployment

### Local Development

```bash
# Terminal 1: Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Demo
.\.venv\Scripts\Activate.ps1
python demo/live_demo.py enroll --name "YourName"
python demo/live_demo.py recognize
```

### Docker

```bash
# Build
docker build -t smart-locker-backend .

# Run (PowerShell)
docker run --rm -p 8000:8000 `
  --env-file backend/.env `
  -v ${PWD}/backend/data:/app/backend/data `
  -v ${PWD}/backend/models_ml:/app/backend/models_ml `
  -v ${PWD}/logs:/app/logs `
  -v insightface-cache:/app/.insightface `
  smart-locker-backend
```

**Docker image details**:
- Base: `python:3.12-slim`
- CPU-only PyTorch (no CUDA wheels — triples image size for no benefit on kiosk hardware).
- System deps: `libglib2.0-0`, `libgl1`, `libgomp1`, `ca-certificates`.
- Non-root user (`app`, uid 1000).
- Healthcheck every 30s against `/api/health`.
- Models, DB, and logs are **volume-mounted**, not baked into the image.
- Demos run on the host (webcam access from Docker on Windows is impractical).

### Production Checklist

- [ ] Set `ENV=production` (hides Swagger/ReDoc)
- [ ] Set a strong `SECRET_KEY` (≥32 random characters)
- [ ] Set `EMBEDDING_ENCRYPTION_ENABLED=true` + `EMBEDDING_ENCRYPTION_KEY`
- [ ] Set `ANTISPOOF_ENABLED=true` + place model weights
- [ ] Configure `OPERATOR_PIN_HASH` for admin endpoints
- [ ] Review and tighten rate limit settings
- [ ] Set up log rotation for `logs/access.jsonl` and `logs/security.log`
- [ ] Consider PostgreSQL for multi-instance deployments

---

## 14. Development Guide

### Adding a New API Endpoint

1. Create a route handler in `backend/api/your_module.py`.
2. Define Pydantic schemas in `backend/models/schemas.py`.
3. Write business logic in `backend/services/`.
4. Register the router in `backend/main.py` via `app.include_router()`.
5. Add DB models to `backend/models/database.py` if needed.

### Adding a New Anomaly Rule

1. Add the rule function in `backend/services/anomaly_detector.py`.
2. Call it from `run_all_checks()`.
3. Add the alert type string to the `Alert.type` column documentation.
4. Add config variables to `backend/config.py`.

### Running Tests

```bash
pytest backend/ -v
```

### Project Conventions

- **No direct InsightFace imports** outside `face_pipeline.py` and `model_manager.py`.
- **All config** goes through `config.py` — no `os.getenv()` in service code.
- **Immutable dataclasses** for pipeline results (`@dataclass(frozen=True)`).
- **Standardized error envelope** on all HTTP errors.
- **Log levels**: `INFO` for normal operations, `WARNING` for recoverable issues, `ERROR` for failures.

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python is not recognized` | Python not on PATH | Reinstall with "Add to PATH" checked |
| `running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `ModuleNotFoundError: No module named 'fastapi'` | venv not activated | Run `.\.venv\Scripts\Activate.ps1` |
| `Cannot open webcam` | Camera in use by another app | Close Zoom/Teams/Chrome |
| `Connection refused` to localhost:8000 | Backend not running | Start uvicorn in Terminal #1 |
| `Address already in use` | Port 8000 taken | Use `--port 8001` |
| `No face detected` | Poor lighting, angle, or distance | Improve lighting, face camera directly |
| `Only N images had a detectable face` | Low-quality captures | Retry with better conditions |
| Face model download fails | Network issue | Retry on a different network |
| `Microsoft Visual C++ 14.0 required` | Missing build tools | Install [VS Build Tools 2022](https://visualstudio.microsoft.com/downloads/) C++ workload |
| `SSL certificate verify failed` | Corporate proxy | Use `--trusted-host pypi.org --trusted-host files.pythonhosted.org` |
| Recognition always denied | Blur threshold too strict | Lower `QUALITY_BLUR_THRESHOLD` (try `40`) |
| Liveness never passes | Only 5-point landmarks available | Head movement required; blink needs 106-point model |

### Reset Everything

```powershell
# Delete database (auto-recreated on next startup)
Remove-Item backend\data\smart_locker.db

# Delete demo locker assignments
Remove-Item demo\locker_assignments.json

# Nuclear option: delete venv and reinstall
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

---

*Generated from source code scan on 2025-05-25. Keep this file updated as the codebase evolves.*
