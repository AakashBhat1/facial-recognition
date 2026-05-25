# Smart Locker — AI-Powered Facial Recognition Access Control

Backend system for an AI-powered smart locker using facial recognition, PIN fallback authentication, anomaly detection, and anti-spoof protection for secure physical locker access.

---

# Features

- Facial recognition using InsightFace embeddings
- Multi-frame verification pipeline
- Blink and head-movement liveness detection
- Anti-spoof detection (YOLOv8 or ONNX backend)
- PIN fallback authentication
- AES-encrypted facial embeddings
- Behavioral anomaly detection
- Rate limiting and brute-force protection
- Interactive demo applications
- REST API with Swagger documentation

---

# Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| API Framework | FastAPI |
| ORM | SQLAlchemy |
| Database | SQLite |
| Face Recognition | InsightFace (RetinaFace + ArcFace) |
| Deep Learning | PyTorch / torchvision |
| Computer Vision | OpenCV |
| ML Utilities | scikit-learn |
| Inference Runtime | ONNX Runtime |
| Anti-Spoof Detection | Ultralytics YOLOv8 |

---

# Quick Start

## 1. Clone the Repository

```bash
git clone <repo-url>
cd smart-locker
```

---

## 2. Create a Virtual Environment

### Using `uv`

```bash
uv venv
uv pip install -r backend/requirements.txt
```

### Using `pip`

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

#### Linux / macOS

```bash
source .venv/bin/activate
pip install -r backend/requirements.txt
```

---

## 3. Environment Configuration

Create a file named:

```text
backend/.env
```

Add the following configuration:

```env
DATABASE_URL=sqlite:///./data/smart_locker.db
SECRET_KEY=replace_with_secure_random_key

USE_BUFFALO_MODEL=true
USE_CUSTOM_FACE_MODEL=false

FACE_MODEL_NAME=buffalo_s

SIMILARITY_THRESHOLD=0.45

ML_SCORING_ENABLED=true
ML_MODEL_PATH=models_ml/anomaly_model.joblib
ML_ANOMALY_THRESHOLD=0.27

ANTISPOOF_ENABLED=false
ANTISPOOF_BACKEND=yolo
ANTISPOOF_THRESHOLD=0.4

ANTISPOOF_MODEL_PATH=models_ml/antispoof.onnx
ANTISPOOF_YOLO_MODEL_PATH=models_ml/l_version_1_300.pt

ANTISPOOF_YOLO_IMGSZ=640
ANTISPOOF_YOLO_CONF=0.25

QUALITY_BLUR_THRESHOLD=40
MULTI_FRAME_MIN_REQUIRED=3
```

---

# Running the Backend

## Start the API Server

```bash
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

Expected output:

```text
Smart Locker API is running.
```

---

## Health Check

```bash
curl http://127.0.0.1:8000/api/health
```

---

# Demo Applications

## Enroll User

```bash
python demo/live_demo.py enroll --name "YourName"
```

Captures webcam frames and enrolls a new user.

---

## Recognize User

```bash
python demo/live_demo.py recognize
```

Blink once during capture for liveness verification.

---

## Continuous Kiosk Mode

```bash
python demo/live_demo.py live
```

---

## Locker Simulation GUI

```bash
python demo/locker_simulation.py
```

---

# Reset Database

Delete the SQLite database manually.

## Windows

```bash
del backend\data\smart_locker.db
```

## Linux / macOS

```bash
rm backend/data/smart_locker.db
```

The schema will be recreated automatically on startup.

---

# API Documentation

## Swagger UI

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/enroll` | Enroll user |
| PUT | `/api/enroll/{id}/re-enroll` | Re-enroll user |
| POST | `/api/auth/recognize-multi` | Multi-frame face verification |
| PUT | `/api/users/{id}/pin` | Set PIN |
| POST | `/api/auth/pin` | PIN authentication |
| GET | `/api/users` | List users |
| DELETE | `/api/users/{id}` | Remove user |
| GET | `/api/locker/status` | Locker state |
| POST | `/api/locker/close` | Close locker |

---

# Recognition Pipeline

1. Face detection using RetinaFace
2. 512-dimensional embedding extraction using ArcFace
3. AES-256-GCM embedding encryption
4. Cosine similarity scoring
5. Threshold validation
6. Multi-frame voting aggregation
7. Liveness verification
8. Anti-spoof classification

---

# Security Features

- Rate limiting
- Brute-force protection
- bcrypt PIN hashing
- AES-256-GCM encrypted embeddings
- Anti-spoof detection
- Multi-frame verification
- Quality filtering
- Blink-based liveness detection
- Restricted CORS policy

---

# Anomaly Detection

## Rule-Based Detection

- Brute-force attempts
- Off-hours access
- Repeated access attempts
- Unknown face repetition

## Machine Learning Detection

RandomForest classifier using behavioral access features.

---

# Project Structure

```text
backend/
├── api/
├── middleware/
├── models/
├── models_ml/
├── services/
├── config.py
└── main.py

demo/
├── live_demo.py
└── locker_simulation.py

docs/
└── project_documentation/
```

---

# Model Files

Place anti-spoof models in one of the following locations:

## ONNX Backend

```text
backend/models_ml/antispoof.onnx
```

## YOLO Backend

```text
backend/models_ml/l_version_1_300.pt
```

InsightFace models are downloaded automatically during first launch.

---

# Notes

- Use `opencv-python` instead of `opencv-python-headless` for GUI demos.
- Webcam quality may require adjusting blur thresholds.
- Only one face recognition backend should be enabled at a time.

---

# Future Improvements

- Docker deployment
- PostgreSQL support
- Mobile companion app
- Web dashboard
- RFID/NFC hybrid authentication
- Cloud synchronization

---

# License

Add your preferred license here.
