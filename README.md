Smart Locker — AI-Powered Facial Recognition Access Control

Backend system for an AI-powered smart locker using facial recognition, PIN fallback authentication, anomaly detection, and anti-spoof protection to control physical locker access.

Features
Face recognition with InsightFace embeddings
Multi-frame verification pipeline
Blink and head-movement liveness detection
Anti-spoof detection (YOLOv8 or ONNX backend)
PIN fallback authentication
AES-encrypted embeddings
Behavioral anomaly detection
Rate limiting and brute-force protection
Interactive demo applications
REST API with Swagger documentation
Tech Stack
Python 3.12
FastAPI
SQLAlchemy
SQLite
InsightFace (RetinaFace + ArcFace)
PyTorch / torchvision
OpenCV
scikit-learn
ONNX Runtime
Ultralytics YOLOv8
Quick Start
1. Clone the Repository
git clone <repo-url>
cd smart-locker
2. Create Virtual Environment

Using uv:

uv venv
uv pip install -r backend/requirements.txt

Or using pip:

python -m venv .venv
Windows
.venv\Scripts\activate
pip install -r backend/requirements.txt
Linux / macOS
source .venv/bin/activate
pip install -r backend/requirements.txt
Environment Configuration

Create a file named backend/.env:

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
Run the Backend
Start API Server
uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload

Expected output:

Smart Locker API is running.
Health Check
curl http://127.0.0.1:8000/api/health
Demo Applications
Enroll User
python demo/live_demo.py enroll --name "YourName"

Captures webcam frames and enrolls a new user.

Recognize User
python demo/live_demo.py recognize

Blink once during capture for liveness verification.

Continuous Kiosk Mode
python demo/live_demo.py live
Locker Simulation GUI
python demo/locker_simulation.py
Reset Database

Delete the SQLite database:

Windows
del backend\data\smart_locker.db
Linux / macOS
rm backend/data/smart_locker.db

The schema is recreated automatically on startup.

API Documentation

Swagger UI:

http://127.0.0.1:8000/docs
API Endpoints
Method	Endpoint	Description
GET	/api/health	Health check
POST	/api/enroll	Enroll user
PUT	/api/enroll/{id}/re-enroll	Re-enroll user
POST	/api/auth/recognize-multi	Multi-frame face verification
PUT	/api/users/{id}/pin	Set PIN
POST	/api/auth/pin	PIN authentication
GET	/api/users	List users
DELETE	/api/users/{id}	Remove user
GET	/api/locker/status	Locker state
POST	/api/locker/close	Close locker
Recognition Pipeline
Face detection using RetinaFace
512-dimensional embedding extraction using ArcFace
AES-256-GCM embedding encryption
Cosine similarity scoring
Threshold validation
Multi-frame voting aggregation
Liveness verification
Anti-spoof classification
Security Features
Rate limiting
Brute-force protection
bcrypt PIN hashing
AES-256-GCM encrypted embeddings
Anti-spoof detection
Multi-frame verification
Quality filtering
Blink-based liveness detection
Restricted CORS policy
Anomaly Detection
Rule-Based Detection
Brute-force attempts
Off-hours access
Repeated access attempts
Unknown face repetition
Machine Learning Detection

RandomForest classifier using behavioral access features.

Project Structure
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
└── project documentation
Model Files

Place the anti-spoof model at backend/models_ml/antispoof.onnx (ONNX backend) or backend/models_ml/l_version_1_300.pt (YOLO backend). InsightFace models download automatically during first launch.

Notes
Use opencv-python instead of opencv-python-headless for GUI demos.
Webcam quality may require lowering blur thresholds.
Only one face backend should be enabled at a time.
License

Add your preferred license here.

Future Improvements
Docker deployment
PostgreSQL support
Mobile companion app
Web dashboard
RFID/NFC hybrid authentication
Cloud synchronization
