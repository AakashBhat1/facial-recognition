import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# Always load backend/.env irrespective of current working directory.
load_dotenv(BASE_DIR / ".env")


def _resolve_local_path(path_value: str) -> Path:
    """Resolve relative paths against backend directory for stable behavior."""
    raw = Path(path_value)
    return raw if raw.is_absolute() else (BASE_DIR / raw).resolve()


def _normalize_database_url(raw_url: str) -> str:
    """Normalize sqlite URLs so relative paths resolve predictably."""
    sqlite_prefix = "sqlite:///"
    if raw_url.startswith(sqlite_prefix):
        raw_path = raw_url[len(sqlite_prefix) :]
        abs_path = _resolve_local_path(raw_path)
        # SQLAlchemy expects forward slashes in sqlite URLs on Windows.
        return f"{sqlite_prefix}{abs_path.as_posix()}"
    return raw_url


def _env_bool(var_name: str, default: bool) -> bool:
    """Parse a boolean env var with strict accepted values."""
    raw = os.getenv(var_name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value for {var_name}: '{raw}'. "
        "Use one of: true/false, 1/0, yes/no, on/off"
    )


def _load_recommended_ml_threshold() -> float | None:
    """Read the last recommended threshold from training metrics when available."""
    metrics_path = BASE_DIR / "models_ml" / "training_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        import json

        metrics = json.loads(metrics_path.read_text())
        value = metrics.get("recommended_threshold")
        return float(value) if value is not None else None
    except Exception:
        return None


DATABASE_URL = _normalize_database_url(
    os.getenv("DATABASE_URL", "sqlite:///./data/smart_locker.db")
)
# Default to project-level logs folder (../logs from backend/).
LOG_DIR = str(_resolve_local_path(os.getenv("LOG_DIR", "../logs")))
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret")

# Environment flag. "production" hides /docs, /redoc, /openapi.json.
ENV = os.getenv("ENV", "development").lower()
IS_PRODUCTION = ENV == "production"

# Operator PIN (bcrypt hash). Required for admin endpoints in production.
OPERATOR_PIN_HASH = os.getenv("OPERATOR_PIN_HASH", "")

# Face recognition
# Tuned via LFW Phase 3: EER is 0.11. Threshold 0.45 ensures False Accept Rate (FAR) = 0.0% 
# while maintaining a False Reject Rate (FRR) < 5%.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.45))

# Face pipeline (Phase 1)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 512))
LEGACY_EMBEDDING_DIM = 128
CENTROID_WEIGHT = float(os.getenv("CENTROID_WEIGHT", 0.7))
MAX_SCORE_WEIGHT = float(os.getenv("MAX_SCORE_WEIGHT", 0.3))
MIN_ENROLLMENT_IMAGES = int(os.getenv("MIN_ENROLLMENT_IMAGES", 5))
MAX_ENROLLMENT_IMAGES = int(os.getenv("MAX_ENROLLMENT_IMAGES", 10))
FACE_DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", 0.5))
# Face backend selector:
# - USE_BUFFALO_MODEL=true, USE_CUSTOM_FACE_MODEL=false  -> current/default behavior
# - USE_BUFFALO_MODEL=false, USE_CUSTOM_FACE_MODEL=true  -> custom InsightFace pack
# Any other combination is invalid and fails fast at startup.
USE_BUFFALO_MODEL = _env_bool("USE_BUFFALO_MODEL", True)
USE_CUSTOM_FACE_MODEL = _env_bool("USE_CUSTOM_FACE_MODEL", False)
if USE_BUFFALO_MODEL == USE_CUSTOM_FACE_MODEL:
    raise ValueError(
        "Invalid face backend selection: set exactly one of "
        "USE_BUFFALO_MODEL or USE_CUSTOM_FACE_MODEL to true"
    )
# buffalo_s: MobileFaceNet backbone (~16MB). 5-10x faster than buffalo_l on CPU,
# ideal for on-device kiosk deployment at close range. 512-dim embeddings.
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "buffalo_s")
CUSTOM_FACE_MODEL_NAME = os.getenv("CUSTOM_FACE_MODEL_NAME", "").strip()
_custom_face_model_root_raw = os.getenv("CUSTOM_FACE_MODEL_ROOT", "").strip()
CUSTOM_FACE_MODEL_ROOT = (
    str(_resolve_local_path(_custom_face_model_root_raw))
    if _custom_face_model_root_raw
    else ""
)
if USE_CUSTOM_FACE_MODEL and not CUSTOM_FACE_MODEL_NAME:
    raise ValueError(
        "CUSTOM_FACE_MODEL_NAME is required when USE_CUSTOM_FACE_MODEL=true"
    )
if USE_CUSTOM_FACE_MODEL and CUSTOM_FACE_MODEL_ROOT and not Path(CUSTOM_FACE_MODEL_ROOT).exists():
    raise ValueError(
        f"CUSTOM_FACE_MODEL_ROOT does not exist: {CUSTOM_FACE_MODEL_ROOT}"
    )
CUSTOM_PT_EMBEDDER_ENABLED = _env_bool("CUSTOM_PT_EMBEDDER_ENABLED", False)
CUSTOM_PT_CHECKPOINT_PATH = str(_resolve_local_path(
    os.getenv("CUSTOM_PT_CHECKPOINT_PATH", "../face_model_scratch/models/best.pt")
))
if CUSTOM_PT_EMBEDDER_ENABLED and not Path(CUSTOM_PT_CHECKPOINT_PATH).exists():
    raise ValueError(
        f"CUSTOM_PT_CHECKPOINT_PATH does not exist: {CUSTOM_PT_CHECKPOINT_PATH}"
    )
# Smaller detection resolution for faster CPU inference at kiosk distance.
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", 320))

# Phase 3: Per-user threshold
PERSONAL_THRESHOLD_K = float(os.getenv("PERSONAL_THRESHOLD_K", "1.5"))
EMBEDDING_DRIFT_CAP = int(os.getenv("EMBEDDING_DRIFT_CAP", "50"))

# Anomaly detection thresholds
BRUTE_FORCE_LIMIT = int(os.getenv("BRUTE_FORCE_LIMIT", 3))
BRUTE_FORCE_WINDOW_SECONDS = int(os.getenv("BRUTE_FORCE_WINDOW_SECONDS", 60))
OFF_HOURS_START = int(os.getenv("OFF_HOURS_START", 22))  # 10 PM
OFF_HOURS_END = int(os.getenv("OFF_HOURS_END", 6))  # 6 AM
RAPID_ACCESS_LIMIT = int(os.getenv("RAPID_ACCESS_LIMIT", 5))
RAPID_ACCESS_WINDOW_MINUTES = int(os.getenv("RAPID_ACCESS_WINDOW_MINUTES", 10))

# Rate limiting
RATE_LIMIT_RECOGNIZE_MAX = int(os.getenv("RATE_LIMIT_RECOGNIZE_MAX", 10))
RATE_LIMIT_RECOGNIZE_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_RECOGNIZE_WINDOW_SECONDS", 60))

# ML anomaly detection
ML_MODEL_PATH = str(_resolve_local_path(os.getenv("ML_MODEL_PATH", "models_ml/anomaly_model.joblib")))
ML_SCORING_ENABLED = os.getenv("ML_SCORING_ENABLED", "true").lower() == "true"
ML_ANOMALY_THRESHOLD = float(
    os.getenv("ML_ANOMALY_THRESHOLD", str(_load_recommended_ml_threshold() or 0.0))
)
ML_CONTAMINATION = float(os.getenv("ML_CONTAMINATION", "0.05"))

# Phase 2: Quality filtering
QUALITY_BLUR_THRESHOLD = float(os.getenv("QUALITY_BLUR_THRESHOLD", "100.0"))
QUALITY_BRIGHTNESS_MIN = float(os.getenv("QUALITY_BRIGHTNESS_MIN", "50.0"))
QUALITY_BRIGHTNESS_MAX = float(os.getenv("QUALITY_BRIGHTNESS_MAX", "200.0"))
QUALITY_MIN_FACE_WIDTH = int(os.getenv("QUALITY_MIN_FACE_WIDTH", "80"))
QUALITY_MAX_YAW = float(os.getenv("QUALITY_MAX_YAW", "30.0"))
QUALITY_MAX_PITCH = float(os.getenv("QUALITY_MAX_PITCH", "25.0"))

# Phase 2: Multi-frame
MULTI_FRAME_MIN_REQUIRED = int(os.getenv("MULTI_FRAME_MIN_REQUIRED", "3"))
MULTI_FRAME_TOP_K = int(os.getenv("MULTI_FRAME_TOP_K", "3"))
DETECTION_INTERVAL_FRAMES = int(os.getenv("DETECTION_INTERVAL_FRAMES", "3"))

# Phase 2: Embedding anomaly
EMBEDDING_ANOMALY_CONTAMINATION = float(os.getenv("EMBEDDING_ANOMALY_CONTAMINATION", "0.01"))
EMBEDDING_ANOMALY_MODEL_PATH = str(_resolve_local_path(
    os.getenv("EMBEDDING_ANOMALY_MODEL_PATH", "models_ml/embedding_isolation_forest.joblib")
))

# Phase 2: Liveness
LIVENESS_EAR_THRESHOLD = float(os.getenv("LIVENESS_EAR_THRESHOLD", "0.21"))
LIVENESS_EAR_CONSEC_FRAMES = int(os.getenv("LIVENESS_EAR_CONSEC_FRAMES", "2"))
LIVENESS_HEAD_MOVEMENT_THRESHOLD = float(os.getenv("LIVENESS_HEAD_MOVEMENT_THRESHOLD", "5.0"))

# Phase 4: Rate limiting cooldown
RATE_LIMIT_COOLDOWN_SECONDS = int(os.getenv("RATE_LIMIT_COOLDOWN_SECONDS", "30"))
RATE_LIMIT_FAILURE_TRIGGER = int(os.getenv("RATE_LIMIT_FAILURE_TRIGGER", "5"))

# Phase 4: Embedding encryption at rest
EMBEDDING_ENCRYPTION_ENABLED = os.getenv("EMBEDDING_ENCRYPTION_ENABLED", "false").lower() == "true"
EMBEDDING_ENCRYPTION_KEY = os.getenv("EMBEDDING_ENCRYPTION_KEY", "")

# Phase 4: Embedding versioning
EMBEDDING_MODEL_VERSION = os.getenv("EMBEDDING_MODEL_VERSION", "buffalo_s_v1")

# Phase 4: Multi-user support
MULTI_USER_MARGIN = float(os.getenv("MULTI_USER_MARGIN", "0.08"))

# Phase 4: Anti-spoof
# Backend switch: "onnx" (MiniFASNetV2, legacy) or "yolo" (custom YOLOv8 real/fake).
ANTISPOOF_ENABLED = os.getenv("ANTISPOOF_ENABLED", "false").lower() == "true"
ANTISPOOF_BACKEND = os.getenv("ANTISPOOF_BACKEND", "onnx").strip().lower()
ANTISPOOF_THRESHOLD = float(os.getenv("ANTISPOOF_THRESHOLD", "0.5"))
# Legacy ONNX (MiniFASNetV2) — face-crop classifier.
ANTISPOOF_MODEL_PATH = str(_resolve_local_path(
    os.getenv("ANTISPOOF_MODEL_PATH", "models_ml/antispoof.onnx")
))
# Custom YOLOv8 real/fake detector (full-frame).
ANTISPOOF_YOLO_MODEL_PATH = str(_resolve_local_path(
    os.getenv("ANTISPOOF_YOLO_MODEL_PATH", "models_ml/l_version_1_300.pt")
))
ANTISPOOF_YOLO_IMGSZ = int(os.getenv("ANTISPOOF_YOLO_IMGSZ", "640"))
ANTISPOOF_YOLO_CONF = float(os.getenv("ANTISPOOF_YOLO_CONF", "0.25"))
if ANTISPOOF_ENABLED and ANTISPOOF_BACKEND == "yolo" and not Path(ANTISPOOF_YOLO_MODEL_PATH).exists():
    raise ValueError(
        f"ANTISPOOF_YOLO_MODEL_PATH does not exist: {ANTISPOOF_YOLO_MODEL_PATH}"
    )
