import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from models.database import init_db
from api import enroll, pin_auth, recognize_multi, users, locker
from middleware.error_handler import register_error_handlers
from services import antispoof_detector, custom_pt_embedder, embedding_anomaly, ml_anomaly_service
from services.model_manager import (
    get_active_face_backend,
    get_active_face_model_name,
    get_insightface_model,
)

logger = logging.getLogger(__name__)

# Hide /docs, /redoc, /openapi.json in production — don't leak API surface on devices.
_docs_url = None if config.IS_PRODUCTION else "/docs"
_redoc_url = None if config.IS_PRODUCTION else "/redoc"
_openapi_url = None if config.IS_PRODUCTION else "/openapi.json"

app = FastAPI(
    title="Smart Locker API",
    description="Backend for the AI-powered Smart Locker System",
    version="2.0.0",
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
)

# Single-instance deployment: backend and kiosk UI run on the same device.
# CORS locked to localhost origins only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

# Register all routers
app.include_router(enroll.router)
app.include_router(pin_auth.router)
app.include_router(pin_auth.users_router)
app.include_router(recognize_multi.router)
app.include_router(users.router)
app.include_router(locker.router)


@app.on_event("startup")
def startup():
    init_db()
    logger.info("Database initialized.")
    if ml_anomaly_service.load_model():
        logger.info("ML anomaly scoring is active.")
    else:
        logger.info("ML anomaly scoring is inactive (model not loaded or disabled).")
    if embedding_anomaly.load_model():
        logger.info("Embedding anomaly model is active.")
    else:
        logger.info("Embedding anomaly model not loaded — embedding anomaly checks disabled.")
    if antispoof_detector.load_model():
        logger.info("Anti-spoof detection is active.")
    else:
        logger.info("Anti-spoof detection is inactive.")
    # Preload face model so first request has zero cold-start delay
    get_insightface_model()
    custom_pt_embedder.preload_if_enabled()
    logger.info(
        "Face backend active: %s (%s)",
        get_active_face_backend(),
        get_active_face_model_name(),
    )
    logger.info(
        "Custom PT embedder enabled: %s",
        "yes" if custom_pt_embedder.is_enabled() else "no",
    )
    logger.info("Smart Locker API v2.0 is running.")


@app.get("/api/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "Smart Locker API", "version": "2.0.0"}


register_error_handlers(app)

