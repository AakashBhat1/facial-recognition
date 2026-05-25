# Smart Locker — Backend Image
#
# What this builds:
#   A CPU-only FastAPI image that serves the recognition / enrollment / locker
#   APIs on port 8000. The webcam-driven demos (demo/live_demo.py and
#   demo/locker_simulation.py) are intentionally NOT containerized — they need
#   host camera + OpenCV GUI windows and run from the host against this image.
#
# Models and the SQLite DB are NOT baked in. Mount them at runtime so they
# survive image rebuilds and can be swapped without a fresh push.
#
# Build:
#   docker build -t smart-locker-backend .
#
# Run (from repo root, PowerShell):
#   docker run --rm -p 8000:8000 `
#     --env-file backend/.env `
#     -v ${PWD}/backend/data:/app/backend/data `
#     -v ${PWD}/backend/models_ml:/app/backend/models_ml `
#     -v ${PWD}/logs:/app/logs `
#     -v insightface-cache:/app/.insightface `
#     smart-locker-backend
#
# Run (bash):
#   docker run --rm -p 8000:8000 \
#     --env-file backend/.env \
#     -v "$(pwd)/backend/data:/app/backend/data" \
#     -v "$(pwd)/backend/models_ml:/app/backend/models_ml" \
#     -v "$(pwd)/logs:/app/logs" \
#     -v insightface-cache:/app/.insightface \
#     smart-locker-backend

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    INSIGHTFACE_HOME=/app/.insightface

# System deps:
#   libglib2.0-0, libgl1 — required by opencv-python-headless transitively
#   libgomp1            — OpenMP runtime for onnxruntime / scikit-learn
#   ca-certificates     — for HTTPS model downloads on first launch
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so source edits don't bust the layer cache.
# Force CPU-only torch — the locker hardware is CPU-only, and CUDA wheels would
# triple the image size for no benefit.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/backend/requirements.txt

# Copy the backend source. .dockerignore excludes models, db, .env, logs, and
# every non-backend tree — keep the layer slim.
COPY backend /app/backend

# Non-root user. Pre-create the runtime dirs so volume mounts inherit ownership.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/backend/data /app/backend/models_ml /app/logs /app/.insightface \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status==200 else 1)"

# Bind to 0.0.0.0 so the host can reach the container on the published port.
# CORS in backend/main.py is locked to localhost origins — fine for the host-side
# demos hitting http://127.0.0.1:8000.
CMD ["uvicorn", "main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
