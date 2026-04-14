# Smart Locker — Demo Runbook for Team Lead

Last updated: 2026-04-04
Presenter: Aakash

================================================================================
PREPARATION (before the demo)
================================================================================

1. Open a terminal and start the backend server:

   cd "c:\Users\zewan\OneDrive\Documents\Claude\Projects\facial recognition"
   .venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000

2. Wait for these two lines in the console:
   - "ML anomaly scoring is active"       ← trained model loaded
   - InsightFace model preloaded           ← no cold start on first request

3. Open a second terminal for running demo commands.

================================================================================
PART A: Automated API Demo (2 minutes)
================================================================================

From the repo root, run all 3 scenarios with full JSON output:

   .venv\Scripts\python.exe demo\run_demo.py --base-url http://127.0.0.1:8000 --scenario all --seed 42 --verbose

WHAT TO NARRATE:

  Scenario 1 — GRANT (Authorized Access)
    - System registers a temp user with a face embedding
    - Same embedding is recognized → access_granted: true
    - Locker action: OPEN
    - Access log entry created with SUCCESS

  Scenario 2 — DENY (Unauthorized Access)
    - Unknown face embedding sent to the system
    - No matching user → access_granted: false
    - Locker action: ACCESS_DENIED
    - Access log entry created with FAILURE

  Scenario 3 — ANOMALY (Attack Detection)
    - Rapid repeated access attempts for the same user
    - System triggers a RAPID_ACCESS anomaly alert
    - ML anomaly score computed in parallel (RandomForest, threshold 0.27)
    - Alert visible in /api/alerts

================================================================================
PART B: Live Webcam Demo (3-5 minutes) — THE IMPRESSIVE PART
================================================================================

Step 1: Enroll yourself with the camera
-----------------------------------------

   cd backend
   ..\.venv\Scripts\python.exe scripts\live_demo.py enroll --name "Aakash"

   - Camera opens automatically
   - Face is auto-detected (green rectangle appears)
   - 10 photos captured automatically — no manual input needed
   - Console shows "Enrolled successfully! Embeddings stored: 11"
   - Narrate: "The system extracts 512-dimensional face embeddings from each
     photo, augments them, and computes a centroid for robust matching."

Step 2: Unlock with your face
-------------------------------

   ..\.venv\Scripts\python.exe scripts\live_demo.py recognize

   - Camera opens, scans your face
   - Shows: ACCESS GRANTED — Welcome, Aakash! (score: 0.835)
   - Narrate: "The system detects the face, extracts the embedding, and
     computes weighted cosine similarity (70% centroid + 30% max) against
     all enrolled users. Score 0.835 is well above our tuned threshold of 0.45."

Step 3: Show it denies strangers
----------------------------------

   Ask Ajit or another team member to stand in front of the camera:

   ..\.venv\Scripts\python.exe scripts\live_demo.py recognize

   - Different person → ACCESS DENIED (score: ~0.10)
   - Narrate: "Unknown faces get very low similarity scores. Our threshold
     was tuned on 13,233 LFW faces to achieve 0% false acceptance rate."

Step 4 (Optional): Kiosk mode
-------------------------------

   ..\.venv\Scripts\python.exe scripts\live_demo.py live

   - Continuous auto-scanning like a real locker terminal
   - Scans automatically when a face is detected, 3-second cooldown between scans
   - Press Q on the camera window to quit

================================================================================
PART C: Show Logs and Alerts (1 minute)
================================================================================

Open these URLs in a browser:

   http://127.0.0.1:8000/api/health     — System health check
   http://127.0.0.1:8000/api/logs       — All access attempts with timestamps
   http://127.0.0.1:8000/api/alerts     — Anomaly alerts raised by the system
   http://127.0.0.1:8000/api/users      — Registered users

Narrate: "Every access attempt — successful or failed — is logged with
timestamps, similarity scores, and IP addresses. The anomaly detection
system monitors these logs in real time."

================================================================================
TALKING POINTS FOR QUESTIONS
================================================================================

Face Recognition Model:
  - InsightFace buffalo_l (ArcFace architecture)
  - 512-dimensional embeddings
  - Pretrained on millions of faces, fine-tuned threshold on LFW dataset

Threshold Tuning:
  - Tuned on 13,233 Labeled Faces in the Wild (LFW) images
  - 9,824 genuine pairs + 9,824 impostor pairs
  - Result: 0% False Accept Rate, <5% False Reject Rate
  - Threshold: 0.45 (cosine similarity)

Anomaly Detection (Dual Layer):
  Layer 1 — Rule-Based:
    - Brute force detection (3 failures in 60 seconds)
    - Off-hours access (10 PM – 6 AM)
    - Rapid repeated access (5 in 10 minutes)
    - Repeated unknown faces (3 in 5 minutes)
  Layer 2 — Machine Learning:
    - RandomForest classifier (300 trees)
    - Trained on 10,000 rows of real InsightFace output data
    - 14 behavioral features (access frequency, IP patterns, timing, etc.)
    - F1 score: 1.0 on held-out test set
    - Auto-retrains from production logs via scripts/retrain_model.py

Security Features:
  - Rate limiting (10 req/60s per IP, 30s cooldown after 5 failures)
  - PIN fallback authentication (SHA-256 + salt)
  - Embedding encryption at rest (AES-256-GCM)
  - Anti-spoof detection (ONNX CNN-based, optional)
  - Quality filtering (blur, brightness, face size, pose angle)
  - Liveness detection (blink + head movement)

Tech Stack:
  - Python 3.12, FastAPI, SQLAlchemy, SQLite
  - InsightFace (RetinaFace + ArcFace), OpenCV
  - scikit-learn (RandomForest anomaly model)

================================================================================
TROUBLESHOOTING
================================================================================

Problem: 429 RATE_LIMITED error
Fix:     Wait 60 seconds, or restart the backend server

Problem: Camera won't open
Fix:     Close other apps using the webcam (Teams, Zoom, etc.)

Problem: Low recognition score
Fix:     Better lighting, face closer to camera, remove obstructions

Problem: ML model not loading
Fix:     Check backend/.env has ML_SCORING_ENABLED=true

Problem: "No face detected" on enrollment
Fix:     Ensure good lighting, face the camera directly, stay still

Problem: Back-to-back demo runs failing
Fix:     Rate limit resets after 60 seconds. Restart backend for instant reset.

================================================================================
DEMO ORDER TIP
================================================================================

1. Start with Part A (automated API demo) — quick, reliable, proves backend works
2. Then Part B (live webcam) — visual, impressive, proves real face recognition
3. End with Part C (logs/alerts) — shows monitoring and security posture

The webcam demo is the most impactful — save it for the middle/end when the
team lead is engaged. Start with the predictable API demo to build confidence.

================================================================================
