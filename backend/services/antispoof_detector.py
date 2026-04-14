"""Anti-spoofing detector with pluggable backends.

Two backends are supported, selected via ``config.ANTISPOOF_BACKEND``:

* ``onnx``  — the pretrained MiniFASNetV2 ONNX model that operates on a
  single face crop and returns a spoof probability. Default, legacy path.
* ``yolo``  — a custom-trained YOLOv8 detection model (``l_version_1_300.pt``)
  with classes ``{0: 'fake', 1: 'real'}``. Runs on the full frame and
  picks the detection whose bbox overlaps the requested face region.

When ``ANTISPOOF_ENABLED`` is False or the selected model file is missing,
checks are skipped and all faces are treated as live.

Public API (stable, called from ``multi_frame_recognizer``):

* ``load_model() -> bool``
* ``check_face(image, bbox) -> (is_spoof: bool, spoof_score: float)``
* ``is_spoof(face_crop) -> (is_spoof: bool, spoof_score: float)``
"""
from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_BACKEND_ONNX = "onnx"
_BACKEND_YOLO = "yolo"

_active_backend: str | None = None
_onnx_session: Any = None
_yolo_model: Any = None


def _selected_backend() -> str:
    backend = (config.ANTISPOOF_BACKEND or _BACKEND_ONNX).strip().lower()
    if backend not in (_BACKEND_ONNX, _BACKEND_YOLO):
        logger.warning(
            "Unknown ANTISPOOF_BACKEND '%s' — falling back to '%s'",
            backend, _BACKEND_ONNX,
        )
        return _BACKEND_ONNX
    return backend


def _load_onnx_backend() -> bool:
    global _onnx_session

    if not os.path.exists(config.ANTISPOOF_MODEL_PATH):
        logger.warning(
            "Anti-spoof ONNX model not found at %s — run download_antispoof_model.py first",
            config.ANTISPOOF_MODEL_PATH,
        )
        return False
    try:
        import onnxruntime as ort
        _onnx_session = ort.InferenceSession(
            config.ANTISPOOF_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Anti-spoof ONNX model loaded from %s", config.ANTISPOOF_MODEL_PATH)
        return True
    except Exception:
        logger.warning("Failed to load anti-spoof ONNX model", exc_info=True)
        return False


def _load_yolo_backend() -> bool:
    global _yolo_model

    if not os.path.exists(config.ANTISPOOF_YOLO_MODEL_PATH):
        logger.warning(
            "Anti-spoof YOLO model not found at %s — set ANTISPOOF_YOLO_MODEL_PATH",
            config.ANTISPOOF_YOLO_MODEL_PATH,
        )
        return False
    try:
        from ultralytics import YOLO
    except Exception:
        logger.warning(
            "ANTISPOOF_BACKEND=yolo but 'ultralytics' is not installed. "
            "Install it or switch ANTISPOOF_BACKEND back to 'onnx'.",
            exc_info=True,
        )
        return False
    try:
        _yolo_model = YOLO(config.ANTISPOOF_YOLO_MODEL_PATH)
        _yolo_model.overrides["verbose"] = False
        logger.info(
            "Anti-spoof YOLO model loaded from %s (classes=%s, imgsz=%d)",
            config.ANTISPOOF_YOLO_MODEL_PATH,
            _yolo_model.names,
            config.ANTISPOOF_YOLO_IMGSZ,
        )
        return True
    except Exception:
        logger.warning("Failed to load anti-spoof YOLO model", exc_info=True)
        return False


def load_model() -> bool:
    """Load the configured anti-spoof backend. Returns True on success."""
    global _active_backend, _onnx_session, _yolo_model

    _active_backend = None
    _onnx_session = None
    _yolo_model = None

    if not config.ANTISPOOF_ENABLED:
        logger.info("Anti-spoof detection is disabled via config.")
        return False

    backend = _selected_backend()
    if backend == _BACKEND_YOLO:
        ok = _load_yolo_backend()
    else:
        ok = _load_onnx_backend()

    if ok:
        _active_backend = backend
    return ok


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _run_onnx(face_crop: np.ndarray) -> tuple[bool, float]:
    if _onnx_session is None:
        return False, 0.0
    try:
        input_name = _onnx_session.get_inputs()[0].name
        input_shape = _onnx_session.get_inputs()[0].shape  # e.g. [1, 3, 80, 80]
        h, w = int(input_shape[2]), int(input_shape[3])

        resized = cv2.resize(face_crop, (w, h))
        blob = np.transpose(resized.astype(np.float32), (2, 0, 1))[np.newaxis, ...]

        outputs = _onnx_session.run(None, {input_name: blob})
        raw = np.asarray(outputs[0])

        if raw.ndim == 2 and raw.shape[-1] >= 2:
            logits = raw[0]
            e = np.exp(logits - np.max(logits))
            probs = e / np.sum(e)
            # MiniFASNetV2: index 0 == fake, index 1 == real. For 3-class
            # Silent-Face heads, class 1 is real and everything else is spoof.
            if len(probs) == 2:
                spoof_score = float(probs[0])
            else:
                real_idx = 1
                spoof_score = float(max(p for i, p in enumerate(probs) if i != real_idx))
        else:
            spoof_score = float(raw.flatten()[0])

        return spoof_score >= config.ANTISPOOF_THRESHOLD, spoof_score
    except Exception:
        logger.warning("Anti-spoof ONNX inference failed — treating as live", exc_info=True)
        return False, 0.0


def _run_yolo(image: np.ndarray, bbox: tuple[float, float, float, float] | None) -> tuple[bool, float]:
    """Run YOLO spoof detection on the full frame and pick the box matching ``bbox``.

    The model emits detections with class ``0=fake`` and ``1=real``. The spoof
    score returned is the confidence of the best ``fake`` detection that overlaps
    the requested face region; if the matched detection is ``real`` we invert
    (``1 - real_conf``) so callers still get a monotonic score in [0, 1].
    """
    if _yolo_model is None:
        return False, 0.0
    try:
        results = _yolo_model.predict(
            source=image,
            imgsz=config.ANTISPOOF_YOLO_IMGSZ,
            conf=config.ANTISPOOF_YOLO_CONF,
            verbose=False,
        )
        if not results:
            return False, 0.0
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            # No detection at all — model is unsure; treat as live but low score.
            return False, 0.0

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        # Pick the detection that best matches the face bbox the caller asked
        # about. When no bbox is supplied (``is_spoof(crop)`` path) we treat the
        # whole image as the region of interest.
        if bbox is not None:
            target = tuple(float(v) for v in bbox)
            best_idx = -1
            best_iou = -1.0
            for i, det_box in enumerate(boxes_xyxy):
                iou = _iou(target, tuple(det_box.tolist()))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx < 0 or best_iou <= 0.0:
                # No overlap — take the highest-confidence detection as a fallback.
                best_idx = int(np.argmax(confs))
        else:
            best_idx = int(np.argmax(confs))

        cls_id = int(classes[best_idx])
        conf = float(confs[best_idx])
        class_name = _yolo_model.names.get(cls_id, str(cls_id)).lower()

        if class_name == "fake":
            spoof_score = conf
        elif class_name == "real":
            spoof_score = 1.0 - conf
        else:
            # Unknown class — stay conservative, treat as spoof.
            spoof_score = conf

        return spoof_score >= config.ANTISPOOF_THRESHOLD, spoof_score
    except Exception:
        logger.warning("Anti-spoof YOLO inference failed — treating as live", exc_info=True)
        return False, 0.0


def is_spoof(face_crop: np.ndarray) -> tuple[bool, float]:
    """Determine if a face crop is a spoof.

    Returns ``(is_spoof, spoof_score)`` where ``spoof_score`` ranges from
    0.0 (definitely live) to 1.0 (definitely spoof). Returns ``(False, 0.0)``
    when no backend is loaded.
    """
    if _active_backend is None:
        return False, 0.0
    if _active_backend == _BACKEND_ONNX:
        return _run_onnx(face_crop)
    if _active_backend == _BACKEND_YOLO:
        return _run_yolo(face_crop, bbox=None)
    return False, 0.0


def check_face(image: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[bool, float]:
    """Check whether the face region given by ``bbox`` in ``image`` is a spoof."""
    if _active_backend is None:
        return False, 0.0

    if _active_backend == _BACKEND_YOLO:
        # YOLO runs on the full frame — it needs the surrounding context.
        return _run_yolo(image, bbox=bbox)

    # ONNX backend: crop the face and hand it to MiniFASNet.
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return False, 0.0

    face_crop = image[y1:y2, x1:x2]
    return _run_onnx(face_crop)
