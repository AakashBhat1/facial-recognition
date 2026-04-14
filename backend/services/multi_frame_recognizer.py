"""Multi-frame face recognition with quality, anomaly, and liveness checks.

Orchestrates all Phase 2 components into a single pipeline:
1. For each frame: detect face → quality filter → extract embedding → anomaly check
2. Compute weighted similarity per valid frame
3. Aggregate via top-K median
4. Run liveness check across frame landmarks
5. Generate adaptive user prompt if rejected
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from services import embedding_anomaly, face_pipeline, liveness_detector, user_prompt, antispoof_detector
from services.face_comparator import compute_confidence, compute_weighted_score, find_best_match_v2
from services.quality_filter import QualityResult, run_quality_checks

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameResult:
    """Per-frame processing outcome."""

    frame_index: int
    quality_passed: bool
    quality_result: QualityResult | None
    antispoof_passed: bool
    anomaly_passed: bool
    embedding: list[float] | None
    score: float | None


@dataclass(frozen=True)
class MultiFrameResult:
    """Final multi-frame recognition outcome."""

    recognized: bool
    user_id: int | None
    user_name: str
    final_score: float
    confidence: float
    frame_results: tuple[FrameResult, ...]
    liveness_result: liveness_detector.LivenessResult | None
    prompt: user_prompt.PromptSuggestion | None
    frames_processed: int
    frames_passed_quality: int
    frames_passed_anomaly: int


def _aggregate_scores(scores: list[float]) -> float:
    """Top-K median aggregation."""
    if not scores:
        return 0.0
    top_k = sorted(scores, reverse=True)[: config.MULTI_FRAME_TOP_K]
    return statistics.median(top_k)


def recognize_multi_frame(
    images: list[np.ndarray],
    users_with_embeddings: list[dict[str, Any]],
    check_liveness: bool = True,
) -> MultiFrameResult:
    """Run the full multi-frame recognition pipeline.

    *images* is a list of BGR numpy arrays (decoded frames).
    *users_with_embeddings* comes from ``enrollment_service.get_all_users_with_embeddings``.
    """
    frame_results: list[FrameResult] = []
    valid_scores: list[float] = []
    valid_embeddings: list[list[float]] = []
    frames_landmarks: list[np.ndarray] = []
    last_quality_result: QualityResult | None = None

    quality_passed_count = 0
    anomaly_passed_count = 0

    for idx, image in enumerate(images):
        # Step 1: detect face
        try:
            faces = face_pipeline.detect_faces(image)
            face = face_pipeline._largest_face(faces)
        except ValueError:
            frame_results.append(FrameResult(
                frame_index=idx, quality_passed=False,
                quality_result=None, antispoof_passed=True,
                anomaly_passed=False, embedding=None, score=None,
            ))
            continue

        # Collect landmarks for liveness
        try:
            landmarks = face_pipeline.get_face_landmarks(face)
            frames_landmarks.append(landmarks)
        except ValueError:
            pass

        # Step 2: quality check
        qr = run_quality_checks(image, face)
        last_quality_result = qr
        if not qr.passed:
            frame_results.append(FrameResult(
                frame_index=idx, quality_passed=False,
                quality_result=qr, antispoof_passed=True,
                anomaly_passed=False, embedding=None, score=None,
            ))
            continue

        quality_passed_count += 1

        # Step 2.5: anti-spoof check
        bbox = face_pipeline.get_bbox(face)
        spoof_detected, spoof_score = antispoof_detector.check_face(image, bbox)
        if spoof_detected:
            logger.warning("Frame %d: spoof detected (score=%.3f)", idx, spoof_score)
            frame_results.append(FrameResult(
                frame_index=idx, quality_passed=True,
                quality_result=qr, antispoof_passed=False,
                anomaly_passed=False, embedding=None, score=None,
            ))
            continue

        # Step 3: extract embedding
        try:
            emb = face_pipeline.extract_embedding(face, image=image)
        except ValueError:
            frame_results.append(FrameResult(
                frame_index=idx, quality_passed=True,
                quality_result=qr, antispoof_passed=True,
                anomaly_passed=False, embedding=None, score=None,
            ))
            continue

        # Step 4: embedding anomaly check
        if embedding_anomaly.is_anomalous(emb):
            frame_results.append(FrameResult(
                frame_index=idx, quality_passed=True,
                quality_result=qr, antispoof_passed=True,
                anomaly_passed=False, embedding=emb, score=None,
            ))
            continue

        anomaly_passed_count += 1

        # Step 5: compute similarity against each enrolled user
        best_score_this_frame = 0.0
        for entry in users_with_embeddings:
            weighted, _, _ = compute_weighted_score(
                emb, entry["embeddings"], entry["centroid"],
            )
            if weighted > best_score_this_frame:
                best_score_this_frame = weighted

        valid_scores.append(best_score_this_frame)
        valid_embeddings.append(emb)
        frame_results.append(FrameResult(
            frame_index=idx, quality_passed=True,
            quality_result=qr, antispoof_passed=True,
            anomaly_passed=True, embedding=emb, score=best_score_this_frame,
        ))

    # Step 6: check minimum valid frames
    if len(valid_scores) < config.MULTI_FRAME_MIN_REQUIRED:
        confidence = compute_confidence(0.0)
        prompt_suggestion = user_prompt.generate_prompt(confidence, last_quality_result)
        return MultiFrameResult(
            recognized=False, user_id=None, user_name="UNKNOWN",
            final_score=0.0, confidence=confidence,
            frame_results=tuple(frame_results),
            liveness_result=None, prompt=prompt_suggestion,
            frames_processed=len(images),
            frames_passed_quality=quality_passed_count,
            frames_passed_anomaly=anomaly_passed_count,
        )

    # Step 7: top-K median aggregation
    final_score = _aggregate_scores(valid_scores)
    confidence = compute_confidence(final_score)
    recognized = final_score >= config.SIMILARITY_THRESHOLD

    # Step 8: liveness check
    live_result: liveness_detector.LivenessResult | None = None
    if check_liveness:
        if not frames_landmarks:
            live_result = liveness_detector.LivenessResult(
                passed=False,
                blink_detected=False,
                head_movement_detected=False,
                reason="No landmarks available for liveness check",
            )
            recognized = False
        else:
            live_result = liveness_detector.check_liveness(frames_landmarks)
            if not live_result.passed:
                recognized = False

    # Step 9: determine matched user via final-pass resolution
    user_id: int | None = None
    user_name = "UNKNOWN"
    if recognized and valid_embeddings:
        # Use the embedding that produced the highest score for user lookup
        best_idx = int(np.argmax(valid_scores))
        best_emb = valid_embeddings[best_idx]
        match, _, _ = find_best_match_v2(best_emb, users_with_embeddings)
        if match is not None:
            user_id = match["user"].id
            user_name = match["user"].name
        else:
            recognized = False

    # Step 10: generate prompt if rejected
    prompt_suggestion = None
    if not recognized:
        prompt_suggestion = user_prompt.generate_prompt(confidence, last_quality_result)

    return MultiFrameResult(
        recognized=recognized,
        user_id=user_id,
        user_name=user_name,
        final_score=round(final_score, 4),
        confidence=round(confidence, 4),
        frame_results=tuple(frame_results),
        liveness_result=live_result,
        prompt=prompt_suggestion,
        frames_processed=len(images),
        frames_passed_quality=quality_passed_count,
        frames_passed_anomaly=anomaly_passed_count,
    )
