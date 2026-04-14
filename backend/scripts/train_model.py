"""Train an ML anomaly model on synthetic or real access-log data.

Usage:
    cd backend
    python -m scripts.train_model --input data/training_data.csv --output models_ml/isolation_forest.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from services.feature_extractor import FEATURE_NAMES
from services.ml_model_scoring import default_threshold_for, get_score_mode, predict_with_threshold, score_samples

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_and_split(
    csv_path: str | Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load CSV and return stratified (X_train, X_test, y_train, y_test)."""
    df = pd.read_csv(csv_path)
    X = df[FEATURE_NAMES]
    y = df["is_anomaly"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_anomaly_model(
    X_train: pd.DataFrame,
    y_train: pd.Series | None = None,
    *,
    model_kind: str = "random_forest",
    contamination: float = 0.05,
    n_estimators: int = 300,
    seed: int = 42,
) -> object:
    """Fit and return the configured anomaly model."""
    if model_kind == "random_forest":
        if y_train is None:
            raise ValueError("y_train is required for random_forest training.")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
        model.fit(X_train, y_train)
        return model

    if model_kind == "hist_gradient_boosting":
        if y_train is None:
            raise ValueError("y_train is required for hist_gradient_boosting training.")
        model = HistGradientBoostingClassifier(
            random_state=seed,
            learning_rate=0.08,
            max_depth=8,
            max_iter=n_estimators,
            min_samples_leaf=20,
        )
        model.fit(X_train, y_train)
        return model

    if model_kind == "xgboost":
        if y_train is None:
            raise ValueError("y_train is required for xgboost training.")
        pos_count = int(y_train.sum())
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)
        model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            learning_rate=0.08,
            max_depth=8,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    if model_kind == "isolation_forest":
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train)
        return model

    raise ValueError(f"Unsupported model kind: {model_kind}")



def select_threshold(
    scores: np.ndarray,
    y_true: pd.Series | np.ndarray,
    *,
    higher_is_more_anomalous: bool,
    min_precision: float = 0.80,
) -> dict[str, float]:
    """Pick a threshold that maximizes F1 with a precision floor when possible."""
    y_true_arr = np.asarray(y_true)
    quantiles = np.linspace(0.01, 0.99, 80)
    candidates = {round(float(np.quantile(scores, q)), 6) for q in quantiles}
    if higher_is_more_anomalous:
        candidates |= {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
    else:
        candidates |= {-0.3, -0.2, -0.1, 0.0, 0.05, 0.1, 0.15}
    sorted_candidates = sorted(candidates)

    best: dict[str, float] | None = None
    best_with_precision_floor: dict[str, float] | None = None
    for threshold in sorted_candidates:
        y_pred = np.where(scores >= threshold, 1, 0) if higher_is_more_anomalous else np.where(scores < threshold, 1, 0)
        precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
        recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
        candidate = {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate
        if precision >= min_precision and (
            best_with_precision_floor is None or candidate["f1"] > best_with_precision_floor["f1"]
        ):
            best_with_precision_floor = candidate

    return best_with_precision_floor or best or {
        "threshold": 0.5 if higher_is_more_anomalous else 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    threshold: float | None = None,
) -> dict[str, object]:
    """Evaluate the model and return metrics dict."""
    score_mode = get_score_mode(model)
    effective_threshold = default_threshold_for(model) if threshold is None else threshold
    y_pred, scores = predict_with_threshold(model, X_test, threshold=effective_threshold)
    y_true = y_test.values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["normal", "anomaly"], zero_division=0)

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "n_test": len(y_test),
        "n_test_anomalies": int(y_true.sum()),
        "predicted_anomalies": int(y_pred.sum()),
        "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
        "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
        "feature_names": FEATURE_NAMES,
        "score_mode": score_mode,
        "threshold_used": round(float(effective_threshold), 6),
        "score_summary": {
            "min": round(float(scores.min()), 6),
            "max": round(float(scores.max()), 6),
            "mean": round(float(scores.mean()), 6),
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


def save_model(model: object, output_path: str | Path) -> Path:
    """Persist the trained model to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_metrics(metrics: dict[str, object], output_dir: str | Path) -> Path:
    """Persist metrics JSON alongside the trained model."""
    path = Path(output_dir) / "training_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML anomaly model.")
    parser.add_argument("--input", type=str, default="data/training_data.csv")
    parser.add_argument("--output", type=str, default="models_ml/isolation_forest.joblib")
    parser.add_argument(
        "--model-kind",
        type=str,
        default="random_forest",
        choices=("random_forest", "hist_gradient_boosting", "xgboost", "isolation_forest"),
    )
    parser.add_argument("--contamination", type=float, default=0.10)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading data from %s", args.input)
    X_train, X_test, y_train, y_test = load_and_split(args.input, seed=args.seed)
    logger.info("Train: %d rows, Test: %d rows", len(X_train), len(X_test))

    logger.info(
        "Training %s model (n_estimators=%d, contamination=%.3f)",
        args.model_kind,
        args.n_estimators,
        args.contamination,
    )
    model = train_anomaly_model(
        X_train,
        y_train,
        model_kind=args.model_kind,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )

    logger.info("Selecting anomaly threshold from holdout split...")
    holdout_scores = score_samples(model, X_test)
    threshold_metrics = select_threshold(
        holdout_scores,
        y_test,
        higher_is_more_anomalous=get_score_mode(model) == "probability",
    )
    threshold = threshold_metrics["threshold"]
    logger.info(
        "Recommended threshold: %.6f (precision=%.4f, recall=%.4f, f1=%.4f)",
        threshold,
        threshold_metrics["precision"],
        threshold_metrics["recall"],
        threshold_metrics["f1"],
    )

    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    metrics["model_kind"] = args.model_kind
    metrics["recommended_threshold"] = round(float(threshold), 6)
    logger.info("\n%s", metrics["classification_report"])
    logger.info(
        "Precision: %.4f | Recall: %.4f | F1: %.4f | Threshold: %.6f | Score mode: %s",
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        threshold,
        metrics["score_mode"],
    )

    model_path = save_model(model, args.output)
    logger.info("Model saved to %s", model_path)

    metrics_path = save_metrics(metrics, Path(args.output).parent)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
