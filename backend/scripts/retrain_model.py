"""Retrain the anomaly model from real access-log data in the database.

Usage:
    cd backend
    python -m scripts.retrain_model --output models_ml/isolation_forest.joblib --min-rows 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import ML_CONTAMINATION
from models.database import AccessLog, SessionLocal
from scripts.train_model import evaluate_model, save_metrics, save_model, select_threshold, train_anomaly_model
from services.feature_extractor import FEATURE_NAMES, extract_features_from_db
from services.ml_model_scoring import get_score_mode, score_samples

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_features_from_db_bulk(min_rows: int = 1000) -> pd.DataFrame | None:
    """Build a dataframe from AccessLog rows plus engineered features."""
    db = SessionLocal()
    try:
        total = db.query(AccessLog).count()
        if total < min_rows:
            logger.warning("Only %d access logs found (minimum %d required). Aborting retrain.", total, min_rows)
            return None

        logs = db.query(AccessLog).order_by(AccessLog.timestamp.asc()).all()
        rows: list[dict[str, float | int]] = []
        for log in logs:
            features = extract_features_from_db(
                db,
                user_id=log.user_id,
                ip_address=log.ip_address,
                timestamp=log.timestamp,
                result=log.result,
                confidence_score=log.confidence_score,
                similarity_score=log.similarity_score,
            )
            features["is_anomaly"] = 1 if log.anomaly_flag else 0
            rows.append(features)

        return pd.DataFrame(rows)
    finally:
        db.close()


def retrain(
    df: pd.DataFrame,
    *,
    model_kind: str = "random_forest",
    contamination: float = ML_CONTAMINATION,
    n_estimators: int = 300,
    seed: int = 42,
) -> tuple[object, dict[str, object]]:
    """Retrain the configured model and return metrics on a holdout split."""
    X = df[FEATURE_NAMES]
    y = df["is_anomaly"]

    anomaly_ratio = y.mean()
    logger.info("Anomaly ratio in data: %.4f (%d / %d)", anomaly_ratio, y.sum(), len(y))
    if y.sum() < 5:
        logger.warning("Very few anomalies (%d) - evaluation metrics may be unreliable.", y.sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y if y.sum() >= 2 else None,
    )

    model = train_anomaly_model(
        X_train,
        y_train,
        model_kind=model_kind,
        contamination=contamination,
        n_estimators=n_estimators,
        seed=seed,
    )
    scores = score_samples(model, X_test)
    threshold_metrics = select_threshold(
        scores,
        y_test,
        higher_is_more_anomalous=get_score_mode(model) == "probability",
    )
    threshold = threshold_metrics["threshold"]

    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    metrics["data_source"] = "production_db"
    metrics["n_total_rows"] = len(df)
    metrics["model_kind"] = model_kind
    metrics["recommended_threshold"] = round(float(threshold), 6)
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain model from production data.")
    parser.add_argument("--output", type=str, default="models_ml/isolation_forest.joblib")
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument(
        "--model-kind",
        type=str,
        default="random_forest",
        choices=("random_forest", "hist_gradient_boosting", "xgboost", "isolation_forest"),
    )
    parser.add_argument("--contamination", type=float, default=ML_CONTAMINATION)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Extracting features from database...")
    df = extract_features_from_db_bulk(min_rows=args.min_rows)
    if df is None:
        sys.exit(1)

    logger.info("Retraining on %d rows...", len(df))
    model, metrics = retrain(
        df,
        model_kind=args.model_kind,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )

    logger.info("\n%s", metrics["classification_report"])
    logger.info(
        "Precision: %.4f | Recall: %.4f | F1: %.4f | Threshold: %.6f",
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["recommended_threshold"],
    )

    model_path = save_model(model, args.output)
    logger.info("Model saved to %s", model_path)

    metrics_path = save_metrics(metrics, Path(args.output).parent)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
