import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler

from config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

access_logger = logging.getLogger("access")
access_logger.setLevel(logging.INFO)
access_logger.propagate = False

security_logger = logging.getLogger("security")
security_logger.setLevel(logging.WARNING)
security_logger.propagate = False


def _attach_rotating_handler_if_missing(
    logger: logging.Logger,
    file_path: str,
    formatter: logging.Formatter,
) -> None:
    """Attach one rotating handler per target file to avoid duplicate logs on reload."""
    target = os.path.abspath(file_path)
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            current = os.path.abspath(getattr(handler, "baseFilename", ""))
            if current == target:
                return

    rotating = RotatingFileHandler(
        target,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    rotating.setFormatter(formatter)
    logger.addHandler(rotating)


_attach_rotating_handler_if_missing(
    access_logger,
    os.path.join(LOG_DIR, "access.jsonl"),
    logging.Formatter("%(message)s"),
)
_attach_rotating_handler_if_missing(
    security_logger,
    os.path.join(LOG_DIR, "security.log"),
    logging.Formatter("%(asctime)s %(levelname)s %(message)s"),
)


def log_access_event(
    log_id: int,
    user_id: int | None,
    user_name: str,
    action: str,
    result: str,
    confidence_score: float,
    similarity_score: float,
    locker_id: str,
    ip_address: str,
    anomaly_flag: bool,
) -> None:
    entry = {
        "log_id": log_id,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "user_id": user_id,
        "user_name": user_name,
        "action": action,
        "result": result,
        "confidence_score": confidence_score,
        "similarity_score": round(similarity_score, 4),
        "locker_id": locker_id,
        "ip_address": ip_address,
        "anomaly_flag": anomaly_flag,
    }
    access_logger.info(json.dumps(entry))

    if result == "FAILURE" or anomaly_flag:
        security_logger.warning(json.dumps(entry))


def log_alert(alert_type: str, severity: str, description: str) -> None:
    security_logger.warning(f"ALERT [{severity}] {alert_type}: {description}")
