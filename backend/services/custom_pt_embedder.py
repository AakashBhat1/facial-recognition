"""PyTorch checkpoint embedder runtime for custom face models.

This module loads a ``face_model_scratch`` style checkpoint (best.pt/last.pt)
and extracts embeddings from detected face crops while preserving the existing
backend pipeline contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_runtime: dict[str, Any] | None = None


def _load_torch_modules():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torchvision.models import mobilenet_v2, resnet50
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "CUSTOM_PT_EMBEDDER_ENABLED is true, but torch/torchvision are not installed"
        ) from exc
    return torch, nn, F, mobilenet_v2, resnet50


def _build_embedding_model(backbone: str, embedding_dim: int):
    torch, nn, F, mobilenet_v2, resnet50 = _load_torch_modules()

    class FaceEmbeddingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if backbone == "resnet50":
                base = resnet50(weights=None)
                in_features = base.fc.in_features
                base.fc = nn.Identity()
                self.backbone = base
            elif backbone == "mobilenet_v2":
                base = mobilenet_v2(weights=None)
                in_features = base.classifier[1].in_features
                base.classifier = nn.Identity()
                self.backbone = base
            else:
                raise ValueError(f"Unsupported checkpoint backbone: {backbone}")

            self.embedding = nn.Linear(in_features, embedding_dim)

        def forward(self, x):
            features = self.backbone(x)
            emb = self.embedding(features)
            return F.normalize(emb, p=2, dim=1)

    return FaceEmbeddingModel(), torch


def _load_checkpoint_payload(path: Path):
    torch, _, _, _, _ = _load_torch_modules()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        logger.warning(
            "Current torch version does not support weights_only loading; "
            "falling back to standard torch.load for trusted checkpoint: %s",
            path,
        )
        return torch.load(path, map_location="cpu")


def _get_runtime() -> dict[str, Any]:
    global _runtime

    if _runtime is not None:
        return _runtime

    if not config.CUSTOM_PT_EMBEDDER_ENABLED:
        raise RuntimeError("Custom PT embedder is disabled")

    checkpoint_path = Path(config.CUSTOM_PT_CHECKPOINT_PATH)
    payload = _load_checkpoint_payload(checkpoint_path)

    model_state = payload.get("model_state")
    cfg = payload.get("config") or {}
    if model_state is None:
        raise ValueError(f"Invalid checkpoint (missing model_state): {checkpoint_path}")

    backbone = str(cfg.get("backbone", "resnet50"))
    embedding_dim = int(cfg.get("embedding_dim", config.EMBEDDING_DIM))
    image_size = int(cfg.get("image_size", 112))

    model, torch = _build_embedding_model(backbone=backbone, embedding_dim=embedding_dim)
    model.load_state_dict(model_state)
    model.eval()

    _runtime = {
        "torch": torch,
        "model": model,
        "image_size": image_size,
        "embedding_dim": embedding_dim,
        "checkpoint": str(checkpoint_path),
    }

    logger.info(
        "Custom PT embedder loaded from %s (backbone=%s, embedding_dim=%d, image_size=%d)",
        checkpoint_path,
        backbone,
        embedding_dim,
        image_size,
    )

    return _runtime


def preload_if_enabled() -> None:
    """Load checkpoint eagerly during startup if enabled."""
    if not config.CUSTOM_PT_EMBEDDER_ENABLED:
        return
    _get_runtime()


def is_enabled() -> bool:
    """Return whether custom PT embedding runtime is enabled."""
    return config.CUSTOM_PT_EMBEDDER_ENABLED


def _crop_face(image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = max(float(x2 - x1), 1.0)
    box_h = max(float(y2 - y1), 1.0)
    # Small margin keeps chin/forehead context while avoiding large background.
    margin_x = box_w * 0.15
    margin_y = box_h * 0.15

    nx1 = max(int(round(x1 - margin_x)), 0)
    ny1 = max(int(round(y1 - margin_y)), 0)
    nx2 = min(int(round(x2 + margin_x)), w)
    ny2 = min(int(round(y2 + margin_y)), h)

    if nx2 <= nx1 or ny2 <= ny1:
        raise ValueError("Invalid face bbox for custom checkpoint embedding")

    crop = image[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        raise ValueError("Empty face crop for custom checkpoint embedding")
    return crop


def extract_embedding(image: np.ndarray, bbox: tuple[float, float, float, float]) -> list[float]:
    """Extract an L2-normalized embedding from image+bbox using custom checkpoint."""
    runtime = _get_runtime()
    torch = runtime["torch"]
    model = runtime["model"]
    image_size = int(runtime["image_size"])

    crop = _crop_face(image, bbox)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5

    with torch.no_grad():
        embedding = model(tensor).squeeze(0).cpu().numpy()

    return embedding.astype(np.float32).tolist()
