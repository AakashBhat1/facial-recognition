"""Conservative image augmentation for enrollment.

Augmentations are kept mild to avoid generating poor-quality embeddings:
- Horizontal flip
- Slight brightness adjustment (+/- 15%)
- Minor rotation (+/- 5 degrees)
"""

from __future__ import annotations

import cv2
import numpy as np


def _flip_horizontal(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def _adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel values by *factor*, clipping to [0, 255]."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *image* by *angle* degrees around its center."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """Generate conservative augmented versions of *image*.

    Returns a list of 3 augmented images (flip, brighter, rotated).
    The original is **not** included in the output.
    """
    return [
        _flip_horizontal(image),
        _adjust_brightness(image, 1.15),
        _rotate(image, 5.0),
    ]


def augment_batch(images: list[np.ndarray]) -> list[np.ndarray]:
    """Augment each image and return a flat list of original + augmented images."""
    result: list[np.ndarray] = []
    for image in images:
        result.append(image)
        result.extend(augment_image(image))
    return result
