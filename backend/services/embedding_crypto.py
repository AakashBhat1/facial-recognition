"""AES-256-GCM encryption/decryption for face embeddings at rest.

When ``EMBEDDING_ENCRYPTION_ENABLED`` is True, all embeddings are
stored as encrypted base-64 blobs instead of raw JSON float arrays.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os

import numpy as np

import config

logger = logging.getLogger(__name__)

# 96-bit nonce for GCM
_NONCE_SIZE = 12
# AES-256-GCM tag length
_TAG_SIZE = 16


def _derive_key() -> bytes:
    """Derive a 256-bit key from config or SECRET_KEY via SHA-256."""
    raw = config.EMBEDDING_ENCRYPTION_KEY or config.SECRET_KEY
    return hashlib.sha256(raw.encode("utf-8")).digest()


def is_enabled() -> bool:
    """Return True if embedding encryption is turned on."""
    return config.EMBEDDING_ENCRYPTION_ENABLED


def encrypt_embedding(vector: list[float]) -> str:
    """Encrypt a float vector and return a base-64 string suitable for DB storage.

    Format: base64(nonce‖ciphertext‖tag)
    """
    if not is_enabled():
        return json.dumps(vector)

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        logger.warning("cryptography package not installed — storing embedding unencrypted")
        return json.dumps(vector)

    key = _derive_key()
    nonce = os.urandom(_NONCE_SIZE)
    plaintext = json.dumps(vector).encode("utf-8")

    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, None)  # ct includes tag

    return base64.b64encode(nonce + ct).decode("ascii")


def decrypt_embedding(stored: str) -> list[float]:
    """Decrypt a stored embedding string back to a float vector.

    Handles both encrypted (base-64) and legacy unencrypted (JSON) formats
    transparently.
    """
    # Legacy unencrypted format starts with '['
    if stored.startswith("["):
        return json.loads(stored)

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        # If we can't decrypt, try plain JSON parse as fallback
        return json.loads(stored)

    raw = base64.b64decode(stored)
    nonce = raw[:_NONCE_SIZE]
    ct = raw[_NONCE_SIZE:]

    key = _derive_key()
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ct, None)
    return json.loads(plaintext.decode("utf-8"))
