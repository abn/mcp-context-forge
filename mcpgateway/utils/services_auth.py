# -*- coding: utf-8 -*-
"""

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

"""

import base64
import hashlib
import json
import os
from typing import Optional, Dict, Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mcpgateway.config import settings


def get_key() -> bytes:
    """
    Generate a 32-byte AES encryption key derived from a passphrase.

    Returns:
        bytes: A 32-byte encryption key.

    Raises:
        ValueError: If the passphrase is not set or empty.
    """
    passphrase = settings.auth_encryption_secret
    if not passphrase:
        raise ValueError("AUTH_ENCRPYPTION_SECRET not set in environment.")
    return hashlib.sha256(passphrase.encode()).digest()  # 32-byte key


def encode_auth(auth_value: Dict[str, str]) -> Optional[str]:
    """
    Encrypt and encode an authentication dictionary into a compact base64-url string.

    Args:
        auth_value (Dict[str, str]): The authentication dictionary to encrypt and encode.

    Returns:
        Optional[str]: A base64-url-safe encrypted string, or None if input is empty or None.
    """
    if not auth_value: # Handles empty dict or if None was somehow passed
        return None
    plaintext = json.dumps(auth_value)
    key = get_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
    combined = nonce + ciphertext
    encoded = base64.urlsafe_b64encode(combined).rstrip(b"=")
    return encoded.decode()


def decode_auth(encoded_value: Optional[str]) -> Optional[Dict[Any, Any]]:
    """
    Decode and decrypt a base64-url-safe encrypted string back into the authentication dictionary.

    Args:
        encoded_value (Optional[str]): The encrypted base64-url string to decode and decrypt.

    Returns:
        Optional[Dict[Any, Any]]: The decrypted authentication dictionary, or None if input is None/empty or on error.
    """
    if not encoded_value:
        return None
    key = get_key()
    aesgcm = AESGCM(key)
    # Fix base64 padding
    padded = encoded_value + "=" * (-len(encoded_value) % 4)
    try:
        combined = base64.urlsafe_b64decode(padded)
        nonce = combined[:12]
        ciphertext = combined[12:]
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode())
    except Exception:  # pylint: disable=broad-except
        # Consider logging the exception here
        return None
