from __future__ import annotations

import json
import os
import secrets
import hashlib
from typing import Optional, Dict, Any

from vibevoice_api.config import CONFIG


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _load_keystore(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"keys": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"keys": []}


def _save_keystore(path: str, data: Dict[str, Any]) -> None:
    _ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key(prefix: str = "sk-") -> str:
    return prefix + secrets.token_urlsafe(32)


def add_api_key(key: str, path: Optional[str] = None) -> None:
    ks_path = path or CONFIG.keystore_path
    data = _load_keystore(ks_path)
    hashed = _hash_key(key)
    keys = set(data.get("keys", []))
    keys.add(hashed)
    data["keys"] = sorted(keys)
    _save_keystore(ks_path, data)


def validate_api_key(key: Optional[str], path: Optional[str] = None) -> bool:
    if not CONFIG.require_api_key:
        return True
    if not key:
        return False
    ks_path = path or CONFIG.keystore_path
    data = _load_keystore(ks_path)
    hashed = _hash_key(key)
    return hashed in set(data.get("keys", []))

