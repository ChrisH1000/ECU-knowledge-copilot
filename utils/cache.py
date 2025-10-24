"""Simple disk-backed response caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _hash_key(payload: Dict[str, Any]) -> str:
    """Create a deterministic cache key for the provided payload."""

    digest_input = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


@dataclass
class ResponseCache:
    """Persists question/answer pairs with basic metadata."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> Dict[str, Any]:
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _write(self, payload: Dict[str, Any]) -> None:
        serialized = json.dumps(payload, indent=2)
        self.path.write_text(serialized + "\n", encoding="utf-8")

    def get(self, signature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a cached entry for the signature if present."""

        key = _hash_key(signature)
        data = self._read()
        return data.get(key)

    def set(self, signature: Dict[str, Any], value: Dict[str, Any]) -> None:
        """Persist an entry for later reuse."""

        key = _hash_key(signature)
        data = self._read()
        data[key] = value
        self._write(data)