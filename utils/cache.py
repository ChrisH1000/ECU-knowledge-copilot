"""Simple disk-backed response caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _hash_key(payload: Dict[str, Any]) -> str:
    """Create a deterministic cache key for the provided payload.

    Converts the payload dict to canonical JSON (sorted keys, no whitespace)
    and hashes with SHA-256 to create a unique, stable identifier.

    Args:
        payload: Dictionary containing question and citation data

    Returns:
        str: 64-character hexadecimal hash
    """
    # Serialize with sorted keys for deterministic output
    digest_input = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


@dataclass
class ResponseCache:
    """Persists question/answer pairs with basic metadata.

    Stores cached responses in a JSON file keyed by content hash. Avoids
    redundant LLM calls when the same question is asked with the same
    retrieved context.

    Attributes:
        path: Filesystem location for the cache JSON file
    """

    path: Path

    def __post_init__(self) -> None:
        """Initialize the cache file and ensure parent directory exists."""
        self.path = Path(self.path)
        # Create cache directory if needed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize empty cache if file doesn't exist
        if not self.path.exists():
            self._write({})

    def _read(self) -> Dict[str, Any]:
        """Load cache data from disk.

        Returns:
            dict: Parsed cache contents (empty dict if file is empty)
        """
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _write(self, payload: Dict[str, Any]) -> None:
        """Write cache data to disk with pretty formatting.

        Args:
            payload: Complete cache dictionary to persist
        """
        serialized = json.dumps(payload, indent=2)
        self.path.write_text(serialized + "\n", encoding="utf-8")

    def get(self, signature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a cached entry for the signature if present.

        Args:
            signature: Dict containing question and citation data

        Returns:
            dict or None: Cached response if found, else None
        """
        key = _hash_key(signature)
        data = self._read()
        return data.get(key)

    def set(self, signature: Dict[str, Any], value: Dict[str, Any]) -> None:
        """Persist an entry for later reuse.

        Loads current cache, adds the new entry under a hashed key,
        and writes back to disk.

        Args:
            signature: Dict containing question and citation data (used as key)
            value: Dict containing answer and citations to cache
        """
        key = _hash_key(signature)
        data = self._read()
        data[key] = value
        self._write(data)