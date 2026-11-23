"""Shared helpers for source adapters.

Adapter contract: `search(query: str, from_year: int | None, n: int) -> list[Record]`.
Adapters must build Records from API metadata only and must not raise on empty results.
"""

from __future__ import annotations

import hashlib
import os
import re
import time

import httpx

CONTACT = os.environ.get("CONTACT_EMAIL", "")
UA = f"research-assistant/0.1 (mailto:{CONTACT})" if CONTACT else "research-assistant/0.1"

_TAG = re.compile(r"<[^>]+>")


def http() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": UA}, timeout=30, follow_redirects=True)


def get_json(c: httpx.Client, url: str, params: dict | None = None, headers: dict | None = None, retries: int = 3):
    for attempt in range(retries):
        r = c.get(url, params=params, headers=headers)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return r.json()


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        d = d.removeprefix(p)
    return d or None


def strip_tags(s: str | None) -> str | None:
    return _TAG.sub("", s).strip() if s else s


def hashed_key(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode()).hexdigest()[:12]}"
