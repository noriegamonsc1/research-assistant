"""Semantic Scholar adapter (strong for CS/engineering; arXiv preprints included)."""

from __future__ import annotations

import os

from ..schemas import Record
from .base import get_json, http, norm_doi

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,venue,externalIds,authors,abstract,citationCount,publicationTypes"


def search(query: str, from_year: int | None = None, n: int = 25) -> list[Record]:
    params: dict = {"query": query, "limit": min(n, 100), "fields": FIELDS}
    if from_year:
        params["year"] = f"{from_year}-"
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else None
    with http() as c:
        data = get_json(c, f"{BASE}/paper/search", params, headers=headers).get("data", [])
    out = []
    for p in data:
        if not p.get("title"):
            continue
        ext = p.get("externalIds") or {}
        types = p.get("publicationTypes") or []
        out.append(Record(
            key=f"S2_{p['paperId']}",
            title=p["title"],
            year=p.get("year"),
            venue=p.get("venue") or None,
            doi=norm_doi(ext.get("DOI")),
            authors=[a["name"] for a in p.get("authors", []) if a.get("name")],
            abstract=p.get("abstract"),
            cites=p.get("citationCount") or 0,
            is_review="Review" in types,
            is_preprint="ArXiv" in ext and not ext.get("DOI"),
            source="semanticscholar",
        ))
    return out
