"""Crossref adapter (broad DOI coverage incl. SSRN, book chapters) + DOI resolution for verification."""

from __future__ import annotations

from ..schemas import Record
from .base import get_json, hashed_key, http, norm_doi, strip_tags

BASE = "https://api.crossref.org"


def _record(m: dict) -> Record | None:
    title = (m.get("title") or [None])[0]
    doi = norm_doi(m.get("DOI"))
    if not title or not doi:
        return None
    parts = (m.get("issued") or {}).get("date-parts") or [[None]]
    authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in m.get("author", [])]
    insts = sorted({af["name"] for a in m.get("author", []) for af in a.get("affiliation", []) if af.get("name")})
    return Record(
        key=hashed_key("CR", doi),
        title=strip_tags(title) or title,
        year=parts[0][0],
        venue=(m.get("container-title") or [None])[0] or m.get("publisher"),
        doi=doi,
        authors=[a for a in authors if a],
        institutions=insts[:10],
        abstract=strip_tags(m.get("abstract")),
        cites=m.get("is-referenced-by-count") or 0,
        is_review="review" in (title or "").lower(),
        is_preprint=m.get("type") == "posted-content",
        language=m.get("language"),
        source="crossref",
    )


def search(query: str, from_year: int | None = None, n: int = 25) -> list[Record]:
    params: dict = {"query.bibliographic": query, "rows": min(n, 100)}
    if from_year:
        params["filter"] = f"from-pub-date:{from_year}"
    with http() as c:
        items = get_json(c, f"{BASE}/works", params)["message"].get("items", [])
    return [r for m in items if (r := _record(m))]


def resolve_title(doi: str) -> str | None:
    """Title registered for a DOI, or None if the DOI does not resolve."""
    with http() as c:
        r = c.get(f"{BASE}/works/{doi}")
    if r.status_code != 200:
        return None
    titles = r.json()["message"].get("title") or []
    return strip_tags(titles[0]) if titles else None
