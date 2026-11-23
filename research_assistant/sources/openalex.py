"""OpenAlex adapter: the universal backbone (all fields, incl. humanities and non-English work)."""

from __future__ import annotations

from ..schemas import Record
from .base import get_json, http, norm_doi

BASE = "https://api.openalex.org"
SELECT = ",".join([
    "id", "doi", "title", "publication_year", "primary_location", "authorships",
    "abstract_inverted_index", "cited_by_count", "type", "language", "referenced_works",
])


def _abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


def _record(w: dict) -> Record:
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    return Record(
        key=w["id"].rsplit("/", 1)[-1],
        title=w.get("title") or "",
        year=w.get("publication_year"),
        venue=src.get("display_name"),
        doi=norm_doi(w.get("doi")),
        authors=[a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author")],
        institutions=sorted({i["display_name"] for a in w.get("authorships", []) for i in a.get("institutions", [])}),
        abstract=_abstract(w.get("abstract_inverted_index")),
        cites=w.get("cited_by_count") or 0,
        is_review=w.get("type") == "review",
        is_preprint=src.get("type") == "repository" or w.get("type") == "preprint",
        language=w.get("language"),
        source="openalex",
    )


def _works(params: dict) -> list[dict]:
    with http() as c:
        return get_json(c, f"{BASE}/works", {**params, "select": SELECT}).get("results", [])


def search(query: str, from_year: int | None = None, n: int = 25, sort: str | None = None) -> list[Record]:
    params: dict = {"search": query, "per_page": min(n, 200)}
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    if sort:
        params["sort"] = sort
    return [_record(w) for w in _works(params) if w.get("title")]


def landmarks(query: str, n: int = 15) -> list[Record]:
    """Most-cited works on the topic regardless of age."""
    return search(query, None, n, sort="cited_by_count:desc")


def cited_by(key: str, n: int = 10) -> list[Record]:
    return [_record(w) for w in _works({"filter": f"cites:{key}", "per_page": n, "sort": "cited_by_count:desc"})]


def references(key: str, n: int = 10) -> list[Record]:
    with http() as c:
        w = get_json(c, f"{BASE}/works/{key}", {"select": "referenced_works"})
    ids = [r.rsplit("/", 1)[-1] for r in w.get("referenced_works", [])][:50]
    if not ids:
        return []
    recs = [_record(x) for x in _works({"filter": f"openalex_id:{'|'.join(ids)}", "per_page": 50})]
    return sorted(recs, key=lambda r: r.cites, reverse=True)[:n]


def field_distribution(query: str) -> dict[str, int]:
    """Count of matching works per OpenAlex top-level field."""
    with http() as c:
        r = get_json(c, f"{BASE}/works", {"search": query, "group_by": "primary_topic.field.id"})
    return {g["key_display_name"]: g["count"] for g in r.get("group_by", []) if g.get("key_display_name")}


def get_by_doi(doi: str) -> Record | None:
    with http() as c:
        try:
            w = get_json(c, f"{BASE}/works/https://doi.org/{doi}", {"select": SELECT})
        except Exception:
            return None
    return _record(w) if w.get("id") else None
