"""Evidence Ledger: dedupe, merge, score and trim retrieved records.

The ledger is the single source of truth for citations. Its keys are the only
thing the LLM may cite.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date
from itertools import combinations

from .schemas import Facet, Record

_WS = re.compile(r"[^a-z0-9]+")


def norm_title(t: str) -> str:
    return _WS.sub(" ", t.lower()).strip()


def _merge(a: Record, b: Record) -> Record:
    """Merge b into a; prefer richer metadata, keep a's key (first-seen)."""
    a = a.model_copy(deep=True)
    a.abstract = a.abstract if a.abstract and len(a.abstract) >= len(b.abstract or "") else b.abstract
    a.cites = max(a.cites, b.cites)
    a.doi = a.doi or b.doi
    a.venue = a.venue or b.venue
    a.year = a.year or b.year
    a.authors = a.authors or b.authors
    a.institutions = sorted(set(a.institutions) | set(b.institutions))
    a.is_review = a.is_review or b.is_review
    a.language = a.language or b.language
    a.found_by = list(dict.fromkeys(a.found_by + b.found_by))
    if b.source not in a.source.split("+"):
        a.source = f"{a.source}+{b.source}"
    return a


def add_records(ledger: dict[str, Record], records: list[Record], tag: str) -> dict[str, str]:
    """Dedupe by DOI, then normalized title, then key (mutates `ledger`).

    Returns {incoming_key: ledger_key} so callers can remap keys of merged duplicates.
    """
    by_doi = {r.doi: k for k, r in ledger.items() if r.doi}
    by_title = {norm_title(r.title): k for k, r in ledger.items()}
    mapping: dict[str, str] = {}
    for r in records:
        r = r.model_copy(update={"found_by": list(dict.fromkeys(r.found_by + [tag]))})
        nt = norm_title(r.title)
        existing = (by_doi.get(r.doi) if r.doi else None) or by_title.get(nt) or (r.key if r.key in ledger else None)
        if existing:
            ledger[existing] = _merge(ledger[existing], r)
            mapping[r.key] = existing
            continue
        ledger[r.key] = r
        if r.doi:
            by_doi[r.doi] = r.key
        by_title[nt] = r.key
        mapping[r.key] = r.key
    return mapping


def _terms(facets: list[Facet]) -> list[str]:
    return [t.lower() for f in facets for t in [f.name, *f.terms]]


def _relevance(r: Record, terms: list[str]) -> float:
    text = f"{r.title} {r.abstract or ''}".lower()
    hits = sum(1 for t in terms if t in text)
    return min(1.0, hits / max(3, len(terms) * 0.25))


def score_and_trim(ledger: dict[str, Record], facets: list[Facet], profile: dict, keep: set[str] = frozenset()) -> dict[str, Record]:
    """Score = relevance x recency x field-normalized citation percentile (+ review bonus).

    Landmark quota: the top 10% by citations bypass the recency penalty, so seminal
    older work survives in fast-moving fields too.
    """
    if not ledger:
        return ledger
    this_year = date.today().year
    window = profile["recency_years"]
    terms = _terms(facets)
    cpy = {k: r.cites / max(1, this_year - (r.year or this_year) + 1) for k, r in ledger.items()}
    ranked = sorted(cpy.values())
    pct = {k: ranked.index(v) / max(1, len(ranked) - 1) for k, v in cpy.items()}
    top_cited = set(sorted(ledger, key=lambda k: ledger[k].cites, reverse=True)[: max(3, len(ledger) // 10)])

    for k, r in ledger.items():
        age = this_year - (r.year or this_year)
        recency = 1.0 if age <= window else math.exp(-(age - window) / window)
        if k in top_cited or profile.get("landmark_any_age"):
            recency = max(recency, 0.8)
        rel = _relevance(r, terms)
        r.score = round((0.2 + rel) * recency * (0.5 + pct[k]) + (0.15 if r.is_review else 0.0), 4)

    n = profile.get("max_records", 100)
    keys = sorted(ledger, key=lambda k: ledger[k].score, reverse=True)[:n]
    keys = list(dict.fromkeys(keys + [k for k in keep if k in ledger]))
    return {k: ledger[k] for k in keys}


def digest(ledger: dict[str, Record], keys: list[str] | None = None, abstract_chars: int = 600) -> str:
    """Compact text view of the ledger for LLM prompts."""
    lines = []
    for k in keys or list(ledger):
        r = ledger[k]
        flags = ",".join(f for f, on in (("review", r.is_review), ("preprint", r.is_preprint)) if on)
        head = f"[{k}] {r.title} ({r.year or 'n.d.'}; {r.venue or 'venue n/a'}; cites={r.cites}{'; ' + flags if flags else ''})"
        body = (r.abstract or "(no abstract)")[:abstract_chars]
        lines.append(f"{head}\n  {body}")
    return "\n".join(lines)


def top_groups(ledger: dict[str, Record], n: int = 8) -> dict[str, list[tuple[str, int]]]:
    """Active authors/institutions computed from metadata (not from LLM memory)."""
    authors = Counter(a for r in ledger.values() for a in r.authors[:5])
    insts = Counter(i for r in ledger.values() for i in r.institutions)
    return {"authors": authors.most_common(n), "institutions": insts.most_common(n)}


def facet_matrix(ledger: dict[str, Record], facets: list[Facet]) -> list[tuple[str, str, int]]:
    """Co-occurrence counts for each facet pair; sparse cells hint at structural gaps."""
    texts = [f"{r.title} {r.abstract or ''}".lower() for r in ledger.values()]
    has = {f.name: {i for i, t in enumerate(texts) if any(x.lower() in t for x in [f.name, *f.terms])} for f in facets}
    return sorted(((a, b, len(has[a] & has[b])) for a, b in combinations(has, 2)), key=lambda x: x[2])


def coverage(ledger: dict[str, Record]) -> dict:
    decades = Counter(f"{(r.year // 10) * 10}s" for r in ledger.values() if r.year)
    sources = Counter(s for r in ledger.values() for s in r.source.split("+"))
    langs = Counter(r.language or "unknown" for r in ledger.values())
    return {
        "n_records": len(ledger),
        "with_abstract": sum(1 for r in ledger.values() if r.abstract),
        "reviews": sum(1 for r in ledger.values() if r.is_review),
        "by_source": dict(sources),
        "by_language": dict(langs),
        "by_decade": dict(sorted(decades.items())),
    }
