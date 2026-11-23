"""PubMed adapter via NCBI E-utilities (esearch -> efetch XML)."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from ..schemas import Record
from .base import CONTACT, get_json, http, norm_doi

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _common() -> dict:
    p = {"tool": "research-assistant"}
    if CONTACT:
        p["email"] = CONTACT
    if os.environ.get("NCBI_API_KEY"):
        p["api_key"] = os.environ["NCBI_API_KEY"]
    return p


def _text(el: ET.Element | None) -> str | None:
    return "".join(el.itertext()).strip() if el is not None else None


def _parse(article: ET.Element) -> Record | None:
    pmid = _text(article.find(".//PMID"))
    title = _text(article.find(".//ArticleTitle"))
    if not pmid or not title:
        return None
    abstract = " ".join(_text(a) or "" for a in article.findall(".//Abstract/AbstractText")) or None
    year = _text(article.find(".//JournalIssue/PubDate/Year")) or (_text(article.find(".//PubDate/MedlineDate")) or "")[:4]
    authors = []
    for a in article.findall(".//AuthorList/Author"):
        last, fore = _text(a.find("LastName")), _text(a.find("ForeName"))
        if last:
            authors.append(f"{fore} {last}" if fore else last)
    insts = sorted({_text(x) for x in article.findall(".//AffiliationInfo/Affiliation") if _text(x)})
    doi = next((_text(x) for x in article.findall(".//ArticleIdList/ArticleId") if x.get("IdType") == "doi"), None)
    ptypes = [_text(x) or "" for x in article.findall(".//PublicationTypeList/PublicationType")]
    return Record(
        key=f"PMID{pmid}",
        title=title,
        year=int(year) if year and year.isdigit() else None,
        venue=_text(article.find(".//Journal/Title")),
        doi=norm_doi(doi),
        authors=authors,
        institutions=insts[:10],
        abstract=abstract,
        cites=0,  # PubMed has no citation counts; ledger merge fills from OpenAlex when DOIs match
        is_review=any(t in ("Review", "Systematic Review", "Meta-Analysis") for t in ptypes),
        language=(_text(article.find(".//Language")) or "")[:2] or None,
        source="pubmed",
    )


def search(query: str, from_year: int | None = None, n: int = 25) -> list[Record]:
    params = {**_common(), "db": "pubmed", "term": query, "retmax": n, "retmode": "json", "sort": "relevance"}
    if from_year:
        params.update({"datetype": "pdat", "mindate": str(from_year), "maxdate": "3000"})
    with http() as c:
        ids = get_json(c, f"{BASE}/esearch.fcgi", params)["esearchresult"].get("idlist", [])
        if not ids:
            return []
        r = c.get(f"{BASE}/efetch.fcgi", params={**_common(), "db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        r.raise_for_status()
    root = ET.fromstring(r.content)
    return [rec for a in root.findall(".//PubmedArticle") if (rec := _parse(a))]
