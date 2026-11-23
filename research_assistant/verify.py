"""S11 verification: citation keys, DOI resolution, claim entailment, generic-language lint.

Produces `claim_status` (path -> verdict, used by the renderer to drop/hedge claims)
and `qc` (the qc_report.json payload).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz import fuzz

from . import prompts as P
from .config import rubric_cfg
from .llm import ask, usage_report
from .schemas import Claim, EntailBatch, Record
from .sources import crossref
from .state import State

log = logging.getLogger("research_assistant")
DOI_TITLE_MIN = 90
ENTAIL_BATCH = 40          # claims per verification call
ENTAIL_ABSTRACT_CHARS = 1200


def collect_claims(s: State) -> list[tuple[str, Claim]]:
    """Every cited claim in the report, with a stable path used by the renderer."""
    out: list[tuple[str, Claim]] = []
    d = s.get("draft")
    if d:
        for part in ("scholarly", "practical", "societal"):
            out += [(f"sig.{part}.{i}", c) for i, c in enumerate(getattr(d, f"significance_{part}"))]
    if s.get("landscape"):
        for ci, cl in enumerate(s["landscape"].clusters):
            out += [(f"cluster.{ci}.{j}", c) for j, c in enumerate(cl.summary)]
    for ri, r in enumerate(s.get("ranked", [])):
        out += [(f"dir.{ri}.{j}", c) for j, c in enumerate(r["direction"].novelty_argument)]
    if s.get("recommendation"):
        out += [(f"rec.{j}", c) for j, c in enumerate(s["recommendation"].rationale)]
    return out


def strip_invalid_cites(claims: list[tuple[str, Claim]], ledger: dict[str, Record]) -> list[str]:
    """Remove keys not in the ledger (mutates claims). Returns the stripped keys."""
    stripped = []
    for _, c in claims:
        bad = [k for k in c.cites if k not in ledger]
        stripped += bad
        c.cites = [k for k in dict.fromkeys(c.cites) if k in ledger]
    return stripped


def doi_title_matches(registered: str | None, ours: str) -> bool:
    return bool(registered) and fuzz.token_set_ratio(registered.lower(), ours.lower()) >= DOI_TITLE_MIN


def _has_evidence(c: Claim, ledger: dict[str, Record]) -> bool:
    return any(ledger[k].abstract for k in c.cites)


def entail_all(claims: list[Claim], ledger: dict[str, Record]) -> list[str]:
    """Verdict per claim. Claims are checked in batches; each cited abstract is sent once per batch.

    Pre-labels without an LLM call: no cites -> "uncited"; cited works lack abstracts -> "unverifiable".
    """
    verdicts = ["uncited" if not c.cites else "unverifiable" if not _has_evidence(c, ledger) else "pending"
                for c in claims]
    pending = [i for i, v in enumerate(verdicts) if v == "pending"]
    for start in range(0, len(pending), ENTAIL_BATCH):
        chunk = pending[start:start + ENTAIL_BATCH]
        keys = list(dict.fromkeys(k for i in chunk for k in claims[i].cites if ledger[k].abstract))
        evidence = "\n\n".join(f"[{k}] {ledger[k].title}. {ledger[k].abstract[:ENTAIL_ABSTRACT_CHARS]}" for k in keys)
        listing = "\n".join(f"{i}. {claims[i].text}  cites={claims[i].cites}" for i in chunk)
        try:
            out = ask(P.ENTAIL, f"CLAIMS:\n{listing}\n\nEVIDENCE:\n{evidence}", EntailBatch,
                      stage="verify", effort="low", bulk=True)
            got = {v.id: v.verdict for v in out.verdicts}
        except Exception as e:
            log.warning("batched entailment failed: %s", e)
            got = {}
        for i in chunk:
            verdicts[i] = got.get(i, "unverifiable")  # missing from the reply -> keep but flag
    return verdicts


def _check_doi(rec: Record) -> bool:
    try:
        return doi_title_matches(crossref.resolve_title(rec.doi), rec.title)
    except Exception:
        return False


def banned_hits(texts: list[str]) -> list[str]:
    phrases = rubric_cfg()["banned_phrases"]
    return sorted({p for t in texts for p in phrases if p in t.lower()})


def verify_node(s: State) -> dict:
    ledger = s["ledger"]
    claims = collect_claims(s)
    stripped = strip_invalid_cites(claims, ledger)

    # Non-claim citation lists get the same ledger check.
    for g in s.get("gaps", []):
        stripped += [k for k in g.supporting if k not in ledger]
        g.supporting = [k for k in g.supporting if k in ledger]
    for r in s.get("ranked", []):
        d = r["direction"]
        stripped += [k for k in d.closest_prior_work if k not in ledger]
        d.closest_prior_work = [k for k in d.closest_prior_work if k in ledger]

    verdicts = entail_all([c for _, c in claims], ledger)
    status = {path: v for (path, _), v in zip(claims, verdicts)}

    cited = sorted({k for _, c in claims for k in c.cites}
                   | {k for g in s.get("gaps", []) for k in g.supporting}
                   | {k for r in s.get("ranked", []) for k in r["direction"].closest_prior_work + r["novelty"].overlap_keys if k in ledger})
    with_doi = [ledger[k] for k in cited if ledger[k].doi]
    with ThreadPoolExecutor(8) as ex:
        doi_ok = dict(zip((r.key for r in with_doi), ex.map(_check_doi, with_doi)))

    texts = [c.text for _, c in claims] + ([s["draft"].summary] if s.get("draft") else [])
    texts += [r["direction"].research_question + " " + r["direction"].method_sketch for r in s.get("ranked", [])]

    n = len(claims) or 1
    counts = {v: verdicts.count(v) for v in ("supported", "partial", "unsupported", "uncited", "unverifiable")}
    th = rubric_cfg()["qc_thresholds"]
    unsupported_rate = (counts["unsupported"] + counts["uncited"]) / n
    doi_rate = (sum(doi_ok.values()) / len(doi_ok)) if doi_ok else 1.0
    qc = {
        "n_claims": len(claims),
        "claim_verdicts": counts,
        "unsupported_rate": round(unsupported_rate, 3),
        "stripped_citations": stripped,
        "n_cited_records": len(cited),
        "doi_checked": len(doi_ok),
        "doi_match_rate": round(doi_rate, 3),
        "doi_failures": [k for k, ok in doi_ok.items() if not ok],
        "banned_phrases_found": banned_hits(texts),
        "coverage": s.get("coverage", {}),
        "fields": s.get("fields", []),
        "profiles": s.get("profile", {}).get("names", []),
        "field_distribution": s.get("field_distribution", {}),
        "critic_verdicts": [c.verdict for c in s.get("critiques", [])],
        "revisions": s.get("revisions", 0),
        "notes": s.get("notes", []),
        "token_usage": usage_report(),
    }
    qc["low_confidence"] = (
        unsupported_rate > th["max_unsupported_claim_rate"]
        or doi_rate < th["min_doi_match_rate"]
        or len(ledger) < th["min_ledger_size"]
    )
    return {"claim_status": status, "qc": qc}
