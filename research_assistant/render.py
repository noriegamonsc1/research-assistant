"""Deterministic Markdown rendering. The bibliography comes from ledger metadata, never from LLM text."""

from __future__ import annotations

from .schemas import Claim, Record
from .state import State

DROP = {"unsupported", "uncited"}


class Refs:
    """Numbers references in order of first citation."""

    def __init__(self, ledger: dict[str, Record], doi_failures: list[str]):
        self.ledger, self.order, self.bad_doi = ledger, [], set(doi_failures)

    def cite(self, keys: list[str]) -> str:
        nums = []
        for k in keys:
            if k not in self.ledger:
                continue
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        return f" [{', '.join(map(str, sorted(set(nums))))}]" if nums else ""

    @staticmethod
    def link(r: Record) -> str:
        if r.doi:
            return f"https://doi.org/{r.doi}"
        if r.key.startswith("W"):
            return f"https://openalex.org/{r.key}"
        if r.key.startswith("PMID"):
            return f"https://pubmed.ncbi.nlm.nih.gov/{r.key[4:]}/"
        if r.key.startswith("S2_"):
            return f"https://www.semanticscholar.org/paper/{r.key[3:]}"
        return ""

    def bibliography(self) -> str:
        lines = []
        for i, k in enumerate(self.order, 1):
            r = self.ledger[k]
            au = ", ".join(r.authors[:3]) + (" et al." if len(r.authors) > 3 else "") or "Anon."
            flags = " [preprint]" if r.is_preprint else ""
            flags += " [DOI not verified]" if k in self.bad_doi else ""
            venue = f" *{r.venue}*." if r.venue else ""
            lines.append(f"{i}. {au} ({r.year or 'n.d.'}). {r.title}.{venue} {self.link(r)}{flags}")
        return "\n".join(lines)


def _claim(path: str, c: Claim, status: dict[str, str], refs: Refs) -> str | None:
    st = status.get(path, "supported")
    if st in DROP:
        return None
    suffix = {"partial": " *(partially supported)*", "unverifiable": " *(not verified: no abstract available)*"}.get(st, "")
    return f"{c.text}{refs.cite(c.cites)}{suffix}"


def _claims(prefix: str, claims: list[Claim], status, refs) -> list[str]:
    return [t for i, c in enumerate(claims) if (t := _claim(f"{prefix}.{i}", c, status, refs))]


def render(s: State) -> str:
    ledger, qc, status = s["ledger"], s.get("qc", {}), s.get("claim_status", {})
    refs = Refs(ledger, qc.get("doi_failures", []))
    ctx, prof = s["context"], s["profile"]
    md: list[str] = []

    if qc.get("low_confidence"):
        md.append("> **LOW CONFIDENCE.** Verification thresholds were not met; see qc_report.json.\n")
    md.append(f"# Research directions: {s['topic']}\n")
    if s.get("draft"):
        md.append(s["draft"].summary + "\n")

    md.append("**Field:** " + (" + ".join(s.get("fields", [])) or "unresolved")
              + f" (profiles: {', '.join(prof.get('names', [])) or 'defaults'}; window: last {prof['recency_years']} years)  ")
    md.append(f"**Researcher context:** stage={ctx.stage or 'n/s'}, horizon={ctx.horizon_months or 'n/s'} months, "
              f"resources={', '.join(ctx.resources) or 'n/s'}, methods={', '.join(ctx.methods) or 'n/s'}  ")
    cov = s.get("coverage", {})
    md.append(f"**Corpus:** {cov.get('n_records', 0)} records; sources {cov.get('by_source', {})}; "
              f"languages {cov.get('by_language', {})}\n")
    if prof.get("low_digital_coverage"):
        md.append("> **Coverage caveat.** Much of this field's scholarship (monographs, archives, non-indexed "
                  "journals) is poorly represented in bibliographic APIs. Treat gaps as hypotheses and check with a "
                  "subject librarian, key monographs, and relevant archives before committing.\n")

    # 1. Significance
    md.append("## 1. Significance\n")
    d = s.get("draft")
    for part in ("scholarly", "practical", "societal"):
        items = _claims(f"sig.{part}", getattr(d, f"significance_{part}"), status, refs) if d else []
        md.append(f"**{part.capitalize()}.** " + (" ".join(items) or "_No verifiable claim survived checking._") + "\n")

    # 2. State of the art
    md.append("## 2. State of the art\n")
    for ci, cl in enumerate(s["landscape"].clusters):
        md.append(f"### {cl.label}\n")
        md += [f"- {t}" for t in _claims(f"cluster.{ci}", cl.summary, status, refs)]
        if cl.methods_or_frameworks:
            md.append(f"- *Dominant methods/frameworks:* {', '.join(cl.methods_or_frameworks)}")
        if cl.landmark_keys:
            md.append("- *Landmark works:* " + "; ".join(f"{ledger[k].title}{refs.cite([k])}" for k in cl.landmark_keys))
        md.append("")
    g = s.get("groups", {})
    if g:
        md.append("**Most active (by count in corpus):** authors: "
                  + ", ".join(f"{a} ({n})" for a, n in g.get("authors", [])[:6])
                  + "; institutions: " + ", ".join(f"{i} ({n})" for i, n in g.get("institutions", [])[:6]) + "\n")

    # 3. Gaps
    md.append("## 3. Gaps and open problems\n")
    for i, gap in enumerate(s["gaps"]):
        md.append(f"{i + 1}. **{gap.statement}**{refs.cite(gap.supporting)} — *{gap.kind}; unsolved because of "
                  f"{gap.why_unsolved}.* {gap.explanation}")
    md.append("")

    # 4. Directions
    md.append("## 4. Candidate directions\n")
    if not s.get("ranked"):
        md.append("_No direction survived the novelty check and critique._\n")
    for ri, r in enumerate(s.get("ranked", [])[:5]):
        dr, nv, cr = r["direction"], r["novelty"], r["critique"]
        md.append(f"### D{ri + 1}. {dr.title}  \n*score {r['score']} · risk {dr.risk} · novelty check: {nv.status}*\n")
        md.append(f"- **Research question:** {dr.research_question}")
        md.append(f"- **Gaps addressed:** {', '.join(str(i + 1) for i in dr.gaps_addressed) or 'n/a'}")
        arg = _claims(f"dir.{ri}", dr.novelty_argument, status, refs)
        md.append("- **Novelty:** " + (" ".join(arg) or "_argument did not survive verification_")
                  + (f" Differs from closest work{refs.cite(nv.overlap_keys)}: {nv.difference}" if nv.status == "partial" else ""))
        if dr.closest_prior_work:
            md.append(f"- **Closest prior work:**{refs.cite(dr.closest_prior_work)}")
        horizon = ctx.horizon_months
        fit = "" if not horizon else (" ✓ within horizon" if dr.est_months <= horizon else f" ⚠ exceeds {horizon}-month horizon")
        md.append(f"- **Feasibility:** ~{dr.est_months} months{fit}; needs {', '.join(dr.resources_needed)}")
        md.append(f"- **Method sketch:** {dr.method_sketch}")
        md.append(f"- **Risks:** {dr.risk_notes}")
        md.append(f"- **Critic** ({cr.verdict}; novelty {cr.novelty}/5, feasibility {cr.feasibility}/5, "
                  f"significance {cr.significance}/5, specificity {cr.specificity}/5): {cr.notes}\n")

    # 5. Recommendation
    md.append("## 5. Recommendation\n")
    rec = s.get("recommendation")
    if rec:
        md.append("**Top pick(s):** " + ", ".join(f"D{i + 1} — {s['ranked'][i]['direction'].title}" for i in rec.chosen) + "\n")
        md += [f"- {t}" for t in _claims("rec", rec.rationale, status, refs)]
        md.append(f"\n**What would change this recommendation:** {rec.what_would_change_it}\n")
    else:
        md.append("_No recommendation: no direction survived critique._\n")

    if s.get("notes"):
        md.append("## Run notes\n" + "\n".join(f"- {n}" for n in s["notes"]) + "\n")
    md.append("## References\n" + refs.bibliography())
    return "\n".join(md)


def render_node(s: State) -> dict:
    return {"report_md": render(s)}


def stop_early_node(s: State) -> dict:
    """Gate: too little literature after every search round. Skip the analysis stages entirely
    and return a short report that says so, lists what was found, and suggests next steps."""
    from .llm import usage_report  # local import keeps render free of API code at import time

    ledger = s.get("ledger", {})
    refs = Refs(ledger, [])
    found = [f"- {r.title}{refs.cite([k])}" for k, r in list(ledger.items())[:15]]
    queries = [f"- {q.source}: {q.text}" for q in s.get("queries", [])]
    md = [
        "> **STOPPED EARLY.** Too little indexed literature was found to support a reliable analysis, "
        "so the remaining stages were skipped to avoid spending tokens on thin evidence.\n",
        f"# Research directions: {s.get('topic', '')}\n",
        f"Only **{len(ledger)}** relevant records were found after {s.get('retrieval_rounds', 0)} search rounds "
        f"(field: {' + '.join(s.get('fields', [])) or 'unresolved'}).\n",
        "## What to try next\n",
        "- Broaden the topic, or name the field explicitly with `--field`.",
        "- Add synonyms or the terms used in the field's own literature.",
        "- For fields with thin digital coverage (humanities, regional studies), consult a subject librarian, "
        "key monographs and archives; this tool can only see what bibliographic APIs index.\n",
        "## What was found\n", *(found or ["_Nothing relevant._"]), "",
        "## Queries tried\n", *queries, "",
        "## References\n" + refs.bibliography(),
    ]
    qc = {
        "stopped_early": True,
        "reason": f"{len(ledger)} records < gates.min_ledger_to_continue",
        "low_confidence": True,
        "coverage": s.get("coverage", {}),
        "fields": s.get("fields", []),
        "notes": s.get("notes", []),
        "token_usage": usage_report(),
    }
    return {"report_md": "\n".join(md), "qc": qc}
