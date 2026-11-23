"""Pipeline stages S0-S10. Each node takes the graph state and returns a partial update.

S11 (verify) lives in verify.py; rendering in render.py.

Token budget notes:
- One shared paper digest (built once in `retrieve`) is sent as a cached system block to
  landscape, gaps, directions, rank and write, instead of five differently-sized copies.
- Effort is `low`/`medium` except where judgment matters most (directions, critique).
- Novelty judging is one batched call; a revision round regenerates only the directions
  the critic flagged and re-critiques only those.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from . import prompts as P
from .config import all_openalex_fields, build_profile, profiles_for_fields, rubric_cfg, stage_weights
from .ledger import add_records, coverage, digest, facet_matrix, score_and_trim, top_groups
from .llm import ask
from .schemas import (
    Critiques, Directions, Facets, FieldGuess, Gaps, Intake, Landscape, NoveltyBatch, NoveltyVerdict, QueryPlan,
    Recommendation, ReportDraft,
)
from .sources import REGISTRY, openalex, semanticscholar
from .state import State

log = logging.getLogger("research_assistant")
MAX_QUERIES_PER_SOURCE = 4
MIN_USABLE = 3            # revise only if fewer directions than this are usable after critique
CRITIC_ABSTRACT_CHARS = 200
MAX_RETRIEVAL_ROUNDS = 2
RISK_LEVEL = {"low": 0, "medium": 1, "high": 2}
SHARED_RECORDS = 60       # papers in the shared, cached digest
SHARED_ABSTRACT_CHARS = 500
NOVELTY_ABSTRACT_CHARS = 300


def _shared(s: State) -> str:
    return s["shared_digest"]


def _field_label(s: State) -> str:
    return " + ".join(s["fields"]) or "the relevant field"


def _ctx(s: State) -> str:
    return s["context"].model_dump_json()


# ------------------------------------------------------------------- S0-S2 --


def intake(s: State) -> dict:
    out = ask(P.INTAKE, s["raw_input"], Intake, stage="intake", effort="low")
    return {
        "topic": out.topic_core,
        "field_hint": s.get("user_field") or out.field_hint,
        "context": out.context,
        "retrieval_rounds": 0,
        "revisions": 0,
    }


def decompose(s: State) -> dict:
    return {"facets": ask(P.DECOMPOSE, f"Topic: {s['topic']}\nContext: {_ctx(s)}", Facets, stage="decompose", effort="low")}


def detect_field(s: State) -> dict:
    """Combine LLM classification with the empirical OpenAlex field distribution."""
    allowed = all_openalex_fields()
    facet_names = ", ".join(f.name for f in s["facets"].facets)
    guess = ask(
        P.FIELD.format(allowed="\n".join(allowed)),
        f"Topic: {s['topic']}\nUser-stated field: {s.get('field_hint') or 'none'}\nFacets: {facet_names}",
        FieldGuess, stage="detect_field", effort="low",
    )
    llm_fields = [f for f in [guess.primary, *guess.secondary] if f in allowed]
    notes: list[str] = []
    try:
        dist = openalex.field_distribution(s["topic"])
    except Exception as e:  # network trouble shouldn't kill the run
        dist, notes = {}, [f"Field distribution unavailable: {e}"]
    total = sum(dist.values()) or 1
    shares = sorted(((f, c / total) for f, c in dist.items()), key=lambda x: -x[1])

    fields = llm_fields[:1]
    if shares:
        top, top_share = shares[0]
        if s.get("field_hint"):
            if top not in llm_fields and top_share > 0.5:
                notes.append(f"User-stated field kept, but {top_share:.0%} of matching literature is indexed under '{top}'.")
        else:
            if top not in fields and top_share >= 0.25:
                fields.append(top)
            if top_share < 0.4:  # no dominant field -> interdisciplinary
                fields += llm_fields[1:]
                fields += [f for f, sh in shares[1:3] if sh >= 0.2]
    elif not fields:
        fields = llm_fields
    fields = list(dict.fromkeys(fields))[:3]
    names = profiles_for_fields(fields)
    if len(names) > 1:
        notes.append(f"Interdisciplinary topic: merged profiles {names}.")
    return {
        "fields": fields,
        "profile": build_profile(names),
        "field_distribution": {f: round(sh, 3) for f, sh in shares[:6]},
        "notes": notes,
    }


# ------------------------------------------------------------------- S3-S4 --


def plan_queries(s: State) -> dict:
    prof, facets = s["profile"], s["facets"]
    retry = ""
    if s.get("retrieval_rounds", 0) > 0:
        retry = (f"Coverage was thin ({len(s.get('ledger', {}))} records). Previous queries:\n"
                 + "\n".join(f"- {q.source}: {q.text}" for q in s["queries"])
                 + "\nBroaden: fewer terms per query, more synonyms, adjacent fields.")
    mesh = f"Prefer these MeSH terms: {facets.mesh_terms}" if prof["query_style"] == "mesh" and facets.mesh_terms else ""
    plan = ask(
        P.PLAN.format(sources=", ".join(prof["sources"]), mesh_hint=mesh, retry_hint=retry),
        f"Topic: {s['topic']}\nFacets: {facets.model_dump_json()}",
        QueryPlan, stage="plan_queries", effort="low",
    )
    per_source: dict[str, int] = {}
    queries = []
    for q in plan.queries:
        if q.source in prof["sources"] and q.source in REGISTRY and per_source.get(q.source, 0) < MAX_QUERIES_PER_SOURCE:
            per_source[q.source] = per_source.get(q.source, 0) + 1
            queries.append(q)
    return {"queries": queries}


def retrieve(s: State) -> dict:
    prof = s["profile"]
    this_year = date.today().year
    window = prof["recency_years"]
    rnd = s.get("retrieval_rounds", 0)
    ledger = dict(s.get("ledger", {}))
    errors: list[str] = []

    def run(i_q):
        i, q = i_q
        from_year = None if q.purpose == "landmark" else this_year - (2 * window if q.purpose == "review" else window)
        try:
            return f"q{rnd}.{i}:{q.source}", REGISTRY[q.source](q.text, from_year, prof["per_query"])
        except Exception as e:
            errors.append(f"{q.source} '{q.text[:60]}': {e}")
            return None, []

    with ThreadPoolExecutor(6) as ex:
        for tag, recs in ex.map(run, enumerate(s["queries"])):
            if tag:
                add_records(ledger, recs, tag)

    try:  # landmark quota: most-cited works on the topic, any age
        add_records(ledger, openalex.landmarks(s["topic"]), "landmark")
    except Exception as e:
        errors.append(f"landmarks: {e}")

    facets = s["facets"].facets
    ledger = score_and_trim(ledger, facets, {**prof, "max_records": 10_000})
    seeds = [k for k in ledger if k.startswith("W")][:5]  # one hop of citation chasing from top OpenAlex seeds
    for k in seeds:
        try:
            add_records(ledger, openalex.cited_by(k, 8) + openalex.references(k, 8), "chase")
        except Exception as e:
            errors.append(f"chase {k}: {e}")
    ledger = score_and_trim(ledger, facets, prof)  # returned sorted by score
    return {
        "ledger": ledger,
        # Built once, byte-identical for every later stage, so the prompt cache can reuse it.
        "shared_digest": P.SHARED_HEADER + digest(ledger, list(ledger)[:SHARED_RECORDS], SHARED_ABSTRACT_CHARS),
        "retrieval_rounds": rnd + 1,
        "coverage": coverage(ledger),
        "notes": [f"Retrieval error: {e}" for e in errors],
    }


def coverage_ok(s: State) -> str:
    enough = len(s["ledger"]) >= rubric_cfg()["qc_thresholds"]["min_ledger_size"]
    return "ok" if enough or s["retrieval_rounds"] >= MAX_RETRIEVAL_ROUNDS else "thin"


# ------------------------------------------------------------------- S5-S6 --


def landscape(s: State) -> dict:
    out = ask(
        P.LANDSCAPE.format(field=_field_label(s), evidence="; ".join(s["profile"]["evidence"])),
        f"Topic: {s['topic']}\nMap the LEDGER provided above.",
        Landscape, stage="landscape", effort="medium", shared_context=_shared(s),
    )
    for c in out.clusters:
        c.landmark_keys = [k for k in c.landmark_keys if k in s["ledger"]]
    return {"landscape": out, "groups": top_groups(s["ledger"])}


def gaps(s: State) -> dict:
    ledger = s["ledger"]
    matrix = facet_matrix(ledger, s["facets"].facets)
    matrix_txt = "\n".join(f"- {a} x {b}: {n}" for a, b, n in matrix)
    clusters = ", ".join(c.label for c in s["landscape"].clusters)
    out = ask(
        P.GAPS.format(field=_field_label(s)),
        f"Topic: {s['topic']}\nClusters: {clusters}\nReviews are flagged 'review' in the LEDGER above.\n\n"
        f"FACET MATRIX (records mentioning both facets, of {len(ledger)}):\n{matrix_txt}",
        Gaps, stage="gaps", effort="medium", shared_context=_shared(s),
    )
    return {"gaps": out.gaps}


# ------------------------------------------------------------------ S7-S10 --


def directions(s: State) -> dict:
    """First round: draft 6-8 directions. Revision round: regenerate only the flagged ones."""
    prof = s["profile"]
    rev, kept, kept_crit = s.get("revisions", 0), [], {}
    n, revision = "6-8", ""
    if s.get("critiques"):
        rev += 1
        by_idx = {c.index: c for c in s["critiques"]}
        to_fix = []
        for i, cand in enumerate(s["candidates"]):
            crit = by_idx.get(i)
            if crit is None or crit.verdict == "accept":
                kept.append(cand["direction"])
                if crit:
                    kept_crit[cand["direction"].research_question] = crit
            else:
                to_fix.append({**cand["direction"].model_dump(), "novelty_check": cand["novelty"].model_dump(),
                               "verdict": crit.verdict, "critic_notes": crit.notes})
        n = f"{len(to_fix)} revised or replacement"
        revision = (
            "REVISION ROUND. Return ONLY new versions of the directions below: revise each 'revise' one following "
            "the critic's notes exactly, and replace each 'reject' one with a different direction. The accepted "
            f"directions are kept as they are, so do not repeat them: {[d.title for d in kept]}.\n"
            f"TO FIX:\n{json.dumps(to_fix, default=str)}"
        )
    gaps_txt = "\n".join(f"[{i}] ({g.kind}; unsolved: {g.why_unsolved}) {g.statement} cites={g.supporting}"
                         for i, g in enumerate(s["gaps"]))
    out = ask(
        P.DIRECTIONS.format(field=_field_label(s), novelty="; ".join(prof["novelty"]),
                            evidence="; ".join(prof["evidence"]), context=_ctx(s), n=n, revision=revision),
        f"Topic: {s['topic']}\n\nGAPS:\n{gaps_txt}",
        Directions, stage="directions", effort="high", max_tokens=32000, shared_context=_shared(s),
    )
    return {"raw_directions": kept + out.directions, "revisions": rev, "kept_critiques": kept_crit}


def novelty_check(s: State) -> dict:
    """Search each research question directly (free APIs), then judge all new directions in ONE call.

    Done -> dropped. Verdicts are cached by research question, so kept directions aren't re-judged.
    """
    ledger = dict(s["ledger"])
    cache: dict = dict(s.get("novelty_cache", {}))
    use_s2 = "semanticscholar" in s["profile"]["sources"]
    todo = [d for d in s["raw_directions"] if d.research_question not in cache]

    def search(d):
        hits = []
        try:
            hits += openalex.search(d.research_question, None, 8)
            if use_s2:
                hits += semanticscholar.search(d.research_question, None, 4)
        except Exception as e:
            log.warning("novelty search failed: %s", e)
        tmp: dict = {}
        add_records(tmp, hits, "novelty")
        return tmp

    with ThreadPoolExecutor(4) as ex:
        hit_sets = list(ex.map(search, todo))

    if todo:
        blocks = [
            f"### DIRECTION {i}: {d.title}\nRQ: {d.research_question}\nMethod: {d.method_sketch}\n"
            f"Hits:\n{digest(tmp, None, NOVELTY_ABSTRACT_CHARS) if tmp else '(no hits)'}"
            for i, (d, tmp) in enumerate(zip(todo, hit_sets))
        ]
        batch = ask(P.NOVELTY, "\n\n".join(blocks), NoveltyBatch, stage="novelty_check", effort="medium", bulk=True)
        by_idx = {v.index: v for v in batch.verdicts}
        for i, (d, tmp) in enumerate(zip(todo, hit_sets)):
            item = by_idx.get(i)
            if item is None:
                continue  # judge skipped it; handled below as "not judged"
            overlap = [tmp[k] for k in item.overlap_keys if k in tmp]
            keys: list[str] = []
            if overlap:  # bring overlapping work into the ledger so it can be cited; remap merged keys
                mapping = add_records(ledger, overlap, "novelty")
                keys = list(dict.fromkeys(mapping[r.key] for r in overlap))
            cache[d.research_question] = NoveltyVerdict(status=item.status, overlap_keys=keys, difference=item.difference)

    candidates, dropped, notes = [], [], []
    for d in s["raw_directions"]:
        v = cache.get(d.research_question)
        if v is None:
            notes.append(f"Novelty not judged for '{d.title}'; treated as open.")
            v = NoveltyVerdict(status="open", overlap_keys=[], difference="")
        (dropped if v.status == "done" else candidates).append({"direction": d, "novelty": v})
    notes += [f"Dropped as already done: '{c['direction'].title}' (overlap: {c['novelty'].overlap_keys})" for c in dropped]
    return {"candidates": candidates, "ledger": ledger, "novelty_cache": cache, "notes": notes}


def critic_view(i: int, d, nv) -> dict:
    """Only the fields the critic scores, in compact form (no schema noise, claims as 'text [keys]')."""
    return {
        "index": i,
        "title": d.title,
        "rq": d.research_question,
        "novelty_argument": [f"{c.text} [{', '.join(c.cites)}]" for c in d.novelty_argument],
        "closest_prior_work": d.closest_prior_work,
        "method": d.method_sketch,
        "resources": d.resources_needed,
        "months": d.est_months,
        "risk": f"{d.risk}: {d.risk_notes}",
        "novelty_check": {"status": nv.status, "overlap": nv.overlap_keys, "difference": nv.difference},
    }


def critique(s: State) -> dict:
    """Independent critic. In a revision round only new or changed directions are sent; accepted ones keep their scores."""
    rub = rubric_cfg()
    cands = s["candidates"]
    kept = s.get("kept_critiques", {}) if s.get("revisions") else {}
    todo = [i for i, c in enumerate(cands) if c["direction"].research_question not in kept]
    new: dict = {}
    if todo:
        criteria = "; ".join(f"{k}: {v}" for k, v in rub["direction_criteria"].items())
        items = [critic_view(i, cands[i]["direction"], cands[i]["novelty"]) for i in todo]
        cited = list(dict.fromkeys(
            k for i in todo for k in cands[i]["direction"].closest_prior_work + cands[i]["novelty"].overlap_keys
            if k in s["ledger"]))
        out = ask(
            P.CRITIC.format(field=_field_label(s), criteria=criteria, context=_ctx(s),
                            banned=", ".join(f'"{b}"' for b in rub["banned_phrases"][:4])),
            f"DIRECTIONS:\n{json.dumps(items, separators=(',', ':'))}\n\n"
            f"CITED RECORDS:\n{digest(s['ledger'], cited, CRITIC_ABSTRACT_CHARS)}",
            Critiques, stage="critique", effort="high",
        )
        horizon = s["context"].horizon_months
        for c in out.critiques:  # deterministic guards the critic can't talk its way past
            if c.index not in todo:
                continue
            cand = cands[c.index]
            if horizon and cand["direction"].est_months > horizon and c.verdict == "accept":
                c.verdict, c.notes = "revise", c.notes + f" [auto] est_months {cand['direction'].est_months} > horizon {horizon}."
            if cand["novelty"].status == "partial" and not cand["novelty"].difference.strip() and c.verdict == "accept":
                c.verdict, c.notes = "revise", c.notes + " [auto] partial overlap with no stated difference."
            new[c.index] = c
    critiques = []
    for i, c in enumerate(cands):
        prev = kept.get(c["direction"].research_question)
        crit = prev.model_copy(update={"index": i}) if prev else new.get(i)
        if crit is not None:
            critiques.append(crit)
    return {"critiques": critiques}


def is_usable(c) -> bool:
    """Accepted, or a 'revise' that is already solid on the two criteria that matter most."""
    return c.verdict == "accept" or (c.verdict == "revise" and min(c.novelty, c.feasibility) >= 3)


def needs_revision(s: State) -> str:
    """Run the (expensive) revision round only when too few directions are usable."""
    if s.get("revisions", 0) >= 1:
        return "done"
    usable = [c for c in s["critiques"] if is_usable(c)]
    return "revise" if len(usable) < MIN_USABLE else "done"


def rank(s: State) -> dict:
    w = stage_weights(s["context"].stage)
    by_idx = {c.index: c for c in s["critiques"]}
    ranked = []
    for i, cand in enumerate(s["candidates"]):
        crit = by_idx.get(i)
        if crit is None or crit.verdict == "reject":
            continue
        base = sum(w[k] * getattr(crit, k) for k in ("novelty", "feasibility", "significance", "specificity"))
        score = base - w["risk_penalty"] * RISK_LEVEL[cand["direction"].risk] - (0.3 if crit.verdict == "revise" else 0)
        ranked.append({**cand, "critique": crit, "score": round(score, 3)})
    ranked.sort(key=lambda x: -x["score"])
    if not ranked:
        return {"ranked": [], "recommendation": None, "notes": ["No direction survived critique."]}
    top = ranked[:3]
    payload = [{"rank": i, "score": r["score"], **r["direction"].model_dump(), "critique": r["critique"].model_dump()}
               for i, r in enumerate(top)]
    rec = ask(P.RECOMMEND, f"Researcher context: {_ctx(s)}\n\nRANKED:\n{json.dumps(payload)}",
              Recommendation, stage="rank", effort="medium", shared_context=_shared(s))
    rec.chosen = [i for i in rec.chosen if 0 <= i < len(top)][:2] or [0]
    return {"ranked": ranked, "recommendation": rec}


def write(s: State) -> dict:
    rub = rubric_cfg()
    gaps_txt = "\n".join(f"- {g.statement}" for g in s["gaps"])
    draft = ask(
        P.WRITER.format(field=_field_label(s), banned=", ".join(f'"{b}"' for b in rub["banned_phrases"])),
        f"Topic: {s['topic']}\nContext: {_ctx(s)}\nGaps:\n{gaps_txt}\n"
        f"Top directions: {[r['direction'].title for r in s.get('ranked', [])[:2]]}",
        ReportDraft, stage="write", effort="medium", shared_context=_shared(s),
    )
    return {"draft": draft}
