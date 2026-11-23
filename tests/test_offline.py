"""Offline tests: no network, no API calls."""

from __future__ import annotations

from collections import Counter

import pytest

from research_assistant import verify
from research_assistant.config import build_profile, fields_cfg, profiles_for_fields, stage_weights
from research_assistant.ledger import add_records, facet_matrix, score_and_trim
from research_assistant.render import render
from research_assistant.schemas import (
    Claim, Cluster, Direction, DirectionCritique, Facet, Gap, Landscape, NoveltyVerdict, Record, Recommendation,
    ReportDraft, ResearcherContext,
)

OPENALEX_26 = [
    "Agricultural and Biological Sciences", "Arts and Humanities", "Biochemistry, Genetics and Molecular Biology",
    "Business, Management and Accounting", "Chemical Engineering", "Chemistry", "Computer Science", "Decision Sciences",
    "Dentistry", "Earth and Planetary Sciences", "Economics, Econometrics and Finance", "Energy", "Engineering",
    "Environmental Science", "Health Professions", "Immunology and Microbiology", "Materials Science", "Mathematics",
    "Medicine", "Neuroscience", "Nursing", "Pharmacology, Toxicology and Pharmaceutics", "Physics and Astronomy",
    "Psychology", "Social Sciences", "Veterinary",
]


def rec(key, title, doi=None, year=2023, cites=10, abstract="fish microplastic freshwater", **kw):
    return Record(key=key, title=title, doi=doi, year=year, cites=cites, abstract=abstract, **kw)


# ------------------------------------------------------------------ config --

def test_every_openalex_field_maps_to_exactly_one_profile():
    counts = Counter(f for p in fields_cfg()["profiles"].values() for f in p["openalex_fields"])
    assert set(counts) == set(OPENALEX_26)
    assert all(n == 1 for n in counts.values()), {f: n for f, n in counts.items() if n > 1}


def test_merge_profiles_for_interdisciplinary_topic():
    names = profiles_for_fields(["Computer Science", "Social Sciences"])
    p = build_profile(names)
    assert names == ["computer_science", "social_sciences"]
    assert p["recency_years"] == 7  # max of 3 and 7
    assert "semanticscholar" in p["sources"] and "crossref" in p["sources"]
    assert p["preprints"] == "accept"  # first profile wins


def test_mesh_wins_and_humanities_flags():
    p = build_profile(["humanities", "biomedicine"])
    assert p["query_style"] == "mesh" and p["low_digital_coverage"] and p["landmark_any_age"]


def test_unknown_profile_falls_back_to_defaults():
    assert build_profile(["nope"])["names"] == []


@pytest.mark.parametrize("stage,expect", [("PhD", 0.35), ("MS thesis", 0.2), ("MBA capstone", 0.15), (None, 0.3)])
def test_stage_weights(stage, expect):
    assert stage_weights(stage)["novelty"] == expect


# ------------------------------------------------------------------ ledger --

def test_dedupe_by_doi_and_title_with_key_mapping():
    ledger: dict = {}
    add_records(ledger, [rec("W1", "Microplastics in Trout", doi="10.1/x", cites=5, source="openalex")], "q1")
    m = add_records(ledger, [
        rec("PMID9", "Microplastics in trout.", doi="10.1/x", cites=0, source="pubmed"),   # same DOI
        rec("CR_abc", "MICROPLASTICS IN TROUT", cites=50, source="crossref"),               # same title
        rec("W2", "Something else"),
    ], "q2")
    assert set(ledger) == {"W1", "W2"}
    assert m == {"PMID9": "W1", "CR_abc": "W1", "W2": "W2"}
    assert ledger["W1"].cites == 50 and set(ledger["W1"].source.split("+")) == {"openalex", "pubmed", "crossref"}
    assert ledger["W1"].found_by == ["q1", "q2"]


def test_score_trim_keeps_old_landmark():
    facets = [Facet(name="microplastic", role="phenomenon", terms=["fish"])]
    ledger: dict = {}
    recent = [rec(f"W{i}", f"Recent paper {i}", year=2025, cites=1) for i in range(20)]
    classic = rec("W999", "Classic old study", year=1995, cites=5000)
    add_records(ledger, recent + [classic], "q")
    kept = score_and_trim(ledger, facets, {**build_profile(["computer_science"]), "max_records": 10})
    assert "W999" in kept and len(kept) == 10


def test_facet_matrix_counts():
    facets = [Facet(name="fish", role="population_or_object", terms=[]),
              Facet(name="tropical", role="context", terms=["amazon"])]
    ledger: dict = {}
    add_records(ledger, [rec("W1", "a", abstract="fish in the amazon"), rec("W2", "b", abstract="fish only")], "q")
    assert facet_matrix(ledger, facets) == [("fish", "tropical", 1)]


# ------------------------------------------------------- verify + render --

def all_supported(system, user, schema, **kw):
    """Fake batched entailment: every claim id in the prompt is 'supported'."""
    import re
    ids = [int(m) for m in re.findall(r"^(\d+)\. ", user.split("EVIDENCE:")[0], re.M)]
    return schema(verdicts=[{"id": i, "verdict": "supported", "reason": ""} for i in ids])


def make_state():
    ledger = {"W1": rec("W1", "Real paper", doi="10.1/real", authors=["A. Author"]),
              "W2": rec("W2", "Paper without abstract", abstract=None)}
    direction = Direction(
        title="Tropical trophic transfer", research_question="Does X transfer?", gaps_addressed=[0],
        novelty_argument=[Claim(text="No tropical studies exist.", cites=["W1", "FAKE123"])],
        closest_prior_work=["W1", "GHOST"], method_sketch="Sample fish.", resources_needed=["boat"],
        est_months=18, risk="medium", risk_notes="Seasonality.",
    )
    return {
        "topic": "Microplastics in freshwater fish",
        "context": ResearcherContext(stage="MS", horizon_months=12, resources=["field access"], methods=[], constraints=[]),
        "profile": build_profile(["life_environmental"]),
        "fields": ["Environmental Science"],
        "ledger": ledger,
        "coverage": {"n_records": 2},
        "landscape": Landscape(clusters=[Cluster(label="Exposure", summary=[
            Claim(text="Supported claim.", cites=["W1"]),
            Claim(text="Hallucinated claim.", cites=["NOPE"]),
            Claim(text="Abstract-less claim.", cites=["W2"]),
        ], methods_or_frameworks=["FTIR"], landmark_keys=["W1"])]),
        "gaps": [Gap(statement="No tropical data", kind="structural", why_unsolved="access", explanation="Remote.", supporting=["W1", "BAD"])],
        "ranked": [{"direction": direction, "novelty": NoveltyVerdict(status="open", overlap_keys=[], difference=""),
                    "critique": DirectionCritique(index=0, verdict="accept", novelty=4, feasibility=3, significance=4,
                                                  specificity=4, notes="ok"), "score": 3.1}],
        "recommendation": Recommendation(chosen=[0], rationale=[Claim(text="Best fit.", cites=["W1"])],
                                         what_would_change_it="Loss of access."),
        "draft": ReportDraft(summary="Summary. This topic has growing interest.",
                             significance_scholarly=[Claim(text="Sig.", cites=["W1"])],
                             significance_practical=[], significance_societal=[]),
        "critiques": [], "notes": [],
    }


def test_verify_strips_fake_keys_and_flags(monkeypatch):
    monkeypatch.setattr(verify, "ask", all_supported)
    monkeypatch.setattr(verify.crossref, "resolve_title", lambda doi: "Real paper")
    s = make_state()
    out = verify.verify_node(s)
    qc, status = out["qc"], out["claim_status"]
    assert set(qc["stripped_citations"]) == {"FAKE123", "NOPE", "BAD", "GHOST"}
    assert status["cluster.0.1"] == "uncited"
    assert status["cluster.0.2"] == "unverifiable"
    assert status["cluster.0.0"] == "supported"
    assert qc["doi_match_rate"] == 1.0
    assert "growing interest" in qc["banned_phrases_found"]
    assert s["ranked"][0]["direction"].closest_prior_work == ["W1"]


def test_doi_mismatch_detected(monkeypatch):
    monkeypatch.setattr(verify, "ask", all_supported)
    monkeypatch.setattr(verify.crossref, "resolve_title", lambda doi: "A totally different article")
    qc = verify.verify_node(make_state())["qc"]
    assert qc["doi_failures"] == ["W1"] and qc["low_confidence"]


def test_render_drops_uncited_and_builds_bibliography_from_ledger(monkeypatch):
    monkeypatch.setattr(verify, "ask", all_supported)
    monkeypatch.setattr(verify.crossref, "resolve_title", lambda doi: "Real paper")
    s = make_state()
    s.update(verify.verify_node(s))
    md = render(s)
    assert "Hallucinated claim" not in md
    assert "Supported claim. [1]" in md
    assert "not verified: no abstract" in md
    assert "https://doi.org/10.1/real" in md
    assert "FAKE123" not in md and "GHOST" not in md
    assert "exceeds 12-month horizon" in md
    assert md.count("\n1. A. Author") == 1  # W1 appears once in bibliography


# ---------------------------------------------------------- token savings --

from research_assistant import llm, nodes  # noqa: E402
from research_assistant.schemas import Directions, NoveltyBatch  # noqa: E402


def test_shared_context_is_cached_and_effort_changed_per_message():
    kw = llm.build_request("stage rules", "do X", effort="high", bulk=False, max_tokens=100, shared_context="LEDGER...")
    assert kw["system"][0] == {"type": "text", "text": "LEDGER...", "cache_control": {"type": "ephemeral"}}
    assert kw["system"][1]["text"] == "stage rules"
    assert kw["output_config"]["effort"] == llm.CACHE_EFFORT  # top level pinned so the cache isn't invalidated
    assert kw["messages"][-1] == {"role": "system", "content": [], "output_config": {"effort": "high"}}
    assert llm.EFFORT_BETA in kw["betas"]


def test_no_effort_message_when_effort_matches_cache_effort():
    kw = llm.build_request("s", "u", effort=llm.CACHE_EFFORT, bulk=False, max_tokens=100, shared_context="L")
    assert len(kw["messages"]) == 1 and llm.EFFORT_BETA not in kw.get("betas", [])


def test_plain_request_keeps_stage_effort():
    kw = llm.build_request("s", "u", effort="low", bulk=True, max_tokens=100, shared_context=None)
    assert kw["system"] == "s" and kw["output_config"]["effort"] == "low" and kw["model"] == llm.BULK_MODEL


def test_usage_accounting():
    llm.reset_usage()
    u = type("U", (), {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 10_000,
                       "cache_creation_input_tokens": 0})()
    llm.record_usage("gaps", "claude-opus-5-5", u)
    llm.record_usage("gaps", "claude-opus-5-5", u)
    rep = llm.usage_report()
    assert rep["by_stage"]["gaps"]["calls"] == 2 and rep["total"]["cache_read_tokens"] == 20_000
    # 2 * (1000*4 + 500*20 + 10000*4*0.05) / 1e6
    assert rep["total"]["est_cost_usd"] == pytest.approx(0.032)


def test_entailment_is_one_call_per_batch(monkeypatch):
    calls = []

    def fake(system, user, schema, **kw):
        calls.append(user)
        return all_supported(system, user, schema)

    monkeypatch.setattr(verify, "ask", fake)
    ledger = {"W1": rec("W1", "t1"), "W2": rec("W2", "t2"), "W3": rec("W3", "t3", abstract=None)}
    claims = [Claim(text=f"c{i}", cites=["W1", "W2"]) for i in range(45)]
    claims += [Claim(text="no cite", cites=[]), Claim(text="no abstract", cites=["W3"])]
    v = verify.entail_all(claims, ledger)
    assert len(calls) == 2  # 45 claims -> batches of 40 + 5
    assert all(calls[0].count(f"[{k}]") == 1 for k in ("W1", "W2"))  # each abstract sent once per batch
    assert v[:45] == ["supported"] * 45 and v[45] == "uncited" and v[46] == "unverifiable"


def _direction(title, rq):
    return Direction(title=title, research_question=rq, gaps_addressed=[0], novelty_argument=[],
                     closest_prior_work=[], method_sketch="m", resources_needed=[], est_months=6,
                     risk="low", risk_notes="")


def test_revision_regenerates_only_flagged_directions(monkeypatch):
    seen = {}

    def fake(system, user, schema, **kw):
        seen["system"] = system
        return Directions(directions=[_direction("B2", "rq-b2")])

    monkeypatch.setattr(nodes, "ask", fake)
    a, b = _direction("A", "rq-a"), _direction("B", "rq-b")
    crit = lambda i, v: DirectionCritique(index=i, verdict=v, novelty=3, feasibility=3, significance=3,  # noqa: E731
                                          specificity=3, notes="fix it")
    s = make_state()
    s.update({"shared_digest": "L", "candidates": [
        {"direction": a, "novelty": NoveltyVerdict(status="open", overlap_keys=[], difference="")},
        {"direction": b, "novelty": NoveltyVerdict(status="open", overlap_keys=[], difference="")}],
        "critiques": [crit(0, "accept"), crit(1, "revise")], "revisions": 0})
    out = nodes.directions(s)
    assert [d.title for d in out["raw_directions"]] == ["A", "B2"]
    assert '"title": "A"' not in seen["system"].split("TO FIX:")[1]  # accepted one not re-sent
    assert set(out["kept_critiques"]) == {"rq-a"}


def test_novelty_is_one_batched_call(monkeypatch):
    calls = []
    monkeypatch.setattr(nodes.openalex, "search", lambda q, y, n: [rec(f"W{abs(hash(q)) % 1000}", f"hit for {q}")])

    def fake(system, user, schema, **kw):
        calls.append(user)
        return NoveltyBatch(verdicts=[
            {"index": 0, "status": "done", "overlap_keys": [], "difference": ""},
            {"index": 1, "status": "open", "overlap_keys": [], "difference": ""}])

    monkeypatch.setattr(nodes, "ask", fake)
    s = make_state()
    s["raw_directions"] = [_direction("A", "rq-a"), _direction("B", "rq-b")]
    out = nodes.novelty_check(s)
    assert len(calls) == 1
    assert [c["direction"].title for c in out["candidates"]] == ["B"]
    # second pass: both cached, no new call
    s.update(out)
    nodes.novelty_check(s)
    assert len(calls) == 1
