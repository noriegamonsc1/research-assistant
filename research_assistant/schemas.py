"""Typed contracts between pipeline stages.

Two kinds of models:
- `Record` is built ONLY from bibliographic API metadata (never by the LLM).
- Everything else is LLM structured output. Those models avoid defaults and dicts
  so they map cleanly onto strict JSON schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- API-side ---


class Record(BaseModel):
    key: str  # ledger key, [A-Za-z0-9_]+ (e.g. W2741809807, PMID12345, S2_ab12.., CR_9f3e..)
    title: str
    year: int | None = None
    venue: str | None = None
    doi: str | None = None  # bare DOI, lowercase, no https://doi.org/ prefix
    authors: list[str] = []
    institutions: list[str] = []
    abstract: str | None = None
    cites: int = 0
    is_review: bool = False
    is_preprint: bool = False
    language: str | None = None
    source: str = ""
    found_by: list[str] = []  # query ids / "chase" / "novelty"
    score: float = 0.0


# --------------------------------------------------------------- LLM-side ----


class ResearcherContext(BaseModel):
    stage: str | None = Field(description="Career stage, e.g. 'MS thesis', 'PhD', 'MBA capstone'; null if not stated")
    horizon_months: int | None = Field(description="Time available in months; null if not stated")
    resources: list[str] = Field(description="Equipment, datasets, archives, participant/field access, funding")
    methods: list[str] = Field(description="Methods expertise stated or clearly implied")
    constraints: list[str] = Field(description="Other stated constraints (language, region, ethics, ...)")


class Intake(BaseModel):
    topic_core: str = Field(description="The research topic alone, 1-2 sentences, context removed")
    field_hint: str | None = Field(description="Field/discipline if the user stated one, else null")
    context: ResearcherContext


class Facet(BaseModel):
    name: str
    role: Literal["population_or_object", "phenomenon", "mechanism", "method", "context", "outcome"]
    terms: list[str] = Field(description="Synonyms and field vocabulary, incl. non-English terms where relevant")


class Facets(BaseModel):
    facets: list[Facet]
    mesh_terms: list[str] = Field(description="MeSH headings if biomedical/health-relevant, else empty")
    languages: list[str] = Field(description="ISO 639-1 codes whose literature matters for this topic, e.g. ['en','es','pt']")


class FieldGuess(BaseModel):
    primary: str = Field(description="Exactly one name from the allowed OpenAlex field list")
    secondary: list[str] = Field(description="0-2 further names from the allowed list if interdisciplinary")


class Query(BaseModel):
    source: str = Field(description="One of the allowed source adapters")
    text: str = Field(description="Query in that source's syntax")
    purpose: Literal["core", "gap_probe", "review", "landmark"]


class QueryPlan(BaseModel):
    queries: list[Query]


class Claim(BaseModel):
    text: str = Field(description="One factual sentence. No citation markers in the text.")
    cites: list[str] = Field(description="Ledger keys supporting this sentence. Only keys from the provided ledger.")


class Cluster(BaseModel):
    label: str
    summary: list[Claim]
    methods_or_frameworks: list[str]
    landmark_keys: list[str]


class Landscape(BaseModel):
    clusters: list[Cluster]


class Gap(BaseModel):
    statement: str
    kind: Literal["stated", "structural", "contested"]
    why_unsolved: Literal["data", "method", "theory", "access", "cost"]
    explanation: str
    supporting: list[str] = Field(description="Ledger keys")


class Gaps(BaseModel):
    gaps: list[Gap]


class Direction(BaseModel):
    title: str
    research_question: str
    gaps_addressed: list[int] = Field(description="Indices into the gap list")
    novelty_argument: list[Claim]
    closest_prior_work: list[str] = Field(description="Ledger keys of the most similar existing work")
    method_sketch: str
    resources_needed: list[str]
    est_months: int
    risk: Literal["low", "medium", "high"]
    risk_notes: str


class Directions(BaseModel):
    directions: list[Direction]


class NoveltyVerdict(BaseModel):
    status: Literal["done", "partial", "open"]
    overlap_keys: list[str]
    difference: str = Field(description="How the direction differs from the overlapping work ('' if done)")


class NoveltyItem(NoveltyVerdict):
    index: int = Field(description="Index of the direction being judged")


class NoveltyBatch(BaseModel):
    verdicts: list[NoveltyItem]


class DirectionCritique(BaseModel):
    index: int
    verdict: Literal["accept", "revise", "reject"]
    novelty: int = Field(description="1-5")
    feasibility: int = Field(description="1-5")
    significance: int = Field(description="1-5")
    specificity: int = Field(description="1-5")
    notes: str = Field(description="Concrete revision instructions or rejection reason")


class Critiques(BaseModel):
    critiques: list[DirectionCritique]


class Recommendation(BaseModel):
    chosen: list[int] = Field(description="1-2 indices into the ranked list")
    rationale: list[Claim]
    what_would_change_it: str


class ReportDraft(BaseModel):
    summary: str = Field(description="3-4 sentence plain-language executive summary, no citations")
    significance_scholarly: list[Claim]
    significance_practical: list[Claim]
    significance_societal: list[Claim]


class Entailment(BaseModel):
    id: int = Field(description="Claim id as given")
    verdict: Literal["supported", "partial", "unsupported"]


class EntailBatch(BaseModel):
    verdicts: list[Entailment]
