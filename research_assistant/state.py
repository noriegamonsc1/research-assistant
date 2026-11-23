"""Graph state. `notes` accumulates across nodes via an add-reducer."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .schemas import Direction, DirectionCritique, Facets, Gap, Landscape, Query, Record, Recommendation, ReportDraft, ResearcherContext


class State(TypedDict, total=False):
    # inputs
    raw_input: str
    user_field: str | None
    # S0-S2
    topic: str
    field_hint: str | None
    context: ResearcherContext
    facets: Facets
    fields: list[str]
    profile: dict
    field_distribution: dict[str, float]
    # S3-S4
    queries: list[Query]
    ledger: dict[str, Record]
    shared_digest: str  # cached system block reused by landscape/gaps/directions/rank/write
    retrieval_rounds: int
    coverage: dict
    # S5-S6
    landscape: Landscape
    groups: dict
    gaps: list[Gap]
    # S7-S10
    raw_directions: list[Direction]
    novelty_cache: dict
    candidates: list[dict[str, Any]]  # {"direction": Direction, "novelty": NoveltyVerdict}
    critiques: list[DirectionCritique]
    kept_critiques: dict[str, DirectionCritique]  # research question -> accepted critique (revision round)
    revisions: int
    ranked: list[dict[str, Any]]
    recommendation: Recommendation | None
    # S11
    draft: ReportDraft
    claim_status: dict[str, str]
    qc: dict
    report_md: str
    # cross-cutting
    notes: Annotated[list[str], operator.add]
