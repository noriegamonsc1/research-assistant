# research-assistant: research direction-finding agent

Give it a 1–3 sentence topic, plus an optional field and researcher context. It returns a cited report with five sections: Significance, State of the art, Gaps, Candidate directions, and Recommendation. It works in any discipline.

Core rule: **the LLM reasons, the APIs remember.** Every reference comes from OpenAlex, Semantic Scholar, PubMed or Crossref metadata retrieved during the run. The model can only cite keys that exist in that Evidence Ledger, and the bibliography is rendered from API metadata.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env    # then set ANTHROPIC_API_KEY and CONTACT_EMAIL, and export them
```
Use the editable install (`-e`). Field profiles are read from `config/` next to the package.

## Working together
- Never commit `.env`. Each person uses their own `ANTHROPIC_API_KEY`, and their runs bill to their own account.
- Work on a branch (`git switch -c your-name/short-topic`) and open a pull request into `main`.
- Run `pytest` before pushing. It runs offline: no API key and no cost.
- Good first contributions that need no Python: a discipline profile in `config/fields.yaml`, or expert `must_mention` lists in `eval/topics.yaml`.
- Generated outputs (`runs/`, `eval/results/`) stay out of git. Share a report by attaching its folder to a PR or an issue.

## Run
```bash
python -m research_assistant.cli "Microplastic accumulation in freshwater fish; MS thesis, 1 year, field sampling access."
python -m research_assistant.cli "LLM-based tutoring for introductory statistics; 6-month project, classroom access." --field "Education"
```
Each run writes these files to `runs/<timestamp>-<slug>/`:
- `report.md`: the report
- `qc_report.json`: claim-support verdicts, stripped citations, DOI match rate, coverage, critic verdicts, and the `low_confidence` flag
- `ledger.json`: every retrieved source
- `trace.json`: the intermediate output of each stage, for debugging

A run makes roughly 15–30 model calls, most at high effort. Start with a single topic.

## Pipeline
```
intake → decompose → detect_field → plan_queries → retrieve ─(thin? ≤2 rounds)→ plan_queries
  → landscape → gaps → directions → novelty_check → critique ─(revise? once)→ directions
  → rank → write → verify → render
```
| Stage | File | Notes |
|---|---|---|
| Field detection | `nodes.detect_field` | LLM label + OpenAlex field distribution; < 40% dominance → interdisciplinary merge |
| Retrieval | `nodes.retrieve`, `ledger.py` | parallel queries, DOI/title dedupe, landmark quota, 1-hop citation chase, field-aware scoring |
| Gaps | `nodes.gaps` | stated (reviews) + structural (sparse facet pairs) + contested |
| Novelty check | `nodes.novelty_check` | targeted search per direction; `done` → dropped |
| Critique | `nodes.critique` | separate role and context, plus hard guards (months vs horizon, partial overlap with no stated difference) |
| Verification | `verify.py` | strips non-ledger keys, entailment per claim, Crossref DOI ↔ title check, banned-phrase lint |

## Adding a discipline
Add a profile to `config/fields.yaml` and give it `openalex_fields`. Each of OpenAlex's 26 fields must map to exactly one profile; `tests/test_offline.py` enforces this. Adding a new *source* takes one adapter module in `research_assistant/sources/` and one line in `sources/__init__.py`.

## Tests and eval
```bash
pytest                                   # offline: no network, no API calls
python eval/run_eval.py --only env-microplastics          # one topic, auto-metrics
python eval/run_eval.py --judge          # all topics + LLM-judge rubric + per-domain fairness check
```
Before trusting `landscape_recall`, fill in `must_mention` in `eval/topics.yaml` with a domain expert. Calibrate the LLM judge against expert scores before using it for anything other than regression testing.

## Token use
Every run records tokens and an estimated cost per stage in `qc_report.json` → `token_usage`. The CLI also prints the totals.
- **Shared, cached paper list.** `retrieve` builds one digest of the top 60 papers. Landscape, gaps, directions, rank and write all get it as the same cached system block, so after the first stage it is read from the prompt cache. Confirm `cache_read_tokens` is non-zero for those stages after a real run.
- **Effort.** `low` for intake, decompose and field detection. `medium` for query planning, landscape, gaps, rank, write and novelty. `high` only for directions and critique. Calls that share the cache keep one top-level effort and switch effort with a per-message setting, which leaves the cache intact.
- **Batched checks.** Novelty judging is one call for all directions. Claim verification checks up to 40 claims per call and sends each cited abstract once.
- **Targeted revision.** A revision round regenerates and re-critiques only the directions the critic flagged.

## Knobs
- `RESEARCH_ASSISTANT_MODEL` / `RESEARCH_ASSISTANT_BULK_MODEL`: the bulk model handles entailment and novelty judging. Try `claude-haiku-5-5` there only after an eval shows quality holds.
- `RESEARCH_ASSISTANT_FALLBACK=0` turns off the server-side refusal fallback.
- `config/rubric.yaml`: career-stage ranking weights, banned phrases, QC thresholds.

## Known MVP limits / v2 ideas
- Relevance is lexical (facet-term overlap). Swap in embeddings for better reranking.
- No full-text gap mining yet (Europe PMC, arXiv and CORE full text are the next step).
- arXiv, Europe PMC, CORE and RePEc adapters are not implemented.
- Non-English coverage depends on OpenAlex/Crossref metadata plus the multilingual queries.
