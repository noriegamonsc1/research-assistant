"""Cross-field evaluation.

  python eval/run_eval.py                 # run all topics, auto-metrics only
  python eval/run_eval.py --judge         # + LLM-as-judge rubric scores
  python eval/run_eval.py --only hum-bolano --judge
  python eval/run_eval.py --rescore eval/results/2026...  # judge existing runs, no pipeline calls

Every pipeline run and judge call spends API credit; start with --only.
The LLM judge is for regression testing; calibrate it against expert scores before trusting it.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_assistant.config import rubric_cfg  # noqa: E402

CRITERIA = list(rubric_cfg()["report_criteria"])


class JudgeScores(BaseModel):
    citation_integrity: int = Field(description="1-5")
    significance_specificity: int = Field(description="1-5")
    landscape_coverage: int = Field(description="1-5")
    gap_validity: int = Field(description="1-5")
    direction_novelty: int = Field(description="1-5")
    feasibility_fit: int = Field(description="1-5")
    actionability: int = Field(description="1-5")
    calibration: int = Field(description="1-5")
    rationale: str


JUDGE = """You are an expert reviewer in {domain} grading a research-direction report for a researcher with this
request: "{input}". Score each criterion 1 (poor) to 5 (excellent) using the field's own standards:
{criteria}
Judge only what is in the report and its QC data. Be strict; 3 = acceptable for an early-stage advisor memo."""


def judge(report: str, qc: dict, topic: dict) -> JudgeScores:
    from research_assistant.llm import ask

    crit = "\n".join(f"- {k}: {v}" for k, v in rubric_cfg()["report_criteria"].items())
    return ask(JUDGE.format(domain=topic["domain"], input=topic["input"], criteria=crit),
               f"REPORT:\n{report}\n\nQC:\n{json.dumps(qc)[:4000]}", JudgeScores, effort="high")


def landscape_recall(report: str, must: list[str]) -> float | None:
    if not must:
        return None
    return sum(1 for m in must if m.lower() in report.lower()) / len(must)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="topic ids to run")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--rescore", help="existing results dir to (re)judge without rerunning the pipeline")
    args = ap.parse_args()

    topics = yaml.safe_load(open(ROOT / "eval" / "topics.yaml", encoding="utf-8"))["topics"]
    if args.only:
        topics = [t for t in topics if t["id"] in args.only]
    res_dir = Path(args.rescore) if args.rescore else ROOT / "eval" / "results" / f"{datetime.now():%Y%m%d-%H%M%S}"

    rows = []
    for t in topics:
        run_dir = res_dir / t["id"]
        if not args.rescore:
            from research_assistant.cli import save_run
            from research_assistant.graph import run

            print(f"[{t['id']}] running pipeline…")
            try:
                save_run(run(t["input"], t.get("field"), stream=False), run_dir)
            except Exception as e:
                print(f"  failed: {e}")
                rows.append({"id": t["id"], "domain": t["domain"], "error": str(e)})
                continue
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        qc = json.loads((run_dir / "qc_report.json").read_text(encoding="utf-8"))
        row = {
            "id": t["id"], "domain": t["domain"],
            "n_records": qc.get("coverage", {}).get("n_records"),
            "n_claims": qc.get("n_claims"),
            "unsupported_rate": qc.get("unsupported_rate"),
            "stripped_citations": len(qc.get("stripped_citations", [])),
            "doi_match_rate": qc.get("doi_match_rate"),
            "banned_phrases": len(qc.get("banned_phrases_found", [])),
            "low_confidence": qc.get("low_confidence"),
            "landscape_recall": landscape_recall(report, t.get("must_mention", [])),
            "llm_calls": qc.get("token_usage", {}).get("total", {}).get("calls"),
            "output_tokens": qc.get("token_usage", {}).get("total", {}).get("output_tokens"),
            "est_cost_usd": qc.get("token_usage", {}).get("total", {}).get("est_cost_usd"),
        }
        if args.judge:
            s = judge(report, qc, t)
            row.update(s.model_dump(exclude={"rationale"}))
            (run_dir / "judge.json").write_text(s.model_dump_json(indent=2), encoding="utf-8")
        rows.append(row)
        print(f"  {row}")

    res_dir.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("id", "domain"), k))
    with open(res_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    if args.judge:  # cross-field fairness: no domain > 1 point below the overall mean on any criterion
        ok_rows = [r for r in rows if "error" not in r]
        print("\nPer-domain means (fairness check):")
        for c in CRITERIA:
            overall = statistics.mean(r[c] for r in ok_rows)
            by_dom = {}
            for r in ok_rows:
                by_dom.setdefault(r["domain"], []).append(r[c])
            flags = [d for d, v in by_dom.items() if statistics.mean(v) < overall - 1]
            print(f"  {c:26s} overall {overall:.2f}  " + "  ".join(f"{d}={statistics.mean(v):.1f}" for d, v in by_dom.items())
                  + (f"  ⚠ below: {flags}" if flags else ""))
    print(f"\nSummary: {res_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
