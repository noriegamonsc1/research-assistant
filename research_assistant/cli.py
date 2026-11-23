"""Command line: python -m research_assistant.cli "topic; context" [--field "..."] [--out runs/]"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


def to_jsonable(x):
    if isinstance(x, BaseModel):
        return x.model_dump()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def save_run(state: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(state.get("report_md", ""), encoding="utf-8")
    (out_dir / "qc_report.json").write_text(json.dumps(state.get("qc", {}), indent=2, default=str), encoding="utf-8")
    (out_dir / "ledger.json").write_text(json.dumps(to_jsonable(state.get("ledger", {})), indent=2, default=str), encoding="utf-8")
    trace_keys = ["raw_input", "topic", "context", "facets", "fields", "profile", "field_distribution", "queries",
                  "gaps", "raw_directions", "candidates", "critiques", "ranked", "recommendation", "claim_status"]
    trace = {k: to_jsonable(state.get(k)) for k in trace_keys}
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Research direction-finding agent")
    ap.add_argument("input", help="Topic (1-3 sentences) plus optional context, e.g. 'X; MS thesis, 1 year, field access.'")
    ap.add_argument("--field", default=None, help="Optional field/discipline (overrides detection)")
    ap.add_argument("--out", default="runs", help="Output root directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    from .graph import run  # import late so --help works without API deps configured

    slug = re.sub(r"[^a-z0-9]+", "-", args.input.lower())[:40].strip("-")
    out = Path(args.out) / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    print(f"Running pipeline → {out}")
    state = run(args.input, args.field)
    save_run(state, out)
    qc = state.get("qc", {})
    if qc.get("stopped_early"):
        print(f"\nStopped early: {qc.get('reason')}. See the report for what was found and what to try next.")
    else:
        print(f"\nDone. {qc.get('n_claims', 0)} claims, unsupported rate {qc.get('unsupported_rate')}, "
              f"DOI match {qc.get('doi_match_rate')}, low_confidence={qc.get('low_confidence')}")
    budget = qc.get("token_usage", {}).get("budget", {})
    if budget.get("exceeded_at_stage"):
        print(f"Cost cap ${budget['cap_usd']:.2f} reached at '{budget['exceeded_at_stage']}': "
              "later calls ran at lower effort and the revision round was skipped.")
    tot = qc.get("token_usage", {}).get("total", {})
    if tot:
        print(f"Tokens: {tot['calls']} calls, input {tot['input_tokens']:,} + cached {tot['cache_read_tokens']:,}, "
              f"output {tot['output_tokens']:,} | est. ${tot['est_cost_usd']:.2f} (see qc_report.json for per-stage)")
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
