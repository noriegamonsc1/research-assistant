"""Load field profiles and rubric; merge profiles for interdisciplinary topics."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("RESEARCH_ASSISTANT_CONFIG_DIR", ROOT / "config"))

LIST_KEYS = ("evidence", "novelty", "venues", "openalex_fields")


@lru_cache
def fields_cfg() -> dict:
    with open(CONFIG_DIR / "fields.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def rubric_cfg() -> dict:
    with open(CONFIG_DIR / "rubric.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_openalex_fields(cfg: dict | None = None) -> list[str]:
    cfg = cfg or fields_cfg()
    return [f for p in cfg["profiles"].values() for f in p.get("openalex_fields", [])]


def profiles_for_fields(openalex_fields: list[str], cfg: dict | None = None) -> list[str]:
    """Map OpenAlex field names to profile names, preserving order, no duplicates."""
    cfg = cfg or fields_cfg()
    out: list[str] = []
    for field in openalex_fields:
        for name, p in cfg["profiles"].items():
            if field in p.get("openalex_fields", []) and name not in out:
                out.append(name)
    return out


def build_profile(names: list[str], cfg: dict | None = None) -> dict:
    """Merge one or more profiles over the defaults.

    Merge rules: sources/list keys = ordered union; recency = max window;
    boolean flags = any; preprints/query_style = first profile's (mesh wins if any).
    """
    cfg = cfg or fields_cfg()
    d = dict(cfg["defaults"])
    ps = [cfg["profiles"][n] for n in names if n in cfg["profiles"]]
    if not ps:
        return {**d, "names": []}

    def union(key: str, fallback: list) -> list:
        return list(dict.fromkeys(x for p in ps for x in p.get(key, fallback)))

    merged = {**d, "names": [n for n in names if n in cfg["profiles"]]}
    merged["sources"] = union("sources", d["sources"])
    for k in LIST_KEYS:
        merged[k] = union(k, d.get(k, []))
    merged["recency_years"] = max(p.get("recency_years", d["recency_years"]) for p in ps)
    merged["low_digital_coverage"] = any(p.get("low_digital_coverage", False) for p in ps)
    merged["landmark_any_age"] = any(p.get("landmark_any_age", False) for p in ps)
    merged["preprints"] = ps[0].get("preprints", d["preprints"])
    merged["query_style"] = "mesh" if any(p.get("query_style") == "mesh" for p in ps) else "keyword"
    return merged


def stage_weights(stage: str | None) -> dict:
    table = rubric_cfg()["stage_weights"]
    s = (stage or "").lower()
    # longest key first so "phd" doesn't lose to a shorter accidental match
    for key in sorted((k for k in table if k != "default"), key=len, reverse=True):
        if key in s:
            return table[key]
    return table["default"]
