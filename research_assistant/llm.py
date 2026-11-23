"""Thin wrapper over the Anthropic SDK's structured-output helper, with token accounting.

Every LLM call goes through `ask()`, which returns a validated Pydantic object and records
token usage per stage (see `usage_report()`; it ends up in qc_report.json).

Token-saving design:
- `shared_context` (the paper digest) goes in the FIRST system block with `cache_control`,
  so stages that reuse it read it from the prompt cache instead of paying full input price.
- All cached-context calls send the same top-level effort (CACHE_EFFORT). A stage that needs
  a different effort adds a per-message effort change after the user turn (beta
  `mid-conversation-output-config-2026-07-01`), which keeps the cached prefix intact.
Check `cache_read_input_tokens` in the usage report to confirm the cache is being hit.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from typing import TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

MODEL = os.environ.get("RESEARCH_ASSISTANT_MODEL", "claude-opus-5-5")
# Bulk checks (claim verification, novelty judging). Same model by default;
# try claude-haiku-5-5 only after an eval shows the verdicts hold.
BULK_MODEL = os.environ.get("RESEARCH_ASSISTANT_BULK_MODEL", MODEL)
USE_FALLBACK = os.environ.get("RESEARCH_ASSISTANT_FALLBACK", "1") == "1"
CACHE_EFFORT = "medium"  # top-level effort for every call that carries the cached digest

FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORT_BETA = "mid-conversation-output-config-2026-07-01"

# USD per million tokens: (input, output). Cache reads/writes priced from input below.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE_READ_X = {"claude-opus-5-5": 0.05, "claude-sonnet-5-5": 0.05}  # others: 0.1
CACHE_WRITE_X = 1.25  # 5-minute cache

_client: anthropic.Anthropic | None = None
_lock = threading.Lock()
_usage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


class LLMRefusal(RuntimeError):
    pass


# --------------------------------------------------------------- accounting --


def reset_usage() -> None:
    with _lock:
        _usage.clear()


def record_usage(stage: str, model: str, usage) -> None:
    get = lambda name: getattr(usage, name, None) or 0  # noqa: E731
    inp, out = get("input_tokens"), get("output_tokens")
    c_read, c_write = get("cache_read_input_tokens"), get("cache_creation_input_tokens")
    p_in, p_out = PRICES.get(model, PRICES["claude-opus-5-5"])
    cost = (inp * p_in + out * p_out + c_read * p_in * CACHE_READ_X.get(model, 0.1)
            + c_write * p_in * CACHE_WRITE_X) / 1e6
    with _lock:
        u = _usage[stage]
        u["calls"] += 1
        u["input_tokens"] += inp
        u["cache_read_tokens"] += c_read
        u["cache_write_tokens"] += c_write
        u["output_tokens"] += out
        u["est_cost_usd"] += cost


def usage_report() -> dict:
    with _lock:
        stages = {s: {k: round(v, 4) if k == "est_cost_usd" else int(v) for k, v in u.items()} for s, u in _usage.items()}
    keys = ["calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "est_cost_usd"]
    total = {k: sum(st.get(k, 0) for st in stages.values()) for k in keys}
    total["est_cost_usd"] = round(total["est_cost_usd"], 4)
    return {"by_stage": stages, "total": total,
            "note": "Cost is estimated from list prices; check the Console for actual billing."}


# ------------------------------------------------------------------ requests --


def build_request(system: str, user: str, *, effort: str, bulk: bool, max_tokens: int,
                  shared_context: str | None) -> dict:
    """Assemble kwargs for messages.parse (pure function; unit-tested)."""
    betas: list[str] = []
    messages: list[dict] = [{"role": "user", "content": user}]
    if shared_context:
        sys_blocks = [
            {"type": "text", "text": shared_context, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": system},
        ]
        top_effort = CACHE_EFFORT
        if effort != CACHE_EFFORT:  # change effort without invalidating the cached prefix
            messages.append({"role": "system", "content": [], "output_config": {"effort": effort}})
            betas.append(EFFORT_BETA)
    else:
        sys_blocks = system
        top_effort = effort
    kwargs: dict = {
        "model": BULK_MODEL if bulk else MODEL,
        "max_tokens": max_tokens,
        "system": sys_blocks,
        "messages": messages,
        "output_config": {"effort": top_effort},  # Opus 5.5 defaults to medium; always explicit
    }
    if USE_FALLBACK:
        betas.append(FALLBACK_BETA)
        kwargs["fallbacks"] = "default"  # server-side reroute if a request is declined
    if betas:
        kwargs["betas"] = betas
    return kwargs


def ask(
    system: str,
    user: str,
    schema: type[T],
    *,
    stage: str,
    effort: str = "medium",
    bulk: bool = False,
    max_tokens: int = 16000,
    shared_context: str | None = None,
) -> T:
    kwargs = build_request(system, user, effort=effort, bulk=bulk, max_tokens=max_tokens, shared_context=shared_context)
    resp = client().beta.messages.parse(output_format=schema, **kwargs)
    record_usage(stage, getattr(resp, "model", None) or kwargs["model"], resp.usage)
    if resp.stop_reason == "refusal":
        raise LLMRefusal(str(resp.stop_details))
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"{stage}/{schema.__name__}: output truncated at max_tokens={max_tokens}")
    if resp.parsed_output is None:
        raise RuntimeError(f"{stage}/{schema.__name__}: no parsed output (stop_reason={resp.stop_reason})")
    return resp.parsed_output
