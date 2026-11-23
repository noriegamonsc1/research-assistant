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

Backends (env RESEARCH_ASSISTANT_BACKEND):
- "api" (default): Anthropic Python SDK, billed to Console API credit. Needs ANTHROPIC_API_KEY.
- "claude-cli": runs `claude -p` (Claude Code headless mode) for each call, so usage counts
  against your Claude plan instead of API credit. Uses Claude Code's own login. The
  per-message effort change and server-side fallbacks are API-only; effort is passed with
  `--effort`, and the shared digest still goes first in the system prompt.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
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

BACKEND = os.environ.get("RESEARCH_ASSISTANT_BACKEND", "api")
# Claude Code model aliases ("opus", "sonnet", "haiku", "fable") or full model names.
CLI_MODEL = os.environ.get("RESEARCH_ASSISTANT_CLI_MODEL", "opus")
CLI_BULK_MODEL = os.environ.get("RESEARCH_ASSISTANT_CLI_BULK_MODEL", CLI_MODEL)
CLI_TIMEOUT_S = int(os.environ.get("RESEARCH_ASSISTANT_CLI_TIMEOUT", "1200"))

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
_budget: dict = {"exceeded_at": None}
STEP_DOWN = {"max": "xhigh", "xhigh": "high", "high": "medium", "medium": "low", "low": "low"}


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
        _budget["exceeded_at"] = None


def cost_cap() -> float | None:
    """Per-run cap in USD: env RESEARCH_ASSISTANT_MAX_COST_USD, else config gates.max_cost_usd."""
    env = os.environ.get("RESEARCH_ASSISTANT_MAX_COST_USD")
    if env:
        return float(env)
    from .config import rubric_cfg
    cap = rubric_cfg().get("gates", {}).get("max_cost_usd")
    return float(cap) if cap is not None else None


def spent_usd() -> float:
    with _lock:
        return sum(u["est_cost_usd"] for u in _usage.values())


def over_budget() -> bool:
    cap = cost_cap()
    return cap is not None and spent_usd() >= cap


def effective_effort(effort: str) -> str:
    """Once the cost cap is reached, every later call runs one effort level lower."""
    return STEP_DOWN.get(effort, effort) if over_budget() else effort


def record_usage(stage: str, model: str, usage) -> None:
    get = lambda name: getattr(usage, name, None) or 0  # noqa: E731
    inp, out = get("input_tokens"), get("output_tokens")
    c_read, c_write = get("cache_read_input_tokens"), get("cache_creation_input_tokens")
    p_in, p_out = PRICES.get(model, PRICES["claude-opus-5-5"])
    cost = (inp * p_in + out * p_out + c_read * p_in * CACHE_READ_X.get(model, 0.1)
            + c_write * p_in * CACHE_WRITE_X) / 1e6
    add_usage(stage, inp, out, c_read, c_write, cost)


def add_usage(stage: str, inp: int, out: int, c_read: int, c_write: int, cost: float) -> None:
    with _lock:
        u = _usage[stage]
        u["calls"] += 1
        u["input_tokens"] += inp
        u["cache_read_tokens"] += c_read
        u["cache_write_tokens"] += c_write
        u["output_tokens"] += out
        u["est_cost_usd"] += cost
    cap = cost_cap()
    if cap is not None and _budget["exceeded_at"] is None and spent_usd() >= cap:
        _budget["exceeded_at"] = stage


def usage_report() -> dict:
    with _lock:
        stages = {s: {k: round(v, 4) if k == "est_cost_usd" else int(v) for k, v in u.items()} for s, u in _usage.items()}
    keys = ["calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "est_cost_usd"]
    total = {k: sum(st.get(k, 0) for st in stages.values()) for k in keys}
    total["est_cost_usd"] = round(total["est_cost_usd"], 4)
    budget = {"cap_usd": cost_cap(), "spent_usd": total["est_cost_usd"], "exceeded_at_stage": _budget["exceeded_at"]}
    note = ("Backend claude-cli: usage counts against your Claude plan; cost is Claude Code's API-equivalent "
            "estimate, not a charge." if BACKEND == "claude-cli"
            else "Cost is estimated from list prices; check the Console for actual billing.")
    return {"by_stage": stages, "total": total, "budget": budget, "backend": BACKEND, "note": note}


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


# ------------------------------------------------------- claude-cli backend --


def claude_bin() -> str:
    exe = os.environ.get("RESEARCH_ASSISTANT_CLAUDE_BIN") or shutil.which("claude")
    if not exe:
        raise RuntimeError("RESEARCH_ASSISTANT_BACKEND=claude-cli but the `claude` command was not found on PATH. "
                           "Install Claude Code or set RESEARCH_ASSISTANT_CLAUDE_BIN to its full path.")
    return exe


def build_cli_command(exe: str, system_file: str, schema: type[BaseModel], *, effort: str, bulk: bool) -> list[str]:
    """Arguments for one headless call (pure function; unit-tested).

    The system prompt goes through a file: the shared paper digest is far longer than the
    Windows command-line limit (~32k characters).
    """
    return [
        exe, "-p",
        "--output-format", "json",
        "--json-schema", json.dumps(schema.model_json_schema(), separators=(",", ":")),
        "--system-prompt-file", system_file,
        "--model", CLI_BULK_MODEL if bulk else CLI_MODEL,
        "--effort", effort,
        "--tools", "",              # no tools: a pure prompt -> JSON call
        "--no-session-persistence",
    ]


def parse_cli_output(stdout: str, schema: type[T], stage: str) -> tuple[T, dict]:
    """Validate the CLI's JSON envelope and return (parsed object, envelope)."""
    try:
        env = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{stage}: claude CLI did not return JSON: {stdout[:300]!r}") from e
    if env.get("is_error") or env.get("subtype") != "success":
        raise RuntimeError(f"{stage}: claude CLI error ({env.get('subtype')}, "
                           f"api_error_status={env.get('api_error_status')}): {str(env.get('result'))[:300]}")
    data = env.get("structured_output")
    if data is None:  # fall back to the text result if no structured block came back
        data = json.loads(env.get("result") or "null")
    return schema.model_validate(data), env


def record_cli_usage(stage: str, env: dict) -> None:
    """Token counts per model from `modelUsage`; cost from Claude Code's own estimate."""
    per_model = env.get("modelUsage") or {}
    if not per_model:
        u = env.get("usage") or {}
        add_usage(stage, u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cache_read_input_tokens", 0),
                  u.get("cache_creation_input_tokens", 0), env.get("total_cost_usd") or 0.0)
        return
    for m in per_model.values():
        add_usage(stage, m.get("inputTokens", 0), m.get("outputTokens", 0), m.get("cacheReadInputTokens", 0),
                  m.get("cacheCreationInputTokens", 0), m.get("costUSD", 0.0))


def ask_cli(system: str, user: str, schema: type[T], *, stage: str, effort: str, bulk: bool,
            shared_context: str | None) -> T:
    # Shared digest first, so consecutive calls share the longest possible identical prefix.
    full_system = f"{shared_context}\n\n{system}" if shared_context else system
    with tempfile.TemporaryDirectory(prefix="ra-cli-") as tmp:  # neutral cwd: no project CLAUDE.md is loaded
        system_file = Path(tmp) / "system.txt"
        system_file.write_text(full_system, encoding="utf-8")
        cmd = build_cli_command(claude_bin(), str(system_file), schema, effort=effort, bulk=bulk)
        try:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                                  cwd=tmp, timeout=CLI_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"{stage}: claude CLI timed out after {CLI_TIMEOUT_S}s") from e
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"{stage}: claude CLI exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    parsed, env = parse_cli_output(proc.stdout, schema, stage)
    record_cli_usage(stage, env)
    return parsed


# ---------------------------------------------------------------------- ask --


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
    effort = effective_effort(effort)
    if BACKEND == "claude-cli":
        return ask_cli(system, user, schema, stage=stage, effort=effort, bulk=bulk, shared_context=shared_context)
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
