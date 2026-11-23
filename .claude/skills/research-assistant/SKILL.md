---
name: research-assistant
description: Run the research-assistant agent on a research topic and summarize the result. It searches real literature databases, finds gaps, proposes and critiques research directions, verifies every citation, and recommends the top 1-2. Use when the user asks to find research directions, gaps or a thesis/dissertation topic for a subject, or invokes /research-assistant with a topic.
---

# Research assistant

Runs this repository's pipeline (`research_assistant` package) on the user's topic through the
Claude Code CLI backend, so it uses the user's Claude plan and needs no API key. Then it summarizes
the report from the output files. The pipeline does the research; your job is to launch it, wait,
and report faithfully.

## 1. Get the input

From the arguments or the message, take:
- **Topic** (required): 1-3 sentences. If it is missing, ask for it and stop.
- **Context** (optional): career stage, time horizon, resources, methods. Keep it in the same
  string after a semicolon, e.g. `Microplastics in freshwater fish; MS thesis, 1 year, field access.`
- **Field** (optional): pass it as `--field "..."` only if the user named one.
- **Model**: `sonnet` by default. Use `opus` only if the user asks for best quality or for Opus.

Don't invent context the user didn't give; the pipeline handles missing context.

## 2. Find the program

Use the first one that exists:
1. the env var `RESEARCH_ASSISTANT_EXE`, if set;
2. `C:\venvs\research-assistant\Scripts\research-assistant.exe` (Windows default);
3. `research-assistant` on PATH.

If none exists, stop and tell the user to set it up once from the repository root:
```
py -m venv C:\venvs\research-assistant
C:\venvs\research-assistant\Scripts\python -m pip install -e ".[dev]"
```

## 3. Run it in the background

A run takes about 5-15 minutes. Run it from the repository root as a background command, so you
are notified when it finishes. Don't poll. Set these environment variables for that command only:
- `RESEARCH_ASSISTANT_BACKEND=claude-cli`
- `RESEARCH_ASSISTANT_CLI_MODEL=<model from step 1>`
- `PYTHONUTF8=1`

Bash form:
```bash
RESEARCH_ASSISTANT_BACKEND=claude-cli RESEARCH_ASSISTANT_CLI_MODEL=sonnet PYTHONUTF8=1 \
  "/c/venvs/research-assistant/Scripts/research-assistant.exe" "<topic; context>" --out runs
```
PowerShell form:
```powershell
$env:RESEARCH_ASSISTANT_BACKEND="claude-cli"; $env:RESEARCH_ASSISTANT_CLI_MODEL="sonnet"; $env:PYTHONUTF8="1"; & "C:\venvs\research-assistant\Scripts\research-assistant.exe" "<topic; context>" --out runs
```
Use `CONTACT_EMAIL` only if it is already set in the environment. Never fill it in with the user's
email address yourself.

Tell the user in one line that the run has started and roughly how long it takes.

## 4. When it finishes

The last lines of the output name the report path. The run folder is `runs/<timestamp>-<slug>/`.

**If the command failed**, show the last error lines and match them against the table below.
Don't retry more than once without the user's go-ahead.

| Error | Meaning / fix |
|---|---|
| `claude command was not found` | Set `RESEARCH_ASSISTANT_CLAUDE_BIN` to the output of `where.exe claude` |
| `claude CLI error ... auth` | The user needs to run `claude` once and log in |
| A usage-limit message | The Claude plan limit was reached; try later or use `haiku` |
| `Stopped early` | Too little literature; suggest broadening the topic or adding `--field` |

**If it succeeded**, read `qc_report.json` first, then `report.md`. Reply with:

1. **Flags first**, if any apply:
   - `stopped_early: true`: say so and give the report's "what to try next" points; skip the rest.
   - `low_confidence: true`: say verification thresholds weren't met, and why (`unsupported_rate`, `doi_match_rate`).
2. **Recommendation:** the top 1-2 directions, each with its research question and a one-line reason.
3. **All candidate directions:** one line each (title, risk, the critic's verdict).
4. **Key gaps:** 2-4 short bullets.
5. **Run stats:** papers found, claims checked, unsupported rate, DOI match rate, and from
   `token_usage.total` the calls, output tokens and estimated cost. On this backend the cost is an
   API-equivalent estimate, not a charge.
6. **Links** to `report.md` and the run folder.

Summarize; don't paste the whole report. Use only what the files say; never add papers, numbers or
directions that aren't in them. After the summary, the user can ask follow-ups about the report;
answer them from the files.
