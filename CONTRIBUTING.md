# Conventions

## Branches

- `main` is always runnable: `uv sync && uv run pytest` passes.
- Work happens on short-lived branches named `<area>/<short-description>`, for example `pipeline/check-step`, `standin/latency-model`, `results/burst-run`.
- Areas match the module names in `src/relay/`, plus `results`, `docs` and `chore`.

## Commit messages

```
<area>: <imperative summary, at most 72 characters>

Why the change was needed, and what it costs if it trades one goal
for another (speed, quality, tokens, availability). Include the
measurement that motivated it when there is one.
```

- The subject is imperative and has no trailing period: `governor: sync buckets from ratelimit headers`, not `Added header syncing.`
- The body explains the why. The diff already shows the what.
- One logical change per commit. Before review, squash commits that belong to the same change. After review starts, add new commits on top and never rewrite pushed history.
- A commit that adds a results run says which code commit produced it: `results: add burst run from a1b2c3d`.

A template is in `.gitmessage`. Enable it with `git config commit.template .gitmessage`.

## Pull requests

- Use the template in `.github/pull_request_template.md`.
- A PR that changes behaviour under load includes before and after numbers from a run, not a description of what should happen.
- Keep PRs to one area where possible, so a reviewer can tie a measurement to a single change.

## Code

- Python 3.12, managed with `uv`. Dependencies are declared in `pyproject.toml`; never `pip install`.
- `uv run ruff format` and `uv run ruff check` must pass.
- No comments that restate the code. Design reasoning goes in commit messages and the write-up.
- Every timeout, limit, threshold and retry count lives in one config module with its unit in the name (`draft_timeout_ms`, `max_retries`), so the write-up can quote it and a reviewer can change it.

## Tests

- Test behaviour through public interfaces: a question in, an answer, refusal or degraded reply out.
- Mock only at the boundary, the model HTTP API. Use the stand-in service for that, not hand-written mocks of internal classes.
- One behaviour per test, with the intent in the test name.

## Results and data

- `results/` is append-only. Each run goes in its own directory, `results/<yyyymmdd-hhmm>-<scenario>/`, with the raw CSV, the config used, and the code commit hash.
- Never edit a raw results file by hand. If a run was wrong, keep it, say why in the run's `NOTES.md`, and run again.
- Every number in the write-up must be computable from `results.csv` by `src/relay/tables.py`.

## Secrets and provider terms

- The Groq key lives only in `.env`, which is git-ignored. Never paste it in code, logs, results or the conversation export.
- Use one free account and stay within its limits. No extra accounts or key rotation to get around limits.
- Plan real requests against the daily budget before running anything against Groq; use the stand-in for everything else.
