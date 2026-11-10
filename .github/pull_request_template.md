## What

<!-- One or two sentences on the change. -->

## Why

<!-- The problem or measurement that motivated it. -->

## Trade-off

<!-- Which goal this gives up for which (speed, quality, tokens, availability), or "none". -->

## Evidence

<!-- Before and after numbers from a run, with the results directory path. Write "not measured" if it was not. -->

- Run:
- Real or stand-in:
- Before:
- After:

## Checklist

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check` and `uv run ruff format --check` pass
- [ ] No API key or `.env` contents in the diff, logs or results
- [ ] Raw results are new files, not edits to existing runs
- [ ] Config values changed here are reflected in the write-up
