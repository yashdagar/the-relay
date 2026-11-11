# Findings log

Internal. Each finding, the run that showed it, and what changed. Runs are under `results/`.

## Real service

- **Per-model buckets: 1,000 requests per day and 8,000 tokens per minute each.** `20261111-0148-probe-limits`. Led to using a different model per step so their budgets add up.
- **Tokens per minute charged as prompt plus `max_tokens` at admission.** `20261111-0148-probe-limits` (142 charged = 78 + 64). The governor and stand-in charge the same way; `max_tokens` became a throughput setting.
- **Daily requests refill continuously, one per 86.4 s.** `20261111-0148-probe-limits`, `20261111-0150-probe-latency` (28m48s after 20 calls). The budget reply says when one request comes back, not "tomorrow".
- **`retry-after` rounds up to whole seconds; the body has the exact wait.** `20261111-0200-probe-tpm` (1 s against 82.5 ms). Retries wait the body's time.
- **33 of 40 got through a limit of 30.** `20261111-0159-probe-rpm`. The stand-in does not model this overshoot.
- **gpt-oss streams hidden reasoning before content.** `20261111-0150-probe-latency`. The client times first chunk and first content token separately.

## Quality

- **Five grader bugs, found by grading real output.** `20261111-0150-probe-latency`: lenticular citation brackets, digits in citation ids, narrow no-break spaces, non-breaking hyphens, digits inside names. Grader normalises text before checking facts and numbers. Later, "eight" for 8.
- **Router refused 7 of 25 answerable questions.** `20261111-0229-eval-300w-k5-bm25`. It only saw paper titles; a glossary of each paper's own names took routing from 27 to 34 of 36 (`20261111-0309-router`).
- **A truncated draft passed the check.** `20261111-0315-eval-300w-k5-bm25` (dc-07). Drafts that stop on length are now redrafted.
- **Vague redraft feedback did not fix a computed number.** `20261111-0315`; feedback now names the number, which fixed dc-04 in `20261111-0320`.
- **Sub-question passages were dropped on the lean path.** `20261111-0315` (x-01). Passages are interleaved by rank, and sub-searches are scoped to the paper the router names.
- **The strict number check rejects false-premise corrections.** `20261111-0320` (inj-03, "160 ms, not 5 ms"). Kept strict by choice.
- **Hybrid refused everything.** `20261111-0850-eval-300w-k5-hybrid`: the 7.0 threshold was on the BM25 scale. Decide now always uses the BM25 score; re-run in `20261111-1657`.
- **Table 3 settings cannot be separated with 20 questions.** `20261111-0837`, `20261111-0844`, `20261111-1657`: one question apart, and dc-04 flips between runs.
- **Still failing:** mos-06 (retrieval miss), dc-07 ("data budget" in the wrong section), x-01 (lean path keeps only the top passage per search), inj-03 (strict numbers).

## Load and limits

- **The governor never learned the daily budget was gone** because it stopped sending before Groq could say so. Scaled load test; now an empty local daily bucket counts as exhausted, and the pipeline answers before any model call.
- **Requests queue rather than retry.** All load shapes: retries per request 0.00 to 0.01 while up to 40% hit a limit. Waits up to 13.7 s in the governor.
- **Zero wrong answers in every load shape.** Degradation shows up as passages instead of answers, never as a worse answer.
- **The check step is 77 to 79% of the cost** at list prices, because qwen3.8-27b is 5x the drafter's price and rereads every passage.

## Process

- **Laptop sleep invalidated runs twice.** `20261111-0348-load-*` and the 08:36 steady and rpm_halved runs. `caffeinate` does not stop lid-close sleep on battery.
- **An unquoted zsh variable produced three empty runs.** `evaluate` now refuses an `--ids` that matches nothing.
