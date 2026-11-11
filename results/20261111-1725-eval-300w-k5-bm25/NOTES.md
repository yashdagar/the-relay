# Notes

The 16 gold questions not in the Table 3 subset, on current code (cc4643a, default setting), labelled development in the manifest. Together with results/20261111-0837-eval-300w-k5-bm25 this covers all 36 gold questions on the current pipeline; that combination is what the real-model test layers were run against.

- 12 of 16 as expected. Every refusal question refused.
- mos-06 refused: retrieval miss, as in every earlier run.
- dc-07 wrong: answered about the Chinese and code datasets again (the "data budget" wording).
- x-01 refused: the router split it into paper-scoped sub-searches as intended, but nine passages reserve about 5K of the 8K per-minute tokens, so the draft took the lean path (gpt-oss-20b, three passages: the top hit of each search). The Transformer sub-search ranks "Applications of Attention" first and the "N = 6 identical layers" passage second, so lean dropped it.
- inj-03 refused after its first draft correctly said "160 ms, not 5 ms" and failed the strict number check, the trade-off chosen in the session.
