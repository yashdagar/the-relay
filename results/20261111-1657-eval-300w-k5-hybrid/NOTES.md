# Notes

Table 3 run for 300w-k5-hybrid with the decide-score fix (3399db1). Same 20 questions and spacing as results/20261111-0837-eval-300w-k5-bm25.

- 15 of 15 answerable correct, inj-02 correct, all 4 refusal questions refused.
- The three Table 3 settings differ by one question each (dc-04 here, dc-01 at 120w-k8, dc-04 at 300w-k5-bm25 where the draft refused). dc-04 has also flipped between runs with identical passages, so 20 questions cannot separate these settings on quality.
- dc-04 took 11395 ms: understand 388 ms; a first draft (560 ms) stating a computed "gain of 6.4 percentage points" that the grounding check rejected; 9057 ms queued in the governor before the second draft, because the first draft's reservation had used the per-minute token budget for gpt-oss-120b; second draft 863 ms; check 461 ms. All before the first token.
