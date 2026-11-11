# Notes

Understand step alone (gpt-oss-20b) on all 36 gold questions, with the glossary prompt from commit 6b5cf62, one call every 8 s.

- 34 of 36 routed as expected, against 27 of 36 implied by the previous full run (7 answerable questions marked out of scope there).
- All 25 answerable questions were in scope except dc-04, whose call timed out at the 4 s step limit with no output. In the pipeline that request would continue without sub-questions.
- un-03 (WMT 2016 English-to-Romanian BLEU) was marked out of scope. It is about the Transformer paper but not answerable from it, so the final reply is still a refusal; only the stated reason differs.
- Latency p50 537 ms, p95 1315 ms, max 4003 ms (the timeout).
- Sub-questions were produced for att-04, mos-07, dc-05, dc-06 and x-01. x-01 split into "original Transformer encoder layers" and "H-Net encoder network layers", which is what the cross-paper search needs.
