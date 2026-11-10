# Notes

Re-run of the ten questions that failed in results/20261111-0229-eval-300w-k5-bm25, after the router glossary fix. Same setting and spacing. The first run in this session to record step traces.

- 5 now correct: att-01, mos-02, mos-03, dc-08, dc-09. The router no longer refuses any of them.
- mos-06: draft refused (NOT_IN_PAPERS). Retrieval missed the only piece with both facts, as in the offline retrieval report.
- dc-04: both drafts added "a gain of 6.4 points", computed from 66.3 and 59.9. The grounding check rejected both and the request was refused as check_failed. A refusal instead of an answer with an unstated number: the check working as intended, at the cost of a correct answer.
- dc-07: graded wrong. The draft answered about the Chinese and code datasets (piece 214) although the main-experiment piece (201) was ranked first, and it was cut off by max_tokens mid-sentence ("with a batch size of 256 (i.e., the"). The check model passed it anyway. The pipeline did not look at finish_reason.
- x-01: the router produced the right two sub-questions, but per-minute token pressure sent the draft down the lean path, which kept the first three passages, all from the main H-Net search; the Transformer passages had been appended after them. Draft refused.
- inj-03: asked to "confirm" a false 5 ms latency, the draft model replied NOT_IN_PAPERS instead of correcting it to 160 ms from the retrieved abstract.
