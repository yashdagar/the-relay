# Retrieval, offline comparison

Internal report. Raw data: `results/20261111-0140-retrieval-offline/` (code at `e99e166`). Regenerate with `uv run python -m relay.retrieval_bench --reps 5`.

## What was measured

- **Questions:** the 27 gold questions that expect an answer (25 answerable, 2 injections with an answerable core). The 9 refusal questions have no evidence, so they only count toward latency.
- **Relevance:** a piece is relevant if it contains a gold evidence phrase on whole normalised words. Evidence items may list alternative phrases for the same fact.
- **Quality:** recall@k (share of evidence items found in the top k), MRR, and precision@k (divided by k), with k equal to the setting's own top_k.
- **Latency:** 5 passes over all 36 questions, 180 searches per setting, after the models were loaded. Apple M1, Python 3.12, CPU only.
- **Not measured here:** answer quality, tokens and time to first token. Those need the model and go in Table 3.

## Results

- **120w-k8-bm25:** recall 0.889, MRR 0.614, precision 0.134; p50 0.8, p95 1.5, p99 1.7 ms; 960 words in context.
- **300w-k5-bm25 (chosen):** recall 0.944, MRR 0.806, precision 0.222; p50 0.4, p95 0.8, p99 0.9 ms; 1500 words.
- **300w-k5-hybrid:** recall 0.907, MRR 0.864, precision 0.215; p50 5.5, p95 8.7, p99 9.5 ms; 1500 words.
- **300w-k5-hybrid-rerank:** recall 0.889, MRR 0.762, precision 0.215; p50 1646, p95 1983, p99 2780 ms; 1500 words.
- **600w-k3-hybrid:** recall 0.852, MRR 0.846, precision 0.296; p50 7.6, p95 12.6, p99 13.7 ms; 1800 words.

For the chosen setting, the first relevant piece was at rank 1 for 19 of 27 questions, rank 2 or 3 for 6, rank 4 for 1, and missing for 1.

## Decision

Default is **300w-k5-bm25**.

- It has the highest recall, and recall is what the draft step depends on: a fact that is not retrieved is either refused or, worse, filled in by the model.
- It is the cheapest and steadiest: under 1 ms at p99, no model to load, no tail.
- **Given up:** hybrid ranks the right piece higher (MRR 0.864 against 0.806). With 5 pieces in the prompt, rank 1 against rank 3 matters less than whether the piece is there at all.

## Findings

- **Reranking made things worse and slower.** The MiniLM cross-encoder lost recall (0.889) and MRR (0.762) and cost about 1.6 s per search, with a 4.4 s max. It was trained on short web passages and truncates 300-word scientific pieces. It stays only as a compared row.
- **Small pieces split facts.** At 120 words, dc-01 (routing and smoothing modules) and dc-05 (Mamba-2 layers, four per network) lose one half of the answer to a neighbouring piece.
- **Large pieces with fewer slots lose coverage.** 600w-k3 has the best precision (0.296) but the worst recall (0.852), because three slots fill up with one paper's long sections.
- **Hybrid latency has an unstable tail.** The query embedding runs on CPU and moves with machine load: an earlier unrecorded run measured hybrid p99 at 43 ms and max 115 ms, against 9.5 ms here. BM25 stayed under 13 ms in both.
- **mos-06 is missed by every 300-word setting.** The only piece stating both Helium's 7B parameters and 2.1T tokens is a bullet in the Moshi introduction. BM25 instead ranks generic "parameters" and "tokens" text, including a piece from the Attention paper's positional encoding section.
- **x-01 is half found everywhere.** It needs one piece from each of two papers, and a single search fills all five slots from H-Net. This is what the decide step's extra retrieval round is for.

## Caveats

- **The gold questions borrow the papers' wording.** That favours BM25. Hybrid's expected advantage on paraphrased questions is untested; adding paraphrased questions is the way to test it.
- **Labels were corrected after the first measurement.** mos-02 and mos-06 gained alternative evidence phrases (commit `d91ca93`) once tables and repeated statements were found to answer them. This raised hybrid recall from 0.870 to 0.907 and did not change BM25 recall.
- **Latency is for one search at a time.** Under load, searches share the CPU with the event loop; BM25's sub-millisecond cost makes that negligible, which hybrid's would not be.

## For the write-up

- **Retrieval under a close limit:** retrieval costs no model tokens, so it is never what gets cut. When the limit is close, reduce pieces sent to the model (k=5 to k=3), not retrieval itself.
- **Naive retriever examples:** mos-06 (generic terms pull in the wrong paper), x-01 (one paper crowds out the other), dc-01 at 120 words (a fact split across pieces).
