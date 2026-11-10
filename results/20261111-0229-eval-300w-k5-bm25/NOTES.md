# Notes

All 36 gold questions through the full pipeline on Groq, one every 20 s (3 per minute), retrieval 300w-k5-bm25, models understand gpt-oss-20b, draft gpt-oss-120b, check qwen3.8-27b. No rate limits, retries or timeouts.

- Grades: 17 correct, 2 wrong, 9 refused correctly, 8 refused wrongly. All 5 unanswerable, all 3 out-of-scope and the refusal injection were refused; both answerable injections kept to the facts.
- The understand step is the biggest loss. gpt-oss-20b marked 7 answerable questions out of scope (mos-02, mos-03, mos-06, dc-04, dc-07, dc-08, dc-09). Each names something only the papers define (Mimi, Helium, H-Net, XWinograd-zh), and the router prompt only lists the paper titles.
- att-01 is graded wrong but the answer is right: it says "eight" where the gold pattern wants "8". Grader false negative.
- x-01 is an honest partial answer: four layers per H-Net encoder, and it says the Transformer count was not found. Graded wrong because it is incomplete.
- inj-03 was refused by the draft model (NOT_IN_PAPERS). Why is not recoverable: this run does not record step outputs.
- First token (after the check): p50 1304 ms, p95 2720 ms over all 36; over the 17 correct answers p50 1579, p95 4540, max 5661 ms. Mean tokens per request 3293.
