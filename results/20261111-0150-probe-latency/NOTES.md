# Notes

20 streamed draft calls per model, run concurrently across models and spaced 22 s apart within a model. Prompt: system instructions, the five 300w-k5-bm25 passages for each of the first 20 answer-expecting gold questions, and the question; max_tokens 400; temperature 0; gpt-oss with reasoning_effort low. 60 requests, none rate-limited or failed.

Latency (ms), n = 20 each, p50 / p95 / max:

- qwen/qwen3.8-27b: first content token 329 / 486 / 1179; total 482 / 1301 / 1766; 2346 tokens in, 121 out on average.
- openai/gpt-oss-20b: first content token 614 / 1877 / 1918; total 680 / 1925 / 1960; 2237 in, 134 out.
- openai/gpt-oss-120b: first content token 481 / 870 / 1526; total 697 / 1611 / 1759; 2237 in, 118 out.

gpt-oss models stream hidden reasoning first, so their first chunk (p50 447 and 292 ms) arrives well before the first content token. With n = 20 the p95 is close to the second-slowest call and should be read as rough.

Draft quality, graded with relay.eval at commit 3aedbba by matching cited ids in the text to the retrieved passages (a throwaway script, not committed):

- qwen/qwen3.8-27b: 17 correct, 3 wrong.
- openai/gpt-oss-20b: 16 correct, 3 wrong, 1 refused wrongly.
- openai/gpt-oss-120b: 16 correct, 3 wrong, 1 refused wrongly.

Wrong answers worth remembering: qwen on att-02 says "The papers do not say" and then quotes warmup_steps = 4000 from the passage; gpt-oss-120b on mos-08 adds "Chen et al., 2022" from memory; gpt-oss-20b on mos-07 misses the 2000 hours because the retrieved piece ends at the page break before "of 2000 hours". All three models refuse mos-06, where retrieval misses the only piece with both facts. gpt-oss-20b on dc-01 cited in the display format "[paper §section p.N]" because this probe's prompt showed both the id and the display citation; the pipeline prompt should show ids only.

After 20 calls the reset-requests header read 28m48s on each model, which is 20 × 86.4 s: the daily request budget refills one request every 86.4 s.
