# Notes

One streamed call per chat model available to this key, max_tokens 64, prompt "Reply with the single word OK." Four requests spent.

- Llama models are not available to this key (`GET /models` lists none), so the gpt-oss, qwen and allam models are the candidates.
- gpt-oss-20b, gpt-oss-120b and qwen3.8-27b each report 1000 requests per day and 8000 tokens per minute; allam-2-7b reports 7000 and 6000. gpt-oss-120b still showed 999 remaining after the gpt-oss-20b call, so buckets are per model.
- Tokens per minute are charged prompt plus max_tokens at admission, not tokens actually generated: gpt-oss-20b used 78 in and 31 out but remaining tokens fell by 142 = 78 + 64. qwen (19 + 64 = 83) and allam (14 + 64 = 78) match the same rule.
- The daily request budget refills continuously: reset-requests read 1m26.4s for a 1000/day model (86400 s / 1000) and 12.342s for allam (86400 s / 7000). One request per 86.4 s comes back, rather than the whole budget at midnight.
- Requests per minute and tokens per day do not appear in any header.
- gpt-oss models stream reasoning before content: first chunk 507 ms and 396 ms, first content token 544 ms and 473 ms. qwen and allam returned content in the first chunk at about 136 ms.
