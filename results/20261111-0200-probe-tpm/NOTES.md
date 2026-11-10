# Notes

Four identical draft calls (1929 prompt tokens, max_tokens 400) sent back to back on openai/gpt-oss-20b, about a minute after the latency probe had used the same model.

- Calls 1 to 3 succeeded; remaining tokens fell 5767, 3884, 1930. Call 4 got HTTP 429 in 91 ms with `type: tokens`: "Limit 8000, Used 5807, Requested 2204. Please try again in 82.499999ms."
- `retry-after` said 1 s while the body said 82.5 ms. The header rounds up to whole seconds, so a client that only reads the header waits about 12 times longer than needed for a token limit.
- "Requested 2204" is less than prompt plus max_tokens (2329), so the admission estimate is not exactly the prompt token count; it is close enough to plan with prompt + max_tokens.
- Waiting 82 ms for about 11 tokens means the per-minute bucket refills continuously at about 133 tokens per second (8000 / 60), not once per minute.
