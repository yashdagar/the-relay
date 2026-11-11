# Notes

40 tiny streamed calls (max_tokens 8) fired at once on allam-2-7b, then one call every 5 s until one succeeded.

- 33 succeeded and 7 got HTTP 429, against a stated limit of 30 requests per minute. The 429 body says "Limit 30, Used 30, Requested 1", so 3 extra calls slipped through under concurrency.
- The 429 came back fast (227 to 455 ms), with `retry-after: 2` and `type: requests`. It is a fast refusal, not a slow one.
- The first recovery poll, 5.1 s after the burst, succeeded. Recovery is between 2 s (retry-after) and 5.1 s; the poll interval was too coarse to say more.
- No separate limit on requests in flight showed up at 40 concurrent calls.
- Error bodies name the Groq organization. This file was redacted once (7 occurrences replaced with org_REDACTED) before its first commit; the client now redacts at capture time.
