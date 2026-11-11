# Notes

- `run.json` field `started` holds the finish time for this run, not the start. The benchmark took the timestamp after running; fixed in the commit after e99e166. The directory name carries the real start time (01:40).
- Run on an Apple M1 laptop with other applications open. Latency tails for the hybrid settings varied between runs: an earlier unrecorded run of the same code measured 300w-k5-hybrid at p95 29.7 ms, p99 43.0 ms, max 115 ms, against p99 9.5 ms here. BM25 stayed under 13 ms at the max in both.
