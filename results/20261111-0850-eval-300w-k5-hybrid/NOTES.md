# Notes

Invalid as a Table 3 row: do not use for quality.

Every answer-expecting question was refused as not_in_papers right after the understand step. The decide step compares the top retrieval score with 7.0, a threshold calibrated on raw BM25 scores, but hybrid hits carry reciprocal-rank-fusion scores (about 0.03), so every question fell below it. Found from the step traces (understand only, no draft). Cost about 11K gpt-oss-20b tokens and no draft or check tokens. Fixed by always deciding on the lexical score; re-run separately.
