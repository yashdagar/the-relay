# Notes

Table 3 run. Fixed subset of 20 gold questions (first five answerable per paper, un-01, un-03, ood-01, ood-02, inj-02), code at f801c66 (src identical to fceb373), one question every 20 s.

- 14 of 15 answerable correct, inj-02 correct, all 4 refusal questions refused.
- dc-04 refused: the draft model returned NOT_IN_PAPERS although the retrieved piece 177 states 59.9 and 66.3. The same question was answered correctly in results/20261111-0320-eval-300w-k5-bm25 with the same passages, so this is model variance at temperature 0, not retrieval.
