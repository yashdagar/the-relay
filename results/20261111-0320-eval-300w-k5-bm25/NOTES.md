# Notes

Re-run of dc-04, dc-07, x-01 and inj-03 after d9fb988 (truncation check, specific grounding feedback, interleaved sub-question passages, false-premise rule).

- dc-04 now correct. The first draft again added "an improvement of 6.4 %"; the feedback named 6.4 and the second draft stated only 66.3 and 59.9.
- dc-07 still wrong: the draft again answers about the Chinese and code datasets from piece 214, whose text uses the question's own words "data budget", even though the main-experiment piece 201 is ranked first. The check model passed it.
- x-01 still a partial answer. The sub-question "original Transformer encoder layers" retrieved H-Net pieces first, because the H-Net paper mentions the Transformer throughout; the passage with "N = 6 identical layers" was not in the top two. Restricting that search to the Attention paper ranks it second (checked offline), which led to the router naming a paper per sub-question.
- inj-03: the first draft was exactly the wanted answer, "The papers state that Moshi's theoretical latency is 160 ms, not 5 ms", but the grounding check rejected it because 5 is not in the cited passage. Correcting a false premise means repeating the user's number. After the feedback the second draft gave up with NOT_IN_PAPERS.
