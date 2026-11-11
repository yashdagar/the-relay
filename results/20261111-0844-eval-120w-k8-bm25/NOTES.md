# Notes

Table 3 run, same 20 questions and code as results/20261111-0837-eval-300w-k5-bm25.

- 14 of 15 answerable correct, inj-02 correct, all 4 refusal questions refused.
- dc-01 graded wrong: with 120-word pieces the passage naming the smoothing module was not retrieved, and the draft answered "a routing module ... and a downsampler" from the figure caption piece. Grounded and passed by the check model, but it names the parts of the chunking layer, not of dynamic chunking. This is the split-fact failure the offline retrieval report predicted for 120-word pieces.
