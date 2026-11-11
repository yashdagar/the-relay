# The Relay

A question-answering pipeline over three arXiv papers that cites every claim, refuses when the papers do not say, and keeps doing both when the model provider is slow, failing, or out of budget.

The library, at pinned versions:

- `1706.03762v7` Attention Is All You Need
- `2410.00037v1` Moshi: a speech-text foundation model for real-time dialogue
- `2507.07955v2` Dynamic Chunking for End-to-End Hierarchical Sequence Modeling

## Architecture

### Components

```mermaid
flowchart LR
    subgraph Load["Load generator"]
        Shapes["Load shapes<br/>steady / burst / overload /<br/>halved RPM / day exhausted"]
    end

    subgraph Relay["Relay process"]
        Admit["Admission queue<br/>per-request deadline"]
        Pipe["Pipeline<br/>understand, retrieve, decide,<br/>draft, check"]
        Gov["Budget governor<br/>RPM, TPM, RPD, TPD buckets<br/>synced from response headers"]
        Client["Model client<br/>OpenAI-compatible, streaming,<br/>timeouts, retry policy"]
        Retr["Retrieval tool<br/>BM25 / hybrid / rerank"]
        Lib[("Chunk index<br/>paper, section, page")]
        Rec["Recorder"]
    end

    subgraph Providers["Model providers"]
        Groq["Groq free tier<br/>(real)"]
        Stand["Stand-in service<br/>(simulated)"]
    end

    Shapes --> Admit --> Pipe
    Pipe --> Retr --> Lib
    Pipe --> Gov --> Client
    Client -->|base URL| Groq
    Client -->|base URL| Stand
    Pipe --> Rec --> CSV[("results/*.csv")]
    Client -. "x-ratelimit-* headers, 429s" .-> Gov
```

- **Model client** speaks the OpenAI-compatible chat completions API with streaming. Groq and the stand-in serve the same protocol, so switching between them is a base URL change and nothing else in the pipeline knows which one it is talking to.
- **Budget governor** is one shared instance per process, because the account's daily limit is shared by everything that runs. It holds local token buckets for every limit and corrects them from the `x-ratelimit-*` headers on each response, since the limits can change without notice.
- **Admission queue** gives each request a deadline (the point where the person gives up). A request whose estimated wait already exceeds its deadline is answered immediately with a degraded reply instead of waiting to time out.
- **Retrieval tool** runs locally and costs no model tokens. It is called through the same timeout wrapper as any other tool.
- **Recorder** writes one row per request in the `results.csv` schema from the brief, with timings split into waiting, working and retrying.

### Request flow

```mermaid
flowchart TD
    Q([Question]) --> U["1. Understand<br/>small model: in scope?, rewritten query,<br/>which paper"]
    U -->|out of scope| R1["Refuse: the papers do not cover this"]
    U --> T["2. Retrieve (tool)<br/>top-k passages with paper, section, page"]
    T --> D{"3. Decide"}
    D -->|"best score below threshold"| R2["Refuse: not found in the papers"]
    D -->|"multi-part question,<br/>at most 2 extra rounds"| T
    D -->|enough evidence| W["4. Draft<br/>large model, streamed,<br/>cites passage ids only"]
    W --> C{"5. Check<br/>citations exist and<br/>support the claims"}
    C -->|pass| A([Cited answer])
    C -->|"fail, first time"| W
    C -->|"fail again"| R3["Refuse rather than send<br/>an unsupported answer"]
```

The branch and loop the brief asks for are the decide step (refuse, retrieve again, or draft) and the check step (redraft once, then refuse).

### The decision that shapes everything: hold the draft until it passes the check

Nothing reaches the person until the draft has passed the check. The draft is generated in full and buffered, checked for grounded citations, and only then streamed out. "First token" in every table means the first token the person sees, which is after the check.

- **Gives up:** time to first token. It becomes roughly draft time plus check time instead of the model's own first-token latency.
- **Gets:** no unchecked text is ever shown, so there is nothing to retract. A wrong or embarrassing answer costs more than a slow one.
- **Consequence for load:** every request holds its slot for the full draft and check, so concurrency limits and deadlines are set on total time, not first-token time.

### Degradation ladder

When the governor says a step cannot be afforded in time, the pipeline steps down instead of queueing until the person gives up:

1. **Full:** gpt-oss-120b drafts; the deterministic grounding check and qwen3.8-27b both check.
2. **Lean:** gpt-oss-20b drafts from three passages; both checks still apply.
3. **Extractive:** no model call; return the best matching passages verbatim with their citations and say that a written answer is unavailable right now.
4. **Honest no:** budget is gone; say so, say when it resets, and do not guess.

A wrong answer is never on the ladder. Losing speed or fluency is acceptable; losing correctness is not.

Models (chosen from `results/20261111-0150-probe-latency`): gpt-oss-20b understands, gpt-oss-120b drafts, qwen3.8-27b checks. Settings live in `src/relay/config.py`.

### Repository layout

```
src/relay/
  library.py         parse PDFs, split into cited pieces
  retrieval.py       BM25, hybrid with local embeddings, rerank
  client.py          streaming model client, timing, rate-limit headers
  governor.py        per-model minute and day buckets, synced from responses
  prompts.py         understand, draft and check prompts
  pipeline.py        the five steps, retries, deadline and ladder
  reply.py           the reply and its per-attempt trace
  eval.py            retrieval metrics, groundedness, grading
  standin.py         stand-in model service
  oracle.py          stand-in answers for gold questions
  probe.py           real-service probes (limits, latency, rpm, tpm)
  route_eval.py      scores the understand step alone
  retrieval_bench.py offline retrieval quality and latency
  evaluate.py        paced real runs over the gold set
  load.py            load shapes against the stand-in
  records.py         results rows and per-request cost
  assemble.py        builds results/results.csv from results/manifest.json
  tables.py          renders the brief's tables to docs/tables.md
data/papers/         the three PDFs at pinned versions
data/questions/      gold questions with evidence and expected facts
results/             one directory per run with NOTES.md, never edited
docs/                write-up (md, tex, pdf), tables, reports, conversation
tests/
```

## Tests

The tests were written before the code and define each layer's contract. Thresholds live in `tests/thresholds.py`; the gold questions in `data/questions/eval.jsonl`.

- **Gold set** (`tests/test_gold_questions.py`): 36 questions (25 answerable, 5 the papers do not answer, 3 out of scope, 3 prompt injections). Every evidence phrase must exist verbatim in its paper and fit inside one piece at every piece size.
- **Metrics** (`tests/unit/`): precision@k, recall@k, MRR, percentiles, groundedness, answer grading and every write-up table, checked on hand-built inputs.
- **Retrieval** (`tests/retrieval/`): recall@k, MRR and precision@k of the default setting on the gold set, search latency, and behaviour such as never returning reference lists.
- **Generation** (`tests/generation/`, `-m real`): groundedness of every sent answer, correct rate, wrong rate, refusal of unanswerable questions, over-refusal.
- **Application** (`tests/application/`): out-of-scope and injection handling against Groq (`-m real`), and pipeline guarantees against the stand-in: unchecked drafts are never sent, first token only after the check, bounded loops, retries, deadlines, daily budget exhaustion.
- **System** (`tests/system/`, `slow`): every load shape against the stand-in, and the submitted `results/results.csv` against the brief's schema.

```sh
uv run pytest                 # everything except Groq calls
uv run pytest -m real         # spends the Groq daily budget
RELAY_EVAL_REPLIES=results/<run>/replies.jsonl uv run pytest -m real   # regrade saved replies for free
```

## Results

- **Write-up:** [`docs/writeup.md`](docs/writeup.md), also as [`docs/writeup.pdf`](docs/writeup.pdf) and a self-contained [`docs/writeup.tex`](docs/writeup.tex).
- **Tables:** [`docs/tables.md`](docs/tables.md), generated from [`results/results.csv`](results/results.csv).
- **Raw runs:** every directory under `results/` has its raw rows, full replies with step traces, `run.json` and a `NOTES.md`. [`results/manifest.json`](results/manifest.json) lists the runs that count; invalid runs stay and say why.
- **Findings log:** [`docs/reports/findings.md`](docs/reports/findings.md).

```sh
uv run python -m relay.probe limits|latency|rpm|tpm   # real service, spends budget
uv run python -m relay.evaluate --setting 300w-k5-bm25 --ids att-01 mos-02
uv run python -m relay.load steady                     # 30 min against the stand-in
uv run python -m relay.assemble && uv run python -m relay.tables
```

## Setup

```sh
uv sync
cp .env.example .env   # add your GROQ_API_KEY
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
