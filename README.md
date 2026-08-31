# RAG over a corpus of academic papers

Large language models are trained on enormous, general corpora — most of which
is irrelevant to any specific topic, and which the model recalls only fuzzily.
This project builds a focused retrieval layer over a small set of academic
papers so that questions are answered from *those papers* rather than from the
model's diffuse training memory. You ask a question; the system finds the most
relevant passages and has an LLM answer using only them. It is a deliberately
minimal, framework-free build (no LangChain or LlamaIndex) — the goal is to
understand and be able to defend every stage of the pipeline, not to ship a
production system.

## Approach

The pipeline runs in six stages, in the order data flows:

1. **Extract** — pull text from the PDFs (PyMuPDF), repairing line-wrap
   artifacts and stripping references and figure-axis noise.
2. **Chunk** — split each paper into ~400-token passages by packing whole
   paragraphs, with overlap between them.
3. **Embed** — turn each chunk into a 1536-dim vector with OpenAI's
   `text-embedding-3-small`, where similar meaning maps to nearby vectors.
4. **Store** — hold the vectors in a FAISS index for fast similarity search.
5. **Retrieve** — embed the question the same way and find the nearest chunks.
6. **Generate** — put the retrieved chunks in a prompt and have an LLM answer
   grounded in them.

The corpus is 11 papers, which chunk to 491 passages. The vectors are computed
once and persisted, so only a change to the papers triggers a rebuild; querying
just loads them.

## Decisions

**No framework.** The stages are written directly against the APIs rather than
through LangChain/LlamaIndex. Those frameworks hide exactly the steps this
project exists to understand, so building the pipeline by hand was the point.

**Chunk size (~400 tokens) with overlap.** Chunk size is a trade-off. Too large
and a chunk's single vector averages several topics together, so it matches any
specific question only weakly. Too small and a chunk loses the surrounding
context that makes it meaningful. Paragraph-packing (rather than blind
token-splitting) keeps chunks coherent; overlap between consecutive chunks means
a fact that straddles a boundary still appears intact in at least one chunk.
Chunks are counted with the same tokeniser the embedding model uses, so the
budget is the real one the model sees.

**Embedding model — `text-embedding-3-small`.** 1536 dimensions, inexpensive,
and adequate for a modest corpus. The larger model is worth reaching for only
when retrieval quality is the bottleneck, which it is not here.

**Vector store — FAISS `IndexFlatIP`.** "Flat" means exact search (it checks
every vector — appropriate at a few hundred vectors, no approximation needed).
"IP" means inner product, which equals cosine similarity *because the embedding
vectors are unit-length* — so the dot product directly measures directional
(semantic) similarity.

**Grounding and refusal — two separate mechanisms.** The answer is grounded in
the corpus by *putting the retrieved chunks in the prompt* and instructing the
model to answer using only that context. Separately, the generation temperature
is set to 0 so the answer is deterministic and faithful to the context rather
than creative — reducing embellishment. The prompt also instructs the model to
say it cannot find the answer when the context does not contain it, rather than
guessing from outside knowledge.

## Demonstration

Three questions test the system's behaviour:

1. **A question answered in the papers** (the rats' behavioural task) — returns a
   correct answer, grounded in the retrieved passages.
2. **Clearly out-of-corpus general knowledge** ("What is the capital of
   France?") — the system refuses, saying it cannot find the answer in the
   papers, even though the model plainly knows it from training. This shows the
   grounding instruction overrides the model's own knowledge.
3. **Plausibly in-domain but absent** (the role of dopamine in reward-prediction
   error) — the harder case: retrieval returns semantically *related* but
   non-answering chunks, and the system still refuses rather than stitching a
   confident answer from tangential text.

Together these show the behaviour that matters most in a RAG system: correct
grounded answers when the corpus contains the answer, and honest refusals when
it does not — including when retrieval returns related-but-wrong context.

## Evaluation

Retrieval was evaluated with a small synthetic golden set: for a sample of
chunks, an LLM was prompted to write a question that chunk answers, in different
wording from the passage (so retrieval is tested on meaning, not shared
vocabulary). Because each question's source chunk is known, recall@k is exact —
run the question through retrieval and check whether the source chunk is in the
top k.

**Recall@5 was 0.60 on a 10-question set.** Inspecting the misses showed that
most were not retrieval failures but a limitation of exact-source matching:
generic questions (e.g. "what statistical methods were used") are answered
equally well by many chunks across the 11 papers, so retrieval returned valid,
on-topic passages from *other* papers rather than the exact source chunk. In
every miss inspected, the retrieved chunks were relevant to the question.
Exact-source recall@k therefore *understates* true semantic retrieval quality
whenever a question has multiple valid answers.

The evaluation has two honest limitations. First, synthetic questions skew
toward the easily-answerable — they test retrieval and basic grounding, not
hard or multi-hop reasoning. Second, exact-source matching (above) penalises
valid non-source hits. A production evaluation would add human-written hard
cases and use LLM-as-judge — scoring whether the retrieved chunks *answer* the
question rather than exact-matching the source — to credit any valid chunk.

## Caveats and limitations

- **Retrieval evaluation is a small synthetic bootstrap.** See Evaluation
  above: recall@5 = 0.60, understated by exact-source matching; no large or
  human-labelled test set.
- **Extraction quality varies by layout.** Single-column papers extract
  cleanly; two-column journals produce more artifacts, and retrieval quality is
  bounded by that noise. Cleaning is deliberately minimal.
- **Untuned and minimal by design.** A single embedding model, a fixed `k=5`, a
  small corpus, and no re-ranking, query rewriting, or hybrid search — all
  deliberately out of scope for a first, understandable build.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# put a .env with OPENAI_API_KEY=... in the project root, then:
python -m scripts.build_embeddings   # embed the papers once
python main.py                       # ask questions interactively
python -m scripts.build_eval         # generate the eval set and measure recall@5
```

## Repository structure

```
src/
├── extract.py     # PDF text extraction and cleaning
├── chunk.py       # paragraph-packing chunker (token-budgeted)
├── embed.py       # embedding + persistence (build/load corpus)
├── store.py       # FAISS index
├── retrieve.py    # query -> nearest chunks
└── generate.py    # prompt building + LLM call (full RAG)
scripts/
├── build_embeddings.py   # one-off: extract -> chunk -> embed -> save
└── build_eval.py         # synthetic eval generation + recall@5
notebooks/
└── demo.ipynb            # step-by-step walkthrough + the three test questions
main.py                   # interactive question-answering CLI
```
