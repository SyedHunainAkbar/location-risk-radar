# Design: Evidence Assistant

## Overview

Two evidence layers over cohort text. Aspect agents (food, service, ambience, plus a
lead) extract grounded quotes with polarity and intensity; Python grounds each quote
and computes every number. A RAG assistant answers standard operator questions from
retrieved snippets only, cites each claim, and is checked by a second-family judge;
answers are precomputed and cached so the app needs no API key.

The LLM discipline from steering is enforced here: models return quotes, labels, and
prose only, never numeric scores. Grounding uses rapidfuzz partial ratio >= 85 and at
least two words; ungrounded quotes are dropped and counted.

Modules:

- `src/lrr/aspects.py`: agent prompts, JSON schema validation with retry, grounding,
  aspect scoring, predicted stars, and the stars evaluation.
- `src/lrr/rag.py`: chunking, the embedding index, retrieval (cosine + optional MMR +
  filters), grounded generation with citations, the judge, and the answer cache.
- `pipeline/06_aspects_rag.py`: the offline driver (stratified aspect sample with a
  JSONL checkpoint, index build, cache precompute, evaluations).

Heavy pieces (sentence-transformers) are imported lazily. LLM calls go through
`lrr.llm` so the mock provider makes everything testable offline.

## aspects.py

```python
ASPECTS = ("food", "service", "ambience")
ASPECT_SCHEMA                                   # list[{quote, polarity, intensity}]
def aspect_prompt(aspect, review_text) -> messages      # few-shot, temp 0
def lead_prompt(review_text) -> messages
def parse_aspect_json(raw) -> list[dict]                # schema-validated
def run_aspect_agent(aspect, text, provider, retries)   # calls llm.chat, retries
def run_lead_agent(text, provider) -> dict
def ground_quote(quote, source) -> bool                 # rapidfuzz >= 85, >= 2 words
def polarity_sign(polarity) -> int                      # +1 / -1 / 0
def aspect_score(items) -> float                        # mean(sign*intensity/3)
def predicted_stars(items) -> float                     # 3 + 2*mean(sign*intensity/3)
def analyze_review(text, provider) -> dict              # all aspects + lead, grounded
def evaluate_predicted_stars(pred, actual) -> dict      # acc, macroF1, MAE, Pearson+CIs
def risk_contrast(location_scores, risk_flag) -> dict   # Mann-Whitney + effect size
```

- `parse_aspect_json` extracts the first JSON array, validates each item has a string
  quote, a polarity in the allowed set, and an integer intensity in 1..3, and raises
  on violation. `run_aspect_agent` retries up to N times (temperature 0), passing the
  schema error back into the prompt.
- `ground_quote` keeps a quote only if it has >= 2 words and rapidfuzz
  `partial_ratio(quote, source) >= 85`. Dropped quotes are counted by the caller.
- Scoring maps polarity to sign (+1/-1/0); score is the mean over grounded items of
  `sign * intensity / 3`; predicted stars is `3 + 2 * score`, clipped to [1, 5].
- `evaluate_predicted_stars`: rounds predicted stars to the nearest star for accuracy
  and macro F1, keeps continuous values for MAE and Pearson r, each with bootstrap
  CIs. `risk_contrast` runs scipy Mann-Whitney U and reports rank-biserial effect size.

## rag.py

```python
def chunk_text(text, max_words=120) -> list[str]
def build_index(reviews, tips, locations, out_dir) -> IndexMeta   # float16 npy + parquet, assert < 50MB
def load_index(index_dir) -> (embeddings, metadata)
def cosine_top_k(query_vec, embeddings, k) -> idx
def mmr(query_vec, embeddings, candidates, k, lambda_) -> idx
def retrieve(query, index, business_id=None, chain=None, k=8, use_mmr=False, max_stars=None, date_range=None)
def answer_prompt(question, snippets) -> messages        # cite [review_id], else "not enough evidence"
def generate_answer(question, snippets, provider) -> str
def judge_prompt(sentence, snippet) -> messages
def judge_sentences(answer, snippets, provider) -> list[(sentence, supported)]
def groundedness_rate(labels) -> float
def visible_answer(answer, labels) -> str                # hide unsupported sentences
def build_rag_cache(index, high_risk_ids, questions, provider) -> DataFrame
```

- `build_index` keeps the 150 most recent reviews plus all tips per location, chunks
  to <= 120 words, embeds with MiniLM (lazy), writes `rag_embeddings.npy` (float16)
  and `rag_metadata.parquet` (`chunk_id, business_id, chain, review_id, date, stars,
  text`), and asserts the total on-disk size is under 50 MB.
- Retrieval filters metadata first (business_id/chain, optional stars <= 2 and date
  range), then cosine top-k over the filtered rows, with optional MMR for diversity.
- Generation uses `lrr.llm` with the NVIDIA model by default; the judge uses a
  different-family model (qwen). The answerer cites `[review_id]`; the judge labels
  each sentence supported/unsupported against its cited snippet. `groundedness_rate`
  is supported / total; `visible_answer` drops unsupported sentences for the UI.
- `build_rag_cache` runs the 6 standard questions per high-risk location and returns
  rows (`business_id, question, answer, visible_answer, groundedness_rate, citations`)
  for `artifacts/rag_cache.parquet`.

## pipeline/06_aspects_rag.py

Two phases. Aspects: build the stratified sample (per chain, top-10 and bottom-10 by
risk from `risk_scores.parquet`, up to 10 recent pre-landmark reviews each, capped by
`--sample-size`), run the agents with a JSONL checkpoint for resume, write
`artifacts/aspect_results.parquet` and the stars evaluation and risk contrast. RAG:
build the index under `data/rag_index/`, precompute the cache for high-risk
locations, and write `artifacts/rag_cache.parquet` plus the eval summary
(`artifacts/rag_eval.parquet`). CLI: `--sample-size`, `--provider`, `--landmark`,
`--checkpoint`, `--phase {aspects,rag,all}`, `--dry-run`.

## Testing strategy

Unit tests, no network/GPU: grounding accept/reject; sign and aspect-score and
predicted-stars math; JSON parse/validate/retry via the mock provider;
`chunk_text` word cap; `cosine_top_k` and `mmr` ordering on tiny vectors; judge
labels -> groundedness rate and `visible_answer`; and the index-size assertion path.
sentence-transformers is never imported in tests (embeddings are injected).
