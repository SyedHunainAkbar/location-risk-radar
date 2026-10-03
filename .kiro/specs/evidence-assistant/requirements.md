# Requirements: Evidence Assistant

## Introduction

We turn review text into human-readable evidence two ways. Aspect agents extract
grounded quotes with polarity and intensity for food, service, and ambience, and
Python computes every number. A retrieval-augmented assistant answers standard
operator questions from retrieved snippets only, cites every claim, and is checked
by a second-family judge model. All answers are precomputed and cached so the app
runs with no API key. This is pipeline stage 06.

## Requirements

### Requirement 1: Aspect agents (JSON only, grounded, Python scores)

#### Acceptance Criteria
1. There SHALL be three aspect agents (food, service, ambience) and one lead agent.
2. Each aspect agent SHALL return JSON only: a list of
   `{quote, polarity in [positive, negative, neutral], intensity in 1..3}`.
3. The lead agent SHALL return overall `polarity` and `intensity`.
4. Prompts SHALL be few-shot with one positive and one negative example, at
   temperature 0.
5. The JSON SHALL be validated against a schema with a bounded retry on failure.
6. Python SHALL ground each quote (rapidfuzz partial ratio >= 85 and >= 2 words);
   ungrounded quotes SHALL be dropped and counted.
7. Python SHALL compute the aspect score = `mean(sign * intensity / 3)` and the
   predicted stars = `3 + 2 * sign * intensity / 3`. The LLM SHALL NOT output numbers.

### Requirement 2: Aspect sampling, evaluation, and risk contrast

#### Acceptance Criteria
1. The sample SHALL be stratified: for each of the 30 chains, the 10 highest-risk and
   10 lowest-risk locations, with up to 10 recent pre-landmark reviews each.
2. The total sample size SHALL be a CLI argument to cap cost.
3. The run SHALL resume from a JSONL checkpoint so partial progress is not lost.
4. Predicted stars SHALL be evaluated against actual stars: accuracy, macro F1, MAE,
   and Pearson r, each with confidence intervals.
5. Aspect scores SHALL be aggregated by location, and a Mann-Whitney test with an
   effect size SHALL test whether high-risk locations differ from low-risk ones.

### Requirement 3: RAG index

#### Acceptance Criteria
1. For each cohort location the index SHALL hold the 150 most recent reviews plus all
   tips, chunked to at most 120 words.
2. Chunks SHALL be embedded with `all-MiniLM-L6-v2`.
3. Embeddings SHALL be stored as float16 `.npy` with parquet metadata, and the total
   index SHALL be under 50 MB (asserted).

### Requirement 4: RAG retrieval

#### Acceptance Criteria
1. Retrieval SHALL filter by `business_id` or `chain`, then take cosine top k = 8.
2. Optional MMR SHALL be available for diversity.
3. Optional filters SHALL be available: stars <= 2 and a date range.

### Requirement 5: RAG generation and judge

#### Acceptance Criteria
1. The answerer (NVIDIA `meta/llama-3.3-70b-instruct`) SHALL answer ONLY from the
   retrieved snippets, cite `[review_id]` for every claim, and say "not enough
   evidence" when the snippets do not support an answer.
2. A judge model from a different family (for example `qwen`) SHALL check each
   sentence against its cited snippet and return supported/unsupported labels.
3. Python SHALL compute a groundedness rate; the UI SHALL hide unsupported sentences.

### Requirement 6: Precomputed answer cache

#### Acceptance Criteria
1. The system SHALL precompute answers for 6 standard questions per high-risk
   location and write them to `artifacts/rag_cache.parquet`.
2. The app SHALL serve cached answers with no API key.

### Requirement 7: Evaluation

#### Acceptance Criteria
1. There SHALL be 30 hand-written question/answer checks.
2. The system SHALL report retrieval hit rate and groundedness rate.

### Requirement 8: Determinism, offline boundary, CLI

#### Acceptance Criteria
1. All LLM calls SHALL use `lrr.llm` (providers nvidia/voyager/mock); keys only from
   env or `st.secrets`.
2. All randomness SHALL use `config.SEED`; all paths SHALL come from `config.py`.
3. `pipeline/06_aspects_rag.py` SHALL accept CLI args, cap cost, and be idempotent.

### Requirement 9: Tests

#### Acceptance Criteria
1. Quote grounding SHALL be tested (accept a grounded quote, drop an ungrounded one).
2. Aspect score and predicted-stars math SHALL be tested on known inputs.
3. JSON schema validation and retry SHALL be tested via the mock provider.
4. Chunking (<= 120 words), cosine/MMR retrieval, judge groundedness rate, and the
   index-size assertion SHALL be tested, with no network and no GPU.
