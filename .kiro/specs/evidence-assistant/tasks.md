# Tasks: Evidence Assistant

- [ ] 1. Add aspect/RAG constants to `config.py`
  - Aspects, polarity set, intensity range, standard questions, chunk size,
    retrieval k, MMR lambda, index size cap, sample sizes, artifact filenames.
  - _Requirements: 1.1, 1.2, 3.1, 3.3, 4.1, 6.1_

- [ ] 2. Implement `src/lrr/aspects.py`
  - Few-shot prompts, JSON schema validation + retry.
  - `ground_quote`, `polarity_sign`, `aspect_score`, `predicted_stars`.
  - `analyze_review`, `evaluate_predicted_stars`, `risk_contrast`.
  - _Requirements: 1.1-1.7, 2.4, 2.5_

- [ ] 3. Implement `src/lrr/rag.py`
  - `chunk_text`, `build_index` (float16 npy + parquet, < 50 MB assert).
  - `cosine_top_k`, `mmr`, `retrieve` (filters).
  - `generate_answer` (cite + abstain), `judge_sentences`, `groundedness_rate`,
    `visible_answer`, `build_rag_cache`.
  - _Requirements: 3.1-3.3, 4.1-4.3, 5.1-5.3, 6.1, 6.2_

- [ ] 4. Implement `pipeline/06_aspects_rag.py`
  - Stratified aspect sample + JSONL checkpoint resume.
  - Build index; precompute RAG cache; evaluations.
  - Idempotent CLI with cost cap.
  - _Requirements: 2.1, 2.2, 2.3, 7.1, 7.2, 8.1, 8.2, 8.3_

- [ ] 5. Tests
  - Grounding, scoring math, JSON validation + retry (mock), chunk cap,
    cosine/MMR, judge groundedness, index-size assert.
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 6. Run pytest and make it pass
  - _Requirements: all_
