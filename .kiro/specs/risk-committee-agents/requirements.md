# Requirements: Risk Committee Agents

## Introduction

We assemble a Location Risk Brief with a three-agent committee orchestrated by
deterministic Python. Every LLM call goes through the gateway. The agents produce
language and labels only; Python computes every number, grounds every quote, and
renders every value. A different-family auditor checks the first two agents, and
Python derives a documented confidence grade. Briefs are precomputed for High and
Elevated cohort locations so the app works with no API key. This is pipeline stage 08.

## Requirements

### Requirement 1: Agent 1 Quant Analyst (placeholder-only templates)

#### Acceptance Criteria
1. The agent SHALL use the gateway role `quant_analyst`.
2. Input SHALL be a Python-built fact-sheet JSON for one location (risk score,
   percentile within chain and cluster, top SHAP drivers with signed contributions,
   review velocity ratio, sentiment slope, topic-share changes vs chain peers,
   survival probabilities).
3. Output schema SHALL be a list of
   `{driver_key, direction, explanation_template}`.
4. Explanation templates SHALL reference values ONLY via placeholders such as
   `{velocity_ratio}`; Python renders them from the fact sheet.
5. Any output containing a literal digit outside a placeholder SHALL be rejected and
   retried (bounded). This is the numeric-fidelity guarantee.

### Requirement 2: Agent 2 Voice of Customer (grounded issues)

#### Acceptance Criteria
1. The agent SHALL use the gateway role `voice_of_customer`.
2. Input SHALL be top-k RAG snippets for the location (pre-landmark reviews and tips,
   MMR diversified, negative weighted).
3. Output schema SHALL be a list of `{quote, review_id, issue, polarity, intensity}`
   where `issue` is one of: `service_speed, staff_attitude, order_accuracy,
   food_quality, cleanliness, value, management_response, drive_thru_takeout, other`
   and `intensity` is 1-3.
4. Python SHALL ground every quote: rapidfuzz partial ratio >= 85, at least 2 words,
   and the cited `review_id` must match the snippet. Ungrounded quotes SHALL be
   dropped and counted.
5. Python SHALL compute per-issue scores = `mean(sign * intensity / 3)` and issue
   prevalence (share of grounded quotes per issue).

### Requirement 3: Agent 3 Risk Auditor (different family)

#### Acceptance Criteria
1. The agent SHALL use the gateway role `risk_auditor` and SHALL be a different model
   family from agents 1 and 2.
2. Input SHALL be agents 1 and 2 outputs with citations plus context flags (review
   volume, pandemic overlap, chain-wide trend, cluster label noise).
3. Output schema SHALL include per-claim `{claim_id, verdict, reason}` where verdict
   is `supported/unsupported/insufficient`, a list of alternative explanations
   considered with verdicts, and a `confidence_grade` of A/B/C.
4. Python SHALL compute a final confidence from the groundedness rate, the evidence
   count, and the auditor grade with a documented formula.

### Requirement 4: Orchestrator and the Location Risk Brief

#### Acceptance Criteria
1. The orchestrator SHALL run agents 1 and 2 concurrently with `asyncio.gather`, then
   agent 3.
2. The brief SHALL assemble: tier, rendered drivers, grounded evidence with
   citations, unsupported claims removed, confidence, and recommended actions from a
   static playbook `config/actions.yaml` keyed by issue.
3. Briefs SHALL be persisted to `artifacts/briefs.parquet` for every High and
   Elevated risk cohort location so the app works without keys.

### Requirement 5: Evaluation

#### Acceptance Criteria
1. `pipeline/08_eval_agents.py` SHALL write `artifacts/agent_eval.parquet` and feed a
   notebook section.
2. It SHALL report numeric fidelity rate (target 100%), groundedness rate, and
   auditor rejection rate.
3. It SHALL compare a single-agent baseline (one LLM writes the whole brief) against
   the committee on 30 held-out locations, scored against a fixed rubric by a blinded
   judge model, plus manual labels on 10.
4. It SHALL report Cohen's kappa (auditor vs manual) and latency and cost per brief
   from the ledger.
5. The TA-style food/service/ambience aspect run SHALL be kept as a separate
   benchmark against stars, clearly credited as adapted from the TA reference.

### Requirement 6: Determinism, safety, CLI

#### Acceptance Criteria
1. Orchestration SHALL be deterministic Python; only the per-agent calls use the LLM.
2. All LLM calls SHALL go through the gateway (keys from env or st.secrets only).
3. All randomness SHALL use `config.SEED`; all paths SHALL come from `config.py`.

### Requirement 7: Tests

#### Acceptance Criteria
1. Tests SHALL cover placeholder enforcement (digit rejection), quote grounding, the
   confidence formula, and orchestrator concurrency with the mock provider.
2. Tests SHALL run with no network and no API keys.
