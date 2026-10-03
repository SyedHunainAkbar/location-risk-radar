# Risk Committee agents

A three-agent committee writes each Location Risk Brief, orchestrated by
deterministic Python. The agents produce language and labels only; Python grounds
every quote, renders every number, and derives the confidence grade. Every call goes
through the LLM gateway (see [gateway.md](gateway.md)).

## Agent 1: Quant Analyst (role `quant_analyst`)

Input is a Python-built fact sheet (risk score, percentile within chain and cluster,
signed SHAP drivers, velocity ratio, sentiment slope, topic-share changes, survival
probabilities). Output is a list of driver explanation **templates** that reference
values only through placeholders such as `{velocity_ratio}`. Python rejects any
template containing a literal digit outside a placeholder and retries, then renders
the final text. This is the numeric-fidelity guarantee: no number in a brief is
written by a model.

## Agent 2: Voice of Customer (role `voice_of_customer`)

Input is the top-k MMR-diversified, negative-weighted review snippets. Output is a
list of `{quote, review_id, issue, polarity, intensity}`. Python grounds every quote
(rapidfuzz partial ratio >= 85, at least two words, matching review id), drops
ungrounded quotes and counts them, and computes per-issue scores
`mean(sign * intensity / 3)` and prevalence.

## Agent 3: Risk Auditor (role `risk_auditor`, different model family)

Input is agents 1 and 2 outputs plus context flags (review volume, pandemic overlap,
chain-wide trend, cluster label noise). Output is a per-claim verdict
(supported / unsupported / insufficient), alternative explanations, and an A/B/C
confidence grade. The orchestrator removes unsupported claims from the brief.

## Confidence formula

```
confidence = groundedness_rate * min(1, grounded_evidence_count / 5) * grade_weight[grade]
grade_weight = {A: 1.0, B: 0.7, C: 0.4}
```

Clamped to [0, 1] and labeled High >= 0.66, Medium >= 0.33, else Low.

## Orchestration and evaluation

Agents 1 and 2 run concurrently (`asyncio.gather`), then the auditor. The brief
carries tier, rendered drivers, grounded evidence with citations, confidence, and
recommended actions from `config/actions.yaml`. Evaluation (`pipeline/08`) reports an
independent numeric-fidelity re-scan (every rendered number must trace to the fact
sheet), groundedness, auditor rejection rate, a committee-vs-single-agent comparison,
auditor-vs-human kappa, and latency and cost per brief from the gateway ledger.
