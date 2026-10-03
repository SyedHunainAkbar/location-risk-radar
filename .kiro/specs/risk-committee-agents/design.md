# Design: Risk Committee Agents

## Overview

A three-agent committee produces a Location Risk Brief. Deterministic Python
orchestrates the agents, grounds every quote, renders every number, and derives the
confidence grade. The agents emit language and labels only. All model calls go
through `lrr.gateway` with the roles `quant_analyst`, `voice_of_customer`, and
`risk_auditor` (the auditor is a different family by routing).

Package `src/lrr/agents/`:

```
agents/
  __init__.py       # public: build_brief, abuild_brief, Brief
  schemas.py        # pydantic schemas for each agent's output
  factsheet.py      # Python builds the Quant fact sheet from artifacts
  quant.py          # Agent 1: placeholder-only driver templates + digit guard
  voc.py            # Agent 2: grounded issues + issue scores/prevalence
  auditor.py        # Agent 3: verdicts, alternatives, confidence grade
  confidence.py     # documented final-confidence formula
  playbook.py       # load config/actions.yaml, map issue -> actions
  orchestrator.py   # asyncio.gather(1, 2) -> 3 -> assemble Brief; persist
```

## Schemas (schemas.py, pydantic)

```python
class Driver(BaseModel): driver_key: str; direction: Literal["up","down"]; explanation_template: str
class QuantOutput(BaseModel): drivers: list[Driver]
class VoCItem(BaseModel): quote: str; review_id: str; issue: Issue; polarity: Polarity; intensity: conint(ge=1,le=3)
class VoCOutput(BaseModel): items: list[VoCItem]
class ClaimVerdict(BaseModel): claim_id: str; verdict: Literal["supported","unsupported","insufficient"]; reason: str
class AuditOutput(BaseModel): claims: list[ClaimVerdict]; alternatives: list[Alt]; confidence_grade: Literal["A","B","C"]
```

`Issue` is the nine-value enum from the requirements. These schemas are passed to the
gateway's JSON mode so a repair retry fires on malformed output.

## Agent 1: Quant Analyst (quant.py)

- `build_prompt(factsheet)`: instructs the model to return driver templates whose
  explanations reference values only via `{placeholder}` tokens drawn from the fact
  sheet keys, never literal numbers.
- `enforce_placeholders(template)`: regex rejects any digit that is not inside a
  `{...}` placeholder. `PLACEHOLDER_RE = r"\{[a-z0-9_]+\}"`; we strip placeholders,
  then if `\d` remains the output is invalid.
- `run_quant(factsheet, retries=2)`: calls the gateway with `QuantOutput` schema; on a
  digit violation, re-prompts with the offending template; raises after retries.
- `render(template, factsheet)`: Python substitutes placeholder values, formatting
  floats consistently. This is where every number enters the text.

## Agent 2: Voice of Customer (voc.py)

- Input is the MMR-diversified, negative-weighted top-k snippets (reused from
  `lrr.rag.retrieve`). The prompt asks for verbatim quotes with an issue label,
  polarity, and intensity.
- `ground_items(items, snippets)`: keeps an item only if its quote has >= 2 words, its
  rapidfuzz partial ratio against the cited snippet text is >= 85, and the cited
  `review_id` matches a snippet id. Dropped items are counted.
- `issue_scores(items)`: per issue, `mean(sign * intensity / 3)`; `issue_prevalence`:
  share of grounded quotes per issue. Python computes both.

## Agent 3: Risk Auditor (auditor.py)

- Input is agents 1 and 2 outputs plus context flags. Claims are built by Python from
  the rendered drivers and the grounded VoC issues, each with a stable `claim_id`.
- The auditor returns a verdict per claim, alternative explanations with verdicts, and
  a confidence grade A/B/C. Unsupported claims are removed by the orchestrator.

## Confidence formula (confidence.py)

Documented and testable:

```
grade_weight = {A: 1.0, B: 0.7, C: 0.4}
evidence_factor = min(1.0, grounded_evidence_count / EVIDENCE_TARGET)   # EVIDENCE_TARGET = 5
confidence = round(groundedness_rate * evidence_factor * grade_weight[grade], 3)
```

`groundedness_rate` is supported_claims / total_claims from the auditor;
`grounded_evidence_count` is the number of grounded VoC quotes. Confidence is in
[0, 1]; the brief also maps it to a label (High >= 0.66, Medium >= 0.33, else Low).

## Orchestrator (orchestrator.py)

```python
async def abuild_brief(location_id, ...) -> Brief
def build_brief(location_id, ...) -> Brief            # sync wrapper
def persist_briefs(briefs, path) -> Path              # artifacts/briefs.parquet
```

Steps: build the fact sheet (Python) and retrieve VoC snippets (Python); run
`asyncio.gather(run_quant, run_voc)`; build claims and run the auditor; drop
unsupported claims; compute confidence; map grounded issues to actions via the
playbook; assemble the `Brief` (tier, rendered drivers, grounded evidence with
citations, confidence, actions). `persist_briefs` writes one row per brief with the
rendered text and citations for the app.

## Playbook (playbook.py + config/actions.yaml)

`config/actions.yaml` maps each issue to an ordered list of recommended actions.
`recommend(issues)` returns the actions for the issues present in the brief, de-duped
and ordered by issue prevalence.

## Evaluation (pipeline/08_eval_agents.py)

- Numeric fidelity: fraction of rendered driver sentences with no stray digit (target
  1.0, enforced by Agent 1's guard).
- Groundedness: grounded quotes / proposed quotes.
- Auditor rejection rate: unsupported claims / total claims.
- Committee vs single-agent: a one-shot "write the whole brief" baseline vs the
  committee on 30 held-out locations, scored by a blinded judge (gateway `risk_auditor`
  role on a fixed rubric), plus manual labels on 10. Cohen's kappa (auditor vs manual).
- Latency and cost per brief are read from the gateway ledger.
- The TA-style food/service/ambience aspect run (existing `lrr.aspects`) is reported
  as a separate benchmark against stars, clearly credited as adapted from the TA's
  Yelp Aspect Agents reference.

Writes `artifacts/agent_eval.parquet`.

## Testing strategy

Offline, mock provider and injected fakes:
- `enforce_placeholders` accepts templates with only placeholders and rejects literal
  digits; `run_quant` retries then raises on a persistent digit.
- `ground_items` keeps a matching quote and drops a hallucinated one and an id
  mismatch.
- `confidence` matches the formula on known inputs and clamps to [0, 1].
- The orchestrator runs agents 1 and 2 concurrently (a fake records overlap) and then
  the auditor, assembling a brief end to end on the mock provider.
