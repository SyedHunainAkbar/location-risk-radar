# Tasks: Risk Committee Agents

- [ ] 1. Config: agent constants in `config.py` + `config/actions.yaml`
  - Issue enum values, confidence weights/target, briefs/eval filenames.
  - _Requirements: 2.3, 3.4, 4.2, 4.3, 5.1_

- [ ] 2. `agents/schemas.py` + `agents/factsheet.py`
  - Pydantic schemas for all three agents; Python fact-sheet builder.
  - _Requirements: 1.2, 1.3, 2.3, 3.3_

- [ ] 3. `agents/quant.py` (Agent 1)
  - Placeholder-only templates, digit guard + retry, Python rendering.
  - _Requirements: 1.1, 1.4, 1.5_

- [ ] 4. `agents/voc.py` (Agent 2)
  - Grounded issue extraction, issue scores and prevalence.
  - _Requirements: 2.1, 2.4, 2.5_

- [ ] 5. `agents/auditor.py` + `agents/confidence.py` (Agent 3)
  - Verdicts, alternatives, grade; documented confidence formula.
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 6. `agents/orchestrator.py` + `agents/playbook.py` + `__init__.py`
  - asyncio.gather(1, 2) then 3; assemble Brief; actions; persist briefs.
  - _Requirements: 4.1, 4.2, 4.3, 6.1_

- [ ] 7. `pipeline/08_eval_agents.py`
  - Fidelity, groundedness, rejection, committee vs single, kappa, latency/cost;
    TA-aspect benchmark credited.
  - _Requirements: 5.1-5.5_

- [ ] 8. Tests
  - Placeholder enforcement, grounding, confidence formula, orchestrator concurrency.
  - _Requirements: 7.1, 7.2_

- [ ] 9. Run pytest and make it pass
  - _Requirements: all_
