# Quality Standards

## Analytical Standard

Elite quant and data science rigor.

- No leakage: features use only data strictly before the landmark date.
- Report uncertainty with bootstrap 95% confidence intervals.
- Always include baselines and ablations.
- Use grouped validation by chain wherever relevant so the same chain does not sit in both train and test.

## Writing Standard (markdown and UI text)

- Concise and precise.
- First person plural ("we").
- Graduate student voice.
- No em dashes.
- No filler.
- Every chart has a one-sentence takeaway.

## Code Standard

- Type hints on functions.
- Docstrings on modules and public functions.
- Small, pure functions.
- pytest for every scoring formula and every data contract.
- Assertions on shapes and counts. The cohort must equal 30 chains (15 Fast Food + 15 Non-Fast Food).
