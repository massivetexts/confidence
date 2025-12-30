# Future ideas (scratch pad)

- Support multi-token score responses (parse multi-token numerals or allow score strings like "4.5").
- Add a "thinking" mode: generate a thought trace first, then append "and the answer is" and switch to a logprob-enabled model for the score token.
- Allow a `--thinking-level` argument to tune how much reasoning to include before scoring.
- Support batch scoring from CSV (task/item/response columns).
- Add a batch JSONL format for large-scale scoring.
- Add caching for repeated prompts to reduce API cost.
- Add schema-based YAML/JSON input validation.
- Provide prompt templates for common tasks (humor, clarity, empathy, relevance, toxicity, etc.).
- Add calibration helpers (e.g., map model scores to human scale with isotonic regression).
- Add tool to plot confidence vs. accuracy or error curves. (i.e. allowing a column with 'truth' answers)