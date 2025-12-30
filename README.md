# confidence

Drop-in utilities for computing confidence-aware scores from token log probabilities.

This repo provides:
- Probability-weighted scores from top-k logprobs for the score token
- Confidence scores (probability mass over valid score-like candidates)
- Small provider helpers for OpenAI, Gemini, and Hugging Face Transformers
- A CLI and `uv run`-friendly drop-in script with mocks

## Quick start

### CLI (installed)

```bash
confidence-dropin openai --model gpt-4o-mini
confidence-dropin gemini --model gemini-2.0-flash
confidence-dropin hf --model gpt2
```

### Script (no install; uses uv)

```bash
uv run examples/confidence_dropin.py openai --mock
uv run examples/confidence_dropin.py gemini --mock
uv run --with transformers --with torch examples/confidence_dropin.py hf --model gpt2
```

## Library usage

```python
from confidence import probability_scores_from_logprobs

score_logprobs = [
    (" 30", -0.51),
    (" 40", -1.61),
    (" 20", -2.30),
]

scores = probability_scores_from_logprobs(score_logprobs)
print(scores)
```

## Notes and limitations

- OpenAI and Gemini expose token log probabilities for the score token; you can compute confidence directly.
- Anthropic Claude does not expose token-level log probabilities via the public API at time of writing, so the logprob-based confidence used here cannot be computed directly for Claude-hosted models.
