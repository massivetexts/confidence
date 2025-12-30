# confidence

Drop-in utilities for computing confidence-aware scores from token log probabilities.

This repo provides:
- Probability-weighted scores from top-k logprobs for the score token
- Confidence scores (probability mass over valid score-like candidates)
- Provider helpers for OpenAI, Gemini, and Hugging Face Transformers
- A CLI and `uv run`-friendly drop-in script with mocks

## Install

```bash
pip install -e .
# or
uv pip install -e .
```

Optional extras:
- `.[openai]` for OpenAI
- `.[gemini]` for Gemini
- `.[hf]` for Hugging Face
- `.[yaml]` for YAML input

## Quick start

### CLI (installed)

```bash
confidence-dropin openai --model gpt-4o-mini --task "Rate the humor of the response from 10-50" --response "A punny joke" --divide-by-10
confidence-dropin gemini --model gemini-2.0-flash --task "Rate the clarity of the response from 10-50" --response "It is clear." --divide-by-10
confidence-dropin hf --model gpt2 --task "Rate the relevance of the response from 10-50" --item "Recycling" --response "Turn it into a planter" --divide-by-10
```

### Script (no install; uses uv)

```bash
uv run examples/confidence_dropin.py openai --mock --divide-by-10
uv run examples/confidence_dropin.py gemini --mock --divide-by-10
uv run --with transformers --with torch examples/confidence_dropin.py hf --model gpt2 --divide-by-10
```

### Mock mode (no API keys required)

```bash
confidence-dropin openai --mock --divide-by-10
confidence-dropin gemini --mock --divide-by-10
```

Expected output (mock mode with `--divide-by-10`):

```
Raw completion: ' 30'
Scores: {'weighted': 3.125, 'weighted_confidence': 0.8, 'top': 3.0, 'top_confidence': 0.5, 'n': 3}
```

## Task and item prompts

Use `--task` for custom instructions, plus `--item` when you have a stimulus:

```bash
confidence-dropin openai --task "Rate the humor of the response from 10-50" --response "A punny joke" --divide-by-10
confidence-dropin openai --task "Rate the empathy of the response from 10-50" --item "Friend is sad" --response "I'm here for you." --divide-by-10
confidence-dropin openai --task "Rate the correctness of the response from 10-50" --item "2+2" --response "4" --divide-by-10
confidence-dropin openai --task "Rate the concision of the response from 10-50" --response "Short and clear." --divide-by-10
confidence-dropin openai --task "Rate the relevance of the response from 10-50" --item "Climate change" --response "Recycle and plant trees." --divide-by-10
confidence-dropin openai --task "Rate the toxicity of the response from 10-50" --response "I hope you have a great day." --divide-by-10
```

If you omit `--task`, the CLI falls back to the default originality prompt with `--prompt`.

## YAML input

You can load `task`, `item`, and `response` from a YAML file:

```yaml
# input.yaml
task: Rate the humor of the response from 10-50
item: A one-liner joke
response: Why did the chicken cross the road?
min_score: 10
max_score: 50
divide_by_10: true
```

```bash
confidence-dropin openai --input-yaml input.yaml
```

## .env support

Create a `.env` file and load it explicitly:

```
OPENAI_API_KEY=your-key
GEMINI_API_KEY=your-key
```

```bash
confidence-dropin openai --env-file .env --task "Rate the humor from 10-50" --response "A pun" --divide-by-10
confidence-dropin gemini --env-file .env --task "Rate the clarity from 10-50" --response "Clear." --divide-by-10
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

## Testing

```bash
pytest
```

To run live API tests (optional):

```bash
export CONFIDENCE_RUN_API_TESTS=1
export OPENAI_API_KEY=your-key
export GEMINI_API_KEY=your-key
pytest -m api
```

## Notes and limitations

- OpenAI and Gemini expose token log probabilities for the score token; you can compute confidence directly.
- Anthropic Claude does not expose token-level log probabilities via the public API at time of writing, so the logprob-based confidence used here cannot be computed directly for Claude-hosted models.

## Future ideas

See `FUTURE_IDEAS.md` for a scratch-pad list of upcoming features.
