# confidence

Drop-in utilities for computing confidence-aware scores from token log probabilities.

This is a demonstrative repo meant to be easy to read and reuse. If you want to
replicate the approach in your own codebase, start with `confidence/core.py`
(especially `openai_chat_completion_top_logprobs`, `gemini_generate_content_top_logprobs`,
`hf_next_token_top_logprobs`, and the probability utilities).

This repo provides:
- Probability-weighted scores from top-k logprobs for the score token
- Confidence scores (probability mass over valid score-like candidates)
- Provider helpers for OpenAI, Gemini, and Hugging Face Transformers
- A CLI and `uv run`-friendly drop-in script with mocks

## Quick start (uv run)

See `uv run` docs: https://docs.astral.sh/uv/reference/cli/#uv-run

```bash
# Run directly from GitHub (no clone required):
uv run --with "confidence[openai] @ git+https://github.com/massivetexts/confidence" confidence-dropin openai --task "Rate the humor of the response from 10-50" --response "A punny joke" --divide-by-10

# Or run locally after cloning:
uv run examples/confidence_dropin.py openai --task "Rate the humor of the response from 10-50" --response "A punny joke" --divide-by-10
uv run examples/confidence_dropin.py gemini --task "Rate the clarity of the response from 10-50" --response "It is clear." --divide-by-10 --score-tokens auto
uv run --with transformers --with torch examples/confidence_dropin.py hf --model gpt2 --task "Rate the relevance of the response from 10-50" --item "Recycling" --response "Turn it into a planter" --divide-by-10
```

Mock mode (no API keys required):

```bash
uv run examples/confidence_dropin.py openai --mock --divide-by-10
uv run examples/confidence_dropin.py gemini --mock --divide-by-10
```

Expected output (mock mode with `--divide-by-10`, JSONL):

```
{"provider":"openai","model":"gpt-4o-mini","inputs":{...},"output":{"completion":" 30","scores":{"weighted":3.125,"weighted_confidence":0.8,"top":3.0,"top_confidence":0.5,"n":3}}}
```

## Multi-token scoring (up to 3 tokens)

Use `--score-tokens` when scores are more than one token. The implementation combines
per-token logprobs into joint probabilities and scores all permutations, following the
method described in the paper.

```bash
uv run examples/confidence_dropin.py gemini --task "Rate the clarity from 10-50" --response "Clear." --divide-by-10 --score-tokens auto
uv run examples/confidence_dropin.py openai --task "Rate from 1.0-5.0" --response "Good" --min-score 1 --max-score 5 --score-tokens 3
```

## Provider docs

- OpenAI Python SDK: https://github.com/openai/openai-python
- OpenAI API reference (chat completions): https://platform.openai.com/docs/api-reference/chat
- Gemini API docs: https://ai.google.dev/gemini-api/docs
- Google Gen AI Python SDK: https://github.com/googleapis/python-genai
- Hugging Face Transformers: https://huggingface.co/docs/transformers/

## Task and item prompts

Use `--task` for custom instructions, plus `--item` when you have a stimulus:

```bash
uv run examples/confidence_dropin.py openai --task "Rate the humor of the response from 10-50" --response "A punny joke" --divide-by-10
uv run examples/confidence_dropin.py openai --task "Rate the empathy of the response from 10-50" --item "Friend is sad" --response "I'm here for you." --divide-by-10
uv run examples/confidence_dropin.py openai --task "Rate the correctness of the response from 10-50" --item "2+2" --response "4" --divide-by-10
uv run examples/confidence_dropin.py openai --task "Rate the concision of the response from 10-50" --response "Short and clear." --divide-by-10
uv run examples/confidence_dropin.py openai --task "Rate the relevance of the response from 10-50" --item "Climate change" --response "Recycle and plant trees." --divide-by-10
uv run examples/confidence_dropin.py openai --task "Rate the toxicity of the response from 10-50" --response "I hope you have a great day." --divide-by-10
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
score_tokens: auto
```

```bash
uv run examples/confidence_dropin.py openai --input-yaml input.yaml
```

## .env support

The CLI uses `python-dotenv` and will automatically load `.env.local` or `.env` if present.
You can also point to a specific file.

```
OPENAI_API_KEY=your-key
GEMINI_API_KEY=your-key
```

```bash
uv run examples/confidence_dropin.py openai --task "Rate the humor from 10-50" --response "A pun" --divide-by-10
uv run examples/confidence_dropin.py gemini --task "Rate the clarity from 10-50" --response "Clear." --divide-by-10 --score-tokens auto
```

```bash
uv run examples/confidence_dropin.py openai --env-file .env.local --task "Rate the humor from 10-50" --response "A pun" --divide-by-10
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
uv run --with pytest --with pyyaml --with python-dotenv --no-project -- python -m pytest
```

To run live API tests (optional):

```bash
export CONFIDENCE_RUN_API_TESTS=1
export OPENAI_API_KEY=your-key
export GEMINI_API_KEY=your-key
uv run --with pytest --with pyyaml --with python-dotenv --no-project -- python -m pytest -m api
```

## Install (optional)

```bash
uv pip install -e .
# or
pip install -e .
```

Optional extras:
- `.[openai]` for OpenAI
- `.[gemini]` for Gemini
- `.[hf]` for Hugging Face
- `.[yaml]` for YAML input
- `.[dotenv]` for `.env` loading

## Notes and limitations

- OpenAI and Gemini expose token log probabilities for the score token; you can compute confidence directly.
- Gemini tokenizes numbers as separate characters, so scaling to 10-50 does not guarantee a single token. Use `--score-tokens` to combine multi-token scores.
- Anthropic Claude does not expose token-level log probabilities via the public API at time of writing, so the logprob-based confidence used here cannot be computed directly for Claude-hosted models.

## Future ideas

See `FUTURE_IDEAS.md` for a scratch-pad list of upcoming features.
