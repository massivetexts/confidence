
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "python-dotenv",
#   "openai>=1.0.0",
#   "google-genai>=0.7.0",
# ]
# ///
"""Drop-in confidence scoring examples (OpenAI, Gemini, Hugging Face).

Examples (real APIs):
  uv run examples/confidence_dropin.py openai --model gpt-4o-mini --task "Rate the humor of this message from 10-50" --response "A punny joke" --divide-by-10
  uv run examples/confidence_dropin.py gemini --model gemini-2.0-flash --task "Rate the clarity of the response from 10-50" --response "It is clear." --divide-by-10 --score-tokens auto
  uv run --with transformers --with torch examples/confidence_dropin.py hf --model gpt2 --task "Rate the relevance of the response from 10-50" --item "Recycling" --response "Turn it into a planter" --divide-by-10

Examples (no API keys required; deterministic mock distribution):
  uv run examples/confidence_dropin.py openai --mock --divide-by-10
  uv run examples/confidence_dropin.py gemini --mock --divide-by-10

Expected output (mock mode with --divide-by-10; exact values, JSONL):
  {"provider":"openai","model":"gpt-4o-mini","inputs":{...},"output":{"completion":" 30","scores":{"weighted":3.125,"weighted_confidence":0.8,"top":3.0,"top_confidence":0.5,"n":3}}}

Note: Anthropic Claude does not expose token-level log probabilities via the public API
at time of writing, so the logprob-based confidence used here cannot be computed directly
for Claude-hosted models.

Gemini note: Gemini tokenizes numbers as separate characters. Use --score-tokens for
multi-token numbers instead of relying on the divide-by-10 single-token trick.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from confidence.cli import main

if __name__ == "__main__":
    main()

# Expected output (OpenAI/Gemini, --mock --divide-by-10):
#   {"provider":"openai","model":"gpt-4o-mini","inputs":{...},"output":{"completion":" 30","scores":{"weighted":3.125,"weighted_confidence":0.8,"top":3.0,"top_confidence":0.5,"n":3}}}

# Example output (HF, gpt2 --divide-by-10):
#   {"provider":"hf","model":"gpt2","inputs":{...},"output":{"completion":" 1","scores":{"weighted":0.2105750280122672,"weighted_confidence":0.10518133743464297,"top":0.1,"top_confidence":0.04655119668576797,"n":8}}}
