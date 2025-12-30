#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "openai>=1.0.0",
#   "google-genai>=0.7.0",
# ]
# ///
"""Drop-in confidence scoring examples (OpenAI, Gemini, Hugging Face).

Examples (real APIs):
  uv run examples/confidence_dropin.py openai --model gpt-4o-mini
  uv run examples/confidence_dropin.py gemini --model gemini-2.0-flash
  uv run --with transformers --with torch examples/confidence_dropin.py hf --model gpt2

Examples (no API keys required; deterministic mock distribution):
  uv run examples/confidence_dropin.py openai --mock
  uv run examples/confidence_dropin.py gemini --mock

Expected output (mock mode; exact values):
  Raw completion: ' 30'
  Scores: {'weighted': 3.125, 'weighted_confidence': 0.8, 'top': 3.0, 'top_confidence': 0.5, 'n': 3}

Note: Anthropic Claude does not expose token-level log probabilities via the public API
at time of writing, so the logprob-based confidence used here cannot be computed directly
for Claude-hosted models.
"""


import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from confidence.cli import main

if __name__ == "__main__":
    main()

# Expected output (OpenAI/Gemini, --mock):
#   Raw completion: ' 30'
#   Scores: {'weighted': 3.125, 'weighted_confidence': 0.8, 'top': 3.0, 'top_confidence': 0.5, 'n': 3}

# Example output (HF, gpt2):
#   Raw completion: ' 1'
#   Scores: {'weighted': 2.105750280122672, 'weighted_confidence': 0.10518133743464297, 'top': 1.0, 'top_confidence': 0.04655119668576797, 'n': 8}

