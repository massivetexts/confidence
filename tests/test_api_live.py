import os

import pytest

from confidence.core import (
    gemini_generate_content_top_logprobs,
    openai_chat_completion_top_logprobs,
)


def _require_api_env(*keys: str) -> None:
    if not os.getenv("CONFIDENCE_RUN_API_TESTS"):
        pytest.skip("Set CONFIDENCE_RUN_API_TESTS=1 to enable live API tests.")
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        pytest.skip("Missing API keys: " + ", ".join(missing))


@pytest.mark.api
def test_openai_live():
    pytest.importorskip("openai")
    _require_api_env("OPENAI_API_KEY")

    messages = [
        {"role": "system", "content": "You are a scoring assistant."},
        {
            "role": "user",
            "content": (
                "Rate the humor of the response from 10 to 50. "
                "Return a single integer token.\n"
                "RESPONSE: A silly joke.\n"
                "SCORE:"
            ),
        },
    ]

    result = openai_chat_completion_top_logprobs(
        model="gpt-4o-mini",
        messages=messages,
        top_logprobs=20,
        max_tokens=1,
        stop="\n",
    )
    assert result.top_logprobs
    assert result.scores["n"] >= 1


@pytest.mark.api
def test_gemini_live():
    try:
        from google import genai  # noqa: F401
    except Exception:
        pytest.skip("google-genai not installed")

    _require_api_env("GEMINI_API_KEY")

    result = gemini_generate_content_top_logprobs(
        model="gemini-2.0-flash",
        contents=(
            "Rate the clarity of the response from 10 to 50. "
            "Return a single integer token.\n"
            "RESPONSE: This answer is clear.\n"
            "SCORE:"
        ),
        top_logprobs=20,
        max_output_tokens=1,
        stop_sequences=("\n",),
    )
    assert result.top_logprobs
    assert result.scores["n"] >= 1
