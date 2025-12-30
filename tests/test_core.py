
import math

import pytest

from confidence.core import (
    claude_logprobs_available,
    claude_top_logprobs_unavailable_error,
    parse_first_number,
    probability_scores_from_logprobs,
    progressive_probability_scores_from_logprobs,
)


def test_parse_first_number():
    assert parse_first_number(" 30") == 30
    assert parse_first_number("score=2.5!") == 2.5
    assert parse_first_number("nope") is None


def test_probability_scores_from_logprobs():
    score_logprobs = [
        (" 30", math.log(0.5)),
        (" 40", math.log(0.2)),
        (" 20", math.log(0.1)),
    ]
    scores = probability_scores_from_logprobs(
        score_logprobs,
        score_transform=lambda x: x / 10,
    )
    assert scores["weighted"] == pytest.approx(3.125)
    assert scores["weighted_confidence"] == pytest.approx(0.8)
    assert scores["top"] == pytest.approx(3.0)
    assert scores["top_confidence"] == pytest.approx(0.5)
    assert scores["n"] == 3


def test_probability_scores_empty_candidates():
    with pytest.raises(ValueError):
        probability_scores_from_logprobs([("x", 0.0)])


def test_progressive_probability_scores():
    score_logprobs = [
        (" 30", math.log(0.5)),
        (" 40", math.log(0.2)),
        (" 20", math.log(0.1)),
    ]
    scores = progressive_probability_scores_from_logprobs(score_logprobs, min_contributors=2)
    assert len(scores) == 2
    assert scores[0]["n"] == 2
    assert scores[1]["n"] == 3


def test_claude_logprobs_unavailable():
    assert claude_logprobs_available() is False
    err = claude_top_logprobs_unavailable_error()
    assert "does not expose" in str(err)
