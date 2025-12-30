"""
Confidence-aware scoring utilities.

Compute confidence scores from top-k token log probabilities for the score token.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

from .types import LogProbPair, ProbScores

ScoreParser = Callable[[str], Optional[float]]
ScoreTransform = Callable[[float], float]

_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_first_number(text: str) -> Optional[float]:
    """Parse the first numeric value from a string (or return None)."""
    match = _NUMBER_RE.search(text.strip())
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _sorted_logprobs(score_logprobs: Sequence[LogProbPair]) -> list[LogProbPair]:
    # Many APIs already return top-k sorted, but we shouldn't rely on it.
    return sorted(score_logprobs, key=lambda x: x[1], reverse=True)


def probability_scores_from_logprobs(
    score_logprobs: Sequence[LogProbPair],
    *,
    parse_score: ScoreParser = parse_first_number,
    score_transform: ScoreTransform = lambda x: x,
) -> ProbScores:
    """Compute top/weighted scores and confidence from a top-k token distribution."""
    if len(score_logprobs) == 0:
        raise ValueError("score_logprobs must be non-empty")

    token_probs: list[tuple[float, float]] = []
    for token, logprob in _sorted_logprobs(score_logprobs):
        parsed = parse_score(token)
        if parsed is None:
            continue
        prob = math.exp(float(logprob))
        token_probs.append((score_transform(parsed), prob))

    if len(token_probs) == 0:
        raise ValueError("No score-like candidates found in score_logprobs")

    top_choice, top_confidence = token_probs[0]
    total_mass = math.fsum(prob for _, prob in token_probs)
    weighted_choice = math.fsum(score * prob for score, prob in token_probs) / total_mass

    return {
        "weighted": weighted_choice,
        "weighted_confidence": total_mass,
        "top": top_choice,
        "top_confidence": top_confidence,
        "n": len(token_probs),
    }


def progressive_probability_scores_from_logprobs(
    score_logprobs: Sequence[LogProbPair],
    *,
    parse_score: ScoreParser = parse_first_number,
    score_transform: ScoreTransform = lambda x: x,
    min_contributors: int = 2,
) -> list[ProbScores]:
    """Compute probability-weighted scores for k=min_contributors..K candidates."""
    if min_contributors < 1:
        raise ValueError("min_contributors must be >= 1")

    sorted_pairs = _sorted_logprobs(score_logprobs)
    if len(sorted_pairs) < min_contributors:
        raise ValueError(
            f"Need at least {min_contributors} candidates, got {len(sorted_pairs)}"
        )

    out: list[ProbScores] = []
    for k in range(min_contributors, len(sorted_pairs) + 1):
        out.append(
            probability_scores_from_logprobs(
                sorted_pairs[:k],
                parse_score=parse_score,
                score_transform=score_transform,
            )
        )
    return out


@dataclass(frozen=True)
class ConfidenceScoringResult:
    completion: str
    top_logprobs: list[LogProbPair]
    scores: ProbScores


def openai_chat_completion_top_logprobs(
    *,
    messages: list[dict],
    model: str,
    top_logprobs: int = 10,
    temperature: float = 0,
    max_tokens: int = 1,
    stop: Optional[str] = "\n",
    api_key: Optional[str] = None,
    parse_score: ScoreParser = parse_first_number,
    score_transform: ScoreTransform = lambda x: x,
) -> ConfidenceScoringResult:
    """Call OpenAI Chat Completions and return top-k logprobs for the 1st output token."""
    try:
        import openai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: openai") from e

    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        n=1,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    choice = resp.choices[0]
    if getattr(choice, "logprobs", None) is None or choice.logprobs.content is None:
        raise ValueError(
            "No logprobs returned (model may not support logprobs or logprobs were disabled)"
        )

    first_token = choice.logprobs.content[0]
    if first_token.top_logprobs is None:
        raise ValueError("No top_logprobs returned for first token")

    pairs = [(x.token, float(x.logprob)) for x in first_token.top_logprobs]
    completion = choice.message.content or ""
    scores = probability_scores_from_logprobs(
        pairs, parse_score=parse_score, score_transform=score_transform
    )
    return ConfidenceScoringResult(completion=completion, top_logprobs=pairs, scores=scores)


def gemini_generate_content_top_logprobs(
    *,
    contents: Union[str, list],
    model: str,
    top_logprobs: int = 10,
    temperature: float = 0,
    max_output_tokens: int = 1,
    stop_sequences: Optional[Sequence[str]] = ("\n",),
    api_key: Optional[str] = None,
    parse_score: ScoreParser = parse_first_number,
    score_transform: ScoreTransform = lambda x: x,
) -> ConfidenceScoringResult:
    """Call Gemini (google-genai) and return top-k logprobs for the 1st output token."""
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: google-genai (pip install google-genai)") from e

    client = genai.Client(
        api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stop_sequences=list(stop_sequences) if stop_sequences is not None else None,
            response_logprobs=True,
            logprobs=top_logprobs,
            candidate_count=1,
        ),
    )

    if not resp.candidates:
        raise ValueError("No candidates returned from Gemini")

    cand = resp.candidates[0]
    completion = ""
    if cand.content and cand.content.parts:
        completion = "".join((p.text or "") for p in cand.content.parts)

    if cand.logprobs_result is None or cand.logprobs_result.top_candidates is None:
        raise ValueError("No logprobs_result.top_candidates returned from Gemini")
    if len(cand.logprobs_result.top_candidates) == 0:
        raise ValueError("Empty logprobs_result.top_candidates from Gemini")

    top0 = cand.logprobs_result.top_candidates[0]
    if top0.candidates is None:
        raise ValueError("Missing candidates in Gemini top_candidates[0]")

    pairs = [
        (c.token, float(c.log_probability))
        for c in top0.candidates
        if c.token is not None
    ]
    if len(pairs) == 0:
        raise ValueError("No (token, log_probability) pairs returned from Gemini")

    scores = probability_scores_from_logprobs(
        pairs, parse_score=parse_score, score_transform=score_transform
    )
    return ConfidenceScoringResult(completion=completion, top_logprobs=pairs, scores=scores)


def hf_next_token_top_logprobs(
    *,
    prompt: str,
    model,
    tokenizer,
    top_logprobs: int = 10,
    device: Optional[str] = None,
    parse_score: ScoreParser = parse_first_number,
    score_transform: ScoreTransform = lambda x: x,
) -> ConfidenceScoringResult:
    """Compute next-token top-k logprobs for a HF causal LM (no generation required)."""
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: torch") from e

    if device is None:
        device = getattr(model, "device", None)
    if device is None:
        device = "cpu"

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        out = model(**encoded)
        logits = out.logits[0, -1]
        logprobs = torch.log_softmax(logits, dim=-1)
        topk = torch.topk(logprobs, k=top_logprobs)

    token_ids = topk.indices.tolist()
    token_logprobs = topk.values.tolist()
    pairs: list[LogProbPair] = []
    for token_id, logprob in zip(token_ids, token_logprobs):
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        pairs.append((token, float(logprob)))

    completion = pairs[0][0] if pairs else ""
    scores = probability_scores_from_logprobs(
        pairs, parse_score=parse_score, score_transform=score_transform
    )
    return ConfidenceScoringResult(completion=completion, top_logprobs=pairs, scores=scores)


def claude_logprobs_available() -> bool:
    """Return whether Claude exposes token logprobs needed by this method."""
    return False


def claude_top_logprobs_unavailable_error() -> RuntimeError:
    return RuntimeError(
        "Anthropic Claude does not expose token-level log probabilities via the public API, "
        "so logprob-based confidence cannot be computed directly."
    )
