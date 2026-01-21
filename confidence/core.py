"""
Confidence-aware scoring utilities.

Compute confidence scores from top-k token log probabilities for the score token.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from itertools import product
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


def joint_score_logprobs(
    token_logprobs: Sequence[Sequence[LogProbPair]],
) -> list[LogProbPair]:
    """Combine per-token top-k logprobs into joint score candidates."""
    if len(token_logprobs) == 0:
        raise ValueError("token_logprobs must be non-empty")
    if len(token_logprobs) == 1:
        return list(token_logprobs[0])

    combos: list[LogProbPair] = []
    for combo in product(*token_logprobs):
        token = "".join(part for part, _ in combo)
        logprob = sum(float(lp) for _, lp in combo)
        combos.append((token, logprob))
    return combos


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
    token_logprobs: Optional[list[list[LogProbPair]]] = None


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
    """Call OpenAI Chat Completions and return top-k logprobs for the score tokens."""
    try:
        import openai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: openai") from e

    client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    try:
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
    except Exception as exc:
        if _openai_logprobs_unavailable_error_match(exc):
            detail = _openai_error_detail(exc)
            raise _openai_logprobs_unavailable_error(model=model, detail=detail) from exc
        raise

    choice = resp.choices[0]
    if getattr(choice, "logprobs", None) is None or choice.logprobs.content is None:
        raise ValueError(
            "No logprobs returned (model may not support logprobs or logprobs were disabled)"
        )

    token_logprobs: list[list[LogProbPair]] = []
    for token_info in choice.logprobs.content[:max_tokens]:
        if token_info.top_logprobs is None:
            raise ValueError("No top_logprobs returned for a score token")
        pairs = [(x.token, float(x.logprob)) for x in token_info.top_logprobs]
        token_logprobs.append(pairs)

    score_logprobs = joint_score_logprobs(token_logprobs)
    completion = choice.message.content or ""
    scores = probability_scores_from_logprobs(
        score_logprobs, parse_score=parse_score, score_transform=score_transform
    )
    return ConfidenceScoringResult(
        completion=completion,
        top_logprobs=score_logprobs,
        scores=scores,
        token_logprobs=token_logprobs,
    )


def _openai_logprobs_unavailable_error_match(exc: Exception) -> bool:
    message = str(exc).lower()
    if "logprobs" not in message:
        return False
    return any(
        phrase in message
        for phrase in (
            "not allowed",
            "not supported",
            "unsupported",
            "does not support",
            "doesn't support",
        )
    )


def _openai_error_detail(exc: Exception) -> Optional[str]:
    detail = str(exc).strip()
    return detail or None


def _openai_logprobs_unavailable_error(*, model: str, detail: Optional[str]) -> RuntimeError:
    hint = (
        "GPT-5 models (gpt-5, gpt-5-mini, gpt-5-nano) do not support logprobs via "
        "chat.completions."
    )
    suggestion = (
        "Try a gpt-4.1-x model instead (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano) or "
        "another logprobs-capable model."
    )
    if detail:
        message = f"OpenAI rejected logprobs for model '{model}'. {detail}. {hint} {suggestion}"
    else:
        message = f"OpenAI rejected logprobs for model '{model}'. {hint} {suggestion}"
    return RuntimeError(message)


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
    """Call Gemini (google-genai) and return top-k logprobs for the score tokens."""
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

    token_logprobs: list[list[LogProbPair]] = []
    for top in cand.logprobs_result.top_candidates[:max_output_tokens]:
        if top.candidates is None:
            raise ValueError("Missing candidates in Gemini top_candidates")
        pairs = [
            (c.token, float(c.log_probability))
            for c in top.candidates
            if c.token is not None
        ]
        if len(pairs) == 0:
            raise ValueError("No (token, log_probability) pairs returned from Gemini")
        token_logprobs.append(pairs)

    score_logprobs = joint_score_logprobs(token_logprobs)
    scores = probability_scores_from_logprobs(
        score_logprobs, parse_score=parse_score, score_transform=score_transform
    )
    return ConfidenceScoringResult(
        completion=completion,
        top_logprobs=score_logprobs,
        scores=scores,
        token_logprobs=token_logprobs,
    )


def hf_next_token_top_logprobs(
    *,
    prompt: str,
    model,
    tokenizer,
    top_logprobs: int = 10,
    score_tokens: int = 1,
    device: Optional[str] = None,
    parse_score: ScoreParser = parse_first_number,
    score_transform: ScoreTransform = lambda x: x,
) -> ConfidenceScoringResult:
    """Compute top-k logprobs for score tokens from a HF causal LM."""
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

    token_logprobs: list[list[LogProbPair]] = []
    completion_ids: list[int] = []

    for _ in range(score_tokens):
        with torch.no_grad():
            out = model(**encoded)
            logits = out.logits[0, -1]
            logprobs = torch.log_softmax(logits, dim=-1)
            topk = torch.topk(logprobs, k=top_logprobs)

        token_ids = topk.indices.tolist()
        token_vals = topk.values.tolist()
        pairs: list[LogProbPair] = []
        for token_id, logprob in zip(token_ids, token_vals):
            token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            pairs.append((token, float(logprob)))
        token_logprobs.append(pairs)

        next_id = token_ids[0] if token_ids else None
        if next_id is None:
            break
        completion_ids.append(next_id)
        next_tensor = torch.tensor([[next_id]], device=device, dtype=encoded["input_ids"].dtype)
        encoded["input_ids"] = torch.cat([encoded["input_ids"], next_tensor], dim=1)
        if "attention_mask" in encoded:
            next_mask = torch.ones((1, 1), device=device, dtype=encoded["attention_mask"].dtype)
            encoded["attention_mask"] = torch.cat([encoded["attention_mask"], next_mask], dim=1)

    score_logprobs = joint_score_logprobs(token_logprobs)
    completion = tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False)
    scores = probability_scores_from_logprobs(
        score_logprobs, parse_score=parse_score, score_transform=score_transform
    )
    return ConfidenceScoringResult(
        completion=completion,
        top_logprobs=score_logprobs,
        scores=scores,
        token_logprobs=token_logprobs,
    )


def claude_logprobs_available() -> bool:
    """Return whether Claude exposes token logprobs needed by this method."""
    return False


def claude_top_logprobs_unavailable_error() -> RuntimeError:
    return RuntimeError(
        "Anthropic Claude does not expose token-level log probabilities via the public API, "
        "so logprob-based confidence cannot be computed directly."
    )
