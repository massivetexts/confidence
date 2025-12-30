from __future__ import annotations

import argparse
import math
import os
from typing import Callable, Optional

from .core import (
    ConfidenceScoringResult,
    claude_logprobs_available,
    claude_top_logprobs_unavailable_error,
    gemini_generate_content_top_logprobs,
    hf_next_token_top_logprobs,
    openai_chat_completion_top_logprobs,
    parse_first_number,
    probability_scores_from_logprobs,
    progressive_probability_scores_from_logprobs,
)


def _build_score_prompt(
    *,
    prompt: str,
    response: str,
    min_score: int,
    max_score: int,
) -> str:
    return (
        "You are a creativity judge. Rate the originality of the RESPONSE for the PROMPT.\n"
        f"Return a single token: an integer from {min_score} to {max_score}, "
        f"where {min_score} is not original at all and {max_score} is extremely original.\n"
        f"PROMPT: {prompt}\n"
        f"RESPONSE: {response}\n"
        "SCORE:"
    )


def _transform_from_divisor(divisor: float) -> Callable[[float], float]:
    if divisor == 1:
        return lambda x: x
    return lambda x: x / divisor


def _score_parser(min_score: int, max_score: int) -> Callable[[str], Optional[float]]:
    def _parse(token: str) -> Optional[float]:
        val = parse_first_number(token)
        if val is None:
            return None
        if val < min_score or val > max_score:
            return None
        return val

    return _parse


def _mock_result(
    *,
    parse_score: Callable[[str], Optional[float]],
    divisor: float,
) -> ConfidenceScoringResult:
    # A tiny synthetic top-k distribution for the score token.
    # The last candidate ("\n") is intentionally non-numeric so confidence < 1.0.
    top_logprobs = [
        (" 30", math.log(0.5)),
        (" 40", math.log(0.2)),
        (" 20", math.log(0.1)),
        ("\n", math.log(0.2)),
    ]
    scores = probability_scores_from_logprobs(
        top_logprobs,
        parse_score=parse_score,
        score_transform=_transform_from_divisor(divisor),
    )
    return ConfidenceScoringResult(
        completion=" 30",
        top_logprobs=top_logprobs,
        scores=scores,
    )


def _print_result(
    result: ConfidenceScoringResult,
    *,
    parse_score: Callable[[str], Optional[float]],
    divisor: float,
    progressive: bool,
) -> None:
    print("Raw completion:", repr(result.completion))
    print("Scores:", result.scores)

    if not progressive:
        return

    filtered = [(t, lp) for (t, lp) in result.top_logprobs if parse_score(t) is not None]
    if len(filtered) < 2:
        print("Progressive weighted: <not enough score-like candidates>")
        return

    prog = progressive_probability_scores_from_logprobs(
        filtered,
        parse_score=parse_score,
        score_transform=_transform_from_divisor(divisor),
        min_contributors=2,
    )
    print("Progressive weighted (k=2..K):")
    for i, scores in enumerate(prog, start=2):
        print(f"  k={i}: {scores}")


def run_openai(args: argparse.Namespace) -> None:
    parse_score = _score_parser(args.min_score, args.max_score)

    if args.mock:
        result = _mock_result(parse_score=parse_score, divisor=args.score_divisor)
        _print_result(
            result,
            parse_score=parse_score,
            divisor=args.score_divisor,
            progressive=args.progressive,
        )
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY (or pass --api-key). For a keyless demo, use --mock."
        )

    prompt_text = _build_score_prompt(
        prompt=args.prompt,
        response=args.response,
        min_score=args.min_score,
        max_score=args.max_score,
    )
    messages = [
        {"role": "system", "content": "You are a creativity judge, scoring originality."},
        {"role": "user", "content": prompt_text},
    ]
    result = openai_chat_completion_top_logprobs(
        model=args.model,
        messages=messages,
        top_logprobs=args.top_logprobs,
        max_tokens=1,
        stop=None,
        api_key=api_key,
        parse_score=parse_score,
        score_transform=_transform_from_divisor(args.score_divisor),
    )
    _print_result(
        result,
        parse_score=parse_score,
        divisor=args.score_divisor,
        progressive=args.progressive,
    )


def run_gemini(args: argparse.Namespace) -> None:
    parse_score = _score_parser(args.min_score, args.max_score)

    if args.mock:
        result = _mock_result(parse_score=parse_score, divisor=args.score_divisor)
        _print_result(
            result,
            parse_score=parse_score,
            divisor=args.score_divisor,
            progressive=args.progressive,
        )
        return

    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GEMINI_API_KEY/GOOGLE_API_KEY (or pass --api-key). For a keyless demo, use --mock."
        )

    prompt_text = _build_score_prompt(
        prompt=args.prompt,
        response=args.response,
        min_score=args.min_score,
        max_score=args.max_score,
    )
    result = gemini_generate_content_top_logprobs(
        model=args.model,
        contents=prompt_text,
        top_logprobs=args.top_logprobs,
        max_output_tokens=1,
        stop_sequences=("\n",),
        api_key=api_key,
        parse_score=parse_score,
        score_transform=_transform_from_divisor(args.score_divisor),
    )
    _print_result(
        result,
        parse_score=parse_score,
        divisor=args.score_divisor,
        progressive=args.progressive,
    )


def run_hf(args: argparse.Namespace) -> None:
    parse_score = _score_parser(args.min_score, args.max_score)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency: transformers. Try: uv run --with transformers --with torch examples/confidence_dropin.py hf ..."
        ) from e

    prompt_text = _build_score_prompt(
        prompt=args.prompt,
        response=args.response,
        min_score=args.min_score,
        max_score=args.max_score,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    result = hf_next_token_top_logprobs(
        prompt=prompt_text,
        model=model,
        tokenizer=tokenizer,
        top_logprobs=args.top_logprobs,
        device=args.device,
        parse_score=parse_score,
        score_transform=_transform_from_divisor(args.score_divisor),
    )
    _print_result(
        result,
        parse_score=parse_score,
        divisor=args.score_divisor,
        progressive=args.progressive,
    )


def run_claude(_: argparse.Namespace) -> None:
    if claude_logprobs_available():
        raise RuntimeError("Unexpected: claude_logprobs_available() returned True")
    raise claude_top_logprobs_unavailable_error()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="provider", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--prompt", default="brick")
    common.add_argument(
        "--response",
        default="use it as a stepping stool to reach a high shelf",
    )
    common.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="Top-k candidates for the score token (OpenAI max is 20).",
    )
    common.add_argument("--min-score", type=int, default=10)
    common.add_argument("--max-score", type=int, default=50)
    common.add_argument(
        "--score-divisor",
        type=float,
        default=10.0,
        help="Divide parsed numeric tokens by this value (default 10 => 10..50 -> 1.0..5.0).",
    )
    common.add_argument("--progressive", action="store_true")

    p_openai = sub.add_parser("openai", parents=[common])
    p_openai.add_argument("--model", default="gpt-4o-mini")
    p_openai.add_argument("--api-key")
    p_openai.add_argument("--mock", action="store_true")
    p_openai.set_defaults(func=run_openai)

    p_gemini = sub.add_parser("gemini", parents=[common])
    p_gemini.add_argument("--model", default="gemini-2.0-flash")
    p_gemini.add_argument("--api-key")
    p_gemini.add_argument("--mock", action="store_true")
    p_gemini.set_defaults(func=run_gemini)

    p_hf = sub.add_parser("hf", parents=[common])
    p_hf.add_argument("--model", default="gpt2")
    p_hf.add_argument("--device", default="cpu")
    p_hf.set_defaults(top_logprobs=50)
    p_hf.set_defaults(func=run_hf)

    p_claude = sub.add_parser("claude", parents=[common])
    p_claude.set_defaults(func=run_claude)

    args = parser.parse_args()
    args.func(args)
