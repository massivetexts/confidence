from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

from .core import (
    ConfidenceScoringResult,
    claude_logprobs_available,
    claude_top_logprobs_unavailable_error,
    gemini_generate_content_top_logprobs,
    hf_next_token_top_logprobs,
    joint_score_logprobs,
    openai_chat_completion_top_logprobs,
    parse_first_number,
    probability_scores_from_logprobs,
    progressive_probability_scores_from_logprobs,
)

DEFAULT_PROMPT = "brick"
DEFAULT_RESPONSE = "use it as a stepping stool to reach a high shelf"
DEFAULT_MIN_SCORE = 10
DEFAULT_MAX_SCORE = 50
DEFAULT_SYSTEM_MESSAGE = "You are a scoring assistant."


def resolve_env_file(path: Optional[str]) -> Optional[str]:
    if path:
        return path
    for candidate in (".env.local", ".env"):
        if Path(candidate).exists():
            return candidate
    return None


def load_env_file(path: str, *, override: bool = False) -> Dict[str, str]:
    """Load environment variables from a .env file."""
    try:
        from dotenv import dotenv_values, load_dotenv  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: python-dotenv. Install with `pip install python-dotenv` "
            "or `uv run --with python-dotenv ...`."
        ) from e

    env_path = Path(path)
    if not env_path.exists():
        raise SystemExit(f"Env file not found: {path}")

    load_dotenv(env_path, override=override)
    values = dotenv_values(env_path)
    return {key: value for key, value in values.items() if value is not None}


def load_input_yaml(path: str) -> Dict[str, object]:
    """Load prompt inputs from a YAML file."""
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: pyyaml. Install with `pip install pyyaml` or "
            "`uv run --with pyyaml ...`."
        ) from e

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit("YAML input must be a mapping (key: value).")
    return data


def apply_yaml_overrides(args: argparse.Namespace, data: Dict[str, object]) -> None:
    """Apply YAML values if the CLI flags were not set."""
    for key in ("task", "item", "prompt", "response"):
        if getattr(args, key) is None and key in data:
            setattr(args, key, data[key])

    if args.min_score is None and "min_score" in data:
        args.min_score = data["min_score"]
    if args.max_score is None and "max_score" in data:
        args.max_score = data["max_score"]

    if args.score_divisor is None and "score_divisor" in data:
        args.score_divisor = data["score_divisor"]
    if not args.divide_by_10 and data.get("divide_by_10"):
        args.divide_by_10 = True
    if args.score_tokens is None and "score_tokens" in data:
        args.score_tokens = data["score_tokens"]


def resolve_score_divisor(score_divisor: Optional[float], divide_by_10: bool) -> float:
    if score_divisor is None:
        return 10.0 if divide_by_10 else 1.0
    return float(score_divisor)


def _auto_score_tokens(
    *,
    provider: str,
    divide_by_10: bool,
    score_divisor: float,
) -> int:
    if provider == "gemini" and (divide_by_10 or score_divisor == 10):
        return 2
    return 1


def _resolve_score_tokens(args: argparse.Namespace) -> int:
    raw = args.score_tokens
    if raw is None:
        return 1
    if isinstance(raw, int):
        tokens = raw
    else:
        raw_str = str(raw).strip().lower()
        if raw_str == "auto":
            return _auto_score_tokens(
                provider=args.provider,
                divide_by_10=args.divide_by_10,
                score_divisor=resolve_score_divisor(args.score_divisor, args.divide_by_10),
            )
        try:
            tokens = int(raw_str)
        except ValueError as exc:
            raise SystemExit("score-tokens must be 1-3 or 'auto'") from exc
    if tokens < 1 or tokens > 3:
        raise SystemExit("score-tokens must be 1-3 or 'auto'")
    return tokens


def normalize_args(args: argparse.Namespace) -> None:
    if args.min_score is None:
        args.min_score = DEFAULT_MIN_SCORE
    if args.max_score is None:
        args.max_score = DEFAULT_MAX_SCORE
    args.min_score = int(args.min_score)
    args.max_score = int(args.max_score)

    if args.min_score >= args.max_score:
        raise SystemExit("min-score must be less than max-score")

    if args.response is None:
        args.response = DEFAULT_RESPONSE

    if args.task is None and args.prompt is None and args.item is None:
        args.prompt = DEFAULT_PROMPT

    if args.top_logprobs is None:
        args.top_logprobs = 50 if args.provider == "hf" else 20
    if args.provider in {"openai", "gemini"} and args.top_logprobs > 20:
        raise SystemExit("top-logprobs must be <= 20 for OpenAI/Gemini.")

    args.score_divisor = resolve_score_divisor(args.score_divisor, args.divide_by_10)
    args.score_tokens = _resolve_score_tokens(args)


def _build_default_prompt(
    *,
    prompt: str,
    response: str,
    min_score: int,
    max_score: int,
) -> str:
    return (
        "You are a creativity judge. Rate the originality of the RESPONSE for the PROMPT.\n"
        f"Return only a number from {min_score} to {max_score}, where {min_score} "
        f"is not original at all and {max_score} is extremely original.\n"
        f"PROMPT: {prompt}\n"
        f"RESPONSE: {response}\n"
        "SCORE:"
    )


def _build_task_prompt(
    *,
    task: str,
    item: Optional[str],
    response: str,
    min_score: int,
    max_score: int,
) -> str:
    lines = [task.strip()]
    lines.append(f"Return only a number from {min_score} to {max_score}.")
    if item:
        lines.append(f"ITEM: {item}")
    lines.append(f"RESPONSE: {response}")
    lines.append("SCORE:")
    return "\n".join(lines)


def build_prompt(
    *,
    task: Optional[str],
    item: Optional[str],
    prompt: Optional[str],
    response: str,
    min_score: int,
    max_score: int,
) -> str:
    if task:
        return _build_task_prompt(
            task=task,
            item=item,
            response=response,
            min_score=min_score,
            max_score=max_score,
        )

    prompt_value = prompt if prompt is not None else (item if item else DEFAULT_PROMPT)
    return _build_default_prompt(
        prompt=prompt_value,
        response=response,
        min_score=min_score,
        max_score=max_score,
    )


def resolve_prompt(args: argparse.Namespace) -> str:
    return build_prompt(
        task=args.task,
        item=args.item,
        prompt=args.prompt,
        response=args.response,
        min_score=args.min_score,
        max_score=args.max_score,
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
    score_tokens: int,
) -> ConfidenceScoringResult:
    # A tiny synthetic top-k distribution for the score tokens.
    # The last candidate ("\n") is intentionally non-numeric so confidence < 1.0.
    if score_tokens == 1:
        token_logprobs = [
            [
                (" 30", math.log(0.5)),
                (" 40", math.log(0.2)),
                (" 20", math.log(0.1)),
                ("\n", math.log(0.2)),
            ]
        ]
    elif score_tokens == 2:
        token_logprobs = [
            [
                (" 2", math.log(0.5)),
                (" 3", math.log(0.2)),
                (" 1", math.log(0.1)),
                ("\n", math.log(0.2)),
            ],
            [
                ("0", math.log(0.6)),
                ("5", math.log(0.2)),
                ("\n", math.log(0.2)),
            ],
        ]
    else:
        token_logprobs = [
            [
                (" 1", math.log(0.5)),
                (" 2", math.log(0.2)),
                (" 3", math.log(0.1)),
                ("\n", math.log(0.2)),
            ],
            [
                (".", math.log(0.7)),
                (" ", math.log(0.3)),
            ],
            [
                ("5", math.log(0.5)),
                ("0", math.log(0.2)),
                ("\n", math.log(0.3)),
            ],
        ]

    score_logprobs = joint_score_logprobs(token_logprobs)
    scores = probability_scores_from_logprobs(
        score_logprobs,
        parse_score=parse_score,
        score_transform=_transform_from_divisor(divisor),
    )
    completion = "".join(parts[0][0] for parts in token_logprobs)
    return ConfidenceScoringResult(
        completion=completion,
        top_logprobs=score_logprobs,
        scores=scores,
        token_logprobs=token_logprobs,
    )


def _print_result(
    result: ConfidenceScoringResult,
    *,
    provider: str,
    model: str,
    prompt_text: str,
    args: argparse.Namespace,
    parse_score: Callable[[str], Optional[float]],
    divisor: float,
    progressive: bool,
) -> None:
    payload: Dict[str, object] = {
        "provider": provider,
        "model": model,
        "inputs": {
            "task": args.task,
            "item": args.item,
            "prompt": args.prompt,
            "response": args.response,
            "prompt_text": prompt_text,
            "min_score": args.min_score,
            "max_score": args.max_score,
            "score_divisor": args.score_divisor,
            "divide_by_10": args.divide_by_10,
            "score_tokens": args.score_tokens,
            "top_logprobs": args.top_logprobs,
            "device": getattr(args, "device", None),
        },
        "output": {
            "completion": result.completion,
            "scores": result.scores,
        },
    }

    if progressive:
        filtered = [
            (t, lp) for (t, lp) in result.top_logprobs if parse_score(t) is not None
        ]
        if len(filtered) >= 2:
            payload["progressive"] = progressive_probability_scores_from_logprobs(
                filtered,
                parse_score=parse_score,
                score_transform=_transform_from_divisor(divisor),
                min_contributors=2,
            )
        else:
            payload["progressive"] = []

    print(json.dumps(payload))


def run_openai(args: argparse.Namespace) -> None:
    parse_score = _score_parser(args.min_score, args.max_score)
    prompt_text = resolve_prompt(args)

    if args.mock:
        result = _mock_result(
            parse_score=parse_score,
            divisor=args.score_divisor,
            score_tokens=args.score_tokens,
        )
        _print_result(
            result,
            provider=args.provider,
            model=args.model,
            prompt_text=prompt_text,
            args=args,
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

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt_text},
    ]
    result = openai_chat_completion_top_logprobs(
        model=args.model,
        messages=messages,
        top_logprobs=args.top_logprobs,
        max_tokens=args.score_tokens,
        stop=None,
        api_key=api_key,
        parse_score=parse_score,
        score_transform=_transform_from_divisor(args.score_divisor),
    )
    _print_result(
        result,
        provider=args.provider,
        model=args.model,
        prompt_text=prompt_text,
        args=args,
        parse_score=parse_score,
        divisor=args.score_divisor,
        progressive=args.progressive,
    )


def run_gemini(args: argparse.Namespace) -> None:
    parse_score = _score_parser(args.min_score, args.max_score)
    prompt_text = resolve_prompt(args)

    if args.mock:
        result = _mock_result(
            parse_score=parse_score,
            divisor=args.score_divisor,
            score_tokens=args.score_tokens,
        )
        _print_result(
            result,
            provider=args.provider,
            model=args.model,
            prompt_text=prompt_text,
            args=args,
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

    if args.score_tokens == 1 and args.divide_by_10:
        print(
            "Warning: Gemini uses one token per digit, so divide-by-10 does not yield a single "
            "token. Use --score-tokens 2 or 3.",
            file=sys.stderr,
        )

    try:
        result = gemini_generate_content_top_logprobs(
            model=args.model,
            contents=prompt_text,
            top_logprobs=args.top_logprobs,
            max_output_tokens=args.score_tokens,
            stop_sequences=("\n",),
            api_key=api_key,
            parse_score=parse_score,
            score_transform=_transform_from_divisor(args.score_divisor),
        )
    except ValueError as exc:
        msg = str(exc)
        if args.score_tokens == 1 and "score-like candidates" in msg:
            raise SystemExit(
                "No score-like candidates found. Gemini splits numbers into multiple tokens; "
                "try --score-tokens 2 or 3."
            ) from exc
        raise
    _print_result(
        result,
        provider=args.provider,
        model=args.model,
        prompt_text=prompt_text,
        args=args,
        parse_score=parse_score,
        divisor=args.score_divisor,
        progressive=args.progressive,
    )


def run_hf(args: argparse.Namespace) -> None:
    parse_score = _score_parser(args.min_score, args.max_score)
    prompt_text = resolve_prompt(args)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing dependency: transformers. Try: uv run --with transformers --with torch examples/confidence_dropin.py hf ..."
        ) from e

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
        score_tokens=args.score_tokens,
        device=args.device,
        parse_score=parse_score,
        score_transform=_transform_from_divisor(args.score_divisor),
    )
    _print_result(
        result,
        provider=args.provider,
        model=args.model,
        prompt_text=prompt_text,
        args=args,
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
    common.add_argument("--task", help="Custom scoring task prompt.")
    common.add_argument("--item", help="Optional item or stimulus for the task.")
    common.add_argument("--prompt", help="Prompt for the default originality template.")
    common.add_argument("--response", help="Response to score.")
    common.add_argument("--input-yaml", help="Load task/item/response from YAML.")
    common.add_argument(
        "--env-file",
        help="Load API keys from a .env file (defaults to .env.local/.env if present).",
    )
    common.add_argument(
        "--env-override",
        action="store_true",
        help="Allow .env values to override existing environment variables.",
    )
    common.add_argument(
        "--top-logprobs",
        type=int,
        default=None,
        help="Top-k candidates for the score token (OpenAI/Gemini max is 20).",
    )
    common.add_argument(
        "--score-tokens",
        type=str,
        default=None,
        help="Number of score tokens to consider (1-3 or auto).",
    )
    common.add_argument("--min-score", type=int)
    common.add_argument("--max-score", type=int)

    divisor_group = common.add_mutually_exclusive_group()
    divisor_group.add_argument(
        "--score-divisor",
        type=float,
        help="Divide parsed numeric tokens by this value.",
    )
    divisor_group.add_argument(
        "--divide-by-10",
        action="store_true",
        help="Transform scores by dividing by 10.",
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
    p_hf.set_defaults(func=run_hf)

    p_claude = sub.add_parser("claude", parents=[common])
    p_claude.set_defaults(func=run_claude)

    args = parser.parse_args()

    env_path = resolve_env_file(args.env_file)
    if env_path:
        load_env_file(env_path, override=args.env_override)

    if args.input_yaml:
        yaml_data = load_input_yaml(args.input_yaml)
        apply_yaml_overrides(args, yaml_data)

    normalize_args(args)
    args.func(args)
