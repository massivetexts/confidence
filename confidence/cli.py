from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Callable, Dict, Optional

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

DEFAULT_PROMPT = "brick"
DEFAULT_RESPONSE = "use it as a stepping stool to reach a high shelf"
DEFAULT_MIN_SCORE = 10
DEFAULT_MAX_SCORE = 50
DEFAULT_SYSTEM_MESSAGE = "You are a scoring assistant."


def load_env_file(path: str, *, override: bool = False) -> Dict[str, str]:
    """Load environment variables from a .env file."""
    env_path = Path(path)
    if not env_path.exists():
        raise SystemExit(f"Env file not found: {path}")

    updates: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith(""") and value.endswith(""")) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        updates[key] = value
    return updates


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


def resolve_score_divisor(score_divisor: Optional[float], divide_by_10: bool) -> float:
    if score_divisor is None:
        return 10.0 if divide_by_10 else 1.0
    return float(score_divisor)


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

    args.score_divisor = resolve_score_divisor(args.score_divisor, args.divide_by_10)


def _build_default_prompt(
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


def _build_task_prompt(
    *,
    task: str,
    item: Optional[str],
    response: str,
    min_score: int,
    max_score: int,
) -> str:
    lines = [task.strip()]
    lines.append(
        f"Return a single token: an integer from {min_score} to {max_score}."
    )
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
    prompt_text = resolve_prompt(args)

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

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
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
    prompt_text = resolve_prompt(args)

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
    common.add_argument("--task", help="Custom scoring task prompt.")
    common.add_argument("--item", help="Optional item or stimulus for the task.")
    common.add_argument("--prompt", help="Prompt for the default originality template.")
    common.add_argument("--response", help="Response to score.")
    common.add_argument("--input-yaml", help="Load task/item/response from YAML.")
    common.add_argument("--env-file", help="Load API keys from a .env file.")
    common.add_argument(
        "--env-override",
        action="store_true",
        help="Allow .env values to override existing environment variables.",
    )
    common.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="Top-k candidates for the score token (OpenAI max is 20).",
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
    p_hf.set_defaults(top_logprobs=50)
    p_hf.set_defaults(func=run_hf)

    p_claude = sub.add_parser("claude", parents=[common])
    p_claude.set_defaults(func=run_claude)

    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file, override=args.env_override)

    if args.input_yaml:
        yaml_data = load_input_yaml(args.input_yaml)
        apply_yaml_overrides(args, yaml_data)

    normalize_args(args)
    args.func(args)
