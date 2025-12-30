
from argparse import Namespace

from confidence import cli


def _base_args():
    return Namespace(
        task=None,
        item=None,
        prompt=None,
        response=None,
        min_score=None,
        max_score=None,
        score_divisor=None,
        divide_by_10=False,
    )


def test_build_prompt_task_and_item():
    prompt = cli.build_prompt(
        task="Rate the humor of the response.",
        item="A one-liner joke",
        prompt=None,
        response="Why did the chicken cross the road?",
        min_score=10,
        max_score=50,
    )
    assert "Rate the humor" in prompt
    assert "ITEM: A one-liner joke" in prompt
    assert "RESPONSE: Why did the chicken" in prompt


def test_build_prompt_default_uses_prompt():
    prompt = cli.build_prompt(
        task=None,
        item=None,
        prompt="brick",
        response="use it as a paperweight",
        min_score=10,
        max_score=50,
    )
    assert "PROMPT: brick" in prompt


def test_build_prompt_default_uses_item():
    prompt = cli.build_prompt(
        task=None,
        item="banana",
        prompt=None,
        response="as a phone",
        min_score=10,
        max_score=50,
    )
    assert "PROMPT: banana" in prompt


def test_resolve_score_divisor():
    assert cli.resolve_score_divisor(None, True) == 10.0
    assert cli.resolve_score_divisor(None, False) == 1.0
    assert cli.resolve_score_divisor(5.0, False) == 5.0


def test_normalize_args_defaults():
    args = _base_args()
    cli.normalize_args(args)
    assert args.min_score == cli.DEFAULT_MIN_SCORE
    assert args.max_score == cli.DEFAULT_MAX_SCORE
    assert args.prompt == cli.DEFAULT_PROMPT
    assert args.response == cli.DEFAULT_RESPONSE
