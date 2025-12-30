import os
from argparse import Namespace

import pytest

from confidence import cli


def test_load_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=abc123\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    updates = cli.load_env_file(str(env_file))
    assert updates["OPENAI_API_KEY"] == "abc123"
    assert os.environ["OPENAI_API_KEY"] == "abc123"


def test_load_env_file_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=new-value\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "old-value")

    cli.load_env_file(str(env_file), override=False)
    assert os.environ["OPENAI_API_KEY"] == "old-value"

    cli.load_env_file(str(env_file), override=True)
    assert os.environ["OPENAI_API_KEY"] == "new-value"


def test_load_input_yaml_mapping(tmp_path):
    yaml_file = tmp_path / "input.yaml"
    yaml_file.write_text(
        "task: Rate the humor from 10-50\nresponse: Hello\nmin_score: 10\n",
        encoding="utf-8",
    )
    data = cli.load_input_yaml(str(yaml_file))
    assert data["task"] == "Rate the humor from 10-50"
    assert data["response"] == "Hello"


def test_load_input_yaml_invalid(tmp_path):
    yaml_file = tmp_path / "input.yaml"
    yaml_file.write_text("- bad\n- data\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.load_input_yaml(str(yaml_file))


def test_apply_yaml_overrides():
    args = Namespace(
        task=None,
        item=None,
        prompt=None,
        response="Already set",
        min_score=None,
        max_score=None,
        score_divisor=None,
        divide_by_10=False,
    )
    data = {
        "task": "Rate humor",
        "response": "Should not override",
        "min_score": 5,
        "max_score": 25,
        "score_divisor": 10,
        "divide_by_10": True,
    }
    cli.apply_yaml_overrides(args, data)
    assert args.task == "Rate humor"
    assert args.response == "Already set"
    assert args.min_score == 5
    assert args.max_score == 25
    assert args.score_divisor == 10
    assert args.divide_by_10 is True
