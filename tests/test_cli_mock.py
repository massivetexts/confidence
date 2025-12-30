
import json
from argparse import Namespace

from confidence import cli


def test_run_openai_mock(capsys):
    args = Namespace(
        provider="openai",
        task="Rate the humor of the response from 10-50.",
        item=None,
        prompt=None,
        response="A short pun.",
        min_score=10,
        max_score=50,
        score_divisor=10.0,
        divide_by_10=False,
        top_logprobs=20,
        score_tokens=1,
        progressive=False,
        model="gpt-4o-mini",
        api_key=None,
        mock=True,
        env_file=None,
        env_override=False,
        input_yaml=None,
    )
    cli.run_openai(args)
    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload["provider"] == "openai"
    assert payload["output"]["completion"]
    assert "weighted" in payload["output"]["scores"]
