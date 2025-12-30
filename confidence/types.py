from typing import TypedDict

ProbScores = TypedDict(
    "ProbScores",
    {
        "weighted": float,
        "weighted_confidence": float,
        "top": float,
        "top_confidence": float,
        "n": int,
    },
)

LogProbPair = tuple[str, float]
