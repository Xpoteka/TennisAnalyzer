"""Training or match, from the structure of the rallies."""

from __future__ import annotations

from tennis.analysis.kind import RallyInfo, classify_session, serve_runs


def test_serve_runs() -> None:
    assert serve_runs([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]) == [4, 5, 1]


def match_rallies() -> list[RallyInfo]:
    out, t = [], 0.0
    for game in range(6):
        for _ in range(5):
            out.append(RallyInfo(t, t + 6, 4, True, game % 2))
            t += 26
    return out


def training_rallies() -> list[RallyInfo]:
    out, t = [], 0.0
    for k in range(30):
        out.append(RallyInfo(t, t + 20, 14 if k % 2 else 6, False, None))
        t += 24
    return out


def test_match_is_recognised() -> None:
    result = classify_session(match_rallies())
    assert result.kind == "match" and result.p_match > 0.8


def test_training_is_recognised() -> None:
    result = classify_session(training_rallies())
    assert result.kind == "training" and result.p_match < 0.2


def test_serve_practice_is_training() -> None:
    rallies = [RallyInfo(k * 8.0, k * 8.0 + 2, 1, True, 0) for k in range(30)]
    assert classify_session(rallies).kind != "match"


def test_too_little_to_say() -> None:
    assert classify_session(match_rallies()[:3]).kind == "unknown"
