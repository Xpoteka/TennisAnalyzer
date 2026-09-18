"""Recognising players across sessions, and the player profile."""

from __future__ import annotations

import numpy as np

from tennis.analysis.profile import player_profile
from tennis.analysis.reid import assign, combine, signature_distance

LOOK_A = [0.0, 0.8, 0.2, 0.0]
LOOK_B = [0.6, 0.0, 0.0, 0.4]


def sig(hand: str, height: float, limbs: list[float], look: list[float]) -> dict[str, object]:
    return {"hand": hand, "height_m": height, "limbs": limbs, "appearance": look}


JERRY = sig("left", 1.80, [0.62, 0.55, 0.85, 0.80], LOOK_A)
JOSI = sig("right", 1.68, [0.58, 0.52, 0.80, 0.78], LOOK_B)


def test_same_player_in_new_clothes_is_close() -> None:
    again = sig("left", 1.81, [0.63, 0.55, 0.84, 0.81], LOOK_B)  # other shirt
    assert signature_distance(again, JERRY) < signature_distance(again, JOSI)
    assert signature_distance(again, JERRY) < 2.2


def test_assignment_keeps_two_players_apart() -> None:
    profiles = {1: JERRY, 2: JOSI}
    session = {
        "A": sig("right", 1.69, [0.58, 0.52, 0.81, 0.78], LOOK_A),
        "B": sig("left", 1.79, [0.62, 0.56, 0.85, 0.80], LOOK_B),
    }
    assert assign(session, profiles) == {"A": 2, "B": 1}


def test_a_stranger_gets_a_new_profile() -> None:
    stranger = sig("right", 1.95, [0.70, 0.60, 0.95, 0.90], [0.25, 0.25, 0.25, 0.25])
    assert assign({"A": stranger}, {1: JERRY}) == {"A": None}


def test_combine_takes_the_majority_hand_and_median_height() -> None:
    out = combine(
        [
            JERRY,
            sig("left", 1.78, JERRY["limbs"], LOOK_B),
            sig("right", 1.9, JERRY["limbs"], LOOK_A),
        ]
    )  # type: ignore[arg-type]
    assert out["hand"] == "left"
    assert out["height_m"] == 1.8


def test_profile_aggregates_and_explains() -> None:
    rng = np.random.default_rng(0)
    shots = []
    metrics = {}
    for i in range(40):
        stroke = "forehand" if i % 2 else "backhand"
        shots.append(
            {
                "id": i,
                "session_id": 1 if i < 20 else 2,
                "stroke": stroke,
                "speed_kmh": float(110 if stroke == "forehand" else 80) + rng.normal(0, 3),
                "in_court": bool(i % 5),
                "net_clearance_m": 0.8,
                "contact_height_m": 1.0,
            }
        )
        metrics[i] = {"knee_bend_deg": 160.0, "split_step": 0.0 if i % 3 else 1.0}
    sessions = [
        {"id": 1, "recorded_at": None, "kind": "training"},
        {"id": 2, "recorded_at": None, "kind": "match"},
    ]
    profile = player_profile(shots, metrics, sessions)
    assert profile["by_stroke"]["forehand"]["count"] == 20
    assert 105 < profile["by_stroke"]["forehand"]["speed_avg"] < 115
    assert len(profile["trend"]) == 2
    texts = " ".join(i["text"] for i in profile["insights"])
    assert "forehand is the weapon" in texts
    assert "Legs stay fairly straight" in texts
    assert "Split step before only" in texts
