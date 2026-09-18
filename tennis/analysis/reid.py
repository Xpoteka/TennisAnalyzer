"""Recognise players across sessions, without asking.

A player's clothes change from one session to the next, and a court camera is too far away
for faces. What stays the same:

- **racket hand** (from serves: the hand above the head at contact); a mismatch is close to
  decisive;
- **height** in metres, measured through the court camera model;
- **body proportions**: forearm, upper arm, thigh and shin, each relative to the torso;
- **clothing colour**, which often repeats but proves nothing, so it weighs little.

Each cue adds to a distance; a cue missing on either side adds a neutral amount, so a thin
profile is neither favoured nor punished. Players of one session are assigned to profiles
together (Hungarian algorithm), since two players in one session cannot be the same person.
A player further than ``MATCH_DISTANCE`` from every free profile gets a new one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from tennis.util.appearance import distance as appearance_distance

MATCH_DISTANCE = 2.2
HEIGHT_UNIT_M = 0.06  # a height difference of this counts as one unit of distance
LIMB_UNIT = 0.08
MISSING = 0.5  # a cue one side does not have


def signature_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    d = 0.0
    ha, hb = a.get("hand"), b.get("hand")
    d += (3.0 if ha != hb else 0.0) if ha and hb else MISSING
    ya, yb = a.get("height_m"), b.get("height_m")
    d += min(abs(ya - yb) / HEIGHT_UNIT_M, 4.0) if ya and yb else MISSING
    la, lb = a.get("limbs"), b.get("limbs")
    if la and lb and len(la) == len(lb):
        d += min(float(np.linalg.norm(np.array(la) - np.array(lb))) / LIMB_UNIT, 4.0)
    else:
        d += MISSING
    aa, ab = a.get("appearance"), b.get("appearance")
    d += 0.8 * appearance_distance(np.array(aa), np.array(ab)) if aa and ab else MISSING * 0.8
    return d


def combine(signatures: list[dict[str, Any]]) -> dict[str, Any]:
    """A profile's signature from its sessions' ones (the latest clothes)."""
    if not signatures:
        return {}
    hands = [s["hand"] for s in signatures if s.get("hand")]
    heights = [s["height_m"] for s in signatures if s.get("height_m")]
    limbs = [s["limbs"] for s in signatures if s.get("limbs")]
    looks = [s["appearance"] for s in signatures if s.get("appearance")]
    out: dict[str, Any] = {}
    if hands:
        out["hand"] = max(set(hands), key=hands.count)
    if heights:
        out["height_m"] = round(float(np.median(heights)), 3)
    if limbs and all(len(x) == len(limbs[0]) for x in limbs):
        out["limbs"] = np.round(np.mean(limbs, axis=0), 4).tolist()
    if looks:
        out["appearance"] = looks[-1]
    return out


def assign(
    players: dict[str, dict[str, Any]], profiles: dict[int, dict[str, Any]]
) -> dict[str, int | None]:
    """Map each session player (by label) to a profile id, or None for a new profile."""
    labels = list(players)
    ids = list(profiles)
    result: dict[str, int | None] = dict.fromkeys(labels)
    if not labels or not ids:
        return result
    cost = np.array(
        [[signature_distance(players[lab], profiles[i]) for i in ids] for lab in labels]
    )
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols, strict=True):
        if cost[r, c] <= MATCH_DISTANCE:
            result[labels[r]] = ids[c]
    return result
