"""Session stage ``players``: link the session's players to profiles.

See :mod:`tennis.analysis.reid` for how players are compared.

For each player of the session, a signature is measured from their full-rate poses:

- **racket hand:** voted over their serves and smashes;
- **height:** where the court camera is known, the nose's height above the feet (a ray
  through the nose pixel, closest to the vertical line over the feet) divided by 0.935,
  the nose's usual share of body height; the 90th percentile, since players crouch;
- **proportions:** limb lengths relative to the torso, from frames where the player is
  large and every joint is seen.

Profiles already linked to this session are kept when they still match, so re-analysing a
session does not create new profiles. A profile left without sessions is removed, unless
it was renamed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from sqlmodel import select

from tennis.analysis.reid import assign, combine, signature_distance
from tennis.db import session_scope
from tennis.db.models import Player, SessionPlayer
from tennis.pipeline.session.shots import swings_around_hits
from tennis.util.io import read_json
from tennis.vision import court as court_mod
from tennis.vision.strokes import KP, racket_hand

if TYPE_CHECKING:
    from tennis.pipeline.session import SessionContext

NOSE_SHARE_OF_HEIGHT = 0.935
ANKLE_HEIGHT_M = 0.08
MIN_BOX_PX = 120  # joints of smaller players are too coarse for proportions
LIMBS = (
    ("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow"),
    ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"), ("r_hip", "r_knee"),
    ("l_knee", "l_ankle"), ("r_knee", "r_ankle"),
)  # fmt: skip


def body_measures(kp: np.ndarray, height_px: np.ndarray) -> list[float] | None:
    """Median limb lengths relative to the torso: upper arm, forearm, thigh, shin."""
    ok = kp[..., 2] >= 0.5
    rows = []
    for k, seen, h in zip(kp, ok, height_px, strict=True):
        if h < MIN_BOX_PX or not seen.all():
            continue
        p = k[:, :2]
        shoulders = (p[KP["l_shoulder"]] + p[KP["r_shoulder"]]) / 2
        hips = (p[KP["l_hip"]] + p[KP["r_hip"]]) / 2
        torso = float(np.linalg.norm(shoulders - hips))
        if torso < 10:
            continue
        lengths = [float(np.linalg.norm(p[KP[a]] - p[KP[b]])) / torso for a, b in LIMBS]
        rows.append([(lengths[i] + lengths[i + 1]) / 2 for i in range(0, 8, 2)])
    if len(rows) < 50:
        return None
    return [round(float(v), 4) for v in np.median(rows, axis=0)]


def standing_height(kp: np.ndarray, cal: court_mod.Calibration | None) -> float | None:
    """Body height in metres from frames with the nose and both ankles seen."""
    if cal is None or not cal.has_camera:
        return None
    ok = (
        (kp[:, KP["nose"], 2] >= 0.5)
        & (kp[:, KP["l_ankle"], 2] >= 0.5)
        & (kp[:, KP["r_ankle"], 2] >= 0.5)
    )
    if ok.sum() < 50:
        return None
    sel = kp[ok][:: max(1, int(ok.sum()) // 600)]  # a few hundred frames are plenty
    feet_px = (sel[:, KP["l_ankle"], :2] + sel[:, KP["r_ankle"], :2]) / 2
    # The ankles are about 8 cm above the ground: where their rays cross that height is
    # where the player stands (projecting them onto the ground would put the feet too far).
    centre, ankle_rays = cal.image_rays(feet_px)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_ankle = (ANKLE_HEIGHT_M - centre[2]) / ankle_rays[:, 2]
    feet = centre[:2] + s_ankle[:, None] * ankle_rays[:, :2]
    _, rays = cal.image_rays(sel[:, KP["nose"], :2])
    heights = []
    for f, d in zip(feet, rays, strict=True):
        if not np.isfinite(f).all():
            continue
        # The point on the ray closest to the vertical line through the feet.
        horiz = d[:2]
        denom = float(horiz @ horiz)
        if denom < 1e-9:
            continue
        s = float((f - centre[:2]) @ horiz) / denom
        z = centre[2] + s * d[2]
        if 0.8 < z < 2.3:
            heights.append(z / NOSE_SHARE_OF_HEIGHT)
    if len(heights) < 30:
        return None
    return round(float(np.percentile(heights, 90)), 3)


def measure(ctx: SessionContext, identities: dict[str, Any]) -> dict[str, dict[str, Any]]:
    kp_by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    box_by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    heights: dict[str, list[float]] = defaultdict(list)
    swings: dict[str, list[Any]] = defaultdict(list)
    for v in ctx.videos:
        labels = {int(k): lab for k, lab in identities["tracks"].get(str(v.id), {}).items()}
        if not labels:
            continue
        m = pq.read_table(v.path("motion.parquet")).to_pydict()
        if not m["t"]:
            continue
        track = np.asarray(m["track"])
        kp = np.stack([np.asarray(m["kp_x"]), np.asarray(m["kp_y"]), np.asarray(m["kp_c"])], axis=2)
        box_h = np.asarray(m["height"], np.float64)
        court = v.read_json("court.json")
        cal = court_mod.Calibration.from_json(court) if court.get("found") else None
        for label in set(labels.values()):
            sel = np.isin(track, [k for k, lab in labels.items() if lab == label])
            kp_by_label[label].append(kp[sel])
            box_by_label[label].append(box_h[sel])
            h = standing_height(kp[sel], cal)
            if h is not None:
                heights[label].append(h)
        hits = pq.read_table(v.path("hits.parquet"), columns=["t", "track"]).to_pydict()
        for i, sw in swings_around_hits(v, hits).items():
            label = labels.get(int(hits["track"][i]))
            if label:
                swings[label].append(sw)
    out: dict[str, dict[str, Any]] = {}
    for label, info in identities["players"].items():
        kps = kp_by_label.get(label)
        limbs = (
            body_measures(np.concatenate(kps), np.concatenate(box_by_label[label])) if kps else None
        )
        out[label] = {
            "hand": racket_hand(swings.get(label, [])),
            "height_m": round(float(np.median(heights[label])), 3) if heights.get(label) else None,
            "limbs": limbs,
            "appearance": info.get("appearance"),
        }
    return out


def run(ctx: SessionContext) -> None:
    identities = read_json(ctx.dir / "identities.json")
    signatures = measure(ctx, identities)
    with session_scope(ctx.data_root) as db:
        mine = {
            sp.label: sp
            for sp in db.exec(
                select(SessionPlayer).where(SessionPlayer.session_id == ctx.session_id)
            )
        }
        # Every other session's players, per profile (merged profiles count as their target).
        others: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sp in db.exec(select(SessionPlayer).where(SessionPlayer.session_id != ctx.session_id)):
            pid = _resolve(db, sp.player_id)
            if sp.signature:
                others[pid].append(sp.signature)
        profiles = {pid: combine(sigs) for pid, sigs in others.items()}
        # This session's current profiles compete too, so a re-analysis keeps them.
        for sp in mine.values():
            pid = _resolve(db, sp.player_id)
            profiles.setdefault(pid, combine([signatures.get(sp.label, {})]))
        chosen = assign(signatures, profiles)
        previous = {sp.label: sp.player_id for sp in mine.values()}
        for label, sig in signatures.items():
            row = mine.get(label)
            if row is None:
                continue
            found = chosen.get(label)
            if found is None:
                player = Player(name="")
                db.add(player)
                db.flush()
                player.name = f"Player {player.id}"
                found = player.id
            pid = found or row.player_id
            row.player_id = pid
            # Recognised: the profile also has other sessions. Otherwise it is new.
            known = pid in others
            row.match_distance = round(signature_distance(sig, profiles[pid]), 3) if known else None
            row.signature = sig
            db.add(row)
        db.flush()
        # Refresh the profiles touched, and drop profiles left without a session.
        touched = set(previous.values()) | {sp.player_id for sp in mine.values()}
        for pid in touched:
            profile = db.get(Player, pid)
            if profile is None:
                continue
            sigs = [
                sp.signature
                for sp in db.exec(select(SessionPlayer).where(SessionPlayer.player_id == pid))
                if sp.signature
            ]
            if not sigs:
                if profile.name.startswith("Player ") and profile.merged_into is None:
                    db.delete(profile)
                continue
            agg = combine(sigs)
            profile.signature = agg
            profile.handedness = agg.get("hand") or profile.handedness
            profile.height_m = agg.get("height_m") or profile.height_m
            if not profile.thumbnail:
                profile.thumbnail = next(
                    (sp.thumbnail for sp in mine.values() if sp.player_id == pid and sp.thumbnail),
                    None,
                )
            db.add(profile)
    ctx.log(
        "players",
        signatures={
            k: {"hand": v["hand"], "height_m": v["height_m"]} for k, v in signatures.items()
        },
        matched={k: v for k, v in chosen.items()},
    )


def _resolve(db: Any, pid: int) -> int:
    seen = set()
    player = db.get(Player, pid)
    while player is not None and player.merged_into is not None and player.id not in seen:
        seen.add(player.id)
        player = db.get(Player, player.merged_into)
    return int(player.id) if player is not None and player.id is not None else pid
