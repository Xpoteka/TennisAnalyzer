"""Session stages ``kind`` and ``scoring``: training or match, and the score of a match.

``kind`` decides from the rally structure (:mod:`tennis.analysis.kind`) unless the user set
it by hand. ``scoring`` only runs for matches between two players:

1. **Points.** A point starts with a serve. Rallies that do not (balls hit back between
   points, warm-up) are not points. A rally that is only a serve, followed within
   ``FAULT_GAP_S`` by another serve from the same player, was a fault: the point goes on
   with the second serve. Two faults in a row are a double fault.
2. **Who won each point, probably.** From the last shot: into the net or out means its
   hitter lost; in and not returned means its hitter won. Without a measured flight the
   last hitter usually loses (most points end in errors).
3. **Score.** :mod:`tennis.analysis.scoring` turns the servers and these probabilities into
   games and sets that follow the rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlmodel import col, select

from tennis.analysis.kind import RallyInfo, classify_session
from tennis.analysis.scoring import Point, keep_score, point_score_text
from tennis.db import session_scope
from tennis.db.models import Rally, Session, SessionPlayer, Shot

if TYPE_CHECKING:
    from tennis.pipeline.session import SessionContext

FAULT_GAP_S = 25.0
# Without a measured flight: most points end in an error by whoever hit last (in the
# Wingfield reference match 46 of 68 points ended in an error, 10 with a winner).
P_LAST_HITTER_UNKNOWN = 0.25


def _load(ctx: SessionContext) -> tuple[list[Rally], dict[int, list[Shot]], dict[int, int]]:
    with session_scope(ctx.data_root) as db:
        rallies = list(
            db.exec(
                select(Rally).where(Rally.session_id == ctx.session_id).order_by(col(Rally.index))
            )
        )
        shots = list(db.exec(select(Shot).where(Shot.session_id == ctx.session_id)))
        players = list(
            db.exec(select(SessionPlayer).where(SessionPlayer.session_id == ctx.session_id))
        )
    by_rally: dict[int, list[Shot]] = {}
    for s in sorted(shots, key=lambda s: s.t):
        if s.rally_id is not None:
            by_rally.setdefault(s.rally_id, []).append(s)
    index = {sp.player_id: i for i, sp in enumerate(sorted(players, key=lambda p: p.label))}
    return rallies, by_rally, index


def run_kind(ctx: SessionContext) -> None:
    rallies, by_rally, index = _load(ctx)
    infos = []
    for r in rallies:
        shots = by_rally.get(r.id or -1, [])
        if not shots:
            continue
        first = shots[0]
        infos.append(
            RallyInfo(
                start=r.start_s,
                end=r.end_s,
                shots=len(shots),
                first_is_serve=first.stroke == "serve",
                server=index.get(first.player_id or -1) if first.stroke == "serve" else None,
            )
        )
    result = classify_session(infos)
    with session_scope(ctx.data_root) as db:
        session = db.get(Session, ctx.session_id)
        if session is None:
            return
        if session.kind_source != "manual":
            session.kind = result.kind
            session.kind_confidence = (
                result.p_match if result.kind == "match" else 1 - result.p_match
            )
        session.summary = {**session.summary, "kind_evidence": result.evidence}
        db.add(session)
    ctx.log("kind", kind=result.kind, p_match=result.p_match, evidence=result.evidence)


def point_probability(last: Shot, server: int, hitter: int) -> tuple[float, str]:
    """P(server wins) from the rally's last shot, and how the point ended."""
    crosses = (last.quality or {}).get("crosses_net")
    if crosses is False:
        p_hitter, how = 0.12, "net"
    elif last.in_court is False:
        p_hitter, how = 0.2, "out"
    elif last.in_court is True:
        p_hitter, how = 0.8, "ace" if last.stroke == "serve" else "winner"
    else:
        p_hitter, how = P_LAST_HITTER_UNKNOWN, "unknown"
    return (p_hitter if hitter == server else 1 - p_hitter), how


def run_scoring(ctx: SessionContext) -> None:
    with session_scope(ctx.data_root) as db:
        session = db.get(Session, ctx.session_id)
        kind = session.kind if session else "unknown"
    rallies, by_rally, index = _load(ctx)
    players = {i: pid for pid, i in index.items()}
    if kind != "match" or len(players) != 2:
        _clear(ctx)
        return

    # Points: serve-started rallies, with first-serve faults folded into the next rally.
    points: list[dict[str, Any]] = []
    faults: list[Rally] = []
    serve_rallies = []
    for r in rallies:
        shots = by_rally.get(r.id or -1, [])
        if shots and shots[0].stroke == "serve" and index.get(shots[0].player_id or -1) is not None:
            serve_rallies.append((r, shots))
    pending_fault: tuple[Rally, int] | None = None
    for k, (r, shots) in enumerate(serve_rallies):
        server = index[shots[0].player_id or -1]
        nxt = serve_rallies[k + 1] if k + 1 < len(serve_rallies) else None
        only_serve = len(shots) == 1
        next_same_server = (
            nxt is not None
            and index.get(nxt[1][0].player_id or -1) == server
            and nxt[0].start_s - r.end_s < FAULT_GAP_S
        )
        if only_serve and next_same_server and shots[0].in_court is not True:
            if pending_fault is not None:  # second fault in a row: double fault
                points.append(
                    {"rally": r, "server": server, "p": 0.08, "how": "double_fault", "faults": 2}
                )
                faults.append(pending_fault[0])
                pending_fault = None
            else:
                pending_fault = (r, server)
            continue
        last = shots[-1]
        hitter = index.get(last.player_id or -1, server)
        p, how = point_probability(last, server, hitter)
        points.append(
            {"rally": r, "server": server, "p": p, "how": how, "faults": 1 if pending_fault else 0}
        )
        if pending_fault is not None:
            faults.append(pending_fault[0])
        pending_fault = None

    score = keep_score([Point(pt["server"], pt["p"]) for pt in points])
    point_rally_ids = {pt["rally"].id for pt in points}
    fault_ids = {r.id for r in faults}
    won = {0: 0, 1: 0}
    served = {0: 0, 1: 0}
    first_in = {0: 0, 1: 0}
    aces = {0: 0, 1: 0}
    double_faults = {0: 0, 1: 0}
    with session_scope(ctx.data_root) as db:
        # Running score before each point, walking the decoded games.
        before: list[str] = []
        sets: list[tuple[int, int]] = []
        games = [0, 0]
        for game in score.games:
            a = b = 0
            for w in game.points:
                srv = game.server
                pts = (a, b)
                prefix = " ".join(f"{x}-{y}" for x, y in sets)
                before.append(
                    f"{prefix + ' ' if prefix else ''}{games[0]}-{games[1]}, "
                    f"{point_score_text(*pts, tiebreak=game.tiebreak)}"
                )
                if w == srv:
                    a += 1
                else:
                    b += 1
            if game.winner is not None:
                games[game.winner] += 1
                done = (max(games) >= 6 and abs(games[0] - games[1]) >= 2) or max(games) == 7
                if done:
                    sets.append((games[0], games[1]))
                    games = [0, 0]
        for i, pt in enumerate(points):
            row = db.get(Rally, pt["rally"].id)
            if row is None:
                continue
            winner = score.point_winners[i]
            row.server_id = players[pt["server"]]
            row.winner_id = players[winner] if winner is not None else None
            row.end_reason = pt["how"]
            row.score_before = {"text": before[i] if i < len(before) else None, "point": i}
            db.add(row)
            if winner is not None:
                won[winner] += 1
            served[pt["server"]] += 1
            if pt["faults"] == 0:
                first_in[pt["server"]] += 1
            if pt["how"] == "ace" and winner == pt["server"]:
                aces[pt["server"]] += 1
            if pt["how"] == "double_fault":
                double_faults[pt["server"]] += 1
        for other in db.exec(select(Rally).where(Rally.session_id == ctx.session_id)):
            if other.id in point_rally_ids:
                continue
            other.winner_id = None
            other.end_reason = "fault" if other.id in fault_ids else "not_a_point"
            other.score_before = None
            db.add(other)
        for sp in db.exec(select(SessionPlayer).where(SessionPlayer.session_id == ctx.session_id)):
            who = index.get(sp.player_id)
            if who is None:
                continue
            sp.stats = {
                **sp.stats,
                "points_won": won[who],
                "service_points": served[who],
                "first_serve_in_pct": round(first_in[who] / served[who], 3)
                if served[who]
                else None,
                "aces": aces[who],
                "double_faults": double_faults[who],
            }
            db.add(sp)
        session = db.get(Session, ctx.session_id)
        if session is not None:
            text = score.text()
            session.summary = {
                **session.summary,
                "score": {
                    "text": text,
                    "sets": [list(s) for s in score.sets],
                    "current": list(score.current),
                    "players": [players[0], players[1]],
                    "points": len(points),
                },
                "headline": {**session.summary.get("headline", {}), "score": text},
            }
            db.add(session)
    ctx.log("score", text=score.text(), points=len(points), faults=len(faults))


def _clear(ctx: SessionContext) -> None:
    with session_scope(ctx.data_root) as db:
        session = db.get(Session, ctx.session_id)
        if session is not None:
            summary = {k: v for k, v in session.summary.items() if k != "score"}
            headline = {k: v for k, v in summary.get("headline", {}).items() if k != "score"}
            session.summary = {**summary, "headline": headline}
            db.add(session)
