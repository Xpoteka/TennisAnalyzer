"""Rallies, serves and faults from the shots of a match."""

from __future__ import annotations

from tennis.db.models import Shot
from tennis.pipeline.session.match import was_fault
from tennis.pipeline.session.shots import continues_rally, serve_position


def test_one_missed_hit_does_not_end_the_rally() -> None:
    assert continues_rally(2.0, -1, 1)
    assert continues_rally(4.5, -1, -1)  # the far player's hit in between was missed
    assert not continues_rally(4.5, -1, 1)  # a ball hit back after the point
    assert not continues_rally(7.0, -1, -1)


def test_serve_from_the_baseline_after_a_pause() -> None:
    assert serve_position((1.2, -12.1), 12.0)
    assert not serve_position((1.2, -12.1), 2.0)  # mid-rally
    assert not serve_position((1.2, -6.0), 12.0)  # from the service line
    assert not serve_position((6.0, -12.1), 12.0)  # from the corner: a ball hit back
    assert not serve_position(None, 12.0)


def serve(x: float, in_court: bool | None = None) -> Shot:
    return Shot(
        session_id=1, video_id=1, t=0.0, stroke="serve", hit_x=x, hit_y=-12.0, in_court=in_court
    )


def rally(x: float, n: int) -> list[Shot]:
    return [serve(x)] + [
        Shot(session_id=1, video_id=1, t=float(k), stroke="forehand") for k in range(1, n)
    ]


def test_fault_is_served_again_from_the_same_box() -> None:
    assert was_fault(rally(1.0, 1), rally(1.2, 1))
    assert was_fault(rally(1.0, 2), rally(0.8, 4))  # the receiver hit the fault back
    assert not was_fault(rally(1.0, 1), rally(-1.0, 1))  # next point, other box
    assert not was_fault(rally(1.0, 4), rally(1.0, 1))  # a real rally was played
    assert not was_fault([serve(1.0, in_court=True)], rally(1.0, 1))  # it was in
    assert not was_fault(rally(1.0, 1), None)
    # Without a clear box, only a lone serve counts as a fault.
    assert was_fault(rally(0.0, 1), rally(0.0, 1))
    assert not was_fault(rally(0.0, 2), rally(0.0, 1))
