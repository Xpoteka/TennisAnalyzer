"""Hit sequence selection and stroke classification on synthetic inputs."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tennis.vision.hits import Option, choose_hits
from tennis.vision.strokes import KP, Swing, classify, racket_hand


def opt(t: float, side: int, score: float) -> Option:
    return Option(
        t=t, side=side, score=score, track=0 if side < 0 else 1, features={}, sources=set()
    )


def test_rally_alternates_and_drops_bounces() -> None:
    # A rally: near, far, near, far hits 1.3 s apart. Each is followed by a loud bounce, too
    # soon after the hit to be the reply, and by a weak sound on the hitter's own side.
    options = []
    for k in range(4):
        side = -1 if k % 2 == 0 else 1
        options.append(opt(10 + 1.3 * k, side, 0.8))
        options.append(opt(10 + 1.3 * k + 0.3, -side, 0.6))  # the bounce
        options.append(opt(10 + 1.3 * k + 0.8, side, 0.5))  # same side again
    hits = choose_hits(options)
    assert [round(h.t, 1) for h in hits] == [10.0, 11.3, 12.6, 13.9]
    assert [h.side for h in hits] == [-1, 1, -1, 1]


def test_weak_hit_kept_when_it_keeps_the_rally_consistent() -> None:
    options = [opt(0.0, -1, 0.9), opt(1.2, 1, 0.38), opt(2.4, -1, 0.9), opt(3.6, 1, 0.9)]
    hits = choose_hits(options)
    assert [h.t for h in hits] == [0.0, 1.2, 2.4, 3.6]


def test_new_rally_can_start_on_either_side() -> None:
    options = [opt(0.0, -1, 0.9), opt(1.3, 1, 0.9), opt(20.0, 1, 0.9), opt(21.3, -1, 0.9)]
    assert [h.side for h in choose_hits(options)] == [-1, 1, 1, -1]


def test_nothing_strong_gives_no_hits() -> None:
    assert choose_hits([opt(0.0, -1, 0.3), opt(5.0, 1, 0.2)]) == []


def swing(
    wrist_dx: float, *, overhead: bool = False, rise: float = 0.0, racket: str = "r_wrist"
) -> Swing:
    """A standing player about 200 px tall seen from behind. The racket wrist swings across
    the body: from ``wrist_dx`` pixels right of the centre to ``wrist_dx`` pixels left of it
    (a forehand for a right-hander when positive). Overheads end above the head."""
    t = np.arange(-0.7, 0.36, 1 / 30)
    kp = np.zeros((len(t), 17, 3))
    kp[..., 2] = 0.9
    base = {
        "nose": (500, 300), "l_eye": (495, 297), "r_eye": (505, 297),
        "l_ear": (490, 300), "r_ear": (510, 300),
        "l_shoulder": (480, 330), "r_shoulder": (520, 330),
        "l_elbow": (470, 370), "r_elbow": (530, 370),
        "l_wrist": (470, 400), "r_wrist": (530, 400),
        "l_hip": (485, 410), "r_hip": (515, 410),
        "l_knee": (485, 450), "r_knee": (515, 450),
        "l_ankle": (485, 490), "r_ankle": (515, 490),
    }  # fmt: skip
    for name, (x, y) in base.items():
        kp[:, KP[name], 0] = x
        kp[:, KP[name], 1] = y
    s = np.clip((t + 0.4) / 0.7, 0, 1)  # the arm swings from 0.4 s before to 0.3 s after contact
    end_y = 250 if overhead else 400 - rise
    start_y = 400 if overhead else 400 + rise
    kp[:, KP[racket], 0] = 500 + wrist_dx - s * (2 * wrist_dx)
    kp[:, KP[racket], 1] = start_y + np.clip((t + 0.4) / 0.4, 0, 1) * (end_y - start_y)
    return Swing(t, kp)


RIGHT: npt.NDArray[np.float64] = np.array([1.0, 0.0])  # near player: their right is image right


def test_forehand_and_backhand_for_a_right_hander() -> None:
    fh = classify(
        swing(60), hand="right", right_dir=RIGHT, first_in_rally=False, distance_from_net_m=11
    )
    bh = classify(
        swing(-60), hand="right", right_dir=RIGHT, first_in_rally=False, distance_from_net_m=11
    )
    assert fh.stroke == "forehand"
    assert bh.stroke == "backhand"


def test_left_hander_is_mirrored() -> None:
    fh = classify(
        swing(-60, racket="l_wrist"),
        hand="left",
        right_dir=RIGHT,
        first_in_rally=False,
        distance_from_net_m=11,
    )
    assert fh.stroke == "forehand"


def test_far_player_faces_the_camera() -> None:
    left = np.array([-1.0, 0.0])  # a far player's right is the image left
    call = classify(
        swing(-60), hand="right", right_dir=left, first_in_rally=False, distance_from_net_m=11
    )
    assert call.stroke == "forehand"


def test_serve_overhead_and_volley() -> None:
    serve = classify(
        swing(10, overhead=True),
        hand="right",
        right_dir=RIGHT,
        first_in_rally=True,
        distance_from_net_m=12,
    )
    smash = classify(
        swing(10, overhead=True),
        hand="right",
        right_dir=RIGHT,
        first_in_rally=False,
        distance_from_net_m=5,
    )
    serve_from_the_back = classify(
        swing(10, overhead=True),
        hand="right",
        right_dir=RIGHT,
        first_in_rally=False,
        distance_from_net_m=12,
    )
    assert serve_from_the_back.stroke == "serve"
    volley = classify(
        swing(60), hand="right", right_dir=RIGHT, first_in_rally=False, distance_from_net_m=3
    )
    assert serve.stroke == "serve"
    assert smash.stroke == "overhead"
    assert volley.stroke == "volley_forehand"


def test_spin_from_the_wrist_path() -> None:
    top = classify(
        swing(60, rise=60),
        hand="right",
        right_dir=RIGHT,
        first_in_rally=False,
        distance_from_net_m=11,
    )
    cut = classify(
        swing(60, rise=-60),
        hand="right",
        right_dir=RIGHT,
        first_in_rally=False,
        distance_from_net_m=11,
    )
    assert top.spin == "topspin"
    assert cut.spin == "slice"


def test_racket_hand_from_serves() -> None:
    assert racket_hand([swing(10, overhead=True) for _ in range(8)]) == "right"
    lefty = [swing(-10, overhead=True, racket="l_wrist") for _ in range(8)]
    assert racket_hand(lefty) == "left"
    assert racket_hand([swing(10, overhead=True)]) is None  # too little evidence
    assert racket_hand([swing(60) for _ in range(8)]) is None  # no overheads, no vote


def test_a_serve_from_inside_the_court_mid_rally_is_a_smash() -> None:
    from tennis.pipeline.session.shots import not_a_serve_mid_rally
    from tennis.vision.strokes import StrokeCall

    serve = StrokeCall("serve", None, 0.8, 0.0)
    behind, inside = (0.5, -12.0), (1.0, -7.0)
    assert not_a_serve_mid_rally(serve, True, inside).stroke == "serve"  # first hit: a serve
    assert not_a_serve_mid_rally(serve, False, behind).stroke == "serve"  # restarts the rally
    assert not_a_serve_mid_rally(serve, False, inside).stroke == "overhead"
    assert not_a_serve_mid_rally(serve, False, None).stroke == "serve"
    forehand = StrokeCall("forehand", "topspin", 0.7, 0.3)
    assert not_a_serve_mid_rally(forehand, False, inside) is forehand
