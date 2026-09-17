"""Stage 7: word matching, manual overrides and eval-labels, on a synthetic transcript.

No model is ever loaded: the test registers a transcriber that returns a fixed word list.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tennis.config import Config, LabelsConfig, PathsConfig
from tennis.errors import UserError
from tennis.session import Session, create_or_reuse_session
from tennis.stages import STAGES_BY_NAME, StageContext, labels
from tennis.stages.labels import Contact, Word
from tennis.util.log import get_logger

AUDIO_START = 0.5  # WAV time t is at PTS t + 0.5
CONTACT_TIMES = [10.0, 20.0, 30.0]


@pytest.fixture
def transcript() -> Iterator[list[Word]]:
    """Register a transcriber whose word list the test can fill in."""
    words: list[Word] = []

    class Fake:
        name = "fake"

        def transcribe(self, audio: Path, language: str) -> list[Word]:
            assert audio.name == "audio.wav"
            return list(words)

    previous = labels.register_transcriber("fake", lambda cfg, root: Fake())
    try:
        yield words
    finally:
        if previous is None:
            labels._REGISTRY.pop("fake", None)
        else:
            labels.register_transcriber("fake", previous)


def _write_session(tmp_path: Path, data_root: Path, confirmed: list[bool] | None = None) -> Session:
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"x")
    session = create_or_reuse_session(data_root, video, datetime(2026, 9, 20, tzinfo=UTC), "v")
    session.path("audio.wav").write_bytes(b"RIFF")
    (session.dir / "metadata.json").write_text(
        f'{{"video_start_s": 0.0, "audio_start_s": {AUDIO_START}}}'
    )
    flags = confirmed if confirmed is not None else [True] * len(CONTACT_TIMES)
    pq.write_table(
        pa.table(
            {
                "contact_id": np.arange(len(CONTACT_TIMES), dtype=np.int64),
                "t_audio": np.array(CONTACT_TIMES),
                "is_self_audio": [True] * len(CONTACT_TIMES),
            }
        ),
        session.path("contacts.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "swing_id": np.arange(len(CONTACT_TIMES), dtype=np.int64),
                "contact_id": np.arange(len(CONTACT_TIMES), dtype=np.int64),
                "t_contact": np.array(CONTACT_TIMES),
                "is_self_confirmed": flags,
                "qc_pass": [True] * len(CONTACT_TIMES),
            }
        ),
        session.path("swing_info.parquet"),
    )
    return session


def _config(data_root: Path, tmp_path: Path, **label_overrides: object) -> Config:
    return Config(
        paths=PathsConfig(data_root=data_root, labels_dir=tmp_path / "labels"),
        labels=LabelsConfig(backend="fake", **label_overrides),  # type: ignore[arg-type]
    )


def _run(session: Session, config: Config) -> list[dict[str, object]]:
    labels.run(StageContext(session, config, "h", get_logger(), "labels"))
    return pq.read_table(session.path("labels.parquet")).to_pylist()


# --- word normalization and the vocabulary ---------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [("Good!", "good"), (" LATE. ", "late"), ("frame,", "frame"), ("...", ""), ("don't", "dont")],
)
def test_normalize_strips_case_and_punctuation(spoken: str, expected: str) -> None:
    assert labels.normalize(spoken) == expected


def test_vocabulary_maps_every_spelling_to_its_label() -> None:
    vocabulary = labels.build_vocabulary(LabelsConfig())
    assert vocabulary["nice"] == "good" and vocabulary["shank"] == "framed"
    assert vocabulary["out"] == "long"


def test_a_word_in_two_labels_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="mapped to both"):
        LabelsConfig(vocabulary={"good": ["nice"], "late": ["nice"]})


# --- matching ---------------------------------------------------------------------------------


def _contacts() -> list[Contact]:
    return [Contact(i, i, t) for i, t in enumerate(CONTACT_TIMES)]


def _match(words: list[Word], max_delay_s: float = 3.0) -> tuple[list[object], list[object]]:
    vocabulary = labels.build_vocabulary(LabelsConfig())
    rows, dropped = labels.match_words(
        words, _contacts(), vocabulary, max_delay_s, audio_start_s=AUDIO_START
    )
    return list(rows), list(dropped)


def test_a_word_goes_to_the_hit_just_before_it() -> None:
    rows, dropped = _match([Word("Good!", 11.0, 11.3, 0.9)])  # PTS 11.5, 1.5 s after hit 0
    assert not dropped
    assert (rows[0].contact_id, rows[0].label, rows[0].source) == (0, "good", "voice")  # type: ignore[attr-defined]
    assert rows[0].t_word == pytest.approx(11.5)  # type: ignore[attr-defined]
    assert rows[0].confidence == pytest.approx(0.9)  # type: ignore[attr-defined]


def test_a_word_said_too_long_after_the_hit_is_dropped() -> None:
    rows, dropped = _match([Word("late", 15.0, 15.2)])  # PTS 15.5: 5.5 s after hit 0
    assert not rows
    assert "after the last own hit" in dropped[0][1]  # type: ignore[index]


def test_a_word_before_the_first_hit_is_dropped() -> None:
    rows, dropped = _match([Word("good", 1.0, 1.2)])
    assert not rows and dropped[0][1] == "no own hit before it"  # type: ignore[index]


def test_words_outside_the_vocabulary_are_dropped_with_a_reason() -> None:
    rows, dropped = _match([Word("the", 11.0, 11.1), Word("ball", 11.2, 11.4)])
    assert not rows
    assert [d[1] for d in dropped] == ["not in the vocabulary"] * 2  # type: ignore[index]


def test_each_hit_takes_the_words_said_after_it() -> None:
    rows, _ = _match([Word("good", 10.6, 10.8), Word("late", 21.0, 21.2), Word("net", 30.9, 31.1)])
    assert [(r.contact_id, r.label) for r in rows] == [  # type: ignore[attr-defined]
        (0, "good"),
        (1, "late"),
        (2, "net"),
    ]


def test_the_audio_offset_is_applied_to_word_times() -> None:
    # WAV 9.6 s is PTS 10.1, just after hit 0; without the offset it would be before it.
    rows, _ = _match([Word("good", 9.6, 9.8)])
    assert rows and rows[0].contact_id == 0  # type: ignore[attr-defined]


# --- the stage ----------------------------------------------------------------------------------


def test_stage_writes_matched_labels(
    tmp_path: Path, data_root: Path, transcript: list[Word]
) -> None:
    session = _write_session(tmp_path, data_root)
    transcript += [Word("Nice!", 10.8, 11.0, 0.8), Word("late", 21.0, 21.3, 0.7)]
    rows = _run(session, _config(data_root, tmp_path))
    assert [(r["swing_id"], r["label"], r["source"]) for r in rows] == [
        (0, "good", "voice"),
        (1, "late", "voice"),
    ]
    assert labels.labels_by_swing(session) == {0: ["good"], 1: ["late"]}


def test_only_confirmed_own_hits_can_be_labelled(
    tmp_path: Path, data_root: Path, transcript: list[Word]
) -> None:
    session = _write_session(tmp_path, data_root, confirmed=[True, False, True])
    transcript += [Word("late", 21.0, 21.3)]  # said just after the unconfirmed hit 1
    rows = _run(session, _config(data_root, tmp_path))
    assert not rows  # hit 0 is 11 s earlier, well past max_delay_s


def test_manual_labels_override_the_spoken_ones(
    tmp_path: Path, data_root: Path, transcript: list[Word]
) -> None:
    session = _write_session(tmp_path, data_root)
    transcript += [Word("good", 10.8, 11.0), Word("net", 30.8, 31.0)]
    manual = tmp_path / "labels" / f"manual_{session.id}.csv"
    manual.parent.mkdir(parents=True, exist_ok=True)
    manual.write_text("contact_id,label\n0,framed\n1,long\n")
    rows = _run(session, _config(data_root, tmp_path))
    by_contact = {r["contact_id"]: r for r in rows}
    assert by_contact[0]["label"] == "framed" and by_contact[0]["source"] == "manual"
    assert by_contact[1]["label"] == "long" and by_contact[1]["source"] == "manual"
    assert by_contact[2]["label"] == "net" and by_contact[2]["source"] == "voice"


def test_a_manual_label_for_an_unknown_contact_is_an_error(
    tmp_path: Path, data_root: Path, transcript: list[Word]
) -> None:
    session = _write_session(tmp_path, data_root)
    manual = tmp_path / "labels" / f"manual_{session.id}.csv"
    manual.parent.mkdir(parents=True, exist_ok=True)
    manual.write_text("contact_id,label\n99,good\n")
    with pytest.raises(UserError, match="not a confirmed own hit"):
        _run(session, _config(data_root, tmp_path))


def test_no_confirmed_hits_writes_an_empty_file_without_transcribing(
    tmp_path: Path, data_root: Path, transcript: list[Word]
) -> None:
    session = _write_session(tmp_path, data_root, confirmed=[False] * 3)
    transcript += [Word("good", 10.8, 11.0)]
    assert _run(session, _config(data_root, tmp_path)) == []


def test_unknown_backend_is_a_user_error(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    config = Config(paths=PathsConfig(data_root=data_root), labels=LabelsConfig(backend="whisperx"))
    with pytest.raises(UserError, match=r"unknown labels\.backend"):
        labels.run(StageContext(session, config, "h", get_logger(), "labels"))


def test_the_real_backend_is_registered_but_never_loaded_here() -> None:
    assert "faster_whisper" in labels.available_transcribers()


# --- optional inputs --------------------------------------------------------------------------


def test_labels_are_an_optional_input_of_the_clips_and_report_stages() -> None:
    for name in ("clips", "report"):
        stage = STAGES_BY_NAME[name]
        assert "labels.parquet" in stage.optional_inputs
        assert "labels.parquet" not in stage.inputs
        assert "labels.parquet" in stage.all_inputs


def test_an_optional_input_appearing_makes_a_stage_stale(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    session.path("out.txt").write_text("x")
    session.write_stamp(
        "s",
        "h",
        inputs=session.fingerprints(("contacts.parquet", "labels.parquet")),
        outputs=session.fingerprints(("out.txt",)),
    )
    args = (("contacts.parquet",), ("out.txt",), "h")
    assert session.stale_reason("s", *args, optional_inputs=("labels.parquet",)) is None
    session.path("labels.parquet").write_text("now here")
    reason = session.stale_reason("s", *args, optional_inputs=("labels.parquet",))
    assert reason == "optional input appeared: labels.parquet"


# --- eval-labels -------------------------------------------------------------------------------


def test_eval_labels_counts_misses_confusions_and_spurious_labels(
    tmp_path: Path, data_root: Path, transcript: list[Word]
) -> None:
    from tennis.evaluation import evaluate_labels, format_label_eval

    session = _write_session(tmp_path, data_root)
    transcript += [
        Word("good", 10.8, 11.0),  # heard right
        Word("late", 21.0, 21.2),  # heard as "late" but "long" was said: a confusion
        Word("net", 30.6, 30.8),  # nothing was said here: spurious
    ]
    _run(session, _config(data_root, tmp_path))
    # "early" was said 9 s after the only nearby stored label, so nothing matches it.
    spoken = [(11.3, "good"), (21.5, "long"), (40.0, "early")]
    ev = evaluate_labels(session, spoken, tolerance_s=2.0)
    assert ev.spoken == 3 and ev.stored == 3
    assert ev.correct == 1 and ev.confused == 1 and ev.missed == 1
    assert ev.spurious == 1  # the "net" row belongs to no spoken word
    assert ev.match_rate == pytest.approx(1 / 3)
    assert not ev.meets_target
    text = format_label_eval(ev)
    assert "1/3" in text and "good" in text


def test_eval_labels_meets_the_target_when_every_word_matches(
    tmp_path: Path, data_root: Path, transcript: list[Word]
) -> None:
    from tennis.evaluation import evaluate_labels

    session = _write_session(tmp_path, data_root)
    transcript += [Word("good", 10.8, 11.0), Word("late", 21.0, 21.2)]
    _run(session, _config(data_root, tmp_path))
    ev = evaluate_labels(session, [(11.0, "good"), (21.2, "late")])
    assert ev.match_rate == 1.0 and ev.meets_target


def test_eval_labels_needs_the_stage_to_have_run(tmp_path: Path, data_root: Path) -> None:
    from tennis.evaluation import evaluate_labels

    session = _write_session(tmp_path, data_root)
    with pytest.raises(UserError, match=r"no labels\.parquet"):
        evaluate_labels(session, [(1.0, "good")])


def test_read_word_labels(tmp_path: Path) -> None:
    from tennis.validation import read_word_labels

    path = tmp_path / "said.csv"
    path.write_text("# what I said\nt,label\n2:03.5,Good\n12,late\n")
    assert read_word_labels(path) == [(12.0, "late"), (123.5, "good")]


def test_a_manual_label_is_reported_apart_and_never_scored(
    tmp_path: Path, data_root: Path, transcript: list[Word]
) -> None:
    """It is a hand-written override, not something the transcriber produced."""
    from tennis.evaluation import evaluate_labels, format_label_eval

    session = _write_session(tmp_path, data_root)
    transcript += [Word("good", 10.8, 11.0), Word("late", 21.0, 21.2)]
    manual = tmp_path / "labels" / f"manual_{session.id}.csv"
    manual.parent.mkdir(parents=True, exist_ok=True)
    manual.write_text("contact_id,label\n2,framed\n")
    rows = _run(session, _config(data_root, tmp_path))
    assert len(rows) == 3 and sum(1 for r in rows if r["source"] == "manual") == 1

    ev = evaluate_labels(session, [(11.0, "good"), (21.2, "late")])
    assert ev.stored == 3 and ev.manual == 1 and ev.scored == 2
    # The manual row inflates neither the match rate nor the spurious count.
    assert ev.correct == 2 and ev.spurious == 0
    assert ev.match_rate == 1.0
    text = format_label_eval(ev)
    assert "1 written by hand and not scored" in text


def test_read_stroke_labels_honours_the_player_argument(tmp_path: Path) -> None:
    from tennis.validation import read_stroke_labels

    path = tmp_path / "s.csv"
    path.write_text("t,player,stroke\n10,self,forehand\n20,other,serve\n30,partner,volley\n")
    assert read_stroke_labels(path) == [(10.0, "forehand")]
    assert read_stroke_labels(path, "me") == [(10.0, "forehand")]  # a self alias
    assert read_stroke_labels(path, "other") == [(20.0, "serve")]
    assert read_stroke_labels(path, "partner") == [(30.0, "volley")]
    with pytest.raises(UserError, match="no usable nobody labels"):
        read_stroke_labels(path, "nobody")
