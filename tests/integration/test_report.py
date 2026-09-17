"""Stages 8 and 9: which swings get a clip, and what the report and trends pages contain."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tennis.config import Config, PathsConfig, ReportConfig
from tennis.errors import UserError
from tennis.session import Session, create_or_reuse_session
from tennis.stages import STAGES_BY_NAME, StageContext, clips, metrics, report
from tennis.util.log import get_logger
from tests.conftest import MakeVideo, needs_ffmpeg

METRIC_NAMES = list(metrics.REGISTRY)
# Anything the browser would have to fetch to render the page: a script, a stylesheet or an
# image from somewhere else. (URLs inside the inlined Plotly source are not fetched.)
_EXTERNAL = re.compile(
    r"""<(?:script|link|img)\b[^>]*\b(?:src|href)\s*=\s*["'](?!clips/)([^"']+)"""
)


def _external_references(html: str) -> list[str]:
    return [url for url in _EXTERNAL.findall(html) if not url.startswith("#")]


def _metric_row(swing_id: int, stroke_type: str, base: float, **overrides: object) -> dict:
    row: dict = {
        "swing_id": swing_id,
        "contact_id": swing_id,
        "t_contact": 1.0 + 0.4 * swing_id,
        "stroke_type": stroke_type,
        "two_handed": False,
        "outlier_score": 0.0,
        "is_outlier": False,
    }
    for i, name in enumerate(METRIC_NAMES):
        row[name] = base + 0.1 * i
    row.update(overrides)
    return row


def _write_metrics(session: Session, rows: list[dict]) -> None:
    pq.write_table(
        pa.Table.from_pylist(
            [{k: r.get(k) for k in metrics.metrics_schema().names} for r in rows],
            schema=metrics.metrics_schema(),
        ),
        session.path("metrics.parquet"),
    )
    pq.write_table(metrics.summarize(rows), session.path("metrics_summary.parquet"))


def _write_session(
    tmp_path: Path,
    data_root: Path,
    session_id: str = "2026-09-20_evening",
    rows: list[dict] | None = None,
    labels: list[dict] | None = None,
    video: Path | None = None,
) -> Session:
    source = video or (tmp_path / f"{session_id}.mp4")
    if video is None:
        source.write_bytes(b"x")
    session = create_or_reuse_session(
        data_root, source, datetime(2026, 9, 20, tzinfo=UTC), session_id
    )
    (session.dir / "metadata.json").write_text(
        json.dumps(
            {
                "video_start_s": 0.0,
                "audio_start_s": 0.0,
                "fps": 30.0,
                "duration_s": 600.0,
                "resolution": "1280x720",
                "warnings": ["frame rate 30 fps is below the recommended 100 fps"],
                "video": {"width": 1280, "height": 720},
            }
        )
    )
    (session.dir / "players.json").write_text(
        json.dumps(
            {
                "me": "A",
                "me_reason": "louder hits",
                "identities_resolved": True,
                "handedness": {"hand": "right"},
                "sides": [{"start": 0.0, "end": 600.0, "side": "near", "windows": 12}],
                "windows": [{"window_id": 0, "margin": 0.4, "near": "A"}],
                "hits": {"self": 10, "other": 4, "unknown": 1},
            }
        )
    )
    rows = rows if rows is not None else [
        _metric_row(i, "forehand" if i % 2 == 0 else "backhand", 1.0 + 0.05 * i)
        for i in range(6)
    ]  # fmt: skip
    _write_metrics(session, rows)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "swing_id": r["swing_id"],
                    "contact_id": r["contact_id"],
                    "t_contact": r["t_contact"],
                    "contact_frame_idx": int(r["t_contact"] * 30),
                    "qc_pass": True,
                    "is_self_confirmed": True,
                    "player_side": "near",
                    "qc_reason": "",
                    "swap_count": 1,
                    "track_reset_count": 0,
                }
                for r in rows
            ]
        ),
        session.path("swing_info.parquet"),
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"swing_id": r["swing_id"], "stroke_type": r["stroke_type"], "two_handed": False}
                for r in rows
            ]
        ),
        session.path("strokes.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "contact_id": np.arange(len(rows) + 4, dtype=np.int64),
                "t_audio": np.arange(len(rows) + 4, dtype=np.float64),
                "is_self_audio": [True] * len(rows) + [False] * 4,
            }
        ),
        session.path("contacts.parquet"),
    )
    if labels is not None:
        pq.write_table(pa.Table.from_pylist(labels), session.path("labels.parquet"))
    return session


def _config(data_root: Path, **overrides: object) -> Config:
    return Config(paths=PathsConfig(data_root=data_root), **overrides)  # type: ignore[arg-type]


# --- clip selection ---------------------------------------------------------------------------


def test_outliers_the_median_and_labelled_swings_are_selected() -> None:
    rows = [_metric_row(i, "forehand", 1.0 + i) for i in range(5)]
    rows[4]["is_outlier"] = True
    chosen = clips.select_swings(rows, {1: ["good"], 3: ["late"]}, max_per_label=10)
    picked = {s.swing_id: s.reasons for s in chosen}
    assert 4 in picked and "outlier" in picked[4]
    assert 1 in picked and picked[1] == ["label good"]
    assert 3 in picked and picked[3] == ["label late"]
    median = next(s for s in chosen if any(r.startswith("median") for r in s.reasons))
    assert median.swing_id == 2  # the middle of 1..5
    assert [s.swing_id for s in chosen] == sorted(s.swing_id for s in chosen)


def test_one_swing_picked_twice_is_rendered_once_with_both_reasons() -> None:
    rows = [_metric_row(i, "forehand", 1.0 + i) for i in range(4)]
    rows[3]["is_outlier"] = True
    chosen = clips.select_swings(rows, {3: ["late"]}, max_per_label=10)
    entry = next(s for s in chosen if s.swing_id == 3)
    assert entry.reasons == ["outlier", "label late"]
    assert sum(1 for s in chosen if s.swing_id == 3) == 1


def test_labels_are_capped_per_label() -> None:
    rows = [_metric_row(i, "forehand", 1.0 + i) for i in range(6)]
    chosen = clips.select_swings(rows, {i: ["good"] for i in range(6)}, max_per_label=2)
    labelled = [s for s in chosen if any(r == "label good" for r in s.reasons)]
    assert len(labelled) == 2
    assert [s.swing_id for s in labelled] == [0, 1]  # the earliest two


def test_a_median_is_picked_per_stroke_type() -> None:
    rows = [_metric_row(i, "forehand", 1.0 + i) for i in range(3)]
    rows += [_metric_row(10 + i, "serve", 5.0 + i) for i in range(3)]
    chosen = clips.select_swings(rows, {}, max_per_label=0)
    reasons = {r for s in chosen for r in s.reasons}
    assert reasons == {"median forehand", "median serve"}


def test_clip_caption_lists_the_stroke_reasons_and_metrics() -> None:
    selection = clips.Selection(
        swing_id=3, t_contact=12.0, stroke_type="forehand",
        reasons=["outlier"], labels=["late"], metrics={"peak_wrist_speed": 18.25},
    )  # fmt: skip
    lines = clips.clip_label_lines(selection)
    assert "swing 3" in lines[0] and "forehand" in lines[0] and "late" in lines[0]
    assert lines[1] == "outlier"
    assert "peak_wrist_speed 18.25" in lines[2]


# --- the clips stage --------------------------------------------------------------------------


@needs_ffmpeg
def test_clips_stage_renders_the_selected_swings(
    tmp_path: Path, data_root: Path, make_video: MakeVideo
) -> None:
    from tennis.stages.ingest import run as ingest_run

    video = make_video("clip-report.mp4", seconds=4.0, fps=30)
    session = _write_session(tmp_path, data_root, video=video)
    ingest_run(StageContext(session, _config(data_root), "h", get_logger(), "ingest"))
    rows = [_metric_row(i, "forehand", 1.0 + i) for i in range(3)]
    rows[2]["is_outlier"] = True
    for i, row in enumerate(rows):
        row["t_contact"] = 1.0 + 0.8 * i
    _write_metrics(session, rows)
    # Stage 8 reads the cleaned pixel keypoints; an empty frame set is enough here.
    pq.write_table(
        pa.table(
            {
                "swing_id": np.zeros(0, np.int64),
                "frame_idx": np.zeros(0, np.int64),
                **{
                    f"{n}_{c}": np.zeros(0, np.float32)
                    for n in ("nose", "l_wrist")
                    for c in ("px", "py", "conf")
                },
            }
        ),
        session.path("swings.parquet"),
    )

    config = _config(data_root)
    _write_full_swings(session, rows)
    clips.run(StageContext(session, config, "h", get_logger(), "clips"))

    index = json.loads(session.path(clips.INDEX_NAME).read_text())
    assert index["session"] == session.id
    rendered = [c for c in index["clips"] if c["rendered"]]
    assert rendered, index
    for entry in rendered:
        path = session.dir / "clips" / entry["file"]
        assert path.exists() and path.stat().st_size > 0
        assert entry["frames"] > 0
    # Rerunning drops clips that are no longer selected.
    _write_metrics(session, rows[:1])
    clips.run(StageContext(session, config, "h", get_logger(), "clips"))
    remaining = {p.name for p in (session.dir / "clips").glob("*.mp4")}
    assert remaining <= {"0.mp4"}


def _write_full_swings(session: Session, rows: list[dict]) -> None:
    from tennis.pose_backends.base import KEYPOINT_NAMES

    columns: dict[str, list[float]] = {"swing_id": [], "frame_idx": []}
    for name in KEYPOINT_NAMES:
        for suffix in ("px", "py", "conf"):
            columns[f"{name}_{suffix}"] = []
    for row in rows:
        contact = int(float(row["t_contact"]) * 30)
        for frame in range(max(0, contact - 30), contact + 20):
            columns["swing_id"].append(float(row["swing_id"]))
            columns["frame_idx"].append(float(frame))
            for k, name in enumerate(KEYPOINT_NAMES):
                columns[f"{name}_px"].append(100.0 + k)
                columns[f"{name}_py"].append(120.0 + k)
                columns[f"{name}_conf"].append(0.9)
    table = pa.table(
        {
            "swing_id": pa.array([int(v) for v in columns["swing_id"]], pa.int64()),
            "frame_idx": pa.array([int(v) for v in columns["frame_idx"]], pa.int64()),
            **{k: pa.array(v, pa.float32()) for k, v in columns.items()
               if k not in ("swing_id", "frame_idx")},
        }
    )  # fmt: skip
    pq.write_table(table, session.path("swings.parquet"))


# --- the report stage --------------------------------------------------------------------------


def _build_report(session: Session, data_root: Path, **overrides: object) -> str:
    config = _config(data_root, **overrides)
    report.run(StageContext(session, config, "h", get_logger(), "report"))
    return session.path("report.html").read_text()


def test_report_has_every_section(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    html = _build_report(session, data_root)
    for anchor in ("summary", "metrics", "trends", "distributions", "diagnostics"):
        assert f'id="{anchor}"' in html, anchor
    assert "Session 2026-09-20_evening" in html
    assert "forehand" in html and "backhand" in html
    assert "contact_height" in html


def test_report_opens_offline(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    html = _build_report(session, data_root)
    assert "Plotly" in html
    assert _external_references(html) == []


def test_report_shows_the_diagnostics_the_spec_asks_for(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    html = _build_report(session, data_root)
    assert "below the recommended 100 fps" in html  # ingest warnings
    assert "Confirmed own hits" in html and "Player-track resets" in html
    assert "Identity margin" in html
    assert "First pass (audio level)" in html


def test_report_without_labels_or_clips_leaves_those_sections_out(
    tmp_path: Path, data_root: Path
) -> None:
    session = _write_session(tmp_path, data_root)
    html = _build_report(session, data_root)
    assert 'id="labels"' not in html and 'id="clips"' not in html


def test_report_includes_the_label_analysis_and_the_gallery(
    tmp_path: Path, data_root: Path
) -> None:
    rows = [_metric_row(i, "forehand", 1.0 + i * 0.3) for i in range(6)]
    label_rows = [
        {
            "swing_id": i,
            "contact_id": i,
            "label": "good" if i < 3 else "late",
            "word": "",
            "t_word": 1.0,
            "t_contact": 1.0,
            "confidence": float("nan"),
            "source": "voice",
        }
        for i in range(6)
    ]
    session = _write_session(tmp_path, data_root, rows=rows, labels=label_rows)
    (session.dir / "clips").mkdir(parents=True, exist_ok=True)
    session.path(clips.INDEX_NAME).write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "swing_id": 0,
                        "file": "0.mp4",
                        "rendered": True,
                        "stroke_type": "forehand",
                        "t_contact": 1.0,
                        "reasons": ["outlier"],
                        "labels": ["good"],
                    }
                ]
            }
        )
    )
    html = _build_report(session, data_root)
    assert 'id="labels"' in html and "Cohen" in html
    assert 'id="clips"' in html and 'src="clips/0.mp4"' in html


def test_deltas_are_coloured_only_for_metrics_with_a_declared_direction(
    tmp_path: Path, data_root: Path
) -> None:
    old = _write_session(tmp_path, data_root, session_id="2026-09-01_evening")
    _write_metrics(old, [_metric_row(i, "forehand", 1.0) for i in range(4)])
    new = _write_session(tmp_path, data_root, session_id="2026-09-20_evening")
    _write_metrics(new, [_metric_row(i, "forehand", 2.0) for i in range(4)])

    plain = _build_report(new, data_root)
    assert "2026-09-01_evening" in plain  # named as the baseline
    assert 'class="good"' not in plain and 'class="bad"' not in plain

    coloured = _build_report(
        new,
        data_root,
        report=ReportConfig(metric_direction={"contact_height": "up", "knee_flex_min": "down"}),
    )
    assert 'class="good"' in coloured and 'class="bad"' in coloured


def test_report_needs_metrics(tmp_path: Path, data_root: Path) -> None:
    session = _write_session(tmp_path, data_root)
    session.path("metrics.parquet").unlink()
    with pytest.raises(UserError, match=r"no metrics\.parquet"):
        _build_report(session, data_root)


# --- trends ----------------------------------------------------------------------------------


def test_trends_reads_every_session_with_duckdb(tmp_path: Path, data_root: Path) -> None:
    from tennis.reporting import trend_table, write_trends_report

    _write_session(tmp_path, data_root, session_id="2026-09-01_evening")
    _write_session(tmp_path, data_root, session_id="2026-09-20_evening")
    rows = trend_table(_config(data_root))
    assert {r["session_id"] for r in rows} == {"2026-09-01_evening", "2026-09-20_evening"}
    assert {r["stroke_type"] for r in rows} == {"forehand", "backhand"}
    assert all(r["date"] == r["session_id"][:10] for r in rows)
    assert any(np.isfinite(float(r["contact_height_mean"])) for r in rows)

    out = write_trends_report(_config(data_root), out=tmp_path / "trends.html")
    html = out.read_text()
    assert "2026-09-01_evening" in html and "2026-09-20_evening" in html
    assert _external_references(html) == []


def test_trends_since_filters_by_date(tmp_path: Path, data_root: Path) -> None:
    from tennis.reporting import trend_table

    _write_session(tmp_path, data_root, session_id="2026-09-01_evening")
    _write_session(tmp_path, data_root, session_id="2026-09-20_evening")
    rows = trend_table(_config(data_root), since="2026-09-10")
    assert {r["session_id"] for r in rows} == {"2026-09-20_evening"}


def test_trends_needs_a_session_with_metrics(data_root: Path, tmp_path: Path) -> None:
    from tennis.reporting import write_trends_report

    with pytest.raises(UserError, match=r"no session"):
        write_trends_report(_config(data_root), out=tmp_path / "t.html")


def test_the_duckdb_free_fallback_agrees_with_duckdb(tmp_path: Path, data_root: Path) -> None:
    from tennis.reporting import _trend_table_without_duckdb, trend_table

    _write_session(tmp_path, data_root, session_id="2026-09-01_evening")
    config = _config(data_root)
    fast = {(r["session_id"], r["stroke_type"]): r for r in trend_table(config)}
    slow = {
        (r["session_id"], r["stroke_type"]): r
        for r in _trend_table_without_duckdb(config, since=None)
    }
    assert set(fast) == set(slow)
    for key, row in fast.items():
        assert row["swings"] == slow[key]["swings"]
        assert float(row["contact_height_mean"]) == pytest.approx(
            float(slow[key]["contact_height_mean"])
        )


# --- Cohen's d ---------------------------------------------------------------------------------


def test_cohens_d_on_known_groups() -> None:
    from tennis.reporting import cohens_d

    assert cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)
    assert cohens_d([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert np.isnan(cohens_d([1.0], [2.0, 3.0]))
    assert np.isnan(cohens_d([1.0, 1.0], [1.0, 1.0]))  # no spread at all


# --- the registry ------------------------------------------------------------------------------


def test_stage_8_declares_index_json_not_the_directory() -> None:
    assert STAGES_BY_NAME["clips"].outputs == ("clips/index.json",)


def test_every_stage_is_implemented_now() -> None:
    from tennis.stages import STAGES

    assert [s.name for s in STAGES if not s.implemented] == []


# --- the clips stage without ffmpeg ---------------------------------------------------------


class _FakeReader:
    """Yields one synthetic frame per source frame inside the requested window."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.windows: list[tuple[float, float]] = []

    def read(self, windows: list) -> list:
        from tennis.util.frames import Frame

        out = []
        for window in windows:
            self.windows.append((window.start, window.end))
            start = round(window.start * 30)
            end = round(window.end * 30)
            for index in range(start, end + 1):
                out.append(
                    Frame(
                        index=index,
                        pts=index / 30,
                        image=np.zeros((720, 1280, 3), np.uint8),
                        window_id=window.id,
                    )
                )
        return out


class _FakeWriter:
    written: ClassVar[dict[Path, int]] = {}

    def __init__(self, path: Path, *args: object, **kwargs: object) -> None:
        self.path = path
        self.count = 0

    def __enter__(self) -> _FakeWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"\x00" * max(1, self.count))
        _FakeWriter.written[self.path] = self.count

    def write(self, image: np.ndarray) -> None:
        self.count += 1


@pytest.fixture
def fake_video(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeWriter.written = {}
    monkeypatch.setattr(clips, "FrameReader", _FakeReader)
    monkeypatch.setattr(clips, "VideoWriter", _FakeWriter)


def test_clips_stage_writes_an_index_and_one_file_per_selection(
    tmp_path: Path, data_root: Path, fake_video: None
) -> None:
    rows = [_metric_row(i, "forehand", 1.0 + i) for i in range(3)]
    for i, row in enumerate(rows):
        row["t_contact"] = 2.0 + 2.0 * i
    rows[2]["is_outlier"] = True
    session = _write_session(tmp_path, data_root, rows=rows)
    _write_full_swings(session, rows)
    pq.write_table(
        pa.table({"frame_idx": np.arange(300, dtype=np.int64), "pts": np.arange(300) / 30}),
        session.path("frame_times.parquet"),
    )
    clips.run(StageContext(session, _config(data_root), "h", get_logger(), "clips"))

    index = json.loads(session.path(clips.INDEX_NAME).read_text())
    chosen = {c["swing_id"]: c for c in index["clips"]}
    assert set(chosen) == {1, 2}  # the median forehand and the outlier
    assert chosen[2]["reasons"] == ["outlier"]
    assert chosen[1]["reasons"] == ["median forehand"]
    for entry in index["clips"]:
        assert entry["rendered"] and entry["frames"] > 0
        assert (session.dir / "clips" / entry["file"]).exists()
    # The window is [t - clips.pre_s, t + clips.post_s], clipped to the video.
    assert chosen[1]["start_s"] == pytest.approx(4.0 - 1.2)
    assert chosen[1]["end_s"] == pytest.approx(4.0 + 0.8)


def test_clips_stage_drops_files_that_are_no_longer_selected(
    tmp_path: Path, data_root: Path, fake_video: None
) -> None:
    rows = [_metric_row(i, "forehand", 1.0 + i) for i in range(3)]
    session = _write_session(tmp_path, data_root, rows=rows)
    _write_full_swings(session, rows)
    pq.write_table(
        pa.table({"frame_idx": np.arange(300, dtype=np.int64), "pts": np.arange(300) / 30}),
        session.path("frame_times.parquet"),
    )
    stale = session.dir / "clips" / "99.mp4"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")
    clips.run(StageContext(session, _config(data_root), "h", get_logger(), "clips"))
    assert not stale.exists()


def test_slow_motion_lowers_the_clip_frame_rate(
    tmp_path: Path, data_root: Path, fake_video: None
) -> None:
    from tennis.config import ClipsConfig

    rows = [_metric_row(0, "forehand", 1.0)]
    session = _write_session(tmp_path, data_root, rows=rows)
    _write_full_swings(session, rows)
    pq.write_table(
        pa.table({"frame_idx": np.arange(300, dtype=np.int64), "pts": np.arange(300) / 30}),
        session.path("frame_times.parquet"),
    )
    config = _config(data_root, clips=ClipsConfig(slow_motion=True, slow_motion_rate=0.25))
    clips.run(StageContext(session, config, "h", get_logger(), "clips"))
    index = json.loads(session.path(clips.INDEX_NAME).read_text())
    assert index["fps"] == pytest.approx(7.5) and index["slow_motion"] is True
