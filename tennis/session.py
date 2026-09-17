"""Session directories, session IDs, and stage caching (spec sections 4 and 5.2)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tennis import __version__
from tennis.errors import UserError
from tennis.util.io import read_json, write_json

SESSIONS_DIR = "sessions"
SOURCE_STEM = "source"
STAMPS_DIR = ".stamps"
LOG_NAME = "pipeline.log"
SOURCE_INPUT = "@source"  # stage input token for the linked raw video

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def time_of_day(ts: datetime) -> str:
    hour = ts.hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def default_session_id(created: datetime) -> str:
    """``YYYY-MM-DD_<time of day>`` in local time, e.g. ``2026-09-20_evening``."""
    local = created.astimezone()
    return f"{local:%Y-%m-%d}_{time_of_day(local)}"


def validate_session_id(session_id: str) -> str:
    if not SESSION_ID_RE.match(session_id):
        raise UserError(
            f"invalid session id '{session_id}': use letters, digits, '_', '-' or '.', "
            "starting with a letter or digit"
        )
    return session_id


@dataclass(frozen=True)
class Session:
    id: str
    dir: Path

    @property
    def log_path(self) -> Path:
        return self.dir / LOG_NAME

    def path(self, name: str) -> Path:
        """Resolve a stage input/output name to a path inside the session."""
        if name == SOURCE_INPUT:
            return self.source_link
        return self.dir / name

    @property
    def source_link(self) -> Path:
        for p in sorted(self.dir.glob(f"{SOURCE_STEM}.*")):
            if p.is_symlink() or p.is_file():
                return p
        raise UserError(f"session '{self.id}' has no {SOURCE_STEM}.* link in {self.dir}")

    def source_target(self) -> Path | None:
        """The raw video the session points to, or None if the link is missing."""
        try:
            return self.source_link.resolve()
        except UserError:
            return None

    # --- stage stamps -------------------------------------------------------------------

    def stamp_path(self, stage: str) -> Path:
        return self.dir / STAMPS_DIR / f"{stage}.json"

    def read_stamp(self, stage: str) -> dict[str, Any] | None:
        path = self.stamp_path(stage)
        if not path.is_file():
            return None
        data = read_json(path)
        return data if isinstance(data, dict) else None

    def fingerprints(self, names: tuple[str, ...]) -> dict[str, list[int] | None]:
        """(mtime_ns, size) of each named file; None if it does not exist."""
        out: dict[str, list[int] | None] = {}
        for name in names:
            try:
                st = self.path(name).stat()  # follows the source symlink
            except FileNotFoundError:
                out[name] = None
                continue
            out[name] = [st.st_mtime_ns, st.st_size]
        return out

    def write_stamp(
        self,
        stage: str,
        config_hash: str,
        inputs: dict[str, list[int] | None] | None = None,
        outputs: dict[str, list[int] | None] | None = None,
        **extra: Any,
    ) -> None:
        stamp: dict[str, Any] = {
            "stage": stage,
            "pipeline_version": __version__,
            "config_hash": config_hash,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **extra,
        }
        if inputs is not None:
            stamp["inputs"] = inputs
        if outputs is not None:
            stamp["outputs"] = outputs
        write_json(self.stamp_path(stage), stamp)

    def clear_stamp(self, stage: str) -> None:
        self.stamp_path(stage).unlink(missing_ok=True)

    def stale_reason(
        self,
        stage: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        config_hash: str,
        optional_inputs: tuple[str, ...] = (),
    ) -> str | None:
        """Why ``stage`` must run, or None when its outputs are up to date.

        Up to date means: the stage finished before (stamp present), with the same pipeline
        version and the same hash of the config values it depends on, all outputs exist,
        and no input or output changed since that run. "Changed" compares each file's
        modification time and size with the ones recorded in the stamp. Stamps written
        before fingerprints existed fall back to "no input newer than the oldest output".
        A config hash is used instead of the config file's mtime so that editing unrelated
        options does not trigger reruns.

        ``optional_inputs`` are files the stage uses when they exist. They may be missing,
        but appearing, changing or disappearing still makes the stage stale, so producing
        the optional voice labels rebuilds what reads them.
        """
        stamp = self.read_stamp(stage)
        if stamp is None:
            return "not run yet"
        if stamp.get("pipeline_version") != __version__:
            return f"pipeline version changed ({stamp.get('pipeline_version')} -> {__version__})"
        if stamp.get("config_hash") != config_hash:
            return "config changed"
        for name in outputs:
            if not self.path(name).exists():
                return f"output missing: {name}"
        recorded_in = stamp.get("inputs")
        recorded_out = stamp.get("outputs")
        if isinstance(recorded_in, dict) and isinstance(recorded_out, dict):
            current_in = self.fingerprints(inputs + optional_inputs)
            for name in inputs:
                if current_in[name] is None:
                    return f"input missing: {name}"
                if recorded_in.get(name) != current_in[name]:
                    return f"input changed: {name}"
            for name in optional_inputs:
                if recorded_in.get(name) != current_in[name]:
                    kind = "appeared" if recorded_in.get(name) is None else "changed"
                    if current_in[name] is None:
                        kind = "disappeared"
                    return f"optional input {kind}: {name}"
            current_out = self.fingerprints(outputs)
            for name in outputs:
                if recorded_out.get(name) != current_out[name]:
                    return f"output changed: {name}"
            return None
        output_mtimes = [self.path(name).stat().st_mtime_ns for name in outputs]
        for name in inputs:
            p = self.path(name)
            if not p.exists():  # follows symlinks: a dangling source link counts as missing
                return f"input missing: {name}"
            if output_mtimes and p.stat().st_mtime_ns > min(output_mtimes):
                return f"input newer than outputs: {name}"
        return None


def sessions_root(data_root: Path) -> Path:
    return data_root / SESSIONS_DIR


def open_session(data_root: Path, session_id: str) -> Session:
    validate_session_id(session_id)
    d = sessions_root(data_root) / session_id
    if not d.is_dir():
        raise UserError(f"unknown session '{session_id}' (looked in {d.parent})")
    return Session(id=session_id, dir=d)


def list_sessions(data_root: Path) -> list[Session]:
    root = sessions_root(data_root)
    if not root.is_dir():
        return []
    return [
        Session(id=d.name, dir=d)
        for d in sorted(root.iterdir())
        if d.is_dir() and SESSION_ID_RE.match(d.name)
    ]


def check_video_file(video: Path) -> Path:
    if not video.exists():
        raise UserError(f"video not found: {video}")
    if not video.is_file():
        raise UserError(f"not a file: {video}")
    if video.suffix.lower() not in VIDEO_SUFFIXES:
        raise UserError(
            f"unsupported file type '{video.suffix}': expected one of "
            + ", ".join(sorted(VIDEO_SUFFIXES))
        )
    if not os.access(video, os.R_OK):
        raise UserError(f"video is not readable: {video}")
    return video.resolve()


def create_or_reuse_session(
    data_root: Path, video: Path, created: datetime, session_id: str | None = None
) -> Session:
    """Find or create the session for ``video`` and link the raw file into it.

    The raw video is never copied or modified; the session holds a symlink to its absolute
    path. Re-processing the same video reuses its session. A derived ID that is already
    taken by a different video gets a numeric suffix (``_2``, ``_3``, ...); an explicit
    ``--session-id`` that is taken by a different video is an error.
    """
    video = video.resolve()
    root = sessions_root(data_root)

    if session_id is not None:
        candidates = [validate_session_id(session_id)]
    else:
        # Same video processed before under any id? Reuse that session.
        for existing in list_sessions(data_root):
            if existing.source_target() == video:
                return existing
        base = default_session_id(created)
        candidates = [base] + [f"{base}_{n}" for n in range(2, 100)]

    for sid in candidates:
        d = root / sid
        session = Session(id=sid, dir=d)
        if not d.exists():
            d.mkdir(parents=True)
            (d / f"{SOURCE_STEM}{video.suffix.lower()}").symlink_to(video)
            return session
        target = session.source_target()
        if target == video:
            return session
        if target is None and not any(d.iterdir()):
            (d / f"{SOURCE_STEM}{video.suffix.lower()}").symlink_to(video)
            return session
        if session_id is not None:
            raise UserError(
                f"session '{sid}' already exists for a different video ({target}); "
                "choose another --session-id"
            )
    raise UserError(f"too many sessions named {candidates[0]}*; pass --session-id")
