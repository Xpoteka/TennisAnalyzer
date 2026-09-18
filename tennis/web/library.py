"""The video library: every video kept on the server, and the uploads that put it there.

Uploaded videos stay in ``<data_root>/uploads/`` for good, so any session can be rerun later
without sending the file again. A file copied into that folder by other means (a network
share, ``rsync``) is part of the library too: the folder is the library.

Uploads arrive in pieces (:data:`CHUNK_LIMIT` at most each) and are appended to a partial
file under ``uploads/.partial/``. Pieces keep every request under the body limits of reverse
proxies and tunnels, and an upload cut short resumes where it stopped: the page asks how much
arrived and sends the rest. The partial file is keyed by the file's name and size, so
dropping the same file again resumes it, and it is renamed into the library once complete.
"""

from __future__ import annotations

import re
import shutil
import threading
from dataclasses import dataclass
from io import BufferedIOBase
from pathlib import Path
from typing import Any, BinaryIO

from tennis.session import VIDEO_SUFFIXES, list_sessions

UPLOADS_DIR = "uploads"
PARTIAL_DIR = ".partial"
CHUNK_LIMIT = 64 * 1024**2  # per request; the page sends 32 MiB
MAX_UPLOAD_BYTES = 64 * 1024**3  # a long 4K session, not a typo'd header
FREE_SPACE_MARGIN = 1024**3  # keep this much free after an upload
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(Exception):
    """A refused upload request; ``kind`` picks the HTTP status."""

    def __init__(self, kind: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.kind = kind  # bad | type | size | space | offset | conflict
        self.message = message
        self.extra = extra


def safe_name(raw: str) -> str:
    """A file name that cannot leave its folder or hide itself."""
    return SAFE_NAME.sub("_", Path(raw).name).lstrip(".")


@dataclass(frozen=True)
class UploadTarget:
    """Where one upload is going: its folder, final name, and expected size."""

    directory: Path
    name: str
    size: int
    partial_root: Path

    @property
    def final(self) -> Path:
        return self.directory / self.name

    @property
    def partial(self) -> Path:
        return self.partial_root / f"{self.name}.{self.size}.part"


class Uploads:
    """Resumable, chunked uploads. Safe to call from many request threads."""

    def __init__(self, partial_root: Path) -> None:
        self.partial_root = partial_root
        self._locks: dict[Path, threading.Lock] = {}
        self._guard = threading.Lock()

    def target(self, directory: Path, raw_name: str, size: int, suffixes: set[str]) -> UploadTarget:
        name = safe_name(raw_name)
        if not name:
            raise UploadError("bad", "missing file name")
        suffix = Path(name).suffix.lower()
        if suffix not in suffixes:
            accepted = ", ".join(sorted(suffixes))
            raise UploadError("type", f"'{suffix or name}' is not accepted: expected {accepted}")
        if size <= 0:
            raise UploadError("bad", "missing or empty file size")
        if size > MAX_UPLOAD_BYTES:
            raise UploadError("size", "file is too large")
        return UploadTarget(directory, name, size, self.partial_root)

    def status(self, target: UploadTarget) -> dict[str, Any]:
        """How far this upload got. A same-named file of the same size is already done."""
        with self._lock(target):
            if target.final.is_file() and target.final.stat().st_size == target.size:
                return self._done(target.final, reused=True)
            received = target.partial.stat().st_size if target.partial.is_file() else 0
            if received == 0:
                self._check_space(target)
            return {"done": False, "received": received, "size": target.size}

    def write(
        self, target: UploadTarget, offset: int, length: int, body: BufferedIOBase
    ) -> dict[str, Any]:
        """Append one piece at ``offset``; finish the upload when the last piece is in."""
        if length <= 0:
            raise UploadError("bad", "empty piece")
        if length > CHUNK_LIMIT:
            raise UploadError("size", f"a piece may be at most {CHUNK_LIMIT // 1024**2} MiB")
        if offset < 0 or offset + length > target.size:
            raise UploadError("bad", "piece lies outside the file")
        with self._lock(target):
            if target.final.is_file() and target.final.stat().st_size == target.size:
                _drain(body, length)
                return self._done(target.final, reused=True)
            received = target.partial.stat().st_size if target.partial.is_file() else 0
            if offset != received:
                # A retried piece, or two tabs sending the same file: say where to go on.
                _drain(body, length)
                raise UploadError(
                    "offset", f"expected the piece at byte {received}", received=received
                )
            if offset == 0:
                self._check_space(target)
            target.partial.parent.mkdir(parents=True, exist_ok=True)
            with target.partial.open("ab") as fh:
                written = _copy(body, fh, length)
            if written != length:
                # Keep what arrived whole; the page resumes from the size on disk.
                with target.partial.open("r+b") as fh:
                    fh.truncate(offset)
                raise UploadError("bad", "the piece was cut short", received=offset)
            if offset + length < target.size:
                return {"done": False, "received": offset + length, "size": target.size}
            target.directory.mkdir(parents=True, exist_ok=True)
            final = _free_path(target.final)
            # A move, not a rename: the labels folder may be on another disk.
            shutil.move(target.partial, final)
            return self._done(final, reused=False)

    def discard(self, target: UploadTarget) -> bool:
        with self._lock(target):
            if target.partial.is_file():
                target.partial.unlink()
                return True
            return False

    def _check_space(self, target: UploadTarget) -> None:
        target.partial_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target.partial_root).free
        if target.size + FREE_SPACE_MARGIN > free:
            raise UploadError(
                "space",
                f"not enough disk space: the file needs {_gib(target.size)}, "
                f"{_gib(free)} is free on the server",
            )

    def _lock(self, target: UploadTarget) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(target.partial, threading.Lock())

    @staticmethod
    def _done(path: Path, reused: bool) -> dict[str, Any]:
        return {
            "done": True,
            "path": str(path.resolve()),
            "name": path.name,
            "reused": reused,
            "received": path.stat().st_size,
            "size": path.stat().st_size,
        }


def _copy(src: BufferedIOBase, dst: BinaryIO, length: int) -> int:
    written = 0
    while written < length:
        chunk = src.read(min(1024 * 1024, length - written))
        if not chunk:
            break
        dst.write(chunk)
        written += len(chunk)
    return written


def _drain(src: BufferedIOBase, length: int) -> None:
    left = length
    while left > 0:
        chunk = src.read(min(1024 * 1024, left))
        if not chunk:
            return
        left -= len(chunk)


def _free_path(path: Path) -> Path:
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise UploadError("conflict", f"too many files named {path.stem}*")


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GB"


# --- the library ---------------------------------------------------------------------------


def list_videos(data_root: Path) -> dict[str, Any]:
    """Every video in the library, which sessions use it, and uploads still in progress."""
    uploads = data_root / UPLOADS_DIR
    used_by: dict[Path, list[str]] = {}
    for session in list_sessions(data_root):
        target = session.source_target()
        if target is not None:
            used_by.setdefault(target, []).append(session.id)

    videos = []
    partial = []
    if uploads.is_dir():
        for p in sorted(uploads.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not p.is_file() or p.suffix.lower() not in VIDEO_SUFFIXES or p.name.startswith("."):
                continue
            st = p.stat()
            videos.append(
                {
                    "name": p.name,
                    "path": str(p.resolve()),
                    "size": st.st_size,
                    "modified": st.st_mtime,
                    "sessions": sorted(used_by.get(p.resolve(), [])),
                    "url": f"/files/data/{UPLOADS_DIR}/{p.name}",
                }
            )
        partial_dir = uploads / PARTIAL_DIR
        if partial_dir.is_dir():
            for p in sorted(partial_dir.glob("*.part")):
                name, _, size = p.name[: -len(".part")].rpartition(".")
                if not size.isdigit():
                    continue
                partial.append(
                    {
                        "name": name,
                        "size": int(size),
                        "received": p.stat().st_size,
                        "modified": p.stat().st_mtime,
                    }
                )

    disk_dir = uploads if uploads.is_dir() else data_root if data_root.is_dir() else Path(".")
    usage = shutil.disk_usage(disk_dir)
    return {
        "dir": str(uploads.resolve()),
        "videos": videos,
        "partial": partial,
        "disk": {"free": usage.free, "total": usage.total},
    }


def delete_video(data_root: Path, raw_name: str) -> dict[str, Any]:
    """Remove one video from the library. Its sessions keep their results."""
    name = safe_name(raw_name)
    path = data_root / UPLOADS_DIR / name
    if not name or not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
        raise UploadError("missing", f"no video named {raw_name!r} in the library")
    resolved = path.resolve()
    sessions = sorted(s.id for s in list_sessions(data_root) if s.source_target() == resolved)
    path.unlink()
    return {"deleted": name, "sessions": sessions}
