"""A local web UI for the pipeline (``tennis ui``).

The server is the standard library's threaded HTTP server: no extra dependency, nothing to
build, and it starts instantly. It does two things — read the session directory to describe
what exists, and run CLI commands as subprocesses through :mod:`tennis.web.jobs`. Every
computation still happens in the CLI, so the browser and the terminal cannot disagree.

It binds to the loopback interface by default and refuses cross-origin writes, because a
server that can start commands must not be reachable from a page the user happens to have
open. On a server (``--host 0.0.0.0``) it also asks for a password: see :mod:`tennis.web.auth`.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import mimetypes
import signal
import threading
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from tennis import __version__
from tennis.config import DEFAULT_CONFIG_NAME, Config, load_config
from tennis.errors import TennisError, UserError
from tennis.session import SESSION_ID_RE, VIDEO_SUFFIXES, Session, list_sessions, sessions_root
from tennis.stages import STAGES, stage_status
from tennis.util.io import is_dataless
from tennis.web import auth as authn
from tennis.web.commands import COMMANDS, COMMANDS_BY_NAME, GROUPS, build_argv
from tennis.web.jobs import JobRunner
from tennis.web.library import (
    PARTIAL_DIR,
    UPLOADS_DIR,
    UploadError,
    Uploads,
    delete_video,
    list_videos,
)

STATIC = Path(__file__).parent / "static"
MAX_BODY_BYTES = 4 * 1024**2  # JSON bodies
LABEL_SUFFIXES = {".csv", ".txt"}
# Reachable without logging in: the login page itself and what it needs to render.
PUBLIC_PATHS = {"/login", "/api/login", "/static/style.css", "/favicon.ico"}
UPLOAD_STATUS = {
    "bad": HTTPStatus.BAD_REQUEST,
    "type": HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
    "size": HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    "space": HTTPStatus.INSUFFICIENT_STORAGE,
    "offset": HTTPStatus.CONFLICT,
    "conflict": HTTPStatus.CONFLICT,
    "missing": HTTPStatus.NOT_FOUND,
}
DEFAULT_CONFIG_TEXT = """# Every key has a default; see config.example.yaml for the full list.
paths:
  data_root: ./data
"""


class WebError(Exception):
    """An HTTP-level failure with the status to send back."""

    def __init__(
        self, status: HTTPStatus, message: str, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra or {}


class App:
    """Everything the request handler needs: where the data is, and what is running."""

    def __init__(
        self,
        config: Config,
        config_path: Path | None,
        project_root: Path,
        auth: authn.Auth | None = None,
    ) -> None:
        self._config = config
        self.config_path = config_path
        self.project_root = project_root
        self.auth = auth
        self.jobs = JobRunner(project_root)
        self._uploads: Uploads | None = None

    @property
    def config(self) -> Config:
        """The config as it is on disk now, so edits in the UI take effect at once."""
        if self.config_path is not None and self.config_path.is_file():
            with contextlib.suppress(TennisError):
                # An invalid edit keeps the last good config; /api/config reports the error.
                self._config = load_config(self.config_path)
        return self._config

    @property
    def writable_config_path(self) -> Path:
        """Where the UI saves: the config in use, or the one it would create beside the data."""
        if self.config_path is not None:
            return self.config_path
        return self.project_root / DEFAULT_CONFIG_NAME

    @property
    def data_root(self) -> Path:
        return self.config.paths.data_root

    @property
    def labels_dir(self) -> Path:
        return self.config.paths.labels_dir

    @property
    def uploads_dir(self) -> Path:
        return self.data_root / UPLOADS_DIR

    @property
    def uploads(self) -> Uploads:
        """The upload tracker for the current data root (the config may move it)."""
        partial_root = self.uploads_dir / PARTIAL_DIR
        if self._uploads is None or self._uploads.partial_root != partial_root:
            self._uploads = Uploads(partial_root)
        return self._uploads

    def session(self, session_id: str) -> Session:
        if not SESSION_ID_RE.match(session_id):
            raise WebError(HTTPStatus.BAD_REQUEST, f"invalid session id {session_id!r}")
        d = sessions_root(self.data_root) / session_id
        if not d.is_dir():
            raise WebError(HTTPStatus.NOT_FOUND, f"unknown session {session_id!r}")
        return Session(id=session_id, dir=d)


# --- reading the session directory -------------------------------------------------------


def _parquet_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return [dict(row) for row in table.to_pylist()]


def _parquet_count(path: Path) -> int:
    if not path.exists():
        return 0
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _json_or_none(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _jsonable(value: Any) -> Any:
    """Make parquet/numpy values safe for :func:`json.dumps` (NaN becomes null)."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, str | int | bool):
        return value
    if hasattr(value, "item"):  # numpy scalar
        return _jsonable(value.item())
    return str(value)


def session_summary(app: App, session: Session, config: Config) -> dict[str, Any]:
    meta = _json_or_none(session.path("metadata.json")) or {}
    stages = {s.name: stage_status(session, s, config) for s in STAGES}
    source = session.source_target()
    return {
        "id": session.id,
        "stages": stages,
        "complete": all(stages[s.name] == "ok" for s in STAGES if s.implemented and not s.optional),
        "duration_s": meta.get("duration_s"),
        "fps": meta.get("fps"),
        "resolution": meta.get("resolution"),
        "creation_time": meta.get("creation_time"),
        "source_path": str(source) if source else None,
        "source_missing": source is not None and not source.exists(),
        "source_offline": source is not None and is_dataless(source),
        "has_report": session.path("report.html").exists(),
        "n_swings": _parquet_count(session.path("metrics.parquet")),
        "modified": session.dir.stat().st_mtime,
    }


def session_detail(app: App, session: Session, config: Config) -> dict[str, Any]:
    meta = _json_or_none(session.path("metadata.json")) or {}
    players = _json_or_none(session.path("players.json")) or {}
    clips_index = _json_or_none(session.path("clips/index.json")) or {}
    stages = []
    for stage in STAGES:
        stamp = session.read_stamp(stage.name) or {}
        stages.append(
            {
                "number": stage.number,
                "name": stage.name,
                "status": stage_status(session, stage, config),
                "optional": stage.optional,
                "implemented": stage.implemented,
                "elapsed_s": stamp.get("elapsed_s"),
                "finished_at": stamp.get("finished_at"),
                "outputs": list(stage.outputs),
            }
        )

    strokes: dict[str, int] = {}
    for row in _parquet_rows(session.path("metrics.parquet"), ["stroke_type"]):
        key = str(row["stroke_type"])
        strokes[key] = strokes.get(key, 0) + 1

    contacts = _parquet_rows(
        session.path("contacts.parquet"), ["is_self_audio", "is_self_confirmed"]
    )
    labels = _parquet_rows(session.path("labels.parquet"))
    label_counts: dict[str, int] = {}
    for row in labels:
        key = str(row.get("label", ""))
        label_counts[key] = label_counts.get(key, 0) + 1

    debug_dir = session.dir / "debug"
    debug = sorted(p.name for p in debug_dir.iterdir()) if debug_dir.is_dir() else []
    source = session.source_target()

    return {
        "id": session.id,
        "dir": str(session.dir),
        # The raw video the session points at: the UI needs it to continue the pipeline,
        # because `tennis process` takes a video rather than a session id.
        "source_path": str(source) if source else None,
        "source_missing": source is None or not source.exists(),
        # Present on disk but evicted to the cloud: reading it would stall, so the UI warns
        # instead of starting a job that hangs.
        "source_offline": source is not None and is_dataless(source),
        "metadata": _jsonable(meta),
        "players": _jsonable(players),
        "stages": stages,
        "counts": {
            "contacts": len(contacts),
            "self_audio": sum(1 for r in contacts if r.get("is_self_audio")),
            "self_confirmed": sum(1 for r in contacts if r.get("is_self_confirmed")),
            "measured": _parquet_count(session.path("metrics.parquet")),
            "clips": sum(1 for c in clips_index.get("clips", []) if c.get("rendered")),
            "labels": len(labels),
        },
        "strokes": strokes,
        "label_counts": label_counts,
        "summary": _jsonable(_parquet_rows(session.path("metrics_summary.parquet"))),
        "has_report": session.path("report.html").exists(),
        "debug_files": debug,
        "log_lines": _tail_log(session.log_path, 200),
    }


def _tail_log(path: Path, count: int) -> list[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return lines[-count:]


def session_swings(app: App, session: Session) -> dict[str, Any]:
    from tennis.stages.metrics import REGISTRY

    rows = _parquet_rows(session.path("metrics.parquet"))
    clips_index = _json_or_none(session.path("clips/index.json")) or {}
    clipped = {
        int(c["swing_id"]): str(c["file"])
        for c in clips_index.get("clips", [])
        if c.get("rendered")
    }
    reasons = {int(c["swing_id"]): list(c.get("reasons", [])) for c in clips_index.get("clips", [])}
    labels_by_swing: dict[int, list[str]] = {}
    for row in _parquet_rows(session.path("labels.parquet")):
        swing = row.get("swing_id")
        if swing is None:
            continue
        labels_by_swing.setdefault(int(swing), []).append(str(row.get("label", "")))

    swings = []
    for row in rows:
        swing_id = int(row["swing_id"])
        swings.append(
            {
                **_jsonable(row),
                "swing_id": swing_id,
                "clip": clipped.get(swing_id),
                "clip_reasons": reasons.get(swing_id, []),
                "labels": labels_by_swing.get(swing_id, []),
            }
        )
    metrics = [
        {"name": m.name, "unit": m.unit, "description": m.description} for m in REGISTRY.values()
    ]
    return {"swings": swings, "metrics": metrics}


# --- the request handler -----------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = f"tennis-analyzer/{__version__}"
    protocol_version = "HTTP/1.1"
    app: App  # set by make_server
    verbose: bool = False
    _body: bytes = b""  # the request body, read up front (not for upload pieces)
    _unread_body: bool = False

    # --- plumbing ----------------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        if self.verbose:
            super().log_message(format, *args)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        # A request body left unread would be taken for the start of the next request on
        # this kept-alive connection. Small bodies are read up front; an upload piece is
        # streamed to disk, and when it is refused unread the connection is closed instead.
        self._body = b""
        self._unread_body = method == "POST" and int(self.headers.get("Content-Length") or 0) > 0
        try:
            if method == "POST":
                if path != "/api/upload":
                    self._body = self._read_body()
                self._check_same_origin()
            if not self._authorized(path):
                if method == "GET" and not path.startswith(("/api/", "/files/")):
                    self._redirect("/login")
                    return
                raise WebError(HTTPStatus.UNAUTHORIZED, "log in first")
            if method == "POST":
                self._route_post(path, query)
            else:
                self._route_get(path, query, head=method == "HEAD")
        except WebError as exc:
            self._send_json({"error": exc.message, **exc.extra}, exc.status)
        except UploadError as exc:
            status = UPLOAD_STATUS.get(exc.kind, HTTPStatus.BAD_REQUEST)
            self._send_json({"error": exc.message, **exc.extra}, status)
        except TennisError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except BrokenPipeError:  # the tab was closed mid-response
            pass
        except Exception as exc:  # never take the server down for one bad request
            self.log_error("%s %s failed: %r", method, path, exc)
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _check_same_origin(self) -> None:
        """Refuse writes from another page: a local server must not be a CSRF target."""
        origin = self.headers.get("Origin")
        if origin is None:
            return  # not a browser fetch (curl, tests): no ambient credentials to abuse
        # Behind a reverse proxy the browser's host arrives as X-Forwarded-Host. Another site
        # cannot set that header on a cross-origin request without a preflight we never grant.
        hosts = {self.headers.get("Host", ""), self.headers.get("X-Forwarded-Host", "")}
        if urlparse(origin).netloc not in hosts - {""}:
            raise WebError(HTTPStatus.FORBIDDEN, f"cross-origin request from {origin} refused")

    def _authorized(self, path: str) -> bool:
        auth = self.app.auth
        if auth is None or path in PUBLIC_PATHS:
            return True
        return auth.verify(authn.cookie_value(self.headers.get("Cookie")))

    @property
    def _https(self) -> bool:
        """Whether the browser reached us over HTTPS (through a TLS-terminating proxy)."""
        return self.headers.get("X-Forwarded-Proto", "").lower() == "https"

    # --- routing -----------------------------------------------------------------------

    def _route_get(self, path: str, query: dict[str, list[str]], head: bool) -> None:
        app = self.app
        if path in {"/", "/index.html"}:
            self._send_file(STATIC / "index.html", head=head, cache=False)
        elif path == "/login":
            if app.auth is None:
                self._redirect("/")
            else:
                self._send_file(STATIC / "login.html", head=head, cache=False)
        elif path.startswith("/static/"):
            self._send_file(_under(STATIC, path[len("/static/") :]), head=head, cache=False)
        elif path == "/api/meta":
            self._send_json(self._meta())
        elif path == "/api/sessions":
            config = app.config
            self._send_json(
                {
                    "sessions": [
                        session_summary(app, s, config) for s in list_sessions(app.data_root)
                    ]
                }
            )
        elif path.startswith("/api/sessions/"):
            rest = path[len("/api/sessions/") :]
            session_id, _, tail = rest.partition("/")
            session = app.session(session_id)
            if tail == "":
                self._send_json(session_detail(app, session, app.config))
            elif tail == "swings":
                self._send_json(session_swings(app, session))
            else:
                raise WebError(HTTPStatus.NOT_FOUND, f"no such endpoint: {path}")
        elif path == "/api/labels":
            self._send_json({"files": self._label_files()})
        elif path == "/api/videos":
            self._send_json(list_videos(app.data_root))
        elif path == "/api/upload":
            self._send_json(app.uploads.status(self._upload_target(query)))
        elif path == "/api/config":
            self._send_json(self._config_payload())
        elif path == "/api/jobs":
            self._send_json({"jobs": [j.as_json() for j in app.jobs.recent()]})
        elif path.startswith("/api/jobs/"):
            job = app.jobs.get(path[len("/api/jobs/") :])
            if job is None:
                raise WebError(HTTPStatus.NOT_FOUND, "no such job")
            since = int((query.get("since") or ["0"])[0])
            lines, next_since = app.jobs.tail(job, since)
            self._send_json({**job.as_json(), "lines": lines, "next_since": next_since})
        elif path.startswith("/files/"):
            self._send_file(self._resolve_file(path[len("/files/") :]), head=head, ranges=True)
        else:
            raise WebError(HTTPStatus.NOT_FOUND, f"no such path: {path}")

    def _route_post(self, path: str, query: dict[str, list[str]]) -> None:
        app = self.app
        if path == "/api/jobs":
            self._send_json(self._start_job(self._read_json()))
        elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path[len("/api/jobs/") : -len("/cancel")]
            if not app.jobs.cancel(job_id):
                raise WebError(HTTPStatus.CONFLICT, "job is not running")
            self._send_json({"ok": True})
        elif path == "/api/upload":
            self._send_json(self._upload(query))
        elif path == "/api/upload/discard":
            self._send_json({"discarded": app.uploads.discard(self._upload_target(query))})
        elif path == "/api/videos/delete":
            self._send_json(delete_video(app.data_root, str(self._read_json().get("name", ""))))
        elif path == "/api/config":
            self._send_json(self._save_config(self._read_json()))
        elif path == "/api/login":
            self._login(self._read_json())
        elif path == "/api/logout":
            self._send_json({"ok": True}, extra={"Set-Cookie": authn.clear_cookie(self._https)})
        else:
            raise WebError(HTTPStatus.NOT_FOUND, f"no such path: {path}")

    # --- endpoint bodies ---------------------------------------------------------------

    def _meta(self) -> dict[str, Any]:
        app = self.app
        return {
            "version": __version__,
            "data_root": str(app.data_root.resolve()),
            "labels_dir": str(app.labels_dir.resolve()),
            "config_path": str(app.config_path) if app.config_path else None,
            "uploads_dir": str(app.uploads_dir),
            "auth": app.auth is not None,
            "video_suffixes": sorted(VIDEO_SUFFIXES),
            "groups": list(GROUPS),
            "commands": [c.as_json() for c in COMMANDS],
            "stages": [
                {
                    "number": s.number,
                    "name": s.name,
                    "milestone": s.milestone,
                    "optional": s.optional,
                    "implemented": s.implemented,
                }
                for s in STAGES
            ],
        }

    def _label_files(self) -> list[dict[str, Any]]:
        directory = self.app.labels_dir
        if not directory.is_dir():
            return []
        return [
            {"name": p.name, "path": str(p), "size": p.stat().st_size}
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() in LABEL_SUFFIXES
        ]

    def _config_payload(self) -> dict[str, Any]:
        """The config as it is, or - when there is none - the example to start from."""
        path = self.app.config_path
        error: str | None = None
        if path is not None and path.is_file():
            text = path.read_text()
            try:
                load_config(path)
            except TennisError as exc:
                error = str(exc)
        else:
            example = self.app.project_root / "config.example.yaml"
            text = example.read_text() if example.is_file() else DEFAULT_CONFIG_TEXT
        return {
            "path": str(path) if path else None,
            "exists": path is not None and path.is_file(),
            "would_create": str(self.app.writable_config_path),
            "text": text,
            "error": error,
        }

    def _save_config(self, body: dict[str, Any]) -> dict[str, Any]:
        path = self.app.writable_config_path
        text = str(body.get("text", ""))
        backup = path.read_text() if path.is_file() else None
        path.write_text(text)
        try:
            load_config(path)
        except TennisError as exc:
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(backup)
            raise WebError(HTTPStatus.BAD_REQUEST, f"{exc} (the file was left unchanged)") from exc
        self.app.config_path = path  # a config created here is the one we use from now on
        return {"ok": True, **self._config_payload()}

    def _start_job(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("command", ""))
        command = COMMANDS_BY_NAME.get(name)
        if command is None:
            raise WebError(HTTPStatus.BAD_REQUEST, f"unknown command {name!r}")
        values = body.get("values")
        if not isinstance(values, dict):
            values = {}
        argv = build_argv(command, values, self.app.config_path)
        session_id = str(values.get("session_id") or "") or None
        job = self.app.jobs.submit(name, argv, command.title, session_id)
        return job.as_json()

    def _upload_target(self, query: dict[str, list[str]]) -> Any:
        """The upload a request is about: ``?kind=video|labels&name=...&size=<bytes>``."""
        kind = (query.get("kind") or ["video"])[0]
        if kind == "video":
            directory, suffixes = self.app.uploads_dir, VIDEO_SUFFIXES
        elif kind == "labels":
            directory, suffixes = self.app.labels_dir, LABEL_SUFFIXES
        else:
            raise WebError(HTTPStatus.BAD_REQUEST, f"unknown upload kind {kind!r}")
        # Without ?size= the body is the whole file, as a small label upload sends it.
        size = (query.get("size") or [self.headers.get("Content-Length") or "0"])[0]
        if not size.isdigit():
            raise WebError(HTTPStatus.BAD_REQUEST, "size must be a number of bytes")
        return self.app.uploads.target(
            directory, (query.get("name") or [""])[0], int(size), suffixes
        )

    def _upload(self, query: dict[str, list[str]]) -> dict[str, Any]:
        """One piece of a file: ``?offset=`` says where it goes. See :mod:`.library`."""
        target = self._upload_target(query)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise WebError(HTTPStatus.LENGTH_REQUIRED, "missing or empty Content-Length")
        offset = (query.get("offset") or ["0"])[0]
        if not offset.isdigit():
            raise WebError(HTTPStatus.BAD_REQUEST, "offset must be a number of bytes")
        result: dict[str, Any] = self.app.uploads.write(target, int(offset), length, self.rfile)
        self._unread_body = False
        return result

    def _login(self, body: dict[str, Any]) -> None:
        auth = self.app.auth
        if auth is None:
            self._send_json({"ok": True})
            return
        if not auth.check_password(str(body.get("password", ""))):
            raise WebError(HTTPStatus.UNAUTHORIZED, "wrong password")
        cookie = authn.set_cookie(auth.issue(), self._https)
        self._send_json({"ok": True}, extra={"Set-Cookie": cookie})

    def _resolve_file(self, rel: str) -> Path:
        """Map ``/files/<root>/<rest>`` onto a directory the UI is allowed to serve."""
        root_name, _, rest = rel.partition("/")
        roots = {
            "data": self.app.data_root,
            "labels": self.app.labels_dir,
            "docs": self.app.project_root / "docs",
        }
        base = roots.get(root_name)
        if base is None:
            raise WebError(HTTPStatus.NOT_FOUND, f"unknown file root {root_name!r}")
        if any(part.startswith(".") for part in Path(rest).parts):
            # Hidden files are internal: the login secret, stamps, unfinished uploads.
            raise WebError(HTTPStatus.NOT_FOUND, "not found")
        return _under(base, rest)

    # --- responses ---------------------------------------------------------------------

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise WebError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
        raw = self.rfile.read(length) if length else b""
        self._unread_body = False
        return raw

    def _read_json(self) -> dict[str, Any]:
        try:
            body = json.loads(self._body or b"{}")
        except json.JSONDecodeError as exc:
            raise WebError(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise WebError(HTTPStatus.BAD_REQUEST, "expected a JSON object")
        return body

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
        extra: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, allow_nan=False, default=str).encode()
        self._send_head(status, "application/json; charset=utf-8", len(body), extra)
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self._send_head(HTTPStatus.SEE_OTHER, "text/plain", 0, {"Location": location})

    def end_headers(self) -> None:
        # Nothing here is meant to be framed by another site or sniffed into another type.
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        super().end_headers()

    def _send_head(
        self,
        status: HTTPStatus,
        content_type: str,
        length: int,
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if self._unread_body:
            self.send_header("Connection", "close")  # see _dispatch
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _send_file(
        self, path: Path, *, head: bool = False, ranges: bool = False, cache: bool = False
    ) -> None:
        if not path.is_file():
            raise WebError(HTTPStatus.NOT_FOUND, f"not found: {path.name}")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        extra = {"Accept-Ranges": "bytes"} if ranges else {}
        if cache:
            extra["Cache-Control"] = "max-age=60"
        header = self.headers.get("Range")
        if ranges and header and header.startswith("bytes="):
            first, _, last = header[len("bytes=") :].partition("-")
            try:
                start = int(first) if first else max(0, size - int(last))
                end = int(last) if first and last else size - 1
            except ValueError:
                start, end = 0, size - 1
            start, end = max(0, start), min(end, size - 1)
            if start > end:
                self._send_head(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    ctype,
                    0,
                    {"Content-Range": f"bytes */{size}"},
                )
                return
            status = HTTPStatus.PARTIAL_CONTENT
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        for key, value in extra.items():
            self.send_header(key, value)
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if head:
            return
        with path.open("rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(256 * 1024, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def _under(base: Path, rel: str) -> Path:
    """``base / rel``, refusing anything that escapes ``base`` (``..``, absolute, symlink)."""
    base = base.resolve()
    candidate = (base / rel.lstrip("/")).resolve()
    if candidate != base and base not in candidate.parents:
        raise WebError(HTTPStatus.FORBIDDEN, "path is outside the served directory")
    return candidate


def make_server(
    config: Config,
    config_path: Path | None,
    project_root: Path,
    host: str,
    port: int,
    verbose: bool = False,
    password: str | None = None,
) -> tuple[ThreadingHTTPServer, App]:
    auth = None
    if password is not None:
        auth = authn.Auth(password, authn.load_secret(config.paths.data_root))
    app = App(config, config_path, project_root, auth)
    handler = type("BoundHandler", (Handler,), {"app": app, "verbose": verbose})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, app


def serve(
    config: Config,
    config_path: Path | None,
    project_root: Path,
    host: str = "127.0.0.1",
    port: int = 8731,
    open_browser: bool = True,
    verbose: bool = False,
    password: str | None = None,
) -> None:
    """Run the UI until interrupted."""
    if not STATIC.is_file() and not (STATIC / "index.html").is_file():
        raise UserError(f"the web UI's files are missing from {STATIC}")
    if password is None and not is_loopback(host):
        raise UserError(
            f"refusing to serve on {host} without a password: anyone who reaches the page "
            f"could run commands on this machine. Set {authn.PASSWORD_ENV} or pass "
            "--password-file (see docs/DEPLOY.md)"
        )
    if password is not None and len(password) < authn.MIN_PASSWORD_LENGTH:
        raise UserError(
            f"the UI password must be at least {authn.MIN_PASSWORD_LENGTH} characters long"
        )
    server, app = make_server(config, config_path, project_root, host, port, verbose, password)
    url = f"http://{host}:{server.server_address[1]}/"
    login = "password required" if password is not None else "no password"
    print(f"tennis ui  {url}   (data root {app.data_root.resolve()}, {login})", flush=True)
    print("press ctrl-c to stop", flush=True)
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    # A service manager stops us with SIGTERM: treat it like ctrl-c, so the running job is
    # stopped too instead of carrying on in its own process group without us.
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        app.jobs.shutdown()
        server.server_close()


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt
