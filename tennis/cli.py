"""Command line: ``tennis serve`` runs the app; the other commands are for scripting and tests."""

from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from tennis.config import Config, load_config
from tennis.errors import TennisError, UserError
from tennis.util.log import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="config.yaml (default: ./config.yaml)")
]
DataRootOpt = Annotated[
    Path | None, typer.Option("--data-root", help="Override paths.data_root from the config")
]


def _load(config: Path | None, data_root: Path | None) -> tuple[Config, Path]:
    cfg = load_config(config)
    root = (data_root or cfg.paths.data_root).expanduser().resolve()
    return cfg, root


@app.command()
def serve(
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    host: Annotated[str, typer.Option(help="Interface to listen on")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8731,
    password_file: Annotated[
        Path | None, typer.Option(help="File holding the login password")
    ] = None,
    secure_cookie: Annotated[bool, typer.Option(help="Behind HTTPS: mark cookies Secure")] = False,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Run the web app and the analysis worker."""
    import ipaddress

    import uvicorn

    from tennis.api import auth
    from tennis.api.app import create_app

    _cfg, root = _load(config, data_root)
    password = auth.read_password(password_file)
    loopback = host == "localhost" or _is_loopback(host, ipaddress)
    if not loopback and password is None:
        raise UserError(
            f"listening on {host} needs a password: set {auth.PASSWORD_ENV} or --password-file"
        )
    if password is not None and len(password) < auth.MIN_PASSWORD_LENGTH:
        raise UserError(f"the password needs at least {auth.MIN_PASSWORD_LENGTH} characters")
    logger = get_logger()
    application = create_app(
        root,
        config_path=config.resolve() if config else None,
        password=password,
        secure_cookie=secure_cookie,
        logger=logger,
    )
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    print(f"tennis  {url}   (data {root}, {'password' if password else 'no password'})")
    if open_browser and loopback:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(application, host=host, port=port, log_level="warning")


def _is_loopback(host: str, ipaddress: object) -> bool:
    try:
        return bool(ipaddress.ip_address(host).is_loopback)  # type: ignore[attr-defined]
    except ValueError:
        return False


@app.command()
def analyze(
    videos: Annotated[list[Path], typer.Argument(help="Video files of one session")],
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    session: Annotated[int | None, typer.Option(help="Add to this session ID")] = None,
    force: Annotated[bool, typer.Option(help="Rerun every stage")] = False,
) -> None:
    """Analyse videos now, in this terminal, without the web app."""
    from sqlmodel import select

    from tennis.db import session_scope
    from tennis.db.models import Session, Video
    from tennis.pipeline import store_video_path
    from tennis.pipeline.runner import analyze_session

    cfg, root = _load(config, data_root)
    paths = [v.expanduser().resolve() for v in videos]
    for p in paths:
        if not p.is_file():
            raise UserError(f"no such file: {p}")
    with session_scope(root) as db:
        if session is None:
            row = Session()
            db.add(row)
            db.flush()
        else:
            found = db.get(Session, session)
            if found is None:
                raise UserError(f"no session {session}")
            row = found
        assert row.id is not None
        session_id = row.id
        known = {v.path for v in db.exec(select(Video).where(Video.session_id == session_id))}
        for p in paths:
            if store_video_path(root, p) not in known:
                db.add(
                    Video(
                        session_id=session_id,
                        filename=p.name,
                        path=store_video_path(root, p),
                        size_bytes=p.stat().st_size,
                    )
                )

    def progress(fraction: float, stage: str, message: str | None) -> None:
        print(f"\r{fraction * 100:5.1f}%  {stage}".ljust(70), end="", file=sys.stderr, flush=True)

    analyze_session(root, cfg, session_id, get_logger(), force=force, on_progress=progress)
    print(file=sys.stderr)
    print(f"session {session_id} ready")


@app.command()
def review(
    video_id: int,
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    start: Annotated[float, typer.Option(help="Seconds from the start of the video")] = 0.0,
    duration: Annotated[float, typer.Option(help="Seconds to render")] = 30.0,
) -> None:
    """Draw the court, players, ball and sound onsets over part of a video."""
    from tennis.db import session_scope
    from tennis.db.models import Video
    from tennis.pipeline import video_source
    from tennis.review import render

    _cfg, root = _load(config, data_root)
    with session_scope(root) as db:
        row = db.get(Video, video_id)
        if row is None:
            raise UserError(f"no video {video_id}")
        source = video_source(root, row.path)
    print(render(root, video_id, source, start, duration))


@app.command("run-job", hidden=True)
def run_job(job_id: int, config: ConfigOpt = None, data_root: DataRootOpt = None) -> None:
    """Run one queued job (the worker starts this in a subprocess)."""
    from tennis.worker import execute_job

    cfg, root = _load(config, data_root)
    execute_job(root, cfg, job_id, get_logger())


def main() -> None:
    try:
        app()
    except TennisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(exc.exit_code)
