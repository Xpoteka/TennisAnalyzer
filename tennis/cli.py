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


@app.command("eval-hits")
def eval_hits(
    video_id: int,
    labels: Annotated[Path, typer.Option(help="CSV with a t column (and optionally player)")],
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    offset: Annotated[float, typer.Option(help="Label clock to video time, seconds")] = 0.0,
    resolution: Annotated[float, typer.Option(help="Label time resolution, seconds")] = 0.0,
    verbose: Annotated[bool, typer.Option(help="List misses and extra detections")] = False,
) -> None:
    """Compare a video's detected hits with hand-made labels."""
    import numpy as np
    import pyarrow.parquet as pq

    from tennis.evaluation import evaluate_hits, parse_time, read_labels
    from tennis.pipeline import video_dir

    _cfg, root = _load(config, data_root)
    rows = read_labels(labels)
    if not rows:
        raise UserError(f"{labels} has no labels")
    key = "t" if "t" in rows[0] else next(iter(rows[0]))
    label_t = np.array([parse_time(r[key]) for r in rows])
    hits = pq.read_table(video_dir(root, video_id) / "hits.parquet").to_pydict()
    hit_t = np.asarray(hits["t"], np.float64)
    report, matches = evaluate_hits(hit_t, label_t, offset=offset, resolution=resolution)
    print(report.text())
    if verbose:
        matched_labels = {m.label_index for m in matches}
        for i in np.argsort(label_t):
            if i not in matched_labels:
                print(f"  missed label {label_t[i]:8.1f}  {rows[i]}")


@app.command("eval-shots")
def eval_shots(
    session_id: int,
    labels: Annotated[Path, typer.Option(help="CSV with t, and player and stroke columns")],
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    offset: Annotated[float, typer.Option(help="Label clock to session time, seconds")] = 0.0,
    resolution: Annotated[float, typer.Option(help="Label time resolution, seconds")] = 0.0,
) -> None:
    """Compare a session's shots (hitter and stroke) with hand-made labels."""
    import numpy as np
    from sqlmodel import col, select

    from tennis.db import session_scope
    from tennis.db.models import SessionPlayer, Shot
    from tennis.evaluation import evaluate_shots, read_labels

    _cfg, root = _load(config, data_root)
    rows = read_labels(labels)
    if not rows:
        raise UserError(f"{labels} has no labels")
    with session_scope(root) as db:
        shots = list(
            db.exec(select(Shot).where(Shot.session_id == session_id).order_by(col(Shot.t)))
        )
        label_of = {
            sp.player_id: sp.label
            for sp in db.exec(select(SessionPlayer).where(SessionPlayer.session_id == session_id))
        }
    report = evaluate_shots(
        np.array([s.t for s in shots]),
        [label_of.get(s.player_id) if s.player_id else None for s in shots],
        [s.stroke for s in shots],
        rows,
        offset=offset,
        resolution=resolution,
    )
    print(report.text())


@app.command("eval-points")
def eval_points(
    session_id: int,
    labels: Annotated[Path, typer.Option(help="CSV with t, end, server, winner per point")],
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    offset: Annotated[float, typer.Option(help="Label clock to session time, seconds")] = 0.0,
) -> None:
    """Compare a match's points (server and winner) with hand-made labels."""
    from sqlmodel import col, select

    from tennis.db import session_scope
    from tennis.db.models import Rally, SessionPlayer
    from tennis.evaluation import parse_time, read_labels

    _cfg, root = _load(config, data_root)
    rows = read_labels(labels)
    with session_scope(root) as db:
        rallies = [
            r
            for r in db.exec(
                select(Rally).where(Rally.session_id == session_id).order_by(col(Rally.start_s))
            )
            if r.score_before is not None
        ]
        label_of = {
            sp.player_id: sp.label
            for sp in db.exec(select(SessionPlayer).where(SessionPlayer.session_id == session_id))
        }
    used: set[int] = set()
    pairs = []
    for row in rows:
        lo = parse_time(row["t"]) + offset - 1.5
        hi = parse_time(row["end"]) + offset + 2.5
        for r in rallies:
            if r.id not in used and lo <= r.start_s <= hi:
                used.add(r.id or 0)
                pairs.append((row, r))
                break
    print(f"points: labelled {len(rows)}, found {len(rallies)}, matched {len(pairs)}")
    if not pairs:
        return
    # Which found player is "self": the one who served most of the labelled self serves.
    votes: dict[str, int] = {}
    for row, r in pairs:
        who = label_of.get(r.server_id or -1)
        if who:
            votes[who] = votes.get(who, 0) + (1 if row["server"] == "self" else -1)
    me = max(votes, key=lambda k: votes[k]) if votes else None

    def side(pid: int | None) -> str | None:
        who = label_of.get(pid or -1)
        return None if who is None else ("self" if who == me else "other")

    server_ok = sum(1 for row, r in pairs if side(r.server_id) == row["server"])
    winner_ok = sum(1 for row, r in pairs if side(r.winner_id) == row["winner"])
    print(f"self = player {me}")
    print(f"server right: {server_ok}/{len(pairs)} ({server_ok / len(pairs):.0%})")
    print(f"winner right: {winner_ok}/{len(pairs)} ({winner_ok / len(pairs):.0%})")
    for who in ("self", "other"):
        truth = sum(1 for row in rows if row["winner"] == who)
        found = sum(1 for r in rallies if side(r.winner_id) == who)
        print(f"points won by {who}: labelled {truth}, found {found}")


@app.command("eval-games")
def eval_games(
    session_id: int,
    labels: Annotated[Path, typer.Option(help="CSV with t, self, other: games won so far")],
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    offset: Annotated[float, typer.Option(help="Label clock to session time, seconds")] = 0.0,
) -> None:
    """Compare a match's decoded games with a scoreboard read at a few moments.

    Each label row gives the games each player had won by time ``t`` (sets added up). The
    found player who agrees best with ``self`` is taken as self.
    """
    from sqlmodel import col, select

    from tennis.db import session_scope
    from tennis.db.models import Rally, Session, SessionPlayer
    from tennis.evaluation import evaluate_games, parse_time, read_labels

    _cfg, root = _load(config, data_root)
    rows = read_labels(labels)
    if not rows:
        raise UserError(f"{labels} has no labels")
    with session_scope(root) as db:
        session = db.get(Session, session_id)
        rallies = [
            r
            for r in db.exec(
                select(Rally).where(Rally.session_id == session_id).order_by(col(Rally.start_s))
            )
            if r.score_before is not None
        ]
        players = sorted(
            db.exec(select(SessionPlayer).where(SessionPlayer.session_id == session_id)),
            key=lambda p: p.label,
        )
        score = (session.summary.get("score") if session is not None else None) or {}
    if len(players) != 2 or not rallies:
        print("no decoded match: players", len(players), "points", len(rallies))
        return
    checkpoints = [(parse_time(r["t"]) + offset, int(r["self"]), int(r["other"])) for r in rows]
    report = evaluate_games(
        [(r.start_s, str((r.score_before or {}).get("text") or "")) for r in rallies],
        checkpoints,
        final=str(score.get("text") or ""),
    )
    print(report.text())


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
