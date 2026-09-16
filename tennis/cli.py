"""Command-line interface (spec section 7).

Exit codes: 0 success, 1 user or config error, 2 stage failure (stage name printed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from tennis import __version__
from tennis.config import Config, load_config
from tennis.errors import TennisError, UserError
from tennis.session import (
    Session,
    check_video_file,
    create_or_reuse_session,
    list_sessions,
    open_session,
)
from tennis.stages import STAGES, run_pipeline, stage_status
from tennis.util import video
from tennis.util.log import get_logger

app = typer.Typer(
    name="tennis",
    help="Measure tennis technique from fixed-camera session videos.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Config YAML (default: ./config.yaml if present)."),
]


def _not_implemented(command: str, milestone: str) -> NoReturn:
    raise UserError(f"'tennis {command}' is not implemented yet (planned for {milestone})")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tennis-analyzer {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    pass


@app.command()
def process(
    video_path: Annotated[Path, typer.Argument(metavar="VIDEO", help="Session video (MP4/MOV).")],
    session_id: Annotated[
        str | None, typer.Option("--session-id", help="Override the derived session id.")
    ] = None,
    config: ConfigOpt = None,
    force: Annotated[bool, typer.Option("--force", help="Rerun every stage.")] = False,
    from_stage: Annotated[
        int | None,
        typer.Option("--from-stage", metavar="N", help="Rerun stage N and everything after it."),
    ] = None,
    no_labels: Annotated[
        bool, typer.Option("--no-labels", help="Skip the voice-label stage.")
    ] = False,
) -> None:
    """Process one session video end to end."""
    cfg = load_config(config)
    src = check_video_file(video_path)
    created = video.probe(src).creation_time or video.file_creation_time(src)
    session = create_or_reuse_session(cfg.paths.data_root, src, created, session_id)
    typer.echo(f"session {session.id}  ({session.dir})", err=True)
    ran = run_pipeline(
        session,
        cfg,
        get_logger(),
        force=force,
        from_stage=from_stage,
        labels_enabled=cfg.labels.enabled and not no_labels,
    )
    typer.echo(f"done: {', '.join(ran) if ran else 'everything up to date'}", err=True)


@app.command("list")
def list_cmd(config: ConfigOpt = None) -> None:
    """List sessions and the status of each stage (ok, stale, - not run, n/a not built yet)."""
    cfg = load_config(config)
    sessions = list_sessions(cfg.paths.data_root)
    if not sessions:
        typer.echo(f"no sessions in {cfg.paths.data_root}")
        return
    _print_status_table(sessions, cfg)


def _print_status_table(sessions: list[Session], cfg: Config) -> None:
    headers = ["session", *(s.name for s in STAGES)]
    rows = [[sess.id, *(stage_status(sess, st, cfg) for st in STAGES)] for sess in sessions]
    widths = [max(len(r[i]) for r in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())


@app.command()
def report(session_id: str, config: ConfigOpt = None) -> None:
    """Rebuild the report for one session."""
    open_session(load_config(config).paths.data_root, session_id)
    _not_implemented("report", "M7")


@app.command()
def trends(
    since: Annotated[str | None, typer.Option(help="Only sessions from this date on.")] = None,
    config: ConfigOpt = None,
) -> None:
    """Build the cross-session trends report."""
    load_config(config)
    _not_implemented("trends", "M7")


@app.command("tune-contacts")
def tune_contacts(
    session_id: str,
    labels: Annotated[Path, typer.Option(help="CSV of labeled impact times.")],
    config: ConfigOpt = None,
) -> None:
    """Grid-search contact detection parameters against labeled impacts."""
    open_session(load_config(config).paths.data_root, session_id)
    _not_implemented("tune-contacts", "M2")


@app.command("eval-classifier")
def eval_classifier(
    session_id: str,
    labels: Annotated[Path, typer.Option(help="CSV of labeled stroke types.")],
    config: ConfigOpt = None,
) -> None:
    """Print a confusion matrix for the stroke classifier."""
    open_session(load_config(config).paths.data_root, session_id)
    _not_implemented("eval-classifier", "M5")


@app.command()
def inspect(session_id: str, swing_id: int, config: ConfigOpt = None) -> None:
    """Print a swing's metrics and open its clip."""
    open_session(load_config(config).paths.data_root, session_id)
    _not_implemented("inspect", "M6")


def main(argv: list[str] | None = None) -> NoReturn:
    """Entry point. Maps every error to the spec's exit codes (click would use 2 for usage)."""
    try:
        rv = app(args=argv, prog_name="tennis", standalone_mode=False)
    except TennisError as exc:
        typer.echo(f"error: {exc}", err=True)
        sys.exit(exc.exit_code)
    except typer.TyperException as exc:  # usage errors: bad option, missing argument, ...
        show = getattr(exc, "show", None)
        if callable(show):
            show()
        else:
            typer.echo(f"error: {exc}", err=True)
        sys.exit(1)
    except typer.Abort:
        typer.echo("aborted", err=True)
        sys.exit(1)
    # In non-standalone mode typer returns the code of an explicit exit (e.g. --help).
    sys.exit(rv if isinstance(rv, int) else 0)


if __name__ == "__main__":
    main()
