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


@app.command("pose-preview")
def pose_preview(
    session_id: str,
    config: ConfigOpt = None,
    count: Annotated[int, typer.Option(min=1, help="Swings (analysis windows) to render.")] = 20,
    seed: Annotated[int, typer.Option(help="Random seed for the sample.")] = 0,
    speed: Annotated[
        float, typer.Option(min=0.05, max=4.0, help="Playback speed, e.g. 0.25 for slow motion.")
    ] = 1.0,
    out: Annotated[
        Path | None, typer.Option(help="Output MP4 (default: <session>/debug/pose_preview.mp4).")
    ] = None,
) -> None:
    """Render sampled swings with the tracked skeleton for review (M3 acceptance)."""
    from tennis.review import render_pose_preview

    cfg = load_config(config)
    session = open_session(cfg.paths.data_root, session_id)
    path, summaries = render_pose_preview(
        session, cfg, count=count, seed=seed, speed=speed, out=out
    )
    typer.echo(f"{'window':>6}  {'start_s':>8}  {'frames':>6}  {'tracked':>7}  {'resets':>6}")
    for s in summaries:
        typer.echo(
            f"{s.window_id:>6}  {s.start:>8.2f}  {s.frames:>6}  "
            f"{s.detected_ratio:>7.1%}  {s.resets:>6}"
        )
    frames = sum(s.frames for s in summaries)
    detected = sum(s.detected for s in summaries)
    ratio = detected / frames if frames else 0.0
    typer.echo(
        f"total: {detected}/{frames} frames with a player ({ratio:.1%}); "
        f"M3 target is at least 95% with the skeleton on the right person (check the video)"
    )
    typer.echo(f"wrote {path} and {path.with_suffix('.csv').name}", err=True)


@app.command()
def players(
    session_id: str,
    config: ConfigOpt = None,
    count: Annotated[int, typer.Option(min=1, help="Thumbnails per player.")] = 6,
) -> None:
    """Show who was identified as you, which side you played on, and your racket hand."""
    from tennis.review import render_players

    cfg = load_config(config)
    session = open_session(cfg.paths.data_root, session_id)
    path, info = render_players(session, cfg, per_player=count)
    hand = info.get("handedness") or {}
    typer.echo(f"you: player {info.get('me')} ({info.get('me_reason')})")
    typer.echo(f"racket hand: {hand.get('hand')} ({hand.get('reason')}; votes {hand.get('votes')})")
    typer.echo(f"hits: {info.get('hits')}")
    if not info.get("identities_resolved"):
        typer.echo("only one player was tracked; everyone near the camera is treated as you")
    for seg in info.get("sides") or []:
        start, end = float(seg["start"]), float(seg["end"])
        span = f"{int(start // 60):3d}:{start % 60:04.1f} - {int(end // 60):3d}:{end % 60:04.1f}"
        typer.echo(f"  {span}  you are {seg['side']:4s} ({seg['windows']} windows)")
    typer.echo(f"wrote {path}; if the wrong player is marked as you, set player.identity "
               "to the other letter in your config")  # fmt: skip


@app.command("swing-plots")
def swing_plots(
    session_id: str,
    config: ConfigOpt = None,
    count: Annotated[int, typer.Option(min=1, help="Swings to plot.")] = 6,
    seed: Annotated[int, typer.Option(help="Random seed for the sample.")] = 0,
    all_swings: Annotated[
        bool, typer.Option("--all", help="Sample from all QC-passing swings, not just confirmed.")
    ] = False,
    out: Annotated[Path | None, typer.Option(help="Output HTML file.")] = None,
) -> None:
    """Plot raw vs cleaned wrist trajectories for sampled swings (M4 review)."""
    from tennis.review import render_swing_plots

    cfg = load_config(config)
    session = open_session(cfg.paths.data_root, session_id)
    path = render_swing_plots(
        session, cfg, count=count, seed=seed, confirmed_only=not all_swings, out=out
    )
    typer.echo(f"wrote {path}", err=True)


@app.command("tune-contacts")
def tune_contacts(
    session_id: str,
    labels: Annotated[Path, typer.Option(help="CSV of impact times (video time, s or m:ss.sss).")],
    config: ConfigOpt = None,
    target: Annotated[
        str,
        typer.Option(help="self: labels are your own hits. any: labels are all players' hits."),
    ] = "self",
    k: Annotated[str | None, typer.Option("--k", help="onset_k values, comma-separated.")] = None,
    cutoff: Annotated[
        str | None, typer.Option("--cutoff", help="highpass_hz values, comma-separated.")
    ] = None,
    db: Annotated[
        str | None, typer.Option("--db", help="own_hit_db_threshold values, comma-separated.")
    ] = None,
    tolerance_ms: Annotated[float, typer.Option(help="Match tolerance.")] = 40.0,
    label_offset: Annotated[
        float, typer.Option(help="Seconds to add to every label (label clock vs video).")
    ] = 0.0,
    label_resolution: Annotated[
        float,
        typer.Option(help="Label precision in seconds: a label t means [t, t + this]."),
    ] = 0.0,
    segments: Annotated[
        Path | None,
        typer.Option(
            help="CSV of start,end ranges to evaluate (e.g. rallies), on the labels' clock."
        ),
    ] = None,
    start: Annotated[
        str | None, typer.Option(help="Start of the labeled range (default: first label - 1 s).")
    ] = None,
    end: Annotated[
        str | None, typer.Option(help="End of the labeled range (default: last label + 1 s).")
    ] = None,
    top: Annotated[int, typer.Option(help="Rows to show.")] = 15,
    report_path: Annotated[
        Path | None, typer.Option("--report", help="Also write the results as markdown.")
    ] = None,
) -> None:
    """Grid-search contact detection parameters against labeled impacts."""
    from tennis.stages import contacts
    from tennis.validation import parse_time, read_segments, read_time_labels

    if label_resolution < 0 or tolerance_ms < 0:
        raise UserError("--label-resolution and --tolerance-ms must not be negative")
    cfg = load_config(config)
    session = open_session(cfg.paths.data_root, session_id)
    spec = contacts.LabelSpec(
        times_s=read_time_labels(labels),
        tolerance_s=tolerance_ms / 1000,
        resolution_s=label_resolution,
        offset_s=label_offset,
    )

    ranges: list[tuple[float, float]] | None = None
    if segments is not None:
        if start is not None or end is not None:
            raise UserError("use either --segments or --start/--end, not both")
        ranges = read_segments(segments)
    elif start is not None or end is not None:
        try:
            lo = parse_time(start) if start is not None else 0.0
            hi = parse_time(end) if end is not None else float("inf")
        except ValueError as exc:
            raise UserError(f"--start/--end: {exc}") from exc
        ranges = [(lo, hi)]

    typer.echo(
        f"tuning on {spec.times_s.size} labels; this runs detection once per cutoff...", err=True
    )
    result = contacts.tune_contacts(
        session,
        cfg.audio,
        spec,
        target=target,
        ks=_float_list(k, "k") or contacts.DEFAULT_GRID_K,
        cutoffs=_float_list(cutoff, "cutoff") or contacts.DEFAULT_GRID_CUTOFF,
        db_thresholds=_float_list(db, "db") or contacts.DEFAULT_GRID_DB,
        segments=ranges,
    )
    typer.echo(contacts.format_tune_report(result, top=top))
    if report_path is not None:
        contacts.write_tune_report(result, report_path, top=top)
        typer.echo(f"wrote {report_path}", err=True)


def _float_list(value: str | None, name: str) -> list[float] | None:
    if value is None:
        return None
    try:
        items = [float(v) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise UserError(f"--{name}: expected comma-separated numbers, got {value!r}") from exc
    if not items:
        raise UserError(f"--{name}: no values given")
    return items


@app.command("eval-contacts")
def eval_contacts(
    session_id: str,
    labels: Annotated[
        Path, typer.Option(help="CSV of your own impact times (video time, s or m:ss.sss).")
    ],
    config: ConfigOpt = None,
    tolerance_ms: Annotated[float, typer.Option(help="Match tolerance.")] = 40.0,
    label_offset: Annotated[
        float, typer.Option(help="Seconds to add to every label (label clock vs video).")
    ] = 0.0,
    label_resolution: Annotated[
        float,
        typer.Option(help="Label precision in seconds: a label t means [t, t + this]."),
    ] = 0.0,
    segments: Annotated[
        Path | None,
        typer.Option(
            help="CSV of start,end ranges to evaluate (e.g. rallies), on the labels' clock."
        ),
    ] = None,
    speeds: Annotated[
        str | None, typer.Option(help="wrist_confirm_min_speed values to sweep.")
    ] = None,
    windows: Annotated[
        str | None, typer.Option(help="wrist_confirm_window_s values to sweep.")
    ] = None,
    top: Annotated[int, typer.Option(help="Sweep rows to show.")] = 12,
    report_path: Annotated[
        Path | None, typer.Option("--report", help="Also write the results as markdown.")
    ] = None,
) -> None:
    """Score own-hit detection (audio first pass and wrist confirmation) against labels."""
    from tennis.evaluation import (
        DEFAULT_SPEEDS,
        DEFAULT_WINDOWS,
        evaluate_contacts,
        format_contact_eval,
        write_contact_eval,
    )
    from tennis.stages.clean import resolved_handedness
    from tennis.stages.contacts import LabelSpec, make_scorer
    from tennis.validation import read_segments, read_time_labels

    cfg = load_config(config)
    session = open_session(cfg.paths.data_root, session_id)
    spec = LabelSpec(
        times_s=read_time_labels(labels),
        tolerance_s=tolerance_ms / 1000,
        resolution_s=label_resolution,
        offset_s=label_offset,
    )
    scorer = make_scorer(session, spec, read_segments(segments) if segments else None)
    result = evaluate_contacts(
        session,
        scorer,
        (
            cfg.audio.wrist_confirm_min_speed,
            cfg.audio.wrist_confirm_window_s,
            cfg.audio.wrist_confirm_wrist,
        ),
        resolved_handedness(session, cfg),
        speeds=_float_list(speeds, "speeds") or DEFAULT_SPEEDS,
        windows=_float_list(windows, "windows") or DEFAULT_WINDOWS,
    )
    typer.echo(format_contact_eval(result, top=top))
    if report_path is not None:
        write_contact_eval(result, report_path, top=top)
        typer.echo(f"wrote {report_path}", err=True)


@app.command("eval-classifier")
def eval_classifier(
    session_id: str,
    labels: Annotated[
        Path, typer.Option(help="CSV of labeled stroke types ('t,stroke' or 't,player,stroke').")
    ],
    config: ConfigOpt = None,
    tolerance_ms: Annotated[float, typer.Option(help="Match tolerance.")] = 40.0,
    label_offset: Annotated[
        float, typer.Option(help="Seconds to add to every label (label clock vs video).")
    ] = 0.0,
    label_resolution: Annotated[
        float, typer.Option(help="Label precision in seconds: a label t means [t, t + this].")
    ] = 0.0,
    segments: Annotated[
        Path | None,
        typer.Option(help="CSV of start,end ranges to evaluate, on the labels' clock."),
    ] = None,
    report_path: Annotated[
        Path | None, typer.Option("--report", help="Also write the results as markdown.")
    ] = None,
) -> None:
    """Print a confusion matrix for the stroke classifier (M5 acceptance: 90% accuracy)."""
    import numpy as np

    from tennis.evaluation import evaluate_classifier, format_classifier_eval
    from tennis.stages.contacts import LabelSpec, make_scorer
    from tennis.validation import read_segments, read_stroke_labels

    cfg = load_config(config)
    session = open_session(cfg.paths.data_root, session_id)
    pairs = read_stroke_labels(labels)
    spec = LabelSpec(
        times_s=np.array([t for t, _ in pairs], dtype=np.float64),
        tolerance_s=tolerance_ms / 1000,
        resolution_s=label_resolution,
        offset_s=label_offset,
    )
    scorer = make_scorer(session, spec, read_segments(segments) if segments else None)
    result = evaluate_classifier(session, scorer, pairs)
    typer.echo(format_classifier_eval(result))
    if report_path is not None:
        from tennis.evaluation import write_classifier_eval

        write_classifier_eval(result, report_path)
        typer.echo(f"wrote {report_path}", err=True)


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
