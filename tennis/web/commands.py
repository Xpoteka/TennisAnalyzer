"""The catalogue of CLI commands the web UI offers, and how to turn a form into argv.

The UI renders one card per command from this list, so every pipeline function is
reachable from the browser. The server builds the argv itself from these declarations:
the browser sends field values, never raw command-line arguments.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tennis.errors import UserError

# Field kinds the frontend knows how to render. "session" is a dropdown of session ids,
# "video" and "labels" are file pickers backed by the upload endpoints.
FieldKind = str


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    kind: FieldKind = "text"
    flag: str | None = None  # None means a positional argument
    default: str = ""
    help: str = ""
    required: bool = False
    choices: tuple[str, ...] = ()
    placeholder: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "help": self.help,
            "required": self.required,
            "choices": list(self.choices),
            "placeholder": self.placeholder,
        }


@dataclass(frozen=True)
class Command:
    name: str  # the CLI subcommand
    title: str
    group: str
    summary: str
    fields: tuple[Field, ...] = ()
    slow: bool = False  # worth a warning that it may take a while
    produces: str = ""  # what to offer when it finishes: report, video, html, image, text
    fixed: tuple[str, ...] = field(default_factory=tuple)  # flags always appended

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "group": self.group,
            "summary": self.summary,
            "slow": self.slow,
            "produces": self.produces,
            "fields": [f.as_json() for f in self.fields],
        }


_SESSION = Field("session_id", "Session", "session", required=True)
_TOLERANCE = Field(
    "tolerance_ms",
    "Match tolerance (ms)",
    "number",
    "--tolerance-ms",
    "40",
    help="How close a detection must be to a label to count as the same hit.",
)
_LABEL_OFFSET = Field(
    "label_offset",
    "Label offset (s)",
    "number",
    "--label-offset",
    "0",
    help="Seconds to add to every label when its clock differs from the video's.",
)
_LABEL_RESOLUTION = Field(
    "label_resolution",
    "Label resolution (s)",
    "number",
    "--label-resolution",
    "0",
    help="Label precision: a label t means the range [t, t + this]. Use 1 for whole-second logs.",
)
_SEGMENTS = Field(
    "segments",
    "Segments CSV",
    "labels",
    "--segments",
    help="Only score inside these start,end ranges (e.g. rallies), on the labels' clock.",
)
_REPORT_OUT = Field(
    "report_path",
    "Write markdown to",
    "text",
    "--report",
    placeholder="docs/validation/report.md",
    help="Optional: also save the result as markdown.",
)


COMMANDS: tuple[Command, ...] = (
    Command(
        "process",
        "Process a session",
        "Pipeline",
        "Run every stage on one video: ingest, contacts, pose, clean, classify, metrics, "
        "voice labels, clips and the report.",
        (
            Field(
                "video_path",
                "Video file",
                "video",
                required=True,
                help="Drop a file above, or give a path to a video already on this machine.",
            ),
            Field(
                "session_id",
                "Session id",
                "text",
                "--session-id",
                placeholder="derived from the video's timestamp",
            ),
            Field("force", "Rerun every stage", "bool", "--force"),
            Field(
                "from_stage",
                "Rerun from stage",
                "number",
                "--from-stage",
                placeholder="1-9",
                help="Rerun this stage and everything after it.",
            ),
            Field("no_labels", "Skip voice labels", "bool", "--no-labels"),
        ),
        slow=True,
        produces="report",
    ),
    Command(
        "report",
        "Rebuild the report",
        "Pipeline",
        "Rerun the clips and report stages for one session.",
        (_SESSION, Field("no_clips", "Keep existing clips", "bool", "--no-clips")),
        slow=True,
        produces="report",
    ),
    Command(
        "trends",
        "Cross-session trends",
        "Pipeline",
        "Build the report that compares every session with your history.",
        (
            Field("since", "Only sessions from", "text", "--since", placeholder="YYYY-MM-DD"),
            Field("out", "Output HTML", "text", "--out", placeholder="<data root>/report.html"),
        ),
        produces="trends",
    ),
    Command(
        "relink",
        "Relink moved videos",
        "Pipeline",
        "Point sessions whose raw video is gone at a file of the same name, e.g. after copying "
        "the data folder to a server. Looks in the uploads folder unless you give another one.",
        (
            Field(
                "dir",
                "Look in folder",
                "text",
                "--dir",
                placeholder="<data root>/uploads",
                help="A folder on the server that holds the videos.",
            ),
            Field("dry_run", "Only show what would change", "bool", "--dry-run"),
        ),
    ),
    Command(
        "players",
        "Who is who",
        "Review",
        "Show which tracked player is you, which side you played on, and your racket hand.",
        (_SESSION, Field("count", "Thumbnails per player", "number", "--count", "6")),
        produces="players",
    ),
    Command(
        "pose-preview",
        "Pose preview video",
        "Review",
        "Render sampled swings with the tracked skeleton drawn in, to check the tracking.",
        (
            _SESSION,
            Field("count", "Swings to render", "number", "--count", "20"),
            Field(
                "speed",
                "Playback speed",
                "number",
                "--speed",
                "0.5",
                help="0.25 is slow motion; 1 is real time.",
            ),
            Field("seed", "Random seed", "number", "--seed", "0"),
        ),
        slow=True,
        produces="pose_preview",
    ),
    Command(
        "swing-plots",
        "Swing trajectory plots",
        "Review",
        "Plot raw against cleaned wrist trajectories for sampled swings.",
        (
            _SESSION,
            Field("count", "Swings to plot", "number", "--count", "6"),
            Field("seed", "Random seed", "number", "--seed", "0"),
            Field(
                "all_swings",
                "All QC-passing swings",
                "bool",
                "--all",
                help="Sample from every swing that passed QC, not only confirmed own hits.",
            ),
        ),
        produces="swing_plots",
    ),
    Command(
        "tune-contacts",
        "Tune contact detection",
        "Tuning",
        "Grid-search the audio settings against your labelled impacts and suggest a config block.",
        (
            _SESSION,
            Field(
                "labels",
                "Impact labels CSV",
                "labels",
                "--labels",
                required=True,
                help="Times of your own hits, one per line, as seconds or m:ss.sss.",
            ),
            Field(
                "target",
                "Labels are",
                "choice",
                "--target",
                "self",
                choices=("self", "any"),
                help="self: only your hits. any: every player's hits.",
            ),
            Field("k", "onset_k values", "text", "--k", placeholder="4,6,8"),
            Field("cutoff", "highpass_hz values", "text", "--cutoff", placeholder="600,800,1200"),
            Field("db", "own_hit_db values", "text", "--db", placeholder="3,6,9"),
            _TOLERANCE,
            _LABEL_OFFSET,
            _LABEL_RESOLUTION,
            _SEGMENTS,
            Field("start", "Start of labelled range", "text", "--start", placeholder="0:00"),
            Field("end", "End of labelled range", "text", "--end", placeholder="10:00"),
            Field("top", "Rows to show", "number", "--top", "15"),
            _REPORT_OUT,
        ),
        slow=True,
        produces="text",
    ),
    Command(
        "eval-contacts",
        "Score contact detection",
        "Tuning",
        "Score the audio pass and the wrist confirmation against labels, and sweep the "
        "confirmation settings.",
        (
            _SESSION,
            Field("labels", "Impact labels CSV", "labels", "--labels", required=True),
            _TOLERANCE,
            _LABEL_OFFSET,
            _LABEL_RESOLUTION,
            _SEGMENTS,
            Field("speeds", "Confirm speed values", "text", "--speeds", placeholder="4,6,8"),
            Field("windows", "Confirm window values", "text", "--windows", placeholder="0.1,0.2"),
            Field("top", "Sweep rows to show", "number", "--top", "12"),
            _REPORT_OUT,
        ),
        produces="text",
    ),
    Command(
        "eval-classifier",
        "Score the stroke classifier",
        "Tuning",
        "Print a confusion matrix of the classified strokes against labelled ones.",
        (
            _SESSION,
            Field(
                "labels",
                "Stroke labels CSV",
                "labels",
                "--labels",
                required=True,
                help="Rows of 't,stroke' or 't,player,stroke'.",
            ),
            _TOLERANCE,
            _LABEL_OFFSET,
            _LABEL_RESOLUTION,
            _SEGMENTS,
            _REPORT_OUT,
        ),
        produces="text",
    ),
    Command(
        "eval-labels",
        "Score the voice labels",
        "Tuning",
        "Check the labels the pipeline attached against the words you actually said.",
        (
            _SESSION,
            Field(
                "labels",
                "Spoken labels CSV",
                "labels",
                "--labels",
                required=True,
                help="Rows of 't,label'.",
            ),
            Field("tolerance_s", "Tolerance (s)", "number", "--tolerance-s", "2"),
            _REPORT_OUT,
        ),
        produces="text",
    ),
    Command(
        "inspect",
        "Inspect one swing",
        "Review",
        "Print a swing's metrics beside the session median for that stroke type.",
        (_SESSION, Field("swing_id", "Swing id", "number", required=True)),
        fixed=("--no-open",),  # the browser shows the clip; don't open a player on the host
        produces="text",
    ),
)

COMMANDS_BY_NAME: dict[str, Command] = {c.name: c for c in COMMANDS}

GROUPS: tuple[str, ...] = ("Pipeline", "Review", "Tuning")


def build_argv(command: Command, values: Mapping[str, object], config: Path | None) -> list[str]:
    """Turn submitted form values into a safe argv for ``tennis <command>``.

    Only the fields declared above are read, and each one is checked against its kind, so a
    value can never become an extra flag.
    """
    positional: list[str] = []
    options: list[str] = []
    for f in command.fields:
        raw = values.get(f.name)
        if f.kind == "bool":
            if _as_bool(raw):
                assert f.flag is not None
                options.append(f.flag)
            continue
        text = "" if raw is None else str(raw).strip()
        if not text:
            if f.required:
                raise UserError(f"{command.name}: {f.label} is required")
            continue
        _check(command, f, text)
        if f.flag is None:
            positional.append(text)
        else:
            options += [f.flag, text]
    argv = [command.name, *positional, *options, *command.fixed]
    if config is not None:
        argv += ["--config", str(config)]
    return argv


def _check(command: Command, f: Field, text: str) -> None:
    if text.startswith("-"):
        raise UserError(f"{command.name}: {f.label} must not start with '-' (got {text!r})")
    if f.kind == "number":
        try:
            float(text)
        except ValueError as exc:
            raise UserError(f"{command.name}: {f.label} must be a number, got {text!r}") from exc
    if f.choices and text not in f.choices:
        raise UserError(
            f"{command.name}: {f.label} must be one of {', '.join(f.choices)}, got {text!r}"
        )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
