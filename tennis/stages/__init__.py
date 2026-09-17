"""Stage registry and runner (spec section 4).

Each stage is a function of its input files and config sections that writes its output
files. Stages share nothing in memory; they communicate only through the session directory.
Stages that are not implemented yet are listed with ``run=None`` and the milestone that
delivers them, so ``tennis list`` and ``--from-stage`` already know the full pipeline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from tennis.config import Config
from tennis.errors import StageError, TennisError, UserError
from tennis.session import SOURCE_INPUT, Session
from tennis.stages import clean, contacts, ingest, pose
from tennis.util.log import log, session_log


@dataclass(frozen=True)
class StageContext:
    session: Session
    config: Config
    config_hash: str
    logger: logging.Logger
    stage: str

    def log(self, event: str, level: int = logging.INFO, **fields: object) -> None:
        log(self.logger, event, level=level, session=self.session.id, stage=self.stage, **fields)


StageFn = Callable[[StageContext], None]


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    config_keys: tuple[str, ...]
    milestone: str
    run: StageFn | None = None
    optional: bool = False

    @property
    def implemented(self) -> bool:
        return self.run is not None

    def config_hash(self, config: Config) -> str:
        return config.section_hash(*self.config_keys)


STAGES: tuple[Stage, ...] = (
    Stage(1, "ingest", (SOURCE_INPUT,), ingest.OUTPUTS, (), "M1", ingest.run),
    Stage(
        2, "contacts", contacts.INPUTS, contacts.OUTPUTS, contacts.CONFIG_KEYS, "M2", contacts.run
    ),
    Stage(3, "pose", pose.INPUTS, pose.OUTPUTS, ("pose", "windows"), "M3", pose.run),
    Stage(
        4,
        "clean",
        clean.INPUTS,
        clean.OUTPUTS,
        clean.CONFIG_KEYS,
        "M4",
        clean.run,
    ),
    Stage(5, "classify", ("swings.parquet",), ("swings.parquet",), ("player", "classify"), "M5"),
    Stage(6, "metrics", ("swings.parquet",), ("metrics.parquet",), ("player", "metrics"), "M6"),
    Stage(
        7,
        "labels",
        ("audio.wav", "contacts.parquet"),
        ("labels.parquet",),
        ("labels",),
        "M8",
        optional=True,
    ),
    Stage(8, "clips", ("metrics.parquet",), ("clips",), ("clips", "player"), "M7"),
    Stage(9, "report", ("metrics.parquet",), ("report.html",), ("report",), "M7"),
)

STAGES_BY_NAME = {s.name: s for s in STAGES}


def stage_status(session: Session, stage: Stage, config: Config) -> str:
    if not stage.implemented:
        return "n/a"
    stamp = session.read_stamp(stage.name)
    if stamp is None:
        return "-"
    reason = session.stale_reason(
        stage.name, stage.inputs, stage.outputs, stage.config_hash(config)
    )
    return "ok" if reason is None else "stale"


def run_pipeline(
    session: Session,
    config: Config,
    logger: logging.Logger,
    *,
    force: bool = False,
    from_stage: int | None = None,
    labels_enabled: bool = True,
) -> list[str]:
    """Run stages in order, skipping up-to-date ones. Returns the names of stages that ran."""
    if from_stage is not None:
        if not 1 <= from_stage <= len(STAGES):
            raise UserError(f"--from-stage must be between 1 and {len(STAGES)}")
        target = STAGES[from_stage - 1]
        if not target.implemented:
            raise UserError(
                f"stage {from_stage} ({target.name}) is not implemented yet "
                f"(planned for {target.milestone})"
            )

    ran: list[str] = []
    with session_log(logger, session.log_path):
        log(logger, "pipeline start", session=session.id, force=force, from_stage=from_stage)
        for stage in STAGES:
            chash = stage.config_hash(config)
            ctx = StageContext(session, config, chash, logger, stage.name)
            if stage.optional and not labels_enabled:
                ctx.log("skipped (disabled)")
                continue
            if not stage.implemented:
                ctx.log(f"not implemented yet (planned for {stage.milestone}); stopping here")
                break

            forced = force or (from_stage is not None and stage.number >= from_stage)
            reason = (
                "forced"
                if forced
                else session.stale_reason(stage.name, stage.inputs, stage.outputs, chash)
            )
            if reason is None:
                ctx.log("up to date, skipping")
                continue

            _check_inputs(session, stage)
            input_prints = session.fingerprints(stage.inputs)
            session.clear_stamp(stage.name)
            ctx.log("running", reason=reason)
            started = time.perf_counter()
            assert stage.run is not None
            try:
                stage.run(ctx)
            except TennisError as exc:
                ctx.log("failed", level=logging.ERROR, error=str(exc))
                if isinstance(exc, StageError):
                    raise
                raise StageError(stage.name, str(exc)) from exc
            except Exception as exc:
                fields = {"session": session.id, "stage": stage.name, "error": repr(exc)}
                logger.error("failed", exc_info=True, extra={"fields": fields})
                raise StageError(stage.name, f"{type(exc).__name__}: {exc}") from exc
            elapsed = round(time.perf_counter() - started, 3)
            session.write_stamp(
                stage.name,
                chash,
                inputs=input_prints,
                outputs=session.fingerprints(stage.outputs),
                elapsed_s=elapsed,
            )
            ctx.log("done", elapsed_s=elapsed)
            ran.append(stage.name)
        log(logger, "pipeline end", session=session.id, ran=ran)
    return ran


def _check_inputs(session: Session, stage: Stage) -> None:
    for name in stage.inputs:
        if session.path(name).exists():
            continue
        producer = next((s for s in STAGES if name in s.outputs), None)
        hint = f"; run stage {producer.number} ({producer.name}) first" if producer else ""
        raise StageError(stage.name, f"missing input {name}{hint}")
