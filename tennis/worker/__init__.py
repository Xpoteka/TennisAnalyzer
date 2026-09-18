"""The job queue: analysis jobs run one at a time, each in its own subprocess.

Jobs live in the database, so the queue survives a restart: a job that was running when the
server stopped is queued again, and its finished stages are skipped when it reruns. Each job
runs as ``tennis run-job <id>`` in a subprocess, so a crash in a model or in ffmpeg cannot
take the web server down, and cancelling a job just ends that process.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from sqlmodel import col, select

from tennis.config import Config
from tennis.db import session_scope
from tennis.db.models import Job, Session, utcnow
from tennis.pipeline.runner import analyze_session
from tennis.util.log import log

POLL_S = 1.0


def enqueue(data_root: Path, session_id: int, *, force: bool = False) -> Job:
    """Queue an analysis of a session, unless one is already queued or running."""
    with session_scope(data_root) as db:
        existing = db.exec(
            select(Job).where(
                Job.session_id == session_id, col(Job.status).in_(("queued", "running"))
            )
        ).first()
        if existing is not None:
            return existing
        job = Job(session_id=session_id, force=force)
        db.add(job)
        session = db.get(Session, session_id)
        if session is not None:
            session.status = "queued"
            db.add(session)
        db.flush()
        db.refresh(job)
        return job


def cancel(data_root: Path, job_id: int) -> bool:
    with session_scope(data_root) as db:
        job = db.get(Job, job_id)
        if job is None or job.status not in ("queued", "running"):
            return False
        if job.status == "running" and job.pid:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(job.pid, signal.SIGTERM)
        job.status = "cancelled"
        job.finished_at = utcnow()
        db.add(job)
        session = db.get(Session, job.session_id)
        if session is not None and session.status in ("queued", "processing"):
            session.status = "failed"
            session.error = "analysis cancelled"
            db.add(session)
        return True


def execute_job(data_root: Path, config: Config, job_id: int, logger: logging.Logger) -> None:
    """Run a job in this process (the body of ``tennis run-job``)."""
    with session_scope(data_root) as db:
        job = db.get(Job, job_id)
        if job is None:
            raise ValueError(f"no job {job_id}")
        job.status = "running"
        job.started_at = utcnow()
        job.pid = os.getpid()
        db.add(job)
        session_id, force = job.session_id, job.force

    def progress(fraction: float, stage: str, message: str | None) -> None:
        with session_scope(data_root) as db:
            row = db.get(Job, job_id)
            if row is not None and row.status == "running":
                row.progress = round(fraction, 4)
                row.stage = stage
                row.message = message
                db.add(row)

    try:
        analyze_session(data_root, config, session_id, logger, force=force, on_progress=progress)
    except Exception as exc:
        _end(data_root, job_id, "failed", str(exc) or type(exc).__name__)
        raise
    _end(data_root, job_id, "done", None)


def _end(data_root: Path, job_id: int, status: str, error: str | None) -> None:
    with session_scope(data_root) as db:
        job = db.get(Job, job_id)
        if job is not None and job.status == "running":
            job.status = status
            job.error = error
            job.finished_at = utcnow()
            if status == "done":
                job.progress = 1.0
            db.add(job)


class Worker:
    """A background thread that starts queued jobs one after the other."""

    def __init__(self, data_root: Path, config_path: Path | None, logger: logging.Logger):
        self.data_root = data_root
        self.config_path = config_path
        self.logger = logger
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        self._recover()
        self._thread = threading.Thread(target=self._loop, name="tennis-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            # Leave the job marked running: the next start queues it again.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGTERM)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _recover(self) -> None:
        """Queue again the jobs that were running when the server stopped."""
        with session_scope(self.data_root) as db:
            for job in db.exec(select(Job).where(Job.status == "running")):
                job.status = "queued"
                job.pid = None
                db.add(job)

    def _next(self) -> int | None:
        with session_scope(self.data_root) as db:
            job = db.exec(select(Job).where(Job.status == "queued").order_by(col(Job.id))).first()
            return job.id if job is not None else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._next()
            if job_id is None:
                self._stop.wait(POLL_S)
                continue
            self._run(job_id)

    def _run(self, job_id: int) -> None:
        cmd = [sys.executable, "-m", "tennis", "run-job", str(job_id)]
        if self.config_path is not None:
            cmd += ["--config", str(self.config_path)]
        cmd += ["--data-root", str(self.data_root)]
        log(self.logger, "job started", job=job_id)
        self._proc = subprocess.Popen(cmd, start_new_session=True)
        code = self._proc.wait()
        self._proc = None
        if self._stop.is_set():
            return
        # The child records its own result; this covers a child that died without doing so.
        with session_scope(self.data_root) as db:
            job = db.get(Job, job_id)
            if job is not None and job.status in ("queued", "running"):
                job.status = "failed"
                job.error = f"the analysis process ended unexpectedly (exit code {code})"
                job.finished_at = utcnow()
                db.add(job)
                session = db.get(Session, job.session_id)
                if session is not None and session.status != "ready":
                    session.status = "failed"
                    session.error = job.error
                    db.add(session)
        log(self.logger, "job finished", job=job_id, exit_code=code)
