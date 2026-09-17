"""Running CLI commands for the web UI, one at a time, with their output kept for the page.

Each job is ``python -m tennis.cli <argv>`` in a subprocess, so the browser gets exactly
what the terminal would print and a stage failure is still the CLI's own exit code. Jobs
are queued and run one after another: two stages writing into the same session at once
would fight over the same files.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_LINES = 5000  # per job; older lines are dropped and counted


@dataclass
class Job:
    id: str
    command: str
    argv: list[str]
    title: str
    session_id: str | None
    created_at: datetime
    state: str = "queued"  # queued | running | done | failed | cancelled
    started_at: datetime | None = None
    finished_at: datetime | None = None
    returncode: int | None = None
    lines: list[str] = field(default_factory=list)
    dropped: int = 0
    _process: subprocess.Popen[str] | None = None
    _cancelled: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "title": self.title,
            "session_id": self.session_id,
            "state": self.state,
            "returncode": self.returncode,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "started_at": self.started_at.isoformat(timespec="seconds")
            if self.started_at
            else None,
            "finished_at": (
                self.finished_at.isoformat(timespec="seconds") if self.finished_at else None
            ),
            "elapsed_s": round(self.elapsed, 1),
            "n_lines": len(self.lines) + self.dropped,
            "argv": self.argv,
        }

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or datetime.now().astimezone()
        return (end - self.started_at).total_seconds()

    @property
    def done(self) -> bool:
        return self.state in {"done", "failed", "cancelled"}


class JobRunner:
    """A single-worker queue of CLI subprocesses."""

    def __init__(self, cwd: Path, keep: int = 50) -> None:
        self.cwd = cwd
        self.keep = keep
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._run_forever, name="tennis-jobs", daemon=True)
        self._worker.start()

    # --- public API ---------------------------------------------------------------------

    def submit(self, command: str, argv: list[str], title: str, session_id: str | None) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            command=command,
            argv=argv,
            title=title,
            session_id=session_id,
            created_at=datetime.now().astimezone(),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._forget_old()
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 30) -> list[Job]:
        with self._lock:
            ids = self._order[-limit:]
            return [self._jobs[i] for i in reversed(ids) if i in self._jobs]

    def tail(self, job: Job, since: int) -> tuple[list[str], int]:
        """Lines from index ``since`` on, and the index the next poll should ask for."""
        with self._lock:
            start = max(0, since - job.dropped)
            return list(job.lines[start:]), job.dropped + len(job.lines)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.done:
                return False
            job._cancelled = True
            process = job._process
            if process is None:  # still queued: the worker will skip it
                job.state = "cancelled"
                job.finished_at = datetime.now().astimezone()
                return True
        process.terminate()
        return True

    def shutdown(self) -> None:
        with self._lock:
            running = [j._process for j in self._jobs.values() if j._process is not None]
        for process in running:
            if process.poll() is None:
                process.terminate()

    # --- worker -------------------------------------------------------------------------

    def _forget_old(self) -> None:
        while len(self._order) > self.keep:
            old = self._order.pop(0)
            job = self._jobs.get(old)
            if job is not None and not job.done:  # never drop a job still on the queue
                self._order.append(old)
                return
            self._jobs.pop(old, None)

    def _run_forever(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None or job.state == "cancelled":
                continue
            try:
                self._run(job)
            except Exception as exc:  # a broken spawn must not kill the worker
                with self._lock:
                    job.lines.append(f"error: could not start the command: {exc}")
                    job.state = "failed"
                    job.returncode = -1
                    job.finished_at = datetime.now().astimezone()

    def _run(self, job: Job) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", TENNIS_NO_COLOR="1", NO_COLOR="1")
        process = subprocess.Popen(
            [sys.executable, "-m", "tennis.cli", *job.argv],
            cwd=self.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self._lock:
            job._process = process
            job.state = "running"
            job.started_at = datetime.now().astimezone()
        assert process.stdout is not None
        for line in process.stdout:
            self._append(job, line.rstrip("\n"))
        process.stdout.close()
        code = process.wait()
        with self._lock:
            job.returncode = code
            job.finished_at = datetime.now().astimezone()
            if job._cancelled:
                job.state = "cancelled"
            else:
                job.state = "done" if code == 0 else "failed"
            job._process = None

    def _append(self, job: Job, line: str) -> None:
        with self._lock:
            job.lines.append(line)
            if len(job.lines) > MAX_LINES:
                cut = len(job.lines) - MAX_LINES
                del job.lines[:cut]
                job.dropped += cut
