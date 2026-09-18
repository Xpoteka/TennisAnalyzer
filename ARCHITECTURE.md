# Architecture (v2)

## Processes

```
browser ──HTTP──▶ tennis serve (FastAPI, uvicorn)
                   ├─ /api/*        JSON API, chunked uploads, video streaming (Range)
                   ├─ /*            the React build (web/ → tennis/api/static)
                   └─ Worker thread ──spawns──▶ tennis run-job <id>   (one at a time)
                                                   └─ pipeline → data/videos/<id>/*, tennis.db
```

- **The server** is FastAPI. It does no analysis itself.
- **The worker** is a thread in the server. It takes queued jobs from the `job` table and starts each one as a subprocess. A crash in a model or in ffmpeg therefore cannot take the server down, and cancelling a job ends its process group.
- **Jobs survive restarts.** A job that was running when the server stopped goes back into the queue, and the stages it already finished are skipped (see caching).
- **SQLite runs in WAL mode**, so the server and the job process can write at the same time. Schema changes are additive only. `tennis.db` creates new tables and adds new nullable columns when it is opened, so an existing database keeps its data.

## Code

```
tennis/
  cli.py                 serve, analyze, run-job (internal)
  config.py              pydantic config; section_hash() for stage caching
  db/                    SQLModel tables (models.py) and the engine (WAL, additive migration)
  worker/                job queue: enqueue, cancel, execute_job, Worker thread
  api/                   app.py (app, auth guard, uploads), routes.py (JSON), uploads.py, auth.py
  pipeline/
    __init__.py          VideoStage, VideoContext, stamps; video path storage
    runner.py            analyze_session(): every video's stages, then the session stages
    video/               per-video stages: ingest, proxy (court, people, ball, audio next)
    session/             per-session stages: timeline (shots, rallies, kind, scoring, players next)
  pose_backends/         PoseBackend protocol + YOLO
  util/                  video/ffprobe, FrameReader, audio onsets, filters, io, overlay
web/                     React + Vite + TypeScript
```

## Pipeline

**Video stages** run once per video. They read the source file and earlier outputs, and write files to `data/videos/<id>/`.

A stage is skipped when its stamp (`.stamps/<stage>.json`) still matches all of these:

- the pipeline version;
- the stage's own version;
- a hash of the config keys it uses;
- the source file's size and modification time.

Every output must also still exist.

| Stage | Writes | Notes |
|---|---|---|
| ingest | `metadata.json`, `frame_times.parquet`, `audio.wav` (if the video has sound) | Frame times are **PTS seconds**; frame index `i` = row `i`. `audio.wav` sample 0 is at PTS `audio_start_s`. |
| proxy | `proxy.mp4`, `proxy.json` | Browser-playable H.264. A source that already plays is symlinked instead. Proxy time = PTS − `start_pts`. |

**Session stages** combine every finished video of a session and write to the database. They are cheap and always run.

| Stage | Does |
|---|---|
| timeline | Places each video on the session clock (`Video.offset_s`), by creation time for now. Audio sync comes with multi-video support. |

**Times in the database are session seconds.** A video's PTS `t` is at session time `t + offset_s`.

## Paths

A video inside the data folder is stored relative to it (`uploads/x.mp4`), so the data folder can move from a Mac to a server as a whole. A file elsewhere is stored with its absolute path.
