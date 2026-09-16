# Tennis Technique Analyzer

A local command-line pipeline. It takes a video of a tennis session filmed from a fixed tripod, finds each ball impact, and measures the player's technique. It compares the player against their own history, not against an absolute standard.

**Status: milestone M1.** Built so far: the scaffold, config, CLI, stage caching and the **ingest** stage. The other stages are registered, and `tennis list` shows them as `n/a` until they are built. `tennis process` stops cleanly after the last stage that exists. See [ARCHITECTURE.md](ARCHITECTURE.md) for how the stages fit together.

## Setup

You need [uv](https://docs.astral.sh/uv/) and ffmpeg (`ffmpeg` and `ffprobe` on `PATH`).

```bash
brew install uv ffmpeg                  # macOS
sudo apt-get install ffmpeg             # Debian/Ubuntu; install uv from its docs
uv sync                                 # creates .venv with Python 3.11 and all dependencies
cp config.example.yaml config.yaml      # optional; every key has a default
```

Run the CLI with `uv run tennis ...`, or activate `.venv` and call `tennis` directly.

## Recording guidelines

- Mount the camera on a tripod in the **same position every session**: behind the baseline, slightly off-centre, raised.
- Use 100–120 fps if you can. The pipeline is designed for 30–240 fps, including variable frame rate. Lower rates (for example 24 fps) still work, but ingest warns because contact timing and wrist speeds will be less precise.
- Record the clip-on microphone **into the camera file**. Ingest fails if the video has no audio track.
- Keep yourself the largest person on the near side of the court. A hitting partner on the far side is fine.
- Keep the camera clock correct. Session IDs come from the video's creation timestamp.
- Leave the raw files where they are. The pipeline never copies or changes them; each session only holds a symlink to the file.

## CLI

```text
tennis process <video> [--session-id ID] [--config PATH] [--force] [--from-stage N] [--no-labels]
tennis list                               # sessions and the status of each stage
tennis report <session-id>                # M7
tennis trends [--since DATE]              # M7
tennis tune-contacts <session-id> --labels PATH    # M2
tennis eval-classifier <session-id> --labels PATH  # M5
tennis inspect <session-id> <swing-id>    # M6
```

- **Session IDs** default to `YYYY-MM-DD_<morning|afternoon|evening|night>`, in local time. If a different video already has that ID, the new session gets `_2`, `_3` and so on. Processing the same file again reuses its existing session.
- **Caching:** a stage is skipped when all of these hold:
  - its outputs exist and are newer than its inputs;
  - the config sections it uses are unchanged;
  - the pipeline version is the same.

  Use `--force` to rerun every stage, or `--from-stage N` to rerun stage N and everything after it.
- **Exit codes:** `0` means success, `1` means a user or config error, and `2` means a stage failed. When a stage fails, its name is printed.
- **Config:** the CLI reads `--config`, or `./config.yaml` if it exists; otherwise it uses the defaults. Unknown keys are rejected. A relative `paths.data_root` is resolved from the config file's directory.

Every run appends JSON lines to `data/sessions/<id>/pipeline.log`.

> **YAML gotcha:** quote `"yes"`, `"no"`, `"on"` and `"off"` in the label vocabulary. Unquoted, YAML reads them as booleans and config validation fails.

## Development

```bash
uv run pytest            # unit + integration tests (integration tests build tiny clips with ffmpeg)
uv run ruff check . && uv run ruff format --check .
uv run mypy              # strict on the tennis package
```

CI (`.github/workflows/ci.yml`) runs the same checks on Ubuntu.

### Adding a stage

1. Create `tennis/stages/<name>.py` with a `run(ctx: StageContext) -> None` function.
2. Write every output with `tennis.util.io.atomic_path`, `write_json` or `write_parquet`. `write_parquet` records the pipeline version, config hash and schema version.
3. In the `STAGES` registry in `tennis/stages/__init__.py`, set `run=` on the stage's entry, and check its inputs, outputs and config sections.
4. Log with `ctx.log(event, **fields)`. When one swing is bad, log it, mark it and continue. Do not raise for it.

### Adding a metric or pose backend

These arrive with M6 and M3. Metrics will be registered functions (`@metric(name, applies_to)`) in `tennis/stages/metrics.py`. Pose backends will implement the `PoseBackend` protocol in `tennis/pose_backends/base.py`.
