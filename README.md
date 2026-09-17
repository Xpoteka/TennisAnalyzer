# Tennis Technique Analyzer

A local command-line pipeline. It takes a video of a tennis session filmed from a fixed tripod, finds each ball impact, and measures the player's technique. It compares the player against their own history, not against an absolute standard.

**Status: milestone M4.** Built so far: the scaffold, config, CLI, stage caching, the **ingest** stage, **contact detection** with its tuning tool, **pose extraction** with a review video, and **cleaning, normalization, QC and own-hit confirmation**. The other stages are registered, and `tennis list` shows them as `n/a` until they are built. `tennis process` stops cleanly after the last stage that exists. See [ARCHITECTURE.md](ARCHITECTURE.md) for how the stages fit together. Continuing development? Start with [docs/HANDOFF.md](docs/HANDOFF.md).

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
- One person per half of the court works best: you and your hitting partner. Changing ends is fine; the pipeline tells the two of you apart by clothing color. Wear something that looks clearly different from your partner.
- Keep the camera clock correct. Session IDs come from the video's creation timestamp.
- Keep the video files stored locally. iCloud's "Optimize Mac Storage" can remove the local copy, and reading such a file then stalls while iCloud downloads it again.
- Leave the raw files where they are. The pipeline never copies or changes them; each session only holds a symlink to the file.

## CLI

```text
tennis process <video> [--session-id ID] [--config PATH] [--force] [--from-stage N] [--no-labels]
tennis list                               # sessions and the status of each stage
tennis report <session-id>                # M7
tennis trends [--since DATE]              # M7
tennis tune-contacts <session-id> --labels PATH [--k ...] [--cutoff ...] [--db ...] [--report PATH]
tennis eval-contacts <session-id> --labels PATH [--speeds ...] [--windows ...] [--report PATH]
tennis pose-preview <session-id> [--count 20] [--speed 0.5]
tennis swing-plots <session-id> [--count 6] [--all]
tennis players <session-id>                 # who is you, which side, racket hand
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

### Pose extraction

- **When it runs:** the pose stage processes only the frames around contacts (`windows.pre_s` / `windows.post_s`).
- **Which contacts:** by default, only those flagged as your own hits by the first audio pass. When the mic isn't on you, as with a court camera, set `pose.contacts: all`.
- **Model and device:** the model is downloaded to `data/models/` on first use. `pose.device: auto` uses an NVIDIA GPU, then the Apple GPU, then the CPU.
- **Speed:** on an Apple M-series Mac, `yolo11m-pose` at 640 px runs at about 28 frames/s on 720p video.

Check the tracking with a review video of 20 random swings. It has the skeleton drawn in (racket side in orange), a red border on contact frames, and a warning on frames with no player:

```bash
uv run tennis pose-preview <session-id> --count 20 --speed 0.5
```

It writes `data/sessions/<id>/debug/pose_preview.mp4`, plus a CSV of the tracked frame ratio and track resets per swing.

### Who is who, and your racket hand

Pose tracks the near-court and the far-court player. Stage 4 matches their clothing colors to two players, A and B, across the whole session, so it follows you when you change ends.

- **Hits:** each hit goes to the player whose wrist speeds up the most at that moment.
- **You:** the player whose hits are clearly louder on your clip-on mic. Without a player-worn mic, you're A, the near player at the start.
- **Racket hand:** `player.handedness: auto` picks the wrist that moves faster at your own near-side hits.
- **Far side:** your far-side swings are detected but not measured, because far-side poses are too small.

Check the result with:

```bash
uv run tennis players <session-id>
```

It prints the decision, the hand, and the times you were on each side, and writes `debug/players.jpg` with thumbnails of A and B. If it picked the wrong player, set `player.identity: B` (or `A`) in your config. You can also fix `player.handedness` by hand. Only stage 4 reruns.

### Cleaning and own-hit confirmation

Stage 4 does the following:

- cleans the keypoints;
- turns each candidate contact into a normalized swing;
- runs the spec's QC checks;
- confirms own hits: the wrist speed must peak near the sound.

To review and tune it:

```bash
uv run tennis swing-plots <session-id> --count 6          # raw vs cleaned wrist trajectories
uv run tennis eval-contacts <session-id> --labels labels/contacts_<session-id>.csv
```

`eval-contacts` accepts the same label options as `tune-contacts`. It scores the audio first pass, the wrist confirmation, and both together, then sweeps the confirmation settings. Nothing is recomputed except the confirmation rule, so it takes seconds.

### Tuning contact detection

Contact detection needs tuning against your own labeled hits. The spec's default settings count far too many sounds as your hits.

1. Pick a stretch of about 5–10 minutes of rallying. Write down the time of **every one of your own hits** in it, as the video player shows it: `83.4` or `1:23.4`. Don't include your partner's hits, bounces or footsteps. See [labels/README.md](labels/README.md) for the file format.
2. Save the times as `labels/contacts_<session-id>.csv`.
3. Run the tuning tool:

   ```bash
   uv run tennis tune-contacts 2026-09-16_evening --labels labels/contacts_2026-09-16_evening.csv \
     --report docs/validation/M2_contacts.md
   ```

   It prints the precision and recall of every combination of `onset_k`, `highpass_hz` and `own_hit_db_threshold`. Only detections inside the labeled stretch count. It also prints a suggested `audio:` config block. Copy that block into `config.yaml`, then run `tennis process` again; only the contacts stage reruns.

**Coarse or external labels.** If the times come from another system, for example Wingfield's whole-second shot log, describe their precision and clock offset with `--label-resolution 1 --label-offset 1`. Then restrict scoring to rallies with `--segments rallies.csv` (`start,end` rows, on the labels' clock). Use `--target any` when the labels include both players' shots. `scripts/wingfield_labels.py` converts a Wingfield `.xlsx` export into these files:

```bash
uv run scripts/wingfield_labels.py ~/Downloads/<export>.xlsx <session-id>
```

In the output, `recall_any` is the recall you'd get if every detected sound counted as your hit. If `recall_any` is low, the detector itself is missing hits. If it's high but `recall` is low, the loudness rule is the problem.

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

### Adding a pose backend

1. Implement the `PoseBackend` protocol from `tennis/pose_backends/base.py`. It has a `name` and an `infer(frames) -> list[list[PersonPose]]` method that takes BGR images and returns COCO-17 keypoints in pixels.
2. Register a factory in `tennis/pose_backends/__init__.py` with `register_backend("name", factory)`.
3. Select it with `pose.backend: name`.

No other stage needs to change.

**License note:** the built-in `yolo` backend uses Ultralytics, which is AGPL-3.0. That's fine for personal use. Before distributing this software, switch to a differently licensed backend and remove `ultralytics` from the dependencies.

### Adding a metric

This arrives with M6. Metrics will be registered functions (`@metric(name, applies_to)`) in `tennis/stages/metrics.py`.
