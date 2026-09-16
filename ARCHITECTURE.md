# Architecture

## Stages

The pipeline is a fixed sequence of stages. Each stage reads files from the session directory and writes files back to it. Stages never share in-memory state.

| # | Stage | Inputs | Outputs | Config sections | Milestone |
|---|-------|--------|---------|-----------------|-----------|
| 1 | ingest | source video | `metadata.json`, `audio.wav` | none | **M1 (done)** |
| 2 | contacts | `audio.wav`, `metadata.json` | `contacts.parquet` | audio | M2 |
| 3 | pose | `contacts.parquet` (and the source video) | `keypoints.parquet` | pose, windows | M3 |
| 4 | clean | `keypoints.parquet`, `contacts.parquet` | `swings.parquet` | player, cleaning, audio | M4 |
| 5 | classify | `swings.parquet` | `swings.parquet` | player, classify | M5 |
| 6 | metrics | `swings.parquet` | `metrics.parquet` | player, metrics | M6 |
| 7 | labels (optional) | `audio.wav`, `contacts.parquet` | `labels.parquet` | labels | M8 |
| 8 | clips | `metrics.parquet` | `clips/` | clips, player | M7 |
| 9 | report | `metrics.parquet` | `report.html` | report | M7 |

The registry lives in `tennis/stages/__init__.py`. The runner (`run_pipeline`) processes the stages in order:

1. It skips stages that are up to date.
2. It stops at the first stage that is not implemented yet.
3. It wraps any exception in `StageError`, so the CLI exits with code 2 and prints the stage name.

## Caching

After a stage succeeds, the runner writes `.stamps/<stage>.json`. The stamp records the pipeline version, a hash of the config sections the stage uses, and the elapsed time.

A stage is **stale** in any of these cases:

- it has no stamp;
- the pipeline version changed;
- the hash of its config sections changed;
- an output is missing;
- an input's mtime is newer than the oldest output.

For the source video, the check follows the symlink to the raw file.

The runner deletes a stage's stamp before running it. Outputs are written to a temporary file and then renamed into place. Together, these mean a crashed run never looks finished.

The spec says outputs must be "newer than config". This implementation compares a config hash per stage instead of the config file's mtime. Editing an unrelated option therefore does not rerun a stage, and changing a relevant option always does, even if the file's timestamp is unchanged. `paths` is never part of the hash, so moving the data root does not invalidate results.

## Time conventions

- All frame times are **PTS seconds** from the container, never `frame_index / fps`.
- `metadata.json` records `video_start_s`, the PTS of the first video frame.
- It also records `audio_start_s`. Sample 0 of `audio.wav` is at that PTS, so a WAV time `t` is at PTS `t + audio_start_s`.

## Data schemas

Every Parquet output carries these key/value metadata entries: `tennis.pipeline_version`, `tennis.config_hash`, `tennis.stage` and `tennis.schema_version`. JSON outputs carry the same fields at the top level.

### `metadata.json` (schema_version 1)

| Field | Meaning |
|-------|---------|
| `duration_s`, `fps`, `is_vfr`, `resolution`, `codec`, `audio_sample_rate`, `creation_time` | The fields the spec names. `fps` is the nominal rate; `audio_sample_rate` is the WAV rate (48000). |
| `warnings` | Human-readable warnings, for example a frame rate outside 30–240 fps. They are also logged at WARNING level. |
| `creation_time_source` | `container` (a metadata tag) or `filesystem` (fallback) |
| `video_start_s`, `audio_start_s` | Stream start PTS values; see "Time conventions" |
| `video.*` | Stream index, codec, pix_fmt, width, height, rotation, nominal and average fps, `nb_frames`, and sampled frame-interval stats (median, p01 and p99 in ms, over the first 30 s) |
| `audio.*` | Source codec, sample rate and channels, plus the WAV format (mono, s16, 48 kHz) |
| `source_path`, `source_size_bytes`, `container` | Provenance |

A stream is flagged VFR in either of these cases:

- its nominal and average frame rates differ by more than 1%;
- the p01–p99 spread of its frame intervals is more than 50% of the median interval.

### `audio.wav`

Mono, 16-bit PCM, 48 kHz, decoded from the first audio stream. It is written with `-bitexact`, so repeated runs produce identical files.

The schemas for the later stage outputs are in spec section 6. They will be documented here as each stage is built.

## Open design points for later milestones

- **Stage 4 writes `is_self_confirmed` back into `contacts.parquet`.** Stage 3 reads that file, so the rewrite would make stage 3 look stale and cause reruns forever. Proposed fix: stage 4 writes the confirmation to its own output (for example, a column in `swings.parquet` or a `contacts_confirmed.parquet` file), and later stages read it from there.
- **Stage 5 modifies `swings.parquet` in place.** A file cannot be both an input and an output under mtime caching. Proposed fix: stage 5 writes a separate `strokes.parquet`, and readers join it on `swing_id`.
