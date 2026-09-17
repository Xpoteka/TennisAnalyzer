# Architecture

## Stages

The pipeline is a fixed sequence of stages. Each stage reads files from the session directory and writes files back to it. Stages never share in-memory state.

| # | Stage | Inputs | Outputs | Config sections | Milestone |
|---|-------|--------|---------|-----------------|-----------|
| 1 | ingest | source video | `metadata.json`, `audio.wav`, `frame_times.parquet` | none | **M1 (done)** |
| 2 | contacts | `audio.wav`, `metadata.json`, `frame_times.parquet` | `contacts.parquet` | audio | **M2 (done, needs tuning)** |
| 3 | pose | `contacts.parquet`, `frame_times.parquet`, `metadata.json`, source video | `keypoints.parquet` | pose, windows | **M3 (done, pending review)** |
| 4 | clean | `keypoints.parquet`, `contacts.parquet`, `frame_times.parquet` | `swings.parquet`, `swing_info.parquet` | player, cleaning, windows, `pose.kp_conf_min`, `pose.contacts`, `audio.wrist_confirm_*` | **M4 (done)** |
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

After a stage succeeds, the runner writes `.stamps/<stage>.json`. The stamp records:

- the pipeline version;
- a hash of the config values the stage uses;
- a fingerprint (mtime and size) of every input and output;
- the elapsed time.

Each stage declares its config keys. A key is either a whole section (`pose`) or a single field (`audio.onset_k`). For example, changing a wrist-confirmation setting reruns only stage 4, not contact detection.

A stage is **stale** in any of these cases:

- it has no stamp;
- the pipeline version changed;
- the hash of its config keys changed;
- an output is missing;
- an input or output fingerprint differs from the one recorded in the stamp.

For the source video, the check follows the symlink to the raw file. Stamps written before fingerprints existed fall back to the older rule: an input is newer than the oldest output.

**Unchanged results don't trigger reruns.** When a stage rewrites an output with identical content, the file keeps its old modification time. For Parquet, "identical" means equal data; files of other types must be byte-identical. Downstream stages therefore don't rerun. For example, re-tuning contact detection in a way that yields the same onsets doesn't repeat the slow pose stage.

The runner deletes a stage's stamp before running it. Outputs are written to a temporary file and then renamed into place. Together, these mean a crashed run never looks finished.

The spec says outputs must be "newer than their inputs and config". This implementation compares recorded fingerprints and a config hash per stage instead of mtimes. Editing an unrelated option therefore does not rerun a stage, and changing a relevant option always does, even if the file's timestamp is unchanged. `paths` is never part of the hash, so moving the data root does not invalidate results.

## Time conventions

- All frame times are **PTS seconds** from the container, never `frame_index / fps`.
- `frame_times.parquet` lists the PTS of every frame. Frame index `i` always means row `i` of this file.
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

### `frame_times.parquet` (schema_version 1)

| Column | Type | Meaning |
|--------|------|---------|
| `frame_idx` | int64 | 0-based frame index in presentation order |
| `pts` | float64 | PTS in seconds |

Ingest reads these from packet headers with ffprobe, which takes about 5 s for a 5 GB file. Packets marked as discarded are skipped. The frame-interval stats in `metadata.json` are computed from every frame.

### `contacts.parquet` (schema_version 1)

| Column | Type | Meaning |
|--------|------|---------|
| `contact_id` | int64 | Sequential, in time order |
| `t_audio` | float64 | Onset time on the PTS timeline (WAV time + `audio_start_s`) |
| `frame_idx`, `t_video` | int64, float64 | The frame whose PTS is nearest to `t_audio`, and that PTS |
| `onset_strength`, `threshold` | float32 | Onset envelope value at the detection, and the adaptive threshold there |
| `peak_db` | float64 | Peak 5 ms RMS level (dBFS) of the high-passed audio within ±20 ms |
| `prominence_db` | float64 | `peak_db` minus the local background level (sliding median over `threshold_window_s`) |
| `is_self_audio` | bool | First pass: `peak_db >= session median + own_hit_db_threshold` |
| `is_self_confirmed` | bool | Second pass using wrist speed. Null until stage 4 exists. |

How detection works (`tennis/util/audio.py`):

1. A high-pass Butterworth filter (`sosfiltfilt`, zero phase) removes low-frequency sound.
2. A 64-band log-mel spectrogram is computed (20 ms window, 5 ms hop, no top-dB clipping), and `librosa.onset.onset_strength` turns it into a spectral-flux envelope.
3. The threshold is `median + k·MAD` over a sliding window. For speed, it is evaluated every 0.25 s and interpolated in between.
4. `find_peaks` picks peaks above the threshold, at least `min_separation_s` apart.
5. Each onset is moved to the rise of the transient in the filtered waveform: the last point below 30% of the peak, at most 10 ms before it. On synthetic clicks this is accurate to about 1 ms.
6. Two measurements are taken: `peak_db` and `prominence_db`.

Audio is processed in 60 s chunks with 1 s of overlap, so memory stays bounded and results don't depend on the chunk size (a test checks this). A 49-minute session takes about 6 s.

Two things differ from the spec:

- **`min_prominence_db` (default 6 dB).** Detections quieter than this relative to the local background are dropped. Without this rule, the log-flux envelope's heavy tail produces several false detections per minute of plain noise.
- **The first-pass own-hit rule is kept as specified, but it has a known limit.** It compares each onset to the *median* onset level, so it only works when your own hits are a minority of all onsets. That holds for real sessions (about 450 loud onsets out of about 3,000 in the first one). It fails when there are few other sounds. The second pass (wrist speed, M4) is the real safeguard.

### `keypoints.parquet` (schema_version 2)

The file has one row per decoded frame inside an analysis window, sorted by `frame_idx`.

| Column | Type | Meaning |
|--------|------|---------|
| `frame_idx`, `t_video` | int64, float64 | Frame index and PTS |
| `slot` | string | `near` or `far`: which tracked player the row belongs to (one row per frame per slot) |
| `appearance` | list<float32> | Clothing-color descriptor of the torso (`tennis/util/appearance.py`) |
| `window_id` | int32 | Merged analysis window |
| `detected` | bool | A player was selected in this frame |
| `track_reset` | bool | The selection fell back to the first-frame rule because nothing overlapped the previous box enough |
| `n_persons` | int16 | Number of people the backend found |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `bbox_conf` | float32 | Selected player's box, in pixels (NaN when not detected) |
| `<kp>_x`, `<kp>_y`, `<kp>_conf` | float32 | 17 COCO keypoints (`nose`, `l_eye`, …, `r_ankle`), in pixels; confidence is 0 when not detected |

The Parquet metadata also records `frame_width`, `frame_height`, `backend`, `model`, `device` and `crop_refine`.

How the stage works:

1. **Windows:** it takes `[t − pre_s, t + post_s]` around each selected contact and merges overlapping ranges. Windows less than `seek_gap_s` apart are decoded in one pass instead of seeking again.
2. **Decoding:** `tennis/util/frames.py` runs `ffmpeg -copyts -ss … -vf showinfo`. ffmpeg writes raw BGR frames, and its `showinfo` output gives each frame's PTS, which is matched to `frame_times.parquet` within 1 ms. ffmpeg applies rotation. PyAV isn't used, because on macOS its bundled FFmpeg clashes with OpenCV's.
3. **Detection:** the backend finds everyone in each batch of frames.
4. **Player selection:** `tennis/util/tracking.py` implements the spec's rule, once for each half of the court.
   - **Near player:** in a window's first frame, the largest person whose box bottom is below `near_court_min_y`. After that, the person with the highest IoU with the previous selection. If the IoU is below `track_iou_min`, it falls back to the first rule and sets `track_reset`. A frame with no detection keeps the previous box.
   - **Far player (`pose.track_far`):** the same rule, applied to people above that line. They are searched both in the full frame and in a full-resolution crop (`pose.far_crop`), because far players are only a few dozen pixels tall. This roughly doubles pose time.
5. **Crop refinement:** if the frame is more than 2.5× the model's input size (for example 4K at 640 px), pose runs again on a full-resolution crop around the player, padded by `crop_pad`.
6. **Model loading:** the model is loaded only when there is at least one window. A decoding failure skips that group of windows with a warning instead of failing the stage.

### Identity and handedness: `players.json`

The spec assumes a single player on the near side (re-identification is a non-goal). Real sessions have two players who change ends, so stage 4 works out who is who (`tennis/util/identity.py`):

1. **Match the two players.** For each pose window, it takes the median appearance of the near and the far player. It then matches these to two identities with a two-cluster assignment, with the rule that the two players in a window are different people. A is the near player of the first window. The window assignments are smoothed by majority vote, because players only change ends between games.
2. **Attribute each contact** to the player whose wrist speed peaks highest near it, if that peak reaches `wrist_confirm_min_speed`.
3. **Decide which identity is you** (`player.identity`):
   - `auto`: the identity whose attributed hits are at least `identity_loudness_db` louder, which works with the clip-on mic; otherwise A;
   - `near_at_start`: always A;
   - `A` or `B`: set explicitly.
4. **Your racket hand** (`player.handedness: auto`): for your own near-side hits, it votes on which wrist peaks faster. With fewer than five votes, it assumes right-handed.
5. **Your swings.** Every candidate contact becomes a swing built from *your* keypoints, on whichever side you are. Far-side swings fail QC with `far side (not measured)`. `is_self_confirmed` also requires the hit to be attributed to you.

`players.json` records:

- the decision and its reason;
- loudness statistics;
- the handedness votes;
- the side segments;
- per-window assignments and margins;
- hit counts;
- the appearance prototypes.

`tennis players` prints this and writes a thumbnail sheet.

### `swings.parquet` and `swing_info.parquet` (schema_version 1)

Stage 4 turns every candidate contact into a swing. By default the candidates are the first-pass own hits; with `pose.contacts: all`, every onset becomes a candidate. The column lists are in the `tennis/stages/clean.py` docstring. In short:

- **`swings.parquet`** has one row per swing per frame. It holds `t_rel`, the normalized keypoints `<kp>_x`/`<kp>_y`, the cleaned pixel positions `<kp>_px`/`<kp>_py`, confidences, wrist velocities and speeds, a per-frame `valid` flag, and the swing's QC fields repeated on each row.
- **`swing_info.parquet`** has one row per swing. It holds the contact time and frame, the normalization (origin and scale in pixels), QC (`valid_frame_ratio`, `swap_count`, `track_reset_count`, `qc_pass`, `qc_reason`), and the confirmation result (`wrist_peak_speed`, `wrist_peak_offset_s`, `is_self_confirmed`). It also records `player_side` (the side you were on), `hitter` (`self`, `other` or null), and the near and far players' peak wrist speeds. `swings.parquet` has a `player_side` column too, and both files store `racket_side` in their metadata.

**Cleaning** runs once per pose window, in the spec's order:

1. mask low-confidence points;
2. fix left/right swaps (each paired group separately; confidences follow the swap);
3. interpolate gaps of up to `max_gap_frames`, using timestamps;
4. smooth. The default is One Euro; its `beta` is divided by the window's median torso length so it applies to torso-normalized speeds, and `zero_phase` averages a forward and a backward pass. Savitzky–Golay is also available.

**Normalization** puts the origin at the hip midpoint at the contact frame (or at the nearest frame with both hips visible). The scale is the swing's median torso length, and y points up.

**`t_rel`** is `t_video − t_contact`, where `t_contact` is the audio onset on the PTS timeline.

A frame is **valid** when the player is detected and the racket-side shoulder, elbow and wrist and both hips are present after cleaning. A swing fails QC when any of these hold:

- `valid frames / expected frames` is below `qc.min_valid_frame_ratio`;
- the racket arm is missing at the contact frame;
- it has more than `qc.max_track_resets` track resets;
- there is no torso length or no hip position.

**Confirmation:** the wrist speed must have a local maximum within ±`wrist_confirm_window_s` of the onset, and that maximum must reach `wrist_confirm_min_speed`. The speed is taken from the faster wrist by default, or only the racket wrist with `wrist_confirm_wrist: racket`. When several contacts share one wrist peak (for example a hit and a bounce), only the contact closest to the peak is confirmed.

`tennis eval-contacts` scores the stored flags against labels and sweeps these settings. `tennis swing-plots` plots raw vs cleaned trajectories. Tuning results are in `docs/validation/M4_cleaning.md`.

### `audio.wav`

Mono, 16-bit PCM, 48 kHz, decoded from the first audio stream. It is written with `-bitexact`, so repeated runs produce identical files.

The schemas for the later stage outputs are in spec section 6. They will be documented here as each stage is built.

## Open design points for later milestones

- **Resolved in M4: `is_self_confirmed` lives in `swing_info.parquet`, not in `contacts.parquet`.** The spec writes it back into `contacts.parquet`, which stages 3 and 4 read. In `contacts.parquet` the column stays null.
- **Stage 5 modifies `swings.parquet` in place.** A file cannot be both an input and an output under mtime caching. Proposed fix: stage 5 writes a separate `strokes.parquet`, and readers join it on `swing_id`.
