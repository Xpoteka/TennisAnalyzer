# Architecture

## Stages

The pipeline is a fixed sequence of stages. Each stage reads files from the session directory and writes files back to it. Stages never share in-memory state.

| # | Stage | Inputs | Outputs | Config sections | Milestone |
|---|-------|--------|---------|-----------------|-----------|
| 1 | ingest | source video | `metadata.json`, `audio.wav`, `frame_times.parquet` | none | **M1 (done)** |
| 2 | contacts | `audio.wav`, `metadata.json`, `frame_times.parquet` | `contacts.parquet` | audio | **M2 (done, needs tuning)** |
| 3 | pose | `contacts.parquet`, `frame_times.parquet`, `metadata.json`, source video | `keypoints.parquet` | pose, windows | **M3 (done, pending review)** |
| 4 | clean | `keypoints.parquet`, `contacts.parquet`, `frame_times.parquet` | `swings.parquet`, `swing_info.parquet` | player, cleaning, windows, `pose.kp_conf_min`, `pose.contacts`, `audio.wrist_confirm_*` | **M4 (done)** |
| 5 | classify | `swings.parquet`, `swing_info.parquet`, `keypoints.parquet`, `metadata.json` | `strokes.parquet` | player, classify | **M5 (done)** |
| 6 | metrics | `swings.parquet`, `swing_info.parquet`, `strokes.parquet` | `metrics.parquet`, `metrics_summary.parquet` | player, metrics | **M6 (done)** |
| 7 | labels (optional) | `audio.wav`, `contacts.parquet`, `swing_info.parquet`, `metadata.json` | `labels.parquet` | labels | **M8 (done)** |
| 8 | clips | source video, `metrics.parquet`, `swings.parquet`, `swing_info.parquet`, `strokes.parquet`, `frame_times.parquet`, `metadata.json` (+ optional `labels.parquet`) | `clips/index.json` and `clips/*.mp4` | clips, player, `pose.hwaccel`, `pose.kp_conf_min` | **M7 (done)** |
| 9 | report | `metrics.parquet`, `metrics_summary.parquet`, `swing_info.parquet`, `strokes.parquet`, `metadata.json` (+ optional `labels.parquet`, `clips/index.json`, `players.json`, `keypoints.parquet`) | `report.html` | report, player | **M7 (done)** |

The registry lives in `tennis/stages/__init__.py`. The runner (`run_pipeline`) processes the stages in order:

1. It skips stages that are up to date.
2. It stops at the first stage that is not implemented yet. Every stage is implemented now, so this only matters for a stage added later.
3. It wraps any exception in `StageError`, so the CLI exits with code 2 and prints the stage name.

**Optional inputs.** A stage may declare `optional_inputs`: files it uses when they exist. They are not required to run it, but the stage is stale when one appears, changes or disappears. That is how switching the voice labels on rebuilds the clips and the report without `--force`, while a session with no labels still runs.

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

For this comparison **two NaNs count as equal** (`tennis.util.io.same_data`). Arrow follows IEEE 754, where they do not, but a keypoint that was never tracked or a metric that does not apply is the same result on the next run. Without this, every stage that writes NaN — stage 4 and stage 6 — would invalidate the stages after it on every rerun.

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

### `strokes.parquet` (schema_version 1)

One row per swing, joined to everything else on `swing_id`.

| Column | Type | Meaning |
|--------|------|---------|
| `swing_id`, `contact_id`, `t_contact` | int64, int64, float64 | Identifiers and the contact time |
| `stroke_type` | string | `serve`, `forehand`, `backhand`, `volley`, or null when the swing was not classified |
| `two_handed` | bool | Both wrists within `classify.two_handed_max_dist` at contact; null when the other wrist is not tracked |
| `classifier_version`, `rule` | string | Which classifier ran, and which rule fired (or why the swing was skipped) |
| `player_side`, `handedness` | string | The side you were on, and your racket hand |
| `racket_wrist_x/_y`, `other_wrist_x/_y`, `nose_y` | float64 | The normalized positions at contact the rules used |
| `wrist_travel` | float64 | Racket-wrist path length over the 0.5 s before contact |
| `wrist_gap` | float64 | Distance between the wrists at contact |
| `bbox_bottom_y` | float64 | Player's box bottom as a fraction of the image height |

Only swings that are `is_self_confirmed`, `qc_pass` and `player_side == "near"` are classified. The rest keep their row with a null `stroke_type` and the reason in `rule`, so joins stay total.

**The rules, in order** (normalized units: y up, origin at the hip midpoint at contact, one unit = the median torso length):

1. **serve** — the racket wrist is more than `serve_wrist_above_nose` above the nose;
2. **volley** — the racket wrist travelled less than `volley_travel_max` in the 0.5 s before contact **and** the box bottom is above `volley_bbox_bottom_max_y` (the player is near the net);
3. **forehand** — the racket wrist is on the racket-hand side of the hip midpoint. Normalized x is image x, so from behind the baseline a right-handed player's forehand has `racket_wrist_x > 0` and a left-handed player's has `racket_wrist_x < 0`;
4. **backhand** — anything else.

`player.camera_side: side_on` is rejected with an error: rule 3 assumes the camera is behind the baseline, and the side-on rule is not specified.

Classifiers go through a registry (`classify.classifier`, `rule` built in), the same pattern as the pose backends, so a learned classifier can replace the rules without touching anything else. `tennis eval-classifier` scores the stored types against labelled shots.

### `metrics.parquet` and `metrics_summary.parquet` (schema_version 1)

`metrics.parquet` has one row per swing that is yours, passes QC, is on the near side and got a stroke type: `swing_id`, `contact_id`, `t_contact`, `stroke_type`, `two_handed`, one float64 column per registered metric, and `outlier_score` / `is_outlier`. A metric is NaN when it does not apply to that stroke type or could not be computed.

`metrics_summary.parquet` has one row per stroke type per metric: `count`, `mean`, `std` (the spec's "consistency"), `median`, `p10` and `p90`. **The aggregates live in their own file** rather than in the report stage, so `tennis trends` and the delta columns read them off disk instead of recomputing them per session. A metric with no finite value gets no summary row.

Metrics are registered functions in `tennis/stages/metrics.py`; see "Adding a metric" in the README. The ones that ship:

| Metric | Applies to | Unit | Meaning |
|--------|-----------|------|---------|
| `contact_height` | all | torso | Racket wrist above the ground (ankle midpoint) at contact |
| `contact_forward` | all | torso | Racket wrist up-court of the body centre at contact, times `player.forward_sign` |
| `contact_lateral` | all | torso | Racket wrist to the racket-hand side of the body centre |
| `contact_reach` | all | torso | Shoulder-to-wrist distance at contact |
| `elbow_angle_contact` | all | deg | Shoulder–elbow–wrist angle at contact; 180 is straight |
| `elbow_angle_min` | all | deg | Smallest elbow angle in the 0.5 s before contact |
| `knee_flex_min` | all | deg | Deepest knee bend of the swing (mean of both knees) |
| `knee_flex_contact` | all | deg | Mean knee angle at contact |
| `peak_wrist_speed` | all | torso/s | Highest racket-wrist speed in the window |
| `peak_speed_offset` | all | s | When that peak happened, relative to contact |
| `shoulder_turn_proxy_min` | serve, forehand, backhand | ratio | Narrowest apparent shoulder width, over its width at `t_rel = −1.0` |
| `unit_turn_lead_time` | forehand, backhand | s | How long before contact that ratio first drops below `metrics.unit_turn_ratio` |
| `follow_through_height` | serve, forehand, backhand | torso | Racket wrist above the hips at the end of the window |
| `torso_lean_contact` | all | deg | Tilt of the hip-to-shoulder line from vertical, towards the racket side |

> **The spec's section 6.6 table was not available in this repository.** The list above is built from the metric names the config and the handoff already referenced, plus the closely related ones the same geometry supports. Reconcile it with the spec before the next milestone; adding or renaming a metric is one edit in `tennis/stages/metrics.py`.

#### The forward/height convention

```
                       net (up the court)
                            ^
   camera behind the       |          Moving up the court and moving UP both
   baseline, raised   ->   |          make image y SMALLER, so both raise the
                           |          normalized y. One camera cannot tell
        o  <- contact      |          them apart.
       /|\                 |
       / \                 |
    ---+-------------------+---  the player, seen from behind
```

Because a single camera behind the baseline projects court depth and height onto the same image axis, the two metrics are referenced differently so that each is still meaningful on its own:

- **`contact_height`** is measured from the ground: the racket wrist's normalized y minus the ankle midpoint's y.
- **`contact_forward`** is measured from the body centre: the racket wrist's normalized y minus the hip midpoint's y, multiplied by `player.forward_sign` (`-1` for a mirrored setup).

They remain correlated. That is a property of the camera, not of the code — the outlier distance uses a ridge-regularized covariance so it copes. **Check this against real Wingfield frames before trusting `contact_forward` as a depth measure**, and raise it with the product owner (open question 7).

#### Outliers

Per stroke type, the swings with all of `metrics.outlier_metrics` finite are scored by Mahalanobis distance from that group's mean. The covariance gets a ridge of `metrics.outlier_ridge` times its mean variance before inversion, so a collinear or degenerate set of metrics still produces finite distances. Below `metrics.min_swings_for_covariance` swings the covariance is not worth estimating, so each column is standardized and the plain Euclidean norm is used instead. Swings **strictly above** the `metrics.outlier_percentile` of their group's scores are flagged; the strict comparison matters because with a flat score distribution the percentile equals every score. Nothing samples and nothing uses a random state, so two runs produce byte-identical files (a test checks this).

### `labels.parquet` (schema_version 1)

One row per label. Stage 7 is optional (`labels.enabled`, `--no-labels`).

| Column | Type | Meaning |
|--------|------|---------|
| `swing_id`, `contact_id` | int64 | The swing the label belongs to |
| `label` | string | The vocabulary entry (`good`, `late`, …) |
| `word` | string | The spoken word it came from; empty for a manual label |
| `t_word` | float64 | When the word was said, on the PTS timeline |
| `t_contact` | float64 | The contact it was attached to |
| `confidence` | float64 | The transcriber's word probability; NaN for a manual label |
| `source` | string | `voice` or `manual` |

The audio is transcribed with word timestamps, each word is normalized (lower-cased, punctuation stripped) and looked up in `labels.vocabulary`, and the match is attached to the latest **confirmed** own contact within `labels.max_delay_s` before it. Whisper timestamps are WAV time, so `audio_start_s` is added to put them on the PTS timeline. Words that map to nothing, and words with no own hit before them, are logged with the reason and dropped.

`<paths.labels_dir>/manual_<session>.csv` (`contact_id,label`) overrides the voice labels for the same contact. A contact id that is not a confirmed own hit is an error, not a silent drop. That file is not a declared stage input, so add or edit it and rerun with `--from-stage 7`.

Transcription goes through a registry (`labels.backend`, `faster_whisper` built in); the model is downloaded to `<data_root>/models/whisper` rather than `~/.cache`. Tests register a backend that returns a fixed word list, so no model is ever loaded in CI. `tennis eval-labels` scores the stored labels against a CSV of what was actually said.

### `clips/index.json` and `clips/*.mp4`

`clips/index.json` is stage 8's **declared output** — not the directory. A directory's modification time changes whenever any file inside it does, which the fingerprint cache would read as a permanent change. It records the session, the playback rate, and per clip: `swing_id`, `t_contact`, `stroke_type`, `reasons`, `labels`, `metrics`, `file`, `start_s`, `end_s`, `frames` and `rendered` (plus `error` when one clip failed).

Which swings get a clip:

- every outlier;
- the swing closest to the median of each stroke type (metrics standardized first, so degrees do not outweigh torso lengths; ties break on the lower swing id, so the choice is stable);
- every labelled swing, at most `clips.max_per_label` per label.

A swing picked by more than one rule is rendered once and lists every reason. Clips no longer selected are deleted on the next run. Each frame carries the skeleton (racket side coloured), the stroke type, the labels and a few metrics; the contact frame gets a red border. `clips.slow_motion` writes the same frames at `fps × clips.slow_motion_rate`.

### `report.html`

One self-contained file. Plotly's script is inlined once and every figure is rendered without its own copy, so the page opens offline with no network and no sibling files; the clips are linked by relative path, so the report and its `clips/` folder travel together. The sections are the ones spec section 6.9 lists: session summary, metrics per stroke type, trends, distributions, label analysis, the clip gallery and diagnostics.

**Deltas.** The metrics table compares this session's mean with the mean of the last `report.rolling_sessions` sessions that have metrics. A delta is coloured green or red **only** for the metrics listed under `report.metric_direction` (`up` or `down`). For every other metric the direction that counts as an improvement has not been decided (open question 7), so the change is shown without a judgement.

**Cross-session data** comes from DuckDB over `data/sessions/*/metrics.parquet`, read with `read_parquet(glob, filename = true)` so each row knows its session. There is a plain-Python fallback for the case where DuckDB cannot be imported; a test holds the two to the same answers. `tennis trends` writes the same figures for every session to `<data_root>/report.html`.

**Label analysis** (after M8) lists Cohen's d between the `good` swings and each other label, per metric, largest effect first.

### `audio.wav`

Mono, 16-bit PCM, 48 kHz, decoded from the first audio stream. It is written with `-bitexact`, so repeated runs produce identical files.

## Changes from the spec

Each of these is also listed in `docs/HANDOFF.md` §2, with the data that justified it.

- **`is_self_confirmed` lives in `swing_info.parquet`, not in `contacts.parquet`** (M4). The spec writes it back into `contacts.parquet`, which stages 3 and 4 read, so the cache would call them stale forever. In `contacts.parquet` the column stays null.
- **Stage 5 writes `strokes.parquet` instead of rewriting `swings.parquet` in place** (M5), for the same reason: a file cannot be both an input and an output under fingerprint caching. Readers join on `swing_id`.
- **The aggregates live in `metrics_summary.parquet`** (M6), so the report and the trends read them instead of recomputing them.
- **`clips/index.json` is stage 8's output, not the `clips/` directory** (M7).
- **`report.metric_direction` is opt-in** (M7). Without it, deltas are shown but not judged.
- **Metrics needed a registry with `applies_to`** (M6), because the four stroke types do not share every metric.
- **Stages may declare optional inputs** (M8), so the optional voice-label stage can feed the clips and the report without making them fail when it is skipped.
