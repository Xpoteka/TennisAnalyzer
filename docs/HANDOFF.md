# Developer handoff: state of the project and plan for M5–M8

*Written 2026-09-17, at the end of M4. Read this first, then [ARCHITECTURE.md](../ARCHITECTURE.md) (how the stages fit together, data schemas, caching) and the [README](../README.md) (setup and CLI). The product spec, "Tennis Technique Analyzer: Software Specification v1.0", is the source of requirements; section numbers below (§6.5 and so on) refer to it.*

---

## 1. Where things stand

### Milestones

| # | Milestone | State | Acceptance |
|---|---|---|---|
| M1 | Scaffold, config, CLI, ingest, CI | Merged (PR #1) | Done. CI has never been seen running on GitHub; see §4. |
| M2 | Contact detection and tuning tool | Merged (PR #2) | **Open.** Needs about 100 frame-accurate own-hit labels from a clip-on-mic session (§10.3). Only coarse Wingfield labels exist so far. |
| M3 | Pose extraction, player selection, review video | Merged (PR #3) | Visual review by the product owner (PO). 97% of frames tracked on 20 Wingfield swings. No formal sign-off yet. |
| M4 | Cleaning, normalization, QC, confirmation, two-player identity | Merged (PR #4) | QC pass rate 86% for near-side own hits (target ≥ 80%). Trajectory plots done. See `docs/validation/M4_cleaning.md`. |
| M5 | Stroke classification and eval tool | Not started | ≥ 90% accuracy (§10.3) |
| M6 | Metrics, aggregation, outliers | Not started | All §6.6 metrics implemented, unit-tested, and identical across reruns |
| M7 | Clips and HTML report with trends | Not started | Report opens offline and shows every §6.9 section, with ≥ 3 sessions |
| M8 | Voice labels and label analysis | Not started | ≥ 80% of spoken label words matched correctly |

Start M5 from an up-to-date `main`.

### Codebase in one page

```
tennis/
  cli.py                typer app; main() maps errors to exit codes 0/1/2
  config.py             pydantic config; section_hash() accepts "section" or "section.field" keys
  session.py            session dirs and IDs; stage stamps with input/output fingerprints
  errors.py             UserError (exit 1), StageError (exit 2)
  stages/__init__.py    STAGES registry and run_pipeline(); unimplemented stages have run=None
  stages/ingest.py      1: metadata.json, audio.wav, frame_times.parquet
  stages/contacts.py    2: contacts.parquet; tune_contacts(); LabelSpec / LabelScorer
  stages/pose.py        3: keypoints.parquet (near and far slots, appearance descriptors)
  stages/clean.py       4: swings.parquet, swing_info.parquet, players.json
  pose_backends/        PoseBackend protocol, registry, YOLO backend
  util/audio.py         onset detection (chunked; spectral flux)
  util/frames.py        FrameReader (ffmpeg subprocess plus showinfo PTS), window merging
  util/tracking.py      near/far PlayerTracker, IoU, crop boxes
  util/appearance.py    clothing-color descriptors
  util/identity.py      A/B identity assignment across end changes
  util/cleaning.py      confidence mask, swap fix
  util/filters.py       One Euro (zero-phase option), Savitzky–Golay, gap filling
  util/geometry.py      angle_deg, midpoint, distance   (M6 builds on these)
  util/overlay.py       draw_pose, draw_label, VideoWriter (H.264 through ffmpeg)   (M7 clips)
  util/io.py            atomic writes; write_parquet with provenance; keeps mtime on identical rewrites
  util/video.py         ffprobe/ffmpeg wrappers, frame PTS, nearest_frames
  evaluation.py         eval-contacts (scores stored flags, sweeps confirmation settings)
  review.py             pose-preview, swing-plots, players thumbnails
  validation.py         label CSV parsing, parse_time, match_events (with label windows), read_segments
scripts/wingfield_labels.py   Wingfield .xlsx export → label CSVs
```

About 5,400 lines of code, and 172 tests (`uv run pytest`, about 8 s, no network, no real model). The same checks run in CI: ruff, ruff format, and mypy in strict mode.

### Data you can use

| What | Where | Notes |
|---|---|---|
| Wingfield match, 2025-01-10 (30 min, 720p, 30 fps, variable frame rate, court mic) | `~/Downloads/2025-01-10-Wingfield-Jerry-Josi.MP4`; session `2025-01-10_wingfield` | Pose was run around **every** onset (`pose.contacts: all`), so run with the scratch config in §5. Pose takes about 45 min on an M-series Mac. |
| Wingfield shot log | `~/Downloads/2025-01-10-Wingfield-Stats-Josi.xlsx` → `labels/*_2025-01-10_wingfield.csv` (git-ignored) | Whole-second times on Wingfield's clock: a label `t` means the hit is in [t+1, t+2] s. It includes **stroke types for both players**, which is the M5 test set. Regenerate with `uv run scripts/wingfield_labels.py <xlsx> 2025-01-10_wingfield`. |
| DJI session, 2026-09-16 (49 min, 1080p HEVC 10-bit, 23.976 fps, clip-on mic) | iCloud Drive: `DJI_20260916195322_0007_D.MP4`; session `2026-09-16_evening` | The camera sits **at court level**, so players are tiny and pose finds nobody. iCloud evicts the file (0 bytes on disk), and reads then stall. Only ingest and contacts have run. |

The PO (the player) is **left-handed** and wore an orange shirt in the Wingfield match. `player.handedness: auto` detects the left hand correctly.

---

## 2. Changes from the spec so far (keep them, or discuss with the PO)

1. **`is_self_confirmed` lives in `swing_info.parquet`, not in `contacts.parquet`.** Writing it back would make stages 3 and 4 stale forever.
2. **Stage 5 must not rewrite `swings.parquet` in place,** for the same reason. See M5 below.
3. **Two-player identity is added,** although §1.2 lists re-identification as a non-goal. Sessions have a partner and the players change ends. Without it, 30% of the Wingfield match measured the partner as "you". Far-side swings are detected but **not measured**; this was the PO's choice.
4. **Handedness `auto`** (default) and **identity `auto`**: louder clip-mic hits, otherwise the near player at the start. `player.identity: A|B` overrides it.
5. **Confirmation rule:** the highest *local* wrist-speed peak within ±0.2 s must reach 6 torso lengths/s. It uses the *faster* wrist, because left/right labels are unreliable from behind.
6. **Tuned defaults:**
   - One Euro `min_cutoff` 3.0 and `beta` 0.5 (spec: 1.0 and 0.05), plus `zero_phase: true`;
   - `wrist_confirm_window_s` 0.2 (spec: 0.15).
7. **`audio.min_prominence_db`** (6 dB) is added: the log-flux envelope produces false onsets on steady noise.
8. **Caching:**
   - config hashes are per stage and per field, not per file mtime;
   - stamps store input and output fingerprints (mtime and size);
   - identical rewrites keep their old mtime so reruns don't cascade;
   - **code changes don't invalidate anything** (see §4).
9. **No PyAV.** On macOS its bundled FFmpeg clashes with OpenCV's (which Ultralytics requires). Frames are decoded by an `ffmpeg` subprocess.
10. **Ultralytics is a core dependency** (AGPL-3.0). The backend stays swappable through the registry.
11. `ingest` also writes `frame_times.parquet`. Frame index `i` means row `i` of this file, always.

---

## 3. Plan for the remaining milestones

The general rules for every new stage are in the README under "Adding a stage":

- `run(ctx)` is a pure function of the stage's files;
- declare `INPUTS`, `OUTPUTS` and `CONFIG_KEYS` (field-level where possible);
- write with `write_parquet` or `write_json`;
- when one swing is bad, log it, mark it and continue;
- add tests with synthetic data (no real model, no network).

After adding a stage, set `run=` in `STAGES`. Tests derive the expected stage list from the registry (`implemented_stages()` in `tests/conftest.py`), so the older tests keep passing.

### M5: stroke classification (§6.5), about 4–5 days

**Registry change:** stage 5 reads `swings.parquet` and `swing_info.parquet` and **writes `strokes.parquet`**, one row per swing: `swing_id`, `stroke_type`, `two_handed`, `classifier_version`, and the features it used. Downstream readers join on `swing_id`. Update the `Stage(5, …)` entry and ARCHITECTURE.md.

**Classifier interface** (`tennis/stages/classify.py`):

```python
class Classifier(Protocol):
    version: str
    def classify(self, swing: SwingFeatures) -> StrokeResult: ...
```

`SwingFeatures` holds the per-swing values the rules need, computed once from `swings.parquet`:

- racket and non-racket wrist positions at contact;
- nose height;
- wrist travel before contact;
- the player's box bottom as a fraction of image height (from `keypoints.parquet`: near slot at the contact frame, or add it to `swings.parquet`);
- `player_side`.

Keep `RuleClassifier` as the v1 implementation. A later gradient-boosting classifier plugs into the same interface. Add a small registry, the same pattern as the pose backends.

**Rules, evaluated in order** (normalized units: y up, origin at the hip midpoint):

1. **Serve:** racket wrist y > nose y + `serve_wrist_above_nose`.
2. **Volley:**
   - the racket wrist's path length over [−0.5, 0] s is below `volley_travel_max`,
   - **and** the box bottom is above `volley_bbox_bottom_max_y` (the player is near the net).
3. **Forehand:** the racket wrist is on the racket-hand side of the hip midpoint. **The sign matters:**
   - normalized x equals image x (not mirrored). From behind the baseline, the player's right is image right;
   - right-handed: forehand when `wrist_x > 0`; left-handed: when `wrist_x < 0`;
   - `camera_side: side_on` needs a different rule, so reject it with a clear error until it's specified;
   - use `resolved_handedness()` from `stages/clean.py` (it reads `players.json`).
4. **Backhand:** otherwise.
5. **Two-handed:** the non-racket wrist is within `two_handed_max_dist` of the racket wrist at contact.

Classify only swings with `is_self_confirmed`, `qc_pass` and `player_side == "near"`. Write `stroke_type = null` for the rest, rather than dropping the rows.

**`tennis eval-classifier <session> --labels …`:**

- **Label files:** accept the Wingfield strokes format (`t,player,stroke,...`, keeping only `player == self`) and a plain `t,stroke` format.
- **Label clock:** reuse `LabelSpec` and `make_scorer` (`--label-offset`, `--label-resolution`, `--segments`) to match labels to confirmed swings with `match_events`. Only matched pairs count toward accuracy.
- **Output:** a confusion matrix plus per-class precision and recall, and optionally a markdown report written to `docs/validation/M5_classifier.md`.
- **Test set:** the Wingfield export has about 144 own shots (43 serves, 55 forehands, 40 backhands, 6 volleys). Only near-side shots can be scored, and only those that are also confirmed swings.

**Risks:**

- the 30 fps, 230 px footage makes wrist positions noisy;
- serves are easy, while volleys are rare and the rule is fragile;
- the Wingfield "VOLLEY" label includes both forehand and backhand volleys; the spec merges them.

**Tests:** hand-built pose fixtures for each class, for **both handedness values**, following the style of `tests/integration/test_clean.py`, plus the eval matching and the confusion matrix.

### M6: metrics (§6.6), about 5 days

**Stage 6 contract:**

- **Inputs:** `swings.parquet`, `swing_info.parquet`, `strokes.parquet`.
- **Output:** `metrics.parquet`, one row per swing:
  - `swing_id`, `stroke_type`;
  - all the metrics, with NaN where a metric doesn't apply;
  - `is_outlier`, `outlier_score`.
- **Which swings:** only those with `qc_pass`, `is_self_confirmed` and near side.

**Registry** (`tennis/stages/metrics.py`). Adding a metric must not require edits anywhere else:

```python
@metric("elbow_angle_contact", applies_to={"serve", "forehand", "backhand", "volley"})
def elbow_angle_contact(s: SwingFrame, racket: Side) -> float: ...
```

`SwingFrame` gives time-indexed access to the normalized keypoints: `at(t_rel)`, `between(a, b)`, and `contact` (the frame nearest `t_rel = 0`). `racket` is `"l"` or `"r"`.

Implement every metric in the §6.6 table. Notes:

- **Angles:** use `util/geometry.angle_deg`.
- **`contact_forward`:** the sign depends on the camera and on handedness. Multiply by `player.forward_sign`, and document the convention with a drawing in ARCHITECTURE.md. From behind, "in front of the body" is up the court, which in 2-D shows up as smaller image y, so larger normalized y. Verify this against the Wingfield frames before trusting it.
- **Wrist speed:** `peak_wrist_speed` and `peak_speed_offset` use the stored `<side>_wrist_speed`. Remember the smoothing trade-off (see `test_light_smoothing_keeps_peak_speed`).
- **`shoulder_turn_proxy_min`:** the ratio of apparent shoulder width to its value at `t_rel = −1.0`. It is NaN if the window starts later than that.

**Aggregation:** per stroke type, compute count, mean, std ("consistency"), median, p10 and p90. Store these either in a small `metrics_summary.parquet` or in the report stage; decide which, and document it.

**Outliers:**

- Mahalanobis distance on `metrics.outlier_metrics`, per stroke type.
- Use a robust or regularized covariance (`np.cov` plus a small ridge; fall back to standardized Euclidean distance when there are fewer than about 10 swings).
- Flag swings above the `outlier_percentile`.
- Keep it deterministic: no random sampling.

**Tests:**

- one test per metric on a hand-built swing with known geometry (the spec requires this);
- a determinism test: run twice and check the Parquet files are byte-identical apart from metadata;
- use `util/io` equality helpers.

### M7: clips and HTML report (§6.8, §6.9, `tennis report`, `tennis trends`, `tennis inspect`), about 6–8 days

**Dependencies:** add `jinja2` and `duckdb` (plotly is already a dependency).

**Stage 8, clips:**

- **Selection:**
  - every outlier;
  - the swing closest to the median, per stroke type;
  - every `good` or `late` swing (after M8), capped at `clips.max_per_label`.
- **Rendering:** decode `[t − 1.2, t + 0.8]` with `FrameReader` and draw with `util/overlay.py`: the skeleton (racket side already colored), a red border on the contact frame, and stroke type plus key metrics as text. Write with `VideoWriter` (H.264, 720p). Optional slow motion: set `fps × clips.slow_motion_rate`.
- **Output:** `clips/<swing_id>.mp4`, plus `clips/index.json` listing why each clip was chosen.
- **Caching:** use `clips/index.json` as the stage output, not the directory. A directory's mtime changes whenever any file changes, which works badly with fingerprints.
- **Performance:** clips need the source video, so add `@source` to the stage inputs. They are slow on long sessions, so decode only the selected swings.

**Stage 9, report:** `templates/report.html.j2` (set up hatch to package `tennis/templates`). It must be a single self-contained file with Plotly inlined (`include_plotlyjs="inline"` once) and clips linked by relative path. It has these sections:

1. **Session summary:**
   - date, duration, contacts, own swings, QC pass rate, swings per stroke type;
   - from `players.json`: who is you, the hand, and the side timeline.
2. **Metrics table:**
   - mean ± std per stroke type;
   - deltas against the rolling mean of the last `report.rolling_sessions` sessions, green or red. Which direction is "good" is metric-specific; ask the PO, otherwise show the delta without judging it.
3. **Trends:** DuckDB over `data/sessions/*/metrics.parquet` (read with `read_parquet(glob, filename=true)`), for each `report.trend_metrics`, as session mean with a ± std band.
4. **Distributions:** a histogram per metric.
5. **Label analysis** (after M8): Cohen's d between `good` and each negative label, per metric, sorted by |d|.
6. **Clip gallery.**
7. **Diagnostics:**
   - ingest warnings (e.g. low fps);
   - contact counts and first-pass/confirmed numbers;
   - swap counts, track resets, far-player detection rate;
   - identity margins.

**CLI commands:**

- `tennis report <id>`: reruns stages 8–9 for that session.
- `tennis trends [--since]`: writes `data/report.html`.
- `tennis inspect <id> <swing_id>`: prints that swing's metrics and opens its clip with `open` on macOS or `xdg-open` on Linux.

**Acceptance:** the report must open offline with ≥ 3 sessions. Only two sessions exist, and one of them (DJI) has no pose, so the PO has to record more; see §6.

### M8: voice labels (§6.7), about 4 days

**Dependencies:** add `faster-whisper`. The model is downloaded on first use into `data/models/whisper`. Pass `download_root` so nothing lands in `~/.cache` unexpectedly.

**Stage 7, labels:**

- **Order:** it is optional (`labels.enabled`, `--no-labels`). Its inputs should be `audio.wav`, `contacts.parquet` and `swing_info.parquet`.
- **Transcription:** run faster-whisper with word timestamps, using the configured model and language.
- **Matching words:** map each word through `labels.vocabulary` (case-insensitive, strip punctuation), then attach it to the latest **confirmed** own contact within `[0, max_delay_s]` before the word. Log unmatched words and discard them.
- **Manual labels:** also read `labels/manual_<session>.csv` (`contact_id,label`); manual labels override voice labels.
- **Output:** `labels.parquet` with `swing_id`, `label`, `word`, `t_word`, `confidence`, and `source` (voice or manual).
- **Order fix:** stage 7 currently runs *after* metrics (6) but *before* clips (8) and report (9), which consume it. That's fine, but make clips and report list `labels.parquet` as an *optional* input: the runner currently requires every input to exist, so add support for optional inputs to `Stage`.

**Audio and matching notes:**

- The clip-on mic records the player's own voice, which is what we want.
- The `audio.wav` WAV time maps to PTS as `t + audio_start_s`.
- Whisper timestamps are relative to the WAV.
- Default language: the PO hasn't said. §14 lists this as an open question; the PO's name suggests French might matter. Make `labels.language` and the vocabulary per session or per config.

**Acceptance:** ≥ 80% of spoken label words correctly matched. The PO needs to record a session saying the words (see §6), then label the truth by hand in a small CSV. Add `tennis eval-labels` alongside `eval-contacts`.

**Tests:** synthetic transcripts through the matcher (mock the whisper call); no model in CI.

### Cross-cutting work, still owed from the spec

- **§10.2 integration fixture:** a real, committed clip of about 20 s with 3–5 impacts. The pipeline must run end to end in CI on the CPU with a small pose model (`yolo11n-pose.pt`, 6 MB, downloaded in CI and cached). Today, tests generate synthetic clips with ffmpeg, and the YOLO backend is stubbed (`_no_real_pose_model` in `tests/conftest.py`). Add the fixture and mark the test `real_yolo`.
- **§10.3 determinism check:** after M6, run a session twice with the cache disabled (`--force`) and require identical metrics. Watch for GPU nondeterminism: compare with tolerance on MPS/CUDA, and exactly on CPU.
- **§9 performance:** a 60-min 4K session at 120 fps with about 600 swings must run in ≤ 45 min on an RTX 3060. It is untested. Ideas:
  - `pose.hwaccel: cuda`;
  - half precision (`half=True` in `YoloBackend.predict`);
  - larger batches;
  - skipping the far pass on frames where the near player is not near a contact;
  - `track_far: false` when the mic identifies the player reliably.

  Measure on the PO's target GPU (§14: "TrueNAS host or desktop?").
- **§9 memory ≤ 8 GB:**
  - `clean.run` loads `keypoints.parquet` whole. With `pose.contacts: all` on a 4K 120 fps session, that can be millions of rows; check it, and stream per window if needed.
  - `swings_table` concatenates all swings in memory.
- **Deliverables (§12):** README (keep it current), `config.example.yaml` (every option commented; a test checks it matches the defaults), ARCHITECTURE.md, and validation notes for M2, M5 and M8.

---

## 4. Known issues and traps

| Issue | Details / what to do |
|---|---|
| **Code changes don't invalidate caches** | Stamps hash config and file fingerprints, not code. After changing a stage's logic, run `tennis process … --from-stage N`. Consider adding a per-stage `VERSION` constant to the stamp. |
| **`gh` isn't logged in on the dev Mac** | Pushing works over SSH (`origin = git@github.com:Xpoteka/TennisAnalyzer.git`), but PRs had to be opened through prefilled compare links. Run `gh auth login --git-protocol ssh --web` once. |
| **CI has never been seen running** | `.github/workflows/ci.yml` exists, but the Actions API showed no runs after the first pushes. Check that Actions is enabled on the repo. On Linux, `uv sync` pulls PyTorch with CUDA (about 2.5 GB). Consider a CPU-only torch index for CI (`[tool.uv.sources]` with a marker) and cache the uv directory. |
| **iCloud-evicted videos** | A file with 0 blocks on disk stalls `ffmpeg` and `ffprobe` while iCloud downloads it. Ingest could warn when `st_blocks == 0 and st_size > 0`; add that. Keep raw footage local or on the NAS (see the README recording guidelines). |
| **DJI camera placement** | At court level, far players are about 30 px tall and near players are rarely framed. For pose to work, the tripod must be raised behind the baseline, as in the spec and as on the Wingfield court. It is a recording issue, not a code issue. |
| **Low frame rate** | The DJI footage is 23.976 fps (ingest warns). Contact timing comes from audio, so it's fine, but wrist speeds and contact frames are coarse. Ask the PO to record at 100–120 fps. |
| **Far-player detection is weak** | 12% of frames on Wingfield. Identity and side timeline are still correct (they vote per window). Before tuning, check `players.jpg` (`tennis players`). The far slot sometimes tracks the partner at the net, or people at the bench. |
| **Swap fix changes 7% of shoulder frames** | These are probably real crossings seen from behind during the shoulder turn. Turning it off doesn't change the confirmation F1. It's kept because the spec requires it; revisit it when M6 shoulder-turn metrics look wrong. |
| **Own-hit detection is F1 0.68 on Wingfield** | That match has a court mic and whole-second labels, so it isn't a fair test of the §10.3 target. Real validation needs a clip-on-mic session with frame-accurate labels (`labels/README.md`). |
| **`--from-stage` on the Wingfield session** | Always pass the scratch config (`pose.contacts: all`). Without it, the default `self_audio` changes the pose windows and triggers a 45-minute rerun. |
| **Session ID from the container date** | The Wingfield MP4 says 2025-01-16 but was recorded on 2025-01-10. Use `--session-id` for exported files. |

---

## 5. Working on it

```bash
uv sync                                   # Python 3.11, all deps (includes torch and ultralytics)
uv run pytest -q                          # ~8 s
uv run ruff check . && uv run ruff format --check . && uv run mypy

# Wingfield session (pose already done; stages 4+ rerun in seconds)
cat > /tmp/wingfield.yaml <<'YAML'
paths:
  data_root: /Users/jeremiehamel/dev/repos/TennisAnalyzer/data
pose:
  contacts: all
YAML
uv run tennis process ~/Downloads/2025-01-10-Wingfield-Jerry-Josi.MP4 \
  --session-id 2025-01-10_wingfield --config /tmp/wingfield.yaml --from-stage 4
uv run tennis players 2025-01-10_wingfield --config /tmp/wingfield.yaml
uv run tennis eval-contacts 2025-01-10_wingfield --config /tmp/wingfield.yaml \
  --labels labels/contacts_2025-01-10_wingfield.csv \
  --label-offset 1 --label-resolution 1 --segments labels/rallies_2025-01-10_wingfield.csv
```

**Conventions:**

- **Branches and PRs:** one branch per milestone (`m5-classify`, …), one PR each. Commit messages end with the co-author trailer used so far.
- **Where to write things down:** design notes go in the module docstrings (see `clean.py` and `pose.py`) and in ARCHITECTURE.md. Validation results go in `docs/validation/`.
- **Changes from the spec:** write each one down in ARCHITECTURE.md and in §2 of this file, with the data that justified it.
- **Tests:** never download a model or touch the network. Use synthetic clips (`tests/conftest.py`: `make_video`, `counter=True` for frame-number checks, `audio_wav=` for clicks) and hand-built keypoint fixtures.

---

## 6. Questions for the product owner

1. **Recording setup:** can the camera be raised behind the baseline (like the Wingfield camera), at 100–120 fps? This blocks useful pose on the DJI footage.
2. **Labels for M2:** about 100 own hits, to the frame, from one DJI clip-on-mic session (format in `labels/README.md`).
3. **More sessions:** M7 needs ≥ 3 sessions with pose, and M8 needs one session with spoken label words.
4. **Voice labels:** which language(s) and vocabulary? (§14)
5. **Target machine for the 45-minute performance budget:** TrueNAS host or desktop, and which GPU? (§14)
6. **Serves:** are they filmed from another camera position? If so, config profiles per session type are needed. (§14)
7. **Good or bad metric changes:** for each metric, which direction is an improvement? This sets the green/red deltas in the report (§6.9).
8. **Changes from the spec:** confirm the changes in §2, especially two-player identity and far-side swings not being measured.
