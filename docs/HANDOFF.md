# Developer handoff: state of the project

*Written 2026-09-17 at the end of M4, updated at the end of M8. Read this first, then [ARCHITECTURE.md](../ARCHITECTURE.md) (how the stages fit together, data schemas, caching) and the [README](../README.md) (setup and CLI). The product spec, "Tennis Technique Analyzer: Software Specification v1.0", is the source of requirements; section numbers below (§6.5 and so on) refer to it.*

**Every stage of the pipeline is now built (M1–M8).** What remains is validation on real footage and the cross-cutting work in §3 — and all of it is blocked on recordings that do not exist yet (§6). Read §1, then §6.

---

## 1. Where things stand

### Milestones

| # | Milestone | State | Acceptance |
|---|---|---|---|
| M1 | Scaffold, config, CLI, ingest, CI | Merged (PR #1) | Done. CI has never been seen running on GitHub; see §4. |
| M2 | Contact detection and tuning tool | Merged (PR #2) | **Open.** Needs about 100 frame-accurate own-hit labels from a clip-on-mic session (§10.3). Only coarse Wingfield labels exist so far. |
| M3 | Pose extraction, player selection, review video | Merged (PR #3) | Visual review by the product owner (PO). 97% of frames tracked on 20 Wingfield swings. No formal sign-off yet. |
| M4 | Cleaning, normalization, QC, confirmation, two-player identity | Merged (PR #4) | QC pass rate 86% for near-side own hits (target ≥ 80%). Trajectory plots done. See `docs/validation/M4_cleaning.md`. |
| M5 | Stroke classification and eval tool | Built | **Fails.** Measured on 2026-09-17 against the Wingfield stroke labels: **58% accuracy** (49/84 matched swings; target ≥ 90%). Serves are the main miss. See §3 and `docs/validation/M5_classifier.md`. |
| M6 | Metrics, aggregation, outliers | Built | Metrics implemented with one unit test each, and a determinism test that requires byte-identical Parquet across reruns. On the Wingfield session, forced reruns of stages 4–6 give byte-identical `strokes`, `metrics` and `metrics_summary` files (checked 2026-09-17). **Caveat: the spec's §6.6 table was not in the repository**, so the metric list was reconstructed — see below. |
| M7 | Clips and HTML report with trends | Built | **Open.** The report opens offline and has every §6.9 section, but "with ≥ 3 sessions" cannot be shown: only two sessions exist and one has no pose. |
| M8 | Voice labels and label analysis | Built | **Open.** `tennis eval-labels` exists and is tested on synthetic transcripts; the ≥ 80% match rate needs a session recorded with the words spoken. |

**The §6.6 metric table was missing.** No copy of the spec is in the repository, so stage 6 ships the metrics that the config and this document already named (`contact_height`, `contact_forward`, `elbow_angle_contact`, `knee_flex_min`, `peak_wrist_speed`, `peak_speed_offset`, `shoulder_turn_proxy_min`, `unit_turn_lead_time`) plus the closely related ones the same geometry supports, fourteen in all; they are listed in ARCHITECTURE.md. **Reconcile that list against §6.6 before calling M6 accepted.** Adding, renaming or dropping one is a single edit in `tennis/stages/metrics.py`.

There is also a real limit worth raising with the PO: with one camera behind the baseline, court depth and height project onto the same image axis, so `contact_forward` and `contact_height` cannot be separated. They are referenced differently (ground vs body centre) so each is meaningful, but `contact_forward` is not a depth measurement. ARCHITECTURE.md has the drawing.

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
  stages/classify.py    5: strokes.parquet; Classifier protocol, RuleClassifier, registry
  stages/metrics.py     6: metrics.parquet, metrics_summary.parquet; @metric registry, SwingFrame
  stages/labels.py      7: labels.parquet; Transcriber registry, faster-whisper, word matching
  stages/clips.py       8: clips/index.json and clips/*.mp4; which swings are worth a clip
  stages/report.py      9: report.html (a thin wrapper around reporting.py)
  reporting.py          session and trends HTML; DuckDB over every session's metrics
  templates/            jinja2: base.html.j2, report.html.j2, trends.html.j2
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
  evaluation/           one module per eval command: contacts.py, strokes.py, labels.py
  review.py             pose-preview, swing-plots, players thumbnails, inspect
  validation.py         label CSV parsing, parse_time, match_events (with label windows), read_segments
scripts/wingfield_labels.py   Wingfield .xlsx export → label CSVs
```

About 8,700 lines of code, and 281 tests (`uv run pytest`, about 45 s with ffmpeg installed, no network, no real model, no whisper). Tests marked `needs_ffmpeg` are **skipped** when ffmpeg is missing, so a green run without ffmpeg proves little: six of them were broken by M8 and went unnoticed until 2026-09-17. The same checks run in CI: ruff, ruff format, and mypy in strict mode.

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
2. **Stage 5 writes `strokes.parquet` instead of rewriting `swings.parquet` in place,** for the same reason. Readers join on `swing_id`.
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
   - for that comparison two NaNs count as equal, unlike Arrow's own equality; without this every stage that writes NaN (4 and 6) would invalidate its successors on every rerun;
   - stages may declare **optional inputs**: files they use when present, whose appearance or disappearance still makes the stage stale;
   - **code changes don't invalidate anything** (see §4).
9. **No PyAV.** On macOS its bundled FFmpeg clashes with OpenCV's (which Ultralytics requires). Frames are decoded by an `ffmpeg` subprocess.
10. **Ultralytics is a core dependency** (AGPL-3.0). The backend stays swappable through the registry.
11. `ingest` also writes `frame_times.parquet`. Frame index `i` means row `i` of this file, always.
12. **Per-stroke-type aggregates live in `metrics_summary.parquet`** (M6), so the report and `tennis trends` read them rather than recomputing them.
13. **`clips/index.json` is stage 8's output, not the `clips/` directory** (M7): a directory's mtime changes whenever any file in it does.
14. **Report deltas are not judged unless asked** (M7). `report.metric_direction` opts a metric into a green/red delta; everything else shows the change without a verdict, because which way is better is open question 7.
15. **`contact_forward` cannot be a true depth measurement** from one camera behind the baseline (M6). It and `contact_height` use different references so each is meaningful, but they stay correlated. ARCHITECTURE.md has the drawing the plan asked for.
16. **Three registries follow the pose-backend pattern:** stroke classifiers (`classify.classifier`), transcription backends (`labels.backend`) and metrics (`@metric`).
17. **New dependencies:** `jinja2` and `duckdb` for the report, `faster-whisper` for the voice labels. The whisper model is cached under `<data_root>/models/whisper`, not `~/.cache`.

---

## 3. What each milestone shipped, and what is still open

The general rules for every new stage are in the README under "Adding a stage":

- `run(ctx)` is a pure function of the stage's files;
- declare `INPUTS`, `OUTPUTS` and `CONFIG_KEYS` (field-level where possible);
- write with `write_parquet` or `write_json`;
- when one swing is bad, log it, mark it and continue;
- add tests with synthetic data (no real model, no network).

Tests derive the expected stage list from the registry (`implemented_stages()` in `tests/conftest.py`), so adding a stage does not break the older tests.

### M5: stroke classification (§6.5)

Stage 5 reads `swings.parquet`, `swing_info.parquet`, `keypoints.parquet` and `metadata.json`, and writes **`strokes.parquet`**, one row per swing. It does *not* rewrite `swings.parquet`: that file is its own input, so rewriting it would make the stage permanently stale.

`RuleClassifier` implements the spec's cascade (serve → volley → forehand → backhand), with the forehand sign taken from `resolved_handedness()`, and marks two-handed strokes from the gap between the wrists. Classifiers go through a registry (`classify.classifier`), so a gradient-boosting classifier plugs into the same `Classifier` protocol later. `player.camera_side: side_on` is rejected with a clear error, as planned.

The box bottom the volley rule needs comes from `keypoints.parquet` (`bbox_y2` of the swing's slot at the contact frame), with the lowest cleaned keypoint as a fallback, rather than being added to `swings.parquet` — that would have meant rerunning stage 4.

**Result (2026-09-17): 58% accuracy, below the 90% target.** The run matched 84 labelled own shots to classified swings, out of 144 labels and 194 swings in range:

| truth \ predicted | serve | forehand | backhand | volley | recall |
|---|---|---|---|---|---|
| serve | 7 | 12 | 7 | 0 | 0.27 |
| forehand | 0 | 33 | 4 | 0 | 0.89 |
| backhand | 0 | 10 | 9 | 0 | 0.47 |
| volley | 0 | 2 | 0 | 0 | 0.00 |

The weakest rule is the serve rule. For labelled serves, the racket wrist at the contact frame is a median 0.1 torso lengths *below* the nose (10th–90th percentile: −1.05 to +0.67), while the rule requires it to be 0.3 above. At 30 fps, the contact frame often misses the wrist's highest point, and from behind the left/right wrist labels can swap. Things to try: the highest point of *either* wrist over a short window around contact (for example [−0.3, +0.1] s) instead of the single contact frame; or add a rule that detects the ball toss. Backhands that are classified as forehands point the same way: the wrist's x position at the single contact frame is a noisy signal. No volley was ever predicted, and only two labelled volleys were matched.

To rerun the measurement:

```bash
uv run tennis eval-classifier 2025-01-10_wingfield --config /tmp/wingfield.yaml \
  --labels labels/strokes_2025-01-10_wingfield.csv \
  --label-offset 1 --label-resolution 1 --segments labels/rallies_2025-01-10_wingfield.csv \
  --report docs/validation/M5_classifier.md
```

Expect the known risks to show: 30 fps and about 230 px of player make wrist positions noisy, serves are easy, volleys are rare and their rule is fragile, and Wingfield merges forehand and backhand volleys the way the spec does.

### M6: metrics (§6.6)

Stage 6 reads `swings.parquet`, `swing_info.parquet` and `strokes.parquet`, and writes `metrics.parquet` (one row per measured swing) and `metrics_summary.parquet` (count, mean, std, median, p10, p90 per stroke type and metric). The summary is its own file so the report and `tennis trends` read the aggregates instead of recomputing them.

Metrics are registered functions of `(SwingFrame, racket side)`; `SwingFrame` has `at()`, `between()`, `contact` and `speed()`. Adding one is a single edit. Each has a unit test on hand-built geometry with a known answer, and a test fails if a registered metric has none.

Outliers use a Mahalanobis distance per stroke type over `metrics.outlier_metrics`, with a ridge on the covariance and a standardized-Euclidean fallback below `metrics.min_swings_for_covariance` swings. Nothing samples, so reruns are identical — the determinism test compares the Parquet bytes.

**Still open:** reconcile the metric list against §6.6 (see §1), and check `contact_forward`'s sign against real Wingfield frames.

### M7: clips and HTML report (§6.8, §6.9)

Stage 8 renders a clip for every outlier, the most typical swing of each stroke type and every labelled swing (capped at `clips.max_per_label`), decoding only those swings. `clips/index.json` is the declared output, not the directory, and it records why each clip was chosen.

Stage 9 writes one self-contained `report.html` with every §6.9 section. Plotly is inlined once, the clips are linked relative, and cross-session numbers come from DuckDB over `data/sessions/*/metrics.parquet` (with a plain-Python fallback that a test holds to the same answers). `tennis report`, `tennis trends` and `tennis inspect` are all wired up.

On the PO question about which direction is an improvement: nothing is judged by default. A delta is coloured only for the metrics listed under `report.metric_direction`, and the report says so on the page.

**Still open:** the acceptance criterion needs ≥ 3 sessions with pose. Only two sessions exist and the DJI one has no usable pose, so this is blocked on recordings (§6).

### M8: voice labels (§6.7)

Stage 7 transcribes `audio.wav` with word timestamps, maps each word through `labels.vocabulary`, and attaches it to the latest confirmed own hit within `labels.max_delay_s` before it. Word times are WAV time, so `audio_start_s` puts them on the PTS timeline. Manual labels in `<paths.labels_dir>/manual_<session>.csv` override the spoken ones.

Transcription goes through a registry (`labels.backend`), with faster-whisper built in and its model cached under `<data_root>/models/whisper`. Tests register a backend returning a fixed word list, so CI never downloads anything.

The optional-input support the plan called for is in: `Stage.optional_inputs` lists files a stage uses when they exist, and the clips and report stages list `labels.parquet` that way. Turning the labels on therefore rebuilds both without `--force`.

**Still open:** `labels.language` still defaults to `en` — the PO has not said which language they call out in (§6, question 4). And the ≥ 80% match rate needs a session recorded with the words spoken, plus a `said_<session>.csv` of what was actually said (format in `labels/README.md`).

### Cross-cutting work, still owed from the spec

None of this was done; it is the top of the next backlog.

- **§10.2 integration fixture:** a real, committed clip of about 20 s with 3–5 impacts. The pipeline must run end to end in CI on the CPU with a small pose model (`yolo11n-pose.pt`, 6 MB, downloaded in CI and cached). Today, tests generate synthetic clips with ffmpeg, and the YOLO backend is stubbed (`_no_real_pose_model` in `tests/conftest.py`). Add the fixture and mark the test `real_yolo`. Stages 5–9 now have their own tests with hand-built data, so the fixture is only needed for the model-facing part.
- **§10.3 determinism check on a real session:** stage 6 has a unit-level determinism test, but the whole-pipeline check is not done. Run a session twice with `--force` and require identical metrics. Watch for GPU nondeterminism: compare with tolerance on MPS/CUDA, exactly on CPU.
- **§9 performance:** a 60-min 4K session at 120 fps with about 600 swings must run in ≤ 45 min on an RTX 3060. Untested. Ideas: `pose.hwaccel: cuda`; half precision (`half=True` in `YoloBackend.predict`); larger batches; skipping the far pass on frames where the near player is not near a contact; `track_far: false` when the mic identifies the player reliably. Stage 8 also decodes the video, but only around the selected swings. Measure on the PO's target GPU (§14: "TrueNAS host or desktop?").
- **§9 memory ≤ 8 GB:**
  - `clean.run` loads `keypoints.parquet` whole. With `pose.contacts: all` on a 4K 120 fps session, that can be millions of rows; check it, and stream per window if needed.
  - `swings_table` concatenates all swings in memory.
  - stage 6's `load_swings` holds every swing's keypoints at once, and `reporting` loads `metrics.parquet` whole — both much smaller than `keypoints.parquet`, but on the same growth curve.
- **Deliverables (§12):** README, `config.example.yaml` (a test now checks it lists every option *and* matches the defaults), ARCHITECTURE.md and this file are current. The validation notes for M5 and M8 are the ones still missing, for the reasons above.

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
| **`manual_<session>.csv` is not a stage input** | Stage 7 reads it, but it is not fingerprinted, so editing it alone does not make the stage stale. Rerun with `--from-stage 7`. |
| **The report is about 5 MB** | Plotly is inlined so the page opens offline, as the spec requires. That is the whole cost; the figures themselves are small. |
| **`ruff format` also formats Markdown code blocks** | It reformatted a Python block in this file. If a docs-only change fails `ruff format --check`, that is why. |

---

## 5. Working on it

```bash
uv sync                                   # Python 3.11, all deps (includes torch and ultralytics)
uv run pytest -q                          # ~8 s
uv run ruff check . && uv run ruff format --check . && uv run mypy

# Wingfield session (pose already done; stages 4-9 rerun in seconds, clips excepted)
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

# M5 acceptance, once the session is up to date (this is the next thing to run)
uv run tennis eval-classifier 2025-01-10_wingfield --config /tmp/wingfield.yaml \
  --labels labels/strokes_2025-01-10_wingfield.csv \
  --label-offset 1 --label-resolution 1 --segments labels/rallies_2025-01-10_wingfield.csv \
  --report docs/validation/M5_classifier.md

uv run tennis report 2025-01-10_wingfield --config /tmp/wingfield.yaml   # clips + report.html
uv run tennis trends --config /tmp/wingfield.yaml                        # data/report.html
```

**Conventions:**

- **Branches and PRs:** one branch per milestone (`m5-classify`, …), one PR each. Commit messages end with the co-author trailer used so far.
- **Where to write things down:** design notes go in the module docstrings (see `clean.py` and `pose.py`) and in ARCHITECTURE.md. Validation results go in `docs/validation/`.
- **Changes from the spec:** write each one down in ARCHITECTURE.md and in §2 of this file, with the data that justified it.
- **Tests:** never download a model or touch the network. Use synthetic clips (`tests/conftest.py`: `make_video`, `counter=True` for frame-number checks, `audio_wav=` for clicks) and hand-built keypoint fixtures.

---

## 6. Questions for the product owner

**Every remaining acceptance criterion is blocked on one of questions 1–3.** The code for M5–M8 is written and tested; what it has never seen is footage good enough to be measured against.

1. **Recording setup:** can the camera be raised behind the baseline (like the Wingfield camera), at 100–120 fps? This blocks useful pose on the DJI footage.
2. **Labels for M2:** about 100 own hits, to the frame, from one DJI clip-on-mic session (format in `labels/README.md`).
3. **More sessions:** M7 needs ≥ 3 sessions with pose, and M8 needs one session with spoken label words.
4. **Voice labels:** which language(s) and vocabulary? `labels.language` defaults to `en` and the vocabulary to English words; both are per-config, so this is a one-line change once you say. (§14)
5. **Target machine for the 45-minute performance budget:** TrueNAS host or desktop, and which GPU? (§14)
6. **Serves:** are they filmed from another camera position? If so, config profiles per session type are needed. (§14)
7. **Good or bad metric changes:** for each metric, which direction is an improvement? This sets the green/red deltas in the report (§6.9). Until you say, the report shows every delta without a verdict; fill in `report.metric_direction` to colour the ones you care about. Worth pairing with a look at `contact_forward`: one camera behind the baseline cannot separate "in front of the body" from "high", so that metric may not mean what its name suggests (ARCHITECTURE.md has the drawing).
8. **Changes from the spec:** confirm the changes in §2, especially two-player identity and far-side swings not being measured.
9. **The §6.6 metric table:** no copy of the spec is in the repository, so stage 6's metric list was reconstructed (§1). Send the table, or confirm the fourteen metrics in ARCHITECTURE.md are the right set.
