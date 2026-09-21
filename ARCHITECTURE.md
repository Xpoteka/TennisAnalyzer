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
| audio | `onsets.parquet`, `envelope.npy` | Impact onsets (PTS) from `util/audio.py`. The 200 Hz onset envelope is for syncing several videos. |
| proxy | `proxy.mp4`, `proxy.json` | Browser-playable H.264. A source that already plays is symlinked instead. Proxy time = PTS − `start_pts`. |
| court | `court.json`, `court.jpg` | Court and camera from the median of 24 frames (`vision/court.py`). `found: false` when it does not fit or the view is too low. |
| people | `people.parquet` | YOLO pose at ~10 fps over the whole video, feet on court in metres, clothing colour, `track_id`. Far players come from a detector on an enlarged crop of the far half, with pose from a tight zoomed crop. |
| ball | `ball.parquet` | Every frame: three-frame motion blobs scored for ball colour and shape, linked into tracklets (`vision/ball.py`). |
| motion | `motion.parquet` | Every frame: pose of each tracked player from an enlarged crop around the tracker's box. The swings at full frame rate. |
| hits | `hits.parquet` | Hit times, side and hitter track (`vision/hits.py`): candidates from sound, ball turns and wrist-speed peaks, scored per side, then the best sequence under the rally rules (alternate sides, ≥0.55 s apart) by dynamic programming. |

### Court and camera (`tennis/vision/court.py`)

Court metres: origin at the centre of the net, `x` across (right as seen from the near baseline), `y` along (towards the far baseline, the one higher in the image), `z` up.

1. **Background.** The white-line mask comes from a top-hat filter plus low saturation, taken on the median background.
2. **Search.** For each of five lens distortions, the image is straightened and Hough lines are found. Every pair of "across" lines and pair of "along" lines is then matched to every ordered pair of court lines, which gives a homography. Each one is scored by the **distinct** line pixels the projected court explains. Counting distinct pixels stops a court squeezed onto one line from winning.
3. **Refinement.** Least squares on the distance transform, with two radial distortion terms. A weak prior on the corners stops it from collapsing.
4. **Camera.** Focal length, rotation and position follow from the homography: principal point at the centre, square pixels. They are then refined against the court lines and the **net tape**, which is at a known height. The court alone cannot separate a longer lens from a camera further away; the net can.
5. **Rejection.** A fit is not a court when:
   - quality (the share of visible line samples on white) is below `court.min_quality`;
   - the outline covers under 5% of the frame or under 15% of its height;
   - the lens model folds over (distortion must grow with radius);
   - the camera is not 1–40 m above the court and looking down at 6° or more.

   A camera at court level (the DJI session) is rejected this way.

**Session stages** combine every finished video of a session and write to the database. They are cheap and always run.

| Stage | Does |
|---|---|
| timeline | Places each video on the session clock (`Video.offset_s`). Overlapping videos are lined up by sound (`analysis/sync.py`: GCC-PHAT on the onset envelopes, accepted only with a clear peak); others by creation time. |
| identities | Joins tracks into players; across cameras, tracks standing on the same spot at the same time are the same person (`identities.json`, `SessionPlayer`). Singles: tracks seen together are different people; the relation is chained along a maximum spanning tree of "seen together", and clothing colour only matches parts that never meet. Thumbnails per player. |
| players | Links the session's players to profiles (`analysis/reid.py`): racket hand from serves, height through the camera, limb proportions, clothing colour (weak); Hungarian assignment, new profile beyond a distance. Profiles left empty are removed unless renamed. |
| shots | One `Shot` per hit: player, stroke (`vision/strokes.py`: overhead → serve/smash, forehand/backhand from the hands' direction across the body, volley near the net, spin from the wrist path), ball flight (`vision/physics.py`: gravity, drag, one bounce, fitted to the ball pixels) → speed, net clearance, apex, bounce, in/out, depth, direction. Rallies: a serve starts one, a 3 s pause ends one. With several cameras, a shot seen twice becomes one, keeping the best of each view. Racket hand from serves (the hand above the head at contact). Technique per shot (`analysis/technique.py`): knee bend, elbow at contact, shoulder turn, arm speed, split step as the opponent hits, recovery to the centre. |
| kind | Training or match (`analysis/kind.py`) from serve runs, serve spacing and rally lengths, unless set by hand. |
| scoring | Matches only (`analysis/scoring.py`): points are serve-started rallies, first-serve faults folded in; point winners from how the rally ended; games from server runs and sets decoded under the rules. |
| summary | Per-player stats and the session headline. In a match, only shots in points count. |

### Measured on the Wingfield match (30 min, hand-labelled shot log)

| What | Result | Note |
|---|---|---|
| Court | quality 0.94 | camera 4.1 m behind the baseline, 4.6 m high |
| Hits found | 87.5% of labelled hits | labels are whole seconds; many "extra" hits are balls hit back between points, and returns missing from the log |
| Who hit | 90.9% | |
| Stroke | 76.6% | serve 83%, forehand 75%, backhand 70%, volleys weak |
| Session type | match (p = 0.83) | |
| Server of a point | 98% | |
| Winner of a point | 55% | **weak**: the last shot of a point is often missed or followed by a ball hit back |

### Review player

The browser draws the overlays itself, on a canvas over the proxy video; nothing is rendered into the video. `GET /api/videos/<id>/overlay` gives what holds for the whole video (its size, the court lines and every bounce projected to pixels, which track is which player). `GET /api/videos/<id>/tracks?start=&end=` gives ball, poses and feet from the stage files in slices (`tennis/api/tracks.py`; PTS seconds, source pixels, gzipped, about 20 KB per ten seconds). What counts as an error is decided in one place, `web/src/review.ts`: a shot whose flight ended in the net or outside the lines; for serves, a fault or double fault.

**Times in the database are session seconds.** A video's PTS `t` is at session time `t + offset_s`.

## Paths

A video inside the data folder is stored relative to it (`uploads/x.mp4`), so the data folder can move from a Mac to a server as a whole. A file elsewhere is stored with its absolute path.
