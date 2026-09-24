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
| hits | `hits.parquet` | Hit times, side and hitter track (`vision/hits.py`): candidates from sound, ball turns and wrist-speed peaks, scored per side, then the best sequence under the rally rules (alternate sides, ≥0.55 s apart) by dynamic programming. People beside the court (median foot position more than 1.5 m outside the doubles sidelines: the bench, the next court through the fence) are never offered as the hitter. |

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
| identities | Joins tracks into players (`identities.json`, `SessionPlayer`). Bystanders (beside the court) are dropped first. How many people are usually on court is counted per analysed frame (the 90th percentile), so a track ending as its replacement starts is not a third person. Singles (≤2): tracks seen together are different people; the relation is chained along a maximum spanning tree of "seen together" seconds, clothing colour only matches parts that never meet, and tracks too short to take part go to the player who was on their end of the court at the time. Doubles: tracks are joined by clothing colour (never two seen together), and the groups with the hits are the players. Across cameras, tracks standing on the same spot at the same time are the same person. Everything is built on one pass over the frames, so a long video with a thousand tracks takes seconds, not an hour. Thumbnails per player. |
| players | Links the session's players to profiles (`analysis/reid.py`): racket hand from serves, height through the camera, limb proportions, clothing colour (weak); Hungarian assignment, new profile beyond a distance. Profiles left empty are removed unless renamed. |
| shots | One `Shot` per hit: player, stroke (`vision/strokes.py`: overhead → serve/smash, forehand/backhand from the hands' direction across the body, volley near the net, spin from the wrist path), ball flight (`vision/physics.py`: gravity, drag, one bounce, fitted to the ball pixels) → speed, net clearance, apex, bounce, in/out, depth, direction. Rallies: a serve starts one, a 3 s pause ends one, unless the next hit within 5 s comes from the same side of the net (one hit in between was missed). A serve is an overhead swing; when the swing cannot tell (the far player is a few dozen pixels tall), the first hit of a rally from behind the baseline near the centre mark after a 5 s pause counts as a serve (`quality.serve_by = position`). With several cameras, a shot seen twice becomes one, keeping the best of each view. Racket hand from serves (the hand above the head at contact). Technique per shot (`analysis/technique.py`): knee bend, elbow at contact, shoulder turn, arm speed, split step as the opponent hits, recovery to the centre. A swing that looks like a serve mid-rally is a smash only when hit from inside the court; from behind the baseline it is a serve and restarts the rally, because the hit detector finds a hit or two in the waiting movements and ball bounces before most serves. The last shot of a rally has no next hit to bound its flight, so it is fitted over shortening windows and the shortest one that shows the bounce wins. |
| kind | Training or match (`analysis/kind.py`) from serve runs, serve spacing and rally lengths, unless set by hand. Only serves seen in the swing count, so feeds from the baseline in training do not look like serves. |
| scoring | Matches only (`analysis/scoring.py`): points are serve-started rallies. A rally of at most two shots followed by the same player serving again from the same service box (same side of the centre mark) was a fault and is folded into the next rally; after a point the server changes box. Point winners from how the rally ended (a serve that was not returned is even: a missed second serve looks like a service winner). The whole match is then decoded at once (a beam search over the rules): the hidden state is the score plus who serves and who stands at which end, and every point's serve is checked against it: the server alternates by game and by pairs of points in a tiebreak, the ends change after odd games and every six tiebreak points, and the service box follows the parity of the points. A detected point may be spurious (a fault's second serve) and a point may have been missed, at a cost. That pins the number of games and finds tiebreaks even when the point winners are poor; who won a game still rests on the point winners. |
| summary | Per-player stats and the session headline. In a match, only shots in points count. |

### Measured on the Wingfield match (30 min, hand-labelled shot log)

| What | Result | Note |
|---|---|---|
| Court | quality 0.94 | camera 4.1 m behind the baseline, 4.6 m high |
| Hits found | 89.9% of labelled hits | labels are whole seconds; many "extra" hits are balls hit back between points, and returns missing from the log |
| Who hit | 89.2% | |
| Stroke | 74.1% | serve 79%, forehand 80%, backhand 70%, volleys weak |
| Session type | match (p = 0.83) | |
| Server of a point | 100% | |
| Winner of a point | 64% | **weak** (57% before the whole-match decoder and the last-shot flight windows; 43% before the rally and fault rules). Measured against the labelled shots, the detector finds 96% of hits and 98% of the last hits of points, but three hits for every real one: spurious hits in the players' movements between points chain onto the rally, so the last hitter, which is what decides the point, is the loser on only about half the points. Trimming silent trailing hits or flipping on the opponent's swing did not help; the ball's bounce is what would, and it is measured for one last shot in four |
| Games | 2.1 games off on average at 12 moments (per-game decoder; not re-measured with the whole-match decoder) | `tennis eval-games` against the game count read off the point log |

### Measured on an 89 min club match (Kitris court camera, 720p, scoreboard burned in)

The scoreboard gives the games at every changeover (`labels/games_2024-11-10_match.csv`, read off the video by hand). Two players are found (before the fix, the identity stage counted a track ending as its replacement started as a third person and made four), and the identity stage takes 40 s instead of 51 min. 229 points and 52 faults are found for 21 true games. Decoded game by game from server runs, the score agreed with the scoreboard at 5 of 12 moments and was 2.5 games off on average: one deuce game at 6-5 went to the wrong player, so the set closed 7-5 and the real tiebreak was read as games of the next set. The whole-match decoder, which checks each serve's player, end and service box against the rules, finds the tiebreak and the set lost 6-7 and is 0.92 games off on average (4 of 12 exact); the second set comes out 3-6 where it was won 6-3, because two game winners rest on point winners that are wrong. The point winners remain the limit: the ball is a few pixels at 720p and mostly lost, so almost every point ends "unknown", and a game between two players goes to the wrong one now and then.

**Times in the database are session seconds.** A video's PTS `t` is at session time `t + offset_s`.

## Paths

A video inside the data folder is stored relative to it (`uploads/x.mp4`), so the data folder can move from a Mac to a server as a whole. A file elsewhere is stored with its absolute path.
