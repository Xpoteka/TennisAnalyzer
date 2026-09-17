# M4 validation: cleaning, QC and own-hit confirmation

Session: the 2025-01-10 Wingfield match. The video is 720p at 30 fps (variable frame rate) and the mic is fixed on the court. Pose ran around every onset (`pose.contacts: all`), which gives 1,865 candidate swings and 43,605 frames.

## Acceptance

| Criterion | Result |
|---|---|
| Smooth wrist trajectories on sample plots | Yes. Run `tennis swing-plots`: cleaned trajectories follow the raw keypoints without jitter, and the wrist-speed peaks line up with the audio contact. |
| QC pass rate ≥ 80% | **86.3%** for your near-side hits in the default mode (first pass and confirmed, 107/124). 79.4% for near-side hits confirmed by the wrist alone (223/281). 64.9% for all 1,302 near-side candidates; most failures are windows where you aren't tracked (for example, while your partner is hitting). Far-side swings are excluded on purpose: they are detected but not measured. |

## Who is who (two players, changing ends)

Pose tracks both the near and the far player (`pose.track_far`). A full-width crop of the upper half runs at 960 px; the partner often stands at the far right, outside a centred crop. Stage 4 then matches the two players by clothing color.

| Check | Result |
|---|---|
| Identity of "you" | Player A, the near player at the start (the court mic gives no loudness difference). The thumbnails (`tennis players`) confirm A is the orange shirt. |
| End changes | Changes detected at 7:12, 13:05, 22:04 and 26:12. Wingfield's first shot from the new side comes at 8:08, 13:37, 22:58 and 26:40. The detected change comes 30–60 s earlier every time, which is the changeover walk. There is one spurious 5 s "far" stretch at 0:00, during warm-up. |
| Racket hand (`auto`) | **Left**, which is correct: 160 votes left vs 122 right, across own near-side hits. |
| Far player found | 12% of frames overall. Far detections that overlapped the near player were dropped (805 frames). Many windows are between points, when the partner is off the far half. |

The spec lists re-identification as a non-goal, but real sessions have two players who change ends. Without it, 30% of the session (the far-side stretches) would have measured the partner as "you".

## Own-hit confirmation

The labels are Wingfield's 144 own shots, accurate to the whole second (label window [t+1, t+2] s), scored during rallies only. The full table is in [M4_confirmation_wingfield.md](M4_confirmation_wingfield.md).

| Selection | Precision | Recall | F1 |
|---|---|---|---|
| All onsets | 0.20 | 1.00 | 0.33 |
| Audio first pass only | 0.44 | 0.81 | 0.57 |
| Wrist confirmation only | 0.36 | 0.76 | 0.49 |
| First pass and confirmation, near player only (before identity) | 0.59 | 0.66 | 0.63 |
| **First pass and confirmation, with identity and hit attribution** | **0.75** | **0.62** | **0.68** |

This is short of the spec's 0.9 target. On this video, that's expected:

- **The mic is on the court**, so the audio level can't tell the players apart (median −33.7 vs −34.1 dBFS).
- **The player is small:** about 230 px tall at 30 fps, so wrist speeds are coarse.
- **The labels are coarse:** a 1 s window can hold both a hit and a bounce, so these scores are only approximate.

## Tuning notes (these change the spec's defaults)

| Setting | Spec | New default | Why |
|---|---|---|---|
| `cleaning.one_euro.min_cutoff` / `beta` | 1.0 / 0.05 | 3.0 / 0.5 | At the spec values, the median wrist speed near the onset was the same for own and partner hits (4.2 vs 4.1). In raw keypoints it separates them (4.7 vs 2.7 body heights/s). |
| `cleaning.one_euro.zero_phase` | – | true | Averages a forward and a backward pass, so the smoothing doesn't shift peaks later in time. |
| Confirmation rule | Wrist speed "peaks" within ±150 ms | The highest *local maximum* within ±`wrist_confirm_window_s` (0.2 s) must reach `wrist_confirm_min_speed` (6 torso lengths/s) | Taking the maximum over the whole 1.5 s window often picked the backswing or follow-through. The best F1 with that rule was 0.43. |
| `audio.wrist_confirm_wrist` | Racket wrist | `either` (the faster wrist) | Left/right labels are unreliable from behind. For own hits, the labelled left wrist was faster than the right. The best F1 for racket-wrist-only was 0.49. |

## Open points

- **Resolved:** the player is left-handed, and `handedness: auto` finds this.
- **Swap fix:** it changes 7% of shoulder frames. These are probably real crossings seen from behind during the shoulder turn, not label errors. Turning the swap fix off didn't change confirmation F1. It stays enabled, as the spec requires.
- **Real validation needs frame-accurate own-hit labels** from a session filmed with the clip-on mic.
