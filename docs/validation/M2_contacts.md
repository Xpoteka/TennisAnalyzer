# M2 validation: contact detection

**Status: not yet accepted.** The acceptance test in spec section 10.3 needs about 100 hand-labeled own impacts from a session recorded with the clip-on mic, matched within ±40 ms. That dataset doesn't exist yet.

## 2025-01-10 Wingfield match (coarse external labels)

| | |
|---|---|
| Source | Wingfield smart-court export: 288 shots (144 own), 97 rallies |
| Video | 1280×720, 30 fps (variable), 30 min |
| Audio | Court camera mic, 16 kHz mono. **Not** the player's clip-on mic. |
| Label precision | Whole seconds. Wingfield's clock is about 1 s behind the video, so a label `t` means the hit is in [t+1, t+2] s. |
| Evaluated | Rally time only (512 s); ball bounces before the serve and between points are excluded |

The offset was measured by comparing all shot labels with all detected onsets: loud onsets cluster sharply between +1.0 and +2.0 s. It was confirmed on video frames for the first serve, where contact is at about 22.5 s against a Wingfield time of 0:21.

### Detecting any hit (both players): [full table](M2_contacts_wingfield_all_hits.md)

| Setting | Precision | Recall |
|---|---|---|
| Current defaults (800 Hz, k=6) | 0.40 | 1.00 |
| Best F1 (400 Hz, k=16) | 0.62 | 0.92 |

The detector finds essentially every hit. The extra detections inside rallies are mostly bounces and footsteps: real impact sounds that audio alone can't reliably reject. The 1 s label window makes these scores optimistic, because a bounce can land in the window of a hit that was missed.

### Selecting own hits by loudness: [full table](M2_contacts_wingfield_own_hits.md)

The best F1 is 0.62 (precision 0.55, recall 0.71). With a fixed court mic, both players' hits have the same level (median −33.7 vs −34.1 dBFS), so the first-pass loudness rule can't separate them. That rule assumes the spec's player-worn mic.

## 2026-09-16 DJI session (no labels yet)

- 49 min, 23.976 fps, clip-on mic recorded into the camera file (stereo, but both channels are equivalent; a mono mix is used).
- 3,040 onsets at the defaults. Their loudness is bimodal, with a separate loud group (about 450 onsets at ≥ −31 dBFS) that plausibly contains the own hits.
- The default `own_hit_db_threshold` of 6 dB selects 854 onsets, likely too many. The loud group suggests about 11 dB, but this is unverified.

## Next steps

1. Label about 100 own hits in a DJI session: see [labels/README.md](../../labels/README.md).
2. Run `tennis tune-contacts` and replace this section with the result.
3. Expect the remaining false positives (bounces near the player) to need the wrist-speed confirmation from M4.
