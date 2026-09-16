# Labels

This folder holds the hand-made labels used by the validation tools. Name files after the session ID.

## `contacts_<session-id>.csv`: your own ball impacts

This file is for `tennis tune-contacts`. Make one row per impact. The file needs a column named `t`, `time`, `time_s`, `t_video`, `timestamp` or `seconds`; if none of those is found, the first column is used. You can add other columns, and they are ignored.

```csv
t,note
1:23.40,forehand
1:26.15,backhand
91.02,
```

- **Times are video time, as the player shows it.** Write seconds (`83.4`) or `m:ss.s` / `h:mm:ss.s`.
- **Label every one of your own hits within one continuous stretch.** A missing hit counts against the detector. Hits outside the stretch are ignored. The stretch runs from the first label minus 1 s to the last label plus 1 s, or set it with `--start` and `--end`.
- **Mark the frame where the ball meets the strings.** At 24 fps the frames are 42 ms apart, so pick the frame closest to the sound.
- **Aim for about 100 hits.** That is roughly 5–10 minutes of rallying.
- **Comments and blank lines are fine.** Lines starting with `#` and blank lines are ignored.

## Files imported from Wingfield

`scripts/wingfield_labels.py` writes these files from a Wingfield export:

- `contacts_<id>.csv`: your own shots
- `hits_all_<id>.csv`: every shot, marked `self` or `other`
- `rallies_<id>.csv`: rally `start,end` times
- `strokes_<id>.csv`: stroke type for each shot (`serve`, `forehand`, `backhand` or `volley`), with the player, rally and shot number

All times are Wingfield's own clock, in whole seconds. Measure the offset to the video before using them; for 2025-01-10 it is +1 s.

## `strokes_<session-id>.csv`: stroke types (M5)

The classifier will use the Wingfield format above: columns `t,player,stroke`.

CSV files in this folder are git-ignored, because they are personal session data.
