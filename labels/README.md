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

## `strokes_<session-id>.csv`: stroke types

For `tennis eval-classifier`. Two shapes are accepted, and the header decides which:

- the Wingfield export above, `t,player,stroke,...`, of which only the rows marked `self` are used;
- a plain `t,stroke` file, where every row is yours.

Stroke names must be `serve`, `forehand`, `backhand` or `volley`; rows with anything else are skipped. Times follow the same clock rules as the contact labels, so pass `--label-offset` and `--label-resolution` for a coarse external clock.

```csv
t,stroke
1:23.4,forehand
1:26.2,backhand
```

## `manual_<session-id>.csv`: hand-written shot labels

Read by **stage 7**, not by a validation tool. Rows are `contact_id,label`, and they override whatever the transcriber heard for that contact:

```csv
contact_id,label
412,framed
418,good
```

The contact id must be a confirmed own hit of that session — look it up in `swing_info.parquet` or in the report. Because this file is not a declared stage input, adding or editing it needs `tennis process ... --from-stage 7`.

The folder is `paths.labels_dir` in the config (`./labels` by default), resolved from the config file's directory.

## `said_<session-id>.csv`: what you actually said

For `tennis eval-labels`. One row per label word you spoke, with the time you said it (video time) and which label it was:

```csv
t,label
3:12.4,good
3:20.0,late
```

CSV files in this folder are git-ignored, because they are personal session data.
