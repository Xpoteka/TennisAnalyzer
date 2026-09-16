# /// script
# requires-python = ">=3.11"
# dependencies = ["openpyxl>=3.1"]
# ///
"""Convert a Wingfield session export (.xlsx) into label files for the validation tools.

Usage:
    uv run scripts/wingfield_labels.py EXPORT.xlsx SESSION_ID [--player NAME] [--out labels]

Writes, with times copied as-is from the export (whole seconds, Wingfield's clock):

    contacts_<id>.csv   your own shots        -> tune-contacts --target self
    hits_all_<id>.csv   every shot            -> tune-contacts --target any
    rallies_<id>.csv    rally start,end       -> tune-contacts --segments
    strokes_<id>.csv    shot type per shot    -> eval-classifier (M5)

Player names are not written; shots are marked ``self`` or ``other``.

Wingfield times are truncated to the second and its clock may not match the video. For
2025-01-10 the hit is in [t + 1, t + 2], so tune with
``--label-offset 1 --label-resolution 1``. Check the offset for each new export.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import openpyxl

STROKE_TYPES = {
    ("SERVE", "DEUCE"): "serve",
    ("SERVE", "AD"): "serve",
    ("GROUNDSTROKE", "FOREHAND"): "forehand",
    ("GROUNDSTROKE", "BACKHAND"): "backhand",
    ("VOLLEY", "FOREHAND"): "volley",
    ("VOLLEY", "BACKHAND"): "volley",
}


def seconds(value: object) -> int:
    hours, minutes, secs = (int(part) for part in str(value).split(":"))
    return hours * 3600 + minutes * 60 + secs


def rows(workbook: openpyxl.Workbook, sheet: str) -> list[dict[str, object]]:
    values = list(workbook[sheet].iter_rows(values_only=True))
    header = [str(h).strip() for h in values[0]]
    return [dict(zip(header, row, strict=False)) for row in values[1:] if any(row)]


def write(path: Path, header: list[str], data: list[list[object]], note: str) -> None:
    with path.open("w", newline="") as fh:
        fh.write(f"# {note}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(data)
    print(f"wrote {path} ({len(data)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("export", type=Path)
    parser.add_argument("session_id")
    parser.add_argument("--player", help="your name as in the export (default: Player 1)")
    parser.add_argument("--out", type=Path, default=Path("labels"))
    args = parser.parse_args()

    wb = openpyxl.load_workbook(args.export, read_only=True, data_only=True)
    player = args.player or str(rows(wb, "info")[0]["Player 1"])
    shots = rows(wb, "shot_stats")
    rallies = rows(wb, "rally_stats")
    names = {str(s["Player Name"]).strip() for s in shots}
    if player.strip() not in names:
        print(f"player {player!r} not in export; found {sorted(names)}", file=sys.stderr)
        return 1

    note = f"from {args.export.name}; Wingfield clock, whole seconds"
    args.out.mkdir(parents=True, exist_ok=True)
    sid = args.session_id
    records = []
    for s in shots:
        who = "self" if str(s["Player Name"]).strip() == player.strip() else "other"
        stroke = STROKE_TYPES.get((str(s["Shot Type"]), str(s["Stroke"])), "unknown")
        records.append((seconds(s["Video Time"]), who, stroke, s["Rally #"], s["Shot # of Rally"]))
    records.sort()

    write(
        args.out / f"contacts_{sid}.csv",
        ["t", "stroke"],
        [[t, stroke] for t, who, stroke, *_ in records if who == "self"],
        note,
    )
    write(
        args.out / f"hits_all_{sid}.csv",
        ["t", "player"],
        [[t, who] for t, who, *_ in records],
        note,
    )
    write(
        args.out / f"strokes_{sid}.csv",
        ["t", "player", "stroke", "rally", "shot"],
        [list(r) for r in records],
        note,
    )
    # Wingfield's end time is truncated too, so extend each rally by a second.
    write(
        args.out / f"rallies_{sid}.csv",
        ["start", "end"],
        [[seconds(r["Start Time"]), seconds(r["End Time"]) + 1] for r in rallies],
        note,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
