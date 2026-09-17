"""Stage 9: the per-session HTML report (spec section 6.9).

The stage is a thin wrapper: everything it does lives in ``tennis.reporting``, which the
``tennis report`` and ``tennis trends`` commands share. The report is one self-contained
file that opens offline, with the clips linked beside it.

``labels.parquet`` and ``clips/index.json`` are optional inputs: without them the label
analysis and the clip gallery are left out, and producing either rebuilds the report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tennis.reporting import write_session_report

if TYPE_CHECKING:
    from tennis.stages import StageContext

INPUTS = (
    "metrics.parquet",
    "metrics_summary.parquet",
    "swing_info.parquet",
    "strokes.parquet",
    "metadata.json",
)
OPTIONAL_INPUTS = ("labels.parquet", "clips/index.json", "players.json", "keypoints.parquet")
OUTPUTS = ("report.html",)
CONFIG_KEYS = ("report", "player", "labels.enabled")


def run(ctx: StageContext) -> None:
    path = write_session_report(ctx.session, ctx.config)
    ctx.log("report written", path=str(path), size_kb=round(path.stat().st_size / 1024, 1))
