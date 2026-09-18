"""Video stage ``court``: find the court and the camera (see :mod:`tennis.vision.court`).

Frames are sampled across the whole video and their per-pixel median becomes the background:
players and balls move, so they vanish from it, and the painted lines stay. Detection runs on
that background. The camera is assumed to stay put for the whole video.

Writes ``court.json`` (the calibration, or ``{"found": false, ...}`` with the reason) and
``court.jpg``, the background with the fitted lines drawn in, to check the result by eye.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import numpy.typing as npt

from tennis.util.io import atomic_path, write_json
from tennis.util.video import grab_frame
from tennis.vision import court

if TYPE_CHECKING:
    from tennis.pipeline import VideoContext

OUTPUTS = ("court.json", "court.jpg")


def background(ctx: VideoContext, n: int) -> npt.NDArray[np.uint8] | None:
    duration = float(ctx.metadata().get("duration_s") or 0.0)
    times = np.linspace(0.05, 0.95, n) * duration if duration > 2 else np.array([0.0])
    frames = []
    for i, t in enumerate(times):
        frame = grab_frame(ctx.source, float(t))
        if frame is not None:
            if frame.shape[1] > court.WORK_WIDTH * 1.5:  # the median needs no more detail
                f = court.WORK_WIDTH * 1.5 / frame.shape[1]
                frame = np.asarray(
                    cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), np.uint8
                )
            frames.append(frame)
        ctx.progress(0.5 * (i + 1) / len(times))
    if not frames:
        return None
    return np.asarray(np.median(np.stack(frames), axis=0), np.uint8)


def draw(image: npt.NDArray[np.uint8], cal: court.Calibration) -> npt.NDArray[np.uint8]:
    out = image.copy()
    for x1, y1, x2, y2 in court.COURT_LINES.values():
        t = np.linspace(0, 1, 80)
        p = cal.court_to_image(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], 1))
        if np.isfinite(p).all():
            cv2.polylines(out, [np.round(p).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
    if cal.has_camera:
        xs = np.linspace(-6.4, 6.4, 40)
        hs = np.interp(np.abs(xs), [0, 6.4], [court.NET_HEIGHT_CENTER, court.NET_HEIGHT_POST])
        p = cal.world_to_image(np.stack([xs, np.zeros_like(xs), hs], 1))
        if np.isfinite(p).all():
            cv2.polylines(out, [np.round(p).astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def run(ctx: VideoContext) -> None:
    cfg = ctx.config.court
    meta = ctx.metadata()
    bg = background(ctx, cfg.frames)
    result: dict[str, Any]
    if bg is None:
        result = {"found": False, "reason": "no frames could be read"}
        image = np.zeros((360, 640, 3), np.uint8)
    else:
        h, w = bg.shape[:2]
        on_bg: court.Calibration | None
        if cfg.manual_corners is not None:
            corners = np.array(cfg.manual_corners, np.float64) * [w, h]
            on_bg = court.from_corners(corners, w, h, court.line_mask(bg))
        else:
            on_bg = court.detect(bg)
        ctx.progress(0.95)
        image = bg
        if on_bg is None:
            result = {"found": False, "reason": "no court lines were found"}
        else:
            image = draw(bg, on_bg)
            # Keep the calibration in the video's own pixels, whatever size the background had.
            factor = float(meta["width"]) / w
            cal = court.with_camera(on_bg.scaled(factor)) if abs(factor - 1) > 1e-6 else on_bg
            found = cal.quality >= cfg.min_quality and court.usable_view(cal)
            result = {
                "found": found,
                "reason": None
                if found
                else f"court lines did not fit well enough (quality {cal.quality:.2f})",
                **cal.to_json(),
            }
    ctx.log("court", found=result["found"], quality=result.get("quality"))
    write_json(ctx.path("court.json"), result)
    with atomic_path(ctx.path("court.jpg")) as tmp:
        cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
