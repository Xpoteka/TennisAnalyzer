"""Skeleton overlays and H.264 output for debug and review videos."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np
import numpy.typing as npt

from tennis.pose_backends.base import KEYPOINT_NAMES, SKELETON
from tennis.util.video import ProbeError, require_tool

# BGR colors
DOMINANT = (0, 140, 255)  # orange: racket side
OTHER_SIDE = (255, 200, 0)  # light blue
CENTER = (230, 230, 230)
BOX = (80, 220, 80)
CONTACT = (0, 0, 255)
TEXT_BG = (20, 20, 20)


def _side(name: str) -> str | None:
    if name.startswith("l_"):
        return "l"
    if name.startswith("r_"):
        return "r"
    return None


def draw_pose(
    image: npt.NDArray[np.uint8],
    keypoints: npt.NDArray[np.float32],
    bbox: tuple[float, float, float, float] | None,
    handedness: str,
    kp_conf_min: float,
) -> None:
    """Draw a skeleton in place. The dominant (racket) side gets its own color."""
    dominant = "r" if handedness == "right" else "l"
    scale = max(1, round(image.shape[0] / 540))

    def color(i: int) -> tuple[int, int, int]:
        side = _side(KEYPOINT_NAMES[i])
        if side is None:
            return CENTER
        return DOMINANT if side == dominant else OTHER_SIDE

    def visible(i: int) -> bool:
        x, y, c = keypoints[i]
        return bool(np.isfinite(x) and np.isfinite(y) and c >= kp_conf_min)

    if bbox is not None and all(np.isfinite(bbox)):
        x1, y1, x2, y2 = (round(v) for v in bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX, scale)
    for a, b in SKELETON:
        if visible(a) and visible(b):
            side_a, side_b = _side(KEYPOINT_NAMES[a]), _side(KEYPOINT_NAMES[b])
            limb = color(a) if side_a == side_b else CENTER
            pa = (round(float(keypoints[a, 0])), round(float(keypoints[a, 1])))
            pb = (round(float(keypoints[b, 0])), round(float(keypoints[b, 1])))
            cv2.line(image, pa, pb, limb, 2 * scale, cv2.LINE_AA)
    for i in range(len(KEYPOINT_NAMES)):
        if visible(i):
            center = (round(float(keypoints[i, 0])), round(float(keypoints[i, 1])))
            cv2.circle(image, center, 3 * scale, color(i), -1, cv2.LINE_AA)


def draw_label(image: npt.NDArray[np.uint8], lines: list[str],
               color: tuple[int, int, int] = (255, 255, 255)) -> None:  # fmt: skip
    """White text on a dark box in the top-left corner."""
    scale = image.shape[0] / 720
    y = round(28 * scale)
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, 2)
        top_left = (round(8 * scale), y - h - round(6 * scale))
        bottom_right = (round(16 * scale) + w, y + round(6 * scale))
        cv2.rectangle(image, top_left, bottom_right, TEXT_BG, -1)
        cv2.putText(image, line, (round(12 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * scale, color, 2, cv2.LINE_AA)  # fmt: skip
        y += h + round(16 * scale)


def draw_border(image: npt.NDArray[np.uint8], color: tuple[int, int, int], width: int) -> None:
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w - 1, h - 1), color, width)


class VideoWriter:
    """Pipe BGR frames to ffmpeg; output is H.264 MP4 scaled to ``height`` pixels."""

    def __init__(self, path: Path, width: int, height: int, fps: float, out_height: int = 720):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.size = (width, height)
        out_height = min(out_height, height) // 2 * 2
        # fmt: off
        cmd = [
            require_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.6f}", "-i", "-",
            "-vf", f"scale=-2:{out_height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ]
        # fmt: on
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, image: npt.NDArray[np.uint8]) -> None:
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError(f"frame is {image.shape[1]}x{image.shape[0]}, expected {self.size}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode() if self._proc.stderr else ""
        if self._proc.wait() != 0:
            raise ProbeError(f"ffmpeg could not write {self.path}: {err.strip()}")

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._proc.kill()
            self._proc.wait()
