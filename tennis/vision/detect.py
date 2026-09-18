"""Object detection with an Ultralytics YOLO detection model (people, and the ball).

The pose model finds people with their keypoints, but it misses small ones. The plain
detector is much better at a far player a few dozen pixels tall; its boxes are then given
to the pose model as tight, enlarged crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

PERSON = 0
SPORTS_BALL = 32


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int


class Detector:
    def __init__(self, model_path: Path, device: str) -> None:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(model_path))
        self.device = device

    def infer(
        self,
        frames: list[npt.NDArray[np.uint8]],
        *,
        imgsz: int,
        conf: float,
        classes: list[int],
    ) -> list[list[Box]]:
        if not frames:
            return []
        results: Any = self._model.predict(
            frames,
            imgsz=imgsz,
            conf=conf,
            classes=classes,
            device=self.device,
            batch=len(frames),
            verbose=False,
            quantize=16 if self.device in ("cuda", "mps") else None,
        )
        out: list[list[Box]] = []
        for r in results:
            boxes = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                    boxes.append(Box(x1, y1, x2, y2, float(cf[i]), int(cl[i])))
            out.append(boxes)
        return out
