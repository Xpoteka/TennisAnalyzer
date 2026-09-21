"""Ultralytics YOLO pose backend (AGPL-3.0; fine for personal use, see the README)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tennis.pose_backends.base import GPU_DEVICES, NUM_KEYPOINTS, Image, PersonPose


class YoloBackend:
    name = "yolo"

    def __init__(self, model_path: Path, device: str, imgsz: int, min_person_conf: float) -> None:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        model_path.parent.mkdir(parents=True, exist_ok=True)
        # A bare model name that does not exist yet is downloaded to model_path on first use.
        self._model = YOLO(str(model_path))
        if self._model.task != "pose":
            raise ValueError(f"{model_path.name} is a '{self._model.task}' model, not a pose model")
        self.device = device
        self.imgsz = imgsz
        self.min_person_conf = min_person_conf

    def infer(self, frames: list[Image]) -> list[list[PersonPose]]:
        if not frames:
            return []
        results: Any = self._model.predict(
            frames,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.min_person_conf,
            classes=[0],
            batch=len(frames),
            verbose=False,
            # Half precision on a GPU: twice as fast, and the keypoints agree to 0.05 px.
            quantize=16 if self.device in GPU_DEVICES else None,
        )
        out: list[list[PersonPose]] = []
        result: Any
        for result in results:
            people: list[PersonPose] = []
            boxes, keypoints = result.boxes, result.keypoints
            if boxes is not None and keypoints is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                data = keypoints.data.cpu().numpy().astype(np.float32)
                if data.shape[-1] == 2:  # model without keypoint confidences
                    data = np.concatenate([data, np.ones((*data.shape[:2], 1), np.float32)], -1)
                for i in range(len(xyxy)):
                    kp = data[i]
                    if kp.shape != (NUM_KEYPOINTS, 3):
                        continue
                    people.append(
                        PersonPose(
                            bbox=(
                                float(xyxy[i, 0]),
                                float(xyxy[i, 1]),
                                float(xyxy[i, 2]),
                                float(xyxy[i, 3]),
                            ),
                            bbox_conf=float(conf[i]),
                            keypoints=kp,
                        )
                    )
            out.append(people)
        return out
