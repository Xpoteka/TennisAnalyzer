"""YAML configuration, validated with pydantic (spec section 8).

Every option has a default, so an empty or missing config file is valid. Unknown keys are
rejected so that typos fail loudly instead of being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tennis.errors import ConfigError

DEFAULT_CONFIG_NAME = "config.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerConfig(_Section):
    handedness: Literal["right", "left"] = "right"
    camera_side: Literal["behind_baseline", "side_on"] = "behind_baseline"
    forward_sign: Literal[1, -1] = 1


class PathsConfig(_Section):
    data_root: Path = Path("./data")


class AudioConfig(_Section):
    highpass_hz: float = Field(800.0, gt=0)
    highpass_order: int = Field(4, ge=1, le=10)
    onset_k: float = Field(6.0, gt=0)
    min_separation_s: float = Field(0.25, gt=0)
    threshold_window_s: float = Field(5.0, gt=0)
    amplitude_window_s: float = Field(0.02, gt=0)
    own_hit_db_threshold: float = 6.0
    wrist_confirm_window_s: float = Field(0.15, gt=0)


class PoseConfig(_Section):
    backend: Literal["yolo", "mediapipe"] = "yolo"
    model: str = "yolo11m-pose.pt"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    batch_size: int = Field(16, ge=1)
    kp_conf_min: float = Field(0.3, ge=0, le=1)
    near_court_min_y: float = Field(0.4, ge=0, le=1)
    track_iou_min: float = Field(0.3, ge=0, le=1)
    crop_pad: float = Field(0.2, ge=0)


class WindowsConfig(_Section):
    pre_s: float = Field(1.0, gt=0)
    post_s: float = Field(0.5, gt=0)


class OneEuroConfig(_Section):
    min_cutoff: float = Field(1.0, gt=0)
    beta: float = Field(0.05, ge=0)
    d_cutoff: float = Field(1.0, gt=0)


class SavgolConfig(_Section):
    window: int = Field(9, ge=3)
    order: int = Field(3, ge=1)

    @model_validator(mode="after")
    def _check_window(self) -> SavgolConfig:
        if self.window % 2 == 0:
            raise ValueError("savgol.window must be odd")
        if self.order >= self.window:
            raise ValueError("savgol.order must be smaller than savgol.window")
        return self


class QcConfig(_Section):
    min_valid_frame_ratio: float = Field(0.8, ge=0, le=1)
    max_track_resets: int = Field(2, ge=0)


class CleaningConfig(_Section):
    smoother: Literal["one_euro", "savgol"] = "one_euro"
    one_euro: OneEuroConfig = OneEuroConfig()
    savgol: SavgolConfig = SavgolConfig()
    max_gap_frames: int = Field(5, ge=0)
    swap_improvement_ratio: float = Field(0.3, ge=0, lt=1)
    qc: QcConfig = QcConfig()


class ClassifyConfig(_Section):
    serve_wrist_above_nose: float = 0.3
    volley_travel_max: float = Field(0.8, gt=0)
    volley_bbox_bottom_max_y: float = Field(0.55, ge=0, le=1)
    two_handed_max_dist: float = Field(0.25, gt=0)


class MetricsConfig(_Section):
    outlier_metrics: list[str] = Field(
        default_factory=lambda: [
            "contact_height",
            "contact_forward",
            "elbow_angle_contact",
            "knee_flex_min",
            "peak_speed_offset",
        ]
    )
    outlier_percentile: float = Field(95.0, gt=0, lt=100)


class ClipsConfig(_Section):
    pre_s: float = Field(1.2, gt=0)
    post_s: float = Field(0.8, gt=0)
    height: int = Field(720, ge=144)
    slow_motion: bool = False
    slow_motion_rate: float = Field(0.25, gt=0, le=1)
    max_per_label: int = Field(10, ge=0)


class ReportConfig(_Section):
    trend_metrics: list[str] = Field(
        default_factory=lambda: [
            "contact_height",
            "contact_forward",
            "knee_flex_min",
            "unit_turn_lead_time",
        ]
    )
    rolling_sessions: int = Field(5, ge=1)


def _default_vocabulary() -> dict[str, list[str]]:
    return {
        "good": ["good", "nice", "yes"],
        "late": ["late"],
        "early": ["early"],
        "framed": ["frame", "framed", "shank"],
        "net": ["net"],
        "long": ["long", "out"],
    }


class LabelsConfig(_Section):
    enabled: bool = True
    whisper_model: str = "small"
    language: str = "en"
    max_delay_s: float = Field(3.0, gt=0)
    vocabulary: dict[str, list[str]] = Field(default_factory=_default_vocabulary)

    @model_validator(mode="after")
    def _check_vocabulary(self) -> LabelsConfig:
        seen: dict[str, str] = {}
        for label, words in self.vocabulary.items():
            for word in words:
                key = word.lower()
                if key in seen and seen[key] != label:
                    raise ValueError(f"word '{word}' is mapped to both '{seen[key]}' and '{label}'")
                seen[key] = label
        return self


class Config(_Section):
    player: PlayerConfig = PlayerConfig()
    paths: PathsConfig = PathsConfig()
    audio: AudioConfig = AudioConfig()
    pose: PoseConfig = PoseConfig()
    windows: WindowsConfig = WindowsConfig()
    cleaning: CleaningConfig = CleaningConfig()
    classify: ClassifyConfig = ClassifyConfig()
    metrics: MetricsConfig = MetricsConfig()
    clips: ClipsConfig = ClipsConfig()
    report: ReportConfig = ReportConfig()
    labels: LabelsConfig = LabelsConfig()

    def section_hash(self, *sections: str) -> str:
        """Stable hash of the named top-level sections, used for stage caching.

        Paths are excluded on purpose: moving the data root must not invalidate results.
        """
        dumped = self.model_dump(mode="json", include=set(sections)) if sections else {}
        blob = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(path: Path | None = None, *, cwd: Path | None = None) -> Config:
    """Load and validate a config file.

    With no path, ``./config.yaml`` is used if present, otherwise all defaults apply.
    A relative ``paths.data_root`` is resolved against the config file's directory
    (or ``cwd`` when there is no file), so results do not depend on where you run from.
    """
    base = cwd or Path.cwd()
    if path is None:
        candidate = base / DEFAULT_CONFIG_NAME
        path = candidate if candidate.is_file() else None

    raw: Any = {}
    if path is not None:
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        base = path.resolve().parent

    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:
        where = str(path) if path is not None else "config"
        raise ConfigError(f"{where}: invalid configuration\n{_format_errors(exc)}") from exc

    data_root = config.paths.data_root.expanduser()
    if not data_root.is_absolute():
        data_root = (base / data_root).resolve()
    return config.model_copy(update={"paths": PathsConfig(data_root=data_root)})


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)
