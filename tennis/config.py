"""YAML configuration, validated with pydantic.

Every option has a default, so an empty or missing config file is valid. Unknown keys are
rejected so that typos fail loudly instead of being silently ignored. Each pipeline stage
declares the config keys it uses; changing one of them reruns that stage and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tennis.errors import ConfigError

DEFAULT_CONFIG_NAME = "config.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Section):
    data_root: Path = Path("./data")
    labels_dir: Path = Path("./labels")


class ProxyConfig(_Section):
    """The browser-playable copy of each video (H.264, AAC, fast start)."""

    height: int = Field(720, ge=144)
    crf: int = Field(26, ge=10, le=40)
    # auto: Apple's hardware encoder on macOS, libx264 elsewhere.
    encoder: Literal["auto", "libx264", "h264_videotoolbox"] = "auto"


class AudioConfig(_Section):
    """Ball-impact onset detection (tennis.util.audio)."""

    highpass_hz: float = Field(800.0, gt=0)
    highpass_order: int = Field(4, ge=1, le=10)
    onset_k: float = Field(6.0, gt=0)
    min_separation_s: float = Field(0.25, gt=0)
    threshold_window_s: float = Field(5.0, gt=0)
    amplitude_window_s: float = Field(0.02, gt=0)
    min_prominence_db: float = Field(6.0, ge=0)


class PoseConfig(_Section):
    backend: str = Field("yolo", min_length=1)  # checked against the backend registry
    model: str = "yolo11m-pose.pt"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    batch_size: int = Field(16, ge=1)
    imgsz: int = Field(960, ge=32)
    min_person_conf: float = Field(0.2, ge=0, le=1)
    kp_conf_min: float = Field(0.3, ge=0, le=1)
    # Frames per second analysed for tracking people over the whole video.
    sample_fps: float = Field(10.0, gt=0)


class CourtConfig(_Section):
    frames: int = Field(24, ge=1)  # frames sampled across the video for calibration
    min_quality: float = Field(0.5, ge=0, le=1)  # below this, no court: ball stats are skipped
    # Manual corners, as fractions of the frame, in the order far-left, far-right,
    # near-right, near-left (doubles court). Overrides detection for every video when set.
    manual_corners: list[tuple[float, float]] | None = None


class BallConfig(_Section):
    detector: Literal["motion", "motion+yolo"] = "motion"
    yolo_model: str = "yolo11m.pt"
    min_area_px: float = Field(2.0, ge=0)
    max_area_frac: float = Field(0.0015, gt=0)  # of the frame area
    max_gap_frames: int = Field(4, ge=0)


class Config(_Section):
    paths: PathsConfig = PathsConfig()
    proxy: ProxyConfig = ProxyConfig()
    audio: AudioConfig = AudioConfig()
    pose: PoseConfig = PoseConfig()
    court: CourtConfig = CourtConfig()
    ball: BallConfig = BallConfig()

    def section_hash(self, *keys: str) -> str:
        """Stable hash of config values, used for stage caching.

        A key is a whole section (``"pose"``) or one field of it (``"audio.onset_k"``).
        Paths are left out on purpose: moving the data root must not invalidate results.
        """
        data = self.model_dump(mode="json")
        picked: dict[str, Any] = {}
        for key in sorted(set(keys), key=lambda k: (k.count("."), k)):
            section, _, name = key.partition(".")
            if section not in data or (name and name not in data[section]):
                raise KeyError(f"unknown config key '{key}'")
            if not name:
                picked[section] = data[section]
            elif section not in keys:  # a whole section already covers its fields
                picked.setdefault(section, {})[name] = data[section][name]
        blob = json.dumps(picked, sort_keys=True, separators=(",", ":"))
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

    resolved = {
        name: _resolve(getattr(config.paths, name), base) for name in PathsConfig.model_fields
    }
    return config.model_copy(update={"paths": PathsConfig(**resolved)})


def _resolve(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)
