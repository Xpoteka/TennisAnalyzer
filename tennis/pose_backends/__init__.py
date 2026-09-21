"""Pose backend registry.

Backends are looked up by the ``pose.backend`` config value. Register another one with
``register_backend`` (tests use this for a model-free fake backend).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tennis.config import PoseConfig
from tennis.errors import UserError
from tennis.pose_backends.base import (
    GPU_DEVICES,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SKELETON,
    Image,
    PersonPose,
    PoseBackend,
)

__all__ = [
    "GPU_DEVICES",
    "KEYPOINT_NAMES",
    "NUM_KEYPOINTS",
    "SKELETON",
    "BackendFactory",
    "Image",
    "PersonPose",
    "PoseBackend",
    "check_backend",
    "create_backend",
    "register_backend",
    "resolve_device",
    "resolve_model_path",
]

BackendFactory = Callable[[PoseConfig, Path, str], PoseBackend]
_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    _REGISTRY[name] = factory


def _yolo(cfg: PoseConfig, model_path: Path, device: str) -> PoseBackend:
    from tennis.pose_backends.yolo import YoloBackend

    return YoloBackend(model_path, device, cfg.imgsz, cfg.min_person_conf)


register_backend("yolo", _yolo)


def resolve_device(requested: str) -> str:
    """``auto`` picks the first GPU there is: NVIDIA (CUDA), Intel (XPU), Apple (MPS); else CPU.

    An unavailable request falls back to CPU.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    # Which GPUs PyTorch can use depends on the build installed (see pyproject.toml).
    available = {
        "cuda": torch.cuda.is_available(),
        "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available(),
    }
    if requested == "auto":
        return next((name for name in GPU_DEVICES if available[name]), "cpu")
    return requested if available.get(requested, True) else "cpu"


def resolve_model_path(model: str, data_root: Path) -> Path:
    """A bare file name lives in ``<data_root>/models``; paths are used as given."""
    path = Path(model).expanduser()
    if path.parent == Path("."):
        return data_root / "models" / path.name
    return path


def check_backend(name: str) -> BackendFactory:
    factory = _REGISTRY.get(name)
    if factory is None:
        raise UserError(f"unknown pose backend '{name}'; available: {', '.join(sorted(_REGISTRY))}")
    return factory


def create_backend(cfg: PoseConfig, data_root: Path) -> tuple[PoseBackend, str]:
    """Build the configured backend. Returns it with the device actually used."""
    factory = check_backend(cfg.backend)
    device = resolve_device(cfg.device)
    return factory(cfg, resolve_model_path(cfg.model, data_root), device), device
