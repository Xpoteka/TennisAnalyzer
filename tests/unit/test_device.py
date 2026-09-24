"""Which device the models run on."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tennis.pose_backends import resolve_device


def _torch(*, cuda: bool = False, xpu: bool | None = False, mps: bool = False) -> SimpleNamespace:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    if xpu is not None:  # None: a PyTorch build that knows no Intel GPUs
        torch.xpu = SimpleNamespace(is_available=lambda: xpu)
    return torch


@pytest.mark.parametrize(
    ("torch", "requested", "expected"),
    [
        (_torch(), "auto", "cpu"),
        (_torch(xpu=True), "auto", "xpu"),
        (_torch(cuda=True, xpu=True), "auto", "cuda"),
        (_torch(mps=True), "auto", "mps"),
        (_torch(xpu=None), "auto", "cpu"),
        (_torch(xpu=None), "xpu", "cpu"),
        (_torch(), "xpu", "cpu"),
        (_torch(xpu=True), "xpu", "xpu"),
        (_torch(xpu=True), "cpu", "cpu"),
        (_torch(), "cuda", "cpu"),
    ],
)
def test_resolve_device(
    monkeypatch: pytest.MonkeyPatch, torch: SimpleNamespace, requested: str, expected: str
) -> None:
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert resolve_device(requested) == expected
