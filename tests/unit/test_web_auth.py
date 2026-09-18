"""The UI login: cookies expire, cannot be forged, and die with a password change."""

from __future__ import annotations

from pathlib import Path

import pytest

from tennis.web.auth import Auth, cookie_value, load_secret, read_password


def test_a_cookie_is_valid_until_it_expires() -> None:
    auth = Auth("a long password", b"s" * 64)
    token = auth.issue(now=1000.0)
    assert auth.verify(token, now=1001.0)
    expiry = int(token.split(".")[0])
    assert not auth.verify(token, now=expiry + 1)


def test_a_cookie_cannot_be_forged_or_extended() -> None:
    auth = Auth("a long password", b"s" * 64)
    token = auth.issue(now=1000.0)
    expiry, signature = token.split(".")
    assert not auth.verify(f"{int(expiry) + 999999}.{signature}", now=1001.0)
    assert not auth.verify("garbage", now=1001.0)
    assert not auth.verify("", now=1001.0)
    assert not auth.verify(None, now=1001.0)


def test_changing_the_password_or_the_secret_logs_everyone_out() -> None:
    token = Auth("a long password", b"s" * 64).issue(now=1000.0)
    assert not Auth("another password", b"s" * 64).verify(token, now=1001.0)
    assert not Auth("a long password", b"t" * 64).verify(token, now=1001.0)


def test_the_secret_is_kept_private_and_reused(tmp_path: Path) -> None:
    first = load_secret(tmp_path)
    assert (tmp_path / ".ui_secret").stat().st_mode & 0o077 == 0
    assert load_secret(tmp_path) == first


def test_the_password_comes_from_a_file_or_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TENNIS_UI_PASSWORD", raising=False)
    assert read_password(None) is None
    monkeypatch.setenv("TENNIS_UI_PASSWORD", "from the env")
    assert read_password(None) == "from the env"
    path = tmp_path / "pw"
    path.write_text("from the file\n")
    assert read_password(path) == "from the file"


def test_the_cookie_is_found_among_others() -> None:
    assert cookie_value("a=1; tennis_session=abc.def; b=2") == "abc.def"
    assert cookie_value("a=1") is None
    assert cookie_value(None) is None
