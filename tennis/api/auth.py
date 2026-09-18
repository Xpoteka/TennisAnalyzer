"""A single password in front of the web UI, for when it runs on a server.

The UI can start commands on the machine it runs on, so once it is reachable from a network
it must know who is asking. There is one user, so there is one password: it comes from the
``TENNIS_UI_PASSWORD`` environment variable or from a file (``--password-file``).

A successful login sets a signed cookie, ``<expiry>.<signature>``. The signing key mixes a
random secret kept in the data folder with the password, so the cookie survives a restart,
and changing the password logs every browser out. The cookie is ``HttpOnly`` and
``SameSite=Strict``: page scripts cannot read it, and another site cannot make the browser
send it. Failed logins are answered slowly, one at a time, so guessing is impractical.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from pathlib import Path

COOKIE_NAME = "tennis_session"
COOKIE_MAX_AGE_S = 30 * 24 * 3600
FAILED_LOGIN_DELAY_S = 1.0
SECRET_NAME = ".ui_secret"
PASSWORD_ENV = "TENNIS_UI_PASSWORD"
MIN_PASSWORD_LENGTH = 10


def read_password(password_file: Path | None) -> str | None:
    """The password from ``--password-file`` or the environment, or None when neither is set."""
    if password_file is not None:
        text = password_file.read_text().strip()
        return text or None
    return os.environ.get(PASSWORD_ENV) or None


def load_secret(directory: Path) -> bytes:
    """The signing secret in ``directory``, created (readable by its owner only) if missing."""
    path = directory / SECRET_NAME
    if path.is_file():
        data = path.read_bytes().strip()
        if len(data) >= 32:
            return data
    directory.mkdir(parents=True, exist_ok=True)
    data = secrets.token_hex(32).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return data


class Auth:
    """Checks the password and issues and verifies the session cookie."""

    def __init__(self, password: str, secret: bytes) -> None:
        self._password = password.encode()
        self._key = hmac.new(secret, b"tennis-ui:" + self._password, hashlib.sha256).digest()
        self._login_lock = threading.Lock()

    def check_password(self, attempt: str) -> bool:
        # One attempt at a time, and a failure holds the lock for a moment: a guesser gets
        # about one try per second however many connections it opens.
        with self._login_lock:
            ok = hmac.compare_digest(attempt.encode(), self._password)
            if not ok:
                time.sleep(FAILED_LOGIN_DELAY_S)
            return ok

    def issue(self, now: float | None = None) -> str:
        expiry = int((now if now is not None else time.time()) + COOKIE_MAX_AGE_S)
        return f"{expiry}.{self._sign(str(expiry))}"

    def verify(self, token: str | None, now: float | None = None) -> bool:
        if not token:
            return False
        expiry, _, signature = token.partition(".")
        if not expiry.isdigit() or not hmac.compare_digest(signature, self._sign(expiry)):
            return False
        return int(expiry) > (now if now is not None else time.time())

    def _sign(self, value: str) -> str:
        return hmac.new(self._key, value.encode(), hashlib.sha256).hexdigest()


def cookie_value(cookie_header: str | None) -> str | None:
    """Our cookie's value from a ``Cookie`` request header."""
    if not cookie_header:
        return None
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name == COOKIE_NAME:
            return value
    return None


def set_cookie(token: str, secure: bool) -> str:
    flags = f"Path=/; Max-Age={COOKIE_MAX_AGE_S}; HttpOnly; SameSite=Strict"
    return f"{COOKIE_NAME}={token}; {flags}{'; Secure' if secure else ''}"


def clear_cookie(secure: bool) -> str:
    flags = "Path=/; Max-Age=0; HttpOnly; SameSite=Strict"
    return f"{COOKIE_NAME}=; {flags}{'; Secure' if secure else ''}"
