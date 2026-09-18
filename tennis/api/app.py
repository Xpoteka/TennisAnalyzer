"""The FastAPI application: JSON API, uploads, video streaming, and the built React UI.

Security model (unchanged from v1): the server binds to loopback by default. To listen on a
network it needs a password, and every ``/api`` route except login then requires the signed
cookie from :mod:`tennis.api.auth`. Writes from another origin are refused either way, so a
page open in the same browser cannot drive the API.
"""

from __future__ import annotations

import io
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from tennis import __version__
from tennis.api import auth as authn
from tennis.api.uploads import PARTIAL_DIR, UPLOADS_DIR, VIDEO_SUFFIXES, UploadError, Uploads
from tennis.config import Config, load_config
from tennis.db import get_engine
from tennis.worker import Worker

STATIC = Path(__file__).parent / "static"
PUBLIC_API = {"/api/login", "/api/me", "/api/health"}
UPLOAD_STATUS = {
    "bad": 400,
    "type": 415,
    "size": 413,
    "space": 507,
    "offset": 409,
    "conflict": 409,
    "missing": 404,
}


@dataclass
class AppState:
    data_root: Path
    config_path: Path | None
    auth: authn.Auth | None
    secure_cookie: bool
    uploads: Uploads
    logger: logging.Logger

    def config(self) -> Config:
        """The config as it is on disk now: edits in the UI apply to the next job."""
        return load_config(self.config_path, cwd=self.data_root.parent)


def create_app(
    data_root: Path,
    *,
    config_path: Path | None = None,
    password: str | None = None,
    secure_cookie: bool = False,
    run_worker: bool = True,
    logger: logging.Logger | None = None,
) -> FastAPI:
    data_root = data_root.resolve()
    logger = logger or logging.getLogger("tennis")
    get_engine(data_root)  # create the database before the first request
    state = AppState(
        data_root=data_root,
        config_path=config_path,
        auth=authn.Auth(password, authn.load_secret(data_root)) if password else None,
        secure_cookie=secure_cookie,
        uploads=Uploads(data_root / UPLOADS_DIR / PARTIAL_DIR),
        logger=logger,
    )
    worker = Worker(data_root, config_path, logger) if run_worker else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if worker is not None:
            worker.start()
        yield
        if worker is not None:
            worker.stop()

    app = FastAPI(title="Tennis Analyzer", version=__version__, lifespan=lifespan)
    app.state.tennis = state

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if request.method not in ("GET", "HEAD", "OPTIONS") and not _same_origin(request):
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        if (
            state.auth is not None
            and path.startswith("/api/")
            and path not in PUBLIC_API
            and not state.auth.verify(request.cookies.get(authn.COOKIE_NAME))
        ):
            return JSONResponse({"detail": "login required"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(UploadError)
    async def upload_error(_request: Request, exc: UploadError) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.message, **exc.extra}, status_code=UPLOAD_STATUS.get(exc.kind, 400)
        )

    _auth_routes(app, state)
    _upload_routes(app, state)

    from tennis.api import routes

    app.include_router(routes.router, prefix="/api")

    if (STATIC / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (STATIC / path).resolve()
            if path and candidate.is_file() and STATIC.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.tennis
    return state


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return True  # not a browser page (curl, tests)
    host = request.headers.get("host", "")
    return urlparse(origin).netloc == host


def _auth_routes(app: FastAPI, state: AppState) -> None:
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.get("/api/me")
    async def me(request: Request) -> dict[str, Any]:
        needs = state.auth is not None
        ok = not needs or state.auth.verify(request.cookies.get(authn.COOKIE_NAME))  # type: ignore[union-attr]
        return {"password_required": needs, "logged_in": ok, "version": __version__}

    @app.post("/api/login")
    async def login(request: Request) -> Response:
        if state.auth is None:
            return JSONResponse({"logged_in": True})
        body = await request.json()
        password = str(body.get("password", "")) if isinstance(body, dict) else ""
        ok = await run_in_threadpool(state.auth.check_password, password)
        if not ok:
            return JSONResponse({"detail": "wrong password"}, status_code=401)
        response = JSONResponse({"logged_in": True})
        response.headers["set-cookie"] = authn.set_cookie(state.auth.issue(), _https(request))
        return response

    @app.post("/api/logout")
    async def logout(request: Request) -> Response:
        response = JSONResponse({"logged_in": False})
        response.headers["set-cookie"] = authn.clear_cookie(_https(request))
        return response

    def _https(request: Request) -> bool:
        """Mark cookies Secure behind HTTPS, including a proxy that terminates it."""
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        return state.secure_cookie or proto.split(",")[0].strip() == "https"


def _upload_routes(app: FastAPI, state: AppState) -> None:
    def target(name: str, size: int) -> Any:
        return state.uploads.target(state.data_root / UPLOADS_DIR, name, size, VIDEO_SUFFIXES)

    @app.get("/api/uploads")
    async def upload_status(name: str, size: int) -> dict[str, Any]:
        return await run_in_threadpool(state.uploads.status, target(name, size))

    @app.put("/api/uploads")
    async def upload_piece(request: Request, name: str, size: int, offset: int) -> dict[str, Any]:
        body = await request.body()
        return await run_in_threadpool(
            state.uploads.write, target(name, size), offset, len(body), io.BytesIO(body)
        )

    @app.delete("/api/uploads")
    async def upload_discard(name: str, size: int) -> dict[str, Any]:
        return {"discarded": await run_in_threadpool(state.uploads.discard, target(name, size))}
