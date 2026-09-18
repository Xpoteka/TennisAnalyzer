"""Find the court in an image and recover the camera.

Court coordinates are metres: origin at the centre of the net, ``x`` across (positive to the
right as seen from the near baseline), ``y`` along the court (positive towards the far
baseline, which is the one higher in the image), ``z`` up.

Detection works on a background image (the median of frames sampled across the video, so
players and balls vanish):

1. **White lines:** a top-hat filter keeps thin bright structures; low saturation drops
   coloured ones. Probabilistic Hough gives segments, merged into long lines.
2. **Model fit:** every choice of two roughly horizontal and two roughly vertical image lines,
   matched to every ordered pair of court lines, gives a homography. The court model is
   projected with it and scored by how much of it lands on white. This is the classic search
   of Farin et al.; it needs no training data.
3. **Refinement:** the best candidate is refined by least squares on the distance to the
   nearest white pixel, with one radial distortion term for wide lenses.
4. **Camera:** focal length, rotation and position follow from the homography, assuming the
   principal point is at the image centre and pixels are square. That turns image points into
   3-D rays, which the ball physics needs.

The quality score is the share of the visible court lines that lie on white pixels. Below
``court.min_quality`` the court counts as not found.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

FloatArray = npt.NDArray[np.float64]

# Dimensions in metres (ITF).
LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
SERVICE_FROM_NET = 6.40
NET_HEIGHT_CENTER = 0.914
NET_HEIGHT_POST = 1.07
NET_POST_X = 6.40  # the posts stand 0.914 m outside the doubles sidelines
HALF_L = LENGTH / 2
HALF_DW = DOUBLES_WIDTH / 2
HALF_SW = SINGLES_WIDTH / 2

# Painted lines as (x1, y1, x2, y2).
COURT_LINES: dict[str, tuple[float, float, float, float]] = {
    "far_baseline": (-HALF_DW, HALF_L, HALF_DW, HALF_L),
    "near_baseline": (-HALF_DW, -HALF_L, HALF_DW, -HALF_L),
    "far_service": (-HALF_SW, SERVICE_FROM_NET, HALF_SW, SERVICE_FROM_NET),
    "near_service": (-HALF_SW, -SERVICE_FROM_NET, HALF_SW, -SERVICE_FROM_NET),
    "left_doubles": (-HALF_DW, -HALF_L, -HALF_DW, HALF_L),
    "right_doubles": (HALF_DW, -HALF_L, HALF_DW, HALF_L),
    "left_singles": (-HALF_SW, -HALF_L, -HALF_SW, HALF_L),
    "right_singles": (HALF_SW, -HALF_L, HALF_SW, HALF_L),
    "centre_service": (0.0, -SERVICE_FROM_NET, 0.0, SERVICE_FROM_NET),
}
# Lines across the court, far to near, and along it, left to right: (name, coordinate).
ACROSS = [
    ("far_baseline", HALF_L),
    ("far_service", SERVICE_FROM_NET),
    ("near_service", -SERVICE_FROM_NET),
    ("near_baseline", -HALF_L),
]
ALONG = [
    ("left_doubles", -HALF_DW),
    ("left_singles", -HALF_SW),
    ("centre_service", 0.0),
    ("right_singles", HALF_SW),
    ("right_doubles", HALF_DW),
]
CORNERS = np.array(
    [[-HALF_DW, HALF_L], [HALF_DW, HALF_L], [HALF_DW, -HALF_L], [-HALF_DW, -HALF_L]], np.float64
)  # far-left, far-right, near-right, near-left

WORK_WIDTH = 1280  # detection runs at most at this width


def court_samples(step_m: float = 0.1) -> FloatArray:
    """Points along every painted line, ``step_m`` apart: (N, 2) court metres."""
    pts = []
    for x1, y1, x2, y2 in COURT_LINES.values():
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step_m) + 1)
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1))
    return np.concatenate(pts)


# --- the camera model --------------------------------------------------------------------


@dataclass
class Calibration:
    """Maps court metres to image pixels (and back, for points on the ground)."""

    H: FloatArray  # 3x3: court (x, y, 1) -> undistorted image pixels
    k1: float  # radial distortion in normalised coordinates (see distort)
    width: int
    height: int
    quality: float
    method: str = "auto"  # auto | manual
    k2: float = 0.0
    # Pinhole camera from H (None if the geometry did not allow it).
    focal: float | None = None
    R: FloatArray | None = None  # world -> camera rotation
    t: FloatArray | None = None

    # Distortion is defined around the image centre, in units of half the diagonal, so it
    # does not depend on the image scale.
    @property
    def _centre(self) -> FloatArray:
        return np.array([self.width / 2, self.height / 2])

    @property
    def _norm(self) -> float:
        return float(np.hypot(self.width, self.height) / 2)

    @property
    def max_radius(self) -> float:
        """Largest undistorted radius the distortion model is valid for.

        Beyond it the radial mapping stops growing and folds back: points far outside the
        picture would land inside it. Such points are not visible and project to NaN.
        """
        r = np.linspace(0.0, 4.0, 801)
        slope = 1 + 3 * self.k1 * r**2 + 5 * self.k2 * r**4
        bad = np.nonzero(slope < 0.02)[0]
        return float(r[bad[0]]) if bad.size else np.inf

    def distort(self, p: FloatArray) -> FloatArray:
        c, s = self._centre, self._norm
        q = (p - c) / s
        r2 = np.sum(q * q, axis=-1, keepdims=True)
        out = c + q * (1 + self.k1 * r2 + self.k2 * r2 * r2) * s
        limit = self.max_radius
        if np.isfinite(limit):
            out = np.where(r2 > limit**2, np.nan, out)
        return np.asarray(out)

    def undistort(self, p: FloatArray) -> FloatArray:
        c, s = self._centre, self._norm
        qd = (np.asarray(p, np.float64) - c) / s
        q = qd.copy()
        for _ in range(12):  # fixed-point iteration; converges fast for moderate distortion
            r2 = np.sum(q * q, axis=-1, keepdims=True)
            q = qd / (1 + self.k1 * r2 + self.k2 * r2 * r2)
        return np.asarray(c + q * s)

    def court_to_image(self, xy: npt.ArrayLike) -> FloatArray:
        """Ground points (N, 2) in metres -> pixels (N, 2)."""
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        h = np.c_[xy, np.ones(len(xy))] @ self.H.T
        with np.errstate(divide="ignore", invalid="ignore"):
            p = h[:, :2] / h[:, 2:3]
        return self.distort(p)

    def image_to_court(self, px: npt.ArrayLike) -> FloatArray:
        """Pixels (N, 2) of points on the ground -> court metres (N, 2)."""
        p = self.undistort(np.asarray(px, np.float64).reshape(-1, 2))
        h = np.c_[p, np.ones(len(p))] @ np.linalg.inv(self.H).T
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(h[:, :2] / h[:, 2:3])

    @property
    def has_camera(self) -> bool:
        return self.focal is not None and self.R is not None and self.t is not None

    @property
    def K(self) -> FloatArray:
        assert self.focal is not None
        return np.array(
            [[self.focal, 0, self.width / 2], [0, self.focal, self.height / 2], [0, 0, 1.0]]
        )

    @property
    def camera_position(self) -> FloatArray | None:
        if self.R is None or self.t is None:
            return None
        return np.asarray(-self.R.T @ self.t)

    def world_to_image(self, xyz: npt.ArrayLike) -> FloatArray:
        """3-D points (N, 3) in metres -> pixels (N, 2). Needs the camera."""
        assert self.R is not None and self.t is not None
        X = np.asarray(xyz, np.float64).reshape(-1, 3)
        cam = X @ self.R.T + self.t
        h = cam @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            p = h[:, :2] / h[:, 2:3]
        return self.distort(p)

    def image_rays(self, px: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Camera centre and unit directions (N, 3) of the rays through pixels (N, 2)."""
        assert self.R is not None and self.t is not None
        p = self.undistort(np.asarray(px, np.float64).reshape(-1, 2))
        d_cam = np.c_[p, np.ones(len(p))] @ np.linalg.inv(self.K).T
        d = d_cam @ self.R  # R.T applied to row vectors
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        centre = self.camera_position
        assert centre is not None
        return centre, d

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "H": self.H.tolist(),
            "k1": self.k1,
            "k2": self.k2,
            "width": self.width,
            "height": self.height,
            "quality": round(self.quality, 4),
            "method": self.method,
        }
        if self.has_camera:
            pos = self.camera_position
            assert pos is not None and self.R is not None and self.t is not None
            out |= {
                "focal": self.focal,
                "R": self.R.tolist(),
                "t": self.t.tolist(),
                "camera_position": [round(float(v), 3) for v in pos],
            }
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Calibration:
        return cls(
            H=np.array(d["H"], np.float64),
            k1=float(d["k1"]),
            k2=float(d.get("k2", 0.0)),
            width=int(d["width"]),
            height=int(d["height"]),
            quality=float(d["quality"]),
            method=str(d.get("method", "auto")),
            focal=d.get("focal"),
            R=np.array(d["R"], np.float64) if d.get("R") is not None else None,
            t=np.array(d["t"], np.float64) if d.get("t") is not None else None,
        )

    def scaled(self, factor: float) -> Calibration:
        """The same calibration for the image resized by ``factor``."""
        S = np.diag([factor, factor, 1.0])
        return Calibration(
            H=S @ self.H,
            k1=self.k1,
            k2=self.k2,
            width=round(self.width * factor),
            height=round(self.height * factor),
            quality=self.quality,
            method=self.method,
        )


def with_camera(cal: Calibration) -> Calibration:
    """Fill in focal length, rotation and translation from the homography, if possible."""
    cx, cy = cal.width / 2, cal.height / 2
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    Hc = T @ cal.H
    a1, b1, c1 = Hc[:, 0]
    a2, b2, c2 = Hc[:, 1]
    estimates = []
    if abs(c1 * c2) > 1e-12:
        f2 = -(a1 * a2 + b1 * b2) / (c1 * c2)
        if f2 > 0:
            estimates.append(f2)
    if abs(c2**2 - c1**2) > 1e-12:
        f2 = (a1**2 + b1**2 - a2**2 - b2**2) / (c2**2 - c1**2)
        if f2 > 0:
            estimates.append(f2)
    if not estimates:
        return cal
    f = float(np.sqrt(np.median(estimates)))
    if not (0.2 * cal.width < f < 20 * cal.width):
        return cal
    Kinv = np.linalg.inv(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]))
    M = Kinv @ cal.H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:  # the court must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return cal
    cal.focal, cal.R, cal.t = f, R, t
    pos = cal.camera_position
    if pos is None or pos[2] <= 0:  # camera below the ground: not a valid solution
        cal.focal = cal.R = cal.t = None
    return cal


# --- detection ---------------------------------------------------------------------------


def line_mask(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Thin, bright, unsaturated structures: candidate court line pixels (0/255)."""
    _h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = max(7, round(w / 110) | 1)
    tophat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    )
    thresh = max(18.0, float(np.percentile(tophat, 97)))
    mask = (tophat > thresh) & (hsv[..., 1] < 90) & (hsv[..., 2] > 90)
    return (mask.astype(np.uint8)) * 255


@dataclass
class ImageLine:
    """An image line a x + b y = c (unit normal), with its supporting segment length."""

    a: float
    b: float
    c: float
    length: float
    p1: tuple[float, float]
    p2: tuple[float, float]

    @property
    def angle(self) -> float:
        """Direction angle in degrees, 0..180 (0 = horizontal)."""
        return float(np.degrees(np.arctan2(self.a, -self.b)) % 180)

    def homogeneous(self) -> FloatArray:
        return np.array([self.a, self.b, -self.c])


def find_lines(mask: npt.NDArray[np.uint8], max_lines: int = 16) -> list[ImageLine]:
    _h, w = mask.shape
    segs = cv2.HoughLinesP(
        mask, 1, np.pi / 360, threshold=40, minLineLength=int(w * 0.04), maxLineGap=int(w * 0.01)
    )
    if segs is None:
        return []
    segs = np.asarray(segs, np.float64).reshape(-1, 4)
    order = np.argsort(-np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]))
    groups: list[list[FloatArray]] = []
    reps: list[ImageLine] = []
    tol_px = max(4.0, w * 0.006)
    for s in segs[order]:
        x1, y1, x2, y2 = s
        placed = False
        for g, rep in zip(groups, reps, strict=True):
            # Same line if both endpoints are close to it and the angle agrees.
            d1 = abs(rep.a * x1 + rep.b * y1 - rep.c)
            d2 = abs(rep.a * x2 + rep.b * y2 - rep.c)
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            dang = min(abs(ang - rep.angle), 180 - abs(ang - rep.angle))
            if d1 < tol_px and d2 < tol_px and dang < 3:
                g.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])
            reps.append(_fit_line([s]))
    lines = [_fit_line(g) for g in groups]
    lines.sort(key=lambda ln: -ln.length)
    return lines[:max_lines]


def _fit_line(segs: list[FloatArray]) -> ImageLine:
    pts = np.concatenate([[s[:2], s[2:]] for s in segs])
    weights = np.repeat([np.hypot(s[2] - s[0], s[3] - s[1]) for s in segs], 2)
    mean = np.average(pts, axis=0, weights=weights)
    cov = np.cov((pts - mean).T, aweights=weights)
    _evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]
    direction = evecs[:, 1]
    proj = (pts - mean) @ direction
    p1 = mean + direction * proj.min()
    p2 = mean + direction * proj.max()
    return ImageLine(
        a=float(normal[0]),
        b=float(normal[1]),
        c=float(normal @ mean),
        length=float(weights.sum() / 2),
        p1=(float(p1[0]), float(p1[1])),
        p2=(float(p2[0]), float(p2[1])),
    )


def _intersect(l1: FloatArray, l2: FloatArray) -> FloatArray | None:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return np.asarray(p[:2] / p[2])


class _Scorer:
    """Scores homographies by how many distinct line pixels the projected court explains.

    Counting distinct pixels, not samples, matters: a court squeezed onto one white line
    would otherwise score perfectly, because every sample lands on white.
    """

    def __init__(self, mask: npt.NDArray[np.uint8]) -> None:
        h, w = mask.shape
        tol = max(2, round(w / 320))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        self.hit = cv2.dilate(mask, kernel) > 0
        self.w, self.h = w, h
        self.samples = court_samples(0.05)
        self.samples_h = np.c_[self.samples, np.ones(len(self.samples))]
        self.cell = max(2, tol)  # pixels are counted on a grid this coarse

    def score_points(self, xy: FloatArray) -> tuple[float, float, float]:
        """(score, hit fraction of the visible court, visible fraction) of projected samples."""
        ok = np.isfinite(xy).all(axis=1)
        inside = ok & (xy[:, 0] >= 0) & (xy[:, 0] < self.w) & (xy[:, 1] >= 0) & (xy[:, 1] < self.h)
        n_in = int(inside.sum())
        if n_in < 0.3 * len(xy):
            return (-np.inf, 0.0, n_in / len(xy))
        ij = xy[inside].astype(np.int64)
        on = self.hit[ij[:, 1], ij[:, 0]]
        cells = (ij[:, 1] // self.cell) * (self.w // self.cell + 1) + ij[:, 0] // self.cell
        hit_cells = np.unique(cells[on]).size
        miss_cells = np.unique(cells[~on]).size
        total = hit_cells + miss_cells
        if total == 0:
            return (-np.inf, 0.0, 0.0)
        return (hit_cells - 0.7 * miss_cells, hit_cells / total, n_in / len(xy))

    def score(self, H: FloatArray) -> tuple[float, float, float]:
        p = self.samples_h @ H.T
        z = p[:, 2]
        if np.any(z <= 0):
            return (-np.inf, 0.0, 0.0)  # part of the court behind the camera
        return self.score_points(p[:, :2] / z[:, None])


def _plausible(H: FloatArray, w: int, h: int) -> bool:
    """The four court corners project to a convex quadrilateral of reasonable size."""
    c = np.c_[CORNERS, np.ones(4)] @ H.T
    if np.any(c[:, 2] <= 0):
        return False
    q = c[:, :2] / c[:, 2:3]
    area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
    if not (0.04 * w * h < area < 25 * w * h):
        return False
    cross = []
    for i in range(4):
        a, b, cc = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        u, v = b - a, cc - b
        cross.append(u[0] * v[1] - u[1] * v[0])
    return bool(all(x > 0 for x in cross) or all(x < 0 for x in cross))


K1_GRID = (0.0, -0.08, -0.16, -0.24, -0.32)  # radial distortions tried (wide lenses bow lines)


def undistort_image(image: npt.NDArray[np.uint8], k1: float) -> npt.NDArray[np.uint8]:
    """The image as a distortion-free camera would see it (same size)."""
    if k1 == 0:
        return image
    h, w = image.shape[:2]
    cal = Calibration(H=np.eye(3), k1=k1, width=w, height=h, quality=0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    src = cal.distort(np.stack([xs, ys], axis=-1).reshape(-1, 2)).reshape(h, w, 2)
    out = cv2.remap(
        image, src[..., 0].astype(np.float32), src[..., 1].astype(np.float32), cv2.INTER_LINEAR
    )
    return np.asarray(out, np.uint8)


def detect(image: npt.NDArray[np.uint8]) -> Calibration | None:
    """Find the court in a (background) BGR image. None when no candidate fits at all."""
    full_h, full_w = image.shape[:2]
    scale = min(1.0, WORK_WIDTH / full_w)
    work = (
        np.asarray(
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), np.uint8
        )
        if scale < 1
        else image
    )
    h, w = work.shape[:2]
    mask = line_mask(work)
    best: tuple[float, FloatArray, float] | None = None
    for k1 in K1_GRID:
        found = _search(line_mask(undistort_image(work, k1)))
        if found is not None and (best is None or found[0] > best[0]):
            best = (found[0], found[1], k1)
    if best is None:
        return None
    H = _orient(best[1], w, h)
    cal = refine(Calibration(H=H, k1=best[2], width=w, height=h, quality=0.0), mask)
    cal = refine_camera(with_camera(cal), mask)
    cal.quality = _quality(cal, _Scorer(mask)) if usable_view(cal) else 0.0
    if scale < 1:
        cal = cal.scaled(1 / scale)
        cal.width, cal.height = full_w, full_h
        cal = with_camera(cal)
    return cal


def _search(mask: npt.NDArray[np.uint8]) -> tuple[float, FloatArray] | None:
    """The best homography for a distortion-free line mask, with its score."""
    h, w = mask.shape
    lines = find_lines(mask)
    if len(lines) < 4:
        return None
    # Lines within 40 degrees of horizontal run across the image; the rest along it. Each
    # family is sorted (top to bottom, left to right), so image order matches court order.
    across = sorted(
        [ln for ln in lines if min(ln.angle, 180 - ln.angle) < 40][:7],
        key=lambda ln: (ln.c - ln.a * w / 2) / ln.b if abs(ln.b) > 1e-9 else 0.0,
    )
    along = sorted(
        [ln for ln in lines if min(ln.angle, 180 - ln.angle) >= 40][:7],
        key=lambda ln: (ln.c - ln.b * h / 2) / ln.a if abs(ln.a) > 1e-9 else 0.0,
    )
    scorer = _Scorer(mask)
    best: tuple[float, FloatArray] | None = None
    # Behind a baseline, image rows are court lines across the court. From the side, image
    # rows run along the court; then left/right in the image is either end, so try both.
    for rows, cols, side_view in ((across, along, False), (along, across, True)):
        if len(rows) < 2 or len(cols) < 2:
            continue
        for la, lb in itertools.combinations(rows, 2):
            for lc, ld in itertools.combinations(cols, 2):
                pts = [
                    _intersect(la.homogeneous(), lc.homogeneous()),
                    _intersect(la.homogeneous(), ld.homogeneous()),
                    _intersect(lb.homogeneous(), lc.homogeneous()),
                    _intersect(lb.homogeneous(), ld.homogeneous()),
                ]
                if any(p is None for p in pts):
                    continue
                img = np.array(pts, np.float32)
                if np.any(np.abs(img) > 4 * max(w, h)):
                    continue
                for (_, ya), (_, yb) in itertools.combinations(ACROSS, 2):
                    for (_, xa), (_, xb) in itertools.combinations(ALONG, 2):
                        if side_view:
                            options = [
                                [(xa, ya), (xa, yb), (xb, ya), (xb, yb)],
                                [(xb, ya), (xb, yb), (xa, ya), (xa, yb)],
                            ]
                        else:
                            options = [[(xa, ya), (xb, ya), (xa, yb), (xb, yb)]]
                        for model in options:
                            m = np.array(model, np.float32)
                            H = cv2.getPerspectiveTransform(m, img).astype(np.float64)
                            if not _plausible(H, w, h):
                                continue
                            s, _, _ = scorer.score(H)
                            if best is None or s > best[0]:
                                best = (s, H)
    return best


def usable_view(cal: Calibration) -> bool:
    """Does the court fill enough of the image to measure the ball on it?

    From very low down (a camera on the ground) the court is a thin sliver, and a line fit
    there is meaningless even when it lands on white pixels.
    """
    edge = np.linspace(0, 1, 40, endpoint=False)[:, None]
    outline = np.concatenate(
        [CORNERS[i] + (CORNERS[(i + 1) % 4] - CORNERS[i]) * edge for i in range(4)]
    )
    q = cal.court_to_image(outline)
    finite = np.isfinite(q).all(axis=1)
    # Most of the outline must map to a real image position (not past the lens model's fold).
    if finite.mean() < 0.85:
        return False
    q = q[finite]
    area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))
    visible_h = min(cal.height, q[:, 1].max()) - max(0.0, q[:, 1].min())
    if area < 0.05 * cal.width * cal.height or visible_h < 0.15 * cal.height:
        return False
    pos = cal.camera_position
    if pos is None:
        return True
    # A camera that sees the court well enough is above it, not too far away, and looks down
    # at the court at a noticeable angle.
    dist = float(np.hypot(pos[0], pos[1]))
    elevation = np.degrees(np.arctan2(pos[2], max(dist, 1e-6)))
    return bool(1.0 <= pos[2] <= 40.0 and dist <= 60.0 and elevation >= 6.0)


def _orient(H: FloatArray, w: int, h: int) -> FloatArray:
    """Make +y point to the far baseline (higher in the image) and +x to the right."""
    far = np.array([0, HALF_L, 1.0]) @ H.T
    near = np.array([0, -HALF_L, 1.0]) @ H.T
    if far[1] / far[2] > near[1] / near[2]:  # "far" is lower in the image: rotate 180 degrees
        H = H @ np.diag([-1.0, -1.0, 1.0])
    right = np.array([HALF_DW, -HALF_L, 1.0]) @ H.T
    left = np.array([-HALF_DW, -HALF_L, 1.0]) @ H.T
    if right[0] / right[2] < left[0] / left[2]:
        H = H @ np.diag([-1.0, 1.0, 1.0])
    return H


def _quality(cal: Calibration, scorer: _Scorer) -> float:
    _, hit, visible = scorer.score_points(cal.court_to_image(scorer.samples))
    # A court that is mostly out of view is a weak basis for the ball physics.
    return hit * min(1.0, visible / 0.8)


def refine(cal: Calibration, mask: npt.NDArray[np.uint8]) -> Calibration:
    """Least-squares refinement of H and k1 on the distance to the nearest line pixel."""
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float64)
    h, w = mask.shape
    samples = court_samples(0.1)
    cap = max(4.0, w / 100)

    def make(params: FloatArray) -> Calibration:
        H = np.append(params[:8], 1.0).reshape(3, 3)
        return Calibration(H=H, k1=params[8], k2=params[9], width=w, height=h, quality=0)

    def residuals(params: FloatArray) -> FloatArray:
        c = make(params)
        p = c.court_to_image(samples)
        ok = (
            np.isfinite(p).all(axis=1)
            & (p[:, 0] >= 0)
            & (p[:, 0] < w - 1)
            & (p[:, 1] >= 0)
            & (p[:, 1] < h - 1)
        )
        r = np.full(len(p), cap)
        q = p[ok]
        r[ok] = np.minimum(_bilinear(dist, q[:, 0], q[:, 1]), cap)
        return r

    H0 = cal.H / cal.H[2, 2]
    x0 = np.concatenate([H0.ravel()[:8], [cal.k1, cal.k2]])
    corners0 = cal.court_to_image(CORNERS)
    slack = 0.08 * w  # the corners may move this far without cost

    def with_prior(params: FloatArray) -> FloatArray:
        c = make(params)
        moved = np.linalg.norm(c.court_to_image(CORNERS) - corners0, axis=1)
        prior = np.maximum(0.0, moved - slack) / 4
        return np.concatenate([residuals(params), np.nan_to_num(prior, nan=1e3)])

    try:
        sol = least_squares(
            with_prior, x0, loss="soft_l1", f_scale=2.0, max_nfev=400, x_scale="jac"
        )
    except (ValueError, np.linalg.LinAlgError):
        return cal
    refined = make(sol.x)
    refined.k1 = float(np.clip(refined.k1, -0.6, 0.6))
    refined.k2 = float(np.clip(refined.k2, -0.6, 0.6))
    refined.quality = cal.quality
    scorer = _Scorer(mask)
    before = scorer.score_points(cal.court_to_image(scorer.samples))[0]
    after = scorer.score_points(refined.court_to_image(scorer.samples))[0]
    return refined if after >= before else cal


def net_samples(step_m: float = 0.1) -> FloatArray:
    """Points along the top of the net (the white tape), in metres: (N, 3)."""
    xs = np.arange(-NET_POST_X, NET_POST_X + 1e-9, step_m)
    zs = np.interp(np.abs(xs), [0.0, NET_POST_X], [NET_HEIGHT_CENTER, NET_HEIGHT_POST])
    return np.stack([xs, np.zeros_like(xs), zs], axis=1)


def _camera_calibration(params: FloatArray, w: int, h: int, k1: float, k2: float) -> Calibration:
    """A calibration from (focal, rotation vector, translation)."""
    f = float(params[0])
    R = np.asarray(cv2.Rodrigues(params[1:4].reshape(3, 1))[0], np.float64)
    t = params[4:7]
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    Hm = K @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    return Calibration(
        H=Hm / Hm[2, 2], k1=k1, k2=k2, width=w, height=h,
        quality=0.0, focal=f, R=R, t=np.asarray(t, np.float64),
    )  # fmt: skip


def refine_camera(cal: Calibration, mask: npt.NDArray[np.uint8]) -> Calibration:
    """Fit the pinhole camera to the court lines and to the top of the net.

    The court is flat, and a view of a plane cannot tell a longer lens from a camera further
    away. The net tape is white too and sits at a known height, which settles it; that is
    what makes heights (net clearance, contact height) come out right. The lens distortion
    found on the court lines is kept as it is.
    """
    if not cal.has_camera:
        return cal
    assert cal.R is not None and cal.t is not None and cal.focal is not None
    h, w = mask.shape
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float64)
    lines = court_samples(0.1)
    net = net_samples(0.1)
    cap = max(4.0, w / 100)
    rvec, _ = cv2.Rodrigues(cal.R)
    x0 = np.concatenate([[cal.focal], rvec.ravel(), cal.t])

    def lookup(p: FloatArray) -> FloatArray:
        ok = (
            np.isfinite(p).all(axis=1)
            & (p[:, 0] >= 0)
            & (p[:, 0] < w - 1)
            & (p[:, 1] >= 0)
            & (p[:, 1] < h - 1)
        )
        r = np.full(len(p), cap)
        r[ok] = np.minimum(_bilinear(dist, p[ok, 0], p[ok, 1]), cap)
        return r

    def residuals(params: FloatArray) -> FloatArray:
        c = _camera_calibration(params, w, h, cal.k1, cal.k2)
        return np.concatenate([lookup(c.court_to_image(lines)), lookup(c.world_to_image(net))])

    try:
        sol = least_squares(residuals, x0, loss="soft_l1", f_scale=1.5, max_nfev=600, x_scale="jac")
    except (ValueError, np.linalg.LinAlgError):
        return cal
    fitted = _camera_calibration(sol.x, w, h, cal.k1, cal.k2)
    pos = fitted.camera_position
    if pos is None or pos[2] <= 0 or not (0.2 * w < (fitted.focal or 0) < 20 * w):
        return cal
    scorer = _Scorer(mask)
    before = scorer.score_points(cal.court_to_image(scorer.samples))[0]
    after = scorer.score_points(fitted.court_to_image(scorer.samples))[0]
    if after < before - 0.03 * abs(before):
        return cal
    fitted.method = cal.method
    return fitted


def _bilinear(img: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return np.asarray(a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy)


def from_corners(
    corners_px: npt.ArrayLike, width: int, height: int, mask: npt.NDArray[np.uint8] | None = None
) -> Calibration:
    """Calibration from four hand-picked corners (far-left, far-right, near-right, near-left)."""
    img = np.asarray(corners_px, np.float32).reshape(4, 2)
    H = cv2.getPerspectiveTransform(CORNERS.astype(np.float32), img).astype(np.float64)
    cal = Calibration(H=H, k1=0.0, width=width, height=height, quality=1.0, method="manual")
    if mask is not None:
        cal = refine(cal, mask)
        cal.method = "manual"
        cal.quality = _quality(cal, _Scorer(mask))
    return with_camera(cal)
