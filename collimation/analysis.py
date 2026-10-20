"""Finding circles in a view down a Newtonian's focuser, and the geometry of collimation.

A circle is picked by clicking three or more points on its edge, then follows that edge frame to
frame: rays from the centre, the strongest step of the same sign along each, a robust fit. Blur
does not matter - a blurred circle still has a sharp centre - which is why one lens can serve the
secondary's edge at 15 cm and the donut at 80 cm.
"""

import math

import cv2
import numpy as np

EDGE_SCALE = 0.005          # the slope along a ray is measured over this share of the radius (1-3 px)
LAMBDA_MM = 550e-6


def fit_circle(pts):
    """Least-squares circle through points: an algebraic start, then a few geometric steps
    (the algebraic fit alone pulls the radius in on a short, noisy arc)."""
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = a / 2, b / 2
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    for _ in range(8):
        dx, dy = x - cx, y - cy
        d = np.maximum(np.hypot(dx, dy), 1e-9)
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])
        step, *_ = np.linalg.lstsq(J, -(d - r), rcond=None)
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-4:
            break
    return float(cx), float(cy), float(abs(r))


def robust_fit(pts, min_sigma=0.5):
    """Circle fit that throws out what is not on the circle - mirror clips, spider vanes, a
    reflection crossing the edge. Returns ((cx, cy, r), inlier mask, rms) or None."""
    pts = np.asarray(pts, float)
    keep = np.ones(len(pts), bool)
    circle = None
    # Loose first, then tight: on the real primary a tenth of the rays stopped on the clips,
    # 22 px inside the rim, all on one side. They pulled the first fit 2 px off, a 3-sigma cut
    # round that threw out the rim itself, and the fit fell apart.
    floors = (12.0, 6.0, 3.0, 0.0, 0.0, 0.0)
    for floor in floors:
        if keep.sum() < 3:
            return None
        circle = fit_circle(pts[keep])
        res = np.hypot(pts[:, 0] - circle[0], pts[:, 1] - circle[1]) - circle[2]
        k = res[keep]
        sigma = max(1.4826 * float(np.median(np.abs(k - np.median(k)))), min_sigma)
        new = np.abs(res) < max(3.0 * sigma, floor)
        if floor == 0.0 and (new == keep).all():
            break
        keep = new
    if keep.sum() < 3:
        return None
    rms = float(np.sqrt(np.mean((np.hypot(pts[keep, 0] - circle[0], pts[keep, 1] - circle[1])
                                 - circle[2]) ** 2)))
    return circle, keep, rms


def prepare(frame):
    """Float image scaled 0..1 and slightly smoothed, for edge finding."""
    img = np.asarray(frame, np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    lo, hi = np.percentile(img[::4, ::4], (1, 99.5))
    img = (img - float(lo)) / max(float(hi - lo), 1e-6)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


def edge_points(img, cx, cy, r, band, polarity=0, n=240, near=None):
    """Where each of n rays from (cx, cy) crosses an edge within r +- band.

    polarity +1 takes only edges that get brighter outwards, -1 darker, 0 either. near (px)
    favours steps close to r over stronger ones further off: the primary's edge sits a few px
    inside the secondary's, and is the stronger of the two. Returns the points, their strength
    and the sign of each step, for the rays that found a clear one."""
    h, w = img.shape
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = np.arange(max(r - band, 1.0), r + band + 0.5, 0.5)
    X = (cx + np.outer(np.cos(th), rs)).astype(np.float32)
    Y = (cy + np.outer(np.sin(th), rs)).astype(np.float32)
    inside = (X >= 0) & (X <= w - 1) & (Y >= 0) & (Y <= h - 1)
    prof = cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # The slope measured over a few px, not between neighbouring samples: a soft edge (the rim a
    # little out of focus, 8 px from bright to dark) has a flat-topped slope whose steepest
    # sample wandered with the noise - the primary's edge jumped 4 px between two fits, frame
    # by frame. Smoothed along the ray, its peak sits still.
    if EDGE_SCALE:
        sigma = float(np.clip(EDGE_SCALE * r, 1.0, 3.0)) / (rs[1] - rs[0])     # in samples
        prof = cv2.GaussianBlur(prof, (0, 0), sigmaX=sigma, sigmaY=1e-3)
    d = np.gradient(prof, axis=1)
    score = d * polarity if polarity else np.abs(d)
    if near:
        score = score * np.exp(-0.5 * ((rs - r) / near) ** 2)
    score = np.where(inside, score, -np.inf)
    score[:, [0, -1]] = -np.inf                 # a peak on the band's edge is not a crossing
    k = np.argmax(score, axis=1)
    rows = np.arange(n)
    best = score[rows, k]
    usable = np.isfinite(best) & (k > 0) & (k < len(rs) - 1)
    # sub-sample peak: a parabola through the best step and its two neighbours
    km, kp = np.clip(k - 1, 0, len(rs) - 1), np.clip(k + 1, 0, len(rs) - 1)
    s0, s1, s2 = score[rows, km], best, score[rows, kp]
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where(usable, 0.5 * (s0 - s2) / (s0 - 2 * s1 + s2), 0.0)
    off = np.clip(np.nan_to_num(off), -0.5, 0.5)
    radius = rs[k] + off * (rs[1] - rs[0])
    if usable.any():
        top = float(np.percentile(best[usable], 80))
        usable &= best > max(0.3 * top, 0.004)
    pts = np.column_stack([cx + np.cos(th) * radius, cy + np.sin(th) * radius])
    return pts[usable], best[usable], np.sign(d[rows, k])[usable]


def follow(img, circle, band=None, polarity=0, min_inliers=12, near=None):
    """Move a circle onto its edge in a new frame. Returns (circle, polarity, quality) or None.

    quality is the share of rays that ended on the circle: low when the edge is faint, half out
    of the frame, or covered."""
    cx, cy, r = circle
    band = band if band is not None else min(max(0.08 * r, 4.0), 25.0)
    if near is None and polarity:
        near = max(0.5 * band, 6.0)   # following: an edge where the circle was beats one further off
    pts, _, signs = edge_points(img, cx, cy, r, band, polarity, near=near)
    if len(pts) < min_inliers:
        return None
    fit = robust_fit(pts)
    if fit is None:
        return None
    new, keep, rms = fit
    if keep.sum() < min_inliers or abs(new[2] - r) > band:
        return None
    if not polarity:
        polarity = int(np.sign(np.median(signs[keep]))) or 1
    return new, polarity, float(keep.sum()) / 240.0


def snap(img, clicks):
    """A circle from points clicked on its edge, moved onto the edge itself.

    Returns (circle, polarity, quality, locked): when no clear edge is found, the circle through
    the clicks is kept, unlocked, so the reading can still be used by eye."""
    clicks = np.asarray(clicks, float)
    c0 = fit_circle(clicks)
    if not np.isfinite(c0).all() or c0[2] < 3:
        return None
    scatter = float(np.abs(np.hypot(clicks[:, 0] - c0[0], clicks[:, 1] - c0[1]) - c0[2]).max())
    band = min(max(scatter + 6.0, 0.06 * c0[2], 6.0), max(0.3 * c0[2], 6.0))
    got = follow(img, c0, band=band, near=max(scatter + 2.0, 3.0))   # the edge you clicked
    if got is None:
        return c0, 0, 0.0, False
    circle, pol, q = got
    again = follow(img, circle, polarity=pol)     # tighter band, now the sign is known
    if again is not None:
        circle, pol, q = again
    return circle, pol, q, True


def rotation_axis(a, b):
    """Where the focuser axis falls in the image, from frames taken before and after turning the
    camera 180 deg in the holder.

    Points on the axis keep their pixel whatever the camera's turn, so b is a turned through
    180 deg about that pixel. Turned about the frame centre instead, a lands shifted from b by
    twice the axis' offset from that centre. Returns ((x, y), response): a low response means
    the two frames did not match - the turn was not 180 deg, or the light changed."""
    fa, fb = prepare(a), prepare(b)
    h, w = fa.shape
    turned = np.ascontiguousarray(fa[::-1, ::-1])
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(turned, np.ascontiguousarray(fb), win)
    return ((w - 1) / 2 + dx / 2, (h - 1) / 2 + dy / 2), float(response)


def coma_tolerance_mm(focal_mm, aperture_mm):
    """How far the primary's axis may miss the focuser axis at the focal plane before coma at the
    field centre grows to the size of the Airy disk: tangential coma 3 theta / (16 F^2) against
    2.44 lambda / D, i.e. 0.0072 F^3 mm (0.9 mm at f/5)."""
    F = focal_mm / aperture_mm
    return 2.44 * LAMBDA_MM * 16.0 * F ** 3 / 3.0


def direction(dx, dy):
    """Which way, in the picture (x right, y down), an offset points."""
    parts = []
    if abs(dx) >= 0.5:
        parts.append(f"{abs(dx):.1f} px {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 0.5:
        parts.append(f"{abs(dy):.1f} px {'down' if dy > 0 else 'up'}")
    return " and ".join(parts) or "on it"
