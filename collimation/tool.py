"""The collimation page's state: picked circles, the focuser axis, the readings, the picture.

One picture down the focuser holds four circles, and collimating is making three pairs of them
concentric:

  S  the secondary mirror's outline        S on the focuser axis  secondary under the focuser
  P  the primary's edge, in the secondary  P on S                 secondary tilt (its 3 screws)
  D  the primary's centre donut            D on C                 primary tilt (its screws)
  C  the camera's reflection in the primary (the dark spot of its lens, in a white card)

The focuser axis is not the sensor centre - the camera never sits dead centre in the holder, and
an M12 lens holder alone can put it 140 px off - so it is measured: turn the camera 180 deg in the
holder, and whatever stayed put is on the axis. D on C needs no reference at all: the primary
sends the camera's own reflection straight back.
"""

import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .analysis import (coma_tolerance_mm, direction, find_ring, follow, gradients, prepare,
                       ring_strength, rotation_axis, snap)

FEATURES = {"S": "secondary outline", "P": "primary edge", "D": "centre donut",
            "C": "camera reflection"}
PICK_HINTS = {
    "S": " The secondary's own outline, round everything else - NOT the dark disk in the middle,"
         " which is its reflection. It shows only with white paper behind the secondary.",
    "P": " The rim of the bright disk, between the clips.",
    "D": " The donut's outer edge, in the middle of the dark disk.",
    "C": " The dark spot of the lens, in the white card's reflection.",
}
COLOURS = {"S": (255, 170, 60), "P": (80, 200, 255), "D": (90, 90, 255), "C": (230, 110, 230)}
# A followed circle keeps the radius it was picked at, to this much (refocusing changed the real
# ones by ~2%; the neighbours they jumped to were 14-25% off), and an edge less than this share
# of its own strength is not taken for it.
RADIUS_TOL, RADIUS_TOL_PX = 0.04, 2.0
STRENGTH_KEEP = 0.4


class Collimator:
    """The collimation page's state: picked circles, the axis, the readings."""

    def __init__(self, cam, settings, path=None, sim=False):
        """settings: lens_mm, focal_mm, aperture_mm (the [collimation] section of the config);
        path: where the picks, the axis and the camera settings are kept between runs."""
        self.cam, self.path, self.sim = cam, path, sim
        saved = {}
        if path is not None and path.exists():
            try:
                saved = json.loads(path.read_text())
            except ValueError:
                saved = {}
        self.lens_mm = float(saved.get("lens_mm") or settings.get("lens_mm") or 12.0)
        self.focal_mm = float(settings.get("focal_mm", 750.0))
        self.aperture_mm = float(settings.get("aperture_mm", 150.0))
        self.pixel_um = cam.pixel_um
        self.axis = tuple(saved["axis"]) if saved.get("axis") else None
        self.circles = {k: dict(v, locked=False) for k, v in (saved.get("circles") or {}).items()
                        if k in FEATURES}
        self.tracking = True
        self.picking, self.clicks = None, []
        self.turn = None                  # first frame of the 180 deg turn test
        self.turn_circles = {}            # where the circles were at that first frame
        self.msg = "Pick the circles: S, P, D and C." if not self.circles else ""
        self._seq = None
        self._img = None
        self._grads = None
        self._lock = threading.RLock()
        self.saved_camera = saved.get("camera") or {}    # exposure/gain last used here
        self.peak = None           # brightest part of the frame, 0..1 of full scale
        self.clipped = 0.0         # share of pixels at full scale
        self.auto = None           # what automatic exposure is doing, while it runs

    # -- persistence
    def save(self):
        if self.path is None:
            return
        data = {"lens_mm": self.lens_mm,
                "camera": {"exposure_ms": self.cam.exposure_ms, "gain": self.cam.gain},
                "axis": list(self.axis) if self.axis else None,
                "circles": {k: {f: v[f] for f in ("x", "y", "r", "pol", "r0") if f in v}
                            for k, v in self.circles.items()},
                "at": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.path)

    # -- frames
    def update(self):
        """Follow every picked circle into the newest frame (once per frame, however many
        browsers are watching). Returns the frame."""
        frame, _, seq = self.cam.latest()
        if frame is None:
            return None
        with self._lock:
            if seq == self._seq:
                return frame
            self._seq = seq
            self._img = prepare(frame)
            full = 255.0 if frame.dtype == np.uint8 else float(np.iinfo(frame.dtype).max) \
                if np.issubdtype(frame.dtype, np.integer) else float(frame.max() or 1)
            sub = np.asarray(frame)[::4, ::4]
            self.peak = float(np.percentile(sub, 99.5)) / full
            self.clipped = float(np.mean(sub >= 0.98 * full))
            self._grads = gradients(frame)
            # not while the camera is being turned for the axis: the circles would follow the
            # turning picture, and the second frame then turned them once more
            if self.tracking and self.turn is None:
                for c in self.circles.values():
                    self._track(c)
        return frame

    def _accept(self, c, got):
        """A fit the circle may move to: the size it was picked at, and an edge about as strong
        as the one it has been following. (circle, pol, q, strength) or None."""
        if got is None:
            return None
        (x, y, r), pol, q = got
        r0 = c["r0"]
        if abs(r - r0) > max(RADIUS_TOL_PX, RADIUS_TOL * r0):
            return None
        s = float(ring_strength(self._grads, [(x, y)], r, pol)[0])
        s0 = c.get("s0")
        if not s > 0 or (s0 and s < STRENGTH_KEEP * s0):
            return None
        return (x, y, r), pol, q, s

    def _track(self, c):
        """Follow one circle into the new frame, or leave it where it was, marked lost.

        Never onto another edge: a circle keeps the size it was picked at, and an edge much
        weaker than its own is not taken. When the edge has moved out of reach of the rays (a
        brisk screw turn, a bumped tube, the camera turned), it is looked for by size nearby."""
        c.setdefault("r0", c["r"])
        circle, pol = (c["x"], c["y"], c["r"]), c["pol"]
        got = self._accept(c, follow(self._img, circle, polarity=pol))
        if got is None and pol:
            r0 = c["r0"]
            window = float(np.clip(0.5 * r0, 30.0, 80.0))
            (x, y), s = find_ring(self._grads, c["x"], c["y"], r0, pol, window)
            s0 = c.get("s0")
            if s > 0 and (not s0 or s >= STRENGTH_KEEP * s0):
                got = self._accept(c, follow(self._img, (x, y, r0), polarity=pol))
        if got is None:
            c["locked"] = False
            if c.get("s0"):
                # lost for a while - the light changed, a card in the way - a weaker edge of the
                # same size may be taken after all: down to the 0.4 share in about 30 frames
                c["s0"] *= 0.97
            return
        # a big jump is caught only part of the way - the rays on the far side miss the edge -
        # so go again from there until it stops moving
        for _ in range(4):
            moved = math.hypot(got[0][0] - circle[0], got[0][1] - circle[1])
            circle = got[0]
            if moved < 0.2:
                break
            got = self._accept(c, follow(self._img, circle, polarity=got[1])) or got
        (c["x"], c["y"], c["r"]), c["pol"], c["q"], s = got
        # the edge's strength drifts with the light and the exposure; a jump is what is refused
        c["s0"] = s if not c.get("s0") else 0.9 * c["s0"] + 0.1 * s
        c["locked"] = True

    def axis_or_centre(self):
        if self.axis:
            return self.axis, True
        return ((self.cam.width - 1) / 2, (self.cam.height - 1) / 2), False

    # -- browser actions
    def action(self, name, q):
        with self._lock:
            if name == "pick" and q.get("what") in FEATURES:
                self.picking, self.clicks = q["what"], []
                self.msg = (f"Click 3 or more points on the edge of {FEATURES[self.picking]} "
                            f"({self.picking}), spread round it, then Done."
                            + PICK_HINTS.get(self.picking, ""))
            elif name == "click":
                self._click(float(q["fx"]) * self.cam.width, float(q["fy"]) * self.cam.height)
            elif name == "done":
                if self.picking and self.picking in self.circles:
                    self.save()
                self.picking, self.clicks = None, []
                self.msg = ""
            elif name == "forget" and q.get("what") in FEATURES:
                self.circles.pop(q["what"], None)
                self.save()
            elif name == "tracking":
                self.tracking = q.get("on") not in ("0", "false")
            elif name == "lens":
                self.lens_mm = max(0.5, float(q["mm"]))
                self.save()
            elif name == "turn_first":
                self._turn_first()
            elif name == "turn_second":
                self._turn_second()
            elif name == "save_frame":
                self._save_frame()
            elif name == "auto_exposure":
                if self.auto is None:
                    self.auto = "starting"
                    threading.Thread(target=self._auto_exposure, name="collimate-exposure",
                                     daemon=True).start()
            elif name == "axis_reset":
                self.axis, self.turn = None, None
                self.msg = "Axis forgotten: the frame centre stands in for it."
                self.save()
            elif self.sim and name.startswith("sim_"):
                self.msg = self.cam.sim_action(name, q) or ""
            return self.state()

    def _save_frame(self):
        """The raw frame as the camera gave it (no stretch, no drawing), for a closer look."""
        frame, _, _ = self.cam.latest()
        if frame is None:
            self.msg = "No frame to save."
            return
        folder = (self.path.parent.parent if self.path else Path(".")) / "captures"
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"collimation-{time.strftime('%Y%m%d-%H%M%S')}.png"
        cv2.imwrite(str(f), np.asarray(frame))
        self.msg = f"Saved {f}"

    def _auto_exposure(self, target=0.75, tries=10):
        """Set the exposure so the brightest parts of the picture sit at ~3/4 of full scale:
        edges need contrast, and a clipped edge has none. Gain goes to its lowest first: this is
        daylight, and the cleanest picture is the one with the least gain."""
        cam = self.cam
        try:
            if cam.gain:
                cam.set_gain(0)
            mine = (cam.exposure_ms, cam.gain)
            for _ in range(tries):
                # judge a frame exposed after the last change. The ASI keeps handing over frames
                # taken before it for a while: at 28 fps two frames on were still the old
                # exposure, and Auto quartered it down to nothing on frames that were clipped
                _, _, seq0 = cam.latest()
                t0 = time.monotonic()
                while True:
                    if (cam.exposure_ms, cam.gain) != mine:
                        # set by hand meanwhile: the hand wins, and the wait sized from the
                        # old exposure must not hold Auto up for half a minute
                        self.msg = "Auto exposure stopped: exposure or gain was set by hand."
                        return
                    el, exp_s = time.monotonic() - t0, cam.exposure_ms / 1000.0
                    if cam.latest()[2] >= seq0 + 5 and el >= max(0.4, 3 * exp_s):
                        break
                    if el > 5 * exp_s + 3.0:
                        break
                    time.sleep(0.02)
                if cam.latest()[2] < seq0 + 5:
                    self.msg = "Auto exposure: no frames from the camera."
                    return
                self._seq = None
                self.update()
                peak, clipped = self.peak, self.clipped
                if clipped > 0.002 or peak >= 0.97:
                    factor = 0.25                # clipped: how far over is unknown, so a big step
                elif peak < 0.02:
                    factor = 8.0
                else:
                    factor = min(max(target / peak, 0.25), 8.0)
                if abs(factor - 1) < 0.15:
                    self.msg = f"Auto exposure: {cam.exposure_ms:.3g} ms, brightest at {100 * peak:.0f}%."
                    self.save()
                    return
                ms = min(max(cam.exposure_ms * factor, 0.032), 4000.0)
                if ms == cam.exposure_ms:
                    self.msg = (f"Auto exposure stopped at the camera's limit, {ms:.3g} ms: "
                                + ("raise the gain or add light." if factor > 1 else "lower the gain."))
                    return
                self.auto = f"{ms:.3g} ms"
                cam.set_exposure(ms)
                mine = (cam.exposure_ms, cam.gain)
            self.msg = "Auto exposure did not settle - is the light changing?"
        except Exception as e:
            self.msg = f"Auto exposure failed: {e}"
        finally:
            self.auto = None

    def _click(self, x, y):
        if not self.picking:
            self.msg = "Choose which circle first (S, P, D or C), then click its edge."
            return
        c = self.circles.get(self.picking)
        if c and c.get("locked") and len(self.clicks) >= 3:
            off = abs(math.hypot(x - c["x"], y - c["y"]) - c["r"])
            if off > max(15.0, 0.15 * c["r"]):
                # seen for real: the next circle's clicks landing in this one, Done not pressed,
                # and a locked primary edge dragged into a 629 px circle through the middle
                self.msg = (f"That click is {off:.0f} px off the edge of {FEATURES[self.picking]} - "
                            "not used. Press Done, then the next circle's button.")
                return
        self.clicks.append((x, y))
        if len(self.clicks) < 3:
            self.msg = f"{FEATURES[self.picking]}: {len(self.clicks)} point(s), need 3 or more."
            return
        if self._img is None:
            self.update()
        got = snap(self._img, self.clicks)
        if got is None:
            self.msg = "Those points do not make a circle - spread them round the edge."
            return
        (x0, y0, r), pol, q, locked = got
        c = {"x": x0, "y": y0, "r": r, "pol": pol, "q": q, "locked": locked, "r0": r}
        if locked:
            # the size and edge strength it is held to while it is followed
            c["s0"] = max(float(ring_strength(self._grads, [(x0, y0)], r, pol)[0]), 0.0) or None
        self.circles[self.picking] = c
        self.msg = (f"{FEATURES[self.picking]}: r = {r:.0f} px, "
                    + ("locked onto the edge" if locked else
                       "no clear edge there - kept where you clicked, not followed")
                    + ". More clicks refine it; Done when happy.")

    def _turn_first(self):
        frame, _, _ = self.cam.latest()
        if frame is None:
            self.msg = "No frame yet."
            return
        self.turn = frame.copy()
        self.turn_circles = {k: (c["x"], c["y"]) for k, c in self.circles.items()}
        self.msg = ("Now turn the camera 180 deg in the holder (use a mark on its body), "
                    "tighten the screw, and press Second frame. The circles hold still till then.")

    def _turn_second(self):
        frame, _, _ = self.cam.latest()
        if self.turn is None or frame is None:
            self.msg = "Take the first frame first."
            return
        (ax, ay), resp = rotation_axis(self.turn, frame)
        self.turn = None
        if resp < 0.05:
            # the circles held still through the turn, so they are where they were - in a
            # picture that has turned: unless the camera is turned back, look for them nearby
            self.turn_circles = {}
            self.msg = (f"The two frames do not match (response {resp:.2f}): was the turn 180 deg "
                        "and the light steady? Try again.")
            return
        self.axis = (ax, ay)
        # the picked circles turned with the camera: put them where they are now - from where
        # they were at the first frame, whatever happened to them in between. The turn is never
        # quite 180 deg (176-177 on the real tube); following them from there takes up the rest.
        for k, c in self.circles.items():
            x, y = self.turn_circles.get(k, (c["x"], c["y"]))
            c["x"], c["y"] = 2 * ax - x, 2 * ay - y
        self.turn_circles = {}
        cx, cy = (self.cam.width - 1) / 2, (self.cam.height - 1) / 2
        self.msg = (f"Focuser axis at ({ax:.1f}, {ay:.1f}): {math.hypot(ax - cx, ay - cy):.1f} px "
                    f"from the frame centre (match {resp:.2f}).")
        self.save()

    # -- readings
    def steps(self):
        """The three readings: one centre against another (or against the focuser axis).

        Each carries its offset in its own unit (vx, vy, value) with its good/close limits, so
        the page can draw it as a target, and both centres in pixels."""
        axis, measured = self.axis_or_centre()
        axis = axis if measured else None   # the frame centre is no stand-in for a reading:
        #                                     the M12 holder alone can put it 140 px off
        tol = coma_tolerance_mm(self.focal_mm, self.aperture_mm)
        S, P, D, C = (self._xy(k) for k in "SPDC")
        r = {k: v["r"] for k, v in self.circles.items()}
        # the camera sees D and its own reflection 2e/f apart, e = how far the primary's axis
        # passes from the camera (the focuser axis) at the focal plane
        mm_per_px = self.pixel_um / 1000.0 / self.lens_mm * self.focal_mm / 2.0
        out = []

        def step(key, title, mirror, screws, ref, ref_name, tgt, tgt_name, unit, scale,
                 good, close, explain):
            item = {"key": key, "title": title, "mirror": mirror, "screws": screws,
                    "have": False, "unit": unit, "good": good, "close": close,
                    "ref_name": ref_name, "tgt_name": tgt_name}
            needs = [n for n, v in ((ref_name, ref), (tgt_name, tgt)) if v is None]
            if needs:
                item["text"] = "needs " + " and ".join(
                    "the focuser axis (turn test)" if n == "axis" else f"circle {n}"
                    for n in needs)
                out.append(item)
                return
            dx, dy = tgt[0] - ref[0], tgt[1] - ref[1]
            px = math.hypot(dx, dy)
            value = px * scale
            item.update(have=True, dx=round(dx, 2), dy=round(dy, 2), px=round(px, 2),
                        vx=round(dx * scale, 3), vy=round(dy * scale, 3), value=round(value, 3),
                        ref_xy=[round(ref[0], 1), round(ref[1], 1)],
                        tgt_xy=[round(tgt[0], 1), round(tgt[1], 1)],
                        grade="good" if value <= good else "close" if value <= close else "adjust",
                        locked=all(self.circles.get(n, {}).get("locked", True)
                                   for n in (ref_name, tgt_name) if n in FEATURES),
                        text=explain(value, px),
                        where=f"{tgt_name} is {direction(dx, dy)} of {ref_name}")
            out.append(item)

        step("focuser", "1. Secondary under the focuser", "secondary",
             "centre bolt and turn (rarely needed)", axis, "axis", S, "S", "%",
             100.0 / r["S"] if S else 0, 3.0, 8.0,
             lambda v, px: f"{v:.1f}% of the secondary's radius ({px:.1f} px)")
        # P on S, or - while the secondary's outline is not picked - P on the focuser axis:
        # either way the question is whether the focuser looks at the primary's centre
        ref2, name2 = (S, "S") if S else (axis, "axis")
        step("secondary", "2. Secondary tilt", "secondary", "3 tilt screws",
             ref2, name2, P, "P", "%", 100.0 / r["P"] if P else 0, 1.5, 4.0,
             lambda v, px: f"{v:.1f}% of the primary's radius ({px:.1f} px)")
        step("primary", "3. Primary tilt", "primary", "collimation screws",
             C, "C", D, "D", "mm", mm_per_px, 0.5 * tol, tol,
             lambda v, px: f"axes {v:.2f} mm apart at focus ({px:.1f} px); "
                           f"coma = Airy disk at {tol:.2f} mm")
        return out

    def _xy(self, k):
        v = self.circles.get(k)
        return (v["x"], v["y"]) if v else None

    def state(self):
        self.update()          # the readings must follow the screws with no picture open too
        (ax, ay), measured = self.axis_or_centre()
        return {"picking": self.picking, "clicks": len(self.clicks), "tracking": self.tracking,
                "lens_mm": self.lens_mm, "focal_mm": self.focal_mm,
                "aperture_mm": self.aperture_mm, "axis": [round(ax, 1), round(ay, 1)],
                "axis_measured": measured, "turn_pending": self.turn is not None,
                "circles": {k: {"x": round(v["x"], 2), "y": round(v["y"], 2),
                                "r": round(v["r"], 1), "locked": v.get("locked", False)}
                            for k, v in self.circles.items()},
                "steps": self.steps(), "msg": self.msg, "sim": self.sim, "auto": self.auto,
                "peak": None if self.peak is None else round(100 * self.peak, 1),
                "clipped": round(100 * self.clipped, 2),
                "cam": {"name": self.cam.name, "exposure_ms": self.cam.exposure_ms,
                        "gain": self.cam.gain, "fps": round(self.cam.fps, 1)}}

    # -- picture
    def render(self, max_width=1280):
        frame = self.update()
        if frame is None:
            return None
        img = np.asarray(frame)
        if img.ndim == 2:
            lo, hi = np.percentile(img[::4, ::4], (0.5, 99.8))
            img = np.clip((img.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)), 0, 255)
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        k = min(1.0, max_width / img.shape[1])
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        with self._lock:
            self._draw(img, k)
        return img

    def _draw(self, img, k):
        def p(x, y):
            return int(round(x * k)), int(round(y * k))

        (ax, ay), measured = self.axis_or_centre()
        axis_col = (0, 220, 0) if measured else (160, 160, 160)
        cv2.drawMarker(img, p(ax, ay), axis_col, cv2.MARKER_CROSS, 48, 1)
        if measured:
            cv2.putText(img, "axis", (p(ax, ay)[0] + 8, p(ax, ay)[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_col, 1, cv2.LINE_AA)
        for name, c in self.circles.items():
            col = COLOURS[name]
            thick = 3 if name == self.picking else 2
            style = cv2.LINE_AA if c.get("locked", False) else cv2.LINE_4
            cv2.circle(img, p(c["x"], c["y"]), max(1, int(round(c["r"] * k))), col, thick, style)
            cv2.drawMarker(img, p(c["x"], c["y"]), col, cv2.MARKER_TILTED_CROSS, 10, 1)
            lx, ly = p(c["x"] + 0.71 * c["r"], c["y"] - 0.71 * c["r"])
            cv2.putText(img, name if c.get("locked", False) else name + "?", (lx + 4, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        # No arrows for the readings here: near collimation D, C, P and the axis all sit within a
        # few px of each other, and two arrows, their numbers and the axis cross on top made a
        # knot nobody could read. The targets beside the picture show each offset, magnified.
        for x, y in self.clicks:
            cv2.circle(img, p(x, y), 4, (0, 255, 255), -1)
