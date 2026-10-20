"""A synthetic, slightly miscollimated Newtonian seen down its focuser - the test bench."""

import math
import time

import cv2
import numpy as np

from .camera import Camera


class SimCollimationCamera(Camera):
    """A synthetic view down a slightly miscollimated Newtonian's focuser, with a white card
    opposite the focuser: secondary outline, the primary in it with its clips, the secondary's
    reflection with the vanes, the camera's collar and lens, and the donut. The camera can be
    turned in its holder and the screws nudged, from the page."""

    def __init__(self, name, cfg, fps=10.0, seed=2):
        super().__init__(name, cfg)
        self.period = 1.0 / fps
        self.rng = np.random.default_rng(seed)
        w, h = self.width, self.height
        self.axis = (0.5 * w + 11.3, 0.5 * h - 7.6)    # the camera sits a little off-centre
        self.turned = 0.0
        self.secondary = np.array([4.0, -3.0])         # S centre - axis
        self.tilt2 = np.array([9.0, 5.5])              # P centre - S centre
        self.tilt1 = np.array([-7.0, 4.0])             # D centre - C centre
        self.r_s = 0.31 * h
        self._next = time.monotonic()

    def scene(self, turned=None):
        turned = self.turned if turned is None else turned
        h, w = self.height, self.width
        ax = np.array(self.axis)
        S = ax + self.secondary
        P = S + self.tilt2
        C = ax + 0.5 * self.secondary            # the camera's reflection, near the axis
        D = C + self.tilt1
        r_s = self.r_s
        r_p = 0.965 * r_s
        img = np.full((h, w), 140, np.uint8)     # the white card behind the secondary
        disk = lambda c, r: cv2.circle(np.zeros((h, w), np.uint8), (int(round(c[0] * 16)),
                                       int(round(c[1] * 16))), int(round(r * 16)), 1, -1,
                                       cv2.LINE_AA, 4).astype(bool)
        in_s = disk(S, r_s)
        img[in_s] = 38                           # tube wall seen in the secondary
        prim = np.full((h, w), 175, np.uint8)    # the lit primary
        for a in (np.radians(30), np.radians(150), np.radians(270)):   # mirror clips
            u = np.array([math.cos(a), math.sin(a)])
            v = np.array([-u[1], u[0]])
            pts = np.array([P + u * (r_p + 4) + v * 16, P + u * (r_p - 12) + v * 16,
                            P + u * (r_p - 12) - v * 16, P + u * (r_p + 4) - v * 16])
            cv2.fillPoly(prim, [np.round(pts).astype(np.int32)], 40)
        sr = 0.36 * r_s                          # the secondary's reflection, round C
        cv2.circle(prim, tuple(np.round(C).astype(int)), int(sr), 30, -1, cv2.LINE_AA)
        for a in np.radians([45, 135, 225, 315]):   # spider vanes
            u = np.array([math.cos(a), math.sin(a)])
            cv2.line(prim, tuple(np.round(C + u * sr).astype(int)),
                     tuple(np.round(C + u * 2 * r_p).astype(int)), 30, 3, cv2.LINE_AA)
        cv2.circle(prim, tuple(np.round(C).astype(int)), int(0.27 * r_s), 150, -1, cv2.LINE_AA)
        cv2.circle(prim, tuple(np.round(D).astype(int)), int(0.14 * r_s), 235, -1, cv2.LINE_AA)
        cv2.circle(prim, tuple(np.round(D).astype(int)), int(0.065 * r_s), 150, -1, cv2.LINE_AA)
        cv2.circle(prim, tuple(np.round(C).astype(int)), int(0.04 * r_s), 12, -1, cv2.LINE_AA)
        m = in_s & disk(P, r_p)
        img[m] = prim[m]
        if turned:
            R = cv2.getRotationMatrix2D((float(ax[0]), float(ax[1])), turned, 1.0)
            img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
        return img

    def truth(self):
        """Where the circles really are in the picture (for tests): axis, S, P, C, D."""
        ax = np.array(self.axis)
        S = ax + self.secondary
        C = ax + 0.5 * self.secondary
        out = {"axis": ax, "S": S, "P": S + self.tilt2, "C": C, "D": C + self.tilt1}
        if self.turned:
            a = math.radians(-self.turned)       # warpAffine turns the picture counterclockwise
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            out = {k: ax + R @ (v - ax) for k, v in out.items()}
        return out

    def frame(self):
        img = self.scene() * (self.exposure_ms / 10.0) * (1 + self.gain / 100.0) / 2.0
        img = img + self.rng.normal(0, 2.0, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _grab(self):
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + self.period, time.monotonic())
        return self.frame(), time.time()

    def sim_action(self, name, q):
        if name == "sim_turn":
            self.turned = (self.turned + 180.0) % 360.0
            return "Simulated camera turned 180 deg in the holder."
        if name == "sim_nudge":
            which = q.get("what")
            attr = {"secondary": "tilt2", "primary": "tilt1"}.get(which)
            if attr is None:
                return None
            v = getattr(self, attr)
            # a screw turn takes off about a third, and pushes a little sideways
            side = np.array([-v[1], v[0]]) * self.rng.normal(0, 0.15)
            setattr(self, attr, v * 0.65 + side)
            return f"Simulated {which} screw turned."
        return None
