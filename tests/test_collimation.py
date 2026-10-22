import math

import numpy as np
import pytest

from collimation.analysis import (coma_tolerance_mm, fit_circle, prepare, robust_fit,
                                  rotation_axis, snap)
from collimation.config import ROOT, load_config
from collimation.sim import SimCollimationCamera
from collimation.tool import Collimator

CCFG = {"focal_mm": 750.0, "aperture_mm": 150.0, "lens_mm": 12.0}


def sim():
    # the example config, never config.toml: the tests must not depend on one tube's settings
    return SimCollimationCamera("sim", load_config(ROOT / "config.example.toml")["camera"])


def radii(cam):
    r_s = cam.r_s
    return {"S": r_s, "P": 0.965 * r_s, "D": 0.14 * r_s, "C": 0.04 * r_s}


def rough_clicks(c, r, angles=(20, 140, 260), err=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return [(c[0] + (r + rng.uniform(-err, err)) * math.cos(math.radians(a)),
             c[1] + (r + rng.uniform(-err, err)) * math.sin(math.radians(a))) for a in angles]


def test_fit_circle_exact_and_robust():
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    pts = np.column_stack([100 + 40 * np.cos(th), 80 + 40 * np.sin(th)])
    assert fit_circle(pts) == pytest.approx((100, 80, 40), abs=1e-6)
    bad = pts.copy()
    bad[:6] += 15.0                       # a clip / a vane: points off the circle
    (cx, cy, r), keep, _ = robust_fit(bad)
    assert (cx, cy, r) == pytest.approx((100, 80, 40), abs=0.05)
    assert keep.sum() == 54


@pytest.mark.parametrize("name", ["S", "P", "D", "C"])
def test_snap_finds_each_circle_from_three_rough_clicks(name):
    """Clicked a few px off the edge, the circle still lands on it - through the clips, the
    vanes, and the primary's edge hiding behind the secondary's on one side."""
    cam = sim()
    img = prepare(cam.frame())
    truth = cam.truth()[name]
    r = radii(cam)[name]
    err = min(3.0, 0.2 * r)
    circle, pol, q, locked = snap(img, rough_clicks(truth, r, err=err))
    assert locked
    # S and P nearly touch on one side here, and lean ~0.7 px towards each other there
    assert math.hypot(circle[0] - truth[0], circle[1] - truth[1]) < (0.9 if name in "SP" else 0.6)
    assert circle[2] == pytest.approx(r, abs=1.5)


def test_rotation_axis_from_a_180_deg_turn():
    cam = sim()
    a = cam.frame()
    cam.sim_action("sim_turn", {})
    b = cam.frame()
    (x, y), resp = rotation_axis(a, b)
    assert resp > 0.2
    assert (x, y) == pytest.approx(cam.axis, abs=0.5)
    # the frame centre is clearly wrong here, so the test means something
    assert math.hypot(cam.axis[0] - (cam.width - 1) / 2, cam.axis[1] - (cam.height - 1) / 2) > 10


def pick(coll, cam, name):
    truth = cam.truth()[name]
    r = radii(cam)[name]
    coll.action("pick", {"what": name})
    for x, y in rough_clicks(truth, r, err=min(3.0, 0.2 * r)):
        coll.action("click", {"fx": x / cam.width, "fy": y / cam.height})
    coll.action("done", {})


def test_collimator_readings_follow_the_screws(tmp_path):
    cam = sim()
    cam._frame, cam._seq = cam.frame(), 1          # one frame, no capture thread
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    for n in "SPDC":
        pick(coll, cam, n)
    # the turn test puts the axis where it really is, and the circles where they went
    coll.action("turn_first", {})
    cam.sim_action("sim_turn", {})
    cam._frame, cam._seq = cam.frame(), 2
    coll.action("turn_second", {})
    coll.update()
    assert coll.axis == pytest.approx(cam.axis, abs=0.5)
    t = cam.truth()
    steps = {s["key"]: s for s in coll.steps()}
    # S and P each lean ~0.7 px towards the other where the primary's edge runs into the
    # secondary's and the two merge in the blur: 0.4% of the radius, a quarter of "good"
    for key, a, b, tol in (("focuser", "axis", "S", 1.0), ("secondary", "S", "P", 1.5),
                           ("primary", "C", "D", 0.5)):
        want = t[b] - t[a]
        assert (steps[key]["dx"], steps[key]["dy"]) == pytest.approx(tuple(want), abs=tol), key
    assert steps["primary"]["grade"] == "adjust"
    # turning the primary's screws moves D onto C, and the reading follows the picture
    for _ in range(12):
        cam.sim_action("sim_nudge", {"what": "primary"})
    cam._frame, cam._seq = cam.frame(), 3
    coll.update()
    assert {s["key"]: s for s in coll.steps()}["primary"]["px"] < 1.0
    # and it all comes back after a restart
    again = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    assert again.axis == pytest.approx(coll.axis)
    assert set(again.circles) == set("SPDC")


def test_coma_tolerance():
    assert coma_tolerance_mm(750, 150) == pytest.approx(0.89, abs=0.01)


def test_auto_exposure_brings_a_clipped_picture_back(tmp_path):
    """Started at the console's night setting, the daytime tube is all white - and an all-white
    frame stretched for display is all black, with no edge to find."""
    cam = sim()
    cam.set_exposure(506.0)
    cam.start()
    try:
        coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
        coll._auto_exposure()
        assert coll.clipped < 0.002
        assert 0.6 < coll.peak < 0.9, coll.msg
        assert cam.exposure_ms < 506.0
        assert coll.auto is None
    finally:
        cam.stop()


def test_a_click_far_from_a_locked_edge_is_refused(tmp_path):
    """Done not pressed, the next circle's clicks used to land in this one and drag it away."""
    cam = sim()
    cam._frame, cam._seq = cam.frame(), 1
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    coll.action("pick", {"what": "P"})
    for x, y in rough_clicks(cam.truth()["P"], radii(cam)["P"]):
        coll.action("click", {"fx": x / cam.width, "fy": y / cam.height})
    r = coll.circles["P"]["r"]
    t = cam.truth()["D"]
    coll.action("click", {"fx": t[0] / cam.width, "fy": t[1] / cam.height})    # meant for D
    assert "not used" in coll.msg
    assert coll.circles["P"]["r"] == pytest.approx(r, abs=1.0)


def test_readings_need_the_measured_axis_and_fall_back_to_p_on_the_axis(tmp_path):
    cam = sim()
    cam._frame, cam._seq = cam.frame(), 1
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    pick(coll, cam, "P")
    steps = {s["key"]: s for s in coll.steps()}
    assert not steps["focuser"]["have"] and not steps["secondary"]["have"]
    assert "turn test" in steps["secondary"]["text"]      # never against the frame centre
    coll.axis = tuple(cam.axis)
    sec = {s["key"]: s for s in coll.steps()}["secondary"]
    assert sec["have"] and sec["ref_name"] == "axis"      # no S: P on the axis instead
    want = cam.truth()["P"] - cam.truth()["axis"]
    assert (sec["dx"], sec["dy"]) == pytest.approx(tuple(want), abs=0.7)


def ring_frame(cam, rings, bg=60):
    """A frame with bright disks (x, y, r, level) on a dark ground, blurred like the real rim."""
    import cv2
    img = np.full((cam.height, cam.width), bg, np.float32)
    for x, y, r, level in rings:
        cv2.circle(img, (int(round(x * 16)), int(round(y * 16))), int(round(r * 16)), level, -1,
                   cv2.LINE_AA, 4)
    img = cv2.GaussianBlur(img, (0, 0), 2.0) + np.random.default_rng(1).normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_find_ring_holds_the_size():
    """A bigger disk of the same edge sign close by does not win: only the size picked can."""
    from collimation.analysis import find_ring, gradients
    cam = sim()
    frame = ring_frame(cam, [(460, 400, 34, 220), (531, 412, 28, 220)])
    (x, y), s = find_ring(gradients(frame), 500, 405, 28, -1, 40)
    # a coarse find - the edge is then followed along rays to a fraction of a px
    assert (x, y) == pytest.approx((531, 412), abs=2.5)
    assert s > 0


def test_a_followed_circle_is_not_pulled_onto_a_neighbour(tmp_path):
    """Seen for real: the donut (r 28) settled on a ring of 32, the camera's reflection (r 78) on
    the lens rim inside it (r 61). When its own edge goes, a circle stays put, marked lost."""
    cam = sim()
    cam._frame, cam._seq = ring_frame(cam, [(640, 480, 28, 220)]), 1
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    coll.action("pick", {"what": "D"})
    for x, y in rough_clicks((640, 480), 28, err=2.0):
        coll.action("click", {"fx": x / cam.width, "fy": y / cam.height})
    coll.action("done", {})
    assert coll.circles["D"]["locked"]
    cam._frame, cam._seq = ring_frame(cam, [(646, 476, 33, 220)]), 2     # only a bigger one
    coll.update()
    d = coll.circles["D"]
    assert not d["locked"]
    assert (d["x"], d["y"], d["r"]) == pytest.approx((640, 480, 28), abs=0.6)
    cam._frame, cam._seq = ring_frame(cam, [(590, 485, 33, 220), (661, 468, 28, 220)]), 3
    coll.update()                                    # its own edge back, moved: found again
    d = coll.circles["D"]
    assert d["locked"]
    assert (d["x"], d["y"], d["r"]) == pytest.approx((661, 468, 28), abs=0.6)


def test_a_big_jump_is_found_again(tmp_path):
    """A brisk screw turn moves the donut further than its edge can be followed along rays."""
    cam = sim()
    cam._frame, cam._seq = cam.frame(), 1
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    for n in "PDC":
        pick(coll, cam, n)
    cam.tilt1 = cam.tilt1 + np.array([24.0, -14.0])  # D jumps ~28 px in one frame
    cam._frame, cam._seq = cam.frame(), 2
    coll.update()
    t = cam.truth()
    for n in "PDC":
        c = coll.circles[n]
        assert c["locked"], n
        assert (c["x"], c["y"]) == pytest.approx(tuple(t[n]), abs=0.9 if n == "P" else 0.6), n
        assert c["r"] == pytest.approx(radii(cam)[n], abs=1.5), n


def test_turn_test_with_the_edges_followed_meanwhile(tmp_path):
    """Seen for real: the circles followed the turning picture, then the second frame turned
    them once more - D ended up where it had been before the turn, on another ring."""
    cam = sim()
    cam.secondary = np.array([30.0, -20.0])      # C and D ~20 px off the axis, as on the real tube
    cam._frame, cam._seq = cam.frame(), 1
    coll = Collimator(cam, CCFG, path=tmp_path / "coll.json", sim=True)
    for n in "SPDC":
        pick(coll, cam, n)
    coll.action("turn_first", {})
    for seq, turned in enumerate(np.arange(5.0, 180.1, 5.0), 2):   # a slow turn, many frames
        cam.turned = float(turned)
        cam._frame, cam._seq = cam.frame(), seq
        coll.update()
    coll.action("turn_second", {})
    cam._frame, cam._seq = cam.frame(), 100
    coll.update()
    t = cam.truth()
    for n in "SPDC":
        c = coll.circles[n]
        assert c["locked"], n
        assert (c["x"], c["y"]) == pytest.approx(tuple(t[n]), abs=0.9 if n in "SP" else 0.6), n
