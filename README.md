# Collimation

An electronic Cheshire for a Newtonian (made for a 150/750). A ZWO ASI120MM Mini has a 1.25" body,
so with an M12 lens on it slides straight into the eyepiece holder and looks down the focuser at
the secondary. A page in the browser draws the circles it finds and shows, as targets, how far
each pair is off - and moves as you turn the screws.

## Install

```
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
```

It needs the ZWO SDK (`libASICamera2.so`): it looks in the usual places and in `/opt/FireCapture*`,
or set `sdk_lib` in `config.toml`. Settings live in `config.example.toml`; copy what differs into
`config.toml`, which is layered over it.

## Run

```
./collimate.sh                # the camera; open http://localhost:8099/
./collimate.sh --sim          # a synthetic, miscollimated tube, with screws to turn
.venv/bin/python -m pytest -q # tests (no camera needed)
```

Ctrl-C (or a plain `kill`) stops the camera properly. The picked circles, the focuser axis, the
lens and the camera's exposure and gain are kept in `data/collimation.json`; **Save frame** writes
the raw frame to `captures/`.

## What it measures

Four circles in one picture, and three pairs that should be concentric:

| | circle | should sit on | adjust |
|---|---|---|---|
| 1 | **S** the secondary's outline | the focuser axis | secondary's centre bolt and turn (rarely needed) |
| 2 | **P** the primary's edge (its clips), seen in the secondary | S - or, until S is picked, the axis | the secondary's 3 tilt screws |
| 3 | **D** the primary's centre donut | **C** the camera's reflection | the primary's collimation screws |

Step 3 needs no reference: the primary sends the camera's own reflection straight back, so D on C
means the primary's axis passes through the camera. The camera sees D and C `2e/f` apart, `e`
being how far the primary's axis passes from the focuser axis at the focal plane and `f` the
telescope's focal length. The page turns the offset into `e` in mm, using the lens focal length
set on the page, and grades it against the coma tolerance `0.0072 F³` mm - coma at the field
centre as large as the Airy disk, 0.9 mm at f/5. *Good* is under half of that.

Steps 1 and 2 are graded as a share of the radius (good: under 3% and 1.5%). Both need the
focuser axis from the 180 deg turn - never the frame centre, which the M12 holder alone put
~140 px off on the real camera. Step 2 with P on the axis asks the same question as P on S (does
the focuser look at the primary's centre?) without the secondary's outline, which only shows with
white paper behind the secondary.

Each reading is drawn as a target: the dot is the offset, magnified, as in the picture; bring it
into the green. A grey tail shows where it has just been, so a small turn of a screw shows which
way that screw moves it. Beside it: which mirror's screws to turn.

## Setting up

* **Lens.** The secondary's edge is ~15-20 cm away, the donut ~80 cm, the camera's reflection
  further still. Blur does not spoil the reading - a blurred circle's centre is still exact - but
  it weakens edges, so put a 1-1.5 mm hole in a card over the lens and focus between the secondary
  and the donut (from infinity, a turn or two outwards; count the turns to go back).
  A 16 mm lens (17.6 x 13.3 deg) fits the whole primary with the focuser racked out; a 12 mm gives
  more room. A webcam lens may not reach focus at all on an M12 holder.
* **Light.** The tube aimed at a bright wall or the daytime sky - never the Sun - and nothing across
  its front (a railing's shadow lies across the primary). **Auto exposure** sets the gain to 0 and
  the exposure so the brightest parts sit at ~75%; leave the other controls alone while it works.
* **White card on the lens.** The card with the hole must be **white on the side facing the
  telescope**: its reflection fills the middle of the secondary's dark reflection, the donut (D)
  shows against it and the hole (C) as a dark spot. With a dark card the middle stays black and
  there is no C.
* **White paper behind the secondary**, opposite the focuser: only then does the secondary's
  outline (S) show; against the black tube it cannot be seen.
* **Seat the camera** square and snug in the holder.

## Doing it

1. **Pick each circle**: its button, then 3-6 clicks spread round its edge, then **Done**. The circle
   snaps onto the nearest edge and follows it from then on (*locked*); one that found no clear edge
   is drawn where you clicked and marked `?`. For P click the rim between the clips; for D the
   donut's outer edge; for C the dark spot of the lens. The dark disk in the middle is the
   secondary's *reflection*, not S. Once a circle has locked, a click far from its edge is refused
   rather than bending it - press Done before the next circle.
2. **Focuser axis**: **1st frame**, turn the camera 180 deg in the holder (a mark on its body helps),
   tighten, **2nd frame**. Whatever stays put when the camera turns is on the axis; the circles are
   moved to where the turn put them; between the two frames they hold still. Redo it whenever the
   camera is reseated: each seating moves the axis a few px (~1% of the primary's radius in step 2).
3. **Adjust in order** 1, 2, 3, then check 2 and 3 again: the secondary's tilt moves the camera's
   reflection too. Step 1 is shown only once S is picked.

A followed circle keeps the size it was picked at and the strength of its edge: when that edge is
lost it stays put, marked `?`, rather than settling on a neighbouring ring, and looks for an edge
of its size nearby (a brisk screw turn moves the donut further than the rays reach).
**Following edges / Frozen** stops the circles moving - handy while a hand is in front of the tube.

## The ASI120MM Mini

It hangs for good - still listed on USB, never another frame, `usb ... reset` every few seconds in
`journalctl -k` - when given a long exposure within the first frames of a stream (three times out
of three on 2026-10-19, each a restart restoring a saved 191 or 506 ms), and when killed in the
middle of a capture. Every stream therefore starts at <= 20 ms and takes a longer exposure after 5
frames, and a kill stops the camera first. If it does hang: unplug it for 5 s.

## How it works

| file | |
|---|---|
| `collimation/analysis.py` | circle fitting, following an edge along rays, the 180 deg turn (phase correlation), the coma tolerance |
| `collimation/tool.py` | the session: picks, axis, readings, Auto exposure, the picture with its overlays |
| `collimation/camera.py` | threaded ZWO capture |
| `collimation/sim.py` | the synthetic tube the tests and `--sim` use |
| `collimation/server.py`, `web/index.html` | the page, the MJPEG picture and the API |

Edges are found along 240 rays from a circle's centre: the strongest step of the circle's sign
(bright-to-dark or dark-to-bright), the slope measured over a few px so a soft, slightly defocused
rim has one peak, then a circle fit that throws out what is not on it (the clips, 22 px inside
the primary's rim, the vanes), loose first and tight last.

## Status

First real-tube session 2026-10-19 (150/750, ASI120MM Mini, 16 mm lens): P locks and holds still
to 0.01 px; D was picked on a faint ring that is either the donut or a glint off the lens. Not yet
done on the real tube: the turn test, S (needs the paper) and C (needs the white card).
Where the primary's edge runs into the secondary's, the two fits lean ~0.7 px towards each other
(seen in the simulator, where they nearly touch).
