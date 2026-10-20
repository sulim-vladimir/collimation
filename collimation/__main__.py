"""python -m collimation [--sim] [--port N]: the collimation page on http://<this machine>:N/"""

import argparse
import signal
import time

from .config import ROOT, load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="collimation",
                                 description="Collimate a Newtonian with a camera in its focuser")
    ap.add_argument("--config", help="config file (default config.toml, else config.example.toml)")
    ap.add_argument("--port", type=int, help="page port (default from the config)")
    ap.add_argument("--sim", action="store_true", help="a synthetic, miscollimated tube")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    from .server import serve
    from .tool import Collimator
    if args.sim:
        from .sim import SimCollimationCamera
        cam = SimCollimationCamera("sim", cfg["camera"]).start()
        state = ROOT / "data" / "collimation-sim.json"
    else:
        from .camera import AsiCamera
        cam = AsiCamera("camera", cfg["camera"]).start()
        state = ROOT / "data" / "collimation.json"
    coll = Collimator(cam, cfg["collimation"], state, sim=args.sim)
    # the exposure and gain last used here (cam.start() opened the stream short already: a long
    # one applied right away is what hung the Mini, see AsiCamera)
    saved = coll.saved_camera
    if "gain" in saved:
        cam.set_gain(saved["gain"])
    if "exposure_ms" in saved:
        cam.set_exposure(saved["exposure_ms"])

    port = args.port or cfg["collimation"]["port"]
    serve(coll, port)
    print(f"collimation page on http://localhost:{port}/   (Ctrl-C to stop)")

    # A plain kill must stop the camera properly too: the ASI120MM Mini, killed mid-capture,
    # came back listed but never sent another frame until it was unplugged.
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
