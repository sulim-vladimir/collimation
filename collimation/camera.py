"""Threaded capture from a ZWO ASI camera (the ASI120MM Mini in the focuser)."""

import glob
import os
import threading
import time


class Camera:
    """Grabs frames on a thread of its own and keeps the newest; restarts a stalled stream."""

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.width = cfg["width"] // cfg["bin"]
        self.height = cfg["height"] // cfg["bin"]
        self.pixel_um = float(cfg["pixel_um"]) * cfg["bin"]
        self.exposure_ms = float(cfg["exposure_ms"])
        self.gain = int(cfg["gain"])
        self.fps = 0.0
        self._frame = None
        self._frame_t = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _open(self):
        pass

    def _close(self):
        pass

    def _grab(self):
        raise NotImplementedError

    def _restart(self, reopen=False):
        """Get a stalled stream going again. Only drivers that can stall need it."""

    def set_exposure(self, ms):
        self.exposure_ms = max(0.001, float(ms))

    def set_gain(self, gain):
        self.gain = max(0, int(gain))

    def start(self):
        self._open()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._close()

    def _run(self):
        n, t_fps = 0, time.monotonic()
        misses = restarts = 0
        while not self._stop.is_set():
            img, t = self._grab()
            if img is None:
                # a rate that only updates when a frame arrives would freeze at its last value
                el = time.monotonic() - t_fps
                if el >= 1.0:
                    self.fps, n, t_fps = n / el, 0, time.monotonic()
                misses += 1
                if misses >= 3:
                    # A restart on the old handle "works" and changes nothing when the camera has
                    # dropped off USB and come back: then open it again from scratch.
                    reopen = restarts >= 1
                    print(f"{self.name}: no frame {misses} times running - "
                          + ("reopening the camera" if reopen else "restarting the stream"))
                    restarts += 1
                    try:
                        self._restart(reopen=reopen)
                    except Exception as e:
                        print(f"{self.name}: restart failed: {e}")
                        time.sleep(1.0)
                    misses = 0
                continue
            misses = restarts = 0
            with self._lock:
                self._frame, self._frame_t = img, t
                self._seq += 1
            n += 1
            el = time.monotonic() - t_fps
            if el >= 1.0:
                self.fps, n, t_fps = n / el, 0, time.monotonic()

    def latest(self):
        """(frame, the time it was exposed, sequence number)."""
        with self._lock:
            return self._frame, self._frame_t, self._seq


SDK_CANDIDATES = [
    "/usr/local/lib/libASICamera2.so",
    "/usr/lib/libASICamera2.so",
    "/usr/lib/x86_64-linux-gnu/libASICamera2.so",
    "/usr/lib/aarch64-linux-gnu/libASICamera2.so",   # Raspberry Pi OS 64-bit
]
ZWO_VENDOR_ID = "03c3"


def usb_diagnosis():
    """Why a ZWO camera that is listed might still refuse to open: listing only reads USB
    descriptors; opening claims the interface (a udev rule), streaming needs USB buffer."""
    notes = []
    try:
        with open("/sys/module/usbcore/parameters/usbfs_memory_mb") as f:
            mb = int(f.read().strip())
        if mb < 200:
            notes.append(f"usbfs_memory_mb is {mb}, ZWO needs about 200 - "
                         "`sudo sh -c 'echo 200 > /sys/module/usbcore/parameters/usbfs_memory_mb'`")
    except (OSError, ValueError):
        pass
    rules = []
    for d in ("/etc/udev/rules.d", "/lib/udev/rules.d", "/usr/lib/udev/rules.d"):
        for path in glob.glob(os.path.join(d, "*.rules")):
            try:
                with open(path, errors="ignore") as f:
                    if ZWO_VENDOR_ID in f.read().lower():
                        rules.append(path)
            except OSError:
                pass
    if not rules and os.geteuid() != 0:
        notes.append(f"no udev rule mentions ZWO's vendor id {ZWO_VENDOR_ID}, so only root may "
                     "claim the camera - install asi.rules from the ZWO SDK")
    return notes


def find_sdk(configured=None):
    """The ZWO SDK ships with FireCapture and INDI as well, so look around before giving up."""
    seen = []
    for path in [configured, *SDK_CANDIDATES, *sorted(glob.glob("/opt/FireCapture*/libASICamera2.so"))]:
        if path and path not in seen:
            seen.append(path)
            if os.path.exists(path):
                return path
    raise RuntimeError("libASICamera2.so not found - install the ZWO SDK, or set sdk_lib in "
                       "[camera]. Looked in: " + ", ".join(seen))


def import_zwoasi():
    """zwoasi logs a scary 'ASI SDK library not found' on import when the SDK is not on the
    linker path; it is loaded by path a moment later, so drop that one message."""
    import logging

    class _Quiet(logging.Filter):
        def filter(self, record):
            return "ASI SDK library not found" not in record.getMessage()

    quiet = _Quiet()
    logging.getLogger().addFilter(quiet)
    try:
        import zwoasi as asi
    finally:
        logging.getLogger().removeFilter(quiet)
    return asi


class AsiCamera(Camera):
    _sdk_ready = False

    # The ASI120MM Mini hung for good - listed on USB, never another frame until unplugged -
    # three times out of three when a long exposure (191, 506 ms) was set within the first frames
    # of a stream: a restart restoring its saved exposure. The same change made once the stream
    # had run a while never did. So a stream starts short and takes a long exposure only later.
    START_EXPOSURE_MS = 20.0
    SETTLE_FRAMES = 5

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.cam = None
        self._caps = None
        self._delivered = 0             # frames since the stream started
        self._exposure_pending = False  # a long exposure waiting for the stream to settle

    def _open(self):
        asi = import_zwoasi()
        if not AsiCamera._sdk_ready:
            lib = find_sdk(self.cfg.get("sdk_lib") or None)
            print(f"using ZWO SDK at {lib}")
            asi.init(lib)
            AsiCamera._sdk_ready = True
        names = asi.list_cameras()
        match = [i for i, n in enumerate(names) if self.cfg["name_match"] in n]
        if not match:
            raise RuntimeError(f"camera '{self.cfg['name_match']}' not found; connected: {names}")
        # The SDK reports almost everything as "General error", so say which step failed.
        step = "open the camera"
        try:
            cam = asi.Camera(match[0])
            self._caps = cam.get_controls()
            step = "stop any capture left running by a previous process"
            cam.stop_video_capture()
            step = "set the USB bandwidth"
            self._control(cam, "BandWidth", asi.ASI_BANDWIDTHOVERLOAD, self.cfg["usb_bandwidth"])
            step = "set the image type"
            cam.set_image_type(asi.ASI_IMG_RAW8)
            step = f"set the ROI to {self.width}x{self.height} bin {self.cfg['bin']}"
            cam.set_roi(width=self.width, height=self.height, bins=self.cfg["bin"])
            self.width, self.height = cam.get_roi()[2:4]
            step = "set gain and exposure"
            self.gain = self._control(cam, "Gain", asi.ASI_GAIN, self.gain)
            # always start short, the real exposure follows once frames flow (see set_exposure)
            start_ms = min(self.exposure_ms, self.START_EXPOSURE_MS)
            self._control(cam, "Exposure", asi.ASI_EXPOSURE, int(start_ms * 1000))
            self._delivered = 0
            self._exposure_pending = start_ms != self.exposure_ms
            step = "start video capture"
            cam.start_video_capture()
        except Exception as e:
            hints = usb_diagnosis()
            raise RuntimeError(
                f"{names[match[0]]} is connected and listed, but the SDK failed to {step}: {e}"
                + ("\n  - " + "\n  - ".join(hints) if hints else "")) from e
        self.cam, self._asi = cam, asi

    def _control(self, cam, name, ctype, value):
        """Set a control only if this camera has it, clamped to the range it advertises: the SDK
        reports an out-of-range value as the same "General error" as a dead camera (the Mini's
        gain runs 0-100, not the 0-600 of the USB3 bodies)."""
        caps = (self._caps or {}).get(name)
        if caps is None:
            return None
        lo, hi = int(caps["MinValue"]), int(caps["MaxValue"])
        clamped = min(max(int(value), lo), hi)
        cam.set_control_value(ctype, clamped)
        return clamped

    def set_exposure(self, ms):
        super().set_exposure(ms)
        if self._delivered < self.SETTLE_FRAMES and ms > self.START_EXPOSURE_MS:
            self._exposure_pending = True     # applied by _grab once the stream has settled
            return
        self._apply_exposure()

    def _apply_exposure(self):
        self._exposure_pending = False
        us = self._control(self.cam, "Exposure", self._asi.ASI_EXPOSURE,
                           int(self.exposure_ms * 1000))
        if us is not None:
            self.exposure_ms = us / 1000.0

    def set_gain(self, gain):
        super().set_gain(gain)
        applied = self._control(self.cam, "Gain", self._asi.ASI_GAIN, self.gain)
        if applied is not None:
            self.gain = applied

    def _restart(self, reopen=False):
        """Restart the stream - or, when the camera dropped off USB and came back as a new
        device, open it all over again."""
        if not reopen:
            try:
                self.cam.stop_video_capture()
                time.sleep(0.2)
                if self.exposure_ms > self.START_EXPOSURE_MS:     # a new stream starts short
                    self._control(self.cam, "Exposure", self._asi.ASI_EXPOSURE,
                                  int(self.START_EXPOSURE_MS * 1000))
                    self._exposure_pending = True
                self._delivered = 0
                self.cam.start_video_capture()
                return
            except Exception as e:
                print(f"{self.name}: stream restart failed ({e}) - reopening the camera")
        try:
            self.cam.close()
        except Exception:
            pass
        self._open()

    def _grab(self):
        try:
            img = self.cam.capture_video_frame(timeout=int(self.exposure_ms * 2 + 500))
        except self._asi.ZWO_IOError:
            return None, None
        self._delivered += 1
        if self._exposure_pending and self._delivered >= self.SETTLE_FRAMES:
            try:
                self._apply_exposure()
            except Exception as e:
                print(f"{self.name}: could not set the exposure: {e}")
        return img, time.time() - self.exposure_ms / 2000.0

    def _close(self):
        if self.cam:
            self.cam.stop_video_capture()
            self.cam.close()
