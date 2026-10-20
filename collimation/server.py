"""The page, the picture as MJPEG, and the little API behind them."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

WEB = Path(__file__).resolve().parent / "web"


def serve(coll, port, fps=5.0):
    """Start the server on a thread of its own; returns it. The page is read from disk on every
    request, so editing it needs only a reload."""
    cam = coll.cam
    period = 1.0 / fps

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, content_type="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _state(self):
            return self._send(json.dumps(coll.state()).encode())

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path
            if path in ("/", "/index.html"):
                return self._send((WEB / "index.html").read_bytes(), "text/html")
            if path == "/api/collimate":
                action = q.pop("action", "state")
                body = coll.state() if action == "state" else coll.action(action, q)
                return self._send(json.dumps(body).encode())
            if path == "/api/exposure":
                cam.set_exposure(float(q["ms"]) if q.get("ms") else cam.exposure_ms * float(q["factor"]))
                coll.save()
                return self._state()
            if path == "/api/gain":
                cam.set_gain(int(float(q["value"])) if q.get("value") else cam.gain + int(float(q["delta"])))
                coll.save()
                return self._state()
            if path == "/collimate.mjpg":
                return self._stream()
            self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    img = coll.render()
                    if img is not None:
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                         + jpg.tobytes() + b"\r\n")
                    time.sleep(period)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    return server
