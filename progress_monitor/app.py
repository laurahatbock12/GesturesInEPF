"""Local web dashboard that tracks 2D/3D pose-processing progress for every
school / participant / recording in the EmbodiedProductiveFailure dataset.

Run (from this folder, using the main conda env from ../config.json, which already has
Flask):
    <conda_envs.main>/bin/python3 app.py

Then open http://127.0.0.1:5050

The filesystem scan is cheap (a couple of syscalls per recording folder) so it
runs synchronously once at startup to warm the cache, then again in a background
thread on a timer. Every HTTP request is served straight from the in-memory
cache, so the UI never blocks on disk I/O.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scanner import scan
from project_config import PROJECT_FOLDER as DEFAULT_PROJECT_FOLDER

app = Flask(__name__)

_lock = threading.Lock()
_state = {"data": None, "project_folder": DEFAULT_PROJECT_FOLDER, "refreshing": False}


def _refresh():
    with _lock:
        if _state["refreshing"]:
            return _state["data"]
        _state["refreshing"] = True
        project_folder = _state["project_folder"]
    try:
        data = scan(project_folder)
    finally:
        with _lock:
            _state["refreshing"] = False
    with _lock:
        _state["data"] = data
    return data


def _background_refresh_loop(interval_seconds):
    while True:
        time.sleep(interval_seconds)
        try:
            _refresh()
        except Exception as exc:  # keep the loop alive across transient FS errors
            print(f"[progress_monitor] background refresh failed: {exc}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    with _lock:
        data = _state["data"]
    if data is None:
        data = _refresh()
    return jsonify(data)


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    data = _refresh()
    return jsonify(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_folder", default=DEFAULT_PROJECT_FOLDER)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--refresh_interval",
        type=int,
        default=30,
        help="Seconds between automatic background rescans (default: 30).",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _state["project_folder"] = args.project_folder

    start = time.monotonic()
    _refresh()
    print(f"[progress_monitor] initial scan took {(time.monotonic() - start) * 1000:.0f} ms")

    threading.Thread(
        target=_background_refresh_loop,
        args=(args.refresh_interval,),
        daemon=True,
    ).start()

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
