#!/usr/bin/env python3
"""
test_recording.py - check recording.py without a camera, projector or keys.

  python test_recording.py
"""

import os
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np

from recording import PacedWriter, Recorder, laptop_screen


def video_info(path):
    """(codec, width, height, frames) of a video file."""
    cap = cv2.VideoCapture(path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    info = (fourcc, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def test_paced_writer_plays_at_real_speed():
    """Frames arriving unevenly at ~7 fps make a steady 15 fps file of the same length."""
    with tempfile.TemporaryDirectory() as d:
        w = PacedWriter(os.path.join(d, "t.mp4"), 15)
        rng = np.random.default_rng(0)
        t = 0.0
        while t < 4.0:
            w.add(np.full((48, 64, 3), int(t * 50), np.uint8), now=t)
            t += rng.uniform(0.05, 0.25)
        seconds = w.close()
        assert abs(seconds - 4.0) < 0.3, seconds
        assert video_info(w.path) == ("h264", 64, 48, w.written)


def test_laptop_screen_is_the_one_at_the_origin():
    xrandr = """Monitors: 2
 0: +DP-2 1280/339x800/212+2560+0  DP-2
 1: +eDP-1 2560/286x1600/179+0+0  eDP-1
"""
    assert laptop_screen(xrandr) == (2560, 1600)


def test_odd_sizes_are_padded_for_h264():
    with tempfile.TemporaryDirectory() as d:
        w = PacedWriter(os.path.join(d, "odd.mp4"), 15)
        for i in range(5):
            w.add(np.zeros((101, 75, 3), np.uint8), now=i / 15)
        w.close()
        assert video_info(w.path)[1:3] == (76, 102)


def test_recorder_keys_start_and_stop_each_recording():
    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(SimpleNamespace(record_dir=d, record_fps=15, screen_scale=0.25), "test")
        cam, proj = np.zeros((480, 640, 3), np.uint8), np.zeros((800, 1280, 3), np.uint8)
        assert rec.handle_key(ord("c")) and rec.handle_key(ord("p"))
        assert not rec.handle_key(ord("g"))  # other scripts' keys are left alone
        for _ in range(10):
            rec.record("camera", cam)
            rec.record("projector", proj)
            view = cam.copy()
            rec.overlay(view)
            assert view.any(), "REC note should be drawn"
            time.sleep(0.03)
        assert not cam.any(), "recording must not draw on the frame it records"
        rec.handle_key(ord("c"))  # stop the camera only
        assert list(rec.active) == ["projector"]
        rec.close()
        files = sorted(os.listdir(d))
        assert len(files) == 2 and files[0].endswith("_test_camera.mp4") and files[1].endswith("_test_projector.mp4")
        sizes = [video_info(os.path.join(d, f))[1:3] for f in files]
        assert sizes == [(640, 480), (1280, 800)], sizes


def test_recorder_only_offers_its_kinds():
    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(SimpleNamespace(record_dir=d, record_fps=15, screen_scale=0.25), "test",
                       kinds=("camera", "screen"))
        assert not rec.handle_key(ord("p")), "no projector recording in this script"
        assert rec.handle_key(ord("c")) and list(rec.active) == ["camera"]
        rec.record("camera", np.zeros((48, 64, 3), np.uint8))
        rec.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
