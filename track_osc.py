#!/usr/bin/env python3
"""
track_osc.py - the headless tracker: send people's floor positions to the
Processing sketches in processing/ (or anything else that speaks OSC).

Same detection and tracking as follow.py (pipeline.py), but with no projector
window - the sketch owns the projector - and no windows at all unless --show.
Each frame it sends /person id x y per tracked person, in metres on the floor
(see osc.py). A status line is printed every few seconds. Ctrl+C to stop.

  --show         a camera window with detections and tracks drawn on. Drag its
                 corner to resize it. Keys in it: c / f = start/stop recording
                 the camera view / full screen (see recording.py; files go in
                 recordings/), q / Esc = quit.

  --simulate N   no camera: N imaginary people wander around the taped area,
                 coming and going, so the sketch can be developed without
                 anyone walking about. Uses the floor size from the calibration.
  --simulate N --with-camera
                 the camera's people as usual, plus N imaginary ones, so you
                 can try a piece for several people on your own. Imaginary
                 people's ids start at 1000000, clear of the tracker's.

Examples:
  python track_osc.py                         # people, camera from the calibration
  python track_osc.py --show                  # also a camera window, to check it
  python track_osc.py --show --window-scale 1 # a smaller camera window
  python track_osc.py --dark-objects          # black objects, for table-top tests
  python track_osc.py --simulate 3            # three imaginary people
  python track_osc.py --simulate 2 --with-camera --show  # you, and two imaginary people
  python track_osc.py --osc-host 192.168.1.20 # sketch on another computer
  python track_osc.py --max-fps 15            # on a laptop: leave CPU for the sketch
"""

import argparse
import sys
import time

import cv2
import numpy as np

from calib_utils import apply_h, fit_scale, laptop_width, load_calibration, resizable_window
from cameras import add_camera_args, open_frames
from osc import OscSender, add_osc_args
from pipeline import FloorPipeline, add_pipeline_args
from recording import Recorder, add_record_args

SHOW_WIN = "track_osc camera (q to quit)"
# --with-camera: imaginary people's ids start here, far above any the tracker reaches
SIM_ID_BASE = 1_000_000


def parse_args():
    p = argparse.ArgumentParser(description="track people and send their floor positions over OSC")
    add_camera_args(p)
    add_pipeline_args(p)
    add_osc_args(p)
    p.add_argument("--show", action="store_true", help="show the camera with detections (for checking)")
    p.add_argument("--window-scale", type=float, default=2.0,
                   help="--show: starting window size as a multiple of the camera image, shrunk "
                        "if need be to fit the laptop screen; drag the corner to resize (default: 2)")
    add_record_args(p)
    p.add_argument("--max-fps", type=float, default=0,
                   help="track at most this many frames a second, to leave CPU for a sketch on the "
                        "same computer, e.g. 15 on a laptop: every 2nd, 3rd... frame the camera gives, "
                        "whichever is the most up to this (default: 0 = every frame, as fast as "
                        "this computer can)")
    p.add_argument("--status-every", type=float, default=5.0,
                   help="seconds between status lines (default: 5)")
    p.add_argument("--simulate", type=int, default=0, metavar="N",
                   help="no camera: send N imaginary people wandering the taped area")
    p.add_argument("--with-camera", action="store_true",
                   help="--simulate: track the camera too, and send its people as well as the imaginary ones")
    p.add_argument("--sim-fps", type=float, default=12.0,
                   help="--simulate: updates per second (default: 12, like a tracker on a laptop's CPU)")
    args = p.parse_args()
    if args.with_camera and not args.simulate:
        p.error("--with-camera goes with --simulate N")
    return args


class SimPerson:
    """An imaginary person walking between random spots, who leaves after a while."""

    def __init__(self, pid, floor_w, floor_h, rng):
        self.id, self.rng = pid, rng
        self.size = np.array([floor_w, floor_h])
        self.pos = self._spot()
        self.target = self._spot()
        self.speed = rng.uniform(0.3, 1.0)  # m/s; slow, since people play with it
        self.leaves_at = time.perf_counter() + rng.uniform(15, 45)

    def _spot(self):
        return self.rng.uniform(0.1, 0.9, 2) * self.size

    def step(self, dt):
        to_go = self.target - self.pos
        dist = np.linalg.norm(to_go)
        if dist < 0.05:
            if self.rng.random() < 0.02:  # sometimes stand still for a bit
                self.target = self._spot()
            return
        self.pos = self.pos + to_go / dist * min(dist, self.speed * dt)


class Crowd:
    """N imaginary people on the taped area: as one leaves, someone new arrives."""

    def __init__(self, args, first_id=1):
        cal = load_calibration(args.calib)
        if "floor" not in cal:
            raise SystemExit(f"No floor size in {args.calib}: run calibrate.py first")
        self.floor_w, self.floor_h = cal["floor"]["width_m"], cal["floor"]["height_m"]
        self.n, self.next_id = args.simulate, first_id
        self.rng = np.random.default_rng()
        self.people = []

    def step(self, now, dt):
        """Move everyone on by dt seconds; returns who is here now."""
        self.people = [p for p in self.people if now < p.leaves_at]
        while len(self.people) < self.n:
            self.people.append(SimPerson(self.next_id, self.floor_w, self.floor_h, self.rng))
            self.next_id += 1
        for p in self.people:
            p.step(dt)
        return self.people


def simulate(args, osc):
    crowd = Crowd(args)
    print(f"Simulating {args.simulate} people on the {crowd.floor_w:g} x {crowd.floor_h:g} m taped area")
    t_last, t_status = time.perf_counter(), 0.0
    while True:
        now = time.perf_counter()
        dt, t_last = now - t_last, now
        people = crowd.step(now, dt)
        osc.send_tracks(people)
        if now - t_status >= args.status_every:
            t_status = now
            print("  " + ", ".join(f"#{p.id} ({p.pos[0]:.2f}, {p.pos[1]:.2f}) m" for p in people))
        time.sleep(max(0.0, 1 / args.sim_fps - (time.perf_counter() - now)))


def track(args, osc):
    pipe = FloorPipeline(args)
    label = "objects" if args.dark_objects else "people"
    fps = 0.0
    t_last, t_status = time.perf_counter(), float("-inf")  # first status line straight away
    rec = None
    crowd = None
    if args.simulate:  # --with-camera: imaginary people too
        crowd = Crowd(args, first_id=SIM_ID_BASE)
        print(f"Adding {args.simulate} imaginary people to the camera's")
    if args.show:
        room = laptop_width(pipe.proj_pos)  # keep it on the laptop screen
        resizable_window(SHOW_WIN, pipe.cam["resolution"],
                         fit_scale(args.window_scale, pipe.cam["resolution"], room and 0.9 * room), (0, 0))
        rec = Recorder(args, "track_osc", kinds=("camera", "screen"))
    for frame in open_frames(args, pipe.source):
        now = time.perf_counter()
        dt, t_last = now - t_last, now
        if pipe.frames > 0:  # the first frame's dt includes camera start-up
            fps = 0.9 * fps + 0.1 / max(dt, 1e-6)
        detections, tracks, _ = pipe.process(frame, dt)
        osc.send_tracks(tracks)
        imaginary = crowd.step(now, dt) if crowd else []
        osc.send_tracks(imaginary)

        if now - t_status >= args.status_every:
            t_status = now
            where = ", ".join(f"#{t.id} ({t.pos[0]:.2f}, {t.pos[1]:.2f})" for t in tracks)
            print(f"{time.strftime('%H:%M:%S')}  {fps:4.1f} fps  {len(tracks)} {label}"
                  + (f": {where} m" if tracks else "")
                  + (f"  (+{len(imaginary)} imaginary)" if crowd else ""))

        if args.show:
            for d in detections:
                cv2.circle(frame, (int(d.cam_xy[0]), int(d.cam_xy[1])), 3,
                           (0, 255, 0) if d.inside else (128, 128, 128), -1)
            for t in tracks:
                x, y = np.round(apply_h(pipe.H_floor_cam, [t.pos])[0]).astype(int)
                cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
                cv2.putText(frame, f"#{t.id}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            for p in imaginary:  # where they'd be standing, in grey
                x, y = np.round(apply_h(pipe.H_floor_cam, [p.pos])[0]).astype(int)
                cv2.circle(frame, (x, y), 8, (160, 160, 160), 1)
            cv2.putText(frame, f"{fps:4.1f} fps  {label}: {len(tracks)}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            rec.record("camera", frame)
            rec.overlay(frame)  # after recording, so the REC note isn't in the video
            cv2.imshow(SHOW_WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            rec.handle_key(key)
            if key in (ord("q"), 27):
                break
    else:
        print("The camera stopped sending frames.")
    if rec:
        rec.close()


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # status lines appear at once, even in a log file
    osc = OscSender(args.osc_host, args.osc_port)
    print(f"Sending /person id x y (metres) to {args.osc_host}:{args.osc_port}. Ctrl+C to stop.")
    try:
        if args.simulate and not args.with_camera:
            simulate(args, osc)
        else:
            track(args, osc)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
