#!/usr/bin/env python3
"""
follow.py - project a circle on the floor under each tracked person or object.

The visual test for tracking. Uses calibration.json from calibrate.py,
including the camera it was calibrated with. Detection and tracking are in
pipeline.py (people with YOLO, or --dark-objects); each tracked position is
mapped floor -> projector and a circle of a fixed real-world size drawn there.

Only detections inside the taped rectangle get a circle. Positions are smoothed
by a floor-space tracker (--smooth), which keeps ids when someone is missed for
a moment (--reattach-time).
With --osc, the positions are also sent as OSC, exactly as track_osc.py sends
them (see osc.py), to test a receiver side by side with the circles.

Windows: one per camera (its feed with detections and tracks drawn on), the
projector output itself, and on the laptop screen a "projector preview": a
copy of what's being projected (not needed, so not shown, with --windowed).
Drag a window's corner to resize it; --window-scale and --projector-scale set
the starting sizes.

Keys (in any window): g = floor grid on / off (starts off), q / Esc = quit,
  c / p / f = start/stop recording the camera window / projector / full screen
  (see recording.py; files go in recordings/).

Examples:
  python follow.py                                  # people, camera from the calibration
  python follow.py --dark-objects                   # black objects instead of people
  python follow.py --size 0.08                      # bigger circles, relative to the projected area
  python follow.py --radius 0.2                     # fixed 20 cm radius
  python follow.py --source walk.mp4 --windowed     # override the camera
  python follow.py --osc --windowed                 # also send OSC (projector in a window)
"""

import argparse
import time

import cv2
import numpy as np

from calib_utils import add_window_args, apply_h, draw_floor_grid, floor_circle, open_windows, show_preview
from cameras import add_camera_args, open_frames
from osc import OscSender, add_osc_args
from pipeline import FloorPipeline, add_pipeline_args
from recording import Recorder, add_record_args

PROJ_WIN = "projector"
SOURCE_COLOURS = {"ankle": (0, 255, 0), "box": (0, 165, 255), "dark": (255, 0, 255)}


def parse_args():
    p = argparse.ArgumentParser(description="project a circle under each tracked person or dark object")
    add_camera_args(p)
    add_pipeline_args(p)
    add_window_args(p)
    add_record_args(p)
    p.add_argument("--size", type=float, default=0.05,
                   help="circle radius as a fraction of the projected area's size, i.e. the square "
                        "root of its area on the floor (default: 0.05)")
    p.add_argument("--radius", type=float, default=None,
                   help="fixed circle radius in metres; overrides --size")
    p.add_argument("--background", type=int, default=None,
                   help="projector background brightness 0-255 (default: 90 with --dark-objects, else 0)")
    p.add_argument("--osc", action="store_true", help="also send the tracked positions as OSC")
    add_osc_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    pipe = FloorPipeline(args)
    floor_w, floor_h = pipe.floor_w, pipe.floor_h
    proj_w, proj_h = pipe.proj_w, pipe.proj_h
    radius = args.radius if args.radius is not None else args.size * pipe.proj_size
    print(f"Projected area is about {pipe.proj_size:.2f} m across; circle radius {radius * 100:.1f} cm")
    osc = OscSender(args.osc_host, args.osc_port) if args.osc else None
    if osc:
        print(f"Sending OSC to {args.osc_host}:{args.osc_port}")

    background = args.background if args.background is not None else (90 if args.dark_objects else 0)
    cam_win = f"camera {args.camera} ({pipe.source})"
    rec = Recorder(args, "follow")
    preview = open_windows(args, cam_win, pipe.cam["resolution"], PROJ_WIN, pipe.proj_pos, (proj_w, proj_h))

    show_grid = False  # g turns it on
    label = "objects" if args.dark_objects else "people"
    legend = "red = dark pixels, blue = search area" if args.dark_objects else "green = ankles, orange = box"
    fps = 0.0
    t_last = time.perf_counter()
    for frame in open_frames(args, pipe.source):
        now = time.perf_counter()
        dt, t_last = now - t_last, now
        if pipe.frames > 0:  # the first frame's dt includes camera start-up
            fps = 0.9 * fps + 0.1 / max(dt, 1e-6)

        detections, tracks, dark = pipe.process(frame, dt)
        if osc:
            osc.send_tracks(tracks)
        if dark is not None:
            frame[dark > 0] = (frame[dark > 0] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
            cv2.drawContours(frame, pipe.mask_outline, -1, (255, 128, 0), 1)

        proj_img = np.full((proj_h, proj_w, 3), background, np.uint8)
        if show_grid:
            draw_floor_grid(proj_img, pipe.H_floor_proj, floor_w, floor_h, colour=(0, 60, 0), thickness=2)
            draw_floor_grid(frame, pipe.H_floor_cam, floor_w, floor_h, colour=(0, 120, 0))

        # raw detections: small dots in the camera view, grey if outside the taped area (not tracked)
        for d in detections:
            colour = SOURCE_COLOURS[d.source] if d.inside else (128, 128, 128)
            cv2.circle(frame, (int(d.cam_xy[0]), int(d.cam_xy[1])), 3, colour, -1)

        # smoothed tracks: rings + IDs in the camera view, circles on the projector
        for t in tracks:
            x, y = np.round(apply_h(pipe.H_floor_cam, [t.pos])[0]).astype(int)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(frame, f"#{t.id} {t.pos[0]:.2f},{t.pos[1]:.2f}m", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.fillPoly(proj_img, [floor_circle(pipe.H_floor_proj, t.pos, radius)], (255, 255, 255), cv2.LINE_AA)

        cv2.putText(frame, f"{fps:4.1f} fps  {label}: {len(tracks)}  ({legend})",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        rec.record("camera", frame)
        rec.record("projector", proj_img)
        rec.overlay(frame)
        cv2.imshow(cam_win, frame)
        cv2.imshow(PROJ_WIN, proj_img)
        if preview:
            show_preview(proj_img)
        key = cv2.waitKey(1) & 0xFF
        rec.handle_key(key)
        if key in (ord("q"), 27):
            break
        if key == ord("g"):
            show_grid = not show_grid

    rec.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
