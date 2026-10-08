#!/usr/bin/env python3
"""
calibrate.py - map a camera and a projector onto floor coordinates (metres).

Put tape marks at the 4 corners of a measured rectangle on the floor, inside
the projected area and visible to the camera. Then (keys work in the camera window):

 1. MARKERS  The projector shows a grid of ArUco markers. With the floor
             clear, press SPACE. The markers are averaged over a few frames
             to give the camera -> projector mapping.
 2. FLOOR    The projector shows plain white. Click the 4 tape marks in the
             camera view, in the order shown (u = undo). This is skipped when
             adding a new camera to an already calibrated projector (the
             projector's floor mapping is reused). Use --reclick to force it.
 3. VERIFY   The projector draws a grid in floor coordinates (spacing scaled to
             the taped area). Its outline should sit on the tape.
             s = save, r = start again, q = quit.

In a dark room, auto exposure brightens the picture for the room and can wash
out the bright markers. If few markers are found, fix the camera's exposure
with --exposure (in the camera's own units: on Linux usually tenths of a ms, so try
80, then 40 or 160; on Windows a power of 2 in seconds, so try -6, then -7 or -5).

In any step, c / p / f start/stop recording the camera window / projector /
full screen (see recording.py; files go in recordings/).

Floor coordinates: the first tape mark is (0, 0), the second is (W, 0) along
the width, the third (W, H), the fourth (0, H).

Examples:
  python calibrate.py --source 0 --proj-pos 1920,0 --proj-size 1920x1080 --floor-size 4x3
  python calibrate.py --source 0 --exposure 80 ...   # fixed exposure, if markers are washed out
  python calibrate.py --source 1 --camera cam1       # add a 2nd camera, same projector
  python calibrate.py --source 0 --windowed          # try it on a single screen
  (python cameras.py lists the cameras' numbers)
"""

import argparse

import cv2
import numpy as np

from calib_utils import (add_window_args, apply_h, combine_detections, detect_markers, draw_floor_grid,
                         fit_camera_to_projector, fit_homography, h_from_json, h_to_json,
                         load_calibration, make_detector, make_marker_pattern, marker_pairs,
                         open_windows, save_calibration, show_preview, timestamp)
from cameras import add_camera_args, open_frames, source_name
from recording import Recorder, add_record_args

CAM_WIN = "camera (calibration)"
PROJ_WIN = "projector"


def size_arg(s):
    w, h = s.lower().split("x")
    return float(w), float(h)


def pos_arg(s):
    x, y = s.split(",")
    return int(x), int(y)


def parse_args():
    p = argparse.ArgumentParser(description="camera / projector / floor calibration")
    add_camera_args(p)
    p.add_argument("--calib", default="calibration.json", help="calibration file (default: calibration.json)")
    p.add_argument("--camera", default="cam0", help="name for this camera (default: cam0)")
    p.add_argument("--projector", default="proj0", help="name for this projector (default: proj0)")
    p.add_argument("--proj-size", type=size_arg, default=None,
                   help="projector resolution, e.g. 1920x1080 (default: saved value, else 1920x1080)")
    p.add_argument("--proj-pos", type=pos_arg, default=None,
                   help="projector's top-left on the desktop, e.g. 1920,0 (default: saved value, else 0,0)")
    add_window_args(p)
    add_record_args(p)
    p.add_argument("--floor-size", type=size_arg, default=None,
                   help="taped rectangle in metres, e.g. 4x3 (default: saved value)")
    p.add_argument("--reclick", action="store_true",
                   help="click the tape marks even when adding a new camera to a calibrated projector")
    p.add_argument("--grid", type=size_arg, default=(8, 5), help="marker columns x rows (default: 8x5)")
    p.add_argument("--frames", type=int, default=15, help="frames to average per capture (default: 15)")
    return p.parse_args()


def put_text(img, text, y, colour=(0, 255, 255), scale=0.55):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def main():
    args = parse_args()
    cal = load_calibration(args.calib)
    known_proj = cal["projectors"].get(args.projector)
    # Reuse the projector's floor mapping only when adding a new camera. Recalibrating
    # a known camera means something moved, so the tape marks get clicked again.
    new_camera = args.camera not in cal["cameras"]
    reuse_floor = known_proj is not None and new_camera and not args.reclick

    # Projector size and position: an existing calibration wins, because the
    # saved floor->projector mapping is only valid for that size.
    if reuse_floor:
        proj_w, proj_h = known_proj["resolution"]
        proj_pos = known_proj["position"]
        floor_w, floor_h = cal["floor"]["width_m"], cal["floor"]["height_m"]
        print(f"Projector '{args.projector}' is already calibrated: reusing its floor mapping "
              f"({floor_w:g} x {floor_h:g} m). Use --reclick to redo it.")
    else:
        saved = known_proj or {}
        proj_w, proj_h = (int(v) for v in (args.proj_size or saved.get("resolution", (1920, 1080))))
        proj_pos = args.proj_pos or tuple(saved.get("position", (0, 0)))
        print(f"Projector '{args.projector}': {proj_w}x{proj_h} at {proj_pos[0]},{proj_pos[1]} on the desktop")
        if args.floor_size:
            floor_w, floor_h = args.floor_size
        elif "floor" in cal:
            floor_w, floor_h = cal["floor"]["width_m"], cal["floor"]["height_m"]
        else:
            raise SystemExit("Give the taped rectangle size, e.g. --floor-size 4x3 (metres)")
        old = cal.get("floor")
        if old and (old["width_m"], old["height_m"]) != (floor_w, floor_h) and cal["cameras"]:
            print("Warning: floor size differs from the saved one; cameras calibrated "
                  "against the old tape marks will no longer agree.")

    floor_pts = [(0, 0), (floor_w, 0), (floor_w, floor_h), (0, floor_h)]
    cols, rows = (int(v) for v in args.grid)
    pattern, proj_corners = make_marker_pattern(proj_w, proj_h, cols, rows)
    white = np.full((proj_h, proj_w, 3), 255, np.uint8)
    detector = make_detector()

    rec = Recorder(args, "calibrate")
    # (the camera window starts at this size, whatever the camera gives; it scales its picture to fit)
    preview = open_windows(args, CAM_WIN, (args.width or 640, args.height or 480), PROJ_WIN, proj_pos, (proj_w, proj_h))

    s = {"state": "markers", "clicks": []}
    captured, capturing = [], False
    H_cam_proj = H_cam_floor = H_floor_proj = None
    marker_info = {}
    message = ""

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and s["state"] == "floor" and len(s["clicks"]) < 4:
            s["clicks"].append((x, y))

    cv2.setMouseCallback(CAM_WIN, on_mouse)

    for frame in open_frames(args):
        view = frame.copy()
        state = s["state"]

        if state == "markers":
            proj_img = pattern
            found = detect_markers(detector, frame)
            if found:
                cv2.aruco.drawDetectedMarkers(
                    view, [c.reshape(1, 4, 2).astype(np.float32) for c in found.values()],
                    np.array(list(found), np.int32).reshape(-1, 1))
            if capturing:
                captured.append(found)
                put_text(view, f"capturing {len(captured)}/{args.frames}...", 22)
                if len(captured) >= args.frames:
                    capturing = False
                    markers = combine_detections(captured)
                    captured = []
                    if len(markers) < 4:
                        message = (f"only {len(markers)} markers found, need 4+. Dim the room, or fix "
                                   f"the exposure (--exposure, see --help), then SPACE")
                    else:
                        cam_pts, proj_pts = marker_pairs(markers, proj_corners)
                        try:
                            H_cam_proj, rms, inliers = fit_camera_to_projector(cam_pts, proj_pts)
                        except ValueError as e:
                            message = f"marker fit failed ({e}). Check the view, then SPACE"
                            continue
                        marker_info = {"markers_used": len(markers), "marker_rms_px": round(rms, 3)}
                        print(f"camera->projector: {len(markers)} markers, {inliers}/{len(cam_pts)} "
                              f"corners kept, rms error {rms:.2f} projector px")
                        if rms > 3:
                            print("  That's high: is the floor flat, and was anything moving?")
                        message = ""
                        if reuse_floor:
                            H_floor_proj = h_from_json(known_proj["H_floor_to_proj"])
                            H_cam_floor = np.linalg.inv(H_floor_proj) @ H_cam_proj
                            s["state"] = "verify"
                        else:
                            s["state"] = "floor"
            else:
                put_text(view, f"{len(found)}/{cols * rows} markers seen   SPACE = capture   q = quit", 22)
            if message:
                put_text(view, message, 46, (0, 0, 255))

        elif state == "floor":
            proj_img = white
            clicks = s["clicks"]
            for i, (x, y) in enumerate(clicks):
                cv2.circle(view, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(view, str(i + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if len(clicks) > 1:
                cv2.polylines(view, [np.array(clicks, np.int32)], len(clicks) == 4, (0, 0, 255), 1)
            if len(clicks) < 4:
                fx, fy = floor_pts[len(clicks)]
                put_text(view, f"click tape mark {len(clicks) + 1}/4 at ({fx:g}, {fy:g}) m   u = undo", 22)
            else:
                try:
                    H_cam_floor, _, _ = fit_homography(clicks, floor_pts)
                    H_floor_proj = H_cam_proj @ np.linalg.inv(H_cam_floor)
                    s["state"], message = "verify", ""
                except (ValueError, np.linalg.LinAlgError):
                    s["clicks"] = []
                    message = "those 4 clicks don't form a usable rectangle; click the tape marks again"
            if message:
                put_text(view, message, 46, (0, 0, 255))

        else:  # verify
            proj_img = np.zeros((proj_h, proj_w, 3), np.uint8)
            draw_floor_grid(proj_img, H_floor_proj, floor_w, floor_h, colour=(0, 255, 0), thickness=3)
            draw_floor_grid(view, np.linalg.inv(H_cam_floor), floor_w, floor_h, colour=(0, 255, 0))
            put_text(view, "projected outline should sit on the tape", 22)
            put_text(view, "s = save   r = redo   q = quit without saving", 46)

        rec.record("camera", view)
        rec.record("projector", proj_img)
        rec.overlay(view)
        cv2.imshow(CAM_WIN, view)
        cv2.imshow(PROJ_WIN, proj_img)
        if preview:
            show_preview(proj_img)
        key = cv2.waitKey(1) & 0xFF
        rec.handle_key(key)

        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        if key == ord(" ") and s["state"] == "markers" and not capturing:
            capturing, captured, message = True, [], ""
        elif key in (ord("u"), 8) and s["state"] == "floor" and s["clicks"]:
            s["clicks"].pop()
        elif key == ord("r"):
            s["state"], s["clicks"], capturing, message = "markers", [], False, ""
        elif key == ord("s") and s["state"] == "verify":
            cam_h, cam_w = frame.shape[:2]
            cal["floor"] = {"width_m": floor_w, "height_m": floor_h}
            cal["cameras"][args.camera] = {
                "source": source_name(args),
                "resolution": [cam_w, cam_h],
                **({"fps": args.fps} if args.fps else {}),
                **({"fourcc": args.fourcc} if args.fourcc else {}),
                "H_cam_to_floor": h_to_json(H_cam_floor),
                **marker_info,
                "updated": timestamp(),
            }
            if not reuse_floor:
                cal["projectors"][args.projector] = {
                    "resolution": [proj_w, proj_h],
                    "position": list(proj_pos),
                    "H_floor_to_proj": h_to_json(H_floor_proj),
                    "updated": timestamp(),
                }
            save_calibration(args.calib, cal)
            centre = apply_h(H_cam_floor, [(cam_w / 2, cam_h / 2)])[0]
            print(f"Saved {args.calib}: camera '{args.camera}' ({cam_w}x{cam_h}), "
                  f"projector '{args.projector}'. Camera image centre is at "
                  f"({centre[0]:.2f}, {centre[1]:.2f}) m on the floor.")
            break

    rec.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
