#!/usr/bin/env python3
"""
foot_calibrate.py - fine-tune where people's feet land, by standing on projected targets.

calibrate.py maps the floor itself exactly, but YOLO doesn't see feet on the
floor: it sees ankles, ~10 cm up and behind the middle of the foot. This adds a
small correction on top, from people standing on known spots.

The projector shows a grid of numbered targets, one lit at a time: a ring with
a cross. Stand with the cross between your feet. Once your detected position
holds still for --hold seconds, the target is captured and the next one lights
up, so one person can do the whole grid alone. A red dot on the floor shows
where you're currently detected (before correction).

After the last target, the projector shows each target (white ring), where you
were detected (red) and where the correction puts you (green), and the errors
are printed. The correction is saved to the camera in calibration.json, and
follow.py and track_osc.py use it. Recalibrating the camera with calibrate.py clears it.

Keys (in any window):
  SPACE  capture the current target now     n  skip it     b  back one target
  after the last one:  s = save   r = start again   q / Esc = quit without saving
  c / p / f  start/stop recording the camera window / projector / full screen (recording.py)

Examples:
  python foot_calibrate.py                  # 3x3 targets, camera from the calibration
  python foot_calibrate.py --targets 4x3    # more targets
  python foot_calibrate.py --hold 2.5       # hold still longer before capturing
"""

import argparse
import time

import cv2
import numpy as np

from calib_utils import (add_window_args, apply_h, correction_errors, fit_floor_correction, floor_circle,
                         h_from_json, h_to_json, load_calibration, open_windows, save_calibration,
                         show_preview, timestamp)
from cameras import add_camera_args, calibrated_camera, open_frames
from recording import Recorder, add_record_args
from models import add_device_args, load_model
from pipeline import PERSON, feet_in_camera

PROJ_WIN = "projector"
WHITE, GREY, RED, GREEN, YELLOW = (255, 255, 255), (90, 90, 90), (0, 0, 255), (0, 255, 0), (0, 255, 255)


def size_arg(s):
    cols, rows = s.lower().split("x")
    return int(cols), int(rows)


def parse_args():
    p = argparse.ArgumentParser(description="fine-tune foot positions by standing on projected targets")
    add_camera_args(p)
    p.add_argument("--calib", default="calibration.json", help="calibration file (default: calibration.json)")
    p.add_argument("--camera", default="cam0", help="camera name in the calibration (default: cam0)")
    p.add_argument("--projector", default="proj0", help="projector name in the calibration (default: proj0)")
    add_window_args(p)
    add_record_args(p)
    p.add_argument("--model", default="yolo26n-pose.pt", help="pose weights (default: yolo26n-pose.pt)")
    p.add_argument("--imgsz", type=int, default=320, help="inference size (default: 320)")
    p.add_argument("--conf", type=float, default=0.35, help="min person confidence (default: 0.35)")
    p.add_argument("--kpt-conf", type=float, default=0.5, help="min ankle confidence (default: 0.5)")
    add_device_args(p)
    p.add_argument("--targets", type=size_arg, default=(3, 3), help="target grid, columns x rows (default: 3x3)")
    p.add_argument("--inset", type=float, default=0.15,
                   help="keep targets this fraction of the taped area in from the tape (default: 0.15)")
    p.add_argument("--ring", type=float, default=0.2, help="target ring radius in metres (default: 0.2)")
    p.add_argument("--hold", type=float, default=1.5, help="seconds to hold still before capture (default: 1.5)")
    p.add_argument("--still", type=float, default=0.05,
                   help="how far in metres the detected position may wander while holding still (default: 0.05)")
    p.add_argument("--near", type=float, default=0.6,
                   help="only count a person within this many metres of the target (default: 0.6)")
    return p.parse_args()


def target_grid(floor_w, floor_h, cols, rows, inset):
    """Target positions in metres, in a snake order so you never walk far between them."""
    xs = np.linspace(inset, 1 - inset, cols) * floor_w if cols > 1 else [floor_w / 2]
    ys = np.linspace(inset, 1 - inset, rows) * floor_h if rows > 1 else [floor_h / 2]
    return [(float(x), float(y)) for r, y in enumerate(ys)
            for x in (xs if r % 2 == 0 else list(reversed(xs)))]


def put_text(img, text, y, colour=YELLOW, scale=0.55):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def floor_pt(H, p):
    return tuple(np.round(apply_h(H, [p])[0]).astype(int))


def draw_cross(img, H, centre, size, colour, thickness):
    x, y = centre
    for a, b in [((x - size, y), (x + size, y)), ((x, y - size), (x, y + size))]:
        cv2.line(img, floor_pt(H, a), floor_pt(H, b), colour, thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    cal = load_calibration(args.calib)
    try:
        cam = cal["cameras"][args.camera]
        proj = cal["projectors"][args.projector]
    except KeyError as e:
        raise SystemExit(f"{e.args[0]} not found in {args.calib}: run calibrate.py first")
    H_cam_floor = h_from_json(cam["H_cam_to_floor"])  # without any old foot correction
    H_floor_cam = np.linalg.inv(H_cam_floor)
    H_floor_proj = h_from_json(proj["H_floor_to_proj"])
    floor_w, floor_h = cal["floor"]["width_m"], cal["floor"]["height_m"]
    proj_w, proj_h = proj["resolution"]
    targets = target_grid(floor_w, floor_h, *args.targets, args.inset)
    print(f"{len(targets)} targets over the {floor_w:g} x {floor_h:g} m taped area")

    source = calibrated_camera(args, cam)

    model = load_model(args)
    cam_win = f"camera {args.camera} ({source})"
    rec = Recorder(args, "foot_calibrate")
    preview = open_windows(args, cam_win, cam["resolution"], PROJ_WIN, proj["position"], (proj_w, proj_h))

    measured = [None] * len(targets)  # detected floor position per target, None = not captured
    current, held = 0, []  # held: [(time, floor position)] while holding still
    result = None  # (C, model, errors) once every target is done

    for frame in open_frames(args, source):
        now = time.perf_counter()
        view = frame
        proj_img = np.zeros((proj_h, proj_w, 3), np.uint8)
        thick = max(2, proj_w // 400)

        if result is None:
            # where is the person nearest the current target?
            r = model.predict(frame, classes=[PERSON], conf=args.conf, imgsz=args.imgsz, device=args.device,
                              verbose=False)[0]
            ankles = [f[:2] for f in feet_in_camera(r, args.kpt_conf) if f[2] == "ankle"]
            target = np.array(targets[current])
            pos = None
            if ankles:
                floor = apply_h(H_cam_floor, ankles)
                dist = np.linalg.norm(floor - target, axis=1)
                nearest = floor[dist.argmin()]
                # must be closer to this target than to any other, so someone still
                # standing on the last target isn't captured for this one
                closest_target = np.argmin(np.linalg.norm(np.array(targets) - nearest, axis=1))
                if dist.min() <= args.near and closest_target == current:
                    pos = nearest
            for x, y in ankles:
                cv2.circle(view, (int(x), int(y)), 4, GREEN, -1)

            # hold still for --hold seconds, then capture the median position
            if pos is None or (held and np.linalg.norm(pos - np.median([p for _, p in held], axis=0)) > args.still):
                held = []
            if pos is not None:
                held.append((now, pos))
            progress = (now - held[0][0]) / args.hold if len(held) > 1 else 0.0
            capture = progress >= 1

            # projector: all targets dim, done ones ticked, the current one bright with a fill-up arc
            for i, t in enumerate(targets):
                done = measured[i] is not None
                colour = WHITE if i == current else (GREEN if done else GREY)
                cv2.polylines(proj_img, [floor_circle(H_floor_proj, t, args.ring)], True, colour,
                              thick * (2 if i == current else 1), cv2.LINE_AA)
                draw_cross(proj_img, H_floor_proj, t, args.ring * 0.4, colour, thick)
                cv2.putText(proj_img, str(i + 1), floor_pt(H_floor_proj, (t[0] + args.ring, t[1] + args.ring)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, thick, cv2.LINE_AA)
            if progress > 0:
                arc = floor_circle(H_floor_proj, target, args.ring * 1.15, n=60)[:max(2, int(60 * min(progress, 1)))]
                cv2.polylines(proj_img, [arc], False, GREEN, thick * 3, cv2.LINE_AA)
            if pos is not None:
                cv2.circle(proj_img, floor_pt(H_floor_proj, pos), thick * 4, RED, -1, cv2.LINE_AA)

            # camera view: targets and the detected position
            for i, t in enumerate(targets):
                cv2.circle(view, floor_pt(H_floor_cam, t), 6, WHITE if i == current else GREY, 2)
            if pos is not None:
                cv2.circle(view, floor_pt(H_floor_cam, pos), 6, RED, -1)
            n_done = sum(m is not None for m in measured)
            put_text(view, f"target {current + 1}/{len(targets)}  ({n_done} captured)   "
                           "stand with the cross between your feet", 22)
            put_text(view, "SPACE = capture now   n = skip   b = back   q = quit" if pos is not None
                     else "nobody detected near the target (ankles must be visible)", 46,
                     YELLOW if pos is not None else RED)
        else:
            C, fit_model, (before, after, loo) = result
            for t, m in zip(targets, measured):
                cv2.polylines(proj_img, [floor_circle(H_floor_proj, t, args.ring)], True, WHITE, thick, cv2.LINE_AA)
                draw_cross(proj_img, H_floor_proj, t, args.ring * 0.4, WHITE, thick)
                if m is not None:
                    cv2.circle(proj_img, floor_pt(H_floor_proj, m), thick * 4, RED, -1, cv2.LINE_AA)
                    cv2.circle(proj_img, floor_pt(H_floor_proj, apply_h(C, [m])[0]), thick * 4, GREEN, -1, cv2.LINE_AA)
            put_text(view, f"{fit_model} fit: error {before * 100:.1f} -> {after * 100:.1f} cm "
                           f"(leave-one-out {loo * 100:.1f} cm)", 22)
            put_text(view, "floor: white = targets, red = detected, green = corrected", 46)
            put_text(view, "s = save   r = start again   q = quit without saving", 70)

        rec.record("camera", view)
        rec.record("projector", proj_img)
        rec.overlay(view)
        cv2.imshow(cam_win, view)
        cv2.imshow(PROJ_WIN, proj_img)
        if preview:
            show_preview(proj_img)
        key = cv2.waitKey(1) & 0xFF
        rec.handle_key(key)

        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        if result is None:
            if key == ord(" ") and len(held) >= 3:
                capture = True
            step = 0
            if capture:
                measured[current] = np.median([p for _, p in held], axis=0)
                err = np.linalg.norm(measured[current] - targets[current]) * 100
                print(f"  target {current + 1}: detected {err:.1f} cm off")
                step = 1
            elif key == ord("n"):
                step = 1
            elif key == ord("b") and current > 0:
                step = -1
            if step:
                current, held = current + step, []
            if current == len(targets):
                pairs = [(m, t) for m, t in zip(measured, targets) if m is not None]
                if not pairs:
                    print("No targets captured; starting again.")
                    current = 0
                    continue
                m, t = (np.array(v) for v in zip(*pairs))
                C, fit_model = fit_floor_correction(m, t)
                result = (C, fit_model, correction_errors(m, t))
                before, after, loo = result[2]
                print(f"{fit_model} correction from {len(m)} targets: error {before * 100:.1f} cm -> "
                      f"{after * 100:.1f} cm (leave-one-out {loo * 100:.1f} cm)")
                if loo > before:
                    print("  Leave-one-out is worse than no correction: the captures disagree. "
                          "Try again, standing the same way on every target.")
        elif key == ord("r"):
            measured, current, held, result = [None] * len(targets), 0, [], None
        elif key == ord("s"):
            C, fit_model, (before, after, loo) = result
            cam["foot_correction"] = {
                "model": fit_model,
                "matrix": h_to_json(C),
                "points": sum(m is not None for m in measured),
                "rms_before_m": round(before, 4),
                "rms_after_m": round(after, 4),
                "rms_leave_one_out_m": round(loo, 4),
                "updated": timestamp(),
            }
            save_calibration(args.calib, cal)
            print(f"Saved the foot correction to camera '{args.camera}' in {args.calib}.")
            break

    rec.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
