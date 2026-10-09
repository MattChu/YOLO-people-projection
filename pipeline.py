"""
pipeline.py - camera frame -> tracked positions on the floor, shared by
follow.py (the visual test) and track_osc.py (the headless tracker).

Per frame, things are detected in the camera image, mapped camera -> floor
(metres), filtered to the taped rectangle and fed to a floor-space tracker.

Two detectors:
  people (default)  YOLO pose: the ankle midpoint, or the bottom of the box when
                    the ankles aren't visible, then nudged by the camera's foot
                    correction from foot_calibrate.py if it has one.
  --dark-objects    black objects, for table-top testing (no YOLO): the floor is
                    lit, and anything much darker than it inside the taped
                    rectangle counts. The search stops a little short of the
                    tape (--inset) so the tape marks aren't detected.
"""

import os

import cv2
import numpy as np

from calib_utils import apply_h, floor_polygon_mask, h_from_json, load_calibration, projector_footprint
from cameras import calibrated_camera
from models import add_device_args, load_model
from tracking import FloorTracker

PERSON = 0
L_ANKLE, R_ANKLE = 15, 16  # COCO keypoint indices


def add_pipeline_args(p):
    """Calibration, detection and tracking options, shared by follow.py and track_osc.py."""
    p.add_argument("--calib", default="calibration.json", help="calibration file (default: calibration.json)")
    p.add_argument("--camera", default="cam0", help="camera name in the calibration (default: cam0)")
    p.add_argument("--projector", default="proj0", help="projector name in the calibration (default: proj0)")
    p.add_argument("--model", default="yolo26n-pose.pt",
                   help="pose (or plain detection) weights (default: yolo26n-pose.pt)")
    p.add_argument("--imgsz", type=int, default=320, help="inference size (default: 320)")
    p.add_argument("--conf", type=float, default=0.35, help="min person confidence (default: 0.35)")
    p.add_argument("--kpt-conf", type=float, default=0.5, help="min ankle confidence (default: 0.5)")
    add_device_args(p)
    p.add_argument("--margin", type=float, default=0.0,
                   help="also accept detections up to this many metres outside the taped area (default: 0)")
    p.add_argument("--smooth", type=float, default=0.5,
                   help="position smoothing, 0 = none, closer to 1 = smoother but laggier (default: 0.5)")
    p.add_argument("--match-dist", type=float, default=None,
                   help="max jump in metres between frames for the same object "
                        "(default: 15%% of the projected area's size)")
    p.add_argument("--max-missed", type=int, default=10,
                   help="frames an object can go undetected before it's lost: about 0.7 s at 14 fps "
                        "(default: 10)")
    p.add_argument("--reattach-time", type=float, default=2.0,
                   help="seconds a lost object is remembered: found again in that time, it keeps "
                        "its id (default: 2)")
    p.add_argument("--reattach-dist", type=float, default=None,
                   help="how close, in metres, to where a lost object was last seen it must be "
                        "found again to keep its id (default: 25%% of the projected area's size)")
    p.add_argument("--no-foot-correction", action="store_true",
                   help="ignore the correction from foot_calibrate.py (to compare with and without)")
    p.add_argument("--dark-objects", action="store_true",
                   help="track black objects instead of people (no YOLO)")
    p.add_argument("--dark-ratio", type=float, default=0.5,
                   help="dark-objects: pixels darker than this fraction of the floor's brightness "
                        "count as object (default: 0.5)")
    p.add_argument("--min-area", type=int, default=40,
                   help="dark-objects: ignore blobs smaller than this many camera pixels (default: 40)")
    p.add_argument("--inset", type=float, default=None,
                   help="dark-objects: stay this many metres inside the tape, so the tape marks "
                        "aren't detected (default: 8%% of the taped area's shorter side)")


def feet_in_camera(result, kpt_conf):
    """[(x, y, source)] foot positions in camera pixels, one per person."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    kpts = kconf = None
    if result.keypoints is not None and result.keypoints.conf is not None:
        kpts = result.keypoints.xy.cpu().numpy()
        kconf = result.keypoints.conf.cpu().numpy()
    feet = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        if kpts is not None:
            ok = kconf[i, [L_ANKLE, R_ANKLE]] >= kpt_conf
            if ok.any():
                x, y = kpts[i, [L_ANKLE, R_ANKLE]][ok].mean(axis=0)
                feet.append((x, y, "ankle"))
                continue
        feet.append(((x1 + x2) / 2, y2, "box"))
    return feet


def dark_search_mask(H_floor_cam, H_floor_proj, floor_w, floor_h, inset, proj_size, cam_size):
    """Camera-image mask of where to look for dark objects: inside the tape (pulled
    in by `inset` metres) and inside the lit area (pulled in slightly, so the unlit
    floor just outside the projection isn't mistaken for an object)."""
    if inset * 2 >= min(floor_w, floor_h):
        return np.zeros(cam_size[::-1], np.uint8)
    taped = [(inset, inset), (floor_w - inset, inset), (floor_w - inset, floor_h - inset), (inset, floor_h - inset)]
    lit = projector_footprint(H_floor_proj, *proj_size, shrink=0.04)
    return cv2.bitwise_and(floor_polygon_mask(H_floor_cam, taped, *cam_size),
                           floor_polygon_mask(H_floor_cam, lit, *cam_size))


def dark_objects_in_camera(frame, mask, ratio, min_area):
    """([(x, y, "dark")] blob centres in camera pixels, binary dark-pixel image)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    floor_level = np.median(gray[mask > 0])  # objects are small, so the median is the floor
    dark = ((gray < floor_level * ratio) & (mask > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, stats, centres = cv2.connectedComponentsWithStats(dark)
    objects = [(cx, cy, "dark") for (cx, cy), area in zip(centres[1:], stats[1:, cv2.CC_STAT_AREA])
               if area >= min_area]
    return objects, dark


def projected_size_m(H_floor_proj, proj_w, proj_h):
    """Square root of the projector's footprint area on the floor, in metres."""
    footprint = projector_footprint(H_floor_proj, proj_w, proj_h).astype(np.float32)
    return float(np.sqrt(cv2.contourArea(footprint)))


class Detection:
    def __init__(self, cam_xy, source, floor_xy, inside):
        self.cam_xy = cam_xy      # camera pixels
        self.source = source      # "ankle", "box" or "dark"
        self.floor_xy = floor_xy  # metres
        self.inside = inside      # within the taped area (plus --margin), so tracked


class FloorPipeline:
    """Calibration, detector and tracker for one camera. Call process() per frame."""

    def __init__(self, args):
        self.args = args
        if not os.path.exists(args.calib):
            raise SystemExit(f"No calibration file {args.calib}: run calibrate.py first")
        cal = load_calibration(args.calib)
        try:
            self.cam = cal["cameras"][args.camera]
            proj = cal["projectors"][args.projector]
        except KeyError as e:
            raise SystemExit(f"{e.args[0]} not found in {args.calib}: run calibrate.py first")
        self.H_cam_floor = h_from_json(self.cam["H_cam_to_floor"])
        self.H_floor_cam = np.linalg.inv(self.H_cam_floor)  # true floor geometry, for drawing
        self.H_floor_proj = h_from_json(proj["H_floor_to_proj"])
        self.floor_w, self.floor_h = cal["floor"]["width_m"], cal["floor"]["height_m"]
        self.proj_pos = proj["position"]
        self.proj_w, self.proj_h = proj["resolution"]
        self.proj_size = projected_size_m(self.H_floor_proj, self.proj_w, self.proj_h)

        self.source = calibrated_camera(args, self.cam)
        print(f"Camera: {self.source}")

        if args.dark_objects:
            inset = args.inset if args.inset is not None else 0.08 * min(self.floor_w, self.floor_h)
            self.mask = dark_search_mask(self.H_floor_cam, self.H_floor_proj, self.floor_w, self.floor_h, inset,
                                         (self.proj_w, self.proj_h), tuple(self.cam["resolution"]))
            if cv2.countNonZero(self.mask) < args.min_area:
                raise SystemExit(f"Nowhere to search: the taped area, {inset * 100:.1f} cm in from the tape, "
                                 "doesn't overlap the projected area. Try a smaller --inset or recalibrate.")
            self.mask_outline, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Searching for dark objects {inset * 100:.1f} cm inside the tape")
        else:
            self.model = load_model(args)  # slow, only needed for people
            # foot_calibrate.py's correction for where the detected feet land on the floor
            foot = self.cam.get("foot_correction")
            if foot and not args.no_foot_correction:
                self.H_cam_floor = h_from_json(foot["matrix"]) @ self.H_cam_floor
                print(f"Foot correction on ({foot['model']}, {foot['points']} points, "
                      f"error {foot['rms_before_m'] * 100:.1f} -> {foot['rms_after_m'] * 100:.1f} cm)")
            elif not foot:
                print("No foot correction yet: run foot_calibrate.py to line positions up with feet")

        self.tracker = FloorTracker(
            smooth=args.smooth, max_missed=args.max_missed, reattach_time=args.reattach_time,
            match_dist=args.match_dist if args.match_dist is not None else 0.15 * self.proj_size,
            reattach_dist=args.reattach_dist if args.reattach_dist is not None else 0.25 * self.proj_size)
        self.frames = 0

    def process(self, frame, dt):
        """Detect and track in one frame. Returns (detections, confirmed tracks,
        dark-pixel image or None)."""
        self.frames += 1
        if self.frames == 1 and list(frame.shape[1::-1]) != self.cam["resolution"]:
            raise SystemExit(f"Camera gives {frame.shape[1]}x{frame.shape[0]} but '{self.args.camera}' was "
                             f"calibrated at {self.cam['resolution'][0]}x{self.cam['resolution'][1]}: "
                             "use matching --width/--height, or recalibrate at this size")
        dark = None
        if self.args.dark_objects:
            feet, dark = dark_objects_in_camera(frame, self.mask, self.args.dark_ratio, self.args.min_area)
        else:
            result = self.model.predict(frame, classes=[PERSON], conf=self.args.conf, imgsz=self.args.imgsz,
                                        device=self.args.device, verbose=False)[0]
            feet = feet_in_camera(result, self.args.kpt_conf)

        m = self.args.margin
        detections = []
        for (x, y, src), (fx, fy) in zip(feet, apply_h(self.H_cam_floor, [f[:2] for f in feet]) if feet else []):
            inside = -m <= fx <= self.floor_w + m and -m <= fy <= self.floor_h + m
            detections.append(Detection((x, y), src, (fx, fy), inside))
        tracks = self.tracker.update([d.floor_xy for d in detections if d.inside], dt)
        return detections, tracks, dark
