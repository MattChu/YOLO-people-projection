"""
calib_utils.py - shared calibration and floor-geometry helpers for calibrate.py and follow.py.

Three coordinate systems, all linked through the floor:
  camera     pixels in a camera image
  floor      metres on the floor; origin at the first taped corner
  projector  pixels in a projector's output image

calibration.json holds one camera->floor homography per camera and one
floor->projector homography per projector, so cameras and projectors can be
added or recalibrated independently.
"""

import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50  # big cells, easy to read when projected
EMPTY_CALIBRATION = {"version": 1, "cameras": {}, "projectors": {}}


# --- projected ArUco pattern ------------------------------------------------

def make_marker_pattern(proj_w, proj_h, cols=8, rows=5, margin=0.05):
    """White image with a grid of ArUco markers.

    Returns (image, {marker_id: 4x2 corner array in projector pixels}).
    White background gives every marker the quiet zone it needs, and lights the
    floor so the camera's auto exposure has something to work with.
    """
    if cols * rows > 50:
        raise ValueError("at most 50 markers in DICT_4X4_50")
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img = np.full((proj_h, proj_w), 255, np.uint8)
    mx, my = proj_w * margin, proj_h * margin
    cell_w = (proj_w - 2 * mx) / cols
    cell_h = (proj_h - 2 * my) / rows
    size = int(min(cell_w, cell_h) * 0.7)
    corners = {}
    for r in range(rows):
        for c in range(cols):
            mid = r * cols + c
            x0 = int(mx + (c + 0.5) * cell_w - size / 2)
            y0 = int(my + (r + 0.5) * cell_h - size / 2)
            img[y0:y0 + size, x0:x0 + size] = cv2.aruco.generateImageMarker(dictionary, mid, size)
            # the detector reports corners on pixel edges, half a pixel outside the pixel centres
            a, b = x0 - 0.5, x0 + size - 0.5
            t, u = y0 - 0.5, y0 + size - 0.5
            corners[mid] = np.array([[a, t], [b, t], [b, u], [a, u]], np.float64)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), corners


def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)


def detect_markers(detector, frame):
    """{marker_id: 4x2 corners in camera pixels} for the markers visible in frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def combine_detections(detections, min_fraction=0.5):
    """Median corner positions over several frames, keeping markers seen in at
    least min_fraction of them. Averages out sensor noise and drops flickers."""
    by_id = defaultdict(list)
    for d in detections:
        for mid, c in d.items():
            by_id[mid].append(c)
    need = max(1, int(len(detections) * min_fraction))
    return {mid: np.median(cs, axis=0) for mid, cs in by_id.items() if len(cs) >= need}


def marker_pairs(cam_markers, proj_corners):
    """Matched (camera_points, projector_points) arrays from marker corners."""
    ids = sorted(set(cam_markers) & set(proj_corners))
    if not ids:
        return np.empty((0, 2)), np.empty((0, 2))
    return (np.concatenate([cam_markers[i] for i in ids]),
            np.concatenate([proj_corners[i] for i in ids]))


# --- homography maths -------------------------------------------------------

def apply_h(H, pts):
    """Map Nx2 points through a 3x3 homography."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 2)


def fit_homography(src, dst, ransac_px=3.0):
    """Fit src->dst. Returns (H, rms error in dst units, inlier count)."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 4:
        raise ValueError(f"need at least 4 point pairs, got {len(src)}")
    if len(src) == 4:
        H, mask = cv2.findHomography(src, dst, 0)
    else:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H is None:
        raise ValueError("homography fit failed (points collinear?)")
    inliers = mask.ravel().astype(bool)
    err = np.linalg.norm(apply_h(H, src[inliers]) - dst[inliers], axis=1)
    return H, float(np.sqrt(np.mean(err ** 2))), int(inliers.sum())


def fit_camera_to_projector(cam_pts, proj_pts, ransac_cam_px=2.0):
    """Camera->projector fit with the outlier threshold given in camera pixels,
    scaled by how many projector pixels one camera pixel covers."""
    scale = np.asarray(proj_pts).std(axis=0).mean() / np.asarray(cam_pts).std(axis=0).mean()
    return fit_homography(cam_pts, proj_pts, ransac_cam_px * scale)


def normalise(H):
    return H / H[2, 2]


# --- foot correction --------------------------------------------------------

def fit_floor_correction(measured, true):
    """3x3 matrix mapping measured floor positions (metres) onto the true ones.

    Affine with 4+ point pairs, otherwise a plain offset. Affine suits foot
    detection: ankles are ~10 cm up, so the camera sees them pushed away from
    the point below it by an amount growing with distance - a scaling about
    that point - plus any constant shift (ankles sit behind the feet's centre).
    Returns (C, model name)."""
    measured = np.asarray(measured, np.float64).reshape(-1, 2)
    true = np.asarray(true, np.float64).reshape(-1, 2)
    if len(measured) == 0:
        raise ValueError("no point pairs")
    C = np.eye(3)
    if len(measured) >= 4:
        A = np.hstack([measured, np.ones((len(measured), 1))])
        sol, *_ = np.linalg.lstsq(A, true, rcond=None)
        C[:2] = sol.T
        return C, "affine"
    C[:2, 2] = (true - measured).mean(axis=0)
    return C, "offset"


def correction_errors(measured, true):
    """(rms before, rms after fitting, rms leave-one-out) in metres. Leave-one-out
    predicts each point from a fit to the others: an honest check that the fit
    isn't just bending itself around a handful of points."""
    measured = np.asarray(measured, np.float64).reshape(-1, 2)
    true = np.asarray(true, np.float64).reshape(-1, 2)
    rms = lambda d: float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
    C, _ = fit_floor_correction(measured, true)
    loo = []
    for i in range(len(measured)):
        keep = np.arange(len(measured)) != i
        if keep.any():
            Ci, _ = fit_floor_correction(measured[keep], true[keep])
            loo.append(apply_h(Ci, measured[i:i + 1])[0] - true[i])
    return (rms(measured - true), rms(apply_h(C, measured) - true),
            rms(np.array(loo)) if loo else float("nan"))


# --- floor geometry ---------------------------------------------------------

def projector_footprint(H_floor_proj, proj_w, proj_h, shrink=0.0):
    """Corners (4x2, metres) of the floor area the projector lights. `shrink`
    pulls each edge in by that fraction of the image size."""
    corners = np.array([(0, 0), (proj_w, 0), (proj_w, proj_h), (0, proj_h)], np.float64)
    centre = corners.mean(axis=0)
    corners = centre + (corners - centre) * (1 - 2 * shrink)
    return apply_h(np.linalg.inv(H_floor_proj), corners)


def floor_polygon_mask(H_floor_cam, polygon, cam_w, cam_h):
    """Camera-image mask (255 inside) of a convex polygon given in floor metres."""
    mask = np.zeros((cam_h, cam_w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(apply_h(H_floor_cam, polygon)).astype(np.int32), 255)
    return mask


def grid_step(floor_w, floor_h):
    """A round grid spacing giving a few divisions across the shorter side:
    5-10 cm for a table top, 0.5 m for a room."""
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0):
        if min(floor_w, floor_h) / step <= 6:
            return step
    return 10.0


# --- calibration file -------------------------------------------------------

def load_calibration(path):
    if not os.path.exists(path):
        return json.loads(json.dumps(EMPTY_CALIBRATION))
    with open(path) as f:
        return json.load(f)


def save_calibration(path, cal):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cal, f, indent=2)
    os.replace(tmp, path)  # never leave a half-written file behind


def h_to_json(H):
    return [[float(v) for v in row] for row in normalise(np.asarray(H, np.float64))]


def h_from_json(rows):
    return np.array(rows, np.float64)


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --- drawing ----------------------------------------------------------------

PREVIEW_WIN = "projector preview"


def add_window_args(p):
    """Add the --windowed / --window-scale / --projector-scale options to a parser."""
    p.add_argument("--windowed", action="store_true",
                   help="projector image in a normal window instead of full screen on the "
                        "projector (testing on one screen)")
    p.add_argument("--window-scale", type=float, default=2.0,
                   help="starting camera window size as a multiple of the camera image; "
                        "drag the corner to resize (default: 2)")
    p.add_argument("--projector-scale", type=float, default=0.5,
                   help="starting size of the projector preview (or the --windowed projector "
                        "window) as a multiple of the projector resolution (default: 0.5)")


def resizable_window(name, size, scale, pos):
    """Window that scales its image (and everything drawn on it) to fit, keeping
    the shape. Mouse clicks still come back in image pixels."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, int(size[0] * scale), int(size[1] * scale))
    cv2.moveWindow(name, *pos)


def open_projector_window(name, pos, size):
    """Borderless full-screen window on the projector."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, pos[0], pos[1])
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def open_windows(args, cam_win, cam_size, proj_win, proj_pos, proj_size):
    """The camera window at the top left of the laptop screen, and beside it a
    preview of what's projected. The projector window goes full screen on the
    projector, or with --windowed sits where the preview would (so no preview).
    Returns whether there's a preview to update with show_preview()."""
    resizable_window(cam_win, cam_size, args.window_scale, (0, 0))
    beside = (int(cam_size[0] * args.window_scale) + 10, 0)
    if args.windowed:
        resizable_window(proj_win, proj_size, args.projector_scale, beside)
        return False
    open_projector_window(proj_win, proj_pos, proj_size)
    resizable_window(PREVIEW_WIN, proj_size, args.projector_scale, beside)
    return True


def show_preview(proj_img):
    cv2.imshow(PREVIEW_WIN, proj_img)


def draw_floor_grid(img, H_floor_to_img, floor_w, floor_h, step=None, colour=(0, 255, 0), thickness=1):
    """Floor-space grid lines (every `step` metres, default from grid_step) drawn
    into img. Straight lines stay straight under a homography, so mapping
    endpoints is enough."""
    step = step or grid_step(floor_w, floor_h)
    def line(p, q, t):
        a, b = apply_h(H_floor_to_img, [p, q])
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, t, cv2.LINE_AA)

    for x in np.arange(0, floor_w + 1e-6, step):
        line((x, 0), (x, floor_h), thickness)
    for y in np.arange(0, floor_h + 1e-6, step):
        line((0, y), (floor_w, y), thickness)
    corners = [(0, 0), (floor_w, 0), (floor_w, floor_h), (0, floor_h)]
    for p, q in zip(corners, corners[1:] + corners[:1]):
        line(p, q, thickness * 3)
    origin = np.round(apply_h(H_floor_to_img, [(0, 0)])[0]).astype(int)
    cv2.circle(img, tuple(origin), 6 * thickness, colour, -1, cv2.LINE_AA)


def floor_circle(H_floor_to_img, centre, radius_m, n=40):
    """Polygon for a circle on the floor, as it should appear in img. Under the
    projector's perspective it becomes an ellipse, so it lands round on the floor."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.asarray(centre) + radius_m * np.stack([np.cos(a), np.sin(a)], axis=1)
    return np.round(apply_h(H_floor_to_img, pts)).astype(np.int32)
