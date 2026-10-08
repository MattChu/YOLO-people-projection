#!/usr/bin/env python3
"""
test_calibration.py - check the calibration maths without a camera or projector.

Builds a fake scene with known geometry: a 4x3 m taped floor, a keystoned
projector, and a camera looking down at an angle. Renders what the camera would
see of the projected markers, runs the same steps as calibrate.py, and checks
the recovered mappings against the truth.

  python test_calibration.py
"""

import cv2
import numpy as np

from calib_utils import (apply_h, combine_detections, detect_markers, fit_camera_to_projector,
                         fit_homography, make_detector, make_marker_pattern, marker_pairs)

FLOOR_W, FLOOR_H = 4.0, 3.0
PROJ_W, PROJ_H = 1920, 1080
CAM_W, CAM_H = 640, 480
FLOOR_PTS = np.array([(0, 0), (FLOOR_W, 0), (FLOOR_W, FLOOR_H), (0, FLOOR_H)], np.float32)


def h(src, dst):
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)).astype(np.float64)


def fake_camera_view(pattern, H_proj_cam, rng):
    """Camera image of the projected pattern: perspective, blur, noise, dim floor."""
    lit = cv2.warpPerspective(pattern, H_proj_cam, (CAM_W, CAM_H), flags=cv2.INTER_AREA)
    img = 30 + lit.astype(np.float64) * 0.75  # unlit floor isn't black, projector isn't full white
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(1)

    # Truth: the projected image lands as a keystoned quad slightly bigger than the tape,
    # and the camera sees the tape as a trapezoid (looking down from one side).
    H_floor_proj_true = h([(-0.4, -0.3), (4.3, -0.2), (4.6, 3.4), (-0.5, 3.2)],
                          [(0, 0), (PROJ_W, 0), (PROJ_W, PROJ_H), (0, PROJ_H)])
    H_floor_cam_true = h(FLOOR_PTS, [(170, 110), (480, 115), (615, 440), (25, 430)])
    H_proj_cam_true = H_floor_cam_true @ np.linalg.inv(H_floor_proj_true)

    # Step 1: markers -> camera<->projector
    pattern, proj_corners = make_marker_pattern(PROJ_W, PROJ_H)
    detector = make_detector()
    detections = [detect_markers(detector, fake_camera_view(pattern, H_proj_cam_true, rng))
                  for _ in range(10)]
    markers = combine_detections(detections)
    cam_pts, proj_pts = marker_pairs(markers, proj_corners)
    H_cam_proj, rms, inliers = fit_camera_to_projector(cam_pts, proj_pts)
    print(f"markers found: {len(markers)}/{len(proj_corners)} (some are outside the camera view)")
    print(f"camera->projector fit: {inliers}/{len(cam_pts)} corners, rms {rms:.2f} projector px")

    # Step 2: clicks on the tape marks: up to a pixel off, then rounded like a real mouse click
    clicks = np.round(apply_h(H_floor_cam_true, FLOOR_PTS) + rng.uniform(-1, 1, (4, 2)))
    H_cam_floor, _, _ = fit_homography(clicks, FLOOR_PTS)
    H_floor_proj = H_cam_proj @ np.linalg.inv(H_cam_floor)

    # Check on a grid of floor points
    gx, gy = np.meshgrid(np.linspace(0, FLOOR_W, 9), np.linspace(0, FLOOR_H, 7))
    floor_grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    cam_grid = apply_h(H_floor_cam_true, floor_grid)

    floor_err_cm = np.linalg.norm(apply_h(H_cam_floor, cam_grid) - floor_grid, axis=1) * 100
    proj_err_px = np.linalg.norm(apply_h(H_floor_proj, floor_grid) - apply_h(H_floor_proj_true, floor_grid), axis=1)
    px_per_cm = PROJ_W / (4.9 * 100)
    print(f"camera->floor error:     max {floor_err_cm.max():.2f} cm, mean {floor_err_cm.mean():.2f} cm")
    print(f"floor->projector error:  max {proj_err_px.max():.2f} px (~{proj_err_px.max() / px_per_cm:.2f} cm on the floor)")

    assert len(markers) >= 10, "too few markers detected"
    assert rms < 3.0, "marker fit too noisy"  # about 1 camera pixel here
    assert floor_err_cm.max() < 3.0, "camera->floor too inaccurate"
    assert proj_err_px.max() / px_per_cm < 3.0, "floor->projector too inaccurate"
    print("OK")


if __name__ == "__main__":
    main()
