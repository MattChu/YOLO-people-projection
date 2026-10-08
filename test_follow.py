#!/usr/bin/env python3
"""
test_follow.py - check the dark-object detection (pipeline.py) and tracking without
a camera or projector.

Builds a fake table-top scene with known geometry (a 44.5 x 33.5 cm taped area,
a projector lighting a larger area, a camera looking down at an angle), renders
what the camera would see, and checks what pipeline.py and tracking.py make of it.

  python test_follow.py
"""

import cv2
import numpy as np

from calib_utils import apply_h, grid_step
from pipeline import dark_objects_in_camera, dark_search_mask, projected_size_m
from tracking import FloorTracker

FLOOR_W, FLOOR_H = 0.445, 0.335
PROJ_W, PROJ_H = 1024, 768
CAM_W, CAM_H = 640, 480
FOOTPRINT = [(-0.15, -0.10), (0.60, -0.10), (0.60, 0.45), (-0.15, 0.45)]  # lit area, metres


def h(src, dst):
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)).astype(np.float64)


H_FLOOR_PROJ = h(FOOTPRINT, [(0, 0), (PROJ_W, 0), (PROJ_W, PROJ_H), (0, PROJ_H)])
H_FLOOR_CAM = h([(0, 0), (FLOOR_W, 0), (FLOOR_W, FLOOR_H), (0, FLOOR_H)],
                [(180, 120), (470, 125), (560, 400), (90, 395)])


def cam_pt(fx, fy):
    return tuple(np.round(apply_h(H_FLOOR_CAM, [(fx, fy)])[0]).astype(int))


def fake_frame(objects, rng):
    """Unlit surroundings, lit footprint, black tape crosses at the corners, black objects."""
    frame = np.full((CAM_H, CAM_W, 3), 25, np.uint8)
    cv2.fillConvexPoly(frame, np.int32([cam_pt(*p) for p in FOOTPRINT]), (150, 150, 150))
    for fx, fy in [(0, 0), (FLOOR_W, 0), (FLOOR_W, FLOOR_H), (0, FLOOR_H)]:
        for dx, dy in [(0.02, 0), (0, 0.02)]:
            cv2.line(frame, cam_pt(fx - dx, fy - dy), cam_pt(fx + dx, fy + dy), (15, 15, 15), 4)
    for fx, fy in objects:
        cv2.circle(frame, cam_pt(fx, fy), 9, (20, 20, 20), -1)
    return np.clip(frame + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)


def test_grid_step():
    assert grid_step(FLOOR_W, FLOOR_H) == 0.1
    assert grid_step(4, 3) == 0.5


def test_projected_size():
    assert abs(projected_size_m(H_FLOOR_PROJ, PROJ_W, PROJ_H) - np.sqrt(0.75 * 0.55)) < 1e-6


def test_dark_objects_only_inside_tape():
    rng = np.random.default_rng(0)
    inside, outside_but_lit = (0.22, 0.17), (0.52, 0.10)
    frame = fake_frame([inside, outside_but_lit], rng)
    mask = dark_search_mask(np.asarray(H_FLOOR_CAM), H_FLOOR_PROJ, FLOOR_W, FLOOR_H,
                            0.08 * min(FLOOR_W, FLOOR_H), (PROJ_W, PROJ_H), (CAM_W, CAM_H))
    objects, _ = dark_objects_in_camera(frame, mask, 0.5, 40)
    assert len(objects) == 1, f"expected only the inside object, got {len(objects)} (tape marks leaking in?)"
    found = apply_h(np.linalg.inv(H_FLOOR_CAM), [objects[0][:2]])[0]
    err_mm = np.hypot(*(found - inside)) * 1000
    assert err_mm < 5, f"object found {err_mm:.1f} mm from where it is"
    print(f"  dark objects: 1 found (tape marks and the outside object ignored), {err_mm:.1f} mm off")


def test_inset_too_big_gives_empty_mask():
    mask = dark_search_mask(np.asarray(H_FLOOR_CAM), H_FLOOR_PROJ, FLOOR_W, FLOOR_H,
                            0.2, (PROJ_W, PROJ_H), (CAM_W, CAM_H))
    assert cv2.countNonZero(mask) == 0


def test_tracker_smooths_and_keeps_id():
    rng = np.random.default_rng(1)
    dt, n = 1 / 30, 90
    true = np.stack([0.08 + 0.1 * np.arange(n) * dt, np.full(n, 0.17)], 1)  # 10 cm/s along x
    noisy = true + rng.normal(0, 0.004, true.shape)  # 4 mm jitter
    dropped = {30, 31, 32}
    tracker = FloorTracker(smooth=0.5, match_dist=0.1)
    ids, errors = set(), []
    for i in range(n):
        tracks = tracker.update([] if i in dropped else [noisy[i]], dt)
        ids |= {t.id for t in tracks}
        if tracks and i >= 10 and i not in dropped:
            errors.append(tracks[0].pos - true[i])
    errors = np.array(errors)
    raw_jitter, jitter = np.std(noisy[10:, 1] - true[10:, 1]), np.std(errors[:, 1])
    assert ids == {1}, f"ID changed: {ids}"
    assert jitter < 0.8 * raw_jitter, "smoothing didn't reduce jitter"
    assert abs(np.mean(errors[:, 0])) < 0.003, "smoothed position lags too far behind"
    print(f"  tracker: jitter {raw_jitter * 1000:.1f} -> {jitter * 1000:.1f} mm, one ID through a 3-frame dropout")


def test_tracker_ignores_one_frame_blip():
    tracker = FloorTracker(min_hits=2)
    assert tracker.update([(0.2, 0.2)], 1 / 30) == []  # seen once: not shown yet
    for _ in range(10):
        assert tracker.update([], 1 / 30) == []  # never confirmed, then dropped
    assert tracker.tracks == []


def test_tracker_reattaches_lost_track():
    dt = 1 / 10
    tracker = FloorTracker(match_dist=0.1, max_missed=2, reattach_time=2.0, reattach_dist=0.5)
    for _ in range(5):
        tracks = tracker.update([(1.0, 1.0)], dt)
    assert [t.id for t in tracks] == [1]
    for _ in range(10):  # a second unseen: lost, not reported
        tracks = tracker.update([], dt)
    assert tracks == []
    tracks = tracker.update([(1.3, 1.1)], dt)  # found again nearby
    assert [t.id for t in tracks] == [1], f"lost track not taken over: {[t.id for t in tracks]}"
    print("  tracker: someone missed for a second keeps their id")


def test_tracker_forgets_lost_track():
    dt = 1 / 10
    tracker = FloorTracker(max_missed=2, reattach_time=2.0, reattach_dist=0.5)
    for _ in range(5):
        tracker.update([(1.0, 1.0)], dt)
    for _ in range(30):  # gone for three seconds
        tracker.update([], dt)
    assert tracker.tracks == []
    tracker.update([(1.0, 1.0)], dt)
    assert [t.id for t in tracker.update([(1.0, 1.0)], dt)] == [2]
    tracker = FloorTracker(max_missed=2, reattach_dist=0.5)
    for _ in range(5):
        tracker.update([(1.0, 1.0)], dt)
    for _ in range(5):
        tracker.update([], dt)
    tracker.update([(2.0, 1.0)], dt)  # too far from where they were lost
    assert [t.id for t in tracker.update([(2.0, 1.0)], dt)] == [2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
