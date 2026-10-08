#!/usr/bin/env python3
"""
test_foot_calibrate.py - check the foot correction maths without a camera or projector.

Simulates why detected feet land in the wrong place: a camera 2.5 m up beside a
2.41 x 1.435 m floor sees ankles 10 cm off the floor, so each appears pushed
away from the point below the camera, and 6 cm behind the middle of the foot.
Fits a correction from people standing on the 3x3 targets and checks it also
fixes positions it wasn't fitted on.

  python test_foot_calibrate.py
"""

import numpy as np

from calib_utils import apply_h, correction_errors, fit_floor_correction
from foot_calibrate import target_grid

FLOOR_W, FLOOR_H = 2.41, 1.435
CAM_XY, CAM_HEIGHT = np.array([1.2, -1.0]), 2.5  # camera above a point just off the floor area
ANKLE_HEIGHT, ANKLE_BACK = 0.10, 0.06
FACING = np.array([0.0, -1.0])  # everyone faces the camera


def detected(feet, rng, noise=0.015):
    """Where the camera thinks feet are: the ankle point, seen along the camera's
    ray and continued down to the floor, plus detection noise."""
    ankles = np.asarray(feet) - ANKLE_BACK * FACING
    stretch = CAM_HEIGHT / (CAM_HEIGHT - ANKLE_HEIGHT)
    return CAM_XY + (ankles - CAM_XY) * stretch + rng.normal(0, noise, ankles.shape)


def test_target_grid_snakes_inside_the_tape():
    t = np.array(target_grid(FLOOR_W, FLOOR_H, 3, 3, 0.15))
    assert len(t) == 9
    assert (t[:, 0] > 0).all() and (t[:, 0] < FLOOR_W).all() and (t[:, 1] > 0).all() and (t[:, 1] < FLOOR_H).all()
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    assert steps.max() < 0.9 * FLOOR_W, "targets should snake, not jump back across the floor"
    assert target_grid(FLOOR_W, FLOOR_H, 1, 1, 0.15) == [(FLOOR_W / 2, FLOOR_H / 2)]


def test_affine_fixes_unseen_positions():
    rng = np.random.default_rng(0)
    targets = np.array(target_grid(FLOOR_W, FLOOR_H, 3, 3, 0.15))
    C, model = fit_floor_correction(detected(targets, rng), targets)
    assert model == "affine"

    # people standing anywhere else on the floor
    feet = rng.uniform([0.2, 0.2], [FLOOR_W - 0.2, FLOOR_H - 0.2], (200, 2))
    seen = detected(feet, rng, noise=0)
    before = np.linalg.norm(seen - feet, axis=1)
    after = np.linalg.norm(apply_h(C, seen) - feet, axis=1)
    print(f"  affine: unseen positions off by {before.mean() * 100:.1f} cm -> {after.mean() * 100:.1f} cm "
          f"(max {before.max() * 100:.1f} -> {after.max() * 100:.1f} cm)")
    assert before.mean() > 0.05, "the simulated bias should be noticeable"
    assert after.mean() < 0.02 and after.max() < 0.035


def test_few_points_fall_back_to_offset():
    rng = np.random.default_rng(1)
    targets = np.array(target_grid(FLOOR_W, FLOOR_H, 3, 1, 0.15))
    seen = detected(targets, rng)
    C, model = fit_floor_correction(seen, targets)
    assert model == "offset"
    assert np.allclose(C[:2, :2], np.eye(2))
    before, after, loo = correction_errors(seen, targets)
    assert after < before and loo < before
    print(f"  offset (3 points): {before * 100:.1f} -> {after * 100:.1f} cm, leave-one-out {loo * 100:.1f} cm")


def test_no_bias_gives_near_identity():
    targets = np.array(target_grid(FLOOR_W, FLOOR_H, 3, 3, 0.15))
    C, _ = fit_floor_correction(targets, targets)
    assert np.allclose(C, np.eye(3), atol=1e-9)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
