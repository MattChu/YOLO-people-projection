"""
tracking.py - smooth, ID-keeping tracker for points on the floor (metres).

Each track is an alpha-beta filter: a constant-velocity prediction nudged
towards each new measurement. Detections are matched to tracks by nearest
predicted position (greedy, fine for a handful of objects). New tracks must be
seen for a couple of frames before they're shown, and tracks coast through a
few missed frames, so one-frame blips and dropouts don't make circles flicker.

A shown track missed for longer is lost: no longer reported, but kept for a
couple of seconds where it was last seen. A new detection close to a lost track
takes it over, so someone the detector misses for a moment keeps their id.
"""

import itertools

import numpy as np


class Track:
    def __init__(self, tid, pos):
        self.id = tid
        self.pos = np.asarray(pos, np.float64)
        self.vel = np.zeros(2)
        self.hits = 1
        self.misses = 0
        self.lost_for = None  # seconds since it was lost, or None while it's tracked


def closest_pairs(points, detections, max_dist, used_d):
    """Greedy nearest-neighbour pairs (point index, detection index) within
    max_dist, closest first, skipping detections in used_d (and adding to it)."""
    pairs = sorted((np.linalg.norm(p - d), pi, di)
                   for pi, p in enumerate(points) for di, d in enumerate(detections) if di not in used_d)
    used_p = set()
    for dist, pi, di in pairs:
        if dist > max_dist:
            break
        if pi in used_p or di in used_d:
            continue
        used_p.add(pi)
        used_d.add(di)
        yield pi, di


class FloorTracker:
    def __init__(self, smooth=0.5, match_dist=0.1, max_missed=5, min_hits=2,
                 reattach_time=2.0, reattach_dist=0.5):
        """
        smooth         0 = raw positions, closer to 1 = smoother but laggier
        match_dist     max distance (m) between a prediction and a detection to pair them
        max_missed     frames a track survives without a detection before it's lost
        min_hits       frames a track must be seen before it's reported
        reattach_time  seconds a lost track is kept for a new detection to take over
        reattach_dist  max distance (m) between a lost track and a detection taking it over
        """
        self.alpha = 1.0 - min(max(smooth, 0.0), 0.95)
        self.beta = self.alpha ** 2 / (2 - self.alpha)  # standard alpha-beta pairing
        self.match_dist = match_dist
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.reattach_time = reattach_time
        self.reattach_dist = reattach_dist
        self.tracks = []
        self._ids = itertools.count(1)

    def update(self, detections, dt):
        """Feed this frame's detections (Nx2 floor metres) and the time since the
        last frame. Returns the confirmed tracks that aren't lost."""
        dt = max(dt, 1e-3)
        detections = np.asarray(detections, np.float64).reshape(-1, 2)
        active = [t for t in self.tracks if t.lost_for is None]
        lost = [t for t in self.tracks if t.lost_for is not None]
        predicted = [t.pos + t.vel * dt for t in active]
        used_d = set()

        # tracked ones first: nearest prediction within match_dist
        matched = set()
        for ti, di in closest_pairs(predicted, detections, self.match_dist, used_d):
            matched.add(ti)
            t = active[ti]
            residual = detections[di] - predicted[ti]
            t.pos = predicted[ti] + self.alpha * residual
            t.vel = t.vel + self.beta * residual / dt
            t.hits += 1
            t.misses = 0

        for ti, t in enumerate(active):
            if ti not in matched:  # coast on the prediction, slowing down
                t.pos = predicted[ti]
                t.vel *= 0.5
                t.misses += 1
                if t.misses > self.max_missed:
                    t.lost_for = 0.0

        # then lost ones: a detection left over near where one was last seen takes it over
        for li, di in closest_pairs([t.pos for t in lost], detections, self.reattach_dist, used_d):
            t = lost[li]
            t.pos = detections[di].copy()
            t.vel = np.zeros(2)
            t.misses = 0
            t.lost_for = None

        # lost ones age, and are forgotten in time. Ones never shown aren't kept at all
        for t in lost:
            if t.lost_for is not None:
                t.lost_for += dt
        self.tracks = [t for t in self.tracks if t.lost_for is None or
                       (t.hits >= self.min_hits and t.lost_for <= self.reattach_time)]

        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(next(self._ids), d))

        return [t for t in self.tracks if t.lost_for is None and t.hits >= self.min_hits]
