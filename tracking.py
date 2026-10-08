"""
tracking.py - smooth, ID-keeping tracker for points on the floor (metres).

Each track is an alpha-beta filter: a constant-velocity prediction nudged
towards each new measurement. Detections are matched to tracks by nearest
predicted position (greedy, fine for a handful of objects). New tracks must be
seen for a couple of frames before they're shown, and tracks coast through a
few missed frames, so one-frame blips and dropouts don't make circles flicker.
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


class FloorTracker:
    def __init__(self, smooth=0.5, match_dist=0.1, max_missed=5, min_hits=2):
        """
        smooth      0 = raw positions, closer to 1 = smoother but laggier
        match_dist  max distance (m) between a prediction and a detection to pair them
        max_missed  frames a track survives without a detection
        min_hits    frames a track must be seen before it's reported
        """
        self.alpha = 1.0 - min(max(smooth, 0.0), 0.95)
        self.beta = self.alpha ** 2 / (2 - self.alpha)  # standard alpha-beta pairing
        self.match_dist = match_dist
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.tracks = []
        self._ids = itertools.count(1)

    def update(self, detections, dt):
        """Feed this frame's detections (Nx2 floor metres) and the time since the
        last frame. Returns the confirmed tracks."""
        dt = max(dt, 1e-3)
        detections = np.asarray(detections, np.float64).reshape(-1, 2)
        predicted = [t.pos + t.vel * dt for t in self.tracks]

        # greedy nearest-neighbour matching within match_dist
        pairs = sorted((np.linalg.norm(p - d), ti, di)
                       for ti, p in enumerate(predicted) for di, d in enumerate(detections))
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if dist > self.match_dist:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            residual = detections[di] - predicted[ti]
            t.pos = predicted[ti] + self.alpha * residual
            t.vel = t.vel + self.beta * residual / dt
            t.hits += 1
            t.misses = 0

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:  # coast on the prediction, slowing down
                t.pos = predicted[ti]
                t.vel *= 0.5
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_missed]
        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(next(self._ids), d))

        return [t for t in self.tracks if t.hits >= self.min_hits]
