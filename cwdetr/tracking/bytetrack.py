"""Compact ByteTrack associator (Zhang et al., 2022).

Used on the deployment path when a deterministic, well-understood tracker is
preferred over end-to-end track queries (e.g. for certification, or when running
the INT8 detector-only engine on Jetson). Two-stage association: match high-score
detections first, then recover low-score ones to bridge occlusions. A constant-
velocity Kalman filter (8-state: cx, cy, aspect, h, + velocities) smooths boxes.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

    def initiate(self, meas):
        mean = np.r_[meas, np.zeros(4)]
        std = [2 * self._std_pos * meas[3], 2 * self._std_pos * meas[3], 1e-2,
               2 * self._std_pos * meas[3], 10 * self._std_vel * meas[3],
               10 * self._std_vel * meas[3], 1e-5, 10 * self._std_vel * meas[3]]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        std = [self._std_pos * mean[3], self._std_pos * mean[3], 1e-2, self._std_pos * mean[3],
               self._std_vel * mean[3], self._std_vel * mean[3], 1e-5, self._std_vel * mean[3]]
        Q = np.diag(np.square(std))
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean, cov, meas):
        std = [self._std_pos * mean[3], self._std_pos * mean[3], 1e-1, self._std_pos * mean[3]]
        R = np.diag(np.square(std))
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        mean = mean + K @ (meas - self._H @ mean)
        cov = (np.eye(len(mean)) - K @ self._H) @ cov
        return mean, cov


def _xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def _xyah_to_xyxy(m):
    w = m[2] * m[3]
    return np.array([m[0] - w / 2, m[1] - m[3] / 2, m[0] + w / 2, m[1] + m[3] / 2])


def _iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), 2)
    area_a = np.prod(a[:, 2:] - a[:, :2], 1)
    area_b = np.prod(b[:, 2:] - b[:, :2], 1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


class STrack:
    _kf = KalmanFilter()
    _count = 0

    def __init__(self, xyxy, score, cls):
        self.mean, self.cov = self._kf.initiate(_xyxy_to_xyah(xyxy))
        self.score, self.cls = score, cls
        self.track_id = -1
        self.time_since_update = 0
        self.hits = 0

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    def predict(self):
        self.mean, self.cov = self._kf.predict(self.mean, self.cov)
        self.time_since_update += 1

    def update(self, xyxy, score):
        self.mean, self.cov = self._kf.update(self.mean, self.cov, _xyxy_to_xyah(xyxy))
        self.score, self.time_since_update, self.hits = score, 0, self.hits + 1

    @property
    def xyxy(self):
        return _xyah_to_xyxy(self.mean)


class BYTETracker:
    def __init__(self, high_thresh=0.6, low_thresh=0.1, match_thresh=0.8, max_age=30):
        self.high, self.low = high_thresh, low_thresh
        self.match_thresh, self.max_age = match_thresh, max_age
        self.tracks: List[STrack] = []

    def _match(self, tracks, dets, scores, clss, thresh):
        if not tracks or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        track_boxes = np.stack([t.xyxy for t in tracks])
        iou = _iou(track_boxes, dets)
        track_classes = np.asarray([t.cls for t in tracks])
        iou = np.where(track_classes[:, None] == clss[None, :], iou, -1.0)
        rows, cols = linear_sum_assignment(-iou)
        matches, um_t, um_d = [], list(range(len(tracks))), list(range(len(dets)))
        for r, c in zip(rows, cols):
            if iou[r, c] >= thresh:
                tracks[r].update(dets[c], scores[c])
                matches.append((r, c))
                um_t.remove(r)
                um_d.remove(c)
        return matches, um_t, um_d

    def update(self, boxes_xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray):
        for t in self.tracks:
            t.predict()
        hi = scores >= self.high
        lo = (scores >= self.low) & (~hi)

        # stage 1: high-score association
        _, um_t, um_d = self._match([t for t in self.tracks], boxes_xyxy[hi],
                                    scores[hi], classes[hi], self.match_thresh)
        remaining = [self.tracks[i] for i in um_t]
        # stage 2: low-score recovery on remaining tracks
        self._match(remaining, boxes_xyxy[lo], scores[lo], classes[lo], 0.5)

        # spawn new tracks from unmatched high-score dets
        hi_boxes, hi_scores, hi_cls = boxes_xyxy[hi], scores[hi], classes[hi]
        for d in um_d:
            t = STrack(hi_boxes[d], hi_scores[d], int(hi_cls[d]))
            t.track_id = STrack.next_id()
            t.hits = 1
            self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [(t.track_id, t.xyxy, t.score, t.cls) for t in self.tracks
                if t.time_since_update == 0 and t.hits >= 2]
