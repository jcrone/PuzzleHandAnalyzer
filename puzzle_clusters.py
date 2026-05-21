"""puzzle_clusters.py - dynamic board detection + assembly milestones.

Runs alongside the hand pipeline in puzzle_hands.py. Every
`survey_interval_s` seconds (default 5s) it:

  1. Builds a stationary-blob list from the current frame using edge
     detection + contour finding + a 1-second stillness check.
  2. DBSCAN-clusters nearby stationary blobs into puzzle clusters.
  3. Matches detected clusters to existing tracked clusters by bounding
     box IoU, carrying IDs across surveys.

At finalize() it classifies clusters as assembly (positive net growth)
vs. sort pile (negative net growth), picks the assembly cluster as the
main one, computes seven milestones, and returns a dynamic board (bbox
of the main cluster at its peak count).

See docs/superpowers/specs/2026-05-20-dynamic-board-and-cluster-milestones-design.md
"""

import numpy as np
import cv2

from puzzle_vision_utils import hand_mask


# --------------------------------------------------------------- primitives

def dbscan_2d(points, eps, min_samples):
    """Minimal DBSCAN for 2D points. Returns int array of cluster labels
    (-1 means noise). Avoids the sklearn dependency."""
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=int)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    # Pairwise distances; fine for n up to a few thousand
    diffs = points[:, None, :] - points[None, :, :]
    d2 = (diffs * diffs).sum(axis=2)
    eps2 = eps * eps

    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbours = np.where(d2[i] <= eps2)[0]
        if len(neighbours) < min_samples:
            continue  # noise (for now)
        labels[i] = cluster_id
        seeds = list(neighbours)
        k = 0
        while k < len(seeds):
            j = seeds[k]
            k += 1
            if not visited[j]:
                visited[j] = True
                more = np.where(d2[j] <= eps2)[0]
                if len(more) >= min_samples:
                    for m in more:
                        if m not in seeds:
                            seeds.append(m)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def bbox_iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def blobs_to_clusters(blobs, eps, min_samples):
    """Cluster a list of stationary blobs using DBSCAN.

    blobs: [{"xy": (x, y), "area": float, ...}, ...] in normalized coords.
    Returns: [{"id": None, "count": int, "centroid": (x, y),
               "bbox": (x1, y1, x2, y2), "members": [xy, ...]}].
    Each cluster has 'id' = None -- IDs are assigned by the matcher.
    """
    if not blobs:
        return []
    pts = np.array([b["xy"] for b in blobs], dtype=float)
    labels = dbscan_2d(pts, eps=eps, min_samples=min_samples)
    out = []
    for cid in sorted(set(labels.tolist())):
        if cid == -1:
            continue
        mask = labels == cid
        members = pts[mask]
        x1, y1 = members.min(axis=0)
        x2, y2 = members.max(axis=0)
        out.append({
            "id": None,
            "count": int(mask.sum()),
            "centroid": (float(members[:, 0].mean()),
                         float(members[:, 1].mean())),
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "members": [tuple(m) for m in members.tolist()],
        })
    return out


# ----------------------------------------------------- the cluster mapper

class ClusterMapper:
    """Discovers and tracks stationary puzzle-piece clusters over time.

    Lifecycle:
        m = ClusterMapper(W, H, num_pieces=500, eff_fps=12.0)
        for each frame:
            m.update(frame, hands_pts, frame_idx, t)
        clusters, milestones = m.finalize(n_frames)
        board = m.inferred_board()    # None or [x1, y1, x2, y2]
    """

    def __init__(self, W, H, num_pieces=None, eff_fps=12.0,
                 survey_interval_s=5.0,
                 area_min_frac=0.0004, area_max_frac=0.012,
                 cluster_eps=0.06, dbscan_min_samples=2,
                 iou_match_threshold=0.3, dormant_after_surveys=2):
        self.W, self.H = int(W), int(H)
        self.num_pieces = num_pieces
        self.eff_fps = float(eff_fps)
        self.survey_interval_s = float(survey_interval_s)
        self.survey_stride_frames = max(1, int(survey_interval_s * eff_fps))
        self.area_min = area_min_frac * W * H
        self.area_max = area_max_frac * W * H
        self.cluster_eps = float(cluster_eps)
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.iou_match_threshold = float(iou_match_threshold)
        self.dormant_after_surveys = int(dormant_after_surveys)

        self.tracks = {}                 # id -> ClusterRecord (dict)
        self.next_id = 1
        self.surveys = []                # list of (t, [clusters_this_survey])
        self.events = []                 # ("born"|"merged"|..., t, payload)
        self._last_survey_frame = -10 ** 9
        self._prev_frame = None          # for stillness comparison
        self._prev_frame_idx = -1

    # ---- per-frame entry point ------------------------------------------
    def update(self, frame, hands_pts, frame_idx, t):
        """Most frames are no-ops; heavy work fires every survey_stride_frames."""
        if frame_idx - self._last_survey_frame < self.survey_stride_frames:
            return
        self._last_survey_frame = frame_idx
        clusters_this_survey = self._survey(frame, hands_pts)
        self._assign_ids(clusters_this_survey, t, frame_idx)
        self.surveys.append((t, clusters_this_survey))

    # ---- internal helpers ------------------------------------------------
    def _survey(self, frame, hands_pts):
        """One full survey: find stationary blobs, cluster them."""
        hm = hand_mask(self.W, self.H, hands_pts)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # canny + dilate + contours
        edges = cv2.Canny(gray, 60, 160)
        edges[hm > 0] = 0
        edges = cv2.dilate(edges, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.area_min or a > self.area_max:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx_n = (x + w / 2.0) / self.W
            cy_n = (y + h / 2.0) / self.H
            if self._is_stationary(frame, x, y, w, h):
                blobs.append({
                    "xy": (cx_n, cy_n),
                    "area": float(a / (self.W * self.H)),
                })
        self._prev_frame = frame.copy()
        return blobs_to_clusters(
            blobs, eps=self.cluster_eps,
            min_samples=self.dbscan_min_samples)

    def _is_stationary(self, frame, x, y, w, h):
        """True if this blob's ROI matches the same ROI in the previous
        survey within a small per-pixel difference threshold."""
        if self._prev_frame is None:
            # First survey -- accept everything as stationary candidate
            return True
        roi_now = frame[y:y + h, x:x + w]
        roi_prev = self._prev_frame[y:y + h, x:x + w]
        if roi_now.shape != roi_prev.shape or roi_now.size == 0:
            return False
        sad = cv2.absdiff(roi_now, roi_prev).mean()
        return sad < 8.0   # tunable; SAD over a settled puzzle piece is tiny

    def _assign_ids(self, clusters_this_survey, t, frame_idx):
        """Match this survey's clusters to existing tracks by IoU.
        Unmatched detections start new tracks; unmatched tracks may go
        dormant after `dormant_after_surveys` consecutive misses."""
        matched_tracks = set()
        # Greedy by IoU descending
        cands = []
        for tid, tr in self.tracks.items():
            for ci, c in enumerate(clusters_this_survey):
                iou = bbox_iou(tr["last_bbox"], c["bbox"])
                if iou >= self.iou_match_threshold:
                    cands.append((-iou, tid, ci))
        cands.sort()
        matched_ci = set()
        for _neg_iou, tid, ci in cands:
            if tid in matched_tracks or ci in matched_ci:
                continue
            matched_tracks.add(tid)
            matched_ci.add(ci)
            c = clusters_this_survey[ci]
            c["id"] = tid
            tr = self.tracks[tid]
            tr["last_seen_t"] = t
            tr["last_bbox"] = c["bbox"]
            tr["history_seconds"].append(t)
            tr["history_counts"].append(c["count"])
            tr["history_bboxes"].append(c["bbox"])
            if c["count"] > tr["peak_count"]:
                tr["peak_count"] = c["count"]
                tr["bbox_at_peak"] = c["bbox"]
            tr["final_count"] = c["count"]
            tr["consecutive_misses"] = 0

        # Increment miss counter for unmatched tracks
        for tid, tr in self.tracks.items():
            if tid not in matched_tracks:
                tr["consecutive_misses"] += 1
                if tr["consecutive_misses"] >= self.dormant_after_surveys:
                    tr["dormant"] = True

        # New tracks for unmatched detections
        for ci, c in enumerate(clusters_this_survey):
            if ci in matched_ci:
                continue
            tid = self.next_id
            self.next_id += 1
            c["id"] = tid
            self.tracks[tid] = {
                "id": tid,
                "born_t": t,
                "last_seen_t": t,
                "initial_count": c["count"],
                "peak_count": c["count"],
                "final_count": c["count"],
                "bbox_at_peak": c["bbox"],
                "last_bbox": c["bbox"],
                "history_seconds": [t],
                "history_counts": [c["count"]],
                "history_bboxes": [c["bbox"]],
                "consecutive_misses": 0,
                "dormant": False,
                "merged_into": None,
                "merged_from": [],
            }
            self.events.append(("born", t, {"cluster_id": tid}))

    # ---- finalize stubs (filled in Tasks 3, 4) --------------------------
    def finalize(self, n_frames):
        """Returns (clusters_list, milestones_list).

        Task 2 leaves this as a passthrough that returns raw cluster data
        with no classification or milestones. Tasks 3 and 4 add real
        logic. Integration in Task 5 wires this into puzzle_hands.py.
        """
        clusters_list = list(self.tracks.values())
        milestones = []
        return clusters_list, milestones

    def inferred_board(self):
        """Returns None for now. Task 3 implements assembly-vs-pile
        classification and returns the main cluster's bbox_at_peak."""
        return None
