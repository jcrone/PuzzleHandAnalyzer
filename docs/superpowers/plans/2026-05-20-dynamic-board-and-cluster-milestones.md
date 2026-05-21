# Dynamic Board Detection + Assembly Milestones — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace manual 4-click board marking with automatic cluster-based detection, and add seven timestamped assembly milestones to `metrics.json`.

**Architecture:** A new `puzzle_clusters.py` module runs in parallel with `puzzle_pieces.py`. Every 5 seconds it surveys the frame, finds stationary piece blobs, clusters them, and tracks each cluster's lifecycle. At finalize time it classifies the assembly cluster (largest positive net growth) vs. sort pile (negative net growth), emits milestones, and feeds its inferred board back into `puzzle_pieces.py` so `placed_on_board` works without manual marking.

**Tech Stack:** Python 3.10+, OpenCV (`cv2`), NumPy, stdlib `unittest`. No new production dependencies (project's launcher constraints — see [CLAUDE.md](../../../CLAUDE.md)).

**Spec:** [docs/superpowers/specs/2026-05-20-dynamic-board-and-cluster-milestones-design.md](../specs/2026-05-20-dynamic-board-and-cluster-milestones-design.md)

---

## File structure

**New files:**
- `puzzle_vision_utils.py` — `hand_mask()` helper shared between `puzzle_pieces.py` and `puzzle_clusters.py`. Single responsibility.
- `puzzle_clusters.py` — `ClusterMapper` class + private helpers (DBSCAN, IoU matching, classification, milestone detection).
- `tests/__init__.py` — empty package marker.
- `tests/test_puzzle_vision_utils.py` — unit tests for the hand-mask helper.
- `tests/test_puzzle_clusters.py` — unit + integration tests for ClusterMapper.

**Modified files:**
- `puzzle_pieces.py` — remove `_hand_mask` method, import from `puzzle_vision_utils`. No behavior change.
- `puzzle_hands.py` — wire `ClusterMapper` next to `PieceTracker`; new top-level JSON keys (`board_source`, `cluster_summary`, `clusters`, `milestones`); new perframe.csv columns (`largest_cluster_size`, `cluster_count`); new CLI flag `--no-clusters`.
- `puzzle_app.py` — restructure step 2 with two radio buttons (Auto-detect default, Manual); mode-aware canvas.
- `~/.claude/skills/puzzle-analyst/references/data-schema.md` — document the new JSON keys + CSV columns.

---

## Task 1: Extract hand_mask into a shared helper

**Why first:** Both `puzzle_pieces.py` (existing) and `puzzle_clusters.py` (new) need the same convex-hull-of-hand mask. Extracting first means Task 2 can import cleanly without copy-paste.

**Files:**
- Create: `puzzle_vision_utils.py`
- Create: `tests/__init__.py`
- Create: `tests/test_puzzle_vision_utils.py`
- Modify: `puzzle_pieces.py:90-110` (remove `_hand_mask` method, import + delegate)

- [ ] **Step 1: Create empty tests package marker**

Run:
```bash
touch /home/jc/Python/PuzzleTracker/tests/__init__.py
```

(The plan uses absolute paths; later steps use relative ones when running from the project root.)

- [ ] **Step 2: Write the failing test**

Create `tests/test_puzzle_vision_utils.py`:

```python
"""Tests for the hand_mask helper extracted into puzzle_vision_utils."""
import unittest
import numpy as np
import cv2

from puzzle_vision_utils import hand_mask


class TestHandMask(unittest.TestCase):

    def test_empty_hands_returns_zero_mask(self):
        m = hand_mask(W=100, H=80, hands_pts=[])
        self.assertEqual(m.shape, (80, 100))
        self.assertEqual(m.dtype, np.uint8)
        self.assertEqual(int(m.sum()), 0)

    def test_single_hand_fills_convex_hull(self):
        # 5 points roughly bounding a 40x40 region centred at (50, 40)
        pts = [(0.3, 0.4), (0.7, 0.4), (0.7, 0.6), (0.3, 0.6), (0.5, 0.5)]
        m = hand_mask(W=100, H=80, hands_pts=[pts])
        # centre should be filled
        self.assertEqual(m[40, 50], 255)
        # corner far from the hand should be empty
        self.assertEqual(m[0, 0], 0)

    def test_short_point_list_is_ignored(self):
        # convexHull needs >= 3 points; 2-point hand should be dropped
        pts = [(0.4, 0.5), (0.6, 0.5)]
        m = hand_mask(W=100, H=80, hands_pts=[pts])
        self.assertEqual(int(m.sum()), 0)

    def test_dilation_extends_mask_outward(self):
        # A tiny hand in the centre should grow outward after dilation
        pts = [(0.49, 0.49), (0.51, 0.49), (0.50, 0.51)]
        # default kernel is 35x35 ellipse — the mask should extend at least
        # 10 pixels from the original hull
        m = hand_mask(W=200, H=200, hands_pts=[pts])
        self.assertEqual(m[100, 100], 255)
        self.assertEqual(m[110, 100], 255)  # 10px below original hull centre


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run from project root:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_vision_utils -v
```

Expected: `ModuleNotFoundError: No module named 'puzzle_vision_utils'`.

- [ ] **Step 4: Create the puzzle_vision_utils.py module**

Create `puzzle_vision_utils.py`:

```python
"""Shared vision helpers used by puzzle_pieces.py and puzzle_clusters.py.

Currently exposes hand_mask: a binary mask of pixels occupied by detected
hands (convex hull + dilation), used to ignore hand pixels when looking
for puzzle-piece blobs.
"""

import numpy as np
import cv2


_DEFAULT_HAND_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))


def hand_mask(W, H, hands_pts, kernel=None):
    """Return an HxW uint8 mask with 255 where hands are, 0 elsewhere.

    hands_pts is a list of hands, each a list of (x, y) tuples in
    normalized 0-1 image coordinates. Hands with fewer than 3 points are
    skipped (convexHull needs >= 3). The resulting mask is dilated by an
    ellipse kernel (default 35x35) to swallow shadows around the hand.
    """
    m = np.zeros((H, W), dtype=np.uint8)
    for pts in hands_pts:
        if not pts or len(pts) < 3:
            continue
        arr = np.array(
            [[int(x * W), int(y * H)] for x, y in pts], dtype=np.int32)
        hull = cv2.convexHull(arr)
        cv2.fillConvexPoly(m, hull, 255)
    if m.any():
        m = cv2.dilate(m, kernel if kernel is not None else _DEFAULT_HAND_KERNEL)
    return m
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_vision_utils -v
```

Expected: `OK` with 4 tests passing.

- [ ] **Step 6: Migrate puzzle_pieces.py to use the shared helper**

In `puzzle_pieces.py`:

Add at the top of the imports (after `import cv2`):
```python
from puzzle_vision_utils import hand_mask as _hand_mask_fn
```

Delete the `_hand_mask` method (currently lines 98-110, the method starting `def _hand_mask(self, hands_pts):` through its `return m`).

Also delete the `self._hand_kernel` attribute (currently line 91) — it's no longer used by this class.

Update the one caller inside `update()` (currently line 116). Replace:
```python
        hm = self._hand_mask(hands_pts)
```
with:
```python
        hm = _hand_mask_fn(self.W, self.H, hands_pts)
```

- [ ] **Step 7: Verify nothing in puzzle_pieces broke**

Quick smoke check: import the modified module and instantiate the tracker.

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -c "
import puzzle_pieces
t = puzzle_pieces.PieceTracker(W=640, H=480)
print('OK, board:', t.board, 'next_id:', t.next_id)
"
```

Expected: `OK, board: None next_id: 1`.

- [ ] **Step 8: Commit**

```bash
cd /home/jc/Python/PuzzleTracker
git add puzzle_vision_utils.py tests/__init__.py tests/test_puzzle_vision_utils.py puzzle_pieces.py
git commit -m "$(cat <<'EOF'
Extract hand_mask helper into puzzle_vision_utils

Both puzzle_pieces.py and the upcoming puzzle_clusters.py need the same
convex-hull-of-hand mask. Pulled into a shared module with unit tests so
the new cluster mapper can import it cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: ClusterMapper skeleton + periodic detector

**Why:** The core CV pipeline — detect stationary blobs every 5 seconds, cluster them, match clusters across surveys. This task builds the detection + matching primitives but leaves classification, milestones, and integration for later tasks.

**Files:**
- Create: `puzzle_clusters.py`
- Modify: `tests/test_puzzle_clusters.py` (created in this task)

- [ ] **Step 1: Write a failing test for DBSCAN clustering**

Create `tests/test_puzzle_clusters.py`:

```python
"""Tests for the cluster mapper: DBSCAN, IoU matching, blob survey."""
import unittest
import numpy as np

from puzzle_clusters import (
    dbscan_2d, bbox_iou, blobs_to_clusters,
)


class TestDbscan2D(unittest.TestCase):

    def test_two_clusters_of_three(self):
        # Two well-separated clusters, eps small enough to keep them apart
        pts = np.array([
            (0.10, 0.10), (0.11, 0.11), (0.12, 0.10),   # cluster A
            (0.80, 0.80), (0.81, 0.80), (0.80, 0.81),   # cluster B
        ])
        labels = dbscan_2d(pts, eps=0.05, min_samples=2)
        # Two distinct labels, each appearing 3 times; no -1 (noise)
        unique = sorted(set(labels.tolist()))
        self.assertEqual(len(unique), 2)
        self.assertNotIn(-1, unique)
        counts = sorted([(labels == u).sum() for u in unique])
        self.assertEqual(counts, [3, 3])

    def test_isolated_point_is_noise(self):
        pts = np.array([
            (0.10, 0.10), (0.11, 0.11),    # cluster
            (0.90, 0.90),                  # noise
        ])
        labels = dbscan_2d(pts, eps=0.05, min_samples=2)
        # The isolated point gets -1
        self.assertEqual((labels == -1).sum(), 1)
        # The other two share a label
        non_noise = labels[labels != -1]
        self.assertEqual(len(set(non_noise.tolist())), 1)

    def test_empty_input(self):
        labels = dbscan_2d(np.empty((0, 2)), eps=0.05, min_samples=2)
        self.assertEqual(len(labels), 0)


class TestBboxIou(unittest.TestCase):

    def test_identical_boxes(self):
        b = (0.1, 0.1, 0.5, 0.5)
        self.assertAlmostEqual(bbox_iou(b, b), 1.0)

    def test_disjoint_boxes(self):
        a = (0.0, 0.0, 0.2, 0.2)
        b = (0.5, 0.5, 0.8, 0.8)
        self.assertEqual(bbox_iou(a, b), 0.0)

    def test_half_overlap(self):
        # b is the right half of a — IoU = 0.5/1.5 = 1/3
        a = (0.0, 0.0, 1.0, 1.0)
        b = (0.5, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(bbox_iou(a, b), 1.0 / 3.0, places=5)


class TestBlobsToClusters(unittest.TestCase):

    def test_groups_nearby_blobs(self):
        blobs = [
            {"xy": (0.10, 0.10), "area": 0.001},
            {"xy": (0.11, 0.11), "area": 0.001},
            {"xy": (0.12, 0.10), "area": 0.001},
            {"xy": (0.80, 0.80), "area": 0.001},
            {"xy": (0.81, 0.80), "area": 0.001},
        ]
        clusters = blobs_to_clusters(blobs, eps=0.05, min_samples=2)
        # Two clusters of size 3 and 2
        sizes = sorted(c["count"] for c in clusters)
        self.assertEqual(sizes, [2, 3])
        # Each cluster has a bbox
        for c in clusters:
            x1, y1, x2, y2 = c["bbox"]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)

    def test_single_isolated_blob_dropped(self):
        # min_samples=2 means a lone blob is noise
        blobs = [{"xy": (0.5, 0.5), "area": 0.001}]
        clusters = blobs_to_clusters(blobs, eps=0.05, min_samples=2)
        self.assertEqual(clusters, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_clusters -v
```

Expected: `ModuleNotFoundError: No module named 'puzzle_clusters'`.

- [ ] **Step 3: Create puzzle_clusters.py with the detection primitives**

Create `puzzle_clusters.py`:

```python
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
    Each cluster has 'id' = None — IDs are assigned by the matcher.
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
            # First survey — accept everything as stationary candidate
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_clusters -v
```

Expected: all three test classes pass (8 tests OK).

- [ ] **Step 5: Add a smoke test that runs update() end-to-end**

Append to `tests/test_puzzle_clusters.py`:

```python
class TestClusterMapperUpdate(unittest.TestCase):

    def _synth_frame(self, W, H, blob_positions):
        """Make a black frame with white squares at the given positions."""
        f = np.zeros((H, W, 3), dtype=np.uint8)
        for (cx, cy) in blob_positions:
            x = int(cx * W); y = int(cy * H)
            cv2.rectangle(f, (x - 6, y - 6), (x + 6, y + 6), (255, 255, 255), -1)
        return f

    def test_update_short_circuits_off_survey_frames(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, eff_fps=10.0,
                          survey_interval_s=5.0,
                          area_min_frac=0.0001, area_max_frac=0.5,
                          cluster_eps=0.10)
        frame = self._synth_frame(200, 200, [(0.3, 0.3), (0.35, 0.3)])
        m.update(frame, hands_pts=[], frame_idx=0, t=0.0)
        # survey stride = 5s * 10fps = 50 frames; frame 1 should be a no-op
        n_after_first = len(m.surveys)
        m.update(frame, hands_pts=[], frame_idx=1, t=0.1)
        self.assertEqual(len(m.surveys), n_after_first)

    def test_two_consecutive_surveys_share_cluster_id(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, eff_fps=10.0,
                          survey_interval_s=5.0,
                          area_min_frac=0.0001, area_max_frac=0.5,
                          cluster_eps=0.10)
        blobs = [(0.3, 0.3), (0.35, 0.3), (0.3, 0.35)]
        f0 = self._synth_frame(200, 200, blobs)
        f1 = self._synth_frame(200, 200, blobs)
        m.update(f0, [], frame_idx=0, t=0.0)
        m.update(f1, [], frame_idx=50, t=5.0)   # next survey window
        # Same blobs in same place across two surveys -> one persistent ID
        self.assertEqual(len(m.tracks), 1)
        tr = list(m.tracks.values())[0]
        self.assertGreaterEqual(len(tr["history_seconds"]), 2)
```

- [ ] **Step 6: Run the new tests**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_clusters -v
```

Expected: 10 tests OK.

- [ ] **Step 7: Commit**

```bash
cd /home/jc/Python/PuzzleTracker
git add puzzle_clusters.py tests/test_puzzle_clusters.py
git commit -m "$(cat <<'EOF'
Add ClusterMapper skeleton with periodic detector

New puzzle_clusters.py module. Implements:
- hand-rolled DBSCAN (no sklearn dependency)
- bbox IoU helper
- blobs_to_clusters: edge+contour detection + stillness gate + DBSCAN
- ClusterMapper.update(): survey-every-5s loop with IoU-based ID matching

finalize() and inferred_board() are stubs - filled in later tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Cluster bookkeeping, pruning, and assembly-vs-pile classification

**Why:** With raw cluster tracks captured, we need to classify which one is the assembly cluster (positive net growth) vs. the sort pile (negative net growth), prune noise, and implement `inferred_board()`. This is the conceptual heart of the spec.

**Files:**
- Modify: `puzzle_clusters.py` (extend `finalize()` and `inferred_board()`)
- Modify: `tests/test_puzzle_clusters.py` (add classification tests)

- [ ] **Step 1: Write failing tests for classification**

Append to `tests/test_puzzle_clusters.py`:

```python
class TestClassification(unittest.TestCase):

    def _make_mapper_with_synthetic_tracks(self, tracks):
        """Build a ClusterMapper, then poke synthetic tracks into it
        directly (bypassing update). tracks is a list of dicts."""
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        for tr in tracks:
            m.tracks[tr["id"]] = tr
        m.next_id = max(t["id"] for t in tracks) + 1
        return m

    def _track(self, tid, initial, final, peak=None, bbox=(0.2, 0.2, 0.7, 0.7),
               born_t=10.0, last_t=100.0):
        if peak is None:
            peak = max(initial, final)
        return {
            "id": tid, "born_t": born_t, "last_seen_t": last_t,
            "initial_count": initial, "peak_count": peak,
            "final_count": final, "bbox_at_peak": bbox,
            "last_bbox": bbox,
            "history_seconds": [born_t, last_t],
            "history_counts": [initial, final],
            "history_bboxes": [bbox, bbox],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }

    def test_assembly_cluster_wins_over_sort_pile(self):
        # Pile: starts at 200, ends at 10 (net growth -190)
        # Assembly: starts at 2, ends at 80 (net growth +78)
        pile = self._track(1, initial=200, final=10,
                           bbox=(0.0, 0.0, 0.3, 0.4))
        asm = self._track(2, initial=2, final=80,
                          bbox=(0.4, 0.2, 0.9, 0.8))
        m = self._make_mapper_with_synthetic_tracks([pile, asm])
        clusters, _milestones = m.finalize(n_frames=1000)
        main = [c for c in clusters if c.get("is_main")]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0]["id"], 2)

    def test_no_main_when_nothing_grows(self):
        # Only sort pile present; no assembly cluster
        pile = self._track(1, initial=200, final=10)
        m = self._make_mapper_with_synthetic_tracks([pile])
        clusters, _ = m.finalize(n_frames=1000)
        main = [c for c in clusters if c.get("is_main")]
        self.assertEqual(len(main), 0)
        self.assertIsNone(m.inferred_board())

    def test_aspect_ratio_tiebreak(self):
        # Both clusters have similar positive growth. The one whose
        # aspect ratio is closer to a typical puzzle (1:2 .. 2:1) wins.
        # ar 4:1 -> elongated, looks like a pile shoved against an edge
        a = self._track(1, initial=2, final=50,
                        bbox=(0.0, 0.45, 0.8, 0.55))   # aspect 8:1
        # ar 1:1 -> classic puzzle shape
        b = self._track(2, initial=2, final=48,
                        bbox=(0.2, 0.2, 0.7, 0.7))     # aspect 1:1
        m = self._make_mapper_with_synthetic_tracks([a, b])
        clusters, _ = m.finalize(n_frames=1000)
        main = [c for c in clusters if c.get("is_main")]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0]["id"], 2)
        # cluster_summary should record the tiebreak path
        self.assertTrue(m.last_summary["tiebreak_used"])

    def test_prunes_tiny_noise_tracks(self):
        # A track that never grew past peak_count=2 should be removed
        noise = self._track(1, initial=2, final=2, peak=2)
        asm = self._track(2, initial=2, final=50, peak=50)
        m = self._make_mapper_with_synthetic_tracks([noise, asm])
        clusters, _ = m.finalize(n_frames=1000)
        ids = {c["id"] for c in clusters}
        self.assertNotIn(1, ids)
        self.assertIn(2, ids)


class TestInferredBoard(unittest.TestCase):

    def test_returns_main_cluster_bbox_at_peak(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        m.tracks[2] = {
            "id": 2, "born_t": 10.0, "last_seen_t": 100.0,
            "initial_count": 2, "peak_count": 80, "final_count": 80,
            "bbox_at_peak": (0.1, 0.2, 0.7, 0.8),
            "last_bbox": (0.1, 0.2, 0.7, 0.8),
            "history_seconds": [10.0, 100.0],
            "history_counts": [2, 80],
            "history_bboxes": [(0.1, 0.2, 0.7, 0.8), (0.1, 0.2, 0.7, 0.8)],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.next_id = 3
        m.finalize(n_frames=1000)   # populates is_main
        board = m.inferred_board()
        self.assertEqual(board, [0.1, 0.2, 0.7, 0.8])
```

- [ ] **Step 2: Run new tests to verify they fail**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_clusters.TestClassification tests.test_puzzle_clusters.TestInferredBoard -v
```

Expected: tests fail because `is_main` is never set, `last_summary` doesn't exist, `inferred_board` returns None.

- [ ] **Step 3: Implement classification + pruning in finalize()**

In `puzzle_clusters.py`, replace the existing `finalize()` and `inferred_board()` stubs with:

```python
    # ---- finalize -------------------------------------------------------
    NET_GROWTH_MIN = 5             # cluster must grow by this much to qualify
    PRUNE_PEAK_BELOW = 3           # drop tracks whose peak_count < this
    PUZZLE_AR_MIN = 0.5            # typical puzzle aspect ratios 1:2 .. 2:1
    PUZZLE_AR_MAX = 2.0

    def finalize(self, n_frames):
        """Classify clusters, prune noise, populate is_main, return
        (clusters_list, milestones_list). Milestones list is filled by
        Task 4; this task leaves it empty."""
        # 1. Prune transient noise
        kept = [tr for tr in self.tracks.values()
                if tr["peak_count"] >= self.PRUNE_PEAK_BELOW]

        # 2. Compute net growth for each surviving track
        for tr in kept:
            tr["net_growth"] = tr["final_count"] - tr["initial_count"]
            tr["is_main"] = False

        # 3. Pick the main (assembly) cluster: max positive net growth.
        #    Tiebreak: aspect ratio closest to 1:1 within the puzzle band.
        candidates = [tr for tr in kept
                      if tr["net_growth"] >= self.NET_GROWTH_MIN]
        tiebreak_used = False
        main = None
        if candidates:
            top_growth = max(tr["net_growth"] for tr in candidates)
            top = [tr for tr in candidates
                   if tr["net_growth"] >= 0.9 * top_growth]
            if len(top) == 1:
                main = top[0]
            else:
                tiebreak_used = True
                def ar_score(tr):
                    x1, y1, x2, y2 = tr["bbox_at_peak"]
                    w = max(1e-6, x2 - x1); h = max(1e-6, y2 - y1)
                    ar = w / h
                    in_band = self.PUZZLE_AR_MIN <= ar <= self.PUZZLE_AR_MAX
                    # closer to 1.0 is better; out-of-band gets penalised
                    return abs(np.log(ar)) + (0.0 if in_band else 10.0)
                main = min(top, key=ar_score)
            main["is_main"] = True

        # 4. Cluster summary (used by puzzle_hands.py later)
        self.last_summary = {
            "total_clusters_observed": len(self.tracks),
            "total_clusters_kept": len(kept),
            "main_cluster_id": main["id"] if main else None,
            "main_cluster_peak_count": main["peak_count"] if main else 0,
            "main_cluster_net_growth": main["net_growth"] if main else 0,
            "merge_events": sum(1 for e in self.events if e[0] == "merged"),
            "survey_interval_s": self.survey_interval_s,
            "tiebreak_used": tiebreak_used,
        }

        milestones = []          # filled in Task 4
        return kept, milestones

    def inferred_board(self):
        """Return [x1, y1, x2, y2] of the main cluster's bbox at peak,
        or None if no assembly cluster was identified.

        finalize() must be called first."""
        for tr in self.tracks.values():
            if tr.get("is_main"):
                x1, y1, x2, y2 = tr["bbox_at_peak"]
                return [float(x1), float(y1), float(x2), float(y2)]
        return None
```

- [ ] **Step 4: Run all cluster tests**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_clusters -v
```

Expected: all tests pass (including TestClassification, TestInferredBoard).

- [ ] **Step 5: Commit**

```bash
cd /home/jc/Python/PuzzleTracker
git add puzzle_clusters.py tests/test_puzzle_clusters.py
git commit -m "$(cat <<'EOF'
Add cluster classification: assembly vs sort pile

ClusterMapper.finalize() now prunes noise (peak < 3), computes net
growth per cluster, and picks the main cluster by largest positive net
growth. Aspect-ratio tiebreak when two clusters are within 10% of each
other - prefers shapes nearer 1:1 within the 1:2..2:1 band.

inferred_board() returns the main cluster's bbox at peak, or None if
no cluster met the net-growth threshold.

Milestones list still empty - next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Milestone detection (seven types)

**Why:** Produce the chronological milestone log the puzzle-analyst skill will surface. All milestones are computed from the cluster history at finalize time.

**Files:**
- Modify: `puzzle_clusters.py` (add `_compute_milestones()`, call from `finalize()`)
- Modify: `tests/test_puzzle_clusters.py` (add milestone tests)

- [ ] **Step 1: Write failing tests for milestones**

Append to `tests/test_puzzle_clusters.py`:

```python
class TestMilestones(unittest.TestCase):

    def _mapper_with_growth_history(self, num_pieces, history):
        """history is [(t, count, bbox), ...] for a single growing cluster."""
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=num_pieces, eff_fps=10.0)
        history_seconds = [h[0] for h in history]
        history_counts = [h[1] for h in history]
        history_bboxes = [h[2] for h in history]
        m.tracks[1] = {
            "id": 1, "born_t": history[0][0],
            "last_seen_t": history[-1][0],
            "initial_count": history[0][1],
            "peak_count": max(history_counts),
            "final_count": history[-1][1],
            "bbox_at_peak": history_bboxes[history_counts.index(max(history_counts))],
            "last_bbox": history_bboxes[-1],
            "history_seconds": history_seconds,
            "history_counts": history_counts,
            "history_bboxes": history_bboxes,
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.next_id = 2
        return m

    def test_first_piece_joined_fires(self):
        bbox = (0.2, 0.2, 0.7, 0.7)
        m = self._mapper_with_growth_history(
            100, [(10.0, 2, bbox), (20.0, 50, bbox)])
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertIn("first_piece_joined", types)
        fp = next(ms for ms in milestones if ms["type"] == "first_piece_joined")
        self.assertAlmostEqual(fp["t"], 10.0)
        self.assertEqual(fp["cluster_id"], 1)

    def test_25_50_75_pct_thresholds(self):
        bbox = (0.2, 0.2, 0.7, 0.7)
        # num_pieces=100 -> 25/50/75 pieces = thresholds
        history = [(10.0 * i, 5 * i, bbox) for i in range(1, 21)]   # 5,10,...,100
        m = self._mapper_with_growth_history(100, history)
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        for t in ("cluster_25pct", "cluster_50pct", "cluster_75pct"):
            self.assertIn(t, types)
        # 25% threshold crossed at count >= 25 -> first such survey is i=5 (count=25)
        ms25 = next(m for m in milestones if m["type"] == "cluster_25pct")
        self.assertEqual(ms25["details"]["count"], 25)

    def test_pct_milestones_omitted_when_num_pieces_none(self):
        bbox = (0.2, 0.2, 0.7, 0.7)
        m = self._mapper_with_growth_history(
            None, [(10.0, 2, bbox), (20.0, 50, bbox), (30.0, 80, bbox)])
        m.num_pieces = None  # explicit
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        for t in ("cluster_25pct", "cluster_50pct", "cluster_75pct"):
            self.assertNotIn(t, types)
        # but first_piece_joined still present
        self.assertIn("first_piece_joined", types)

    def test_frame_complete_fires_on_ring_shape(self):
        from puzzle_clusters import ClusterMapper, _perimeter_fraction
        # Build a roughly 25x20 ring of piece positions inside a bbox
        bbox = (0.1, 0.1, 0.9, 0.7)
        x1, y1, x2, y2 = bbox
        cols, rows = 25, 20
        ring = []
        for c in range(cols):
            for r in range(rows):
                if c == 0 or c == cols - 1 or r == 0 or r == rows - 1:
                    fx = x1 + (c + 0.5) * (x2 - x1) / cols
                    fy = y1 + (r + 0.5) * (y2 - y1) / rows
                    ring.append((fx, fy))
        count = len(ring)   # 2*(25+20)-4 = 86
        m = self._mapper_with_growth_history(
            500, [(10.0, 2, bbox), (1800.0, count, bbox)])
        # Patch history_members on the track so frame-complete can inspect it
        m.tracks[1]["history_members"] = [ring, ring]
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertIn("frame_complete", types)

    def test_frame_complete_does_not_fire_on_filled_rectangle(self):
        # A 25x20 fully-filled rectangle (500 pieces) — high interior fraction
        bbox = (0.1, 0.1, 0.9, 0.7)
        x1, y1, x2, y2 = bbox
        cols, rows = 25, 20
        filled = []
        for c in range(cols):
            for r in range(rows):
                fx = x1 + (c + 0.5) * (x2 - x1) / cols
                fy = y1 + (r + 0.5) * (y2 - y1) / rows
                filled.append((fx, fy))
        m = self._mapper_with_growth_history(
            500, [(10.0, 2, bbox), (1800.0, len(filled), bbox)])
        m.tracks[1]["history_members"] = [filled, filled]
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertNotIn("frame_complete", types)

    def test_islands_merged_event_becomes_milestone(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        bbox = (0.2, 0.2, 0.7, 0.7)
        m.tracks[1] = {
            "id": 1, "born_t": 10.0, "last_seen_t": 100.0,
            "initial_count": 2, "peak_count": 30, "final_count": 30,
            "bbox_at_peak": bbox, "last_bbox": bbox,
            "history_seconds": [10.0, 100.0],
            "history_counts": [2, 30],
            "history_bboxes": [bbox, bbox],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.tracks[2] = {
            "id": 2, "born_t": 50.0, "last_seen_t": 200.0,
            "initial_count": 2, "peak_count": 60, "final_count": 60,
            "bbox_at_peak": bbox, "last_bbox": bbox,
            "history_seconds": [50.0, 200.0],
            "history_counts": [2, 60],
            "history_bboxes": [bbox, bbox],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [1],
        }
        m.next_id = 3
        m.events.append(("merged", 180.0,
                         {"from_id": 1, "into_id": 2,
                          "from_peak": 30, "into_peak": 60}))
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertIn("islands_merged", types)

    def test_sort_pile_cleared_fires(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        bbox_pile = (0.0, 0.0, 0.3, 0.4)
        bbox_asm = (0.4, 0.2, 0.9, 0.8)
        # Pile shrinks 200 -> 15 (below 20% of peak=200, i.e. 40)
        m.tracks[1] = {
            "id": 1, "born_t": 0.0, "last_seen_t": 900.0,
            "initial_count": 200, "peak_count": 200, "final_count": 15,
            "bbox_at_peak": bbox_pile, "last_bbox": bbox_pile,
            "history_seconds": [0.0, 300.0, 600.0, 900.0],
            "history_counts": [200, 100, 50, 15],
            "history_bboxes": [bbox_pile] * 4,
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.tracks[2] = {
            "id": 2, "born_t": 100.0, "last_seen_t": 1800.0,
            "initial_count": 2, "peak_count": 80, "final_count": 80,
            "bbox_at_peak": bbox_asm, "last_bbox": bbox_asm,
            "history_seconds": [100.0, 1800.0],
            "history_counts": [2, 80],
            "history_bboxes": [bbox_asm] * 2,
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.next_id = 3
        _, milestones = m.finalize(n_frames=2000)
        types = [ms["type"] for ms in milestones]
        self.assertIn("sort_pile_cleared", types)
        sp = next(ms for ms in milestones if ms["type"] == "sort_pile_cleared")
        # Pile drops below 20% of peak (= 40) somewhere between count=50 and count=15
        # Expected at t=900 (where count dropped to 15).
        self.assertEqual(sp["details"]["count"], 15)
```

- [ ] **Step 2: Run new tests to verify they fail**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_clusters.TestMilestones -v
```

Expected: 7 test failures (milestones list always empty).

- [ ] **Step 3: Implement milestone detection in puzzle_clusters.py**

Add this module-level helper near the top of `puzzle_clusters.py` (after `bbox_iou`):

```python
def _perimeter_fraction(members, bbox, piece_size_x, piece_size_y):
    """Fraction of cluster members within 1 piece-width of the bbox edge."""
    if not members:
        return 0.0
    x1, y1, x2, y2 = bbox
    n_perim = 0
    for (mx, my) in members:
        if (mx - x1 < piece_size_x or x2 - mx < piece_size_x or
                my - y1 < piece_size_y or y2 - my < piece_size_y):
            n_perim += 1
    return n_perim / len(members)
```

Then add this method to the `ClusterMapper` class (replace the empty milestones list in `finalize` with a call):

In `finalize()`, replace `milestones = []` with `milestones = self._compute_milestones(kept, main)`.

Also, update `_assign_ids` so each track has a `history_members` list populated from the survey clusters. In `_assign_ids` where matched tracks are updated, after `tr["history_bboxes"].append(c["bbox"])` add:
```python
            tr.setdefault("history_members", []).append(c["members"])
```
And for new tracks where `self.tracks[tid] = { ... }`, add the key:
```python
                "history_members": [c["members"]],
```

Now add the milestone computer:

```python
    # Milestones
    FRAME_CLUSTER_SIZE_MIN_FRAC = 0.80
    FRAME_CLUSTER_SIZE_MAX_FRAC = 1.30
    FRAME_PERIMETER_FRACTION = 0.85
    ISLAND_MIN_PEAK = 10
    PILE_CLEARED_PEAK_FRAC = 0.20
    PILE_CLEARED_ABS_MIN = 20

    def _compute_milestones(self, kept, main):
        ms = []

        # first_piece_joined - first survey where main reached >= 2
        if main is not None:
            for t, count in zip(main["history_seconds"], main["history_counts"]):
                if count >= 2:
                    ms.append({
                        "t": float(t),
                        "type": "first_piece_joined",
                        "label": "First two pieces joined",
                        "cluster_id": main["id"],
                        "confidence": "high",
                    })
                    break

        # cluster_25/50/75 - only if num_pieces is supplied
        if main is not None and self.num_pieces:
            n = float(self.num_pieces)
            for frac, label in [(0.25, "cluster_25pct"),
                                (0.50, "cluster_50pct"),
                                (0.75, "cluster_75pct")]:
                threshold = frac * n
                for t, count in zip(main["history_seconds"],
                                    main["history_counts"]):
                    if count >= threshold:
                        ms.append({
                            "t": float(t),
                            "type": label,
                            "label": "Cluster reached %d%% of pieces (%d / %d)"
                                     % (int(frac * 100), int(count), int(n)),
                            "cluster_id": main["id"],
                            "confidence": "high",
                            "details": {"count": int(count),
                                        "fraction": round(count / n, 3)},
                        })
                        break

        # frame_complete - heuristic on main cluster's history
        if main is not None and main.get("history_members"):
            for t, count, bbox, members in zip(
                    main["history_seconds"], main["history_counts"],
                    main["history_bboxes"], main["history_members"]):
                # Estimate puzzle grid dimensions from bbox + median piece spacing
                x1, y1, x2, y2 = bbox
                w_box, h_box = x2 - x1, y2 - y1
                # Estimate piece dimensions: use sqrt(area) as proxy, but here
                # use the cluster's median spacing along each axis instead.
                if len(members) < 4:
                    continue
                ms_arr = np.array(members)
                # Pairwise spacings along each axis (small percentile = a piece-width)
                xs = np.sort(np.unique(ms_arr[:, 0]))
                ys = np.sort(np.unique(ms_arr[:, 1]))
                if len(xs) < 2 or len(ys) < 2:
                    continue
                piece_w = float(np.median(np.diff(xs)))
                piece_h = float(np.median(np.diff(ys)))
                if piece_w <= 0 or piece_h <= 0:
                    continue
                cols = max(1, round(w_box / piece_w))
                rows = max(1, round(h_box / piece_h))
                expected_frame = 2 * (cols + rows) - 4
                size_ok = (self.FRAME_CLUSTER_SIZE_MIN_FRAC * expected_frame
                           <= count <=
                           self.FRAME_CLUSTER_SIZE_MAX_FRAC * expected_frame)
                if not size_ok:
                    continue
                pf = _perimeter_fraction(members, bbox, piece_w, piece_h)
                if pf > self.FRAME_PERIMETER_FRACTION:
                    ms.append({
                        "t": float(t),
                        "type": "frame_complete",
                        "label": "Outer frame appears complete",
                        "cluster_id": main["id"],
                        "confidence": "medium",
                        "details": {
                            "perimeter_fraction": round(pf, 3),
                            "expected_frame_pieces": int(expected_frame),
                        },
                    })
                    break

        # islands_merged - one per qualifying merge event
        for ev_type, t, payload in self.events:
            if ev_type != "merged":
                continue
            if (payload.get("from_peak", 0) >= self.ISLAND_MIN_PEAK and
                    payload.get("into_peak", 0) >= self.ISLAND_MIN_PEAK):
                ms.append({
                    "t": float(t),
                    "type": "islands_merged",
                    "label": "Two cluster islands merged",
                    "cluster_id": payload.get("into_id"),
                    "confidence": "high",
                    "details": {"from_id": payload.get("from_id"),
                                "from_peak": payload.get("from_peak"),
                                "into_peak": payload.get("into_peak")},
                })

        # sort_pile_cleared - cluster with most negative net growth drops
        # below 20% of its peak (or below 20 pieces, whichever is lower)
        piles = sorted(
            (tr for tr in kept
             if tr["final_count"] - tr["initial_count"] < 0),
            key=lambda tr: tr["final_count"] - tr["initial_count"])
        if piles:
            pile = piles[0]
            threshold = min(self.PILE_CLEARED_ABS_MIN,
                            self.PILE_CLEARED_PEAK_FRAC * pile["peak_count"])
            for t, count in zip(pile["history_seconds"],
                                pile["history_counts"]):
                if count < threshold:
                    ms.append({
                        "t": float(t),
                        "type": "sort_pile_cleared",
                        "label": "Sort pile mostly worked through",
                        "cluster_id": pile["id"],
                        "confidence": "medium",
                        "details": {
                            "count": int(count),
                            "peak": int(pile["peak_count"]),
                            "threshold": float(threshold),
                        },
                    })
                    break

        ms.sort(key=lambda m: m["t"])
        return ms
```

- [ ] **Step 4: Run all cluster tests**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest tests.test_puzzle_clusters -v
```

Expected: all tests pass (17 total).

- [ ] **Step 5: Commit**

```bash
cd /home/jc/Python/PuzzleTracker
git add puzzle_clusters.py tests/test_puzzle_clusters.py
git commit -m "$(cat <<'EOF'
Add milestone detection (7 types)

ClusterMapper.finalize() now emits a chronological milestones list:
- first_piece_joined (main cluster reaches count >= 2)
- cluster_25/50/75pct (when num_pieces is supplied)
- frame_complete (ring shape: perimeter_fraction > 0.85 and size in
  80-130% of expected frame count)
- islands_merged (merge events between clusters that each reached peak >= 10)
- sort_pile_cleared (most-negative-net-growth cluster drops below 20%
  of peak, or below 20 pieces)

Each milestone has type, label, t, cluster_id, confidence, details.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Integration into puzzle_hands.py — wiring, JSON, CSV, CLI

**Why:** The ClusterMapper now works in isolation. This task wires it into the analyzer pipeline so its output reaches users via `metrics.json` and `perframe.csv`, and the inferred board reaches `puzzle_pieces.py` so `placed_on_board` finally works.

**Files:**
- Modify: `puzzle_hands.py` (multiple edits)

- [ ] **Step 1: Add the `track_clusters` parameter and instance build**

In `puzzle_hands.py`, update the `analyze()` signature (currently line 126-129). The existing signature ends with `track_pieces=True` — add `track_clusters=True` after it:

```python
def analyze(video, start, duration, proc_fps, move_thresh, finger_thresh,
            grip_thresh, board, seg_seconds, preview_seconds, swap,
            want_video, outdir, puzzle_name=None, num_pieces=None,
            difficulty=None, track_pieces=True, track_clusters=True):
```

Add `import puzzle_clusters` at the top alongside `import puzzle_pieces`.

Right after the `piece_tracker = ...` block (line 174-176), add:

```python
    cluster_mapper = (puzzle_clusters.ClusterMapper(
                         W, H, num_pieces=num_pieces, eff_fps=eff_fps)
                      if track_clusters else None)
```

- [ ] **Step 2: Call `cluster_mapper.update()` parallel to piece_tracker**

In the same loop where `piece_tracker.update(...)` is called (currently line 202-203), add immediately after:

```python
            if cluster_mapper is not None:
                cluster_mapper.update(frame, hands_pts_frame, j, j / eff_fps)
```

- [ ] **Step 3: Add finalize, inferred board, and piece-tracker board hand-off**

In the finalize section near line 363-388, two things need to change:

**3a.** Locate this block (around line 366-388 — search for `piece_summary = None`):

```python
    pieces_list = []
    pieces_visible = np.zeros(N, dtype=int)
    pieces_on_board = np.zeros(N, dtype=int)
    piece_summary = None
    if piece_tracker is not None:
        ...
        pieces_list, pieces_visible, pieces_on_board = \
            piece_tracker.finalize(N, grip_events)
        piece_summary = puzzle_pieces.PieceTracker.summarize(
            pieces_list, dur_s)
```

**Insert these lines IMMEDIATELY BEFORE** the `pieces_list = []` line (do NOT remove that line — only add new lines above it):

```python
    cluster_history = []
    milestones = []
    cluster_summary = None
    inferred_board = None
    board_source = "manual" if board is not None else "none"

    if cluster_mapper is not None:
        cluster_history, milestones = cluster_mapper.finalize(N)
        cluster_summary = cluster_mapper.last_summary
        inferred_board = cluster_mapper.inferred_board()
        if board is None and inferred_board is not None:
            board_source = "auto-detected"

```

**3b.** Inside the existing `if piece_tracker is not None:` block, add this hand-off as the **first** statement (before the existing `piece_tracker.finalize(...)` call):

```python
        if piece_tracker.board is None and inferred_board is not None:
            piece_tracker.board = inferred_board
```

**Trade-off to note:** the inferred board flows into `piece_tracker.placed_on_board` (the headline value) but does NOT retroactively populate `series[s]["in_board"]` (already computed upstream from the parameter `board`), and is NOT passed to `_session_chart(..., board, ...)` or `_calibration(..., board, ...)`. Net effect in auto-detect mode: `metrics.json` has the correct `board_region` / `board_source` / `placed_on_board`, but the per-frame `right_in_board` / `left_in_board` CSV columns and the session chart's board outline are absent. That's an acceptable scope cut — the puzzle-analyst skill reads metrics.json primarily. Wiring the inferred board into the chart calls is a small follow-up if needed.

- [ ] **Step 4: Add the new top-level keys to the summary dict**

Find the `if piece_tracker is not None:` block that adds `summary["piece_summary"]` and `summary["pieces"]` (around line 498-500). Add the cluster keys **after that if-block closes** — i.e., at the same indentation level as `if piece_tracker is not None:` itself (4 spaces), so they are added unconditionally:

```python
    summary["board_source"] = board_source
    if board_source == "auto-detected" and inferred_board is not None:
        summary["board_region"] = inferred_board
    if cluster_summary is not None:
        # Quality flag computation
        notes = []
        quality = "high"
        hand_det = (summary["left"]["detected_pct"] +
                    summary["right"]["detected_pct"]) / 2.0
        main_peak = cluster_summary["main_cluster_peak_count"]
        if hand_det < 60:
            quality = "low"
            notes.append(
                "Hand detection rate %.0f%% - piece detections may be affected"
                % hand_det)
        elif hand_det < 75:
            quality = "medium"
            notes.append(
                "Hand detection rate %.0f%% - some piece detections may be obscured"
                % hand_det)
        if main_peak < 20:
            quality = "low"
            notes.append(
                "Main cluster peak count %d - very little assembly detected"
                % main_peak)
        if cluster_summary["tiebreak_used"]:
            quality = "medium" if quality == "high" else quality
            notes.append("Tiebreak used to pick main cluster")
        cluster_summary["detection_quality"] = quality
        cluster_summary["quality_notes"] = notes
        summary["cluster_summary"] = cluster_summary
        summary["clusters"] = _serialize_clusters(cluster_history)
        summary["milestones"] = milestones
```

Add this helper near the top of the module (after the existing imports):

```python
def _serialize_clusters(clusters):
    """Convert ClusterRecord dicts into the JSON-friendly schema."""
    out = []
    for tr in clusters:
        out.append({
            "id": int(tr["id"]),
            "born_t": round(float(tr["born_t"]), 2),
            "last_seen_t": round(float(tr["last_seen_t"]), 2),
            "initial_count": int(tr["initial_count"]),
            "peak_count": int(tr["peak_count"]),
            "final_count": int(tr["final_count"]),
            "net_growth": int(tr["final_count"] - tr["initial_count"]),
            "bbox_at_peak": [round(float(v), 4)
                             for v in tr["bbox_at_peak"]],
            "is_main": bool(tr.get("is_main", False)),
            "merged_into": tr.get("merged_into"),
            "merged_from": list(tr.get("merged_from", [])),
            "history_seconds": [round(float(t), 2)
                                for t in tr["history_seconds"]],
            "history_counts": [int(c) for c in tr["history_counts"]],
        })
    return out
```

- [ ] **Step 5: Add the two new perframe.csv columns**

Find the perframe CSV emission near line 516-531. The block currently looks like:

```python
        if piece_tracker is not None:
            cols += ["pieces_visible", "pieces_on_board"]
        ...
            if piece_tracker is not None:
                row += [int(pieces_visible[i]), int(pieces_on_board[i])]
```

After the piece_tracker column-add, append cluster columns:

```python
        if cluster_mapper is not None:
            cols += ["largest_cluster_size", "cluster_count"]
```

And in the per-row loop after the piece_tracker row-extend, add:

```python
            if cluster_mapper is not None:
                t_now = i / eff_fps
                row += _cluster_counts_at(cluster_history, t_now)
```

Add the helper near `_serialize_clusters`:

```python
def _cluster_counts_at(clusters, t_now):
    """Look up (largest_cluster_size, cluster_count) at time t_now.

    For each cluster, finds the last survey at or before t_now and uses
    its count. Clusters whose first survey is after t_now are skipped.
    """
    largest, active = 0, 0
    for tr in clusters:
        # find the last history index with time <= t_now
        hist_t = tr["history_seconds"]
        hist_c = tr["history_counts"]
        last_idx = -1
        for i in range(len(hist_t)):
            if hist_t[i] <= t_now:
                last_idx = i
            else:
                break
        if last_idx < 0:
            continue   # cluster hasn't appeared yet
        active += 1
        if hist_c[last_idx] > largest:
            largest = hist_c[last_idx]
    return [int(largest), int(active)]
```

- [ ] **Step 6: Add the CLI flag**

Find the existing `--no-pieces` flag near line 751:

```python
    ap.add_argument("--no-pieces", action="store_true",
                    help="skip visual piece tracking (faster pass 1)")
```

Add immediately after:

```python
    ap.add_argument("--no-clusters", action="store_true",
                    help="skip cluster detection / milestones (faster pass 1)")
```

Update the `analyze()` call at line 762-770 (search for `summary = analyze(...)`) to pass through the new flag — append `track_clusters=not a.no_clusters` to the kwargs:

```python
    summary = analyze(a.video, a.start, a.duration, a.fps, a.move_threshold,
                      a.finger_threshold, a.grip_threshold,
                      _parse_board(a.board),
                      a.segment_seconds, a.preview_seconds, a.swap_hands,
                      not a.no_video, a.outdir,
                      puzzle_name=a.puzzle_name, num_pieces=a.pieces,
                      difficulty=a.difficulty,
                      track_pieces=not a.no_pieces,
                      track_clusters=not a.no_clusters)
```

- [ ] **Step 7: Smoke test the import paths and signature**

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python -c "
import puzzle_hands
import inspect
sig = inspect.signature(puzzle_hands.analyze)
assert 'track_clusters' in sig.parameters
print('OK, parameters:', list(sig.parameters.keys()))
"
```

Expected: parameter list ends with `..., track_pieces, track_clusters`.

- [ ] **Step 8: Commit**

```bash
cd /home/jc/Python/PuzzleTracker
git add puzzle_hands.py
git commit -m "$(cat <<'EOF'
Wire ClusterMapper into puzzle_hands pipeline

- Build ClusterMapper next to PieceTracker (track_clusters=True default)
- Call cluster_mapper.update() in the per-frame loop
- At finalize: classify clusters, compute milestones, infer board
- Inferred board flows into piece_tracker so placed_on_board now works
  without manual marking
- metrics.json gains: board_source, cluster_summary, clusters, milestones
- perframe.csv gains: largest_cluster_size, cluster_count columns
- New CLI flag --no-clusters mirrors --no-pieces

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: GUI radio buttons in puzzle_app.py

**Why:** Surface auto-detect as the default in the desktop app and let users still draw a manual box when they want.

**Files:**
- Modify: `puzzle_app.py` (restructure step 2)

- [ ] **Step 1: Add the board_mode state variable**

In the `__init__` method of the main App class (near line 179 where `self.board = None`), add:

```python
        self.board = None
        self.board_mode = tk.StringVar(value="auto")    # "auto" | "manual"
```

- [ ] **Step 2: Restructure the step-2 UI block**

Find the existing step-2 block (lines 206-227 — search for `"2.  (Optional) Mark the puzzle board"`). Replace the entire block (from `ttk.Label(self.inner, text="2.  ...` through `self.btn_clear.pack(**pad)`) with:

```python
        ttk.Label(self.inner, text="2.  Board area (auto-detected by default)",
                  style="Step.TLabel").pack(fill="x", **pad)

        ttk.Radiobutton(
            self.inner,
            text="Auto-detect board",
            variable=self.board_mode, value="auto",
            command=self._on_board_mode_change,
        ).pack(anchor="w", padx=28)
        ttk.Label(
            self.inner,
            text="    Recommended if your board area changes during play\n"
                 "    (e.g. sort pile in the middle, then pushed aside).",
            style="Hint.TLabel", justify="left").pack(anchor="w", padx=28)

        ttk.Radiobutton(
            self.inner,
            text="Set puzzle board manually",
            variable=self.board_mode, value="manual",
            command=self._on_board_mode_change,
        ).pack(anchor="w", padx=28, pady=(6, 0))
        ttk.Label(
            self.inner,
            text="    Draw a rectangle on the preview to mark the assembly zone.",
            style="Hint.TLabel", justify="left").pack(anchor="w", padx=28)

        self.canvas = tk.Canvas(self.inner, width=PREVIEW_W, height=200,
                                bg="#222", highlightthickness=1,
                                highlightbackground="#666",
                                cursor="crosshair")
        self.canvas.pack(**pad)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover)
        self.lbl_board = ttk.Label(
            self.inner,
            text="Board: will be auto-detected from cluster footprint",
            style="Mono.TLabel")
        self.lbl_board.pack(**pad)
        self.btn_clear = ttk.Button(self.inner, text="Reset to auto-detect",
                                    command=self.clear_board, state="disabled")
        self.btn_clear.pack(**pad)
```

- [ ] **Step 3: Add the `_on_board_mode_change` handler**

Add this method in the App class (anywhere alongside `clear_board`):

```python
    def _on_board_mode_change(self):
        """Toggle canvas interactivity when the user switches modes."""
        if self.board_mode.get() == "auto":
            # Wipe any manual rectangle and disable drawing
            self.board = None
            self.canvas.delete("board")
            self.canvas.config(cursor="arrow")
            self.lbl_board.config(
                text="Board: will be auto-detected from cluster footprint")
            self.btn_clear.config(state="disabled")
        else:
            self.canvas.config(cursor="crosshair")
            self.lbl_board.config(text="Board: (not set - draw a rectangle)")
            self.btn_clear.config(state="normal")
```

- [ ] **Step 4: Guard the canvas mouse handlers against auto mode**

The existing `on_press`, `on_drag`, `on_release` methods (search for `def on_press` around line 385+) must early-return when the mode is "auto." Add at the top of each of those three methods:

```python
        if self.board_mode.get() == "auto":
            return
```

- [ ] **Step 5: Update `clear_board()` to flip the mode back to auto**

Find the existing `clear_board` method (around line 519). Replace its body so that "Clear" really means "go back to auto-detect":

```python
    def clear_board(self):
        """Reset to auto-detect mode."""
        self.board_mode.set("auto")
        self._on_board_mode_change()
```

- [ ] **Step 6: Smoke test the GUI module loads**

```bash
cd /home/jc/Python/PuzzleTracker && python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('puzzle_app',
    '/home/jc/Python/PuzzleTracker/puzzle_app.py')
mod = importlib.util.module_from_spec(spec)
# Just verify the module imports without running mainloop
try:
    spec.loader.exec_module(mod)
    print('OK: puzzle_app imports cleanly')
except SystemExit:
    print('OK: puzzle_app imports (exited mainloop)')
"
```

Expected: `OK: puzzle_app imports cleanly` (the module's `if __name__ == "__main__":` guard prevents the GUI from launching during import).

- [ ] **Step 7: Manual GUI smoke test (only if a display is available)**

If a display is available, launch the GUI and verify:
1. Step 2 shows "Board area (auto-detected by default)" with two radio buttons.
2. Auto is selected by default; the canvas cursor is an arrow.
3. Clicking the canvas in auto mode does nothing.
4. Selecting "Set puzzle board manually" changes the cursor to crosshair; click-drag draws a rectangle.
5. Clicking "Reset to auto-detect" flips the radio back to Auto and wipes the rectangle.

Run:
```bash
cd /home/jc/Python/PuzzleTracker && python puzzle_app.py
```

If no display: skip and rely on the import test above.

- [ ] **Step 8: Commit**

```bash
cd /home/jc/Python/PuzzleTracker
git add puzzle_app.py
git commit -m "$(cat <<'EOF'
GUI: add auto-detect / manual board mode radio buttons

Step 2 now defaults to "Auto-detect board" (recommended), with manual
4-click marking available as an alternative. Canvas is read-only in
auto mode; the "Clear board" button is now "Reset to auto-detect" and
flips the radio back instead of just clearing the rectangle.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: End-to-end smoke test + analyst schema docs

**Why:** Verify the full pipeline runs on a real video and produces a valid `metrics.json` with the new keys. Update the puzzle-analyst skill's data-schema reference so future analyses know about the new fields.

**Files:**
- Modify: `~/.claude/skills/puzzle-analyst/references/data-schema.md`
- (no code changes here — this is verification + docs)

- [ ] **Step 1: Run the analyzer on the existing RollingHills video**

Use a fresh output folder so the previous run isn't overwritten:

```bash
cd /home/jc/Python/PuzzleTracker && python puzzle_hands.py \
    /home/jc/Python/PuzzleTracker/RollingHills.mp4 \
    --outdir /home/jc/Python/PuzzleTracker/RollingHills_v3_analysis \
    --puzzle-name "Rolling Hills" --pieces 500 --difficulty Medium \
    --no-video 2>&1 | tail -30
```

Expected: analyzer completes without crash. Last lines should show `pass2`/`done` and write `RollingHills_metrics.json` + `RollingHills_perframe.csv` to the output dir.

- [ ] **Step 2: Verify the new JSON keys are populated**

```bash
python3 -c "
import json
d = json.load(open('/home/jc/Python/PuzzleTracker/RollingHills_v3_analysis/RollingHills_metrics.json'))
print('board_source:', d.get('board_source'))
print('board_region:', d.get('board_region'))
print('cluster_summary:', json.dumps(d.get('cluster_summary'), indent=2))
print('clusters tracked:', len(d.get('clusters', [])))
print('milestones:')
for m in d.get('milestones', []):
    print(' ', m.get('t'), m.get('type'), '-', m.get('label'))
print('piece_summary placed_on_board:', d.get('piece_summary', {}).get('placed_on_board'))
"
```

Expected output should include:
- `board_source: auto-detected` (the inferred board took effect)
- a `cluster_summary` dict with detection_quality and main_cluster_id populated
- a non-empty milestones list (at least `first_piece_joined` and likely `sort_pile_cleared`)
- `piece_summary.placed_on_board > 0` (since piece_tracker now has a board to test against)

If `board_source: none` or no milestones fire, investigate before continuing — likely the cluster_eps or stillness SAD threshold needs tuning. Likely tuning knobs in `puzzle_clusters.py`:
- `cluster_eps` (default 0.06): raise to 0.08-0.10 if real puzzles produce small disconnected sub-clusters.
- SAD threshold in `_is_stationary` (currently 8.0): raise if stationary pieces are being rejected; lower if too much hand motion leaks through.

- [ ] **Step 3: Verify the new perframe.csv columns**

```bash
head -1 /home/jc/Python/PuzzleTracker/RollingHills_v3_analysis/RollingHills_perframe.csv
```

Expected: the header line should end with `,pieces_visible,pieces_on_board,largest_cluster_size,cluster_count`.

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('/home/jc/Python/PuzzleTracker/RollingHills_v3_analysis/RollingHills_perframe.csv')
print('largest_cluster_size: min %d, max %d, last %d'
      % (df.largest_cluster_size.min(), df.largest_cluster_size.max(), df.largest_cluster_size.iloc[-1]))
print('cluster_count: min %d, max %d'
      % (df.cluster_count.min(), df.cluster_count.max()))
"
```

Expected: largest_cluster_size grows from 0 to something larger than 0 by the end (cluster mapper actually saw clusters).

- [ ] **Step 4: Update the puzzle-analyst schema reference**

Edit `~/.claude/skills/puzzle-analyst/references/data-schema.md`. Add this section after the existing `phases` section (around the line "Useful for: how often the puzzler is bimanual vs. one-handed..."):

```markdown
---

### `board_source` (new in v0.3)

String describing how `board_region` was determined:

| Value | Meaning |
|---|---|
| `"manual"` | The user clicked 4 corners in the GUI before analysis |
| `"auto-detected"` | Discovered by the cluster mapper from where stationary pieces accumulated. Slightly less precise than a hand-drawn rectangle but works without any pre-analysis input |
| `"none"` | No board could be inferred (puzzler only sorted, or detection quality was too low). `board_region` is null in this case |

If `board_source` is missing from the JSON, the file is from before this feature shipped — treat as `"none"`.

### `cluster_summary` (new in v0.3)

| Key | Meaning |
|---|---|
| `total_clusters_observed` | Total tracks before pruning |
| `total_clusters_kept` | After dropping tracks with peak_count < 3 |
| `main_cluster_id` | ID of the assembly cluster (largest positive net growth), or null |
| `main_cluster_peak_count` | Peak member count of the main cluster |
| `main_cluster_net_growth` | final_count - initial_count of the main cluster |
| `merge_events` | Count of qualifying island-merge events |
| `survey_interval_s` | Seconds between cluster surveys (default 5.0) |
| `tiebreak_used` | True if aspect-ratio tiebreak was used to pick the main cluster |
| `detection_quality` | "high" / "medium" / "low" — see below |
| `quality_notes` | List of human-readable caveats |

`detection_quality` is `"low"` when hand detection is poor or no real assembly was observed; `"medium"` when a tiebreak was used or hand detection was borderline; `"high"` otherwise. The analyst skill should footnote milestones whenever `detection_quality != "high"`.

### `clusters` (new in v0.3)

List of per-cluster lifecycles. Each entry:

| Key | Meaning |
|---|---|
| `id` | Stable across the session |
| `born_t` | First survey time the cluster appeared |
| `last_seen_t` | Most recent survey time the cluster was detected |
| `initial_count` / `peak_count` / `final_count` | Member counts at born / max / end |
| `net_growth` | `final_count - initial_count` (positive = assembling, negative = sort pile) |
| `bbox_at_peak` | `[x1, y1, x2, y2]` of the cluster at peak count |
| `is_main` | True for the assembly cluster (one per session, or none if `board_source == "none"`) |
| `merged_into` / `merged_from` | Cluster IDs involved in merges |
| `history_seconds` / `history_counts` | Parallel arrays — count over time |

### `milestones` (new in v0.3)

Chronological list of detected assembly milestones. Each entry has `t`, `type`, `label`, `cluster_id`, `confidence` (`"high"` or `"medium"`), and optional `details`. Types:

| Type | Triggers |
|---|---|
| `first_piece_joined` | Main cluster reached size 2 |
| `cluster_25pct` / `cluster_50pct` / `cluster_75pct` | Main cluster crossed 25/50/75% of `num_pieces`. Only emitted if `num_pieces` was supplied at analysis time |
| `frame_complete` | The puzzle's outer ring appears assembled (heuristic — medium confidence) |
| `islands_merged` | Two clusters that each had peak_count ≥ 10 merged |
| `sort_pile_cleared` | The largest negative-growth cluster dropped below 20% of peak (or below 20 pieces) |

When reporting milestones to the puzzler, group them in chronological order and show timestamps as mm:ss. Frame-complete and sort-pile-cleared are medium confidence — footnote them, especially if `detection_quality != "high"`.

### perframe.csv — new columns (when clusters are enabled)

| Column | Meaning |
|---|---|
| `largest_cluster_size` | Member count of the biggest cluster at that frame (carried forward between 5s surveys, 0 before first detection) |
| `cluster_count` | Number of active (non-dormant) clusters at that frame |

These are useful for cluster-growth plots and for detecting moments when many clusters existed simultaneously (lots of small islands).
```

- [ ] **Step 5: Commit the schema docs update**

The puzzle-analyst skill lives outside the project repo, so:

```bash
cd /home/jc/.claude/skills/puzzle-analyst && git status
```

If it's a git repo, commit there:

```bash
cd /home/jc/.claude/skills/puzzle-analyst && \
    git add references/data-schema.md && \
    git commit -m "Document v0.3 schema additions: board_source, clusters, milestones"
```

If not a git repo (single-user skill folder), just save the file — no commit needed. The file path itself is `~/.claude/skills/puzzle-analyst/references/data-schema.md`.

- [ ] **Step 6: Final smoke - run the unit tests once more**

```bash
cd /home/jc/Python/PuzzleTracker && python -m unittest discover tests -v
```

Expected: all tests pass (4 vision_utils + 17 cluster tests = 21 total).

- [ ] **Step 7: Final commit (if any docs changes in the project repo)**

Nothing more to commit in the project repo for this task — all the verification was read-only and the schema docs live in the skill folder. If there are stray modifications, run `git status` to see them and commit if appropriate.

```bash
cd /home/jc/Python/PuzzleTracker && git status
```

---

## What's not in this plan (deliberately deferred)

- **Per-piece identity tracking** (phase 2). The spec mentions this as "deferred to a later phase." Skip until the puzzler asks for per-piece narratives.
- **Debug artifact `*_clusters_debug.png`** (mentioned in spec section 8d). The spec includes a `--debug-clusters` flag but this plan doesn't implement it — add later if the cluster pipeline misbehaves on real data and we want a diagnostic.
- **Analyst-skill chart additions** (cluster-growth-over-time plot, milestone-on-pace overlay). The analyst skill will pick up the new keys automatically via dict.get(), but new chart helpers are out of scope here.
- **Multi-session aggregation of milestones** (e.g. "your frame-complete time has improved across the last 5 sessions"). Defer until enough sessions exist for the comparison to be meaningful.
