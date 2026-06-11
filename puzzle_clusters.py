"""puzzle_clusters.py - dynamic board detection + assembly milestones.

Runs alongside the hand pipeline in puzzle_hands.py. Every
`survey_interval_s` seconds (default 5s) it:

  1. Builds a stationary-blob list from the current frame using edge
     detection + contour finding + a survey-interval stillness gate.
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


ASSEMBLY_MIN_COUNT = 3        # pieces in the main cluster to call it "assembling"
ASSEMBLY_SUSTAIN = 2          # surveys of continued growth required


def assembly_onset(history_seconds, history_counts,
                   min_count=ASSEMBLY_MIN_COUNT, sustain=ASSEMBLY_SUSTAIN):
    """Timestamp where the main assembly cluster begins SUSTAINED growth.

    Returns (t, confidence, note). t is seconds, or None if no sustained
    growth is found. confidence is 'high' (clean near-empty pre-phase then
    growth), 'medium' (already growing at first survey, or gradual), or
    'unavailable' (no onset). Pure function: operates only on the parallel
    history arrays of the main cluster.
    """
    n = len(history_counts)
    if n == 0:
        return None, "unavailable", "no cluster history"
    for i in range(n):
        if history_counts[i] < min_count:
            continue
        end = min(n, i + sustain + 1)
        window = history_counts[i:end]
        grows = (len(window) >= 2 and window[-1] > window[0] and
                 all(window[k + 1] >= window[k] for k in range(len(window) - 1)))
        if not grows:
            continue
        pre_clean = all(c < min_count for c in history_counts[:i])
        conf = "high" if (i > 0 and pre_clean) else "medium"
        note = ("clean flip-phase boundary" if conf == "high"
                else "assembly already underway at first survey")
        return round(float(history_seconds[i]), 1), conf, note
    return None, "unavailable", "no sustained growth detected"


def placement_splits(history_seconds, history_counts, num_pieces,
                     fractions=(0.25, 0.50, 0.75, 1.0)):
    """First timestamp the assembled count reaches each fraction of
    num_pieces. Returns {'25pct': t_or_None, ...}, or {} if num_pieces is
    falsy."""
    if not num_pieces:
        return {}
    out = {}
    for fr in fractions:
        target = fr * num_pieces
        hit = None
        for ts, c in zip(history_seconds, history_counts):
            if c >= target:
                hit = round(float(ts), 1)
                break
        out["%dpct" % int(round(fr * 100))] = hit
    return out


STALL_MIN_GAP_S = 30.0


def detect_stalls(history_seconds, history_counts, min_gap_s=STALL_MIN_GAP_S):
    """Flat spots in the assembled-count curve: maximal spans during which
    the count never exceeds its value at the span start, lasting longer than
    min_gap_s. Returns a list of
    {start_t, duration_s, count_at_stall} dicts.

    A span is measured relative to the count at its start, so a dip-and-recover
    back to the start value still counts as a stall (count_at_stall is that
    start value)."""
    stalls = []
    n = len(history_counts)
    i = 0
    while i < n - 1:
        j = i
        while j + 1 < n and history_counts[j + 1] <= history_counts[i]:
            j += 1
        if j > i:
            gap = history_seconds[j] - history_seconds[i]
            if gap >= min_gap_s:
                stalls.append({
                    "start_t": round(float(history_seconds[i]), 1),
                    "duration_s": round(float(gap), 1),
                    "count_at_stall": int(history_counts[i]),
                })
            i = j
        else:
            i += 1
    return stalls


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

    NET_GROWTH_MIN = 5             # cluster must grow by this much to qualify
    PRUNE_PEAK_BELOW = 3           # drop tracks whose peak_count < this
    PUZZLE_AR_MIN = 0.5            # typical puzzle aspect ratios 1:2 .. 2:1
    PUZZLE_AR_MAX = 2.0
    TIEBREAK_BAND = 0.9            # relative growth threshold for tiebreak candidates

    # Milestone thresholds
    FRAME_CLUSTER_SIZE_MIN_FRAC = 0.80
    FRAME_CLUSTER_SIZE_MAX_FRAC = 1.30
    FRAME_PERIMETER_FRACTION = 0.85
    ISLAND_MIN_PEAK = 10
    PILE_CLEARED_PEAK_FRAC = 0.20
    PILE_CLEARED_ABS_MIN = 20

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
        self.last_summary = {}           # populated by finalize(); safe to read anytime
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
        # A blob that was absent last survey (dark background) counts as
        # newly placed — accept it rather than treating it as a moving hand.
        if roi_prev.mean() < 8.0:
            return True
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
            tr.setdefault("history_members", []).append(c["members"])
            if c["count"] > tr["peak_count"]:
                tr["peak_count"] = c["count"]
                tr["bbox_at_peak"] = c["bbox"]
            tr["final_count"] = c["count"]
            tr["consecutive_misses"] = 0
            tr["dormant"] = False

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
                "history_members": [c["members"]],
                "consecutive_misses": 0,
                "dormant": False,
                "merged_into": None,
                "merged_from": [],
            }
            self.events.append(("born", t, {"cluster_id": tid}))

    # ---- finalize (Tasks 3, 4) ------------------------------------------
    def finalize(self, n_frames):
        """Classify clusters, prune noise, populate is_main, return
        (clusters_list, milestones_list). Milestones are computed by
        _compute_milestones from the finalized track history."""
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
            threshold = max(self.TIEBREAK_BAND * top_growth, top_growth - 10)
            top = [tr for tr in candidates if tr["net_growth"] >= threshold]
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

        milestones = self._compute_milestones(kept, main)
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
                if len(members) < 4:
                    continue
                ms_arr = np.array(members)
                # Median gap between unique x (or y) positions ≈ one piece-width
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
