#!/usr/bin/env python3
"""
puzzle_pieces.py - visual puzzle-piece tracking.

Runs alongside the hand pipeline in puzzle_hands.py. For each processed
frame it:

  1. Builds a foreground mask via KNN background subtraction (slow
     learning, so a piece becomes background only after sitting still for
     ~30s).
  2. Erases pixels inside the convex hull of each detected hand so the
     hand itself is never counted as a piece.
  3. Cleans the mask (open + close), finds piece-sized contours, and
     records each blob's normalized centroid + an HSV color histogram.
  4. Matches detections to existing tracks by centroid distance weighted
     by histogram similarity. New unmatched detections start new tracks
     with fresh IDs.

After the full pass, `finalize()` links each track's first appearance to
the nearest hand grip-release (best guess for who placed it) and each
track's last appearance to the nearest hand grip-onset (best guess for
who picked it back up). Pieces whose final position falls inside the
board rectangle are flagged `placed_on_board`.

This is a heuristic, not ground truth. Dense piles may show up as one
merged blob; lighting / occlusion can split a single piece across two
IDs. Treat `pieces` as best-effort and `piece_summary` counts as
approximate.
"""

import numpy as np
import cv2

from puzzle_vision_utils import hand_mask as _hand_mask_fn


# ----------------------------------------------------------------- helpers
def _color_hist(roi):
    """HSV (H,S) 2D histogram of a piece's bounding-box ROI."""
    if roi is None or roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


class _Track:
    """One piece's life: first/last frame, centroid trajectory endpoints,
    color signature, and how many frames it was actually detected on."""

    __slots__ = ("id", "first_frame", "last_frame", "first_t", "last_t",
                 "first_xy", "last_xy", "area_norm", "hist",
                 "frames_visible", "active")

    def __init__(self, pid, frame_idx, t, xy, area_norm, hist):
        self.id = pid
        self.first_frame = self.last_frame = frame_idx
        self.first_t = self.last_t = t
        self.first_xy = self.last_xy = xy
        self.area_norm = area_norm
        self.hist = hist
        self.frames_visible = 1
        self.active = True


# -------------------------------------------------------------- the tracker
class PieceTracker:
    """Maintains stable piece tracks across the processed frames of a video."""

    def __init__(self, W, H, board=None, eff_fps=15.0,
                 min_frames=4, match_radius=0.07, max_gap=8,
                 area_min_frac=0.0004, area_max_frac=0.012):
        self.W, self.H = int(W), int(H)
        self.board = board
        self.eff_fps = float(eff_fps)
        self.min_frames = min_frames
        self.match_radius = match_radius
        self.max_gap = max_gap
        self.area_min = area_min_frac * W * H
        self.area_max = area_max_frac * W * H

        # KNN was chosen over MOG2: it produces cleaner masks on textured
        # tabletops with small moving objects, and the slow learning rate
        # keeps stationary pieces visible long enough to track.
        self.bg = cv2.createBackgroundSubtractorKNN(
            history=400, dist2Threshold=350.0, detectShadows=False)

        self.tracks = {}
        self.next_id = 1

        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5))
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (11, 11))

    # -------- per-frame ----------------------------------------------------
    def update(self, frame, hands_pts, frame_idx, t):
        """Run detection + tracking on one BGR frame."""
        fg = self.bg.apply(frame, learningRate=0.005)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        hm = _hand_mask_fn(self.W, self.H, hands_pts)
        fg[hm > 0] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._open_kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._close_kernel)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.area_min or a > self.area_max:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx_n = (x + w / 2.0) / self.W
            cy_n = (y + h / 2.0) / self.H
            hist = _color_hist(frame[y:y + h, x:x + w])
            if hist is None:
                continue
            dets.append({"xy": (cx_n, cy_n),
                         "area_norm": a / (self.W * self.H),
                         "hist": hist})

        # greedy assignment: cheapest (centroid distance, histogram-weighted)
        # matches first, each track and detection used at most once
        matched_t = set()
        matched_d = set()
        cands = []
        for tid, tr in self.tracks.items():
            if not tr.active:
                continue
            if frame_idx - tr.last_frame > self.max_gap:
                tr.active = False
                continue
            for di, d in enumerate(dets):
                dx = d["xy"][0] - tr.last_xy[0]
                dy = d["xy"][1] - tr.last_xy[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self.match_radius:
                    continue
                sim = cv2.compareHist(tr.hist, d["hist"], cv2.HISTCMP_CORREL)
                cost = dist * (1.5 - max(0.0, sim))
                cands.append((cost, tid, di))
        cands.sort()
        for _cost, tid, di in cands:
            if tid in matched_t or di in matched_d:
                continue
            tr = self.tracks[tid]
            d = dets[di]
            tr.last_frame = frame_idx
            tr.last_t = t
            tr.last_xy = d["xy"]
            tr.frames_visible += 1
            tr.hist = cv2.addWeighted(tr.hist, 0.85, d["hist"], 0.15, 0)
            matched_t.add(tid)
            matched_d.add(di)

        for di, d in enumerate(dets):
            if di in matched_d:
                continue
            pid = self.next_id
            self.next_id += 1
            self.tracks[pid] = _Track(pid, frame_idx, t,
                                      d["xy"], d["area_norm"], d["hist"])

    # -------- post-processing ---------------------------------------------
    def _in_board(self, xy):
        if self.board is None:
            return False
        x1, y1, x2, y2 = self.board
        return x1 <= xy[0] <= x2 and y1 <= xy[1] <= y2

    @staticmethod
    def _nearest_event(events, frame_idx, xy, max_gap, dist_radius):
        """Pick the event closest in (time, distance) within the gates.
        Returns (cost, (frame, t, (x,y))) or (None, None)."""
        best = None
        best_cost = None
        for f, t, exy in events:
            df = abs(f - frame_idx)
            if df > max_gap:
                continue
            d = ((exy[0] - xy[0]) ** 2 + (exy[1] - xy[1]) ** 2) ** 0.5
            if d > dist_radius:
                continue
            cost = d + 0.3 * (df / max(max_gap, 1))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = (f, t, exy)
        return best_cost, best

    def finalize(self, n_frames, grip_events):
        """Build the per-piece summary plus per-frame visible counts.

        grip_events:
            {"right": {"onset":  [(frame, t, (x,y)), ...],
                       "offset": [(frame, t, (x,y)), ...]},
             "left":  {...}}

        Returns:
            pieces        - list of dicts (one per kept track)
            visible       - np.int array, count of tracks visible per frame
            on_board      - np.int array, count of tracks currently in board
        """
        visible = np.zeros(n_frames, dtype=int)
        on_board = np.zeros(n_frames, dtype=int)

        keepers = [tr for tr in self.tracks.values()
                   if tr.frames_visible >= self.min_frames]
        keepers.sort(key=lambda tr: tr.first_frame)
        # renumber by first-appearance order so JSON ids read 1..N in time
        for new_id, tr in enumerate(keepers, 1):
            tr.id = new_id

        gap_window = max(1, int(2.0 * self.eff_fps))  # ±2s
        dist_radius = 0.12                            # ~12% of frame width

        pieces = []
        for tr in keepers:
            f_lo = max(0, tr.first_frame)
            f_hi = min(tr.last_frame + 1, n_frames)
            visible[f_lo:f_hi] += 1
            placed = self._in_board(tr.last_xy)
            if placed:
                # cumulative count: once a piece's track ends on the board,
                # treat it as "placed" from its first appearance there
                # onward, and keep counting it as on-board until the end
                # of the session (KNN merges stationary pieces into the
                # background, but the piece itself is still physically there)
                on_board[f_lo:n_frames] += 1

            placement_hand = None
            best_place_cost = None
            for side in ("right", "left"):
                cost, ev = self._nearest_event(
                    grip_events[side]["offset"],
                    tr.first_frame, tr.first_xy, gap_window, dist_radius)
                if ev is not None and (best_place_cost is None
                                       or cost < best_place_cost):
                    best_place_cost = cost
                    placement_hand = (side, ev)

            pickup_hand = None
            best_pick_cost = None
            if not placed:
                for side in ("right", "left"):
                    cost, ev = self._nearest_event(
                        grip_events[side]["onset"],
                        tr.last_frame, tr.last_xy, gap_window, dist_radius)
                    if ev is not None and (best_pick_cost is None
                                           or cost < best_pick_cost):
                        best_pick_cost = cost
                        pickup_hand = (side, ev)

            pieces.append({
                "id": tr.id,
                "first_seen_t": round(tr.first_t, 2),
                "last_seen_t": round(tr.last_t, 2),
                "first_xy": [round(tr.first_xy[0], 3),
                             round(tr.first_xy[1], 3)],
                "last_xy": [round(tr.last_xy[0], 3),
                            round(tr.last_xy[1], 3)],
                "frames_visible": int(tr.frames_visible),
                "duration_s": round(tr.last_t - tr.first_t, 2),
                "area_norm": round(float(tr.area_norm), 5),
                "placed_on_board": bool(placed),
                "placed_by_hand": placement_hand[0] if placement_hand else None,
                "placed_at_t": (round(placement_hand[1][1], 2)
                                if placement_hand else None),
                "picked_up_by_hand": pickup_hand[0] if pickup_hand else None,
                "picked_up_at_t": (round(pickup_hand[1][1], 2)
                                   if pickup_hand else None),
            })

        return pieces, visible, on_board

    # -------- session-level summary ---------------------------------------
    @staticmethod
    def summarize(pieces, dur_s):
        if not pieces:
            return {
                "total_tracks": 0,
                "placed_on_board": 0,
                "off_board_final": 0,
                "placements_per_min": 0.0,
                "mean_visible_duration_s": None,
                "mean_time_to_place_s": None,
            }
        placed = [p for p in pieces if p["placed_on_board"]]
        durs = [p["duration_s"] for p in pieces if p["duration_s"] > 0]
        place_times = [p["placed_at_t"] for p in placed
                       if p.get("placed_at_t") is not None]
        # rough "time to place" = time from first session frame to placement
        # event; only meaningful relative to other pieces in the session
        return {
            "total_tracks": len(pieces),
            "placed_on_board": len(placed),
            "off_board_final": len(pieces) - len(placed),
            "placements_per_min": (round(len(placed) / dur_s * 60.0, 2)
                                   if dur_s > 0 else 0.0),
            "mean_visible_duration_s": (round(float(np.mean(durs)), 2)
                                        if durs else None),
            "mean_placement_time_s": (round(float(np.mean(place_times)), 2)
                                      if place_times else None),
        }
