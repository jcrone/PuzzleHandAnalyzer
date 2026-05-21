#!/usr/bin/env python3
"""
puzzle_hands.py - Hand & finger activity analysis for speed-puzzling video.

Built to run on full-length sessions (45+ min). Video is streamed frame by
frame, so memory stays flat regardless of length - only lightweight landmark
data is kept, never the decoded frames.

WHAT IT MEASURES
  GROSS MOVEMENT  per-hand travel across the table
  FINGER WORK     finger articulation (isolated from travel) + grip cycles
  PACE & RHYTHM   piece-handling rate, grip hold time, tempo steadiness
  BIMANUAL        how the two hands share / parallelise the work
  SEGMENTED       all of the above broken into time segments
  FATIGUE TREND   how pace / tempo drift from start to end of the session
  SORT vs ASSEMBLE  (needs --board) time at the pile vs. on the board,
                  trips between zones, and zone-aware pickups / placements

HANDEDNESS is derived from each hand's own geometry, so it is consistent
regardless of camera mirroring. Check *_calibration.jpg (it has a coordinate
grid - use it to read off your --board rectangle). If reversed, --swap-hands.

USAGE
  python puzzle_hands.py VIDEO.mp4
  python puzzle_hands.py VIDEO.mp4 --board 0.42,0.20,0.92,0.78
  python puzzle_hands.py VIDEO.mp4 --segment-seconds 60 --no-video
  python puzzle_hands.py VIDEO.mp4 --fps 8           # faster first pass
  python puzzle_hands.py VIDEO.mp4 --preview-seconds 120   # limit video out

Requires: mediapipe>=0.10.21, opencv-python-headless, numpy, matplotlib
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

import puzzle_pieces
import puzzle_report

HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "hand_landmarker.task")

TIPS = [4, 8, 12, 16, 20]
MCPS = [0, 5, 9, 13, 17]
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
               (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
               (0, 17)]


# ----------------------------------------------------------- small helpers
def onsets(mask):
    m = np.asarray(mask)
    return np.where(m[1:] & ~m[:-1])[0] + 1


def offsets(mask):
    m = np.asarray(mask)
    return np.where(~m[1:] & m[:-1])[0] + 1


def runs(mask):
    m = np.asarray(mask).astype(int)
    dd = np.diff(m, prepend=0, append=0)
    return list(zip(np.where(dd == 1)[0], np.where(dd == -1)[0]))


def trend_label(first, last):
    if first is None or last is None or first <= 1e-6:
        return "n/a", None
    pct = (last - first) / first * 100.0
    if pct <= -15:
        word = "declining"
    elif pct >= 15:
        word = "rising"
    else:
        word = "stable"
    return word, round(pct, 1)


# --------------------------------------------------------------- handedness
def geometric_handedness(lm):
    """'left' / 'right' / None from 21 normalized (x, y) landmarks."""
    w, mid = np.array(lm[0]), np.array(lm[9])
    idx, pky = np.array(lm[5]), np.array(lm[17])
    fwd, across = mid - w, idx - pky
    cross = fwd[0] * across[1] - fwd[1] * across[0]
    if abs(cross) < 1e-6:
        return None
    return "right" if cross < 0 else "left"


# ------------------------------------------------------------ hand features
def hand_features(lm):
    """Pose-independent features from one hand's 21 landmarks."""
    p = np.array(lm)
    w, mid = p[0], p[9]
    fwd = mid - w
    L = float(np.linalg.norm(fwd))
    if L < 1e-6:
        return None
    u = fwd / L
    v = np.array([-u[1], u[0]])
    feat = []
    for t in TIPS:
        rel = p[t] - w
        feat += [float(np.dot(rel, u)) / L, float(np.dot(rel, v)) / L]
    palm = p[MCPS].mean(axis=0)
    openness = float(np.mean([np.linalg.norm(p[t] - palm) for t in TIPS])) / L
    pinch = float(np.linalg.norm(p[4] - p[8])) / L
    centroid = (float(p[:, 0].mean()), float(p[:, 1].mean()))
    return {"c": centroid, "L": L, "feat": np.array(feat),
            "pinch": pinch, "openness": openness, "pts": p.tolist()}


# ----------------------------------------------------------------- analysis
def analyze(video, start, duration, proc_fps, move_thresh, finger_thresh,
            grip_thresh, board, seg_seconds, preview_seconds, swap,
            want_video, outdir, puzzle_name=None, num_pieces=None,
            difficulty=None, track_pieces=True):
    name = os.path.splitext(os.path.basename(video))[0]
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: %s" % video)
    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Phone VFR recordings sometimes report nonsense (e.g. 1000.0 or 90000.0)
    # in container metadata. Clamp so step/eff_fps stay sane.
    native_fps = max(5.0, min(raw_fps, 120.0))
    if abs(native_fps - raw_fps) > 0.01:
        print("Warning: source reports %.2f fps, clamped to %.2f"
              % (raw_fps, native_fps))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, round(native_fps / proc_fps))
    eff_fps = native_fps / step
    start_f = int(start * native_fps)
    end_f = total if duration is None else min(total, start_f + int(duration * native_fps))
    print("%s: native %.1ffps -> processing %.1ffps, frames %d-%d (%.1f min)"
          % (name, native_fps, eff_fps, start_f, end_f,
             (end_f - start_f) / native_fps / 60.0))

    if not os.path.exists(HAND_MODEL_PATH):
        sys.exit("Missing hand_landmarker.task next to puzzle_hands.py - "
                 "re-download the release zip or run "
                 "`curl -L -o hand_landmarker.task https://storage.googleapis.com"
                 "/mediapipe-models/hand_landmarker/hand_landmarker/float16/1"
                 "/hand_landmarker.task` in the project folder.")
    landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))

    # ---- PASS 1: stream, detect, keep only lightweight landmark data -------
    expected_N = max(1, (end_f - start_f) // step)
    dets_per_frame = []
    bg_frame, bg_idx = None, -1
    piece_tracker = (puzzle_pieces.PieceTracker(W, H, board=board,
                                                eff_fps=eff_fps)
                     if track_pieces else None)
    t0 = time.time()
    fi = start_f
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    while fi < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - start_f) % step == 0:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts_ms = int((fi - start_f) * 1000 / native_fps)
            res = landmarker.detect_for_video(mp_image, ts_ms)
            dets = []
            hands_pts_frame = []
            if res.hand_landmarks:
                for hlm in res.hand_landmarks:
                    pts = [(p.x, p.y) for p in hlm]
                    feats = hand_features(pts)
                    if feats is None:
                        continue
                    feats["hand"] = geometric_handedness(pts)
                    dets.append(feats)
                    hands_pts_frame.append(pts)
            j = len(dets_per_frame)
            dets_per_frame.append(dets)
            if piece_tracker is not None:
                piece_tracker.update(frame, hands_pts_frame, j, j / eff_fps)
            if bg_idx < 0 and len(dets) == 2:        # first 2-hand frame
                bg_frame, bg_idx = frame.copy(), j
            if j % 200 == 0 and j:
                elapsed = time.time() - t0
                rate = j / elapsed
                eta = max(0, (expected_N - j) / rate) if rate > 0 else 0
                print("  pass1: %d/%d frames (%ds elapsed, ~%ds remaining)"
                      % (j, expected_N, int(elapsed), int(eta)))
                # structured line for the GUI to parse (fraction + eta seconds)
                print("__PROGRESS__ %.4f %d"
                      % (j / expected_N, int(eta)), flush=True)
        fi += 1
    print("__PROGRESS__ 1.0 0", flush=True)
    print("__STAGE__ Generating reports and charts", flush=True)
    cap.release()
    landmarker.close()
    N = len(dets_per_frame)
    if bg_frame is None and N:                       # fallback bg frame
        bg_idx = N // 2
    print("analyzed %d frames in %.0fs" % (N, time.time() - t0))
    if N < 2:
        raise RuntimeError("Not enough frames analyzed. Try a longer or "
                           "more readable video.")

    # ---- assign detections to a stable left / right track ------------------
    rec = {"left": [None] * N, "right": [None] * N}
    last = {"left": None, "right": None}

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    for i, dets in enumerate(dets_per_frame):
        chosen = {}
        hl = [x for x in dets if x["hand"]]
        if len(dets) == 2 and len(hl) == 2 and hl[0]["hand"] != hl[1]["hand"]:
            for x in dets:
                chosen[x["hand"]] = x
        else:
            for x in dets:
                cands = [s for s in ("left", "right")
                         if s not in chosen and last[s] is not None]
                if not cands:
                    cands = [s for s in ("left", "right") if s not in chosen]
                if not cands:
                    continue
                if x["hand"] in cands:
                    side = x["hand"]
                else:
                    # Only tie-break by distance for sides we've actually seen;
                    # if neither has a known last position, take the first cand.
                    with_last = [s for s in cands if last[s] is not None]
                    if with_last:
                        side = min(with_last,
                                   key=lambda s: dist(x["c"], last[s]))
                    else:
                        side = cands[0]
                chosen[side] = x
        for side, x in chosen.items():
            rec[side][i] = x
            last[side] = x["c"]

    if swap:
        rec = {"left": rec["right"], "right": rec["left"]}

    # ---- per-hand time series ----------------------------------------------
    def fill(arr):
        a = np.array(arr, dtype=float)
        ix = np.arange(len(a))
        good = ~np.isnan(a)
        if good.sum() >= 2:
            a[~good] = np.interp(ix[~good], ix[good], a[good])
        elif good.sum() == 1:
            a[:] = a[good][0]
        return a

    def smooth(a, k=5):
        k = min(k, max(1, len(a)))
        return np.convolve(a, np.ones(k) / k, "same")

    series = {}
    for s in ("left", "right"):
        r = rec[s]
        det = np.array([x is not None for x in r])
        cx = smooth(fill([x["c"][0] if x else np.nan for x in r]))
        cy = smooth(fill([x["c"][1] if x else np.nan for x in r]))
        pinch = smooth(fill([x["pinch"] if x else np.nan for x in r]), 3)
        openn = smooth(fill([x["openness"] if x else np.nan for x in r]), 3)
        feat = np.array([x["feat"] if x else [np.nan] * 10 for x in r])
        for c in range(feat.shape[1]):
            feat[:, c] = fill(feat[:, c])
        artic = np.linalg.norm(np.diff(feat, axis=0, prepend=feat[:1]),
                               axis=1) * eff_fps
        artic = smooth(artic, 3)
        vx = np.diff(cx, prepend=cx[0]) * eff_fps
        vy = np.diff(cy, prepend=cy[0]) * eff_fps
        spd = np.sqrt(vx ** 2 + vy ** 2)
        spd[0], spd[-1] = spd[1], spd[-2]
        artic[0], artic[-1] = artic[1], artic[-2]
        moving = spd > move_thresh
        fingering = artic > finger_thresh
        gripping = pinch < grip_thresh
        in_board = None
        if board is not None:
            x1, y1, x2, y2 = board
            in_board = ((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2))
        series[s] = dict(det=det, cx=cx, cy=cy, spd=spd, artic=artic,
                         pinch=pinch, openness=openn, moving=moving,
                         fingering=fingering, gripping=gripping,
                         in_board=in_board)

    dur_s = N / eff_fps

    # ---- metric computation for an index slice -----------------------------
    def slice_metrics(s, a, b):
        S = series[s]
        mv, fg, gr = S["moving"][a:b], S["fingering"][a:b], S["gripping"][a:b]
        secs = max((b - a) / eff_fps, 1e-6)
        gc = len(onsets(gr))
        on = onsets(gr)
        m = {
            "moving_pct": round(100 * mv.mean(), 1),
            "finger_active_pct": round(100 * fg.mean(), 1),
            "grip_cycles": gc,
            "grips_per_min": round(gc / secs * 60, 1),
        }
        if len(on) >= 2:
            iv = np.diff(on) / eff_fps
            m["median_grip_interval_s"] = round(float(np.median(iv)), 2)
            m["grip_rhythm_cv"] = round(float(np.std(iv) / max(np.mean(iv), 1e-6)), 2)
        else:
            m["median_grip_interval_s"] = None
            m["grip_rhythm_cv"] = None
        return m

    def hand_block(s):
        S = series[s]
        m = slice_metrics(s, 0, N)
        m["detected_pct"] = round(100 * S["det"].mean(), 1)
        m["movement_events"] = len(onsets(S["moving"]))
        m["movements_per_min"] = round(m["movement_events"] / dur_s * 60, 1)
        m["path_length"] = round(float(np.sum(S["spd"]) / eff_fps), 2)
        m["finger_motion_index"] = round(float(S["artic"].mean()), 3)
        m["mean_openness"] = round(float(S["openness"].mean()), 3)
        m["active_pct"] = round(100 * np.mean(S["moving"] | S["fingering"]), 1)
        holds = [(e - st) / eff_fps for st, e in runs(S["gripping"])]
        m["mean_grip_hold_s"] = round(float(np.mean(holds)), 2) if holds else None
        if S["in_board"] is not None:
            ib = S["in_board"]
            m["assemble_pct"] = round(100 * ib.mean(), 1)
            m["sort_pct"] = round(100 * (1 - ib.mean()), 1)
            m["zone_trips"] = len(onsets(ib))            # pile -> board
            rel = offsets(S["gripping"])
            on = onsets(S["gripping"])
            m["placements"] = int(np.sum(ib[rel])) if len(rel) else 0
            m["pickups"] = int(np.sum(~ib[on])) if len(on) else 0
        return m

    per_hand = {s: hand_block(s) for s in ("left", "right")}

    # ---- piece tracks ------------------------------------------------------
    # Pieces are linked to a hand by the nearest grip-event in time + space:
    # grip-releases attribute placements, grip-onsets attribute pickups.
    pieces_list = []
    pieces_visible = np.zeros(N, dtype=int)
    pieces_on_board = np.zeros(N, dtype=int)
    piece_summary = None
    if piece_tracker is not None:
        grip_events = {"right": {"onset": [], "offset": []},
                       "left":  {"onset": [], "offset": []}}
        for s in ("right", "left"):
            S = series[s]
            for f in onsets(S["gripping"]):
                f = int(f)
                grip_events[s]["onset"].append(
                    (f, f / eff_fps,
                     (float(S["cx"][f]), float(S["cy"][f]))))
            for f in offsets(S["gripping"]):
                f = int(f)
                grip_events[s]["offset"].append(
                    (f, f / eff_fps,
                     (float(S["cx"][f]), float(S["cy"][f]))))
        pieces_list, pieces_visible, pieces_on_board = \
            piece_tracker.finalize(N, grip_events)
        piece_summary = puzzle_pieces.PieceTracker.summarize(
            pieces_list, dur_s)

    # ---- bimanual + pace ---------------------------------------------------
    rm, lm_ = series["right"]["moving"], series["left"]["moving"]
    rA = series["right"]["moving"] | series["right"]["fingering"]
    lA = series["left"]["moving"] | series["left"]["fingering"]
    rmv, lmv = per_hand["right"]["moving_pct"], per_hand["left"]["moving_pct"]
    dominant = "right" if rmv >= lmv else "left"
    ratio = round(max(rmv, lmv) / max(min(rmv, lmv), 0.1), 2)
    total_grips = per_hand["right"]["grip_cycles"] + per_hand["left"]["grip_cycles"]

    # ---- segmented analysis ------------------------------------------------
    seg_frames = max(1, int(seg_seconds * eff_fps))
    bounds = list(range(0, N, seg_frames)) + [N]
    if len(bounds) > 2 and (bounds[-1] - bounds[-2]) < 0.5 * seg_frames:
        bounds.pop(-2)               # merge a tiny tail into the prior segment
    segments = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        seg = {"t_start": round(a / eff_fps, 1),
               "t_end": round(b / eff_fps, 1),
               "left": slice_metrics("left", a, b),
               "right": slice_metrics("right", a, b)}
        if board is not None:
            for s in ("left", "right"):
                ib = series[s]["in_board"][a:b]
                seg[s]["assemble_pct"] = round(100 * ib.mean(), 1)
        segments.append(seg)

    # ---- fatigue / drift trend --------------------------------------------
    def combined(seg, key):
        vals = [seg["left"].get(key), seg["right"].get(key)]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else 0.0

    fatigue = {"segment_seconds": seg_seconds}
    if len(segments) >= 2:
        gpm = [combined(sg, "grips_per_min") for sg in segments]
        mvp = [(sg["left"]["moving_pct"] + sg["right"]["moving_pct"]) / 2
               for sg in segments]
        cvs = [v for v in
               [sg[dominant]["grip_rhythm_cv"] for sg in segments]
               if v is not None]
        # trend from a linear fit (robust to single noisy segments)
        mid = np.array([(sg["t_start"] + sg["t_end"]) / 2 / 60.0
                        for sg in segments])
        gco = np.polyfit(mid, gpm, 1)
        mco = np.polyfit(mid, mvp, 1)
        g0, g1 = float(np.polyval(gco, mid[0])), float(np.polyval(gco, mid[-1]))
        m0, m1 = float(np.polyval(mco, mid[0])), float(np.polyval(mco, mid[-1]))
        w_grip, p_grip = trend_label(g0, g1)
        w_move, p_move = trend_label(m0, m1)
        fatigue.update({
            "grips_per_min_trendline_start": round(g0, 1),
            "grips_per_min_trendline_end": round(g1, 1),
            "grips_per_min_change_pct": p_grip,
            "grips_per_min_slope_per_min": round(float(gco[0]), 3),
            "pace_trend": w_grip,
            "movement_trend": w_move,
            "rhythm_cv_first": cvs[0] if cvs else None,
            "rhythm_cv_last": cvs[-1] if cvs else None,
            "note": ("piece-handling pace is %s across the session "
                     "(linear-fit %s%% start to end)"
                     % (w_grip, p_grip if p_grip is not None else "n/a")),
        })
    else:
        fatigue["note"] = "session too short for a trend (need 2+ segments)"

    # ---- phases (3s windows) ----------------------------------------------
    win = max(1, int(3 * eff_fps))
    phases = []
    for i in range(0, N, win):
        r = float(np.mean(rm[i:i + win])) if i < N else 0.0
        l = float(np.mean(lm_[i:i + win])) if i < N else 0.0
        if r > 1.6 * max(l, 1e-6):
            tag = "right-dominant"
        elif l > 1.6 * max(r, 1e-6):
            tag = "left-dominant"
        else:
            tag = "balanced"
        phases.append({"t_start": round(i / eff_fps, 1),
                       "t_end": round(min(N, i + win) / eff_fps, 1),
                       "phase": tag})

    summary = {
        "video": os.path.basename(video),
        "puzzle_name": puzzle_name,
        "num_pieces": num_pieces,
        "difficulty": difficulty,
        "analyzed_seconds": round(dur_s, 1),
        "processing_fps": round(eff_fps, 1),
        "hands_swapped": swap,
        "board_region": board,
        "left": per_hand["left"],
        "right": per_hand["right"],
        "bimanual": {
            "both_moving_pct": round(100 * np.mean(rm & lm_), 1),
            "both_idle_pct": round(100 * np.mean(~(rA | lA)), 1),
            "parallel_active_pct": round(100 * np.mean(rA & lA), 1),
            "dominant_hand": dominant,
            "dominance_ratio": ratio,
        },
        "pace": {
            "total_grip_cycles": total_grips,
            "combined_grips_per_min": round(total_grips / dur_s * 60, 1),
        },
        "fatigue": fatigue,
        "segments": segments,
        "phases": phases,
    }
    if piece_tracker is not None:
        summary["piece_summary"] = piece_summary
        summary["pieces"] = pieces_list

    # ---- write data --------------------------------------------------------
    base = os.path.join(outdir, name)
    with open(base + "_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(base + "_perframe.csv", "w", newline="") as f:
        wr = csv.writer(f)
        cols = ["frame", "time_s"]
        for s in ("right", "left"):
            cols += ["%s_x" % s, "%s_y" % s, "%s_speed" % s, "%s_moving" % s,
                     "%s_articulation" % s, "%s_finger_active" % s,
                     "%s_pinch" % s, "%s_gripping" % s, "%s_openness" % s]
            if board is not None:
                cols.append("%s_in_board" % s)
        if piece_tracker is not None:
            cols += ["pieces_visible", "pieces_on_board"]
        wr.writerow(cols)
        for i in range(N):
            row = [i, round(i / eff_fps, 3)]
            for s in ("right", "left"):
                S = series[s]
                row += [round(S["cx"][i], 4), round(S["cy"][i], 4),
                        round(float(S["spd"][i]), 4), int(S["moving"][i]),
                        round(float(S["artic"][i]), 4), int(S["fingering"][i]),
                        round(float(S["pinch"][i]), 4), int(S["gripping"][i]),
                        round(float(S["openness"][i]), 4)]
                if board is not None:
                    row.append(int(S["in_board"][i]))
            if piece_tracker is not None:
                row += [int(pieces_visible[i]), int(pieces_on_board[i])]
            wr.writerow(row)

    _chart(base, N, eff_fps, series, summary)
    _session_chart(base, segments, seg_seconds, board, dominant)
    if bg_frame is not None:
        cv2.imwrite(base + "_frame.jpg", bg_frame)   # clean bg for report
    try:
        puzzle_report.make_all(base + "_metrics.json")
    except Exception as exc:
        print("report generation failed:", exc)
    _calibration(base, bg_frame, rec, bg_idx, board, W, H)
    if want_video:
        _video(base, video, rec, series, start_f, step, eff_fps, W, H,
               preview_seconds)

    # JSON dump is printed only by the CLI (see __main__); the GUI doesn't
    # need it on stdout and the file is already on disk at <base>_metrics.json.
    return summary


# -------------------------------------------------------------------- chart
def _chart(base, N, fps, series, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(N) / fps
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 1.2]})
    R, L = series["right"], series["left"]
    cR, cL = "#3c5af0", "#f0aa28"
    ax[0].plot(t, R["spd"], color=cR, lw=1.0, label="Right travel")
    ax[0].plot(t, L["spd"], color=cL, lw=1.0, label="Left travel")
    ax[0].set_ylabel("Hand travel speed")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("%s - hand & finger activity (%ss)"
                    % (summary["video"], summary["analyzed_seconds"]),
                    fontsize=12, weight="bold")
    ax[0].grid(alpha=0.25)
    ax[1].plot(t, R["artic"], color=cR, lw=1.0, label="Right finger work")
    ax[1].plot(t, L["artic"], color=cL, lw=1.0, label="Left finger work")
    ax[1].set_ylabel("Finger articulation")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=0.25)
    for S, y, c in ((R, 2, cR), (L, 1, cL)):
        g = onsets(S["gripping"])
        ax[2].vlines(g / fps, y - 0.35, y + 0.35, color=c, lw=1.4)
    ax[2].set_yticks([2, 1])
    ax[2].set_yticklabels(["Right", "Left"])
    ax[2].set_ylim(0.3, 2.7)
    ax[2].set_xlabel("Time (seconds)")
    ax[2].set_title("Grip cycles (each tick = a pinch / piece handled)",
                    fontsize=10)
    ax[2].grid(alpha=0.2, axis="x")
    plt.tight_layout()
    plt.savefig(base + "_timeline.png", dpi=130)
    plt.close()


# ----------------------------------------------------------- session chart
def _session_chart(base, segments, seg_seconds, board, dominant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(segments) < 2:
        return
    mins = [(sg["t_start"] + sg["t_end"]) / 2 / 60.0 for sg in segments]
    cR, cL = "#3c5af0", "#f0aa28"
    rows = 3 if board is not None else 2
    fig, ax = plt.subplots(rows, 1, figsize=(11, 2.6 * rows), sharex=True)

    gR = [sg["right"]["grips_per_min"] for sg in segments]
    gL = [sg["left"]["grips_per_min"] for sg in segments]
    ax[0].plot(mins, gR, "o-", color=cR, label="Right")
    ax[0].plot(mins, gL, "o-", color=cL, label="Left")
    ax[0].set_ylabel("Grips / min")
    ax[0].set_title("Session trend - piece-handling pace, tempo, zones",
                    fontsize=12, weight="bold")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)

    cvR = [sg["right"]["grip_rhythm_cv"] or np.nan for sg in segments]
    cvL = [sg["left"]["grip_rhythm_cv"] or np.nan for sg in segments]
    ax[1].plot(mins, cvR, "o-", color=cR, label="Right")
    ax[1].plot(mins, cvL, "o-", color=cL, label="Left")
    ax[1].set_ylabel("Rhythm CV\n(lower = steadier)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)

    if board is not None:
        aR = [sg["right"].get("assemble_pct", np.nan) for sg in segments]
        aL = [sg["left"].get("assemble_pct", np.nan) for sg in segments]
        ax[2].plot(mins, aR, "o-", color=cR, label="Right")
        ax[2].plot(mins, aL, "o-", color=cL, label="Left")
        ax[2].set_ylabel("% time on board\n(assembling)")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.25)
    ax[-1].set_xlabel("Session time (minutes)")
    plt.tight_layout()
    plt.savefig(base + "_session.png", dpi=130)
    plt.close()


# -------------------------------------------------------------- calibration
def _calibration(base, bg, rec, bg_idx, board, W, H):
    if bg is None:
        return
    f = bg.copy()
    for gx in np.arange(0.1, 1.0, 0.1):
        x = int(gx * W)
        cv2.line(f, (x, 0), (x, H), (0, 200, 0), 1)
        cv2.putText(f, "%.1f" % gx, (x + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    for gy in np.arange(0.1, 1.0, 0.1):
        y = int(gy * H)
        cv2.line(f, (0, y), (W, y), (0, 200, 0), 1)
        cv2.putText(f, "%.1f" % gy, (2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    if board is not None:
        x1, y1, x2, y2 = board
        cv2.rectangle(f, (int(x1 * W), int(y1 * H)),
                      (int(x2 * W), int(y2 * H)), (0, 255, 255), 2)
    col = {"right": (60, 90, 240), "left": (40, 170, 240)}
    for side in ("right", "left"):
        x = rec[side][bg_idx] if 0 <= bg_idx < len(rec[side]) else None
        if x is None:
            continue
        cx, cy = int(x["c"][0] * W), int(x["c"][1] * H)
        cv2.circle(f, (cx, cy), 16, col[side], 3)
        cv2.putText(f, side.upper(), (cx - 30, cy - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col[side], 2)
    cv2.imwrite(base + "_calibration.jpg", f)


# -------------------------------------------------------------------- video
def _video(base, video, rec, series, start_f, step, fps, W, H, preview_s):
    out = cv2.VideoWriter(base + "_annotated.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    col = {"right": (60, 90, 240), "left": (40, 170, 240)}
    limit = len(rec["right"]) if preview_s <= 0 else int(preview_s * fps)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    fi, j = start_f, 0
    while j < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if (fi - start_f) % step == 0:
            if j >= len(rec["right"]):
                break
            for side in ("right", "left"):
                x = rec[side][j]
                if x is None:
                    continue
                c = col[side]
                pts = [(int(px * W), int(py * H)) for px, py in x["pts"]]
                for a, b in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], c, 1)
                for p in pts:
                    cv2.circle(frame, p, 2, c, -1)
                cx, cy = int(x["c"][0] * W), int(x["c"][1] * H)
                grip = series[side]["gripping"][j]
                cv2.circle(frame, (cx, cy), 15, c, -1 if grip else 3)
                tags = []
                if series[side]["moving"][j]:
                    tags.append("MOVE")
                if series[side]["fingering"][j]:
                    tags.append("FINGER")
                if grip:
                    tags.append("GRIP")
                cv2.putText(frame, "%s: %s" % (side.upper(),
                            "+".join(tags) if tags else "still"),
                            (cx - 60, cy - 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, c, 2)
            cv2.rectangle(frame, (0, 0), (220, 30), (0, 0, 0), -1)
            cv2.putText(frame, "t=%6.1fs" % (j / fps), (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            out.write(frame)
            j += 1
        fi += 1
    cap.release()
    out.release()


# --------------------------------------------------------------------- main
def _parse_board(s):
    if not s:
        return None
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        sys.exit("--board needs 4 comma-separated values: x1,y1,x2,y2")
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Hand & finger analysis for speed-puzzling video.")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0, help="start time (s)")
    ap.add_argument("--duration", type=float, default=None, help="length (s)")
    ap.add_argument("--fps", type=float, default=15, help="processing fps")
    ap.add_argument("--move-threshold", type=float, default=0.35,
                    help="gross-travel sensitivity (norm units/sec)")
    ap.add_argument("--finger-threshold", type=float, default=2.5,
                    help="finger-articulation sensitivity (tuned for ~15fps)")
    ap.add_argument("--grip-threshold", type=float, default=0.5,
                    help="thumb-index distance (hand-widths) for a grip")
    ap.add_argument("--board", type=str, default=None,
                    help="puzzle board rectangle x1,y1,x2,y2 in 0-1 coords "
                         "(read it off the calibration grid)")
    ap.add_argument("--segment-seconds", type=float, default=60,
                    help="length of each analysis segment")
    ap.add_argument("--preview-seconds", type=float, default=120,
                    help="length of annotated video to render (0 = full)")
    ap.add_argument("--swap-hands", action="store_true",
                    help="reverse left/right labels for this camera setup")
    ap.add_argument("--no-video", action="store_true",
                    help="skip the annotated video (faster)")
    ap.add_argument("--no-pieces", action="store_true",
                    help="skip visual piece tracking (faster pass 1)")
    ap.add_argument("--outdir", default="puzzle_output")
    ap.add_argument("--puzzle-name", default=None,
                    help="puzzle title (shown on the report)")
    ap.add_argument("--pieces", type=int, default=None,
                    help="number of puzzle pieces (shown on the report)")
    ap.add_argument("--difficulty", default=None,
                    choices=["Easy", "Medium", "Hard", "Expert"],
                    help="self-rated difficulty (shown on the report)")
    a = ap.parse_args()
    summary = analyze(a.video, a.start, a.duration, a.fps, a.move_threshold,
                      a.finger_threshold, a.grip_threshold,
                      _parse_board(a.board),
                      a.segment_seconds, a.preview_seconds, a.swap_hands,
                      not a.no_video, a.outdir,
                      puzzle_name=a.puzzle_name, num_pieces=a.pieces,
                      difficulty=a.difficulty,
                      track_pieces=not a.no_pieces)
    print(json.dumps(summary, indent=2))
