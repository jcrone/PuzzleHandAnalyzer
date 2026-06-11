#!/usr/bin/env python3
"""
puzzle_highlight.py - a short, shareable "highlight reel" of a solve.

Reads a puzzle_hands.py output folder (a <base>_metrics.json plus the matching
<base>_perframe.csv) and writes:

  <base>_highlight.mp4   a ~45s sped-up clip of the whole solve, each hand
                         drawn as a glowing comet-trail moving over the clean
                         board frame. Small (~1-3 MB) and built for sharing -
                         distinct from the full-resolution <base>_annotated.mp4.

Can be run standalone on an existing analysis folder:

    python3 puzzle_highlight.py path/to/<base>_metrics.json
    python3 puzzle_highlight.py path/to/<base>_metrics.json --video orig.mp4

If <base>_frame.jpg (a clean background frame) is present next to the JSON, or
--video is given, it's used (dimmed) as the backdrop. Otherwise the trails are
drawn on a neutral dark canvas.
"""

import argparse

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# reuse the loaders/backdrop logic so we stay in lock-step with the report
from puzzle_report import load_data, find_bg_frame

# glow colors as RGB (matching the report's C_RIGHT / C_LEFT); converted to
# BGR at draw time for cv2
COL_RIGHT_RGB = (60, 90, 240)     # blue
COL_LEFT_RGB  = (240, 170, 40)    # orange

TARGET_DURATION_S = 45.0          # target clip length
OUT_FPS = 30                      # output frame rate
TRAIL_SECONDS = 1.5               # how much real motion the comet tail spans
MAX_HEIGHT = 720                  # output is downscaled to at most this tall
BG_DIM = 0.35                     # background brightness so trails pop
DEFAULT_SIZE = (1280, 720)        # canvas when no background frame is available


def _rgb_to_bgr(c):
    return (float(c[2]), float(c[1]), float(c[0]))


def _prep_background(bg):
    """Returns a dimmed BGR float32 canvas and its (W, H), downscaled so the
    height is at most MAX_HEIGHT. bg is an RGB array or None."""
    if bg is None:
        w, h = DEFAULT_SIZE
        canvas = np.full((h, w, 3), 18.0, dtype=np.float32)  # near-black
        return canvas, w, h
    bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if h > MAX_HEIGHT:
        scale = MAX_HEIGHT / float(h)
        w, h = int(round(w * scale)), MAX_HEIGHT
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    canvas = bgr.astype(np.float32) * BG_DIM
    return canvas, w, h


def _draw_comet(glow, xs, ys, gripping, color_bgr, W, H, head_r):
    """Additively render one hand's comet onto the float glow layer.

    xs/ys are the trail's normalized coords, oldest-first; the last entry is
    the head. Intensity ramps from ~0 at the tail to 1 at the head; the head
    gets a brighter, larger core when the hand is gripping.

    Drawing/blurring is confined to the trail's padded bounding box so the
    cost scales with the comet, not the whole frame.
    """
    n = len(xs)
    if n == 0:
        return
    px = np.clip(xs, 0.0, 1.0) * (W - 1)
    py = np.clip(ys, 0.0, 1.0) * (H - 1)
    sigma = max(1.5, head_r * 0.6)
    core_r = head_r * (1.7 if gripping else 1.0)
    pad = int(np.ceil(3 * sigma + core_r)) + 2

    x0 = max(0, int(np.floor(px.min())) - pad)
    y0 = max(0, int(np.floor(py.min())) - pad)
    x1 = min(W, int(np.ceil(px.max())) + pad)
    y1 = min(H, int(np.ceil(py.max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return

    inten = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    pts = [(int(round(px[k])) - x0, int(round(py[k])) - y0) for k in range(n)]
    for k in range(n):
        recency = k / (n - 1) if n > 1 else 1.0
        r = max(1, int(round(head_r * (0.30 + 0.70 * recency))))
        val = recency ** 1.5
        cv2.circle(inten, pts[k], r, val, -1, lineType=cv2.LINE_AA)
        if k > 0:
            thick = max(1, int(round(head_r * (0.25 + 0.55 * recency))))
            cv2.line(inten, pts[k - 1], pts[k], val, thick,
                     lineType=cv2.LINE_AA)
    # head core - brighter, and a bigger flare while gripping
    cv2.circle(inten, pts[-1], max(1, int(round(core_r))),
               1.6 if gripping else 1.1, -1, lineType=cv2.LINE_AA)
    # bloom
    inten = cv2.GaussianBlur(inten, (0, 0), sigma)
    glow[y0:y1, x0:x1] += inten[:, :, None] * np.array(color_bgr,
                                                       dtype=np.float32)


def make_highlight(metrics_path, video_path=None, out_path=None):
    """Render <base>_highlight.mp4. Returns the output path, or None if it
    couldn't be produced (no cv2, no writable codec, or no usable data)."""
    if cv2 is None:
        print("highlight: OpenCV not available, skipping")
        return None

    summary, cols, base = load_data(metrics_path)
    if out_path is None:
        out_path = base + "_highlight.mp4"

    bg = find_bg_frame(base, video_path)
    canvas, W, H = _prep_background(bg)

    n_src = len(cols["frame"])
    if n_src < 2:
        print("highlight: not enough frames, skipping")
        return None

    src_fps = float(summary.get("processing_fps") or OUT_FPS)
    trail_n = max(2, int(round(TRAIL_SECONDS * src_fps)))
    head_r = max(3, int(round(H * 0.012)))

    target_frames = max(1, int(round(TARGET_DURATION_S * OUT_FPS)))
    step = max(1.0, n_src / float(target_frames))
    n_out = int(np.ceil(n_src / step))

    fourcc = (cv2.VideoWriter_fourcc(*"avc1")
              or cv2.VideoWriter_fourcc(*"mp4v"))
    writer = cv2.VideoWriter(out_path, fourcc, OUT_FPS, (W, H))
    if not writer.isOpened():
        writer = cv2.VideoWriter(out_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 OUT_FPS, (W, H))
    if not writer.isOpened():
        print("highlight: no usable video codec, skipping")
        return None

    rx, ry = cols["right_x"], cols["right_y"]
    lx, ly = cols["left_x"], cols["left_y"]
    rg = cols["right_gripping"].astype(bool)
    lg = cols["left_gripping"].astype(bool)
    col_r = _rgb_to_bgr(COL_RIGHT_RGB)
    col_l = _rgb_to_bgr(COL_LEFT_RGB)

    for i in range(n_out):
        src = min(n_src - 1, int(round(i * step)))
        lo = max(0, src - trail_n + 1)
        sl = slice(lo, src + 1)
        glow = np.zeros((H, W, 3), dtype=np.float32)
        _draw_comet(glow, lx[sl], ly[sl], bool(lg[src]), col_l, W, H, head_r)
        _draw_comet(glow, rx[sl], ry[sl], bool(rg[src]), col_r, W, H, head_r)
        frame = np.clip(canvas + glow, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Short sped-up highlight reel (glowing hand trails) from "
                    "a puzzle_hands.py metrics.json file.")
    ap.add_argument("metrics_json", help="path to <base>_metrics.json")
    ap.add_argument("--video", default=None,
                    help="path to original video, used to grab a clean "
                         "background frame if <base>_frame.jpg is absent")
    a = ap.parse_args()
    out = make_highlight(a.metrics_json, a.video)
    if out:
        print("wrote:\n  " + out)
