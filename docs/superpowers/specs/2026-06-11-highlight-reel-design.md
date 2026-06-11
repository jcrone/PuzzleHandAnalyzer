# Highlight Reel — design

## Goal

Add a flashy, shareable output to the Puzzle Hand Analyzer: a short
(~45 s) sped-up "highlight reel" of the solve, showing each hand as a
glowing comet-trail moving over the clean board frame. Distinct from the
existing full-resolution `_annotated.mp4` (hundreds of MB) — this one is
small (~1–3 MB) and built for posting to Discord/Reddit.

## Output

One new artifact per analysis: `<base>_highlight.mp4`.

## Module

`puzzle_highlight.py` — mirrors `puzzle_report.py`:

- importable `make_highlight(metrics_path, video_path=None)`
- standalone CLI: `python3 puzzle_highlight.py <base>_metrics.json [--video V]`
- reuses `load_data` and `find_bg_frame` from `puzzle_report` (no duplication)

## Visual design

- Just the two hands. No HUD, no progress bar, no text overlays.
- Left = orange, right = blue (the report palette, converted to BGR for cv2).
- Each hand is a comet: a stack of filled circles over the last
  `TRAIL_SECONDS` (~1.5 s) of real motion, radius + brightness ramping up
  toward the head, drawn on a separate glow layer and **additively** blended
  onto the background so overlapping trail produces real bloom.
- The trail fades fully to transparent at its tail (no persistent ghost path).
- The head dot pulses brighter/larger when that hand is `gripping` — the one
  data-driven flourish.
- Background: the clean `_frame.jpg`, dimmed to ~0.35 brightness so the trails
  pop. No frame available → neutral dark canvas at 1280×720.

## Render pipeline (OpenCV additive glow)

1. Load per-frame `right_x/y`, `left_x/y`, `right_gripping`, `left_gripping`
   and the (dimmed) background. Coordinates are normalized [0,1]; clip the
   occasional slight out-of-range value before mapping to pixels. y is
   top-origin (matches the report's `extent`).
2. Compress the full solve to `TARGET_DURATION_S` at `OUT_FPS`
   (45 s × 30 fps ≈ 1350 output frames). `step = n_source / n_output`;
   output frame *i* maps to source index `round(i*step)`. For short sessions
   `step` clamps to ≥ ~1 (less speed-up, fewer frames).
3. For each output frame, draw both hands' trails onto a fresh glow layer,
   additively blend onto the dimmed background, write via `cv2.VideoWriter`
   (try `avc1`, fall back to `mp4v` — no external ffmpeg binary required).

## Tunable constants (module top)

`TARGET_DURATION_S=45`, `OUT_FPS=30`, `TRAIL_SECONDS=1.5`, `MAX_HEIGHT=720`,
`BG_DIM=0.35`, glow colors for left/right.

## Integration

Add a guarded `puzzle_highlight.make_highlight(base + "_metrics.json")` call
right after the `puzzle_report.make_all(...)` call in `puzzle_hands.py`
(~line 772), wrapped in its own try/except so a render failure never breaks
analysis. Add `puzzle_highlight.py` to the `FILES` array in
`build_release.sh`.

## Edge cases

- Missing background frame → dark canvas, default 1280×720.
- `cv2.VideoWriter` opens with neither codec → log and skip (no crash).
- Very short / very long sessions handled by the `step` clamp.

## Non-goals

- No audio. No GUI button (auto-runs with every analysis; standalone CLI
  covers existing folders). No GIF variant for now.
