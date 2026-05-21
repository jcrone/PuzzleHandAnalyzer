# Dynamic Board Detection + Assembly Milestones — Design

**Date:** 2026-05-20
**Status:** Draft (awaiting user review)
**Branch:** `claude/track-puzzle-placement-Det2G`

## Summary

The puzzle hand analyzer currently requires the user to mark the puzzle board with a 4-click rectangle before analysis. Without it, `placed_on_board` is always False and sort-vs-assemble metrics are unavailable. This spec replaces the manual-marking requirement with **automatic discovery** of the assembly area by tracking clusters of stationary puzzle pieces over time, and adds **timestamped milestones** (first piece joined, 25/50/75% complete, frame complete, islands merged, sort pile cleared).

Manual marking remains available as a fallback. Auto-detect is the default and is the recommended option when the assembly area moves during play (e.g. sort pile starts in the middle, then is pushed aside).

## Goals

- Make piece-placement metrics work without any pre-analysis clicks.
- Detect the actual assembly cluster — not the sort pile, which is often the largest cluster early in a session.
- Emit chronological milestones the puzzle-analyst skill can surface.
- Keep the existing piece-tracking and hand-tracking pipelines working unchanged; the new system runs in parallel.

## Non-goals

- Per-piece identity tracking across the full session (still hard; deferred to a later phase).
- Real-time / live operation (this is batch video analysis).
- Detecting puzzle completion as a confident yes/no signal (we report progress %, not a "you're done" flag).

## Architecture

### Module layout

A new file **`puzzle_clusters.py`** runs in parallel with the existing `puzzle_pieces.py`. They have different jobs and do not share state.

| Module | Job | Key signal |
|---|---|---|
| `puzzle_pieces.py` (existing) | Track moving pieces; correlate placements with hand grip events | "left hand placed something around t=520" |
| `puzzle_clusters.py` (new) | Map stationary cluster footprint over time | "main cluster reached 250 pieces at t=1140" |

A small **`puzzle_vision_utils.py`** module is added to hold the shared `hand_mask()` helper (convex-hull-of-fingers + dilation), previously a private method of `PieceTracker`. Both files import from it.

### Integration into `puzzle_hands.py`

A new `cluster_mapper` instance is built next to `piece_tracker` (currently around line 174):

```python
cluster_mapper = puzzle_clusters.ClusterMapper(
    W, H,
    num_pieces=num_pieces,
    eff_fps=eff_fps,
    survey_interval_s=5.0,
) if track_clusters else None
```

Per-frame call shape mirrors `piece_tracker.update(...)`. Internally `ClusterMapper.update()` short-circuits on most frames — heavy work only fires every `survey_interval_s` seconds (default 5s).

At finalize, the cluster mapper produces its history and a discovered board, which is fed to the piece tracker as a dynamic board:

```python
cluster_history, milestones = cluster_mapper.finalize(N)
inferred_board = cluster_mapper.inferred_board()

if piece_tracker.board is None and inferred_board is not None:
    piece_tracker.board = inferred_board
pieces_list, pieces_visible, pieces_on_board = piece_tracker.finalize(N, grip_events)
```

A new CLI flag `--no-clusters` mirrors the existing `--no-pieces`, allowing the user to skip cluster mapping for a faster pass-1.

## The periodic detector

Runs every `survey_interval_s` seconds (default 5s). On each survey frame:

1. **Hand mask** — same helper as the piece tracker. Erases pixels inside the convex hull of each hand (dilated by 35px).
2. **Find candidate piece blobs** using two strategies together:
   - **Edge-based:** Canny edges → dilate → contour finding. Each contour bounded by area gates `area_min_frac = 0.0004` and `area_max_frac = 0.012` (same as `PieceTracker`).
   - **Stillness check:** Compare each candidate's small ROI to the same ROI 1s earlier. If pixel-wise SAD is below a threshold, mark as `stationary`. Moving pieces are filtered out — the cluster mapper only cares about pieces that have settled.
3. **Cluster the stationary blobs** with DBSCAN:
   - `eps` ≈ 2× the median piece radius in normalized coordinates (tuned at first survey, fixed thereafter for the session).
   - `min_samples = 2` (a cluster is ≥2 adjacent pieces).
   - Output per survey: list of `(centroid, bbox, member_xys, member_count)`.
4. **Match this survey's clusters to existing tracked clusters** by IoU of bounding boxes (>0.3 = same cluster, carry over its ID). Unmatched detections start new cluster IDs. Tracked clusters with no match for 2 consecutive surveys (10s+) are marked `dormant` but kept — a cluster doesn't disappear just because pieces were temporarily moved or occluded.
5. **Detect merges.** If two previously-distinct cluster IDs are now both contained inside one larger detected cluster, that's a merge event — record it and consolidate to the larger cluster's ID.

Rationale for 5s surveys (not per-frame): stationary clusters change slowly. 5s × ~470 surveys for a 39-min session vs. ~27,000 per-frame runs — ~60× cheaper. Resolution we don't need.

## Cluster bookkeeping

Each cluster gets a lifecycle record:

```python
class ClusterRecord:
    id: int                    # stable across surveys
    born_t: float              # first survey it appeared
    last_seen_t: float         # most recent survey it was detected
    initial_count: int         # count at born_t
    history: list[tuple]       # (t, count, bbox, centroid) per survey
    peak_count: int            # max members ever
    merged_into: int | None    # if absorbed into another cluster
    merged_from: list[int]     # cluster IDs absorbed into this one
```

Events logged during update():

| Event | Trigger |
|---|---|
| `cluster_born` | New cluster ID first appears (count starts at 2) |
| `cluster_grew` | Member count increased since last survey (internal — drives milestones) |
| `clusters_merged` | Two previously-distinct cluster IDs are now inside one detected cluster |
| `cluster_dormant` | Cluster has had no matching survey for 2 consecutive surveys (10s+) |

**Pruning at finalize:** Drop any cluster whose `peak_count < 3`. Almost certainly noise (transient shadows, hands paused over the pile).

### Sort pile vs. assembly cluster

This is the critical classification step. The sort pile is often the *largest* cluster for the first 10-20 minutes of a session, so "main cluster = biggest cluster" is wrong.

**Classification rule:** Compute `net_growth = final_count - initial_count` for each kept cluster.

| Cluster type | Net growth |
|---|---|
| Sort pile (depleted as pieces are sorted) | Strongly **negative** |
| Assembly cluster (pieces added over session) | Strongly **positive** |
| Static debris | ≈ 0 |

**Main cluster** = the cluster with the largest positive net growth ≥ 5 pieces. If no cluster meets that threshold, there is no assembly cluster (`main_cluster_id: null`, `board_source: "none"`).

**Tiebreak when two clusters both have positive net growth:** prefer the one whose bbox aspect ratio is closer to a typical puzzle (between 1:2 and 2:1). Logged as `tiebreak_used: true`.

## Dynamic board hand-off

After the cluster mapper finalizes, the main cluster's bbox at its peak count is passed to the piece tracker as a dynamic board substitute.

`ClusterMapper.inferred_board()` returns `[x1, y1, x2, y2]` in normalized coords (same shape as the existing manual board), or `None` if no assembly cluster was identified.

**Precedence rule** (in `puzzle_hands.py.run()`):

1. If the user manually marked a board → use the manual board (respect explicit input).
2. Else if a cluster was discovered → use the inferred board.
3. Else → no board (current behavior).

The peak-bbox is used rather than the moment-to-moment bbox so that `placements_per_min` is computed against a stable final assembly area, not a moving target. A piece placed at minute 5 counts the same as one placed at minute 35.

A new top-level `board_source` field in `metrics.json` records which path was taken (`"manual"`, `"auto-detected"`, or `"none"`), so the puzzle-analyst skill can footnote inferred-board sessions appropriately.

## Milestone detection

Computed at `finalize()` from the cluster history. Each becomes one entry in a chronological `milestones` list:

```jsonc
{ "t": <seconds>, "type": "...", "label": "...", "cluster_id": <id>, "confidence": "high" | "medium", "details": {...} }
```

### Rules

| Milestone | Detection | Confidence |
|---|---|---|
| `first_piece_joined` | The earliest survey where the **main** (assembly) cluster — determined retrospectively at finalize — crossed count ≥ 2. Computed after classification so a transient noise blob can't fire it | high |
| `cluster_25pct` / `cluster_50pct` / `cluster_75pct` | The earliest survey where the **main** (assembly) cluster's count crosses 0.25 / 0.50 / 0.75 × `num_pieces` | high (if `num_pieces` supplied; else milestone is omitted) |
| `frame_complete` | Main cluster's count is between 80%-130% of expected frame size AND `perimeter_fraction > 0.85`, where `perimeter_fraction` = (members within 1 piece-width of any bbox edge) / (total members in cluster). High perimeter_fraction means the cluster is a hollow ring rather than a filled rectangle | medium (heuristic) |
| `islands_merged` | A `clusters_merged` event where both merging clusters had `peak_count ≥ 10` (filters tiny absorbs) | high |
| `sort_pile_cleared` | The cluster with the most negative net growth drops below 20% of its peak (or below 20 pieces, whichever is lower) | medium |

### Frame-complete heuristic detail

At each survey, for the main cluster:
1. Fit a bounding box to its members.
2. Estimate puzzle grid dimensions: `cols = bbox_width / median_piece_width`, `rows = bbox_height / median_piece_height`.
3. Expected frame piece count = `2 * (cols + rows) - 4`.
4. Compute `perimeter_fraction` = (cluster members within 1 piece-width of any bbox edge) / (total cluster members).
5. Fire `frame_complete` if total count is within 80%-130% of expected frame count AND `perimeter_fraction > 0.85`.

### Deduplication

Each milestone type fires at most once per session — the *first* crossing only. If the main cluster crosses 50%, dips below (pieces temporarily moved), then crosses again, the 50% milestone fires only on the first crossing.

## GUI changes (`puzzle_app.py`)

Step 2 of the GUI is restructured.

### Before

> 2. *(Optional)* Mark the puzzle board
> [canvas where the user draws a rectangle]
> Board: (not set) | [Clear board]

### After

> 2. Board area (auto-detected by default)
>
> ( ) Auto-detect board    ← default
>     Recommended if your board area changes during play
>     (e.g. sort pile in the middle, then pushed aside)
>
> ( ) Set puzzle board manually
>     Draw a rectangle on the preview to mark the assembly zone
>
> [canvas — read-only in auto mode, interactive in manual mode]
> Board: will be auto-detected from cluster footprint | [Reset to auto-detect]

### Behavior

- **Auto-detect mode (default):** canvas shows the preview frame but is read-only. Status label: *"Board: will be auto-detected from cluster footprint."*
- **Manual mode:** enables the existing click-and-drag interaction. Status label shows current rectangle. The "Clear board" button is relabeled "Reset to auto-detect" and switches the radio back to Auto.
- Switching Manual → Auto wipes the rectangle. Switching Auto → Manual leaves the canvas blank until the user draws.

### Wiring

- Auto-detect → `puzzle_hands.run(..., board=None)`. The cluster mapper discovers the board and feeds it to the piece tracker.
- Manual → existing path, `board=(x1, y1, x2, y2)` passed through.

No CLI changes are required. The existing `puzzle_hands.py` CLI has no `--board` flag today, so auto-detect just becomes the natural CLI behavior. A future `--board` CLI override is out of scope.

## Output schema

### `metrics.json` — new top-level keys

```jsonc
"board_region": [0.21, 0.13, 0.78, 0.86],   // unchanged shape
"board_source": "auto-detected",            // NEW: "manual" | "auto-detected" | "none"

"cluster_summary": {                        // NEW
    "total_clusters_observed": 12,
    "total_clusters_kept": 4,
    "main_cluster_id": 2,
    "main_cluster_peak_count": 287,
    "main_cluster_net_growth": 285,
    "merge_events": 3,
    "survey_interval_s": 5.0,
    "tiebreak_used": false,
    "detection_quality": "high",
    "quality_notes": []
},

"clusters": [                               // NEW
    {
        "id": 2,
        "born_t": 124.5,
        "last_seen_t": 1820.0,
        "initial_count": 2,
        "peak_count": 287,
        "final_count": 287,
        "net_growth": 285,
        "bbox_at_peak": [0.21, 0.13, 0.78, 0.86],
        "is_main": true,
        "merged_into": null,
        "merged_from": [1, 5],
        "history_seconds": [125, 130, 135],
        "history_counts":  [2,   4,   7]
    }
],

"milestones": [                             // NEW
    {
        "t": 14.2,
        "type": "first_piece_joined",
        "label": "First two pieces joined",
        "cluster_id": 2,
        "confidence": "high"
    },
    {
        "t": 1140.0,
        "type": "cluster_50pct",
        "label": "Cluster reached 50% of pieces (252 / 500)",
        "cluster_id": 2,
        "confidence": "high",
        "details": { "count": 252, "fraction": 0.504 }
    },
    {
        "t": 1820.5,
        "type": "frame_complete",
        "label": "Outer frame appears complete",
        "cluster_id": 2,
        "confidence": "medium",
        "details": { "perimeter_fraction": 0.91, "expected_frame_pieces": 86 }
    }
]
```

### `perframe.csv` — two new columns (when clusters are enabled)

| Column | Type | Meaning |
|---|---|---|
| `largest_cluster_size` | int | Member count of the biggest cluster at that frame (carried forward between surveys, 0 before first detection) |
| `cluster_count` | int | Number of active (non-dormant) clusters at that frame |

### Existing blocks

`pieces` and `piece_summary` remain unchanged in shape but will now populate properly (with `placed_on_board: true` for placed pieces) because they receive the inferred board.

### Backwards compatibility

Older `metrics.json` files (pre-cluster) lack the new keys. The `puzzle-analyst` skill must use `dict.get(...)` patterns when reading them — already its existing pattern. The `data-schema.md` in the analyst skill will be updated as part of implementation.

## Failure modes & data quality

### Detection-side problems

| Problem | Mitigation |
|---|---|
| Lighting changes mid-session trigger spurious blobs | DBSCAN `min_samples=2` filters isolated noise. `peak_count >= 3` prune at finalize drops the rest |
| Hand cast shadows look like piece blobs | Hand mask is dilated by 35px (existing constant) |
| Pieces stacked / partially overlapping | One blob may contain multiple pieces. Count under-estimates. Accept it — cluster *shape* and *growth direction* are more reliable than absolute member count |
| Hands resting on assembled section occlude pieces | Cluster goes briefly dormant; 2-survey tolerance absorbs this |

### Classification-side problems

| Problem | Mitigation |
|---|---|
| Sort pile gets shuffled enough to show positive net growth | Aspect-ratio tiebreak (1:2 to 2:1); logged as `tiebreak_used: true` |
| Puzzler only sorts, never starts assembly | No cluster has positive net growth ≥ 5. `board_source: "none"`, `main_cluster_id: null`. Milestones list is empty except possibly `sort_pile_cleared` |
| `frame_complete` fires on a pile that happened to land rectangular | Frame complete requires `perimeter_fraction > 0.85` AND total count near expected frame size (not filled-in). A pile of similar bbox size will have most pieces in the interior, failing the perimeter_fraction test |

### Quality flags

`cluster_summary` includes:

```jsonc
"detection_quality": "high" | "medium" | "low",
"quality_notes": ["…", "…"]
```

Quality is:
- **`"low"`** if any of: hand detection < 60%, no cluster ever exceeded `peak_count` of 20, fewer than 5 surveys produced any clusters.
- **`"medium"`** if hand detection 60-75%, or `tiebreak_used: true`, or main cluster bbox aspect ratio outside 1:2.5 to 2.5:1.
- **`"high"`** otherwise.

The `puzzle-analyst` skill will footnote any milestone whose session has low detection quality, so the puzzler sees the caveat next to the claim.

### Debug artifact

A new optional output file `*_clusters_debug.png` saved when `--debug-clusters` is passed. A 4-panel image showing:
1. The final survey's hand-masked frame.
2. Detected piece blobs (color-coded by cluster assignment).
3. The inferred board rectangle overlaid on the frame.
4. The sort-pile (if found) outlined in a different color.

Helps diagnose problematic sessions without re-running the whole pipeline.

## Testing

Unit tests (new file `tests/test_puzzle_clusters.py`):

- **DBSCAN cluster matching** — two synthetic blob lists, verify IoU-based ID continuity across surveys.
- **Net-growth classification** — feed a synthetic timeline where one cluster grows and one shrinks. Assert the growing one is selected as main.
- **Frame-complete heuristic** — synthesize a 25×20 ring of blob positions, assert `frame_complete` fires. Synthesize a dense pile, assert it does not.
- **Dormancy tolerance** — feed surveys that drop a cluster for 1 then 3 consecutive surveys, assert it stays active for 1 but is dormant after 3.
- **Tiebreak path** — feed two equally-grown clusters with different aspect ratios, assert the more puzzle-shaped one wins.

Integration test:

- A short pre-recorded fixture video with known ground-truth piece positions (1 min, ~20 pieces, hand-labeled cluster boundaries). Asserts `main_cluster_id`, `peak_count` within ±10%, and that `first_piece_joined` fires within 10s of the ground-truth time.

End-to-end smoke test:

- Re-run on `RollingHills.mp4`, confirm `board_source: "auto-detected"`, a non-zero `placed_on_board` count, and at least `first_piece_joined` + `sort_pile_cleared` milestones present. (We expect this real-world video to be noisier than the synthetic fixture; this test asserts the pipeline runs end-to-end without crash or null output, not exact numbers.)

## Performance budget

The periodic detector adds work every 5s of video. Profiling target: total analysis time ≤ 1.5× the current baseline for a 40-min session (currently ~3 min on the user's machine). If this is exceeded, the survey interval can be raised to 10s with no schema change.

## Open questions

None for v1. Per-piece identity tracking (where individual pieces are followed across surveys) is explicitly deferred to phase 2 and tracked separately.
