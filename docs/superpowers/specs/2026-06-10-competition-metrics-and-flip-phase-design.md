# Competition metrics, flip-phase detection, and progress tracking — design

**Date:** 2026-06-10
**Status:** approved (design); implementation plan to follow

## Purpose

Reframe the Puzzle Hand Analyzer around what a *professional / competitive*
puzzler actually trains on. Today every metric is a proxy for **effort**
(grips, travel, articulation). A competitor is scored on one thing —
**pieces placed per minute** — and improves by **comparing sessions over
time**. Neither is a first-class output yet.

This design adds five capabilities, almost entirely on top of data the
pipeline already computes (per-hand activity, bimanual split, fatigue, the
dynamic-board cluster history and milestones). The guiding constraint:
**never imply more certainty than the heuristic data supports** — every new
number that rides on heuristic piece/cluster data carries a confidence flag
or an honest label, and degrades to "not available" rather than fabricating
a value.

## Features

1. **Placement rate + splits** — headline pieces/min and milestone split times.
2. **Elite benchmark overlay** — editable, confidence-tagged reference data.
3. **Efficiency + stalls** — productive-vs-dead time, plus dead-spot detection.
4. **Flip/prep phase detection** — separate the opening face-up/sort phase
   from real assembly grip cycles.
5. **Cross-session progress tracking** — a standalone trends command.

Features 1, 3, and 4 are interdependent through the **assembly-start
timestamp**, so that boundary is computed first and shared.

---

## Component: assembly-onset boundary (`puzzle_clusters.py`)

The keystone. The flip/prep phase runs from session start to
`assembly_start_t`; the assembly phase runs from there to the end.

`assembly_start_t` is derived from the main cluster's growth history
(`history_counts` / `history_seconds`, already tracked): the first survey
where the main cluster begins **sustained** net growth — count climbs and
keeps climbing — *not* a single transient join. This is intentionally
stricter than the existing `first_piece_joined` milestone, which can fire on
one early piece.

**Confidence** (mirrors the existing `cluster_summary.detection_quality` /
`quality_notes` pattern):

- `high` — board/clusters detected, clean growth onset
- `medium` — onset ambiguous (slow ramp) or tiebreak was used to pick the
  main cluster
- `low` / `unavailable` — no board or no usable clusters → boundary reported
  as not detected, with a note; never a fabricated timestamp

Output: `assembly_start_t` (seconds, or `None`), plus `confidence` and an
optional `note`, exposed on `cluster_summary` so `puzzle_hands.py` can
consume it.

---

## Component: flip/prep phase block (`puzzle_hands.py`, Feature 4)

When `assembly_start_t` is available, emit a top-level `flip_phase` block:

- `duration_s`, `end_t`, `confidence`, `note`
- **`manipulations`** — count of grip-like contacts whose onset falls before
  `assembly_start_t`. Labelled **"manipulations (flipping/sorting)"** in all
  output — NOT "flips." The landmark data cannot prove a given contact was a
  flip vs. a sort-grip, so the honest label is mandatory.
- `manipulations_per_min`. (A phase-scoped bimanual split was *deferred* from
  Plan 1; the session-level bimanual numbers remain in `summary["bimanual"]`.)

**Effect on existing metrics:** the whole-session `pace` / grip metrics are
left unchanged (nothing breaks). The **placement rate is split** into an
overall (whole-session) and an assembly-phase variant
(`overall_pieces_per_min` / `assembly_pieces_per_min`, the latter computed
over `assembly_start_t → end`), so the real *placing* pace is not diluted by
the flipping warm-up. This split is part of Feature 1. The **combined grip
pace** is also split by phase (`pace.prep_grips_per_min` /
`pace.assembly_grips_per_min`, boundary at `flip_phase.end_t`) for the
grip-tempo view; these live in the JSON and are consumed by the
analyst/trends tooling (not surfaced on the one-pager, whose competition
panel is at its 4-line layout limit). Per-hand pace remains whole-session
only.

When `assembly_start_t` is `None`, `flip_phase` is reported as not detected
(with the note) and `assembly_pieces_per_min` falls back to the overall
value.

---

## Component: placement rate + splits (`puzzle_hands.py` + `puzzle_report.py`, Feature 1)

**Backbone data source:** the main assembly cluster's count-over-time
(`history_counts` / `history_seconds`) is the most robust "pieces assembled
at time *t*" signal — more reliable than summing individual heuristic piece
tracks. Cross-checked against board-zone `placements` when available.

**Headline numbers** (both reported side by side):

- `overall_pieces_per_min` = final assembled count ÷ total session minutes
  (comparable to competition total time, which includes prep)
- `assembly_pieces_per_min` = pieces placed ÷ assembly-phase minutes
  (placing skill only, excludes flip/prep)
- `percent_complete` = final assembled count ÷ `--pieces N` — only when N is
  supplied

**Splits:** time-to-25/50/75/100% of assembled pieces, read off the
cumulative cluster-count curve. The PDF also gains a placement-rate curve.

**Degradation:** if cluster `detection_quality` is `low` or clusters are
unavailable, the entire block reports "not available" with a note.

---

## Component: elite benchmark (`benchmarks.json` + `puzzle_report.py`, Feature 2)

New `benchmarks.json` shipped in the bundle. Per-metric schema:

```json
{
  "metric_key": {
    "value": 1.4,
    "source": "Tammy McLeod (Guinness 500pc), 2 solo sessions",
    "n": 2,
    "confidence": "robust",        // robust | weak | artifact
    "direction": "higher_is_better",
    "note": "modest but real bimanual symmetry edge"
  }
}
```

Seeded from the measured Tammy data (see
`reference_elite-puzzler-baselines.md`):

- `dominance_ratio` → **robust**
- fatigue-resistance (pace decline start→end) → **robust**
- `parallel_active_pct` → **weak** (reversed in the data — elite is not
  higher)
- non-dominant-hand activity → **weak** (reversed)
- mean grip hold → **artifact** (≈8s would mean continuous gripping)
- absolute combined pace → **weak** (recorded sessions below competition
  ceiling)

The report overlays the puzzler's value vs. the reference **with the
confidence tag visible**: robust signals shown prominently, weak/artifact
shown muted with the caveat. Never rendered as a hard "elite threshold"
line. The user edits the JSON to grow it (add solvers, adjust tags) without
touching code.

---

## Component: efficiency + stalls (`puzzle_hands.py` + `puzzle_report.py`, Feature 3)

Promote the existing `both_idle_pct` / `active_pct` to headline
**"productive engagement %"** and **"dead time %."**

**Stall detection:** scan the cumulative placement curve for flat spots — no
assembled growth for longer than a threshold (default ~30s). Report:

- `stall_count`
- `longest_stall_s`
- per-stall location: `{ start_t, duration_s, percent_complete_at_stall }`

Rides entirely on the cluster-count data already trusted. Degrades with the
placement block when clusters are unavailable.

---

## Component: progress tracking (`puzzle_trends.py`, Feature 5)

New standalone script matching the `puzzle_report.py` pattern. Given a
folder of analysis outputs:

1. Load every `*_metrics.json` found beneath the folder.
2. Group by `puzzle_name`.
3. Within each group, sort chronologically. The metrics summary currently
   has **no timestamp field**, so this design adds a `created` ISO-8601
   timestamp to the summary dict in `puzzle_hands.py` going forward.
   `puzzle_trends.py` sorts by `created` when present and falls back to the
   `*_metrics.json` file modification time for older runs that predate the
   field.
4. Emit a trend chart (PNG + PDF) of the headline metrics over time, per
   puzzle: combined pace, `overall_pieces_per_min`, `assembly_pieces_per_min`,
   dead-time %, fatigue (pace decline), `dominance_ratio`.

Older runs missing the newer fields are handled gracefully — the metric is
simply absent from that point on the line, not an error. A puzzle group with
a single session still renders (one point).

CLI: `python3 puzzle_trends.py <folder> [--puzzle NAME] [--outdir DIR]`.

---

## Cross-cutting concerns

**Graceful degradation.** No board/clusters, or no `--pieces`, → affected
blocks report "not available" with a note, consistent with the existing
`cluster_summary.quality_notes` mechanism. Nothing crashes; nothing is
fabricated.

**Honest naming.** "manipulations (flipping/sorting)", confidence tags on
benchmarks, "not available" notes — no metric implies more certainty than
the heuristic data supports.

**Testing.** Unit tests for the new pure functions, using synthetic
cluster-history fixtures (no video required):

- assembly-onset boundary detection (clean ramp, transient-join, no-growth,
  ambiguous-ramp cases)
- placement rate + splits computation
- stall detection (flat-spot identification, threshold edges)
- benchmark loading + confidence tagging
- trends aggregation (grouping, chronological sort, missing-field tolerance)

Added to the existing `tests/` suite.

**Build / packaging.** Add `benchmarks.json` and `puzzle_trends.py` to the
`FILES` list in `build_release.sh`.

**Cross-platform.** All new file I/O uses `encoding="utf-8"`; any CSV uses
`newline=""`; charts use the Agg backend — consistent with the rest of the
codebase, so Windows/Linux/Mac parity is preserved.

## Out of scope (parked as future work)

- **Edge-first / border-phase strategy detection** — needs piece-shape
  reasoning the heuristic tracker can't reliably support. Real value, low
  feasibility.
- **Color-sort / pile organization quality** — same constraint.
- **GUI integration of trends** — `puzzle_trends.py` is CLI-first; a History
  tab in the Tkinter app can come later.
