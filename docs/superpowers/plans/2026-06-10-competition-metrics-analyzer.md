# Competition Metrics (Analyzer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add competition-relevant metrics to the analyzer — a flip/prep-phase split, pieces-per-minute with milestone splits, and an efficiency/dead-time + stall report — all derived from the cluster growth-history the pipeline already tracks.

**Architecture:** Three pure helper functions are added to `puzzle_clusters.py` (assembly-onset, placement-splits, stall detection), unit-tested with the existing synthetic-track fixtures. `puzzle_hands.py` consumes them to assemble new `flip_phase`, `placement`, and `efficiency` blocks in the metrics summary, plus a `created` timestamp. `puzzle_report.py` gains panels that render these blocks, degrading to "not available" when cluster data is missing.

**Tech Stack:** Python 3.8+, NumPy, OpenCV, Matplotlib (Agg), `unittest` (run via `pytest` or `python -m unittest`).

**Scope note:** This is Plan 1 of 3. Plan 2 (elite benchmark) and Plan 3 (`puzzle_trends.py`) are separate documents; Plan 3 depends on the new summary fields created here.

**Data shapes (confirmed from existing code):**
- A cluster track has `history_seconds: list[float]` and `history_counts: list[int]` (parallel arrays), plus `final_count`, `peak_count`, `is_main`, `bbox_at_peak`. See `tests/test_puzzle_clusters.py` `_mapper_with_growth_history` / `_track` for the canonical synthetic shape.
- `ClusterMapper.finalize(n_frames)` returns `(kept_clusters, milestones)` and populates `self.last_summary` (a dict with `detection_quality`, `quality_notes`, `main_cluster_peak_count`, `tiebreak_used`, ...).
- `puzzle_hands.analyze(...)` builds `summary` (puzzle_hands.py:543-571), attaches cluster data (573-605), writes JSON at `puzzle_hands.py:609`, returns at `:658`.

---

## File Structure

- `puzzle_clusters.py` — **modify**: add three module-level pure functions (`assembly_onset`, `placement_splits`, `detect_stalls`). Module-level (not methods) so they are trivially unit-testable and reusable by `puzzle_trends.py` later.
- `tests/test_puzzle_clusters.py` — **modify**: add test classes for the three new functions.
- `puzzle_hands.py` — **modify**: consume the helpers; add `created`, `flip_phase`, `placement`, `efficiency` to `summary`; add assembly-phase pace variant.
- `tests/test_competition_metrics.py` — **create**: unit tests for the pure `puzzle_hands` helpers (flip-manipulation count, assembly-phase pace, pieces-per-min).
- `puzzle_report.py` — **modify**: render the new blocks in `make_one_pager`.

---

## Task 1: `assembly_onset` helper

**Files:**
- Modify: `puzzle_clusters.py` (add module-level function near the other primitives, after `dbscan_2d`)
- Test: `tests/test_puzzle_clusters.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_puzzle_clusters.py`:

```python
class TestAssemblyOnset(unittest.TestCase):

    def test_clean_phase_boundary_high_confidence(self):
        from puzzle_clusters import assembly_onset
        # Flat near-zero pile-flip phase, then sustained growth at t=60
        secs = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        cnts = [0,    1,    1,    5,    20,   40]
        t, conf, note = assembly_onset(secs, cnts)
        self.assertEqual(t, 60.0)
        self.assertEqual(conf, "high")

    def test_gradual_ramp_is_medium(self):
        from puzzle_clusters import assembly_onset
        # Counts already above the floor at the first survey -> we cannot
        # observe a clean pre-phase, so confidence is medium.
        secs = [0.0, 20.0, 40.0, 60.0]
        cnts = [4,   8,    14,   22]
        t, conf, note = assembly_onset(secs, cnts)
        self.assertEqual(t, 0.0)
        self.assertEqual(conf, "medium")

    def test_no_sustained_growth_unavailable(self):
        from puzzle_clusters import assembly_onset
        # Count never climbs past the floor in a sustained way
        secs = [0.0, 20.0, 40.0, 60.0]
        cnts = [0,   1,    0,    1]
        t, conf, note = assembly_onset(secs, cnts)
        self.assertIsNone(t)
        self.assertEqual(conf, "unavailable")

    def test_empty_history_unavailable(self):
        from puzzle_clusters import assembly_onset
        t, conf, note = assembly_onset([], [])
        self.assertIsNone(t)
        self.assertEqual(conf, "unavailable")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_puzzle_clusters.TestAssemblyOnset -v`
Expected: FAIL with `ImportError: cannot import name 'assembly_onset'`

- [ ] **Step 3: Implement `assembly_onset`**

Add to `puzzle_clusters.py` (module level, after `dbscan_2d`):

```python
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
        return history_seconds[i], conf, note
    return None, "unavailable", "no sustained growth detected"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_puzzle_clusters.TestAssemblyOnset -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add puzzle_clusters.py tests/test_puzzle_clusters.py
git commit -m "Add assembly_onset helper for flip/prep-phase boundary"
```

---

## Task 2: `placement_splits` helper

**Files:**
- Modify: `puzzle_clusters.py`
- Test: `tests/test_puzzle_clusters.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestPlacementSplits(unittest.TestCase):

    def test_quartile_splits(self):
        from puzzle_clusters import placement_splits
        # num_pieces=100; counts reach 25 at t=40, 50 at t=60, 75 at t=80,
        # 100 at t=100
        secs = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        cnts = [0,   10,   25,   50,   75,   100]
        out = placement_splits(secs, cnts, num_pieces=100)
        self.assertEqual(out["25pct"], 40.0)
        self.assertEqual(out["50pct"], 60.0)
        self.assertEqual(out["75pct"], 80.0)
        self.assertEqual(out["100pct"], 100.0)

    def test_unreached_split_is_none(self):
        from puzzle_clusters import placement_splits
        secs = [0.0, 20.0, 40.0]
        cnts = [0,   10,   30]      # never reaches 50 of 100
        out = placement_splits(secs, cnts, num_pieces=100)
        self.assertEqual(out["25pct"], 40.0)
        self.assertIsNone(out["50pct"])

    def test_no_num_pieces_returns_empty(self):
        from puzzle_clusters import placement_splits
        out = placement_splits([0.0, 10.0], [0, 5], num_pieces=None)
        self.assertEqual(out, {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_puzzle_clusters.TestPlacementSplits -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `placement_splits`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_puzzle_clusters.TestPlacementSplits -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add puzzle_clusters.py tests/test_puzzle_clusters.py
git commit -m "Add placement_splits helper for milestone split times"
```

---

## Task 3: `detect_stalls` helper

**Files:**
- Modify: `puzzle_clusters.py`
- Test: `tests/test_puzzle_clusters.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestDetectStalls(unittest.TestCase):

    def test_finds_a_flat_spot(self):
        from puzzle_clusters import detect_stalls
        # Grows, then flat from t=40 to t=120 (80s, no new pieces), then grows
        secs = [0.0, 20.0, 40.0, 80.0, 120.0, 140.0]
        cnts = [0,   10,   30,   30,   30,    45]
        stalls = detect_stalls(secs, cnts, min_gap_s=30.0)
        self.assertEqual(len(stalls), 1)
        self.assertEqual(stalls[0]["start_t"], 40.0)
        self.assertEqual(stalls[0]["duration_s"], 80.0)
        self.assertEqual(stalls[0]["count_at_stall"], 30)

    def test_ignores_short_pauses(self):
        from puzzle_clusters import detect_stalls
        # 20s flat is below the 30s threshold
        secs = [0.0, 20.0, 40.0, 60.0]
        cnts = [0,   10,   10,   25]
        stalls = detect_stalls(secs, cnts, min_gap_s=30.0)
        self.assertEqual(stalls, [])

    def test_steady_growth_has_no_stalls(self):
        from puzzle_clusters import detect_stalls
        secs = [0.0, 20.0, 40.0, 60.0]
        cnts = [0,   10,   20,   30]
        self.assertEqual(detect_stalls(secs, cnts, min_gap_s=30.0), [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_puzzle_clusters.TestDetectStalls -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `detect_stalls`**

```python
STALL_MIN_GAP_S = 30.0


def detect_stalls(history_seconds, history_counts, min_gap_s=STALL_MIN_GAP_S):
    """Flat spots in the assembled-count curve: maximal spans during which
    the count never exceeds its value at the span start, lasting longer than
    min_gap_s. Returns a list of
    {start_t, duration_s, count_at_stall} dicts."""
    stalls = []
    n = len(history_counts)
    i = 0
    while i < n - 1:
        # extend while no new growth beyond the count at i
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
            i = j  # resume from where growth resumed
        else:
            i += 1
    return stalls
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_puzzle_clusters.TestDetectStalls -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add puzzle_clusters.py tests/test_puzzle_clusters.py
git commit -m "Add detect_stalls helper for dead-spot detection"
```

---

## Task 4: `puzzle_hands` pure helpers (manipulation count, assembly-phase pace, pieces/min)

These are extracted as small module-level pure functions in `puzzle_hands.py` so they can be unit-tested without running video.

**Files:**
- Modify: `puzzle_hands.py` (add module-level helpers near the other top-level helpers, before `analyze`)
- Test: `tests/test_competition_metrics.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_competition_metrics.py`:

```python
"""Tests for the competition-metric pure helpers in puzzle_hands."""
import unittest

from puzzle_hands import (
    count_before, pieces_per_min,
)


class TestCountBefore(unittest.TestCase):
    def test_counts_onsets_before_boundary(self):
        # grip onset times in seconds for both hands, pooled
        onset_times = [2.0, 5.0, 9.0, 30.0, 61.0, 90.0]
        self.assertEqual(count_before(onset_times, 60.0), 4)

    def test_none_boundary_counts_all(self):
        onset_times = [2.0, 5.0, 9.0]
        self.assertEqual(count_before(onset_times, None), 0)


class TestPiecesPerMin(unittest.TestCase):
    def test_overall_and_assembly(self):
        # final assembled 80 pieces; total 1200s; onset at 300s, count 5 there
        res = pieces_per_min(final_count=80, total_s=1200.0,
                             onset_t=300.0, count_at_onset=5)
        self.assertAlmostEqual(res["overall_pieces_per_min"], 4.0)   # 80/20min
        # assembly: (80-5)=75 pieces over (1200-300)=900s = 15min -> 5.0
        self.assertAlmostEqual(res["assembly_pieces_per_min"], 5.0)

    def test_no_onset_falls_back_to_overall(self):
        res = pieces_per_min(final_count=80, total_s=1200.0,
                             onset_t=None, count_at_onset=0)
        self.assertAlmostEqual(res["overall_pieces_per_min"], 4.0)
        self.assertAlmostEqual(res["assembly_pieces_per_min"], 4.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_competition_metrics -v`
Expected: FAIL with `ImportError: cannot import name 'count_before'`

- [ ] **Step 3: Implement the helpers**

Add to `puzzle_hands.py` (module level, before `def analyze`):

```python
def count_before(onset_times, boundary_t):
    """How many grip onsets occurred before boundary_t. If boundary_t is
    None (no flip phase detected) returns 0 — nothing is attributed to a
    phase we couldn't bound."""
    if boundary_t is None:
        return 0
    return sum(1 for t in onset_times if t < boundary_t)


def pieces_per_min(final_count, total_s, onset_t, count_at_onset):
    """Overall and assembly-phase placement rates.

    overall: final_count over the whole session.
    assembly: pieces added after onset, over the post-onset duration.
    Falls back to overall when onset_t is None or degenerate.
    """
    total_min = max(total_s, 1e-6) / 60.0
    overall = final_count / total_min
    if onset_t is None or (total_s - onset_t) <= 1e-6:
        assembly = overall
    else:
        asm_min = (total_s - onset_t) / 60.0
        assembly = (final_count - count_at_onset) / asm_min
    return {
        "overall_pieces_per_min": round(float(overall), 2),
        "assembly_pieces_per_min": round(float(assembly), 2),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_competition_metrics -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add puzzle_hands.py tests/test_competition_metrics.py
git commit -m "Add pure helpers for manipulation count and placement rate"
```

---

## Task 5: Assemble `flip_phase`, `placement`, `efficiency` blocks in the summary

Wire the helpers into `analyze()`. This is integration: it reads the main cluster's history, the pooled grip-onset times, and the existing `bimanual` numbers, and attaches three new blocks plus a `created` timestamp.

**Files:**
- Modify: `puzzle_hands.py` — inside `analyze`, after cluster finalize (`:426-428`) and before/within the `summary` assembly (`:543-605`)

**Context you need:**
- `cluster_history` (list of cluster dicts) and `cluster_summary` exist by line 428. The main cluster is the one with `is_main == True`.
- Grip onsets: the per-hand grip boolean series and the `onsets()` helper already exist (used around `puzzle_hands.py:397, 409`). Pool left+right onset frame indices, convert to seconds via `eff_fps`.
- `dur_s` (session seconds) and `eff_fps` are in scope. `num_pieces` is in scope.
- `summary["bimanual"]["both_idle_pct"]` is computed at `:555`.

- [ ] **Step 1: Add the integration code**

In `analyze()`, after the cluster block populates `cluster_summary`/`cluster_history` and after the per-frame series exist, compute:

```python
# ---- competition metrics: flip phase, placement rate, efficiency --------
from puzzle_clusters import assembly_onset, placement_splits, detect_stalls
from datetime import datetime, timezone

main_cluster = None
if cluster_history:
    main_cluster = next((c for c in cluster_history if c.get("is_main")), None)

flip_phase = None
placement = None
stalls = []
onset_t = None
count_at_onset = 0
if main_cluster is not None and main_cluster.get("history_counts"):
    hs = main_cluster["history_seconds"]
    hc = main_cluster["history_counts"]
    onset_t, onset_conf, onset_note = assembly_onset(hs, hc)
    # downgrade confidence if cluster detection quality is weak
    if cluster_summary and cluster_summary.get("detection_quality") == "low":
        onset_conf, onset_note = "low", "cluster detection quality is low"
    if onset_t is not None:
        count_at_onset = next((c for t, c in zip(hs, hc) if t >= onset_t), 0)

    # pooled grip onsets in seconds (both hands). `series` is the per-hand
    # dict built at puzzle_hands.py:364; grip state key is "gripping".
    onset_times = sorted(
        list(onsets(series["left"]["gripping"]) / eff_fps) +
        list(onsets(series["right"]["gripping"]) / eff_fps))
    manips = count_before(onset_times, onset_t)
    flip_dur = onset_t if onset_t is not None else 0.0
    flip_phase = {
        "end_t": round(onset_t, 1) if onset_t is not None else None,
        "duration_s": round(flip_dur, 1),
        "confidence": onset_conf if onset_t is not None else "unavailable",
        "note": onset_note,
        "manipulations": manips,
        "manipulations_per_min": (round(manips / flip_dur * 60, 1)
                                  if flip_dur > 1e-6 else None),
    }

    final_count = main_cluster.get("final_count", hc[-1] if hc else 0)
    placement = dict(pieces_per_min(final_count, dur_s, onset_t, count_at_onset))
    placement["final_assembled"] = int(final_count)
    if num_pieces:
        placement["percent_complete"] = round(100.0 * final_count / num_pieces, 1)
    placement["splits"] = placement_splits(hs, hc, num_pieces)
    stalls = detect_stalls(hs, hc)

efficiency = {
    "dead_time_pct": summary["bimanual"]["both_idle_pct"],
    "productive_pct": round(100.0 - summary["bimanual"]["both_idle_pct"], 1),
    "stall_count": len(stalls),
    "longest_stall_s": (max(s["duration_s"] for s in stalls) if stalls else 0.0),
    "stalls": stalls,
}

summary["created"] = datetime.now(timezone.utc).isoformat()
if flip_phase is not None:
    summary["flip_phase"] = flip_phase
else:
    summary["flip_phase"] = {"confidence": "unavailable",
                             "note": "no assembly cluster detected"}
if placement is not None:
    summary["placement"] = placement
summary["efficiency"] = efficiency
```

> NOTE: confirmed against `puzzle_hands.py:364` — the per-hand series dict is `series["left"]`/`series["right"]` with grip key `"gripping"`, and `onsets(bool_array)` returns frame indices (see `slice_metrics`, `:376-377`). `summary["efficiency"]` is always emitted (it only needs `both_idle_pct`, which always exists); `flip_phase`/`placement` degrade gracefully.

- [ ] **Step 2: Smoke-test on an existing run**

Run against the smallest existing analysis video-free path. Since `analyze` needs video, instead verify the JSON shape by running the full CLI on a short clip if available, or rely on Task 4 unit tests for the pure logic. Minimum check — import stays clean:

Run: `python -c "import puzzle_hands"`
Expected: no error.

- [ ] **Step 3: Run the full test suite**

Run: `python -m unittest discover -s tests -v`
Expected: PASS (all existing + new tests).

- [ ] **Step 4: Commit**

```bash
git add puzzle_hands.py
git commit -m "Emit flip_phase, placement, efficiency blocks + created timestamp"
```

---

## Task 6: Render the new blocks in the one-pager report

Add display panels to `make_one_pager` (`puzzle_report.py:268`). Matplotlib layout isn't unit-tested; verify by regenerating a report on an existing analysis folder and eyeballing the PNG.

**Files:**
- Modify: `puzzle_report.py` — `make_one_pager` (reads `summary` dict)

- [ ] **Step 1: Add a competition-metrics text/figure panel**

Inside `make_one_pager`, after the existing panels, add a block that reads `summary.get("placement")`, `summary.get("flip_phase")`, `summary.get("efficiency")` and renders:
- Headline: `overall_pieces_per_min` and `assembly_pieces_per_min` side by side; `percent_complete` if present.
- Splits row: 25/50/75/100% times (format with the existing `_fmt_time`-style helper if present in report, else `%.0f s`/`%.1f min`).
- Flip phase: "Flip/prep: <duration> · <manipulations> manipulations (flipping/sorting) · confidence <tag>". When `flip_phase["confidence"] == "unavailable"`, render "Flip/prep phase: not detected (<note>)".
- Efficiency: "Productive <productive_pct>% · Dead time <dead_time_pct>% · Stalls: <stall_count> (longest <longest_stall_s>s)".

Each value guarded with `.get(...)` and an "n/a" fallback so a summary lacking the new blocks (older runs) still renders.

```python
# --- competition metrics panel ---
pl = summary.get("placement")
fp = summary.get("flip_phase", {})
ef = summary.get("efficiency", {})
lines = []
if pl:
    line = ("Placing pace:  %.2f pieces/min overall   ·   %.2f pieces/min assembly"
            % (pl.get("overall_pieces_per_min", 0),
               pl.get("assembly_pieces_per_min", 0)))
    if "percent_complete" in pl:
        line += "   ·   %.0f%% complete" % pl["percent_complete"]
    lines.append(line)
    sp = pl.get("splits", {})
    if sp:
        def _s(v):
            return "—" if v is None else "%.0fs" % v
        lines.append("Splits:  25%% %s   50%% %s   75%% %s   100%% %s" % (
            _s(sp.get("25pct")), _s(sp.get("50pct")),
            _s(sp.get("75pct")), _s(sp.get("100pct"))))
if fp:
    if fp.get("confidence") == "unavailable":
        lines.append("Flip/prep phase: not detected (%s)" % fp.get("note", ""))
    else:
        lines.append("Flip/prep: %.0fs · %d manipulations (flipping/sorting) · confidence %s"
                     % (fp.get("duration_s", 0), fp.get("manipulations", 0),
                        fp.get("confidence", "?")))
if ef:
    lines.append("Productive %.0f%% · Dead time %.0f%% · Stalls: %d (longest %.0fs)"
                 % (ef.get("productive_pct", 0), ef.get("dead_time_pct", 0),
                    ef.get("stall_count", 0), ef.get("longest_stall_s", 0)))
# draw `lines` into a dedicated axis using ax.text(), matching the style of
# the existing text panels in make_one_pager (fontsize ~10, monospace).
```

- [ ] **Step 2: Regenerate a report on an existing analysis**

Pick an existing folder that has cluster data (e.g. one of the `RollingHills_*_analysis` runs with a `*_metrics.json`). Note: older runs won't have the new blocks — to exercise the new panel, first re-run the analyzer (Task 5) on a short clip, OR hand-add the new keys to a copy of a metrics JSON for a one-off visual check.

Run: `python3 puzzle_report.py <folder>/<base>_metrics.json`
Expected: a `<base>_report.png` regenerates without error; the new panel shows "not detected"/"n/a" gracefully on an old JSON.

- [ ] **Step 3: Commit**

```bash
git add puzzle_report.py
git commit -m "Render placement/flip-phase/efficiency panel in one-pager"
```

---

## Task 7: Full-suite green + end-to-end sanity

- [ ] **Step 1: Run the whole suite**

Run: `python -m unittest discover -s tests -v`
Expected: PASS — all existing tests plus `TestAssemblyOnset`, `TestPlacementSplits`, `TestDetectStalls`, and `tests/test_competition_metrics.py`.

- [ ] **Step 2: End-to-end on a real clip (manual)**

Run the analyzer on a short real clip with a known board so cluster data is produced:

Run: `python3 puzzle_hands.py <clip>.mp4 --pieces 100 --no-video`
Expected: the written `*_metrics.json` contains `created`, `flip_phase`, `placement` (with `overall_pieces_per_min`, `assembly_pieces_per_min`, `splits`), and `efficiency` (with `stalls`). Confirm `flip_phase.confidence` is one of high/medium/low/unavailable and the manipulation label reads "flipping/sorting", not "flips".

- [ ] **Step 3: Final commit if any fixups were needed**

```bash
git add -A
git commit -m "Competition metrics: end-to-end fixups"
```

---

## Self-Review checklist (done while writing)

- **Spec coverage:** flip-phase detection (Tasks 1, 5), assembly-phase pace variant (Task 4/5), placement rate both overall+assembly (Task 4/5), splits (Task 2/5), efficiency headline (Task 5), stalls (Task 3/5), `created` timestamp for Plan 3 (Task 5), graceful degradation + honest naming (Tasks 5/6). Benchmark (Feature 2) and trends (Feature 5) are deliberately in Plans 2 and 3.
- **Type consistency:** `assembly_onset` returns `(t, conf, note)` everywhere; `placement_splits` keys are `"25pct".."100pct"` in both helper and report; `detect_stalls` dict keys `start_t/duration_s/count_at_stall` match between Task 3 and Task 5; `pieces_per_min` keys `overall_pieces_per_min`/`assembly_pieces_per_min` match Task 4 ↔ Task 6.
- **Grip-series access (verified):** per-hand series is `series["left"]/["right"]`, grip key `"gripping"`, and `onsets(bool_array)` returns frame indices — confirmed against `puzzle_hands.py:364, 376-377`. No open assumptions remain.
