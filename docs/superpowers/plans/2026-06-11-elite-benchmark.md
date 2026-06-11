# Elite Benchmark Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Compare a session's metrics against an editable, confidence-tagged reference of measured elite numbers, rendered as a second page of the report PDF.

**Architecture:** A shipped `benchmarks.json` holds measured reference values, each tagged `robust`/`weak`/`artifact` with a dotted `path` into the metrics summary. Pure helpers in `puzzle_report.py` load it, extract the session's value per path, and build comparison rows (unit-tested). `make_all`/`make_one_pager` render those rows on a new second PDF page (and a `*_benchmark.png`), with robust signals prominent and weak/artifact muted — never as hard "elite thresholds."

**Tech Stack:** Python 3 (run as `python3` — system `python` is 2.7), Matplotlib (Agg) with `matplotlib.backends.backend_pdf.PdfPages` for multipage output, `unittest`.

**Honesty constraint (from prior user feedback):** baselines are MEASURED (N=2 Tammy McLeod + context), not invented. Only `dominance_ratio` and `fatigue_decline` are robust; `parallel_active` reversed in data; `grip_hold` is an artifact. The report must show the confidence tag and N, and must NOT present any value as a pass/fail threshold.

**Confirmed summary key-paths** (verified against an existing metrics JSON):
`bimanual.dominance_ratio`, `bimanual.parallel_active_pct`, `bimanual.both_idle_pct`, `pace.combined_grips_per_min`, `fatigue.grips_per_min_change_pct`, `right.mean_grip_hold_s`.

**Report facts:** `make_one_pager(summary, windows, sa, base)` (puzzle_report.py:304) builds one `fig=plt.figure(figsize=(8.5,11))` and saves `base+"_report.pdf"` and `base+"_report.png"` (lines 584-585). `make_all(metrics_path, video_path, n_windows)` at :590 is the entry.

---

## File Structure
- `benchmarks.json` — **create** (repo root, next to puzzle_report.py). Shipped in the bundle.
- `build_release.sh` — **modify**: add `benchmarks.json` to `FILES`.
- `puzzle_report.py` — **modify**: add pure `load_benchmarks`, `_get_by_path`, `benchmark_rows`; add a benchmark-page figure builder; switch PDF writing to multipage and emit `*_benchmark.png`.
- `tests/test_benchmark.py` — **create**: unit tests for the pure helpers.

---

## Task 1: `benchmarks.json` + bundle inclusion

**Files:** Create `benchmarks.json`; Modify `build_release.sh`.

- [ ] **Step 1: Create `benchmarks.json`** with this exact content:

```json
{
  "dominance_ratio": {
    "path": "bimanual.dominance_ratio",
    "label": "Bimanual symmetry (dominance ratio)",
    "value": 1.4,
    "source": "Tammy McLeod (Guinness 500pc), 2 solo sessions",
    "n": 2,
    "confidence": "robust",
    "direction": "lower_is_better",
    "note": "lower = more two-handed symmetry; elite ~1.4 vs a casual top-3% solver ~2.25"
  },
  "fatigue_decline_pct": {
    "path": "fatigue.grips_per_min_change_pct",
    "label": "Fatigue resistance (pace change start to end)",
    "value": -36,
    "source": "Tammy McLeod, 2 solo sessions",
    "n": 2,
    "confidence": "robust",
    "direction": "higher_is_better",
    "note": "less negative = holds pace better through the session; elite around -36%"
  },
  "combined_grips_per_min": {
    "path": "pace.combined_grips_per_min",
    "label": "Combined grip pace (grips/min)",
    "value": 24,
    "source": "Tammy McLeod recorded sessions",
    "n": 2,
    "confidence": "weak",
    "direction": "higher_is_better",
    "note": "recorded casually; her competition ceiling is ~40. Loose reference only."
  },
  "parallel_active_pct": {
    "path": "bimanual.parallel_active_pct",
    "label": "Parallel-active (both hands working at once)",
    "value": 23,
    "source": "Tammy McLeod, 2 sessions",
    "n": 2,
    "confidence": "weak",
    "direction": null,
    "note": "REVERSED in the data vs a top-3% solver; not a reliable elite marker"
  },
  "both_idle_pct": {
    "path": "bimanual.both_idle_pct",
    "label": "Both hands idle",
    "value": 35,
    "source": "Tammy McLeod, 2 sessions",
    "n": 2,
    "confidence": "weak",
    "direction": "lower_is_better",
    "note": "weak signal"
  },
  "mean_grip_hold_s": {
    "path": "right.mean_grip_hold_s",
    "label": "Mean grip hold (s)",
    "value": 7.0,
    "source": "Tammy McLeod footage",
    "n": 2,
    "confidence": "artifact",
    "direction": null,
    "note": "almost certainly an analyzer artifact on her footage; do not trust"
  }
}
```

- [ ] **Step 2: Validate it parses**

Run: `python3 -c "import json; d=json.load(open('benchmarks.json')); print(len(d), 'metrics')"`
Expected: `6 metrics`

- [ ] **Step 3: Add to build_release.sh FILES**

In `build_release.sh`, add `benchmarks.json` to the `FILES=( ... )` array (e.g. right after `requirements.txt`).

- [ ] **Step 4: Verify the build includes it**

Run: `./build_release.sh 2>&1 | grep benchmarks.json`
Expected: a line showing `PuzzleAnalyzer/benchmarks.json`.

- [ ] **Step 5: Commit** (stage only `benchmarks.json` and `build_release.sh`; never `git add -A`)

```bash
git add benchmarks.json build_release.sh
git commit -m "Add benchmarks.json (measured elite reference) + bundle it"
```

---

## Task 2: pure helpers `load_benchmarks`, `_get_by_path`, `benchmark_rows`

**Files:** Modify `puzzle_report.py` (add near other module helpers, e.g. after `_competition_lines`); Create `tests/test_benchmark.py`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_benchmark.py`:

```python
"""Tests for the elite-benchmark comparison helpers."""
import unittest
from puzzle_report import load_benchmarks, _get_by_path, benchmark_rows


class TestGetByPath(unittest.TestCase):
    def test_nested_lookup(self):
        s = {"bimanual": {"dominance_ratio": 2.25}}
        self.assertEqual(_get_by_path(s, "bimanual.dominance_ratio"), 2.25)

    def test_missing_returns_none(self):
        self.assertIsNone(_get_by_path({"bimanual": {}}, "bimanual.dominance_ratio"))
        self.assertIsNone(_get_by_path({}, "pace.combined_grips_per_min"))


class TestLoadBenchmarks(unittest.TestCase):
    def test_loads_shipped_file(self):
        b = load_benchmarks()
        self.assertIn("dominance_ratio", b)
        self.assertEqual(b["dominance_ratio"]["confidence"], "robust")

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_benchmarks("/no/such/benchmarks.json"), {})


class TestBenchmarkRows(unittest.TestCase):
    def _bench(self):
        return {
            "dominance_ratio": {"path": "bimanual.dominance_ratio",
                "label": "Dom", "value": 1.4, "n": 2, "confidence": "robust",
                "direction": "lower_is_better", "note": "x"},
            "combined_grips_per_min": {"path": "pace.combined_grips_per_min",
                "label": "Pace", "value": 24, "n": 2, "confidence": "weak",
                "direction": "higher_is_better", "note": "y"},
            "mean_grip_hold_s": {"path": "right.mean_grip_hold_s",
                "label": "Hold", "value": 7.0, "n": 2, "confidence": "artifact",
                "direction": None, "note": "z"},
        }

    def _summary(self):
        return {"bimanual": {"dominance_ratio": 2.25},
                "pace": {"combined_grips_per_min": 35.1},
                "right": {"mean_grip_hold_s": 1.45}}

    def test_rows_sorted_robust_first(self):
        rows = benchmark_rows(self._summary(), self._bench())
        confs = [r["confidence"] for r in rows]
        self.assertEqual(confs, ["robust", "weak", "artifact"])

    def test_extracts_session_value_and_verdict(self):
        rows = benchmark_rows(self._summary(), self._bench())
        dom = next(r for r in rows if r["key"] == "dominance_ratio")
        self.assertEqual(dom["you"], 2.25)
        self.assertEqual(dom["elite"], 1.4)
        # lower_is_better and you(2.25) > elite(1.4) -> behind
        self.assertEqual(dom["verdict"], "behind")
        pace = next(r for r in rows if r["key"] == "combined_grips_per_min")
        # higher_is_better and you(35.1) > elite(24) -> ahead
        self.assertEqual(pace["verdict"], "ahead")

    def test_artifact_and_no_direction_have_no_verdict(self):
        rows = benchmark_rows(self._summary(), self._bench())
        hold = next(r for r in rows if r["key"] == "mean_grip_hold_s")
        self.assertIsNone(hold["verdict"])

    def test_missing_session_value_marks_na(self):
        rows = benchmark_rows({}, self._bench())   # no metrics present
        dom = next(r for r in rows if r["key"] == "dominance_ratio")
        self.assertIsNone(dom["you"])
        self.assertIsNone(dom["verdict"])

    def test_empty_benchmarks(self):
        self.assertEqual(benchmark_rows(self._summary(), {}), [])
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest tests.test_benchmark -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement the helpers** in `puzzle_report.py`:

```python
import os

_CONF_ORDER = {"robust": 0, "weak": 1, "artifact": 2}


def load_benchmarks(path=None):
    """Load the elite-reference benchmarks JSON. Returns {} if the file is
    missing or unreadable (graceful — benchmarking is optional)."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmarks.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _get_by_path(summary, dotted):
    """Walk a dotted key path into the summary dict; None if any key is
    missing or a non-dict is hit."""
    cur = summary
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _bench_verdict(you, elite, direction):
    if you is None or direction not in ("lower_is_better", "higher_is_better"):
        return None
    if abs(you - elite) < 1e-9:
        return "even"
    better = (you < elite) if direction == "lower_is_better" else (you > elite)
    return "ahead" if better else "behind"


def benchmark_rows(summary, benchmarks):
    """Build comparison rows from the session summary and the benchmark
    reference. Pure. Each row: {key, label, you, elite, n, confidence,
    direction, verdict, note}. Sorted robust -> weak -> artifact. `you` is
    None when the session lacks that metric (verdict then None)."""
    rows = []
    for key, b in benchmarks.items():
        you = _get_by_path(summary, b.get("path", ""))
        rows.append({
            "key": key,
            "label": b.get("label", key),
            "you": you,
            "elite": b.get("value"),
            "n": b.get("n"),
            "confidence": b.get("confidence", "weak"),
            "direction": b.get("direction"),
            "verdict": _bench_verdict(you, b.get("value"), b.get("direction")),
            "note": b.get("note", ""),
        })
    rows.sort(key=lambda r: _CONF_ORDER.get(r["confidence"], 1))
    return rows
```

> NOTE (verified): `json` (line 28) and `os` (line 29) are ALREADY imported at the top of puzzle_report.py — do NOT re-add either. Add the new functions below the existing helpers.

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m unittest tests.test_benchmark -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit** (stage only `puzzle_report.py`, `tests/test_benchmark.py`)

```bash
git add puzzle_report.py tests/test_benchmark.py
git commit -m "Add pure benchmark-comparison helpers"
```

---

## Task 3: render the benchmark as a second PDF page + benchmark PNG

**Files:** Modify `puzzle_report.py` — add a page-2 figure builder and switch report output to multipage PDF.

- [ ] **Step 1: Add a benchmark-page figure builder** `_make_benchmark_page(summary)` returning a Matplotlib `Figure` (figsize `(8.5, 11)`), or `None` when there are no rows / no benchmarks file. It renders:
  - Title: "Elite benchmark comparison".
  - A subtitle caveat (small, italic-ish): "Reference = measured from N=2 Tammy McLeod sessions + 1 subject. Small sample. Confidence tag shown per row. NOT a pass/fail threshold."
  - A table from `benchmark_rows(summary, load_benchmarks())`, columns: Metric | You | Elite ref | n | Confidence | Note. Robust rows in normal weight; weak/artifact rows greyed (e.g. color `#888`). For `you is None`, show "n/a". Show `verdict` as a small word (ahead/behind/even/blank) only for robust+weak rows that have one.
  - Use `ax.axis("off")` + `ax.table(...)` or manual `ax.text` rows — match the existing report's neutral style. ASCII only.

```python
def _make_benchmark_page(summary):
    rows = benchmark_rows(summary, load_benchmarks())
    if not rows:
        return None
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis("off")
    ax.text(0, 1.0, "Elite benchmark comparison", fontsize=20, weight="bold",
            va="top", transform=ax.transAxes)
    ax.text(0, 0.95,
            "Reference = measured from N=2 Tammy McLeod sessions (+1 subject). "
            "Small sample; confidence tag per row. Not a pass/fail threshold.",
            fontsize=9, color="#555", va="top", transform=ax.transAxes)
    y = 0.88
    for r in rows:
        grey = r["confidence"] in ("weak", "artifact")
        col = "#888888" if grey else "#111111"
        you = "n/a" if r["you"] is None else ("%g" % r["you"])
        verdict = (" [%s]" % r["verdict"]) if r["verdict"] else ""
        head = "%s:  you %s   vs elite %g   (n=%s, %s)%s" % (
            r["label"], you, r["elite"], r["n"], r["confidence"], verdict)
        ax.text(0, y, head, fontsize=11, color=col, va="top",
                weight=("bold" if not grey else "normal"), transform=ax.transAxes)
        ax.text(0.02, y - 0.028, r["note"], fontsize=8, color="#777",
                va="top", transform=ax.transAxes)
        y -= 0.075
    return fig
```

- [ ] **Step 2: Switch PDF output to multipage** in `make_one_pager` (or wherever the one-pager fig is saved). Replace the single `fig.savefig(base+"_report.pdf")` with a `PdfPages` write of page 1 (the existing one-pager fig) then page 2 (`_make_benchmark_page(summary)` if not None). Keep `base+"_report.png"` as page 1 unchanged, and ALSO write the benchmark page to `base+"_benchmark.png"` when it exists.

```python
from matplotlib.backends.backend_pdf import PdfPages   # add to imports at top
# ... at the save site (currently lines ~584-585):
bench_fig = _make_benchmark_page(summary)
with PdfPages(base + "_report.pdf") as pdf:
    pdf.savefig(fig, bbox_inches="tight")
    if bench_fig is not None:
        pdf.savefig(bench_fig, bbox_inches="tight")
fig.savefig(base + "_report.png", dpi=130, bbox_inches="tight")
if bench_fig is not None:
    bench_fig.savefig(base + "_benchmark.png", dpi=130, bbox_inches="tight")
    plt.close(bench_fig)
```

> VERIFIED: the one-pager figure is `fig` at the save site (puzzle_report.py:584-585, followed by `plt.close(fig)`). Replace those two savefig lines with the block above; keep the existing `plt.close(fig)` after it. Put the `PdfPages` import with the other matplotlib imports at the top of the file (after line 36), not inline.

- [ ] **Step 3: Run the full suite**

Run: `python3 -m unittest discover -s tests`
Expected: PASS (all prior + 8 benchmark tests).

- [ ] **Step 4: MANDATORY visual verification.** Render a real report and inspect the new page:
  - Find a metrics JSON with the benchmarked fields (e.g. `RollingHills_analysis/RollingHills_metrics.json` has bimanual/pace/fatigue). Run `python3 puzzle_report.py <that>_metrics.json` to a temp/working copy so you don't clobber tracked outputs — copy the metrics+perframe to /tmp first and run there.
  - Confirm `<base>_report.pdf` has 2 pages and `<base>_benchmark.png` was written. **Read the `_benchmark.png`** (use the Read tool — you can see images) and confirm: title, caveat, all 6 rows present, robust rows bold/dark and weak/artifact rows greyed, verdicts shown for robust/weak directional rows, notes legible, nothing clipped off the page.
  - Delete all /tmp artifacts; confirm `git status` has no new tracked files.

- [ ] **Step 5: Commit** (stage only `puzzle_report.py`)

```bash
git add puzzle_report.py
git commit -m "Render elite benchmark as a second report page + benchmark PNG"
```

---

## Task 4: full-suite + end-to-end + bundle check

- [ ] **Step 1: Full suite green**

Run: `python3 -m unittest discover -s tests`
Expected: PASS.

- [ ] **Step 2: Bundle includes benchmarks.json and imports clean**

Run: `./build_release.sh 2>&1 | grep -E "benchmarks.json|files"` then extract the zip to /tmp and `python3 -c "import puzzle_report"` from the extracted dir; confirm clean and the json is present. Delete the /tmp extract.

- [ ] **Step 3: Commit any fixups** (targeted `git add`, no `-A`).

---

## Self-Review (done while writing)
- **Spec coverage (Feature 2):** editable benchmarks.json (Task 1), confidence tags + per-metric source/N (data + rows Task 2), report overlay with robust-prominent / weak-muted and no hard thresholds (Task 3 page + caveat), graceful when file missing or metric absent (`load_benchmarks` -> {}, `you=None` -> "n/a"). Bundled in build (Task 1/4).
- **Honesty:** measured values only; parallel_active flagged reversed; grip_hold flagged artifact; caveat line about small N; verdicts omitted where direction is null.
- **Type consistency:** `benchmark_rows` row keys `{key,label,you,elite,n,confidence,direction,verdict,note}` used identically in Task 2 tests and Task 3 rendering; `_get_by_path` / `load_benchmarks` signatures match tests.
- **Open detail (flagged):** confirm the one-pager figure variable name at the save site (assumed `fig`) and that `json`/`os` imports aren't duplicated — to be checked against the file at Task 2/3, not assumed.
