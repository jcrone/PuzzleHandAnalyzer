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
        self.assertEqual(dom["verdict"], "behind")
        pace = next(r for r in rows if r["key"] == "combined_grips_per_min")
        self.assertEqual(pace["verdict"], "ahead")

    def test_artifact_and_no_direction_have_no_verdict(self):
        rows = benchmark_rows(self._summary(), self._bench())
        hold = next(r for r in rows if r["key"] == "mean_grip_hold_s")
        self.assertIsNone(hold["verdict"])

    def test_missing_session_value_marks_na(self):
        rows = benchmark_rows({}, self._bench())
        dom = next(r for r in rows if r["key"] == "dominance_ratio")
        self.assertIsNone(dom["you"])
        self.assertIsNone(dom["verdict"])

    def test_empty_benchmarks(self):
        self.assertEqual(benchmark_rows(self._summary(), {}), [])

    def test_malformed_entry_skipped(self):
        bench = dict(self._bench())
        bench["junk"] = "not a dict"
        rows = benchmark_rows(self._summary(), bench)
        self.assertNotIn("junk", [r["key"] for r in rows])
        self.assertEqual(len(rows), 3)

    def test_missing_value_no_crash(self):
        bench = {"x": {"path": "bimanual.dominance_ratio", "label": "X",
                       "n": 1, "confidence": "robust",
                       "direction": "lower_is_better", "note": ""}}  # no 'value'
        rows = benchmark_rows(self._summary(), bench)
        self.assertEqual(rows[0]["elite"], None)
        self.assertIsNone(rows[0]["verdict"])


class TestBenchmarkPage(unittest.TestCase):
    def test_no_benchmarks_returns_none(self):
        from puzzle_report import _make_benchmark_page
        # empty benchmarks file path -> no rows -> None (no figure)
        import puzzle_report
        orig = puzzle_report.load_benchmarks
        puzzle_report.load_benchmarks = lambda *a, **k: {}
        try:
            self.assertIsNone(_make_benchmark_page({"bimanual": {}}))
        finally:
            puzzle_report.load_benchmarks = orig

    def test_renders_with_missing_value_entry(self):
        # a benchmark entry without 'value' must not crash the page builder
        from puzzle_report import _make_benchmark_page
        import puzzle_report, matplotlib
        orig = puzzle_report.load_benchmarks
        puzzle_report.load_benchmarks = lambda *a, **k: {
            "x": {"path": "bimanual.dominance_ratio", "label": "X", "n": 1,
                  "confidence": "robust", "direction": "lower_is_better",
                  "note": "no value key"}}
        try:
            fig = _make_benchmark_page({"bimanual": {"dominance_ratio": 2.0}})
            self.assertIsNotNone(fig)
            matplotlib.pyplot.close(fig)
        finally:
            puzzle_report.load_benchmarks = orig
