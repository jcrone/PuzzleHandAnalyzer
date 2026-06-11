"""Tests for the competition-metrics report formatter."""
import unittest
from puzzle_report import _competition_lines


class TestCompetitionLines(unittest.TestCase):
    def test_full_summary(self):
        summary = {
            "placement": {"overall_pieces_per_min": 4.0,
                          "assembly_pieces_per_min": 5.0,
                          "final_assembled": 80, "percent_complete": 80.0,
                          "splits": {"25pct": 40.0, "50pct": 60.0,
                                     "75pct": None, "100pct": None}},
            "flip_phase": {"duration_s": 60.0, "confidence": "high",
                           "manipulations": 12},
            "efficiency": {"productive_pct": 65.0, "dead_time_pct": 35.0,
                           "stall_count": 2, "longest_stall_s": 90.0},
        }
        lines = _competition_lines(summary)
        joined = "\n".join(lines)
        self.assertIn("pieces/min overall", joined)
        self.assertIn("80% complete", joined)
        self.assertIn("manipulations (flipping/sorting)", joined)
        self.assertIn("Dead time 35%", joined)
        # splits: 75/100 unreached render as "-"
        self.assertIn("75% -", joined)

    def test_flip_unavailable(self):
        summary = {"flip_phase": {"confidence": "unavailable",
                                  "note": "no assembly cluster detected"},
                   "efficiency": {"productive_pct": 70.0, "dead_time_pct": 30.0,
                                  "stall_count": 0, "longest_stall_s": 0.0}}
        joined = "\n".join(_competition_lines(summary))
        self.assertIn("Flip/prep phase: not detected", joined)

    def test_old_summary_no_blocks(self):
        # an older metrics dict without the new keys -> no crash, no competition lines
        self.assertEqual(_competition_lines({"video": "x.mp4"}), [])


if __name__ == "__main__":
    unittest.main()
