"""Tests for the competition-metric pure helpers in puzzle_hands."""
import unittest

from puzzle_hands import (
    count_before, pieces_per_min,
)


class TestCountBefore(unittest.TestCase):
    def test_counts_onsets_before_boundary(self):
        onset_times = [2.0, 5.0, 9.0, 30.0, 61.0, 90.0]
        self.assertEqual(count_before(onset_times, 60.0), 4)

    def test_none_boundary_counts_all(self):
        onset_times = [2.0, 5.0, 9.0]
        self.assertEqual(count_before(onset_times, None), 0)


class TestPiecesPerMin(unittest.TestCase):
    def test_overall_and_assembly(self):
        res = pieces_per_min(final_count=80, total_s=1200.0,
                             onset_t=300.0, count_at_onset=5)
        self.assertAlmostEqual(res["overall_pieces_per_min"], 4.0)
        self.assertAlmostEqual(res["assembly_pieces_per_min"], 5.0)

    def test_no_onset_falls_back_to_overall(self):
        res = pieces_per_min(final_count=80, total_s=1200.0,
                             onset_t=None, count_at_onset=0)
        self.assertAlmostEqual(res["overall_pieces_per_min"], 4.0)
        self.assertAlmostEqual(res["assembly_pieces_per_min"], 4.0)
