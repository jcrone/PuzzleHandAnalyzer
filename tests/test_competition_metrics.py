"""Tests for the competition-metric pure helpers in puzzle_hands."""
import unittest

from puzzle_hands import (
    count_before, pieces_per_min,
)


class TestCountBefore(unittest.TestCase):
    def test_counts_onsets_before_boundary(self):
        onset_times = [2.0, 5.0, 9.0, 30.0, 61.0, 90.0]
        self.assertEqual(count_before(onset_times, 60.0), 4)

    def test_none_boundary_counts_zero(self):
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


class TestCompetitionBlocks(unittest.TestCase):
    def _main_cluster(self):
        # clean flip phase 0-60s, then growth to 80 by 1200s
        return {
            "is_main": True,
            "history_seconds": [0.0, 20.0, 40.0, 60.0, 600.0, 1200.0],
            "history_counts":  [0,   1,    1,    5,    45,    80],
            "final_count": 80,
        }

    def test_full_blocks_with_cluster(self):
        from puzzle_hands import competition_blocks
        onset_times = [2.0, 5.0, 9.0, 30.0, 800.0]   # 4 before t=60
        out = competition_blocks(self._main_cluster(), {"detection_quality": "high"},
                                 onset_times, dur_s=1200.0, num_pieces=100,
                                 both_idle_pct=35.0)
        self.assertEqual(out["flip_phase"]["confidence"], "high")
        self.assertEqual(out["flip_phase"]["end_t"], 60.0)
        self.assertEqual(out["flip_phase"]["manipulations"], 4)
        self.assertAlmostEqual(out["placement"]["overall_pieces_per_min"], 4.0)
        self.assertEqual(out["placement"]["percent_complete"], 80.0)
        self.assertIn("25pct", out["placement"]["splits"])
        self.assertEqual(out["efficiency"]["dead_time_pct"], 35.0)
        self.assertEqual(out["efficiency"]["productive_pct"], 65.0)

    def test_no_cluster_degrades(self):
        from puzzle_hands import competition_blocks
        out = competition_blocks(None, None, [1.0, 2.0], dur_s=1200.0,
                                 num_pieces=100, both_idle_pct=40.0)
        self.assertEqual(out["flip_phase"]["confidence"], "unavailable")
        self.assertNotIn("placement", out)
        self.assertEqual(out["efficiency"]["productive_pct"], 60.0)
        self.assertEqual(out["efficiency"]["stall_count"], 0)

    def test_low_quality_downgrades_confidence(self):
        from puzzle_hands import competition_blocks
        out = competition_blocks(self._main_cluster(), {"detection_quality": "low"},
                                 [1.0], dur_s=1200.0, num_pieces=100,
                                 both_idle_pct=30.0)
        self.assertEqual(out["flip_phase"]["confidence"], "low")

    def test_stall_surfaces_into_efficiency(self):
        from puzzle_hands import competition_blocks
        cluster = {
            "is_main": True,
            # grows to 30 by t=60, then flat 60s->130s (70s stall), then resumes
            "history_seconds": [0.0, 60.0, 130.0, 160.0],
            "history_counts":  [0,   30,   30,    50],
            "final_count": 50,
        }
        out = competition_blocks(cluster, {"detection_quality": "high"},
                                 [1.0], dur_s=200.0, num_pieces=100,
                                 both_idle_pct=20.0)
        self.assertEqual(out["efficiency"]["stall_count"], 1)
        self.assertEqual(out["efficiency"]["longest_stall_s"], 70.0)
        self.assertEqual(len(out["efficiency"]["stalls"]), 1)

    def test_num_pieces_none_omits_percent_and_splits(self):
        from puzzle_hands import competition_blocks
        out = competition_blocks(self._main_cluster(), {"detection_quality": "high"},
                                 [1.0], dur_s=1200.0, num_pieces=None,
                                 both_idle_pct=25.0)
        self.assertNotIn("percent_complete", out["placement"])
        self.assertEqual(out["placement"]["splits"], {})


class TestPhaseGripPace(unittest.TestCase):
    def _grip(self, onset_frames, N):
        import numpy as np
        a = np.zeros(N, dtype=bool)
        for f in onset_frames:          # isolated True -> one rising edge each
            a[f] = True
        return a

    def test_split_prep_assembly(self):
        from puzzle_hands import phase_grip_pace
        # eff_fps=10, N=200 (20s); boundary at frame 100 (10s)
        # prep: 3 grips over 10s -> 18/min ; assembly: 2 grips over 10s -> 12/min
        L = self._grip([10, 30, 50, 120, 160], 200)
        R = self._grip([], 200)
        out = phase_grip_pace(L, R, boundary_frame=100, eff_fps=10.0, N=200)
        self.assertEqual(out["prep_grips_per_min"], 18.0)
        self.assertEqual(out["assembly_grips_per_min"], 12.0)

    def test_combines_both_hands(self):
        from puzzle_hands import phase_grip_pace
        L = self._grip([10, 30], 200)        # 2 in prep
        R = self._grip([40, 120], 200)       # 1 in prep, 1 in assembly
        out = phase_grip_pace(L, R, boundary_frame=100, eff_fps=10.0, N=200)
        self.assertEqual(out["prep_grips_per_min"], 18.0)   # 3 over 10s
        self.assertEqual(out["assembly_grips_per_min"], 6.0)  # 1 over 10s

    def test_no_boundary_falls_back_to_whole_session(self):
        from puzzle_hands import phase_grip_pace
        L = self._grip([10, 30, 50, 120, 160], 200)   # 5 grips over 20s
        R = self._grip([], 200)
        out = phase_grip_pace(L, R, boundary_frame=None, eff_fps=10.0, N=200)
        self.assertIsNone(out["prep_grips_per_min"])
        self.assertEqual(out["assembly_grips_per_min"], 15.0)
