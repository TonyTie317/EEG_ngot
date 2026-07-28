from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from video_jar_gnn.extract import (
    _load_cached_info,
    add_temporal_features,
    build_adjacency,
    cached_configuration_mismatches,
    load_active_and_preceding_frame_times,
    preserve_unselected_manifest_rows,
    sampling_fingerprint,
    select_frames_by_time,
    select_preceding_frames_by_time,
)
from video_jar_gnn.expression import (
    build_expression_adjacency,
    expression_metadata,
)


class ExtractUtilityTest(unittest.TestCase):
    def test_partial_extraction_preserves_prior_unselected_status(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "graph_manifest.csv"
            header = (
                "sample_id,subject_id,ma_mau,repeat,jar,start_frame,end_frame,"
                "video_path,frame_label_path,graph_path,extract_status,"
                "detection_ratio\n"
            )
            output.write_text(
                header
                + "A,P001,189,1,3,1,2,v.mp4,f.csv,/old/A.npz,cached,0.9\n"
                + "B,P002,258,2,2,3,4,v.mp4,f.csv,/old/B.npz,cached,0.8\n",
                encoding="utf-8",
            )
            rows = [
                {
                    "sample_id": "A",
                    "subject_id": "P001",
                    "ma_mau": "189",
                    "repeat": "1",
                },
                {
                    "sample_id": "B",
                    "subject_id": "P002",
                    "ma_mau": "258.0",
                    "repeat": "2.0",
                },
            ]
            copied = preserve_unselected_manifest_rows(
                rows, selected_ids={"A"}, output_manifest=output
            )
            self.assertEqual(copied, 1)
            self.assertNotIn("extract_status", rows[0])
            self.assertEqual(rows[1]["extract_status"], "cached")
            self.assertEqual(rows[1]["graph_path"], "/old/B.npz")

    def test_cache_configuration_mismatch_is_explicit(self):
        metadata = {
            "duration_mode": "lsl",
            "window_seconds": 10.0,
            "extractor_version": 1,
        }
        cached = {
            "num_frames": 96,
            "metadata": dict(metadata),
        }
        self.assertEqual(
            cached_configuration_mismatches(
                cached,
                expected_metadata=metadata,
                expected_num_frames=96,
            ),
            {},
        )
        mismatch = cached_configuration_mismatches(
            cached,
            expected_metadata={
                **metadata,
                "duration_mode": "labelled",
                "extractor_version": 4,
            },
            expected_num_frames=128,
        )
        self.assertIn("num_frames", mismatch)
        self.assertIn("duration_mode", mismatch)
        self.assertIn("extractor_version", mismatch)

    def test_cache_integrity_is_checked_before_reuse(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.npz"
            graph = np.zeros((8, 15, 10), dtype=np.float32)
            adjacency = build_adjacency()
            sampling = np.arange(8)
            fingerprint_metadata = {
                "sampling_fingerprint": sampling_fingerprint(
                    sampling,
                    sampling.astype(float),
                    sampling.astype(float),
                )
            }
            np.savez_compressed(
                path,
                graph_seq=graph,
                adj=adjacency,
                sampled_frame_idx=sampling,
                sampled_lsl=sampling.astype(float),
                target_lsl=sampling.astype(float),
                detection_ratio=np.asarray(1.0, dtype=np.float32),
                meta=np.asarray(json.dumps(fingerprint_metadata)),
            )
            self.assertEqual(_load_cached_info(path)["num_frames"], 8)

            corrupted_sampling = sampling.astype(float)
            corrupted_sampling[-1] += 0.25
            np.savez_compressed(
                path,
                graph_seq=graph,
                adj=adjacency,
                sampled_frame_idx=sampling,
                sampled_lsl=corrupted_sampling,
                target_lsl=sampling.astype(float),
                meta=np.asarray(json.dumps(fingerprint_metadata)),
            )
            with self.assertRaisesRegex(RuntimeError, "do not match their fingerprint"):
                _load_cached_info(path)

    def test_expression_v2_cache_round_trip_is_schema_aware(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "expression.npz"
            graph = np.zeros((8, 20, 8), dtype=np.float32)
            graph[:, :, 6] = 1.0
            sampling = np.arange(8)
            metadata = {
                **expression_metadata(),
                "sampling_fingerprint": sampling_fingerprint(
                    sampling,
                    sampling.astype(float),
                    sampling.astype(float),
                ),
            }
            np.savez_compressed(
                path,
                graph_seq=graph,
                adj=build_expression_adjacency(),
                sampled_frame_idx=sampling,
                sampled_lsl=sampling.astype(float),
                target_lsl=sampling.astype(float),
                meta=np.asarray(json.dumps(metadata)),
            )
            cached = _load_cached_info(path)
            self.assertEqual(cached["num_frames"], 8)
            self.assertEqual(
                cached["metadata"]["representation"], "expression_v2"
            )
            self.assertEqual(cached["detection_ratio"], 1.0)

    def test_lsl_mode_truncates_slow_capture_to_ten_seconds(self):
        frames = [(index, index * 12.0 / 599.0) for index in range(600)]
        selected, actual, target = select_frames_by_time(
            frames,
            num_frames=21,
            duration_mode="lsl",
            window_seconds=10.0,
        )
        self.assertEqual(len(selected), 21)
        self.assertLess(selected[-1], 550)
        self.assertAlmostEqual(target[-1] - target[0], 10.0)
        self.assertLessEqual(abs(actual[-1] - 10.0), 0.02)

        labelled, _, labelled_target = select_frames_by_time(
            frames,
            num_frames=21,
            duration_mode="labelled",
            window_seconds=10.0,
        )
        self.assertEqual(labelled[-1], 599)
        self.assertAlmostEqual(labelled_target[-1], 12.0)

    def test_preceding_context_is_bound_to_the_following_trial(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frames.csv"
            path.write_text(
                "frame_idx,t_lsl,ma_mau,lan_lap\n"
                "0,0.00,0,0\n"
                "1,0.02,0,0\n"
                "2,0.04,258,1\n"
                "3,0.06,258,1\n"
                "4,0.08,0,0\n"
                "5,0.10,0,0\n"
                "6,0.12,0,0\n"
                "7,0.14,694,1\n"
                "8,0.16,694,1\n",
                encoding="utf-8",
            )
            active, preceding = load_active_and_preceding_frame_times(path)
            self.assertEqual([item[0] for item in active[(258, 1)]], [2, 3])
            self.assertEqual(
                [item[0] for item in preceding[(258, 1)]], [0, 1]
            )
            self.assertEqual(
                [item[0] for item in preceding[(694, 1)]], [4, 5, 6]
            )

    def test_preceding_context_resamples_the_interval_tail(self):
        frames = [
            (index, index / 60.0)
            for index in range(601)
        ]
        selected, actual, target = select_preceding_frames_by_time(
            frames,
            num_frames=25,
            window_seconds=2.0,
        )
        self.assertEqual(len(selected), 25)
        self.assertGreaterEqual(selected[0], 480)
        self.assertEqual(selected[-1], 600)
        self.assertAlmostEqual(target[-1] - target[0], 2.0)
        self.assertLessEqual(np.max(np.abs(actual - target)), 1.0 / 60.0)

    def test_cache_validates_optional_neutral_baseline(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph_with_baseline.npz"
            graph = np.zeros((8, 15, 10), dtype=np.float32)
            baseline = np.zeros((4, 15, 10), dtype=np.float32)
            adjacency = build_adjacency()
            sampling = np.arange(8)
            baseline_sampling = np.arange(4)
            metadata = {
                "pre_context_seconds": 2.0,
                "baseline_sampling_fingerprint": sampling_fingerprint(
                    baseline_sampling,
                    baseline_sampling.astype(float),
                    baseline_sampling.astype(float),
                ),
            }
            np.savez_compressed(
                path,
                graph_seq=graph,
                adj=adjacency,
                sampled_frame_idx=sampling,
                sampled_lsl=sampling.astype(float),
                target_lsl=sampling.astype(float),
                baseline_seq=baseline,
                baseline_sampled_frame_idx=baseline_sampling,
                baseline_sampled_lsl=baseline_sampling.astype(float),
                baseline_target_lsl=baseline_sampling.astype(float),
                meta=np.asarray(json.dumps(metadata)),
            )
            cached = _load_cached_info(path)
            self.assertEqual(cached["baseline"]["num_frames"], 4)
            mismatch = cached_configuration_mismatches(
                cached,
                expected_metadata=metadata,
                expected_num_frames=8,
                expected_baseline_frames=5,
            )
            self.assertEqual(mismatch["baseline_frames"], (4, 5))

    def test_graph_contract_has_finite_temporal_features(self):
        adjacency = build_adjacency()
        self.assertEqual(adjacency.shape, (15, 15))
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(np.diag(adjacency), np.zeros(15))

        base = np.zeros((8, 15, 6), dtype=np.float32)
        base[:, :, 0] = np.arange(8, dtype=np.float32)[:, None] / 10.0
        base[:, :, 3] = 1.0
        graph = add_temporal_features(base, np.linspace(0.0, 0.7, 8))
        self.assertEqual(graph.shape, (8, 15, 10))
        self.assertTrue(np.isfinite(graph).all())
        self.assertTrue(np.allclose(graph[1:, :, 6], 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
