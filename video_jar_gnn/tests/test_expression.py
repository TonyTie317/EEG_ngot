from __future__ import annotations

import unittest

import numpy as np

from video_jar_gnn.expression import (
    EXPRESSION_FEATURES,
    EXPRESSION_NODES,
    build_expression_adjacency,
    build_expression_graph,
    canonicalize_landmarks,
    expression_metadata,
    expression_proxy_values,
)


def _synthetic_face(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 0.08, size=(468, 3))
    points[:, 0] += np.linspace(-0.2, 0.2, 468)
    points[33] = (-0.32, 0.0, 0.01)
    points[133] = (-0.12, 0.0, 0.0)
    points[362] = (0.12, 0.0, 0.0)
    points[263] = (0.32, 0.0, 0.01)
    points[1] = (0.0, 0.18, -0.02)
    points[2] = (0.0, 0.20, -0.02)
    points[4] = (0.0, 0.22, -0.01)
    points[6] = (0.0, 0.10, -0.01)
    points[168] = (0.0, 0.04, 0.0)
    return points.astype(np.float64)


class ExpressionRepresentationTest(unittest.TestCase):
    def test_contract_and_adjacency(self):
        self.assertEqual(len(EXPRESSION_NODES), 20)
        self.assertEqual(len(EXPRESSION_FEATURES), 8)
        metadata = expression_metadata()
        self.assertEqual(metadata["representation"], "expression_v2")
        self.assertEqual(metadata["observed_mask_indices"], [6, 7])

        adjacency = build_expression_adjacency()
        self.assertEqual(adjacency.shape, (20, 20))
        np.testing.assert_array_equal(adjacency, adjacency.T)
        np.testing.assert_array_equal(np.diag(adjacency), np.zeros(20))
        reached = {0}
        while True:
            expanded = reached | {
                int(index)
                for node in reached
                for index in np.flatnonzero(adjacency[node])
            }
            if expanded == reached:
                break
            reached = expanded
        self.assertEqual(reached, set(range(20)))

    def test_similarity_and_rotation_invariance(self):
        face = _synthetic_face()
        angle = 0.71
        rotation = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        transformed = face @ rotation.T * 2.8 + np.asarray((4.0, -1.0, 0.7))
        first, _ = canonicalize_landmarks(face)
        second, _ = canonicalize_landmarks(transformed)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        np.testing.assert_allclose(first, second, atol=2e-5)
        np.testing.assert_allclose(
            expression_proxy_values(first),
            expression_proxy_values(second),
            atol=2e-5,
        )

    def test_proxy_responds_to_inner_lip_motion(self):
        canonical, _ = canonicalize_landmarks(_synthetic_face())
        before = expression_proxy_values(canonical)
        changed = canonical.copy()
        changed[14, 1] += 0.15
        after = expression_proxy_values(changed)
        self.assertAlmostEqual(after[16] - before[16], 0.15, places=5)

    def test_short_gap_is_imputed_and_long_gap_stays_missing(self):
        face = _synthetic_face()
        times = np.arange(12, dtype=float) * 0.1
        sequence = np.repeat(face[None, :, :], len(times), axis=0)
        sequence[2] = np.nan
        sequence[5:11] = np.nan
        graph, pose = build_expression_graph(
            sequence,
            times,
            trial_baseline_seconds=0.2,
            max_impute_gap_sec=0.5,
        )
        self.assertEqual(graph.shape, (12, 20, 8))
        self.assertEqual(pose.shape, (12, 4))
        self.assertTrue(np.isfinite(graph).all())
        self.assertTrue(np.all(graph[2, :, 6] == 0.0))
        self.assertTrue(np.all(graph[2, :, 7] == 1.0))
        self.assertTrue(np.all(graph[5:11, :, 6:8] == 0.0))
        self.assertTrue(np.all(graph[5:11, :, :6] == 0.0))
        self.assertTrue(np.all(graph[5:11, :, 2:5] == 0.0))

    def test_source_timestamps_are_resampled_before_derivatives(self):
        source_times = np.arange(5, dtype=float) * 0.2
        target_times = np.arange(9, dtype=float) * 0.1
        faces = []
        expected_values = []
        for timestamp in source_times:
            face = _synthetic_face()
            face[14, 1] += 0.02 * timestamp
            faces.append(face)
            canonical, _ = canonicalize_landmarks(face)
            expected_values.append(expression_proxy_values(canonical)[16])
        graph, _ = build_expression_graph(
            np.stack(faces),
            source_times,
            output_times=target_times,
            trial_baseline_seconds=0.2,
        )
        self.assertAlmostEqual(
            float(graph[1, 16, 0]),
            float((expected_values[0] + expected_values[1]) / 2.0),
            places=4,
        )
        expected_slope = (
            float(expected_values[-1] - expected_values[0])
            / float(source_times[-1] - source_times[0])
        )
        np.testing.assert_allclose(
            graph[2:-2, 16, 2], expected_slope, rtol=0.03, atol=2e-3
        )
        self.assertLess(float(np.max(np.abs(graph[:, 16, 3]))), 0.05)

    def test_all_missing_trial_remains_finite_and_explicitly_missing(self):
        times = np.linspace(0.0, 1.0, 61)
        sequence = np.full((len(times), 468, 3), np.nan)
        graph, pose = build_expression_graph(sequence, times)
        self.assertTrue(np.isfinite(graph).all())
        self.assertTrue(np.all(graph == 0.0))
        self.assertTrue(np.all(pose == 0.0))


if __name__ == "__main__":
    unittest.main()
