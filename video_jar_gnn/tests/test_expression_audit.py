from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from video_jar_gnn.expression_audit import (
    CacheIdentity,
    ExpressionSchema,
    ResponseWindow,
    audit_window,
    feature_reliability,
    discover_cache_records,
    load_expression_cache,
    make_parser,
    run,
    summarize_expression_window,
)


class ExpressionAuditTest(unittest.TestCase):
    schema = ExpressionSchema(
        node_names=("mouth", "eyes"),
        feature_names=("signal", "observed"),
        observed_mask_indices=(1,),
    )

    def _write_cache(
        self,
        path: Path,
        *,
        subject: str,
        condition: str,
        repeat: int,
        subject_offset: float = 0.0,
        condition_offset: float = 0.0,
        repeat_noise: float = 0.0,
        representation: str = "expression_v2",
    ) -> None:
        times = np.linspace(100.0, 110.0, 21)
        relative = times - times[0]
        graph = np.zeros((len(times), 2, 2), dtype=np.float32)
        graph[:, :, 0] = (
            subject_offset
            + condition_offset
            + 3.0 * relative[:, None]
            + repeat_noise
        )
        graph[:, :, 1] = 1.0
        meta = {
            "representation": representation,
            "subject_id": subject,
            "condition_id": condition,
            "repeat": repeat,
            "node_names": list(self.schema.node_names),
            "feature_names": list(self.schema.feature_names),
            "observed_mask_indices": list(
                self.schema.observed_mask_indices
            ),
        }
        np.savez_compressed(
            path,
            graph_seq=graph,
            adj=np.eye(2, dtype=np.float32),
            target_lsl=times,
            meta=np.asarray(json.dumps(meta)),
        )

    def test_real_time_window_has_per_second_slope(self):
        times = np.linspace(0.0, 10.0, 21)
        graph = np.zeros((21, 2, 2), dtype=np.float32)
        graph[:, :, 0] = 2.0 + 3.0 * times[:, None]
        graph[:, :, 1] = 1.0
        summary = summarize_expression_window(
            graph,
            times,
            self.schema,
            ResponseWindow(2.0, 6.0),
        )
        self.assertEqual(summary.shape, (12,))
        # Each node has one signal feature × six statistics.
        self.assertAlmostEqual(float(summary[5]), 3.0, places=6)
        self.assertAlmostEqual(float(summary[11]), 3.0, places=6)

    def test_cache_schema_and_representation_are_strictly_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.npz"
            self._write_cache(
                valid, subject="P001", condition="A", repeat=1
            )
            cache = load_expression_cache(valid)
            self.assertEqual(cache.identity, CacheIdentity("P001", "A", 1))
            self.assertEqual(cache.schema, self.schema)
            np.testing.assert_allclose(
                cache.time_seconds[[0, -1]], [0.0, 10.0]
            )

            wrong = root / "wrong.npz"
            self._write_cache(
                wrong,
                subject="P001",
                condition="A",
                repeat=1,
                representation="legacy_face_graph",
            )
            with self.assertRaisesRegex(ValueError, "expression_v2"):
                load_expression_cache(wrong)

    def test_partial_extraction_manifest_skips_unselected_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "one.npz"
            self._write_cache(
                cache, subject="P001", condition="189", repeat=1
            )
            manifest = root / "partial.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "graph_path",
                        "subject_id",
                        "ma_mau",
                        "repeat",
                        "extract_status",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "graph_path": str(cache),
                        "subject_id": "P001",
                        "ma_mau": 189,
                        "repeat": 1,
                        "extract_status": "ok",
                    }
                )
                writer.writerow(
                    {
                        "subject_id": "P002",
                        "ma_mau": 258,
                        "repeat": 1,
                        "extract_status": "not_selected",
                    }
                )
            records = discover_cache_records(
                cache_dir=None, manifest=manifest
            )
            self.assertEqual(len(records), 1)

    def test_singular_observation_metadata_and_imputed_channel(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "expression.npz"
            times = np.linspace(0.0, 10.0, 11)
            graph = np.zeros((11, 1, 3), dtype=np.float32)
            graph[:, :, 0] = times[:, None]
            graph[:, :, 1] = 1.0
            meta = {
                "representation": "expression_v2",
                "subject_id": "P001",
                "ma_mau": 189,
                "repeat": 1,
                "node_names": ["mouth"],
                "feature_names": ["value", "observed", "imputed"],
                "observed_feature_index": 1,
                "imputed_feature_index": 2,
            }
            np.savez_compressed(
                path,
                graph_seq=graph,
                adj=np.eye(1, dtype=np.float32),
                target_lsl=times,
                meta=np.asarray(json.dumps(meta)),
            )
            cache = load_expression_cache(path)
            self.assertEqual(cache.schema.observed_mask_indices, (1,))
            self.assertEqual(cache.schema.excluded_feature_indices, (2,))
            self.assertEqual(cache.schema.signal_feature_indices, (0,))

    def test_identical_condition_repeats_are_separable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            caches = []
            for subject_number in range(1, 4):
                for condition_number in range(3):
                    for repeat in range(1, 4):
                        path = (
                            root
                            / f"P{subject_number}_{condition_number}_{repeat}.npz"
                        )
                        self._write_cache(
                            path,
                            subject=f"P{subject_number}",
                            condition=str(condition_number),
                            repeat=repeat,
                            subject_offset=subject_number * 50.0,
                            condition_offset=condition_number * 4.0,
                        )
                        caches.append(load_expression_cache(path))
            metrics, _, subjects = audit_window(
                caches,
                ResponseWindow(0.0, 10.0),
                min_repeats=3,
            )
            self.assertAlmostEqual(
                metrics["within_condition_distance_median"], 0.0, places=7
            )
            self.assertEqual(metrics["pair_auc"], 1.0)
            self.assertGreater(
                metrics["subject_centered_icc1_median"], 0.99
            )
            self.assertEqual(len(subjects), 3)

    def test_subject_centered_icc_removes_identity_only_repeatability(self):
        values = []
        subjects = []
        groups = []
        repeat_pattern = (-1.0, 0.0, 1.0)
        for subject_number in range(1, 5):
            for condition in ("A", "B"):
                for noise in repeat_pattern:
                    values.append([subject_number * 100.0 + noise])
                    subjects.append(f"P{subject_number}")
                    groups.append(f"P{subject_number}:{condition}")
        rows = feature_reliability(
            np.asarray(values, dtype=np.float64),
            np.asarray(subjects, dtype=object),
            np.asarray(groups, dtype=object),
        )
        self.assertGreater(rows[0]["raw_icc1"], 0.99)
        self.assertLess(rows[0]["subject_centered_icc1"], 0.1)

    def test_cli_writes_label_free_window_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir = root / "caches"
            cache_dir.mkdir()
            rows = []
            for subject_number in range(1, 4):
                for condition_number in range(2):
                    for repeat in range(1, 4):
                        path = (
                            cache_dir
                            / f"P{subject_number}_{condition_number}_{repeat}.npz"
                        )
                        self._write_cache(
                            path,
                            subject=f"P{subject_number}",
                            condition=str(condition_number),
                            repeat=repeat,
                            subject_offset=subject_number,
                            condition_offset=condition_number * 2.0,
                            repeat_noise=repeat * 0.01,
                        )
                        rows.append(
                            {
                                "graph_path": str(path),
                                "subject_id": f"P{subject_number}",
                                "condition_id": str(condition_number),
                                "repeat": repeat,
                            }
                        )
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "graph_path",
                        "subject_id",
                        "condition_id",
                        "repeat",
                    ),
                )
                writer.writeheader()
                writer.writerows(rows)
            output = root / "audit"
            args = make_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(output),
                    "--windows",
                    "0:2",
                    "2:10",
                    "--min-repeats",
                    "3",
                ]
            )
            summary = run(args)
            self.assertFalse(summary["uses_jar_labels"])
            self.assertFalse(summary["uses_condition_id_as_predictor"])
            self.assertFalse(summary["water_included"])
            self.assertIn("605", summary["excluded_conditions"])
            self.assertEqual(len(summary["windows"]), 2)
            for name in (
                "window_metrics.csv",
                "feature_reliability.csv",
                "subject_distances.csv",
                "summary.json",
            ):
                self.assertTrue((output / name).is_file())
            written = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                written["audit_type"], "label_free_repeat_reliability"
            )


if __name__ == "__main__":
    unittest.main()
