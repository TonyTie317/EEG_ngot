from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np

from video_jar_gnn.train_classical import (
    ALL_EXPERIMENTS,
    describe_summary_feature,
    fit_selected_video_pipeline,
    fuse_probabilities,
    make_parser,
    make_video_feature_parser,
    run,
    run_video_features,
    summarize_condition_graph,
)


class ClassicalConditionTrainerTest(unittest.TestCase):
    def test_summary_and_fusion_contract(self):
        graph = np.zeros((10, 15, 10), dtype=np.float32)
        graph[:, :, 0] = np.arange(10, dtype=np.float32)[:, None]
        graph[:, :, 3] = 1.0
        graph[:, :, 6] = -2.0
        summary = summarize_condition_graph(graph)
        # 15 nodes × (9 continuous × 6 statistics
        #             + 4 motion × 4 statistics).
        self.assertEqual(summary.shape, (1050,))
        self.assertTrue(np.isfinite(summary).all())

        code = np.asarray([[0.8, 0.2], [0.3, 0.7]])
        face = np.asarray([[0.4, 0.6], [0.9, 0.1]])
        np.testing.assert_allclose(
            fuse_probabilities(code, face, alpha_face=0.0), code
        )
        np.testing.assert_allclose(
            fuse_probabilities(code, face, alpha_face=1.0), face
        )

    def test_video_pipeline_fits_all_transforms_on_training_rows_only(self):
        rng = np.random.default_rng(9)
        features = rng.normal(size=(12, 8))
        labels = np.asarray([0, 1, 2] * 4, dtype=np.int64)
        train = np.arange(9, dtype=np.int64)
        # This column is constant in train but highly variable outside train.
        # A globally fitted VarianceThreshold would incorrectly retain it.
        features[:, 0] = 0.0
        features[9:, 0] = [10.0, 20.0, 30.0]
        pipeline = fit_selected_video_pipeline(
            features,
            labels,
            train,
            k_features=3,
            c_value=0.1,
            max_iter=500,
            seed=2,
            num_classes=3,
        )
        self.assertEqual(
            list(pipeline.named_steps),
            ["variance", "standardizer", "selector", "classifier"],
        )
        self.assertFalse(
            bool(pipeline.named_steps["variance"].get_support()[0])
        )
        self.assertEqual(
            pipeline.named_steps["classifier"].class_weight, "balanced"
        )
        all_features = fit_selected_video_pipeline(
            features,
            labels,
            train,
            k_features=0,
            c_value=0.1,
            max_iter=500,
            seed=2,
            num_classes=3,
        )
        self.assertEqual(all_features.named_steps["selector"].k, "all")

    def test_summary_feature_indices_are_human_readable(self):
        self.assertEqual(
            describe_summary_feature(0),
            {
                "node": "brow_left_inner",
                "source_feature": "cx",
                "statistic": "mean",
                "feature_family": "temporal",
            },
        )
        self.assertEqual(
            describe_summary_feature(1049),
            {
                "node": "chin_center",
                "source_feature": "velocity_aspect",
                "statistic": "absolute_max",
                "feature_family": "absolute_motion",
            },
        )

    def _fixture(self, root: Path) -> Path:
        graph_dir = root / "graphs"
        graph_dir.mkdir()
        manifest = root / "manifest.csv"
        fields = [
            "sample_id",
            "subject_id",
            "ma_mau",
            "repeat",
            "jar",
            "jar3_label",
            "binary_label",
            "graph_path",
            "detection_ratio",
            "extract_status",
        ]
        rows = []
        adjacency = np.eye(15, dtype=np.float32)
        conditions = (
            (189, 1, 0),
            (258, 3, 1),
            (453, 5, 2),
            (605, 3, 1),
        )
        for subject_number in range(1, 10):
            subject = f"P{subject_number:03d}"
            for code, jar, jar3 in conditions:
                for repeat in (1, 2, 3):
                    rng = np.random.default_rng(
                        subject_number * 10000 + code * 10 + repeat
                    )
                    graph = rng.normal(
                        loc=jar3 * 0.15,
                        scale=0.5,
                        size=(12, 15, 10),
                    ).astype(np.float32)
                    graph[:, :, 3] = 1.0
                    path = graph_dir / f"{subject}_{code}_R{repeat}.npz"
                    np.savez_compressed(
                        path, graph_seq=graph, adj=adjacency
                    )
                    rows.append(
                        {
                            "sample_id": path.stem,
                            "subject_id": subject,
                            "ma_mau": code,
                            "repeat": repeat,
                            "jar": jar,
                            "jar3_label": jar3,
                            "binary_label": int(jar == 3),
                            "graph_path": path,
                            "detection_ratio": 1.0,
                            "extract_status": "ok",
                        }
                    )
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_partial_nested_cv_all_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            output = root / "classical"
            args = make_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--task",
                    "jar3",
                    "--output-dir",
                    str(output),
                    "--cv-folds",
                    "3",
                    "--inner-folds",
                    "2",
                    "--fold-index",
                    "0",
                    "--min-repeats",
                    "3",
                    "--baseline-frames",
                    "3",
                    "--c-grid",
                    "0.1",
                    "--alpha-grid",
                    "0",
                    "0.5",
                    "1",
                    "--max-iter",
                    "500",
                    "--bootstrap-samples",
                    "0",
                ]
            )
            summary = run(args)
            self.assertTrue(summary["partial_cv"])
            self.assertTrue(summary["water_used_as_unlabelled_reference"])
            self.assertEqual(
                summary["condition_audit"]["conditions_included"], 27
            )
            self.assertEqual(
                set(summary["experiments"]), set(ALL_EXPERIMENTS)
            )
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "ablation_metrics.csv").is_file())
            for experiment in ALL_EXPERIMENTS:
                self.assertTrue(
                    (
                        output
                        / "fold_01"
                        / f"{experiment}.joblib"
                    ).is_file()
                )

    def test_partial_video_only_selected_feature_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            output = root / "video_only"
            args = make_video_feature_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--task",
                    "jar3",
                    "--output-dir",
                    str(output),
                    "--modes",
                    "raw",
                    "trial_delta",
                    "water_delta",
                    "--cv-folds",
                    "3",
                    "--inner-folds",
                    "2",
                    "--fold-index",
                    "0",
                    "--min-repeats",
                    "3",
                    "--baseline-frames",
                    "3",
                    "--k-grid",
                    "4",
                    "--c-grid",
                    "0.1",
                    "--max-iter",
                    "500",
                    "--bootstrap-samples",
                    "0",
                ]
            )
            with mock.patch(
                "video_jar_gnn.train_classical.code_feature_matrix",
                side_effect=AssertionError(
                    "video-only path attempted to construct code features"
                ),
            ):
                summary = run_video_features(args)
            self.assertEqual(summary["input_contract"], "video_summary_only")
            self.assertFalse(summary["uses_ma_mau_as_predictor"])
            self.assertTrue(summary["water_used_as_unlabelled_reference"])
            self.assertEqual(
                summary["feature_selection_stability"]["video_raw"]["n_folds"],
                1,
            )
            self.assertEqual(
                summary["condition_audit"]["conditions_included"], 27
            )
            config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertFalse(config["uses_ma_mau_as_predictor"])
            self.assertNotIn("sweet_code_order", config)
            for mode in ("raw", "trial_delta", "water_delta"):
                artifact = joblib.load(
                    output / "fold_01" / f"video_{mode}.joblib"
                )
                self.assertFalse(artifact["uses_ma_mau_as_predictor"])
                self.assertNotIn("code_model", artifact)
                self.assertEqual(
                    list(artifact["pipeline"].named_steps),
                    [
                        "variance",
                        "standardizer",
                        "selector",
                        "classifier",
                    ],
                )
            with (
                output / "predictions_condition.csv"
            ).open(newline="", encoding="utf-8") as handle:
                fields = csv.DictReader(handle).fieldnames or []
            self.assertIn("ma_mau", fields)
            self.assertNotIn("selected_code_C", fields)
            self.assertTrue((output / "feature_metrics.csv").is_file())
            self.assertTrue(
                (output / "selected_feature_indices.csv").is_file()
            )
            with (
                output / "selected_feature_indices.csv"
            ).open(newline="", encoding="utf-8") as handle:
                selected_fields = csv.DictReader(handle).fieldnames or []
            for field in (
                "rank_by_training_f_score",
                "node",
                "source_feature",
                "statistic",
                "training_f_score",
            ):
                self.assertIn(field, selected_fields)


if __name__ == "__main__":
    unittest.main()
