from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from video_jar_gnn.advanced_dataset import (
    EXPRESSION_FEATURE_NAMES,
    EXPRESSION_NODE_NAMES,
    EXPRESSION_OBSERVED_MASK_INDICES,
    EXPRESSION_STATIC_FEATURE_INDICES,
    EXPRESSION_VELOCITY_FEATURE_INDICES,
    AdvancedGraphStore,
    AdvancedStandardizer,
    ConditionGraphDataset,
    append_relational_features,
    build_condition_units,
    canonicalize_eye_rotation,
    preprocess_graph,
)
from video_jar_gnn.advanced_model import RepeatSetClassifier, build_encoder
from video_jar_gnn.dataset import load_graph_records
from video_jar_gnn.train_advanced import make_parser, pooled_inner_epoch, run


class AdvancedPipelineTest(unittest.TestCase):
    def test_delta_and_water_preprocessing_contract(self):
        graph = np.zeros((12, 15, 10), dtype=np.float32)
        graph[:, :, 0] = np.arange(12, dtype=np.float32)[:, None]
        graph[:, :, 3] = 1.0
        graph[:, :, 6:10] = -2.0
        raw_copy = graph.copy()
        delta = preprocess_graph(
            graph, mode="trial_delta", baseline_frames=4
        )
        np.testing.assert_array_equal(graph, raw_copy)
        np.testing.assert_array_equal(delta[:, :, 3], graph[:, :, 3])
        np.testing.assert_array_equal(delta[:, :, 6:10], graph[:, :, 6:10])
        self.assertAlmostEqual(float(np.median(delta[:4, :, 0])), 0.0)

        motion = preprocess_graph(
            graph, mode="trial_delta_motion", baseline_frames=4
        )
        self.assertEqual(motion.shape, (12, 15, 14))
        np.testing.assert_array_equal(motion[:, :, 10:14], 2.0)

        water = graph.copy()
        water[:, :, 0] -= 3.0
        combined = preprocess_graph(
            graph,
            mode="absolute_water_delta",
            water_reference=water,
        )
        self.assertEqual(combined.shape, (12, 15, 19))
        np.testing.assert_allclose(combined[:, :, 10], 3.0)
        np.testing.assert_array_equal(combined[:, :, 3], 1.0)
        empty_neutral = np.zeros((6, 15, 10), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "no detected face"):
            preprocess_graph(
                graph,
                mode="neutral_delta",
                neutral_baseline=empty_neutral,
            )

    def test_rotation_relations_and_segment_pooling(self):
        graph = np.zeros((8, 15, 10), dtype=np.float32)
        graph[:, :, 3] = 1.0
        graph[:, 4:6, 0] = -1.0
        graph[:, 4:6, 1] = -1.0
        graph[:, 6:8, 0] = 1.0
        graph[:, 6:8, 1] = 1.0
        graph[:, :, 6] = 2.0
        graph[:, :, 7] = 2.0
        rotated = canonicalize_eye_rotation(graph)
        left_y = rotated[:, 4:6, 1].mean(axis=1)
        right_y = rotated[:, 6:8, 1].mean(axis=1)
        np.testing.assert_allclose(left_y, right_y, atol=1e-6)
        np.testing.assert_allclose(rotated[:, :, 7], 0.0, atol=1e-6)
        related = append_relational_features(rotated, mode="raw")
        self.assertEqual(related.shape, (8, 15, 17))

        absolute_delta = np.concatenate(
            (rotated, rotated[:, :, [0, 1, 2, 4, 5, 6, 7, 8, 9]]),
            axis=2,
        )
        related_absolute = append_relational_features(
            absolute_delta, mode="absolute_water_delta"
        )
        self.assertEqual(related_absolute.shape, (8, 15, 33))

        encoder = build_encoder(
            "tcn",
            num_features=10,
            num_nodes=15,
            hidden_channels=4,
            dropout=0.0,
            temporal_pooling="segments",
        )
        encoded = encoder(
            torch.randn(2, 15, 24, 10),
            torch.eye(15).repeat(2, 1, 1),
        )
        self.assertEqual(tuple(encoded.shape), (2, 16))

    def test_repeat_set_models_ignore_padding_and_order(self):
        torch.manual_seed(1)
        graphs = torch.randn(2, 5, 15, 24, 10)
        adjacency = torch.eye(15).repeat(2, 1, 1)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, False]]
        )
        for name in ("stgcn", "tcn", "gru"):
            encoder = build_encoder(
                name,
                num_features=10,
                num_nodes=15,
                hidden_channels=4,
                dropout=0.0,
            )
            model = RepeatSetClassifier(
                encoder,
                num_classes=3,
                hidden_channels=4,
                dropout=0.0,
                aggregation="mean_std",
                objective="ordinal",
            )
            model.eval()
            output = model(graphs, adjacency, mask)
            probabilities = model.probabilities(output)
            self.assertEqual(tuple(output.shape), (2, 2))
            self.assertTrue(
                torch.allclose(
                    probabilities.sum(dim=1), torch.ones(2), atol=1e-6
                )
            )
            decisions = model.predictions(
                torch.tensor(
                    [[-1.0, -2.0], [1.0, -1.0], [2.0, 1.0]]
                )
            )
            torch.testing.assert_close(
                decisions, torch.tensor([0, 1, 2])
            )

            changed_padding = graphs.clone()
            changed_padding[~mask] = 1000.0
            output_padding = model(changed_padding, adjacency, mask)
            self.assertTrue(torch.allclose(output, output_padding, atol=1e-6))

            permutation = torch.tensor([2, 0, 1, 3, 4])
            output_permuted = model(
                graphs[:, permutation],
                adjacency,
                mask[:, permutation],
            )
            self.assertTrue(torch.allclose(output, output_permuted, atol=1e-6))
            output.sum().backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.parameters())
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
        adjacency = np.zeros((15, 15), dtype=np.float32)
        codes = (258, 189, 893)
        jars = (1, 3, 5)
        for subject_number in range(1, 10):
            subject = f"P{subject_number:03d}"
            for label, (code, jar) in enumerate(zip(codes, jars)):
                for repeat in (1, 2, 3):
                    rng = np.random.default_rng(
                        subject_number * 100 + label * 10 + repeat
                    )
                    graph = rng.normal(size=(16, 15, 10)).astype(np.float32)
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
                            "jar3_label": label,
                            "binary_label": int(label == 1),
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

    def _expression_fixture(self, root: Path) -> Path:
        manifest = self._fixture(root)
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        node_names = list(EXPRESSION_NODE_NAMES)
        feature_names = list(EXPRESSION_FEATURE_NAMES)
        metadata = {
            "representation": "expression_v2",
            "representation_version": 1,
            "node_names": node_names,
            "feature_names": feature_names,
            "observed_mask_indices": list(
                EXPRESSION_OBSERVED_MASK_INDICES
            ),
            "static_feature_indices": list(
                EXPRESSION_STATIC_FEATURE_INDICES
            ),
            "velocity_feature_indices": list(
                EXPRESSION_VELOCITY_FEATURE_INDICES
            ),
            "observed_feature_index": 6,
            "imputed_feature_index": 7,
        }
        adjacency = np.zeros((len(node_names), len(node_names)), dtype=np.float32)
        for index in range(len(node_names) - 1):
            adjacency[index, index + 1] = 1.0
            adjacency[index + 1, index] = 1.0
        for row_index, row in enumerate(rows):
            rng = np.random.default_rng(1000 + row_index)
            graph = rng.normal(
                size=(16, len(node_names), len(feature_names))
            ).astype(np.float32)
            observed = (np.arange(16) % 3 != 0).astype(np.float32)
            graph[:, :, 6] = observed[:, None]
            graph[:, :, 7] = (1.0 - observed[:, None]) * (
                np.arange(16)[:, None] % 2 == 0
            )
            np.savez_compressed(
                Path(row["graph_path"]),
                graph_seq=graph,
                adj=adjacency,
                meta=np.asarray(json.dumps(metadata)),
            )
        return manifest

    def test_condition_dataset_and_training_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            records, _ = load_graph_records(manifest)
            store = AdvancedGraphStore(
                records, preprocess="trial_delta_motion", baseline_frames=4
            )
            units, audit = build_condition_units(records, min_repeats=3)
            self.assertEqual(len(units), 27)
            self.assertEqual(audit["repeat_count_distribution"], {"3": 27})
            normalizer = AdvancedStandardizer.fit(
                store, units, list(range(18))
            )
            dataset = ConditionGraphDataset(
                store,
                units,
                [0],
                task="jar3",
                normalizer=normalizer,
                training=False,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["graphs"].shape), (5, 15, 16, 14))
            self.assertEqual(int(item["repeat_mask"].sum()), 3)
            self.assertNotIn("code_index", item)

            output = root / "run"
            args = make_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--task",
                    "jar3",
                    "--model",
                    "tcn",
                    "--objective",
                    "ce",
                    "--preprocess",
                    "trial_delta",
                    "--aggregation",
                    "mean",
                    "--output-dir",
                    str(output),
                    "--cv-folds",
                    "3",
                    "--inner-folds",
                    "2",
                    "--fold-index",
                    "0",
                    "--epochs",
                    "1",
                    "--min-epochs",
                    "1",
                    "--patience",
                    "1",
                    "--hidden-channels",
                    "4",
                    "--batch-size",
                    "8",
                    "--min-repeats",
                    "3",
                    "--repeat-dropout",
                    "0",
                    "--bootstrap-samples",
                    "0",
                    "--device",
                    "cpu",
                ]
            )
            summary = run(args)
            self.assertTrue(summary["partial_cv"])
            self.assertEqual(summary["condition_audit"]["conditions_included"], 27)
            self.assertEqual(summary["input_contract"], "video_graph_only")
            self.assertFalse(summary["uses_ma_mau_as_model_feature"])
            self.assertTrue((output / "fold_01" / "model.pt").is_file())
            self.assertTrue((output / "summary.json").is_file())
            checkpoint = torch.load(
                output / "fold_01" / "model.pt",
                map_location="cpu",
                weights_only=False,
            )
            for key in ("code_prior_log_probabilities", "code_to_index", "fusion"):
                self.assertNotIn(key, checkpoint)
            self.assertFalse(
                any(
                    token in name.lower()
                    for name in checkpoint["model_state_dict"]
                    for token in ("code_embedding", "prior", "fusion")
                )
            )

    def test_expression_schema_masks_and_training_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._expression_fixture(root)
            records, _ = load_graph_records(manifest)
            store = AdvancedGraphStore(
                records,
                preprocess="trial_delta_motion",
                baseline_frames=4,
                representation="expression_v2",
            )
            self.assertEqual(
                (store.sequence_length, store.num_nodes, store.num_features),
                (16, 20, 11),
            )
            self.assertEqual(store.observed_mask_indices, (6, 7))
            self.assertEqual(
                store.input_schema.static_feature_indices, (0, 1, 5)
            )
            self.assertEqual(
                store.input_schema.velocity_feature_indices, (2, 3, 4)
            )
            units, _ = build_condition_units(records, min_repeats=3)
            normalizer = AdvancedStandardizer.fit(
                store, units, list(range(len(units)))
            )
            for index in (6, 7):
                self.assertEqual(float(normalizer.mean[index]), 0.0)
                self.assertEqual(float(normalizer.scale[index]), 1.0)
            dataset = ConditionGraphDataset(
                store,
                units,
                [0],
                task="jar3",
                normalizer=normalizer,
                training=True,
                temporal_crop_min=0.6,
                noise_std=0.5,
            )
            item = dataset[0]
            for index in (6, 7):
                mask_values = set(
                    item["graphs"][0, :, :, index].unique().tolist()
                )
                self.assertTrue(mask_values.issubset({0.0, 1.0}))
            with self.assertRaisesRegex(
                ValueError, "relational features.*legacy"
            ):
                AdvancedGraphStore(
                    records,
                    preprocess="raw",
                    baseline_frames=4,
                    representation="expression_v2",
                    relational_features=True,
                )
            with self.assertRaisesRegex(
                ValueError, "expected 'legacy'"
            ):
                AdvancedGraphStore(
                    records,
                    preprocess="raw",
                    baseline_frames=4,
                    representation="legacy",
                )

            output = root / "expression_run"
            args = make_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--representation",
                    "expression_v2",
                    "--task",
                    "jar3",
                    "--model",
                    "tcn",
                    "--objective",
                    "ce",
                    "--preprocess",
                    "raw",
                    "--aggregation",
                    "mean",
                    "--output-dir",
                    str(output),
                    "--cv-folds",
                    "3",
                    "--inner-folds",
                    "2",
                    "--fold-index",
                    "0",
                    "--epochs",
                    "1",
                    "--min-epochs",
                    "1",
                    "--patience",
                    "1",
                    "--hidden-channels",
                    "4",
                    "--batch-size",
                    "8",
                    "--min-repeats",
                    "3",
                    "--repeat-dropout",
                    "0",
                    "--bootstrap-samples",
                    "0",
                    "--device",
                    "cpu",
                ]
            )
            summary = run(args)
            self.assertEqual(summary["representation"], "expression_v2")
            self.assertFalse(summary["canonical_rotation"])
            self.assertEqual(
                summary["representation_schema"]["observed_mask_indices"],
                [6, 7],
            )
            checkpoint = torch.load(
                output / "fold_01" / "model.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["representation"], "expression_v2")
            self.assertEqual(checkpoint["num_nodes"], 20)
            self.assertEqual(checkpoint["num_features"], 8)

            corrupted_path = records[0].graph_path
            with np.load(corrupted_path, allow_pickle=False) as data:
                corrupted_graph = np.asarray(data["graph_seq"])
                corrupted_adjacency = np.asarray(data["adj"])
                corrupted_meta = json.loads(str(data["meta"].item()))
            corrupted_meta["feature_names"][0:2] = [
                "delta",
                "value",
            ]
            np.savez_compressed(
                corrupted_path,
                graph_seq=corrupted_graph,
                adj=corrupted_adjacency,
                meta=np.asarray(json.dumps(corrupted_meta)),
            )
            with self.assertRaisesRegex(
                ValueError, "incompatible expression_v2 schema"
            ):
                AdvancedGraphStore(
                    records,
                    preprocess="raw",
                    baseline_frames=4,
                    representation="expression_v2",
                )

    def test_pooled_epoch_uses_mean_inner_curve(self):
        histories = [
            [
                {
                    "epoch": 1,
                    "validation_balanced_accuracy": 0.60,
                    "validation_macro_f1": 0.58,
                    "validation_loss": 0.8,
                },
                {
                    "epoch": 2,
                    "validation_balanced_accuracy": 0.50,
                    "validation_macro_f1": 0.49,
                    "validation_loss": 0.9,
                },
            ],
            [
                {
                    "epoch": 1,
                    "validation_balanced_accuracy": 0.40,
                    "validation_macro_f1": 0.39,
                    "validation_loss": 1.0,
                },
                {
                    "epoch": 2,
                    "validation_balanced_accuracy": 0.70,
                    "validation_macro_f1": 0.68,
                    "validation_loss": 0.7,
                },
            ],
        ]
        selected, curve = pooled_inner_epoch(histories)
        self.assertEqual(selected, 2)
        self.assertEqual(len(curve), 2)
        self.assertAlmostEqual(
            float(curve[1]["mean_validation_balanced_accuracy"]), 0.60
        )


if __name__ == "__main__":
    unittest.main()
