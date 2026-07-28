from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from video_jar_gnn.advanced_dataset import ConditionUnit
from video_jar_gnn.expression import (
    EXPRESSION_FEATURES,
    EXPRESSION_NODES,
    build_expression_adjacency,
    expression_metadata,
)
from video_jar_gnn.expression_audit import ResponseWindow
from video_jar_gnn.train_advanced import make_unit_splits
from video_jar_gnn.train_expression import (
    assert_subject_disjoint_splits,
    make_parser,
    run,
    select_expression_candidate,
)


class ExpressionWindowTrainerTest(unittest.TestCase):
    def test_outer_test_labels_cannot_change_selected_window(self):
        windows = (ResponseWindow(0.0, 2.0), ResponseWindow(2.0, 4.0))
        outer_train = np.asarray([0, 1, 2, 3], dtype=np.int64)
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        first = np.asarray(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )
        second = 1.0 - first
        oof = {
            ("0:2", 2, 0.1): first,
            ("2:4", 2, 0.1): second,
        }
        selected, history = select_expression_candidate(
            oof,
            labels,
            outer_train,
            windows,
            num_classes=2,
        )
        changed = labels.copy()
        changed[4:] = 1 - changed[4:]
        selected_changed, history_changed = select_expression_candidate(
            oof,
            changed,
            outer_train,
            windows,
            num_classes=2,
        )
        self.assertEqual(selected, selected_changed)
        self.assertEqual(history, history_changed)
        self.assertEqual(selected["window"], "0:2")

    def test_subject_splits_are_disjoint(self):
        units = []
        for subject_number in range(1, 10):
            for label in (0, 1):
                units.append(
                    ConditionUnit(
                        subject_id=f"P{subject_number:03d}",
                        ma_mau=100 + label,
                        jar=3 if label else 1,
                        jar3_label=1 if label else 0,
                        binary_label=label,
                        record_indices=(0, 1),
                        repeats=(1, 2),
                    )
                )
        splits = make_unit_splits(
            units, "binary", n_splits=3, seed=7
        )
        assert_subject_disjoint_splits(units, splits)
        subjects = np.asarray([unit.subject_id for unit in units])
        for train, test in splits:
            self.assertTrue(
                set(subjects[train]).isdisjoint(subjects[test])
            )

    def _fixture(self, root: Path) -> Path:
        graph_dir = root / "graphs_expression_v2"
        graph_dir.mkdir()
        manifest = root / "manifest_expression_v2.csv"
        fields = (
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
        )
        rows = []
        adjacency = build_expression_adjacency()
        conditions = (
            (189, 1, 0),
            (258, 3, 1),
            (453, 5, 2),
            # Must be filtered before supervised condition construction.
            (605, 3, 1),
        )
        for subject_number in range(1, 10):
            subject = f"P{subject_number:03d}"
            for code, jar, jar3 in conditions:
                for repeat in (1, 2):
                    rng = np.random.default_rng(
                        subject_number * 100000 + code * 10 + repeat
                    )
                    target_lsl = np.linspace(500.0, 510.0, 41)
                    relative = target_lsl - target_lsl[0]
                    graph = rng.normal(
                        0.0,
                        0.02,
                        size=(
                            len(relative),
                            len(EXPRESSION_NODES),
                            len(EXPRESSION_FEATURES),
                        ),
                    ).astype(np.float32)
                    response = (
                        jar3 * (relative >= 2.0)[:, None]
                        + 0.02 * relative[:, None]
                    )
                    graph[:, :, 0] += response.astype(np.float32)
                    graph[:, :, 1] += (0.5 * response).astype(np.float32)
                    graph[:, :, 2] += (0.1 * jar3)
                    graph[:, :, 3] += (0.05 * jar3)
                    graph[:, :, 4] = np.abs(graph[:, :, 2])
                    graph[:, :, 5] += (0.25 * response).astype(np.float32)
                    graph[:, :, 6] = 1.0
                    graph[:, :, 7] = 0.0
                    path = (
                        graph_dir
                        / f"{subject}_{code}_R{repeat}.npz"
                    )
                    meta = {
                        **expression_metadata(),
                        "subject_id": subject,
                        "condition_id": str(code),
                        "repeat": repeat,
                    }
                    np.savez_compressed(
                        path,
                        graph_seq=graph,
                        adj=adjacency,
                        target_lsl=target_lsl,
                        meta=np.asarray(json.dumps(meta)),
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

    def test_partial_nested_cv_smoke_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            output = root / "run"
            args = make_parser().parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--task",
                    "jar3",
                    "--output-dir",
                    str(output),
                    "--windows",
                    "0:2",
                    "2:4",
                    "--min-repeats",
                    "2",
                    "--cv-folds",
                    "3",
                    "--inner-folds",
                    "2",
                    "--fold-index",
                    "0",
                    "--k-grid",
                    "2",
                    "--no-include-all-features",
                    "--c-grid",
                    "0.1",
                    "--max-iter",
                    "500",
                    "--bootstrap-samples",
                    "0",
                ]
            )
            summary = run(args)
            self.assertTrue(summary["partial_cv"])
            self.assertFalse(summary["uses_ma_mau_as_predictor"])
            self.assertFalse(summary["water_605_supervised"])
            self.assertEqual(summary["representation"], "expression_v2")
            config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["input_graph_shape"], ["T", 20, 8])
            self.assertEqual(
                config["record_exclusions"]["excluded_water"], 18
            )
            self.assertIn("inner CV", config["window_selection_scope"])
            for name in (
                "predictions_condition.csv",
                "fold_metrics.csv",
                "selection_history.csv",
                "selected_features.csv",
                "confusion.npy",
                "summary.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue(
                (
                    output
                    / "fold_01"
                    / "expression_logistic.joblib"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
