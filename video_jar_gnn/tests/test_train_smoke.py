from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from video_jar_gnn.train import (
    confusion_and_metrics,
    make_parser,
    run_cross_validation,
)


class TrainSmokeTest(unittest.TestCase):
    def test_macro_f1_penalizes_prediction_of_absent_class(self):
        _, metrics = confusion_and_metrics([0, 0], [0, 2], num_classes=3)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0 / 3.0)

    def test_one_nested_group_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            codes = [258, 189, 893]
            jars = [1, 3, 5]
            for subject_number in range(1, 10):
                subject = f"P{subject_number:03d}"
                for label, (code, jar) in enumerate(zip(codes, jars)):
                    for repeat in (1, 2):
                        rng = np.random.default_rng(subject_number * 100 + label * 10 + repeat)
                        graph = rng.normal(size=(16, 15, 10)).astype(np.float32)
                        graph[:, :, 3] = 1.0
                        graph_path = graph_dir / f"{subject}_{code}_R{repeat}.npz"
                        np.savez_compressed(
                            graph_path, graph_seq=graph, adj=adjacency
                        )
                        rows.append(
                            {
                                "sample_id": graph_path.stem,
                                "subject_id": subject,
                                "ma_mau": code,
                                "repeat": repeat,
                                "jar": jar,
                                "jar3_label": label,
                                "binary_label": int(label == 1),
                                "graph_path": graph_path,
                                "detection_ratio": 1.0,
                                "extract_status": "ok",
                            }
                        )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            output = root / "run"
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
                    "--fold-index",
                    "0",
                    "--epochs",
                    "1",
                    "--patience",
                    "1",
                    "--hidden-channels",
                    "4",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                ]
            )
            summary = run_cross_validation(args)
            self.assertTrue(summary["partial_cv"])
            self.assertEqual(summary["folds_run"], [1])
            self.assertTrue((output / "fold_01" / "model.pt").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertGreater(
                summary["subject_condition_level"]["metrics"]["n"], 0
            )


if __name__ == "__main__":
    unittest.main()
