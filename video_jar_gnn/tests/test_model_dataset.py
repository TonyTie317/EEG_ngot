from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from video_jar_gnn.dataset import (
    FeatureStandardizer,
    GraphDataset,
    GraphStore,
    load_graph_records,
)
from video_jar_gnn.model import FacialSTGCN


class ModelDatasetTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        manifest = root / "graphs.csv"
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
        for index, (subject, label) in enumerate(
            [("P001", 0), ("P002", 1), ("P003", 2)]
        ):
            path = root / f"graph_{index}.npz"
            graph = np.random.default_rng(index).normal(
                size=(24, 15, 10)
            ).astype(np.float32)
            graph[:, :, 3] = 1.0
            if index == 0:
                graph[::3, :, 3] = 0.0
            np.savez_compressed(path, graph_seq=graph, adj=adjacency)
            rows.append(
                {
                    "sample_id": f"{subject}_189_R1",
                    "subject_id": subject,
                    "ma_mau": 189,
                    "repeat": 1,
                    "jar": label + 2,
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

    def test_dataset_and_model_forward_backward(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._fixture(Path(temporary))
            records, _ = load_graph_records(manifest)
            store = GraphStore(records)
            normalizer = FeatureStandardizer.fit(store, [0, 1])
            dataset = GraphDataset(
                store,
                [0, 1],
                task="jar3",
                normalizer=normalizer,
                training=False,
            )
            graph = torch.stack([dataset[0][0], dataset[1][0]])
            adjacency = torch.stack([dataset[0][1], dataset[1][1]])
            model = FacialSTGCN(num_classes=3, hidden_channels=8)
            logits = model(graph, adjacency)
            self.assertEqual(tuple(logits.shape), (2, 3))
            logits.sum().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
            # Detection mask remains exactly binary after fold normalization.
            detection = graph[:, :, :, 3]
            self.assertTrue(torch.all((detection == 0.0) | (detection == 1.0)))

            augmented = GraphDataset(
                store,
                [0],
                task="jar3",
                normalizer=normalizer,
                training=True,
                temporal_crop_min=0.75,
                noise_std=0.01,
            )[0][0]
            detection = augmented[:, :, 3]
            self.assertTrue(torch.all((detection == 0.0) | (detection == 1.0)))


if __name__ == "__main__":
    unittest.main()
