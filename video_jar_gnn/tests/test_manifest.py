from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from video_jar_gnn.manifest import ManifestError, build_manifest


class ManifestTest(unittest.TestCase):
    def test_repeats_of_one_condition_must_share_jar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_dir = root / "videos"
            label_dir = root / "labels"
            jar_dir = root / "jar"
            for directory in (video_dir, label_dir, jar_dir):
                directory.mkdir()
            (video_dir / "N01_vid-001.mp4").touch()
            with (label_dir / "N01_vid.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["frame_idx", "t_lsl", "ma_mau", "lan_lap"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"frame_idx": 0, "t_lsl": 1.0, "ma_mau": 189, "lan_lap": 1},
                        {"frame_idx": 1, "t_lsl": 1.1, "ma_mau": 0, "lan_lap": 0},
                        {"frame_idx": 2, "t_lsl": 1.2, "ma_mau": 189, "lan_lap": 2},
                    ]
                )
            with (jar_dir / "sub-P001_ses-S001_task-X_eeg.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["ma_mau", "repeat", "JAR"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"ma_mau": 189, "repeat": 1, "JAR": 2},
                        {"ma_mau": 189, "repeat": 2, "JAR": 3},
                    ]
                )
            with self.assertRaises(ManifestError):
                build_manifest(
                    video_dir,
                    label_dir,
                    jar_dir,
                    strict_complete=False,
                )

    def test_join_uses_subject_code_and_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_dir = root / "videos"
            label_dir = root / "labels"
            jar_dir = root / "jar"
            for directory in (video_dir, label_dir, jar_dir):
                directory.mkdir()
            (video_dir / "N01_vid-001.mp4").touch()

            with (label_dir / "N01_vid.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["frame_idx", "t_lsl", "t_epoch", "ma_mau", "lan_lap"],
                )
                writer.writeheader()
                rows = [
                    (0, 100.0, 0, 0),
                    (1, 100.1, 189, 2),
                    (2, 100.2, 189, 2),
                    (3, 100.3, 0, 0),
                    # Deliberately later and with another repeat.
                    (4, 100.4, 605, 1),
                    (5, 100.5, 605, 1),
                ]
                for frame, timestamp, code, repeat in rows:
                    writer.writerow(
                        {
                            "frame_idx": frame,
                            "t_lsl": timestamp,
                            "t_epoch": timestamp + 1000,
                            "ma_mau": code,
                            "lan_lap": repeat,
                        }
                    )

            with (jar_dir / "sub-P001_ses-S001_task-X_eeg.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["ma_mau", "repeat", "JAR"]
                )
                writer.writeheader()
                # Reverse order verifies that row order is never used for join.
                writer.writerow({"ma_mau": 605, "repeat": 1, "JAR": 1})
                writer.writerow({"ma_mau": 189, "repeat": 2, "JAR": 3})

            rows, audit = build_manifest(
                video_dir,
                label_dir,
                jar_dir,
                strict_complete=False,
            )
            by_key = {(row["ma_mau"], row["repeat"]): row for row in rows}
            self.assertEqual(by_key[(189, 2)]["subject_id"], "P001")
            self.assertEqual(by_key[(189, 2)]["binary_label"], 1)
            self.assertEqual(by_key[(189, 2)]["jar3_name"], "Vua_phai")
            self.assertEqual(by_key[(605, 1)]["jar3_label"], 0)
            self.assertEqual(audit["n_trials"], 2)


if __name__ == "__main__":
    unittest.main()
