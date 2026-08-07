import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts.audit_dssm_training_and_artifacts import audit, parse_pipeline_settings, ratio
from training.dssm.model import DSSM


class DSSMAuditHelpersTest(unittest.TestCase):
    def test_ratio(self):
        self.assertEqual(ratio(1, 4), 0.25)
        self.assertIsNone(ratio(1, 0))

    def test_parse_pipeline_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.sh"
            path.write_text(
                "python train.py --train_rows 1000000 \\\n"
                "  --vocab_rows 2000000 --batch_size 4096 --epochs 10\n"
                "python export.py --profile_rows 3000000\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_pipeline_settings(path),
                {
                    "train_rows": 1000000,
                    "vocab_rows": 2000000,
                    "profile_rows": 3000000,
                    "batch_size": 4096,
                    "epochs": 10,
                },
            )

    def test_synthetic_artifacts_are_audited_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dssm = root / "dssm_out"
            export = dssm / "export"
            deepfm = root / "deepfm_out"
            export.mkdir(parents=True)
            deepfm.mkdir()
            vocab = {
                "user2idx": {"1": 1},
                "item2idx": {"10": 1, "20": 2},
                "cat2idx": {"3": 1},
                "gender2idx": {"1": 1},
                "age2idx": {"2": 1},
            }
            (dssm / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            sizes = {"user": 2, "item": 3, "cat": 2, "gender": 2, "age": 2}
            model = DSSM(sizes, embed_dim=4, out_dim=4)
            torch.save({
                "model": model.state_dict(),
                "epoch": 2,
                "avg_loss": 1.5,
                "config": {
                    "embed_dim": 4,
                    "out_dim": 4,
                    "temperature": 0.05,
                    "vocab_sizes": sizes,
                },
            }, dssm / "dssm_model.pt")
            with torch.no_grad():
                vectors = model.item_tower(torch.tensor([1, 2]), torch.tensor([1, 1])).numpy()
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            np.save(export / "item_vectors.npy", vectors)
            (export / "item_ids.json").write_text("[10, 20]", encoding="utf-8")
            (export / "item_categories.json").write_text('{"10": 3, "20": 3}', encoding="utf-8")
            (export / "user_profiles.json").write_text(
                '{"1": {"gender": 1, "age": 2, "hist": [10]}}', encoding="utf-8"
            )
            (deepfm / "feature_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            (deepfm / "item_categories.json").write_text(
                '{"10": 3, "20": 3, "30": 3}', encoding="utf-8"
            )
            (deepfm / "training_summary.json").write_text(json.dumps({
                "best_val_auc": 0.7,
                "best_val_logloss": 0.5,
                "history": [{"train_examples": 90, "val_examples": 10}],
            }), encoding="utf-8")
            csv_path = root / "data.csv"
            row = {
                "user_id": 1, "item_id": 10, "video_category": 3,
                "gender": 1, "age": 2,
            }
            row.update({column: 10 for column in [f"hist_{i}" for i in range(1, 11)]})
            pd.DataFrame([row]).to_csv(csv_path, index=False)
            pipeline = root / "pipeline.sh"
            pipeline.write_text(
                "train --train_rows 10 --vocab_rows 10 --batch_size 4 --epochs 2\n"
                "export --profile_rows 10\n",
                encoding="utf-8",
            )
            report = audit(SimpleNamespace(
                dssm_dir=str(dssm), deepfm_dir=str(deepfm), csv_path=str(csv_path),
                csv_scan_rows="1", pipeline_script=str(pipeline),
            ))
            self.assertEqual(report["result"], "WARN")
            self.assertEqual(report["failures"], [])
            self.assertEqual(report["training_scope"]["expected_batches_per_epoch"], 2)
            self.assertAlmostEqual(report["coverage"]["full_target_item_coverage"], 2 / 3)
            self.assertTrue(report["coverage"]["deepfm_feature_vocab_matches_dssm"])


if __name__ == "__main__":
    unittest.main()
