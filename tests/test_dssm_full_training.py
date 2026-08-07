import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.deepfm.dataset import iter_split_batches
from training.dssm.dataset import build_or_load_vocab
from training.dssm.evaluate_retrieval import sample_validation_positives
from training.dssm.model import in_batch_recall_counts, in_batch_softmax_loss


def frame(row_count=200):
    data = {
        "user_id": np.arange(row_count) % 17 + 1,
        "item_id": np.arange(row_count) % 31 + 100,
        "click": np.arange(row_count) % 3 != 0,
        "follow": 0,
        "like": 0,
        "share": 0,
        "video_category": np.arange(row_count) % 5 + 1,
        "watching_times": 1,
        "gender": np.arange(row_count) % 2,
        "age": np.arange(row_count) % 4 + 1,
    }
    for index in range(1, 11):
        data[f"hist_{index}"] = (np.arange(row_count) + index) % 31 + 100
    return pd.DataFrame(data)


class DSSMFullTrainingTest(unittest.TestCase):
    def test_multi_positive_loss_does_not_treat_duplicate_items_as_negatives(self):
        users = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        items = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        labels = torch.ones(3)
        item_ids = torch.tensor([7, 7, 7])
        loss = in_batch_softmax_loss(users, items, labels, item_ids=item_ids)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        counts = in_batch_recall_counts(users, items, labels, item_ids, topk=(1,))
        self.assertEqual(counts, {"queries": 3, "hits_at_1": 3})

    def test_zero_vocab_rows_scans_full_csv_and_positive_limit_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "data.csv"
            frame(200).to_csv(csv_path, index=False)
            full = build_or_load_vocab(csv_path, root / "full.json", 0)
            partial = build_or_load_vocab(csv_path, root / "partial.json", 10)
            self.assertGreater(len(full["user2idx"]), len(partial["user2idx"]))
            self.assertGreater(len(full["item2idx"]), len(partial["item2idx"]))

            rows, evidence = sample_validation_positives(csv_path, 7, 0.5, 20260807)
            self.assertEqual(len(rows), 7)
            self.assertEqual(evidence["csv_rows_scanned"], 200)

    def test_training_shuffle_is_deterministic_and_changes_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "data.csv"
            frame(200).to_csv(csv_path, index=False)
            vocab = build_or_load_vocab(csv_path, root / "vocab.json", 0)
            plain_batches = list(iter_split_batches(
                csv_path, vocab, 32, "train", 0.1, 20260807, 0,
                chunk_size=200, shuffle=False,
            ))
            first_batches = list(iter_split_batches(
                csv_path, vocab, 32, "train", 0.1, 20260807, 0,
                chunk_size=200, shuffle=True, shuffle_seed=9,
            ))
            second_batches = list(iter_split_batches(
                csv_path, vocab, 32, "train", 0.1, 20260807, 0,
                chunk_size=200, shuffle=True, shuffle_seed=9,
            ))
            plain = plain_batches[0]["item_id"]
            first = first_batches[0]["item_id"]
            second = second_batches[0]["item_id"]
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(np.array_equal(first, plain))

    def test_tiny_train_export_and_full_corpus_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "data.csv"
            output = root / "output"
            frame(200).to_csv(csv_path, index=False)
            env = os.environ.copy()
            env.update({
                "CSV_PATH": str(csv_path), "OUTPUT_DIR": str(output),
                "DEVICE": "cpu", "TRAIN_ROWS": "0", "VOCAB_ROWS": "0",
                "PROFILE_ROWS": "0", "BATCH_SIZE": "32", "EPOCHS": "1",
                "PATIENCE": "1", "VAL_FRACTION": "0.2",
                "EVAL_QUERY_LIMIT": "10", "EVAL_QUERY_BATCH_SIZE": "4",
            })
            subprocess.run(
                ["bash", "scripts/run_dssm_full_retrain.sh"], env=env,
                check=True, capture_output=True, text=True,
            )
            training = json.loads((output / "training_summary.json").read_text())
            evaluation = json.loads((output / "retrieval_evaluation.json").read_text())
            self.assertEqual(training["best_epoch"], 1)
            self.assertEqual(evaluation["status"], "PASS")
            self.assertEqual(evaluation["evaluated_queries"], 10)
            self.assertIn("full_corpus_recall_at_50", evaluation["metrics"])

            env.update({"START_STAGE": "evaluate", "EVAL_QUERY_BATCH_SIZE": "2"})
            subprocess.run(
                ["bash", "scripts/run_dssm_full_retrain.sh"], env=env,
                check=True, capture_output=True, text=True,
            )


if __name__ == "__main__":
    unittest.main()
