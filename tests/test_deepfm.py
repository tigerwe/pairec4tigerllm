import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from inference.deepfm_rank_server import DeepFMRankRuntime, create_app
from training.deepfm.model import DeepFM


class DeepFMModelTest(unittest.TestCase):
    def test_forward_shape_and_finite_logits(self):
        model = DeepFM(
            {"user": 4, "item": 8, "cat": 3, "gender": 3, "age": 5},
            embed_dim=4,
            hidden_dims=(8, 4),
            dropout=0,
        )
        batch = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([3, 4]),
            "category": torch.tensor([1, 2]),
            "gender": torch.tensor([1, 2]),
            "age": torch.tensor([2, 3]),
            "hist": torch.tensor([[1] * 10, [2] * 10]),
        }
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())


class FakeRankRuntime:
    expected_candidates = 50
    model_version = "unit-test-v1"
    model_role = "engineering"
    checkpoint = {"epoch": 3, "val_auc": 0.7, "val_logloss": 0.5}
    vocab_sizes = {"user": 2, "item": 51, "cat": 2, "gender": 2, "age": 2}
    profiles = {"1": {}}
    categories = {str(index): 1 for index in range(50)}

    def score(self, user_id, item_ids):
        if len(item_ids) != self.expected_candidates:
            raise ValueError("wrong candidate count")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("duplicate item")
        scores = np.linspace(0.0, 1.0, len(item_ids)).tolist()
        coverage = {
            "profile_missing": False,
            "user_oov": False,
            "item_oov_count": 0,
            "category_oov_count": 0,
            "gender_oov": False,
            "age_oov": False,
            "history_valid_count": 10,
            "history_oov_count": 0,
            "score_unique_count": len(scores),
            "score_min": min(scores),
            "score_max": max(scores),
        }
        return scores, 0.1, 0.2, coverage


class DeepFMRankServerTest(unittest.TestCase):
    def setUp(self):
        self.client = create_app(FakeRankRuntime()).test_client()
        self.payload = {
            "request_id": "unit-request",
            "user_id": "1",
            "items": [{"item_id": str(index)} for index in range(50)],
        }

    def test_strict_success(self):
        response = self.client.post("/rank", json=self.payload).get_json()
        self.assertEqual(response["code"], 200)
        self.assertEqual(response["request_id"], "unit-request")
        self.assertEqual(response["model_version"], "unit-test-v1")
        self.assertEqual(response["model_role"], "engineering")
        self.assertEqual(len(response["items"]), 50)
        self.assertEqual(response["trace"]["item_oov_count"], 0)
        self.assertEqual(response["trace"]["score_unique_count"], 50)

    def test_health_exposes_model_identity_and_scope(self):
        response = self.client.get("/health").get_json()
        self.assertEqual(response["model_role"], "engineering")
        self.assertEqual(response["checkpoint_epoch"], 3)
        self.assertEqual(response["vocab_sizes"]["item"], 51)

    def test_rejects_short_and_duplicate_candidates(self):
        short = {**self.payload, "items": self.payload["items"][:-1]}
        rejected = self.client.post("/rank", json=short).get_json()
        self.assertEqual(rejected["code"], 400)
        self.assertEqual(rejected["message"], "wrong candidate count")
        self.assertEqual(rejected["msg"], rejected["message"])
        duplicate = {
            **self.payload,
            "items": self.payload["items"][:-1] + [self.payload["items"][0]],
        }
        self.assertEqual(self.client.post("/rank", json=duplicate).get_json()["code"], 400)

    def test_accepts_protobuf_json_request_identity(self):
        payload = dict(self.payload)
        payload.pop("request_id")
        payload["context"] = {
            "request_id": "brpc-request",
            "contract_version": "pairec.pipeline_trace.v1",
        }
        response = self.client.post("/rank", json=payload).get_json()
        self.assertEqual(response["code"], 200)
        self.assertEqual(response["request_id"], "brpc-request")


class DeepFMRankRuntimeTest(unittest.TestCase):
    def test_reports_feature_coverage_and_checks_vocab_identity(self):
        vocab = {
            "user2idx": {"1": 1},
            "item2idx": {"10": 1, "11": 2},
            "cat2idx": {"3": 1},
            "gender2idx": {"1": 1},
            "age2idx": {"2": 1},
        }
        sizes = {name[:-4] if name.endswith("2idx") else name: len(values) + 1
                 for name, values in vocab.items()}
        model = DeepFM(sizes, embed_dim=4, hidden_dims=(8,), dropout=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.pt"
            torch.save({
                "model": model.state_dict(),
                "model_config": model.config(sizes),
                "epoch": 1,
                "val_auc": 0.6,
                "val_logloss": 0.7,
            }, model_path)
            artifacts = {
                "vocab.json": vocab,
                "profiles.json": {"1": {"gender": 1, "age": 2, "hist": [10]}},
                "categories.json": {"10": 3},
            }
            for name, value in artifacts.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            runtime = DeepFMRankRuntime(
                model_path, root / "vocab.json", root / "profiles.json",
                root / "categories.json", expected_candidates=2)
            _, _, _, coverage = runtime.score("1", ["10", "99"])
            self.assertFalse(coverage["user_oov"])
            self.assertEqual(coverage["item_oov_count"], 1)
            self.assertEqual(coverage["category_oov_count"], 1)
            self.assertEqual(coverage["history_oov_count"], 0)

            vocab["item2idx"]["12"] = 3
            (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
                DeepFMRankRuntime(
                    model_path, root / "vocab.json", root / "profiles.json",
                    root / "categories.json", expected_candidates=2)

    def test_rank_container_overrides_read_only_mount_workdir(self):
        wrapper = Path("scripts/run_deepfm_rank_container.sh").read_text(
            encoding="utf-8")
        self.assertIn('-v "$REPO_DIR:/workspace:ro"', wrapper)
        self.assertIn("--workdir /workspace", wrapper)
        self.assertIn('PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}"', wrapper)
        self.assertIn("python -m inference.deepfm_rank_server", wrapper)

    def test_rank_deployment_derives_phase_dir_after_local_assignment(self):
        wrapper = Path(
            "scripts/deploy_and_validate_pairec_deepfm_rank.sh").read_text(
                encoding="utf-8")
        self.assertIn('local phase="$1" requests="$2" size="$3"\n'
                      '  local phase_dir="$OUTPUT_DIR/$phase"', wrapper)
        self.assertNotIn('size="$3" phase_dir="$OUTPUT_DIR/$phase"', wrapper)


if __name__ == "__main__":
    unittest.main()
