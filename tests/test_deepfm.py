import unittest

import numpy as np
import torch

from inference.deepfm_rank_server import create_app
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

    def score(self, user_id, item_ids):
        if len(item_ids) != self.expected_candidates:
            raise ValueError("wrong candidate count")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("duplicate item")
        return np.linspace(0.0, 1.0, len(item_ids)).tolist(), 0.1, 0.2


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
        self.assertEqual(len(response["items"]), 50)

    def test_rejects_short_and_duplicate_candidates(self):
        short = {**self.payload, "items": self.payload["items"][:-1]}
        self.assertEqual(self.client.post("/rank", json=short).get_json()["code"], 400)
        duplicate = {
            **self.payload,
            "items": self.payload["items"][:-1] + [self.payload["items"][0]],
        }
        self.assertEqual(self.client.post("/rank", json=duplicate).get_json()["code"], 400)


if __name__ == "__main__":
    unittest.main()
