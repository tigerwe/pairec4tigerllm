"""Strict HTTP scoring service for the PyTorch DeepFM ranker."""

import hashlib
import json
import math
import os
import time

import torch
from flask import Flask, jsonify, request

from training.deepfm.model import load_deepfm_checkpoint


class DeepFMRankRuntime:
    def __init__(self, model_path, vocab_path, profiles_path, categories_path,
                 expected_candidates=50, device="cpu"):
        self.device = torch.device(device)
        self.expected_candidates = expected_candidates
        self.model, self.checkpoint = load_deepfm_checkpoint(model_path, self.device)
        self.model.to(self.device).eval()
        self.vocab = self._read_json(vocab_path)
        self.profiles = self._read_json(profiles_path)
        self.categories = self._read_json(categories_path)
        with open(model_path, "rb") as handle:
            self.model_sha256 = hashlib.sha256(handle.read()).hexdigest()
        self.model_version = os.getenv("DEEPFM_MODEL_VERSION", self.model_sha256[:16])

    @staticmethod
    def _read_json(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _map(mapping, value):
        try:
            key = str(int(value))
        except (TypeError, ValueError):
            return 0
        return mapping.get(key, 0)

    def score(self, user_id, item_ids):
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a non-empty string")
        if len(item_ids) != self.expected_candidates:
            raise ValueError(f"items must contain exactly {self.expected_candidates} candidates")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("items must not contain duplicate item_id values")

        feature_started = time.perf_counter()
        profile = self.profiles.get(user_id, {})
        history = list(profile.get("hist", []))[:10]
        history += [-1] * (10 - len(history))
        count = len(item_ids)
        mapped_history = [self._map(self.vocab["item2idx"], value) for value in history]
        batch = {
            "user_id": torch.full(
                (count,), self._map(self.vocab["user2idx"], user_id), dtype=torch.long),
            "item_id": torch.tensor(
                [self._map(self.vocab["item2idx"], value) for value in item_ids],
                dtype=torch.long),
            "category": torch.tensor(
                [self._map(self.vocab["cat2idx"], self.categories.get(value, -1))
                 for value in item_ids], dtype=torch.long),
            "gender": torch.full(
                (count,), self._map(self.vocab["gender2idx"], profile.get("gender", -1)),
                dtype=torch.long),
            "age": torch.full(
                (count,), self._map(self.vocab["age2idx"], profile.get("age", -1)),
                dtype=torch.long),
            "hist": torch.tensor([mapped_history] * count, dtype=torch.long),
        }
        batch = {key: value.to(self.device) for key, value in batch.items()}
        feature_ms = (time.perf_counter() - feature_started) * 1000

        forward_started = time.perf_counter()
        with torch.inference_mode():
            scores = torch.sigmoid(self.model(batch)).cpu().tolist()
        forward_ms = (time.perf_counter() - forward_started) * 1000
        if len(scores) != count or any(not math.isfinite(score) for score in scores):
            raise RuntimeError("model returned invalid scores")
        return scores, feature_ms, forward_ms


def create_app(runtime):
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({
            "code": 200,
            "status": "healthy",
            "backend": "deepfm_rank",
            "model_version": runtime.model_version,
            "expected_candidates": runtime.expected_candidates,
        })

    @app.post("/rank")
    def rank():
        started = time.perf_counter()
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"code": 400, "msg": "invalid JSON object", "items": []})
        request_id = payload.get("request_id", "")
        raw_items = payload.get("items")
        if not isinstance(request_id, str) or not request_id:
            return jsonify({"code": 400, "msg": "request_id is required", "items": []})
        if not isinstance(raw_items, list) or any(not isinstance(item, dict) for item in raw_items):
            return jsonify({"code": 400, "request_id": request_id,
                            "msg": "items must be an array of objects", "items": []})
        item_ids = [item.get("item_id") for item in raw_items]
        if any(not isinstance(item_id, str) or not item_id for item_id in item_ids):
            return jsonify({"code": 400, "request_id": request_id,
                            "msg": "every item_id must be a non-empty string", "items": []})
        try:
            scores, feature_ms, forward_ms = runtime.score(payload.get("user_id"), item_ids)
        except ValueError as exc:
            return jsonify({"code": 400, "request_id": request_id,
                            "msg": str(exc), "items": []})
        except Exception as exc:
            app.logger.exception("DeepFM scoring failed request_id=%s", request_id)
            return jsonify({"code": 500, "request_id": request_id,
                            "msg": "deepfm scoring failed", "items": []})
        total_ms = (time.perf_counter() - started) * 1000
        return jsonify({
            "code": 200,
            "request_id": request_id,
            "model_version": runtime.model_version,
            "items": [{"item_id": item_id, "score": score}
                      for item_id, score in zip(item_ids, scores)],
            "trace": {
                "feature_ms": round(feature_ms, 3),
                "forward_ms": round(forward_ms, 3),
                "total_ms": round(total_ms, 3),
            },
        })

    return app


def main():
    required = {
        "model_path": os.getenv("DEEPFM_MODEL_PATH", "/models/deepfm/deepfm_best.pt"),
        "vocab_path": os.getenv("DEEPFM_VOCAB_PATH", "/models/deepfm/feature_vocab.json"),
        "profiles_path": os.getenv("DEEPFM_PROFILES_PATH", "/models/deepfm/user_profiles.json"),
        "categories_path": os.getenv("DEEPFM_CATEGORIES_PATH", "/models/deepfm/item_categories.json"),
    }
    for name, path in required.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")
    runtime = DeepFMRankRuntime(
        **required,
        expected_candidates=int(os.getenv("DEEPFM_EXPECTED_CANDIDATES", "50")),
        device=os.getenv("DEEPFM_DEVICE", "cpu"),
    )
    print(json.dumps({
        "event": "deepfm_rank_ready",
        "model_version": runtime.model_version,
        "expected_candidates": runtime.expected_candidates,
        "device": str(runtime.device),
    }), flush=True)
    create_app(runtime).run(
        host="0.0.0.0",
        port=int(os.getenv("DEEPFM_RANK_PORT", "18210")),
        threaded=True,
    )


if __name__ == "__main__":
    main()
