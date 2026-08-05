# inference/dssm_recall_server.py
#
# DSSM 向量召回 HTTP 服务.
# 输入 user_id -> 查 user 画像 -> user tower 前向 -> Milvus topK -> 返回 item 列表.
# Milvus 不可用时回退到本地 numpy 暴力检索 (smoke/降级用).
#
# 环境变量:
#   DSSM_CHECKPOINT     默认 checkpoints/dssm/dssm_model.pt
#   DSSM_VOCAB          默认 checkpoints/dssm/vocab.json
#   DSSM_EXPORT_DIR     默认 checkpoints/dssm/export (含 item_vectors.npy 等)
#   MILVUS_HOST         空 = 不连 Milvus, 用本地暴力检索
#   MILVUS_PORT         默认 19530
#   MILVUS_COLLECTION   默认 dssm_item_vectors
#   DSSM_RECALL_PORT    默认 18200

import json
import os
import time

import numpy as np
import torch
from flask import Flask, jsonify, request

from training.dssm.model import DSSM

CHECKPOINT = os.environ.get("DSSM_CHECKPOINT", "checkpoints/dssm/dssm_model.pt")
VOCAB_PATH = os.environ.get("DSSM_VOCAB", "checkpoints/dssm/vocab.json")
EXPORT_DIR = os.environ.get("DSSM_EXPORT_DIR", "checkpoints/dssm/export")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "dssm_item_vectors")
PORT = int(os.environ.get("DSSM_RECALL_PORT", "18200"))

app = Flask(__name__)

model = None
vocab = None
profiles = None
item_vectors = None   # 本地暴力检索用 (N, dim), 已归一化
item_ids = None       # 本地暴力检索用
milvus_collection = None
milvus_err = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_resources():
    global model, vocab, profiles, item_vectors, item_ids
    global milvus_collection, milvus_err

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    state = torch.load(CHECKPOINT, map_location=device)
    model = DSSM(state["config"]["vocab_sizes"],
                 embed_dim=state["config"]["embed_dim"],
                 out_dim=state["config"]["out_dim"])
    model.load_state_dict(state["model"])
    model.to(device).eval()

    with open(os.path.join(EXPORT_DIR, "user_profiles.json")) as f:
        profiles = json.load(f)

    if MILVUS_HOST:
        try:
            from pymilvus import Collection, connections
            connections.connect(host=MILVUS_HOST, port=str(MILVUS_PORT))
            milvus_collection = Collection(MILVUS_COLLECTION)
            milvus_collection.load()
            print(f"[dssm-recall] milvus connected {MILVUS_HOST}:{MILVUS_PORT} "
                  f"collection={MILVUS_COLLECTION}")
        except Exception as e:  # noqa: BLE001
            milvus_err = str(e)
            print(f"[dssm-recall] milvus connect failed: {e}; fallback to local search")
    if milvus_collection is None:
        item_vectors = np.load(os.path.join(EXPORT_DIR, "item_vectors.npy"))
        with open(os.path.join(EXPORT_DIR, "item_ids.json")) as f:
            item_ids = json.load(f)
        print(f"[dssm-recall] local search fallback: {len(item_ids)} items")


def build_user_vector(user_id):
    """返回 (user_vec_np, error_msg)."""
    uid = str(user_id)
    user2idx = vocab["user2idx"]
    profile = profiles.get(uid)
    if profile is None:
        return None, f"user {uid} not found in profiles"
    item2idx = vocab["item2idx"]
    hist = [item2idx.get(str(int(v)), 0) for v in profile["hist"]]
    gender2idx = vocab["gender2idx"]
    age2idx = vocab["age2idx"]
    feats = {
        "user_id": torch.tensor([user2idx.get(uid, 0)], device=device),
        "gender": torch.tensor([gender2idx.get(str(profile["gender"]), 0)], device=device),
        "age": torch.tensor([age2idx.get(str(profile["age"]), 0)], device=device),
        "hist": torch.tensor([hist], device=device),
    }
    with torch.no_grad():
        vec = model.user_tower(feats["user_id"], feats["gender"],
                               feats["age"], feats["hist"])
    vec = vec.cpu().numpy()[0]
    vec = vec / max(np.linalg.norm(vec), 1e-12)
    return vec.astype(np.float32), None


def search_milvus(vec, topk):
    res = milvus_collection.search(
        data=[vec.tolist()], anns_field="vector",
        param={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=topk, output_fields=["item_id"])
    hits = res[0]
    return [{"item_id": str(h.entity.get("item_id")), "score": float(h.distance)}
            for h in hits]


def search_local(vec, topk):
    scores = item_vectors @ vec
    k = min(topk, len(item_ids))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [{"item_id": str(item_ids[i]), "score": float(scores[i])} for i in idx]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "code": 200,
        "status": "healthy",
        "backend": "dssm_recall",
        "milvus": milvus_collection is not None,
        "milvus_err": milvus_err,
    })


@app.route("/recall", methods=["POST"])
def recall():
    t0 = time.time()
    payload = request.get_json(force=True, silent=True) or {}
    user_id = payload.get("user_id")
    topk = int(payload.get("topk", 20))
    if user_id is None:
        return jsonify({"code": 400, "msg": "user_id required", "items": []})

    vec, err = build_user_vector(user_id)
    if err:
        return jsonify({"code": 404, "msg": err, "items": []})

    if milvus_collection is not None:
        items = search_milvus(vec, topk)
        source = "milvus"
    else:
        items = search_local(vec, topk)
        source = "local"
    return jsonify({
        "code": 200,
        "msg": "success",
        "items": items,
        "source": source,
        "latency_ms": round((time.time() - t0) * 1000, 3),
    })


if __name__ == "__main__":
    load_resources()
    print(f"[dssm-recall] listening on 0.0.0.0:{PORT} device={device}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
