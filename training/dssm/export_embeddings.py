# training/dssm/export_embeddings.py
#
# 导出 DSSM item 向量 + user 画像, 供 Milvus 灌库和 user tower 服务使用.
#
# 用法:
#   python -m training.dssm.export_embeddings \
#     --checkpoint checkpoints/dssm/dssm_model.pt \
#     --vocab_path checkpoints/dssm/vocab.json \
#     --csv_path /home/vivwimp/Tenrec/ctr_data_1M.csv \
#     --out_dir checkpoints/dssm/export \
#     --profile_rows 5000000
#
# 产物:
#   item_vectors.npy    (N, out_dim) float32, 已 L2 归一化
#   item_ids.json       与 item_vectors 行序一致的 item_id 列表 (int)
#   user_profiles.json  {user_id: {gender, age, hist: [...]}}

import argparse
import json
import os

import numpy as np
import torch

from training.dssm.dataset import (export_item_categories,
                                   export_user_profiles)
from training.dssm.model import DSSM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/dssm/dssm_model.pt")
    p.add_argument("--vocab_path", default="checkpoints/dssm/vocab.json")
    p.add_argument("--csv_path", default="/home/vivwimp/Tenrec/ctr_data_1M.csv")
    p.add_argument("--out_dir", default="checkpoints/dssm/export")
    p.add_argument("--profile_rows", type=int, default=5_000_000)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.vocab_path) as f:
        vocab = json.load(f)
    state = torch.load(args.checkpoint, map_location=args.device)
    sizes = state["config"]["vocab_sizes"]
    model = DSSM(sizes,
                 embed_dim=state["config"]["embed_dim"],
                 out_dim=state["config"]["out_dim"])
    model.load_state_dict(state["model"])
    model.to(args.device).eval()

    # item idx -> 原始 item_id (idx 0 为 padding, 跳过)
    idx2item = {idx: int(item_id) for item_id, idx in vocab["item2idx"].items()}
    idxs = sorted(idx2item.keys())
    print(f"[export] {len(idxs)} items to export")

    vectors = np.zeros((len(idxs), state["config"]["out_dim"]), dtype=np.float32)

    # item -> category 映射 (与训练时的 category 输入一致)
    cat_map_path = os.path.join(args.out_dir, "item_categories.json")
    if os.path.exists(cat_map_path):
        with open(cat_map_path) as f:
            item_cat = json.load(f)
    else:
        n = export_item_categories(args.csv_path, args.profile_rows, cat_map_path)
        print(f"[export] wrote item_categories.json ({n} items)")
        with open(cat_map_path) as f:
            item_cat = json.load(f)
    cat_lookup = vocab["cat2idx"]

    with torch.no_grad():
        for start in range(0, len(idxs), args.batch_size):
            part = idxs[start:start + args.batch_size]
            item_t = torch.tensor(part, dtype=torch.long, device=args.device)
            cat_t = torch.tensor(
                [cat_lookup.get(str(item_cat.get(str(idx2item[i]), "")), 0)
                 for i in part],
                dtype=torch.long, device=args.device)
            vec = model.item_tower(item_t, cat_t)
            vectors[start:start + len(part)] = vec.cpu().numpy()

    # L2 归一化 (IP 距离等价 cosine)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norm, 1e-12, None)

    np.save(os.path.join(args.out_dir, "item_vectors.npy"), vectors)
    with open(os.path.join(args.out_dir, "item_ids.json"), "w") as f:
        json.dump([idx2item[i] for i in idxs], f)
    print(f"[export] wrote item_vectors.npy ({vectors.shape}) and item_ids.json")

    n = export_user_profiles(
        args.csv_path, vocab, args.profile_rows,
        os.path.join(args.out_dir, "user_profiles.json"))
    print(f"[export] wrote user_profiles.json ({n} users)")


if __name__ == "__main__":
    main()
